from types import SimpleNamespace

from tests.support import *


def test_summary_model_uses_openrouter_reported_cost():
    from api.memory.summary import call_summary_model

    response = SimpleNamespace(
        id="generation-summary",
        model="deepseek/deepseek-v4-flash",
        provider="DeepInfra",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            cost="0.00010442124",
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock(return_value=response)))
    )

    text, cost, segment = call_summary_model(
        [{"role": "user", "content": "summarize"}],
        get_client=lambda: client,
        estimate_tokens=lambda _messages: 100,
        estimate_cost=lambda *_args: 999_999,
        model="~deepseek/deepseek-v4-flash-latest",
        max_tokens=100,
        logger=MagicMock(),
    )

    assert text == "summary"
    assert cost == 105
    assert segment is not None
    assert segment["model"] == "deepseek/deepseek-v4-flash"
    assert segment["metadata"]["requested_model"] == "~deepseek/deepseek-v4-flash-latest"
    assert segment["metadata"]["upstream_provider"] == "DeepInfra"
    assert segment["usage"]["cost"] == "0.00010442124"


def test_summary_model_uses_routed_endpoint_price_when_upstream_cost_is_free():
    from api.memory.summary import call_summary_model

    response = SimpleNamespace(
        model="deepseek/deepseek-v4-flash-0731",
        provider="DeepInfra",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=50,
            cost="0.000001",
            cost_details={"upstream_inference_cost": 0},
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock(return_value=response)))
    )
    lookup = MagicMock(
        return_value={
            "input_per_million": 80_000,
            "cached_input_per_million": 16_000,
            "output_per_million": 180_000,
        }
    )

    _text, cost, segment = call_summary_model(
        [{"role": "user", "content": "summarize"}],
        get_client=lambda: client,
        estimate_tokens=lambda _messages: 1_000,
        estimate_cost=lambda *_args: 999_999,
        model="~deepseek/deepseek-v4-flash-latest",
        max_tokens=100,
        logger=MagicMock(),
        get_provider_pricing=lookup,
    )

    assert cost == 89
    assert segment is not None
    assert segment["metadata"]["provider_pricing_source"] == "openrouter_endpoints"
    lookup.assert_called_once_with("deepseek/deepseek-v4-flash-0731", "DeepInfra")


def test_incremental_summary_helper_uses_only_messages_after_marker():
    _build_incremental_summary_source = index.app_runtime.summary.build_incremental_source

    history = [{"id": f"m{i}", "role": "user", "content": f"msg {i}"} for i in range(1, 6)]

    source = _build_incremental_summary_source(history, "old summary", "m3")

    assert [msg["id"] for msg in source.delta_messages] == ["m4", "m5"]
    assert source.is_zero_delta is False
    assert source.next_marker == "m5"


def test_incremental_summary_helper_reports_zero_delta_without_history_fallback():
    _build_incremental_summary_source = index.app_runtime.summary.build_incremental_source

    history = [{"id": f"m{i}", "role": "user", "content": f"msg {i}"} for i in range(1, 4)]

    source = _build_incremental_summary_source(history, "old summary", "m3")

    assert source.delta_messages == []
    assert source.is_zero_delta is True
    assert source.next_marker is None


def test_incremental_summary_helper_falls_back_to_all_history_when_marker_missing():
    _build_incremental_summary_source = index.app_runtime.summary.build_incremental_source

    history = [{"id": f"m{i}", "role": "user", "content": f"msg {i}"} for i in range(1, 4)]

    source = _build_incremental_summary_source(history, "old summary", "m99")

    assert [msg["id"] for msg in source.delta_messages] == ["m1", "m2", "m3"]
    assert source.is_zero_delta is False
    assert source.next_marker == "m3"


def test_compact_chat_memory_absorbs_only_uncompacted_messages_once():
    compact_chat_memory = index.app_runtime.summary.compact_memory

    redis_client = MagicMock()
    messages = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i} for i in range(1, 21)
    ]

    summary, kept, marker, cost = compact_chat_memory(
        redis_client,
        "123",
        messages,
        "old summary",
        "m10",
        compact_fn=lambda msgs, prior: ("new summary", 1),
        compaction_threshold=8,
        compaction_keep=5,
    )

    assert summary == "new summary"
    assert marker == "m15"
    assert [msg["id"] for msg in kept] == ["m16", "m17", "m18", "m19", "m20"]
    assert cost == 1


def test_build_ai_messages_uses_summary_and_retrieved_messages():
    from api.index import build_ai_messages

    chat_history = [
        {"id": "m96", "role": "user", "text": "msg 96", "timestamp": 96},
        {"id": "m100", "role": "assistant", "text": "msg 100", "timestamp": 100},
    ]

    result = build_ai_messages(
        {"from": {"first_name": "astro"}, "chat": {"type": "group"}},
        chat_history,
        "que paso hoy",
        summary_text="summary abc",
        retrieved_messages=[{"role": "user", "text": "old hit"}],
    )

    rendered = [
        item["content"] if isinstance(item["content"], str) else item["content"][0]["text"]
        for item in result
    ]
    assert any("summary abc" in part for part in rendered)
    assert any("old hit" in part for part in rendered)
    assert any("msg 100" in part for part in rendered)


def test_prepare_chat_memory_uses_searchable_full_history_for_long_gap(monkeypatch):
    prepare_chat_memory = index.app_runtime.summary.prepare_memory

    recent_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i} for i in range(81, 101)
    ]
    full_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i} for i in range(1, 101)
    ]

    monkeypatch.setattr("api.index.app_runtime.state.get_chat_summary", lambda *_: None)
    monkeypatch.setattr("api.index.app_runtime.state.get_chat_compacted_until", lambda *_: None)
    monkeypatch.setattr(
        "api.index.app_runtime.state.fetch_for_compaction",
        lambda *_args, **_kwargs: full_history,
    )
    monkeypatch.setattr(
        "api.index.app_runtime.state.search_history",
        lambda *_args, **_kwargs: [
            {"id": "m12", "role": "user", "text": "old hit", "timestamp": 12}
        ],
    )
    visible_history, summary_text, retrieved_messages, summary_cost, plan = prepare_chat_memory(
        MagicMock(),
        "123",
        recent_history,
        "que paso hoy",
    )

    assert summary_text is None
    assert [msg["id"] for msg in visible_history] == [f"m{i}" for i in range(1, 101)]
    assert retrieved_messages == [{"id": "m12", "role": "user", "text": "old hit", "timestamp": 12}]
    assert summary_cost == 0
    assert plan is not None
    assert plan.target_marker == "m75"
    assert [msg["id"] for msg in plan.messages] == [f"m{i}" for i in range(1, 76)]


def test_prepare_chat_memory_ignores_marker_without_internal_summary(monkeypatch):
    prepare_chat_memory = index.app_runtime.summary.prepare_memory

    chat_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i} for i in range(1, 101)
    ]
    monkeypatch.setattr("api.index.app_runtime.state.get_chat_summary", lambda *_: None)
    monkeypatch.setattr("api.index.app_runtime.state.get_chat_compacted_until", lambda *_: "m80")
    monkeypatch.setattr(
        "api.index.app_runtime.state.fetch_for_compaction",
        lambda *_args, **_kwargs: chat_history,
    )
    monkeypatch.setattr("api.index.app_runtime.state.search_history", lambda *_args, **_kwargs: [])
    prepared = prepare_chat_memory(MagicMock(), "123", chat_history, "que paso")

    plan = prepared[4]
    assert plan is not None
    assert plan.expected_marker is None
    assert plan.prior_summary is None


def test_stream_summary_command_uses_internal_chat_memory(monkeypatch):
    stream_summary_command = index.app_runtime.summary.stream_command

    redis_client = MagicMock()
    history = [
        {"id": "m1", "role": "user", "text": "msg 1", "timestamp": 1},
        {"id": "m2", "role": "user", "text": "msg 2", "timestamp": 2},
    ]
    response_meta = {}
    stream_chunk = SimpleNamespace(
        id="generation-summary",
        model="deepseek/deepseek-v4-flash",
        provider="DeepInfra",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content="resumen",
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.00001,
        },
    )
    create_completion = MagicMock(return_value=iter([stream_chunk]))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_completion),
        )
    )

    monkeypatch.setattr("api.index.app_runtime.state.get_history", lambda *_: history)
    monkeypatch.setattr(
        "api.index.app_runtime.summary.prepare_memory",
        lambda *_args, **_kwargs: (
            [{"id": "m2", "role": "user", "text": "msg 2", "timestamp": 2}],
            "[contexto anterior: msg 1]",
            [],
            0,
        ),
    )
    monkeypatch.setattr("api.index.app_runtime.summary.load_personality", lambda: "bot")
    monkeypatch.setattr(
        "api.index.app_runtime.summary._deps.provider.get_openrouter_client",
        lambda **_kwargs: client,
    )

    iterator, pending_marker = stream_summary_command(
        "123",
        redis_client,
        "resumen",
        response_meta=response_meta,
    )

    assert list(iterator) == [("openrouter", "resumen")]
    assert pending_marker is None
    create_completion.assert_called_once()
    request = create_completion.call_args.kwargs
    assert request["stream"] is True
    assert "tools" not in request
    assert request["messages"][0] == {"role": "system", "content": "bot"}
    assert request["messages"][1] == {
        "role": "assistant",
        "content": "[contexto anterior: msg 1]",
    }
    assert request["messages"][2]["content"] == "msg 2"
    assert response_meta["billing_segments"][0]["kind"] == "summary"
    assert response_meta["billing_segments"][0]["model"] == "deepseek/deepseek-v4-flash"
    assert response_meta["billing_segments"][0]["metadata"]["upstream_provider"] == "DeepInfra"
    assert response_meta["billing_segments"][0]["usage"]["cost"] == 0.00001


def test_fetch_chat_messages_for_compaction_uses_tag_only_query():
    from api.memory.state import fetch_chat_messages_for_compaction

    redis_client = MagicMock()
    redis_client.execute_command.side_effect = [[0], [0]]

    fetch_chat_messages_for_compaction(redis_client, "123456789")

    query = redis_client.execute_command.call_args.args[2]
    assert query == "@chat_id:{123456789}"
    assert "*" not in query


def test_fetch_chat_messages_for_compaction_fetches_newest_window_then_sorts():
    from api.memory.state import fetch_chat_messages_for_compaction

    redis_client = MagicMock()
    redis_client.execute_command.return_value = [
        3,
        "chatmsg:123:103",
        ["message_id", "103", "id", "103", "text", "newest", "timestamp", "3"],
        "chatmsg:123:102",
        ["message_id", "102", "id", "102", "text", "middle", "timestamp", "2"],
        "chatmsg:123:101",
        ["message_id", "101", "id", "101", "text", "oldest", "timestamp", "1"],
    ]

    rows = fetch_chat_messages_for_compaction(redis_client, "123", limit=3)

    command_args = redis_client.execute_command.call_args.args
    sortby_idx = command_args.index("SORTBY")
    assert command_args[sortby_idx : sortby_idx + 3] == ("SORTBY", "timestamp", "DESC")
    assert [row["message_id"] for row in rows] == ["101", "102", "103"]
    assert [row["id"] for row in rows] == ["101", "102", "103"]


def test_build_incremental_summary_source_with_text_field():
    _build_incremental_summary_source = index.app_runtime.summary.build_incremental_source

    history = [
        {"id": "m1", "role": "user", "text": "hola"},
        {"id": "m2", "role": "assistant", "text": "chau"},
    ]
    source = _build_incremental_summary_source(history, None, None)
    assert source.is_zero_delta is False
    assert [msg["text"] for msg in source.delta_messages] == ["hola", "chau"]
