from tests.support import *


def test_incremental_summary_helper_uses_only_messages_after_marker():
    from api.index import _build_incremental_summary_source

    history = [
        {"id": f"m{i}", "role": "user", "content": f"msg {i}"}
        for i in range(1, 6)
    ]

    source = _build_incremental_summary_source(history, "old summary", "m3")

    assert [msg["id"] for msg in source.delta_messages] == ["m4", "m5"]
    assert source.is_zero_delta is False
    assert source.next_marker == "m5"


def test_incremental_summary_helper_reports_zero_delta_without_history_fallback():
    from api.index import _build_incremental_summary_source

    history = [
        {"id": f"m{i}", "role": "user", "content": f"msg {i}"}
        for i in range(1, 4)
    ]

    source = _build_incremental_summary_source(history, "old summary", "m3")

    assert source.delta_messages == []
    assert source.is_zero_delta is True
    assert source.next_marker is None


def test_incremental_summary_helper_falls_back_to_all_history_when_marker_missing():
    from api.index import _build_incremental_summary_source

    history = [
        {"id": f"m{i}", "role": "user", "content": f"msg {i}"}
        for i in range(1, 4)
    ]

    source = _build_incremental_summary_source(history, "old summary", "m99")

    assert [msg["id"] for msg in source.delta_messages] == ["m1", "m2", "m3"]
    assert source.is_zero_delta is False
    assert source.next_marker == "m3"


def test_compact_chat_memory_absorbs_only_uncompacted_messages_once():
    from api.index import compact_chat_memory

    redis_client = MagicMock()
    messages = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i}
        for i in range(1, 21)
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
    from api.index import prepare_chat_memory

    recent_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i}
        for i in range(81, 101)
    ]
    full_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i}
        for i in range(1, 101)
    ]

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: None)
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: None)
    monkeypatch.setattr(
        "api.index._state_fetch_chat_messages_for_compaction",
        lambda *_args, **_kwargs: full_history,
    )
    monkeypatch.setattr(
        "api.index._state_search_chat_history",
        lambda *_args, **_kwargs: [{"id": "m12", "role": "user", "text": "old hit", "timestamp": 12}],
    )
    monkeypatch.setattr(
        "api.index.compact_chat_memory",
        lambda *_args, **_kwargs: (
            "summary from full history",
            full_history[-5:],
            "m95",
            1,
        ),
    )

    visible_history, summary_text, retrieved_messages, summary_cost = prepare_chat_memory(
        MagicMock(),
        "123",
        recent_history,
        "que paso hoy",
    )

    assert summary_text == "summary from full history"
    assert [msg["id"] for msg in visible_history] == ["m96", "m97", "m98", "m99", "m100"]
    assert retrieved_messages == [{"id": "m12", "role": "user", "text": "old hit", "timestamp": 12}]
    assert summary_cost == 1


def test_prepare_chat_memory_ignores_marker_without_internal_summary(monkeypatch):
    from api.index import prepare_chat_memory

    chat_history = [
        {"id": f"m{i}", "role": "user", "text": f"msg {i}", "timestamp": i}
        for i in range(1, 101)
    ]
    captured = {}

    def fake_compact_chat_memory(
        _redis_client,
        _chat_id,
        messages,
        existing_summary,
        compacted_until,
        **_kwargs,
    ):
        captured["existing_summary"] = existing_summary
        captured["compacted_until"] = compacted_until
        return existing_summary, messages, compacted_until, 0

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: None)
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: "m80")
    monkeypatch.setattr(
        "api.index._state_fetch_chat_messages_for_compaction",
        lambda *_args, **_kwargs: chat_history,
    )
    monkeypatch.setattr("api.index._state_search_chat_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("api.index.compact_chat_memory", fake_compact_chat_memory)

    prepare_chat_memory(MagicMock(), "123", chat_history, "que paso")

    assert captured["existing_summary"] is None
    assert captured["compacted_until"] is None


def test_stream_summary_command_uses_user_compaction_marker(monkeypatch):
    from api.index import stream_summary_command

    redis_client = MagicMock()
    history = [
        {"id": "m1", "role": "user", "text": "msg 1", "timestamp": 1},
        {"id": "m2", "role": "user", "text": "msg 2", "timestamp": 2},
    ]

    monkeypatch.setattr("api.index._state_get_user_chat_summary", lambda *_: "cached summary")
    monkeypatch.setattr("api.index._state_get_user_chat_compacted_until", lambda *_: "m2")
    monkeypatch.setattr(
        "api.index._state_get_chat_compacted_until",
        lambda *_: (_ for _ in ()).throw(AssertionError("used internal compaction marker")),
    )
    monkeypatch.setattr("api.index.get_chat_history", lambda *_: history)

    iterator, pending_marker = stream_summary_command("123", redis_client, "resumen")

    assert list(iterator) == [("cache", "cached summary")]
    assert pending_marker is None


def test_fetch_chat_messages_for_compaction_uses_tag_only_query():
    from api.message_state import fetch_chat_messages_for_compaction

    redis_client = MagicMock()
    redis_client.execute_command.side_effect = [[0], [0]]

    fetch_chat_messages_for_compaction(redis_client, "5162530")

    query = redis_client.execute_command.call_args_list[1].args[2]
    assert query == "@chat_id:{5162530}"
    assert "*" not in query


def test_fetch_chat_messages_for_compaction_fetches_newest_window_then_sorts():
    from api.message_state import fetch_chat_messages_for_compaction

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
    from api.index import _build_incremental_summary_source

    history = [
        {"id": "m1", "role": "user", "text": "hola"},
        {"id": "m2", "role": "assistant", "text": "chau"},
    ]
    source = _build_incremental_summary_source(history, None, None)
    assert source.is_zero_delta is False
    assert [msg["text"] for msg in source.delta_messages] == ["hola", "chau"]
