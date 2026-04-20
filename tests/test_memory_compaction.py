from tests.support import *


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
        compact_fn=lambda text: ("new summary", 1),
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
