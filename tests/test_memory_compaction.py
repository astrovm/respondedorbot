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
