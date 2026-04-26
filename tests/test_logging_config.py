from api.logging_config import format_log_context


def test_format_log_context_includes_stable_debug_fields():
    context = format_log_context(
        {
            "chat_id": "-1001",
            "chat_title": "debug chat",
            "message_id": 42,
            "user_id": 7,
            "ignored": "value",
        }
    )

    assert context == " chat_id=-1001 chat_title=debug chat message_id=42 user_id=7"


def test_format_log_context_omits_empty_values_and_truncates_long_values():
    context = format_log_context(
        {
            "source": "unit-test",
            "chat_id": "",
            "chat_title": "x" * 130,
            "user_id": None,
        }
    )

    assert context.startswith(" source=unit-test chat_title=")
    assert context.endswith("...")
    assert "chat_id=" not in context
    assert "user_id=" not in context
