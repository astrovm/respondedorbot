from api.chat_context import (
    extract_numeric_chat_id,
    extract_user_id,
    format_user_identity,
    is_group_chat_type,
)


def test_chat_context_helpers_handle_common_shapes():
    assert is_group_chat_type("group") is True
    assert is_group_chat_type("private") is False
    assert extract_numeric_chat_id("-100123") == -100123
    assert extract_numeric_chat_id("abc") is None
    assert extract_user_id({"from": {"id": "42"}}) == 42
    assert extract_user_id({"from": {}}) is None


def test_format_user_identity_uses_first_name_and_username():
    assert format_user_identity({"first_name": "Ana", "username": "ana"}) == "Ana (ana)"
    assert format_user_identity({"first_name": "Ana"}) == "Ana"
