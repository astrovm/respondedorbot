from unittest.mock import MagicMock

from api.ai_billing import AIMessageBilling, build_insufficient_credits_message, parse_topup_payload


def test_parse_topup_payload_accepts_optional_user_id():
    assert parse_topup_payload("topup:p250:99") == ("p250", 99)
    assert parse_topup_payload("topup:p250") == ("p250", None)
    assert parse_topup_payload("other") == (None, None)


def test_build_insufficient_credits_message_mentions_group_balances():
    message = build_insufficient_credits_message(
        chat_type="group",
        user_balance=2,
        chat_balance=5,
    )
    assert "lo tuyo: 2" in message
    assert "lo del grupo: 5" in message


def test_ai_message_billing_transcribe_success_response_prefixes():
    billing = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        get_ai_credits_per_response_fn=lambda: 1,
        command="/transcribe",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        message={"from": {"first_name": "Ana"}},
    )

    assert billing.is_transcribe_success_response("🎵 te saqué esto del audio: hola")
    assert billing.is_transcribe_success_response("🖼️ en la imagen veo: foto")
    assert not billing.is_transcribe_success_response("error")
