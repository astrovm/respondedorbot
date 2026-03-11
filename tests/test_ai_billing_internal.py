from unittest.mock import MagicMock

from api.ai_billing import (
    AIMessageBilling,
    build_insufficient_credits_message,
    get_ai_billing_packs,
    parse_topup_payload,
)
from api.groq_billing import (
    calculate_billing_for_segments,
    estimate_compound_reserve_credits,
    estimate_vision_reserve_credits,
)




def test_get_ai_billing_packs_default_includes_50_credit_option(monkeypatch):
    monkeypatch.delenv("AI_STARS_PACKS_JSON", raising=False)

    packs = get_ai_billing_packs()

    assert packs[0] == {"id": "p50", "credits": 50, "xtr": 25}

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


def test_calculate_billing_for_segments_ignores_cached_token_discount():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 1_000,
                    "input_cached_tokens": 900,
                    "input_non_cached_tokens": 100,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 2_500
    assert breakdown["charged_credits"] == 1
    assert breakdown["model_breakdown"] == [
        {
            "model": "moonshotai/kimi-k2-instruct-0905",
            "usd_micros": 2_500,
            "input_tokens": 1_000,
            "output_tokens": 500,
        }
    ]


def test_calculate_billing_for_segments_reads_compound_usage_breakdown_models_and_tools():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "model": "groq/compound",
                "usage_breakdown": {
                    "models": [
                        {
                            "model": "openai/gpt-oss-120b",
                            "input_tokens": 10_000,
                            "output_tokens": 500,
                        }
                    ]
                },
                "executed_tools": [
                    {"type": "search", "mode": "basic", "count": 2},
                    {"type": "visit", "count": 3},
                    {"name": "python", "duration_seconds": 10},
                    {"type": "browser", "duration_seconds": 15},
                ],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 15_633
    assert breakdown["charged_credits"] == 4
    assert breakdown["model_breakdown"] == [
        {
            "model": "openai/gpt-oss-120b",
            "usd_micros": 1_800,
            "input_tokens": 10_000,
            "output_tokens": 500,
        }
    ]
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "search",
            "usd_micros": 10_000,
            "count": 2,
            "duration_seconds": 0.0,
            "note": "",
        },
        {
            "tool": "visit",
            "usd_micros": 3_000,
            "count": 3,
            "duration_seconds": 0.0,
            "note": "",
        },
        {
            "tool": "python",
            "usd_micros": 500,
            "count": 1,
            "duration_seconds": 10.0,
            "note": "",
        },
        {
            "tool": "browser",
            "usd_micros": 333,
            "count": 1,
            "duration_seconds": 15.0,
            "note": "",
        },
    ]


def test_calculate_billing_for_segments_uses_max_estimate_for_missing_duration_tools():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "model": "groq/compound",
                "usage_breakdown": {
                    "models": [
                        {
                            "model": "openai/gpt-oss-120b",
                            "input_tokens": 10_000,
                            "output_tokens": 500,
                        }
                    ]
                },
                "executed_tools": [
                    {"type": "browser_automation"},
                    {"type": "browser_automation"},
                    {"name": "python"},
                    {"type": "search", "mode": "basic", "count": 1},
                ],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 9_800
    assert breakdown["charged_credits"] == 3
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "browser_automation",
            "usd_micros": 0,
            "count": 1,
            "duration_seconds": 0.0,
            "note": "estimated_request_cap_shared",
        },
        {
            "tool": "browser_automation",
            "usd_micros": 0,
            "count": 1,
            "duration_seconds": 0.0,
            "note": "estimated_request_cap_shared",
        },
        {
            "tool": "python",
            "usd_micros": 3_000,
            "count": 1,
            "duration_seconds": 0.0,
            "note": "estimated_max_request_duration",
        },
        {
            "tool": "search",
            "usd_micros": 5_000,
            "count": 1,
            "duration_seconds": 0.0,
            "note": "",
        },
    ]
    assert breakdown["unsupported_notes"] == []


def test_estimate_vision_reserve_credits_uses_real_image_payload_size():
    small = estimate_vision_reserve_credits(
        prompt_text="Describe what you see in this image in detail.",
        image_data=b"a" * 128,
    )
    large = estimate_vision_reserve_credits(
        prompt_text="Describe what you see in this image in detail.",
        image_data=b"a" * 200_000,
    )

    assert small == 1
    assert large > small


def test_estimate_compound_reserve_credits_only_reserves_predictable_request_tools():
    reserve = estimate_compound_reserve_credits(
        system_message={"role": "system", "content": "search the web"},
        messages=[{"role": "user", "content": "btc news"}],
        enabled_tools=["web_search", "visit_website", "code_interpreter", "browser_automation"],
    )

    assert reserve == 3


def _build_billing_helper() -> AIMessageBilling:
    return AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        get_ai_credits_per_response_fn=lambda: 1,
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        message={"from": {"first_name": "Ana"}},
    )


def test_settle_reserved_ai_credits_does_not_refund_successful_unused_reserve():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credits": 3,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.charge_ai_credits.assert_not_called()


def test_settle_reserved_ai_credits_charges_extra_when_actual_exceeds_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {"ok": True}

    billing.settle_reserved_ai_credits(
        {
            "reserved_credits": 1,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 2000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 2
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["event_type"] == "ai_settlement_charge"
    billing.credits_db_service.refund_ai_charge.assert_not_called()


def test_settle_reserved_ai_credits_records_debt_when_extra_charge_fails():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {"ok": False}

    billing.settle_reserved_ai_credits(
        {
            "reserved_credits": 1,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 2000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    billing.credits_db_service.apply_ai_debt.assert_called_once()
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["amount"] == 2
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["source"] == "user"
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["event_type"] == "ai_settlement_debt"
    billing.credits_db_service.refund_ai_charge.assert_not_called()


def test_settle_reserved_ai_credits_without_usage_keeps_reserved_charge():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credits": 2,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "image_context_media",
        },
        [],
        reason="image_context_media_success",
    )

    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.admin_reporter.assert_called_once_with(
        "respuesta IA exitosa sin usage billing; se mantiene cobro por reserva (sin reintegro)",
        None,
        {
            "chat_id": "1",
            "user_id": 1,
            "reason": "image_context_media_success",
            "reserved_credits": 2,
        },
    )
