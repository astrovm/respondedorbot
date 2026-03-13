from unittest.mock import MagicMock

from api.ai_billing import (
    AIMessageBilling,
    build_insufficient_credits_message,
    get_ai_billing_packs,
    parse_topup_payload,
)
from api.credit_units import whole_credits_to_units
from api.groq_billing import (
    MAX_UNDOCUMENTED_TIME_BASED_TOOL_SECONDS_PER_REQUEST,
    calculate_billing_for_segments,
    estimate_compound_reserve_credits,
    estimate_vision_reserve_credits,
)




def test_get_ai_billing_packs_default_includes_50_credit_option(monkeypatch):
    monkeypatch.delenv("AI_STARS_PACKS_JSON", raising=False)

    packs = get_ai_billing_packs()

    assert packs[0] == {"id": "p50", "credits": 500, "xtr": 25}

def test_parse_topup_payload_accepts_optional_user_id():
    assert parse_topup_payload("topup:p250:99") == ("p250", 99)
    assert parse_topup_payload("topup:p250") == ("p250", None)
    assert parse_topup_payload("other") == (None, None)



def test_get_ai_billing_packs_accept_decimal_credits(monkeypatch):
    monkeypatch.setenv(
        "AI_STARS_PACKS_JSON",
        '[{"id":"p15","credits":1.5,"xtr":5}]',
    )

    assert get_ai_billing_packs() == [{"id": "p15", "credits": 15, "xtr": 5}]


def test_build_insufficient_credits_message_mentions_group_balances():
    message = build_insufficient_credits_message(
        chat_type="group",
        user_balance=whole_credits_to_units(2),
        chat_balance=whole_credits_to_units(5),
    )
    assert "lo tuyo: 2.0" in message
    assert "lo del grupo: 5.0" in message


def test_ai_message_billing_transcribe_success_response_prefixes():
    billing = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
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


def test_calculate_billing_for_segments_applies_cached_token_discount():
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

    assert breakdown["raw_usd_micros"] == 2_050
    assert breakdown["charged_credit_units"] == 5
    assert breakdown["charged_credits_display"] == "0.5"
    assert breakdown["model_breakdown"] == [
        {
            "model": "moonshotai/kimi-k2-instruct-0905",
            "usd_micros": 2_050,
            "input_tokens": 1_000,
            "input_cached_tokens": 900,
            "input_non_cached_tokens": 100,
            "output_tokens": 500,
        }
    ]


def test_calculate_billing_for_segments_reads_cached_tokens_from_prompt_token_details():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "prompt_tokens": 2_000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 1_500},
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 1_550
    assert breakdown["charged_credit_units"] == 4
    assert breakdown["charged_credits_display"] == "0.4"
    assert breakdown["model_breakdown"] == [
        {
            "model": "moonshotai/kimi-k2-instruct-0905",
            "usd_micros": 1_550,
            "input_tokens": 2_000,
            "input_cached_tokens": 1_500,
            "input_non_cached_tokens": 500,
            "output_tokens": 100,
        }
    ]


def test_calculate_billing_for_segments_skips_cached_source_segments():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "source": "cache",
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
                "executed_tools": [{"type": "search", "mode": "basic", "count": 1}],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["charged_credit_units"] == 0
    assert breakdown["charged_credits_display"] == "0.0"
    assert breakdown["model_breakdown"] == []
    assert breakdown["tool_breakdown"] == []
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_reads_compound_usage_breakdown_models_and_tools():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "model": "groq/compound",
                "metadata": {"request_elapsed_seconds": 30},
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
                    {"name": "python"},
                    {"type": "browser"},
                ],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 16_300
    assert breakdown["charged_credit_units"] == 33
    assert breakdown["charged_credits_display"] == "3.3"
    assert breakdown["model_breakdown"] == [
        {
            "model": "openai/gpt-oss-120b",
            "usd_micros": 1_800,
            "input_tokens": 10_000,
            "input_cached_tokens": 0,
            "input_non_cached_tokens": 10_000,
            "output_tokens": 500,
        }
    ]
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "search",
            "usd_micros": 10_000,
            "count": 2,
            "note": "",
        },
        {
            "tool": "visit",
            "usd_micros": 3_000,
            "count": 3,
            "note": "",
        },
        {
            "tool": "python",
            "usd_micros": 1_500,
            "count": 1,
            "note": "estimated_from_request_elapsed_seconds",
        },
        {
            "tool": "browser",
            "usd_micros": 0,
            "count": 1,
            "note": "estimated_shared_cap_from:estimated_from_request_elapsed_seconds",
        },
    ]
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_uses_usage_total_time_for_browser_tools():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "model": "groq/compound",
                "usage": {"total_time": 30},
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
                    {"type": "browser"},
                    {"type": "browser_automation"},
                    {"type": "search", "mode": "basic", "count": 1},
                ],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 7_466
    assert breakdown["charged_credit_units"] == 15
    assert breakdown["charged_credits_display"] == "1.5"
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "browser",
            "usd_micros": 666,
            "count": 1,
            "note": "estimated_from_usage_total_time",
        },
        {
            "tool": "browser_automation",
            "usd_micros": 0,
            "count": 1,
            "note": "estimated_shared_cap_from:estimated_from_usage_total_time",
        },
        {
            "tool": "search",
            "usd_micros": 5_000,
            "count": 1,
            "note": "",
        },
    ]
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_falls_back_to_120_second_time_cap():
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
                    {"type": "browser"},
                    {"type": "search", "mode": "basic", "count": 1},
                ],
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 9_466
    assert breakdown["charged_credit_units"] == 19
    assert breakdown["charged_credits_display"] == "1.9"
    assert breakdown["model_breakdown"] == [
        {
            "model": "openai/gpt-oss-120b",
            "usd_micros": 1_800,
            "input_tokens": 10_000,
            "input_cached_tokens": 0,
            "input_non_cached_tokens": 10_000,
            "output_tokens": 500,
        }
    ]
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "browser",
            "usd_micros": 2_666,
            "count": 1,
            "note": "estimated_max_120_second_request_cap",
        },
        {
            "tool": "search",
            "usd_micros": 5_000,
            "count": 1,
            "note": "",
        },
    ]
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_clamps_measured_time_to_120_seconds():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "compound",
                "model": "groq/compound",
                "metadata": {"request_elapsed_seconds": 180},
                "usage_breakdown": {
                    "models": [
                        {
                            "model": "openai/gpt-oss-120b",
                            "input_tokens": 10_000,
                            "output_tokens": 500,
                        }
                    ]
                },
                "executed_tools": [{"name": "python"}],
            }
        ]
    )

    assert MAX_UNDOCUMENTED_TIME_BASED_TOOL_SECONDS_PER_REQUEST == 120.0
    assert breakdown["tool_breakdown"] == [
        {
            "tool": "python",
            "usd_micros": 6_000,
            "count": 1,
            "note": "estimated_from_request_elapsed_seconds",
        }
    ]


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

    assert reserve == 19


def _build_billing_helper() -> AIMessageBilling:
    return AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        message={"from": {"first_name": "Ana"}},
    )


def test_settle_reserved_ai_credits_refunds_successful_unused_reserve():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(3),
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

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 29
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_charges_extra_when_actual_exceeds_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {"ok": True}

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(1),
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
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 10
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["event_type"] == "ai_settlement_charge"
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_reserve_ai_credits_reuses_persisted_reservation_without_new_charge():
    persisted_reservation = {
        "reserved_credit_units": whole_credits_to_units(2),
        "chat_scope_id": 1,
        "source": "user",
        "usage_tag": "ai_response_base",
        "metadata": {"cached": True},
    }
    billing = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        message={"from": {"first_name": "Ana"}},
        load_persisted_reservation_fn=lambda usage_tag: (
            persisted_reservation if usage_tag == "ai_response_base" else None
        ),
    )

    reservation_meta, error = billing.reserve_ai_credits(
        "ai_response_base",
        whole_credits_to_units(3),
    )

    assert error is None
    assert reservation_meta == persisted_reservation
    billing.credits_db_service.charge_ai_credits.assert_not_called()


def test_build_insufficient_credits_reply_uses_username_when_first_name_is_missing():
    billing = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda name: f"random:{name}",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        message={"from": {"username": "ana_user"}},
    )

    reply = billing._build_insufficient_credits_reply(
        {
            "user_balance_credit_units": 0,
            "chat_balance_credit_units": 0,
        }
    )

    assert reply.startswith("random:ana_user")


def test_settle_reserved_ai_credits_records_debt_when_extra_charge_fails():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {"ok": False}

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(1),
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
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["amount"] == 10
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["source"] == "user"
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["event_type"] == "ai_settlement_debt"
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_batch_converts_to_credits_once_and_refunds_overreserve():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits_batch(
        [
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "ai_response_base",
            },
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "image_context_media",
            },
        ],
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "vision",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 19
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_batch_charges_extra_once_when_total_exceeds_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {"ok": True}

    billing.settle_reserved_ai_credits_batch(
        [
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "ai_response_base",
            },
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "image_context_media",
            },
        ],
        [
            {
                "kind": "vision",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 2000,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_keeps_reserve_when_groq_reports_zero_usage():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(3),
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["billing_zero_usage_fallback"] is True
    assert metadata["settled_credit_units"] == 30
    assert metadata["refunded_credit_units"] == 0


def test_settle_reserved_ai_credits_batch_keeps_full_reserve_when_total_usage_is_zero():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits_batch(
        [
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "ai_response_base",
            },
            {
                "reserved_credit_units": whole_credits_to_units(1),
                "chat_scope_id": 1,
                "source": "user",
                "usage_tag": "image_context_media",
            },
        ],
        [
            {
                "kind": "chat",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
            {
                "kind": "vision",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["billing_zero_usage_fallback"] is True
    assert metadata["settled_credit_units"] == 20
    assert metadata["refunded_credit_units"] == 0


def test_settle_reserved_ai_credits_without_usage_keeps_reserved_charge():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(2),
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "image_context_media",
        },
        [],
        reason="image_context_media_success",
    )

    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    billing.admin_reporter.assert_called_once_with(
        "respuesta IA exitosa sin usage billing; se mantiene cobro por reserva (sin reintegro)",
        None,
        {
            "chat_id": "1",
            "user_id": 1,
            "reason": "image_context_media_success",
            "reserved_credit_units": 20,
        },
    )
