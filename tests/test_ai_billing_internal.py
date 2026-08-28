from unittest.mock import MagicMock

from api.billing.ai import (
    build_insufficient_credits_message,
    get_ai_billing_packs,
    parse_topup_payload,
)
from tests.support import make_ai_message_billing
from api.billing.credit_units import CREDIT_SCALE, whole_credits_to_units
from api.ai.pricing import (
    CHAT_OUTPUT_TOKEN_LIMIT,
    REASONING_CHAT_OUTPUT_TOKEN_LIMIT,
    calculate_billing_for_segments,
    chat_output_token_limit,
    estimate_vision_reserve_credits,
)


def test_get_ai_billing_packs_includes_50_credit_option():
    packs = get_ai_billing_packs()

    assert packs[0] == {"id": "p50", "credits": 5_000, "xtr": 25}


def test_parse_topup_payload_accepts_optional_user_id():
    assert parse_topup_payload("topup:p250:99") == ("p250", 99)
    assert parse_topup_payload("topup:p250") == ("p250", None)
    assert parse_topup_payload("other") == (None, None)


def test_get_ai_billing_packs_returns_default_packs():
    packs = get_ai_billing_packs()

    assert len(packs) == 6
    assert packs[0] == {"id": "p50", "credits": 5_000, "xtr": 25}
    assert packs[-1] == {"id": "p2500", "credits": 250_000, "xtr": 1250}


def test_build_insufficient_credits_message_mentions_group_balances():
    message = build_insufficient_credits_message(
        chat_type="group",
        user_balance=whole_credits_to_units(2),
        chat_balance=whole_credits_to_units(5),
    )
    assert "lo tuyo: 2.00" in message
    assert "lo del grupo: 5.00" in message


def test_chat_output_token_limit_is_model_specific():
    assert (
        chat_output_token_limit("deepseek/deepseek-v4-flash-0731")
        == REASONING_CHAT_OUTPUT_TOKEN_LIMIT
        == 8192
    )
    assert chat_output_token_limit("deepseek/deepseek-v4-flash-0731:exacto") == 8192
    assert chat_output_token_limit("other/model") == CHAT_OUTPUT_TOKEN_LIMIT == 1024


def test_ai_message_billing_transcribe_success_response_prefixes():
    billing = make_ai_message_billing(
        command="/transcribe",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
    )

    assert billing.is_transcribe_success_response("🎵 te saqué esto del audio: hola")
    assert billing.is_transcribe_success_response("🖼️ en la imagen veo: foto")
    assert not billing.is_transcribe_success_response("error")


def test_calculate_billing_for_segments_applies_cached_token_discount():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 1_000,
                    "input_cached_tokens": 900,
                    "input_non_cached_tokens": 100,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 59
    assert breakdown["charged_credit_units"] == 2
    assert breakdown["charged_credits_display"] == "0.02"
    assert breakdown["model_breakdown"] == [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usd_micros": 59,
            "input_tokens": 1_000,
            "input_cached_tokens": 900,
            "input_non_cached_tokens": 100,
            "output_tokens": 500,
        }
    ]


def test_calculate_billing_for_segments_normalizes_billing_model_ids():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "transcribe",
                "model": "groq/whisper-large-v3",
                "audio_seconds": 60,
            },
        ]
    )

    assert breakdown["raw_usd_micros"] > 0
    assert [item["model"] for item in breakdown["model_breakdown"]] == [
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-flash-0731",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-flash-lite-preview",
        "groq/whisper-large-v3",
    ]


def test_calculate_billing_for_segments_bumps_pricing_version_for_deepseek_search_billing():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "metadata": {"web_search_requests": 1},
            }
        ]
    )

    assert breakdown["pricing_version"] != "2026-03-06"


def test_calculate_billing_for_segments_reads_cached_tokens_from_prompt_token_details():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 2_000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 1_500},
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 35
    assert breakdown["charged_credit_units"] == 1
    assert breakdown["charged_credits_display"] == "0.01"
    assert breakdown["model_breakdown"] == [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usd_micros": 35,
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
                "kind": "chat",
                "source": "cache",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 10_000,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["charged_credit_units"] == 0
    assert breakdown["charged_credits_display"] == "0.00"
    assert breakdown["model_breakdown"] == []
    assert breakdown["tool_breakdown"] == []
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_bills_successful_firecrawl_credits():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "metadata": {
                    "web_search_requests": 2,
                    "firecrawl_credits_used": 4,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 3_328
    assert breakdown["charged_credit_units"] == 67
    assert breakdown["tool_breakdown"] == [{"tool": "web_search", "count": 2, "usd_micros": 3_320}]


def test_calculate_billing_for_segments_refunds_cache_only_usage_to_zero():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "source": "cache",
                "usage": {
                    "input_tokens": 10_000,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["charged_credit_units"] == 0
    assert breakdown["charged_credits_display"] == "0.00"


def test_estimate_vision_reserve_credits_uses_real_image_payload_size():
    small = estimate_vision_reserve_credits(
        prompt_text="Describe what you see in this image in detail.",
        image_data=b"a" * 128,
    )
    large = estimate_vision_reserve_credits(
        prompt_text="Describe what you see in this image in detail.",
        image_data=b"a" * 200_000,
    )

    assert small == 16
    assert large > small


def _build_billing_helper():
    return make_ai_message_billing(
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 299
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_unresolved_operation_persists_every_segment_before_reconciliation():
    billing = _build_billing_helper()
    reservation = {
        "reserved_credit_units": 20,
        "chat_scope_id": None,
        "source": "user",
        "usage_tag": "ai_response_base",
        "metadata": {"operation_id": "operation-1"},
    }
    pending = {
        "kind": "chat",
        "model": "deepseek/deepseek-v4-flash-0731",
        "source": "openrouter",
        "metadata": {
            "provider_generation_id": "generation-pending",
            "provider_usage_pending": True,
        },
    }
    completed = {
        "kind": "chat",
        "model": "deepseek/deepseek-v4-flash-0731",
        "source": "openrouter",
        "usage": {"cost": 0.0001},
        "metadata": {"provider_generation_id": "generation-complete"},
    }

    billing.settle_reserved_ai_credits_batch(
        [reservation],
        [pending, completed],
        reason="ai_response_success",
    )

    persisted = billing.credits_db_service.record_ai_provider_usage.call_args_list
    assert [call.kwargs["segment"] for call in persisted] == [pending, completed]
    billing.credits_db_service.settle_ai_operation_once.assert_not_called()


def test_settle_reserved_ai_credits_charges_extra_when_actual_exceeds_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {
        "ok": True,
        "source": "chat",
    }

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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 40_000,
                    "output_tokens": 55_000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 34
    assert (
        billing.credits_db_service.charge_ai_credits.call_args.kwargs["event_type"]
        == "ai_settlement_charge"
    )
    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["payer_scope"] == "mixed"
    assert metadata["payer_breakdown"] == [
        {"scope": "user", "credit_units": 100},
        {"scope": "chat", "credit_units": 34},
    ]


def test_record_ai_settlement_result_retries_idempotent_write():
    billing = _build_billing_helper()
    billing.credits_db_service.record_ai_settlement_result.side_effect = [
        RuntimeError("temporary database failure"),
        None,
    ]

    billing._record_ai_settlement_result(
        chat_scope_id=1,
        settlement_metadata={"settlement_id": "1:1:1:ai_response_base"},
    )

    assert billing.credits_db_service.record_ai_settlement_result.call_count == 2
    billing.admin_reporter.assert_not_called()


def test_reserve_ai_credits_reuses_persisted_reservation_without_new_charge():
    persisted_reservation = {
        "reserved_credit_units": whole_credits_to_units(2),
        "chat_scope_id": 1,
        "source": "user",
        "usage_tag": "ai_response_base",
        "metadata": {"cached": True},
        "credit_scale": CREDIT_SCALE,
    }
    billing = make_ai_message_billing(
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
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


def test_reserve_ai_credits_rescales_legacy_persisted_reservation():
    billing = make_ai_message_billing(
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        load_persisted_reservation_fn=lambda _usage_tag: {
            "reserved_credit_units": 20,
            "source": "user",
            "usage_tag": "ai_response_base",
            "metadata": {"cached": True},
        },
    )

    reservation_meta, error = billing.reserve_ai_credits(
        "ai_response_base",
        whole_credits_to_units(3),
    )

    assert error is None
    assert reservation_meta["reserved_credit_units"] == 200
    assert reservation_meta["credit_scale"] == CREDIT_SCALE
    billing.credits_db_service.charge_ai_credits.assert_not_called()


def test_build_insufficient_credits_reply_uses_username_when_first_name_is_missing():
    billing = make_ai_message_billing(
        command="/ask",
        chat_id="1",
        chat_type="private",
        user_id=1,
        numeric_chat_id=1,
        gen_random_fn=lambda name: f"random:{name}",
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 40_000,
                    "output_tokens": 55_000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    billing.credits_db_service.apply_ai_debt.assert_called_once()
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["amount"] == 34
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["source"] == "user"
    assert (
        billing.credits_db_service.apply_ai_debt.call_args.kwargs["event_type"]
        == "ai_settlement_debt"
    )
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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 199
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_batch_mixed_sources_refunds_later_reserves():
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
                "source": "chat",
                "usage_tag": "image_context_media",
            },
        ],
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        ],
        reason="ai_response_success",
    )

    assert billing.credits_db_service.refund_ai_charge.call_count == 2
    refunds = billing.credits_db_service.refund_ai_charge.call_args_list
    assert [call.kwargs["amount"] for call in refunds] == [99, 100]
    assert [call.kwargs["source"] for call in refunds] == ["user", "chat"]
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["charged_credit_units_total"] == 1
    assert metadata["settlement_ids"] == [
        "1:1:unknown:ai_response_base",
        "1:1:unknown:image_context_media",
    ]


def test_settle_reserved_ai_credits_batch_mixed_sources_with_missing_billing_keeps_reserved_charge():
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
                "source": "chat",
                "usage_tag": "image_context_media",
            },
        ],
        None,
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["missing_usage_billing"] is True
    assert metadata["billing_zero_usage_fallback"] is False
    assert metadata["refunded_credit_units"] == 0
    assert metadata["payer_scope"] == "mixed"
    assert metadata["payer_breakdown"] == [
        {"scope": "user", "credit_units": 100},
        {"scope": "chat", "credit_units": 100},
    ]
    assert metadata["charged_credit_units_total"] == 200


def test_settle_reserved_ai_credits_batch_empty_segments_keeps_reserved_charge():
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
        [],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
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
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 2000,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 193
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
                "model": "deepseek/deepseek-v4-flash-0731",
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
    assert metadata["settled_credit_units"] == 300
    assert metadata["refunded_credit_units"] == 0


def test_incomplete_pricing_never_reduces_known_overage_to_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {
        "ok": True,
        "source": "user",
    }
    reserved_credit_units = 16
    segments = [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {"cost": "0.0019"},
            "metadata": {"provider": "openrouter"},
        },
        {
            "kind": "chat",
            "model": "unknown/model",
            "usage": {},
            "metadata": {"provider": "openrouter"},
        },
    ]

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": reserved_credit_units,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        segments,
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 22
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["settled_credit_units"] == 38
    assert metadata["pricing_complete"] is False


def test_incomplete_batch_pricing_never_reduces_known_overage_to_reserve():
    billing = _build_billing_helper()
    billing.credits_db_service.charge_ai_credits.return_value = {
        "ok": True,
        "source": "user",
    }
    reservations = [
        {
            "reserved_credit_units": 8,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        {
            "reserved_credit_units": 8,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "image_context_media",
        },
    ]
    segments = [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {"cost": "0.0019"},
            "metadata": {"provider": "openrouter"},
        },
        {
            "kind": "vision",
            "model": "unknown/model",
            "usage": {},
            "metadata": {"provider": "openrouter"},
        },
    ]

    billing.settle_reserved_ai_credits_batch(reservations, segments, reason="ok")

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 22
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["settled_credit_units"] == 38


def test_settle_reserved_ai_credits_refunds_cache_only_usage():
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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 1_000,
                    "prompt_tokens_details": {"cached_tokens": 900},
                    "completion_tokens": 50,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 299
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["settled_credit_units"] == 1
    assert metadata["refunded_credit_units"] == 299
    assert metadata["charged_credit_units_total"] == 1
    assert metadata["payer_scope"] == "user"


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
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
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
    assert metadata["settled_credit_units"] == 200
    assert metadata["refunded_credit_units"] == 0


def test_settle_reserved_ai_credits_batch_ignores_none_segments():
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
            None,
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            },
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.record_ai_settlement_result.call_count == 1


def test_settle_reserved_ai_credits_ignores_none_segments():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(1),
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        [None],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_not_called()
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()


def test_settle_reserved_ai_credits_refunds_transcribe_partial_usage():
    billing = _build_billing_helper()
    reserved_credit_units = whole_credits_to_units(3)
    segments = [
        {
            "kind": "transcribe",
            "model": "groq/whisper-large-v3",
            "audio_seconds": 60,
        }
    ]
    breakdown = calculate_billing_for_segments(segments)

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": reserved_credit_units,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_transcribe",
        },
        segments,
        reason="ok",
    )

    expected_refund = reserved_credit_units - breakdown["charged_credit_units"]
    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == expected_refund
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["settled_credit_units"] == breakdown["charged_credit_units"]
    assert metadata["refunded_credit_units"] == expected_refund


def test_settle_reserved_ai_credits_refunds_partial_chat_usage():
    billing = _build_billing_helper()
    reserved_credit_units = whole_credits_to_units(3)
    segments = [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 100,
            },
        }
    ]
    breakdown = calculate_billing_for_segments(segments)

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": reserved_credit_units,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
        },
        segments,
        reason="ok",
    )

    expected_refund = reserved_credit_units - breakdown["charged_credit_units"]
    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == expected_refund
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs["metadata"]
    assert metadata["settled_credit_units"] == breakdown["charged_credit_units"]
    assert metadata["refunded_credit_units"] == expected_refund


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


def test_settle_reserved_ai_credits_without_billing_segments_keeps_reserved_charge():
    billing = _build_billing_helper()

    billing.settle_reserved_ai_credits(
        {
            "reserved_credit_units": whole_credits_to_units(2),
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "image_context_media",
        },
        None,
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
            "reserved_credit_units": 200,
        },
    )


def test_calculate_billing_uses_gateway_cost_when_higher_than_local():
    # Local pricing for 100 input + 50 output deepseek tokens:
    # (100 * 400_000 + 50 * 1_200_000) // 1_000_000 = 100 usd_micros
    # Gateway cost of $0.005 USD = 5_000 usd_micros -> should win
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 50,
                    "cost": 0.005,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 5_000
    assert breakdown["model_breakdown"][0]["usd_micros"] == 5_000


def test_calculate_billing_uses_reported_gateway_cost_when_lower_than_local():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 4_000,
                    "completion_tokens": 2_000,
                    "cost": 0.001,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 1_000
    assert breakdown["model_breakdown"][0]["usd_micros"] == 1_000


def test_calculate_billing_without_gateway_cost_uses_local():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 8
    assert breakdown["model_breakdown"][0]["usd_micros"] == 8


def test_zero_reported_openrouter_cost_is_incomplete():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 50,
                    "cost": 0,
                },
                "metadata": {
                    "provider": "openrouter",
                    "upstream_provider": "DeepInfra",
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["pricing_complete"] is False
    assert breakdown["segment_breakdown"][0]["pricing_basis"] == "missing"


def test_submicro_provider_costs_are_summed_before_final_rounding():
    segments = [
        {
            "kind": "chat",
            "model": "unknown/model",
            "usage": {"cost": "0.00000003"},
            "metadata": {"provider": "openrouter"},
        },
        {
            "kind": "chat",
            "model": "unknown/model",
            "usage": {"cost": "0.00000003"},
            "metadata": {"provider": "openrouter"},
        },
    ]

    breakdown = calculate_billing_for_segments(segments)

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["raw_usd_micros_exact"] == "0.06000000"
    assert breakdown["charged_credit_units"] == 1
    assert breakdown["pricing_complete"] is True


def test_upstream_inference_cost_wins_over_discounted_gateway_cost():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": "0.000001",
                    "cost_details": {"upstream_inference_cost": "0.00010442124"},
                },
                "metadata": {"provider": "openrouter"},
            }
        ]
    )

    assert breakdown["raw_usd_micros_exact"] == "104.42124000000"
    assert breakdown["charged_credit_units"] == 3
    assert breakdown["segment_breakdown"][0]["pricing_basis"] == "provider_reported"


def test_zero_upstream_cost_uses_reported_gateway_cost():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 50,
                    "cost": "0.000001",
                    "cost_details": {"upstream_inference_cost": 0},
                },
                "metadata": {
                    "provider": "openrouter",
                    "upstream_provider": "DeepInfra",
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros_exact"] == "1.000000"
    assert breakdown["segment_breakdown"][0]["pricing_basis"] == "provider_reported"


def test_invalid_upstream_cost_uses_reported_gateway_cost():
    for upstream_cost in (None, "invalid"):
        breakdown = calculate_billing_for_segments(
            [
                {
                    "kind": "chat",
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "usage": {
                        "prompt_tokens": 1_000,
                        "completion_tokens": 50,
                        "cost": "0.000001",
                        "cost_details": {"upstream_inference_cost": upstream_cost},
                    },
                    "metadata": {"provider": "openrouter"},
                }
            ]
        )

        assert breakdown["raw_usd_micros_exact"] == "1.000000"
        assert breakdown["pricing_complete"] is True
        assert breakdown["segment_breakdown"][0]["pricing_basis"] == "provider_reported"


def test_internal_cache_is_authoritative_zero_cost():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "summary",
                "model": "deepseek/deepseek-v4-flash-0731",
                "source": "cache",
                "cached": True,
            }
        ]
    )

    assert breakdown["charged_credit_units"] == 0
    assert breakdown["pricing_complete"] is True
    assert breakdown["segment_breakdown"][0]["pricing_basis"] == "internal_cache"


def test_calculate_billing_adds_direct_firecrawl_cost_to_reported_model_cost():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.005006,
                },
                "metadata": {
                    "web_search_requests": 1,
                    "firecrawl_credits_used": 2,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 6_666
    assert breakdown["tool_breakdown"] == [{"tool": "web_search", "count": 1, "usd_micros": 1_660}]


def test_calculate_billing_does_not_bill_failed_firecrawl_search():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.005006,
                },
                "metadata": {"web_search_requests": 1},
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 5_006
    assert breakdown["tool_breakdown"] == []


def test_openrouter_transcription_uses_reported_model_cost():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "transcribe",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {
                    "prompt_tokens": 2_000,
                    "completion_tokens": 100,
                    "cost": 0.00125,
                },
                "audio_seconds": 60,
                "metadata": {"provider": "openrouter"},
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 1_250
    assert breakdown["model_breakdown"][0]["model"] == ("google/gemini-3.1-flash-lite-preview")


def test_gemini_local_fallback_prices_cache_audio_and_cache_writes():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {
                        "cached_tokens": 200,
                        "audio_tokens": 300,
                        "cache_write_tokens": 100,
                    },
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 413


def test_openrouter_local_fallback_without_current_endpoint_price_is_incomplete():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "openai/gpt-oss-120b",
                "usage": {"prompt_tokens": 1_000, "completion_tokens": 500},
                "metadata": {
                    "provider": "openrouter",
                    "upstream_provider": "DeepInfra",
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["pricing_complete"] is False


def test_openrouter_without_reported_cost_does_not_use_upstream_provider_rate():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "openai/gpt-oss-120b",
                "usage": {"prompt_tokens": 1_000, "completion_tokens": 500, "cost": 0},
                "metadata": {
                    "provider": "openrouter",
                    "upstream_provider": "Groq",
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["pricing_complete"] is False
    assert breakdown["segment_breakdown"][0]["upstream_provider"] == "groq"


def test_groq_transcription_applies_ten_second_minimum():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "transcribe",
                "model": "groq/whisper-large-v3",
                "audio_seconds": 1,
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 309


def _make_group_billing(*, limit: int, redis_client=None):
    mock_redis = redis_client or MagicMock()
    db = MagicMock()
    db.is_configured.return_value = True
    db.charge_ai_credits.return_value = {"ok": True, "source": "chat"}
    billing = make_ai_message_billing(
        command="/ask",
        chat_id="-100",
        chat_type="group",
        user_id=42,
        numeric_chat_id=100,
        credits_db_service=db,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        redis_client=mock_redis,
        creditless_user_hourly_limit=limit,
    )
    return billing


def test_creditless_cap_allows_under_limit():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1  # first use
    billing = _make_group_billing(limit=3, redis_client=mock_redis)

    result, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert error is None
    assert result is not None
    mock_redis.incr.assert_called_once_with("creditless_cap:-100:42")
    mock_redis.expire.assert_called_once_with("creditless_cap:-100:42", 3600)


def test_creditless_cap_counts_once_for_incremental_reservations():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1
    billing = _make_group_billing(limit=3, redis_client=mock_redis)

    base, base_error = billing.reserve_ai_credits("ai_response_base", 10)
    extension, extension_error = billing.reserve_ai_credits(
        "ai_response_context_extension",
        5,
    )

    assert base_error is None
    assert extension_error is None
    assert base is not None
    assert extension is not None
    assert base["message_cap_counted"] is True
    assert extension["message_cap_counted"] is False
    mock_redis.incr.assert_called_once_with("creditless_cap:-100:42")

    billing.refund_reserved_ai_credits(base, reason="ai_response_fallback")
    billing.refund_reserved_ai_credits(extension, reason="ai_response_fallback")

    mock_redis.decr.assert_called_once_with("creditless_cap:-100:42")


def test_idempotent_reservation_replay_does_not_increment_message_cap():
    mock_redis = MagicMock()
    billing = _make_group_billing(limit=3, redis_client=mock_redis)
    billing.credits_db_service.charge_ai_credits.return_value = {
        "ok": True,
        "applied": False,
        "source": "chat",
        "amount": 10,
    }

    reservation, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert error is None
    assert reservation is not None
    assert reservation["message_cap_counted"] is False
    mock_redis.incr.assert_not_called()


def test_reloaded_idempotent_reservation_does_not_roll_back_message_cap():
    mock_redis = MagicMock()
    persisted_reservation = {
        "reserved_credit_units": 10,
        "chat_scope_id": 100,
        "source": "chat",
        "usage_tag": "ai_response_base",
        "metadata": {},
        "credit_scale": CREDIT_SCALE,
        "message_cap_counted": False,
    }
    billing = make_ai_message_billing(
        command="/ask",
        chat_id="-100",
        chat_type="group",
        user_id=42,
        numeric_chat_id=100,
        credits_db_service=MagicMock(),
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        redis_client=mock_redis,
        creditless_user_hourly_limit=3,
        load_persisted_reservation_fn=lambda _usage_tag: persisted_reservation,
    )

    reservation, error = billing.reserve_ai_credits("ai_response_base", 10)
    billing.refund_reserved_ai_credits(reservation, reason="ai_response_fallback")

    assert error is None
    assert reservation is not None
    assert reservation["message_cap_counted"] is False
    mock_redis.incr.assert_not_called()
    mock_redis.decr.assert_not_called()


def test_background_reservation_does_not_consume_message_cap():
    mock_redis = MagicMock()
    billing = _make_group_billing(limit=0, redis_client=mock_redis)

    result, error = billing.reserve_background_ai_credits(
        "memory_compaction:-100:m1",
        10,
    )

    assert error is None
    assert result is not None
    assert result["source"] == "chat"
    mock_redis.incr.assert_not_called()


def test_creditless_cap_blocks_over_limit_and_refunds():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 4  # over limit=3
    billing = _make_group_billing(limit=3, redis_client=mock_redis)

    result, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert result is None
    assert error is not None
    assert "3" in error
    assert "mensajes de ia pagados por el grupo por hora" in error
    billing.credits_db_service.refund_ai_charge.assert_called_once()
    refund_kwargs = billing.credits_db_service.refund_ai_charge.call_args.kwargs
    assert refund_kwargs["source"] == "chat"
    assert refund_kwargs["event_type"] == "ai_refund"
    assert refund_kwargs["amount"] == 10


def test_creditless_cap_disabled_when_limit_negative():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 999
    billing = _make_group_billing(limit=-1, redis_client=mock_redis)

    result, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert error is None
    assert result is not None
    mock_redis.incr.assert_not_called()


def test_creditless_cap_blocks_always_when_limit_zero():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1
    billing = _make_group_billing(limit=0, redis_client=mock_redis)

    result, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert result is None
    assert error is not None
    assert "0" in error
    billing.credits_db_service.refund_ai_charge.assert_called_once()


def test_refund_reserved_ai_credits_rolls_back_creditless_cap_for_chat_source():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1
    billing = _make_group_billing(limit=3, redis_client=mock_redis)

    reservation, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert error is None
    assert reservation is not None

    billing.refund_reserved_ai_credits(reservation, reason="ai_response_fallback")

    mock_redis.decr.assert_called_once_with("creditless_cap:-100:42")


def test_replayed_reservation_refund_does_not_roll_back_creditless_cap_twice():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1
    billing = _make_group_billing(limit=3, redis_client=mock_redis)
    billing.credits_db_service.refund_ai_charge.side_effect = [
        {"applied": True},
        {"applied": False},
    ]

    reservation, error = billing.reserve_ai_credits("ai_response_base", 10)

    assert error is None
    assert reservation is not None

    billing.refund_reserved_ai_credits(reservation, reason="first_path")
    billing.refund_reserved_ai_credits(reservation, reason="retry_path")

    mock_redis.decr.assert_called_once_with("creditless_cap:-100:42")


def test_reservation_refund_preserves_identity_and_is_reason_independent():
    billing = make_ai_message_billing(
        message={"message_id": 44, "from": {"first_name": "Ana"}},
    )
    billing.credits_db_service.charge_ai_credits.return_value = {
        "ok": True,
        "applied": True,
        "source": "user",
        "amount": 10,
    }
    operation_id = billing.operation_id("ai_response")
    reservation, error = billing.reserve_ai_credits(
        "ai_response_base",
        10,
        metadata={"operation_id": operation_id},
    )
    assert error is None
    assert reservation is not None

    billing.refund_reserved_ai_credits(reservation, reason="first_path")
    billing.refund_reserved_ai_credits(reservation, reason="second_path")

    calls = billing.credits_db_service.refund_ai_charge.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["metadata"]["operation_id"] == operation_id
    assert calls[1].kwargs["metadata"]["operation_id"] == operation_id
    assert calls[0].kwargs["idempotency_key"] == calls[1].kwargs["idempotency_key"]
    assert calls[0].kwargs["idempotency_key"].endswith(":refund")
