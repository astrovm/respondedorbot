from unittest.mock import MagicMock

from api.ai_billing import (
    build_insufficient_credits_message,
    get_ai_billing_packs,
    parse_topup_payload,
)
from tests.support import make_ai_message_billing
from api.credit_units import whole_credits_to_units
from api.ai_pricing import (
    calculate_billing_for_segments,
    estimate_chat_reserve_credits,
    estimate_vision_reserve_credits,
)


def test_get_ai_billing_packs_includes_50_credit_option():
    packs = get_ai_billing_packs()

    assert packs[0] == {"id": "p50", "credits": 500, "xtr": 25}


def test_parse_topup_payload_accepts_optional_user_id():
    assert parse_topup_payload("topup:p250:99") == ("p250", 99)
    assert parse_topup_payload("topup:p250") == ("p250", None)
    assert parse_topup_payload("other") == (None, None)


def test_get_ai_billing_packs_returns_default_packs():
    packs = get_ai_billing_packs()

    assert len(packs) == 6
    assert packs[0] == {"id": "p50", "credits": 500, "xtr": 25}
    assert packs[-1] == {"id": "p2500", "credits": 25000, "xtr": 1250}


def test_build_insufficient_credits_message_mentions_group_balances():
    message = build_insufficient_credits_message(
        chat_type="group",
        user_balance=whole_credits_to_units(2),
        chat_balance=whole_credits_to_units(5),
    )
    assert "lo tuyo: 2.0" in message
    assert "lo del grupo: 5.0" in message


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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1_000,
                    "input_cached_tokens": 900,
                    "input_non_cached_tokens": 100,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 1_000
    assert breakdown["charged_credit_units"] == 2
    assert breakdown["charged_credits_display"] == "0.2"
    assert breakdown["model_breakdown"] == [
        {
            "model": "deepseek/deepseek-v4-flash",
            "usd_micros": 1_000,
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "vision",
                "model": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            {
                "kind": "vision",
                "model": "meta-llama/llama-4-scout",
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
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-scout",
        "whisper-large-v3",
    ]


def test_calculate_billing_for_segments_bumps_pricing_version_for_deepseek_search_billing():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 2_000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 1_500},
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 920
    assert breakdown["charged_credit_units"] == 2
    assert breakdown["charged_credits_display"] == "0.2"
    assert breakdown["model_breakdown"] == [
        {
            "model": "deepseek/deepseek-v4-flash",
            "usd_micros": 920,
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 10_000,
                    "output_tokens": 500,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 0
    assert breakdown["charged_credit_units"] == 0
    assert breakdown["charged_credits_display"] == "0.0"
    assert breakdown["model_breakdown"] == []
    assert breakdown["tool_breakdown"] == []
    assert breakdown["unsupported_notes"] == []


def test_calculate_billing_for_segments_bills_web_search_requests():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "metadata": {"web_search_requests": 2},
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 3_420
    assert breakdown["charged_credit_units"] == 7
    assert breakdown["tool_breakdown"] == [
        {"tool": "web_search", "count": 2, "usd_micros": 3_320}
    ]


def test_calculate_billing_for_segments_refunds_cache_only_usage_to_zero():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
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
    assert breakdown["charged_credits_display"] == "0.0"


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
                "model": "deepseek/deepseek-v4-flash",
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 3000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    assert billing.credits_db_service.charge_ai_credits.call_args.kwargs["amount"] == 1
    assert (
        billing.credits_db_service.charge_ai_credits.call_args.kwargs["event_type"]
        == "ai_settlement_charge"
    )
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 4000,
                    "output_tokens": 3000,
                },
            }
        ],
        reason="ok",
    )

    billing.credits_db_service.charge_ai_credits.assert_called_once()
    billing.credits_db_service.apply_ai_debt.assert_called_once()
    assert billing.credits_db_service.apply_ai_debt.call_args.kwargs["amount"] == 1
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
                "model": "deepseek/deepseek-v4-flash",
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        ],
        reason="ai_response_success",
    )

    billing.credits_db_service.refund_ai_charge.assert_called_once()
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 9
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    assert billing.credits_db_service.record_ai_settlement_result.call_count == 2


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
    assert billing.credits_db_service.record_ai_settlement_result.call_count == 2
    first_metadata = (
        billing.credits_db_service.record_ai_settlement_result.call_args_list[0].kwargs[
            "metadata"
        ]
    )
    second_metadata = (
        billing.credits_db_service.record_ai_settlement_result.call_args_list[1].kwargs[
            "metadata"
        ]
    )
    assert first_metadata["missing_usage_billing"] is True
    assert first_metadata["refunded_credit_units"] == 0
    assert second_metadata["billing_zero_usage_fallback"] is True
    assert second_metadata["refunded_credit_units"] == 0


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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
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
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 12
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
                "model": "deepseek/deepseek-v4-flash",
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
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs[
        "metadata"
    ]
    assert metadata["billing_zero_usage_fallback"] is True
    assert metadata["settled_credit_units"] == 30
    assert metadata["refunded_credit_units"] == 0


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
                "model": "deepseek/deepseek-v4-flash",
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
    assert billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"] == 29
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    billing.credits_db_service.record_ai_settlement_result.assert_called_once()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs[
        "metadata"
    ]
    assert metadata["settled_credit_units"] == 1
    assert metadata["refunded_credit_units"] == 29


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
                "model": "deepseek/deepseek-v4-flash",
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
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs[
        "metadata"
    ]
    assert metadata["billing_zero_usage_fallback"] is True
    assert metadata["settled_credit_units"] == 20
    assert metadata["refunded_credit_units"] == 0


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
    assert (
        billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"]
        == expected_refund
    )
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs[
        "metadata"
    ]
    assert metadata["settled_credit_units"] == breakdown["charged_credit_units"]
    assert metadata["refunded_credit_units"] == expected_refund


def test_settle_reserved_ai_credits_refunds_partial_chat_usage():
    billing = _build_billing_helper()
    reserved_credit_units = whole_credits_to_units(3)
    segments = [
        {
            "kind": "chat",
            "model": "deepseek/deepseek-v4-flash",
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
    assert (
        billing.credits_db_service.refund_ai_charge.call_args.kwargs["amount"]
        == expected_refund
    )
    billing.credits_db_service.charge_ai_credits.assert_not_called()
    metadata = billing.credits_db_service.record_ai_settlement_result.call_args.kwargs[
        "metadata"
    ]
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
            "reserved_credit_units": 20,
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
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.005,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 5_000
    assert breakdown["model_breakdown"][0]["usd_micros"] == 5_000


def test_calculate_billing_keeps_local_cost_when_higher_than_gateway():
    # Local pricing for 4000 input + 2000 output deepseek tokens:
    # (4000 * 400_000 + 2000 * 1_200_000) // 1_000_000 = 4_000 usd_micros
    # Gateway cost of $0.001 USD = 1_000 usd_micros -> local wins
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 4_000,
                    "completion_tokens": 2_000,
                    "cost": 0.001,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 4_000
    assert breakdown["model_breakdown"][0]["usd_micros"] == 4_000


def test_calculate_billing_without_gateway_cost_uses_local():
    breakdown = calculate_billing_for_segments(
        [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        ]
    )

    assert breakdown["raw_usd_micros"] == 100
    assert breakdown["model_breakdown"][0]["usd_micros"] == 130


def test_estimate_chat_reserve_credits_reasoning_adds_headroom():
    messages = [{"role": "user", "content": "hello"}]
    system_message = {"role": "system", "content": "sys"}

    without_reasoning = estimate_chat_reserve_credits(
        system_message=system_message,
        messages=messages,
        reasoning=False,
    )
    with_reasoning = estimate_chat_reserve_credits(
        system_message=system_message,
        messages=messages,
        reasoning=True,
    )

    assert with_reasoning > without_reasoning


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
