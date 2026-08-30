"""Verify the compiled Rust extension against language-neutral contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_bridge(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("respondedorbot_rs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rust bridge from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "contracts" / "credit_units.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("credit-unit contract must be a JSON object")
    return loaded


def verify_credit_units(bridge: ModuleType) -> None:
    contract = _load_contract()
    for case in contract["parse"]:
        assert bridge.parse_credit_units(case["input"]) == case["expected"]
    for case in contract["format"]:
        assert bridge.format_credit_units(case["units"]) == case["expected"]
    for case in contract["rescale"]:
        source_scale = case["source_scale"] or 10
        if case["error"] is None:
            assert bridge.rescale_credit_units(case["units"], source_scale) == case["expected"]
        else:
            try:
                bridge.rescale_credit_units(case["units"], source_scale)
            except ValueError as error:
                assert str(error) == case["error"]
            else:
                raise AssertionError(f"expected rescale failure for {case}")
    for case in contract["whole"]:
        assert bridge.whole_credits_to_units(case["credits"]) == case["expected"]


def verify_command_parsing(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "command_parsing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        assert bridge.parse_command(case["input"], case["bot_name"]) == (
            case["command"],
            case["message_text"],
        )


def verify_command_normalization(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "command_normalization.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["normalization"]:
        assert bridge.normalize_command_text(case["input"]) == case["expected"], case["name"]


def verify_task_triggers(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "task_triggers.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.parse_task_trigger(
                json.dumps(case["input"], separators=(",", ":")),
            )
        )
        assert actual == case["expected"], case["name"]


def verify_price_queries(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "price_queries.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    valid = json.dumps(contract["valid_timeframes"], separators=(",", ":"))
    for case in contract["cases"]:
        actual = json.loads(bridge.parse_price_query(case["input"], valid))
        assert actual == case["expected"], case["name"]


def verify_market_context(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "market_context.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.format_market_info(
            json.dumps(case["input"], separators=(",", ":")),
        )
        assert actual == case["expected"], case["name"]


def verify_market_models(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "market_models.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.evaluate_market_model(
                case["model"], case["elapsed_days"], case["market_price"]
            )
        )
        assert actual == case["expected"], case["name"]


def verify_satoshi(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "satoshi.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.format_satoshi_quote(case["price_usd"], case["price_ars"])
        assert actual == case["expected"], case["name"]
    for case in contract["errors"]:
        try:
            bridge.format_satoshi_quote(case["price_usd"], case["price_ars"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected satoshi failure for {case['name']}")


def verify_devo(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "devo.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["parse"]:
        actual = bridge.parse_devo_input(case["input"])
        assert actual == (case["expected_kind"], case["fee"], case["purchase"]), case
    for case in contract["calculate"]:
        actual = json.loads(bridge.calculate_devo(*case["input"]))
        assert actual == case["expected"], case["name"]


def verify_rulo(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "rulo.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.evaluate_rulo(json.dumps(case["input"], separators=(",", ":")))
        )
        assert actual == case["expected"], case["name"]


def verify_weather_selection(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "weather_selection.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["locations"]:
        actual = bridge.select_weather_location(case["qualifiers"], case["candidates"])
        assert actual == case["expected"], case["name"]
    for case in contract["hours"]:
        actual = bridge.select_weather_hour(
            case["forecast"], case["provider"], case["local"]
        )
        assert actual == case["expected"], case["name"]


def verify_polymarket_ranking(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "polymarket_ranking.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.rank_polymarket_outcomes(
                json.dumps(case["input"], separators=(",", ":")),
                case["limit"],
            )
        )
        assert actual == case["expected"], case["name"]


def verify_hacker_news(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "hacker_news.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["normalize"]:
        actual = json.loads(
            bridge.normalize_hacker_news_item(
                case["title"], case["url"], case["description"]
            )
        )
        assert actual == case["expected"], case["name"]
    for case in contract["format"]:
        actual = bridge.format_hacker_news_items(
            json.dumps(case["items"], separators=(",", ":")),
            case["include_discussion"],
            case["no_data"],
            case["comments_label"],
        )
        assert actual == case["expected"], case["name"]


def verify_config_callbacks(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "config_callbacks.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.evaluate_config_callback(
                json.dumps(case["input"], separators=(",", ":"))
            )
        )
        assert actual == case["expected"], case["name"]


def verify_link_parsing(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "link_parsing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["utf16"]:
        actual = bridge.slice_telegram_utf16(
            case["text"], case["offset"], case["length"]
        )
        assert actual == case["expected"], case["name"]
    for case in contract["trim"]:
        assert bridge.trim_detected_url(case["input"]) == case["expected"], case["name"]
    for case in contract["unique"]:
        actual = bridge.select_unique_urls(case["candidates"], case["max_links"])
        assert actual == case["expected"], case["name"]


def verify_admin_reports(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "admin_reports.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["limits"]:
        assert bridge.parse_creditlog_limit(case["input"]) == case["expected"], case["name"]
    for case in contract["truncate"]:
        actual = bridge.truncate_admin_report(
            case["text"], case["max_length"], case["label"]
        )
        assert actual == case["expected"], case["name"]


def verify_cache_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "cache_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["decisions"]:
        actual = bridge.evaluate_cache_policy(
            case["timestamp"], case["now"], case["ttl"], case["stale_grace"]
        )
        assert actual == case["expected"], case["name"]
    for case in contract["keys"]:
        if case["kind"] == "request":
            actual = bridge.request_cache_key(case["request_hash"])
        else:
            actual = bridge.request_cache_history_key(
                case["hour_key"], case["request_hash"]
            )
        assert actual == case["expected"], case["name"]
    for case in contract["ttls"]:
        if case["kind"] == "request":
            actual = bridge.request_cache_ttl(case["ttl"])
        else:
            actual = bridge.last_success_ttl(case["ttl"], case["stale_grace"])
        assert actual == case["expected"], case["name"]


def verify_message_state(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "message_state.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["writes"]:
        request = case["input"]
        actual = json.loads(
            bridge.prepare_message_write(
                request["chat_id"],
                request["message_id"],
                request["text"],
                request["timestamp"],
                request["role"],
                request["user_id"],
                request["username"],
                request["reply_to_message_id"],
                request["mentions_bot"],
            )
        )
        actual["history_entry"] = json.loads(actual["history_entry"])
        if "expected" in case:
            assert actual == case["expected"], case["name"]
        else:
            assert actual["role"] == case["expected_role"], case["name"]
    for case in contract["search_escape"]:
        assert bridge.escape_message_search_text(case["input"]) == case["expected"], case[
            "name"
        ]
    for case in contract["tag_escape"]:
        assert bridge.escape_message_search_tag(case["input"]) == case["expected"], case[
            "name"
        ]
    ranking = contract["ranking"]
    actual_ranking = json.loads(
        bridge.rank_message_search_results(
            json.dumps(ranking["candidates"], separators=(",", ":")),
            ranking["search_text"],
            ranking["reply_to_message_id"],
            ranking["excluded_message_ids"],
            ranking["limit"],
        )
    )
    assert actual_ranking == ranking["expected"]
    for case in contract["auxiliary_keys"]:
        actual = bridge.message_state_key(
            case["kind"], case["chat_id"], case["message_id"]
        )
        assert actual == case["expected"], case["name"]
    member = contract["member"]
    actual_member = json.loads(
        bridge.prepare_chat_member(
            member["first_name"], member["username"], member["last_seen"]
        )
    )
    assert actual_member == member["expected"]


def verify_compaction_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "compaction_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["due"]:
        actual = bridge.compaction_job_is_due(case["next_attempt_at"], case["now"])
        assert actual is case["expected"], case["name"]
    for case in contract["dispositions"]:
        request = case["input"]
        actual = bridge.evaluate_compaction_policy(
            request["current_summary"],
            request["current_marker"],
            request["prior_summary"],
            request["expected_marker"],
            request["result_summary"],
            request["target_marker"],
        )
        assert actual == case["expected"], case["name"]
    for case in contract["retries"]:
        actual = json.loads(
            bridge.compaction_retry_transition(
                case["attempts"],
                case["now"],
                case["has_billing_segment"],
            )
        )
        assert actual == case["expected"], case["name"]


def verify_compaction_jobs(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "compaction_jobs.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    legacy = contract["legacy"]
    normalized = json.loads(
        bridge.normalize_compaction_job(json.dumps(legacy, separators=(",", ":")))
    )
    for key, value in legacy.items():
        assert normalized[key] == value, key
    for key, value in contract["expected_defaults"].items():
        assert normalized[key] == value, key
    future = {**legacy, "schema_version": contract["unsupported_version"]}
    try:
        bridge.normalize_compaction_job(json.dumps(future, separators=(",", ":")))
    except ValueError:
        pass
    else:
        raise AssertionError("future compaction-job version must be rejected")


def verify_compaction_queue(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "compaction_queue.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    queue = bridge.RedisCompactionQueue("localhost", 6379, None)
    for case in contract["lock_ttl_validation"]:
        try:
            queue.acquire_lock("synthetic-chat", "synthetic-token", case["ttl"])
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_message_aux_io(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "message_aux_io.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    state = bridge.RedisMessageState("localhost", 6379, None)
    for case in contract["ttl_validation"]:
        try:
            state.set_value("synthetic-key", "synthetic-value", case["ttl"])
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_message_history_io(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "message_history_io.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    state = bridge.RedisMessageState("localhost", 6379, None)
    for case in contract["ttl_validation"]:
        try:
            state.save_message(
                "synthetic-chat",
                "1",
                "synthetic text",
                1,
                None,
                "7",
                "synthetic-user",
                None,
                False,
                case["ttl"],
                400,
            )
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_media_cache(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "media_cache.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["keys"]:
        actual = bridge.redis_media_cache_key(case["prefix"], case["file_id"])
        assert actual == case["expected"], case["name"]


def verify_chat_admin_cache(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "chat_admin_cache.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["keys"]:
        actual = bridge.redis_chat_admin_key(case["chat_id"], case["user_id"])
        assert actual == case["expected"], case["name"]


def verify_request_cache_io(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "request_cache_io.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    cache = bridge.RedisJsonCache("invalid.invalid", 1, None)
    for case in contract["ttl_validation"]:
        try:
            cache.set("synthetic", "value", ex=case["ttl"])
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_stale_cache_io(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "stale_cache_io.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    cache = bridge.RedisJsonCache("invalid.invalid", 1, None)
    for case in contract["lock_ttl_validation"]:
        try:
            cache.set("synthetic:lock", "1", nx=True, ex=case["ttl"])
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_redis_maintenance(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "redis_maintenance.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["ttl_targets"]["giphy_pool_stale:*"] == 7 * 24 * 60 * 60
    assert callable(bridge.run_redis_maintenance)


def verify_task_store_io(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "task_store_io.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    store = bridge.RedisTaskStore("invalid.invalid", 1, None)
    for case in contract["ttl_validation"]:
        try:
            store.setex(contract["keys"]["payload"], case["ttl"], "{}")
        except ValueError:
            continue
        raise AssertionError(f"{case['name']} must be rejected")


def verify_billing_reads(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_reads.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for scope in contract["invalid_scopes"]:
        for function_name in (
            "billing_read_balance",
            "billing_get_or_create_balance",
        ):
            function = getattr(bridge, function_name)
            try:
                function(
                    "postgresql://invalid.invalid/db?sslmode=require",
                    scope,
                    1,
                )
            except ValueError:
                continue
            raise AssertionError(
                f"invalid billing scope must be rejected by {function_name}: {scope}"
            )


def verify_billing_schema(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_schema.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["current_credit_scale"] != 100:
        raise AssertionError("billing schema credit scale changed")
    if contract["advisory_lock_keys"] != [48610002, 48610003, 48610004, 48610005]:
        raise AssertionError("billing schema advisory locks changed")
    if not callable(bridge.billing_ensure_schema):
        raise AssertionError("Rust billing schema bridge is unavailable")


def verify_billing_onboarding(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_onboarding.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["idempotency_key"] != "user_id":
        raise AssertionError("onboarding idempotency must remain user-scoped")
    if not callable(bridge.billing_grant_onboarding):
        raise AssertionError("Rust onboarding bridge is unavailable")


def verify_billing_star_payments(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_star_payments.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["idempotency_key"] != "telegram_payment_charge_id":
        raise AssertionError("Stars payment idempotency must remain charge-scoped")
    if not callable(bridge.billing_record_star_payment):
        raise AssertionError("Rust Stars payment bridge is unavailable")


def verify_billing_manual_credits(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_manual_credits.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["transfer"]["lock_order"] != ["user", "chat"]:
        raise AssertionError("manual credit transfer lock order changed")
    for function_name in (
        "billing_mint_user_credits",
        "billing_transfer_user_to_chat",
    ):
        if not callable(getattr(bridge, function_name)):
            raise AssertionError(f"Rust manual-credit bridge unavailable: {function_name}")


def verify_billing_chat_ai_credits(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_chat_ai_credits.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["account_scope"] != "chat":
        raise AssertionError("chat AI billing account scope changed")
    for function_name in (
        "billing_charge_chat_ai_credits",
        "billing_refund_chat_ai_credits",
        "billing_apply_chat_ai_debt",
    ):
        if not callable(getattr(bridge, function_name)):
            raise AssertionError(f"Rust chat AI bridge unavailable: {function_name}")


def verify_billing_ai_debt(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_ai_debt.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["lock_order"] != ["user", "chat"]:
        raise AssertionError("AI debt lock order changed")
    if not callable(bridge.billing_apply_ai_debt):
        raise AssertionError("Rust AI debt bridge is unavailable")


def verify_billing_ai_refunds(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_ai_refunds.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["lock_order"] != ["user", "chat"]:
        raise AssertionError("AI refund lock order changed")
    if not callable(bridge.billing_refund_ai_charge):
        raise AssertionError("Rust AI refund bridge is unavailable")


def verify_billing_ai_charges(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_ai_charges.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["payer_order"] != ["user", "chat"]:
        raise AssertionError("AI charge payer order changed")
    if not callable(bridge.billing_charge_ai_credits):
        raise AssertionError("Rust AI charge bridge is unavailable")


def verify_billing_provider_usage(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_provider_usage.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["idempotency_key"] != ["operation_id", "segment_id"]:
        raise AssertionError("provider usage idempotency changed")
    for function_name in (
        "billing_record_ai_provider_usage",
        "billing_list_ai_provider_segments",
        "billing_update_ai_provider_usage",
    ):
        if not callable(getattr(bridge, function_name)):
            raise AssertionError(f"Rust provider-usage bridge unavailable: {function_name}")


def verify_billing_ai_settlements(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_ai_settlements.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["lock_order"] != ["user", "chat"]:
        raise AssertionError("AI settlement lock order changed")
    if contract["idempotency_key"] != ["user_id", "operation_id"]:
        raise AssertionError("AI settlement idempotency changed")
    if not callable(bridge.billing_settle_ai_operation_once):
        raise AssertionError("Rust AI settlement bridge is unavailable")


def verify_billing_legacy_settlements(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_legacy_settlements.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["lock_order"] != ["user", "chat"]:
        raise AssertionError("legacy AI settlement lock order changed")
    if contract["idempotency_scope"] != "global":
        raise AssertionError("legacy AI settlement idempotency scope changed")
    if not callable(bridge.billing_settle_legacy_ai_reservation_once):
        raise AssertionError("Rust legacy AI settlement bridge is unavailable")


def verify_billing_audit_writes(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_audit_writes.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["idempotency_key"] != ["user_id", "metadata.settlement_id"]:
        raise AssertionError("AI settlement audit idempotency changed")
    if contract["amount"] != 0:
        raise AssertionError("AI settlement audit amount changed")
    if not callable(bridge.billing_record_ai_settlement_result):
        raise AssertionError("Rust AI settlement audit bridge is unavailable")


def verify_billing_audit_reads(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_audit_reads.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["ordering"] != ["created_at descending", "id descending"]:
        raise AssertionError("AI settlement audit ordering changed")
    if contract["limit"]["maximum"] != 50:
        raise AssertionError("AI settlement audit limit changed")
    if not callable(bridge.billing_list_recent_ai_settlement_results):
        raise AssertionError("Rust AI settlement audit read bridge is unavailable")


def verify_billing_reconciliation_reads(bridge: ModuleType) -> None:
    path = (
        Path(__file__).parents[1]
        / "contracts"
        / "billing_reconciliation_reads.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["grouping_key"] != "operation_id":
        raise AssertionError("AI reconciliation grouping changed")
    if contract["provider_segment_order"] != "ledger id ascending":
        raise AssertionError("AI reconciliation provider ordering changed")
    if not callable(bridge.billing_list_unsettled_ai_operations):
        raise AssertionError("Rust AI reconciliation read bridge is unavailable")


def verify_billing_maintenance(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_maintenance.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["retention_days_minimum"] != 1:
        raise AssertionError("AI ledger minimum retention changed")
    if "memory_compaction_settlement" not in contract["event_types"]:
        raise AssertionError("AI ledger retention event allowlist changed")
    if not callable(bridge.billing_purge_expired_ai_ledger_events):
        raise AssertionError("Rust AI ledger maintenance bridge is unavailable")


def verify_billing_charge_history(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "billing_charge_history.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["directions"] != ["older", "newer"]:
        raise AssertionError("AI charge-history directions changed")
    if contract["limit"] != {"minimum": 1, "maximum": 20, "default": 10}:
        raise AssertionError("AI charge-history pagination limits changed")
    if not callable(bridge.billing_list_user_ai_charge_rows):
        raise AssertionError("Rust AI charge-history bridge is unavailable")


def verify_media_routing(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "media_routing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.should_auto_process_media(
            case["chat_type"],
            case["known_command"],
            case["message_text"],
            case["bot_username"],
            case["reply_username"],
        )
        assert actual is case["expected"], case["name"]


def verify_response_routing(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "response_routing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.evaluate_response_routing(
            json.dumps(case["input"], separators=(",", ":")),
        )
        assert actual == case["expected"], case["name"]


def verify_base_conversion(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "base_conversion.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(bridge.convert_base(case["input"]))
        assert actual == case["expected"], case["name"]


def verify_random_selection(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "random_selection.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(bridge.parse_random_selection(case["input"]))
        assert actual == case["expected"], case["name"]


def verify_random_reply(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "random_reply.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.evaluate_random_reply(case["response"], case["suffix"])
        assert list(actual) == case["expected"], case


def verify_billing_contracts(bridge: ModuleType) -> None:
    verify_billing_schema(bridge)
    verify_billing_reads(bridge)
    verify_billing_onboarding(bridge)
    verify_billing_star_payments(bridge)
    verify_billing_manual_credits(bridge)
    verify_billing_chat_ai_credits(bridge)
    verify_billing_ai_debt(bridge)
    verify_billing_ai_refunds(bridge)
    verify_billing_ai_charges(bridge)
    verify_billing_provider_usage(bridge)
    verify_billing_ai_settlements(bridge)
    verify_billing_legacy_settlements(bridge)
    verify_billing_audit_writes(bridge)
    verify_billing_audit_reads(bridge)
    verify_billing_reconciliation_reads(bridge)
    verify_billing_maintenance(bridge)
    verify_billing_charge_history(bridge)


def verify_ai_usage_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "ai_usage_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["identity_cases"]:
        canonical = json.dumps(
            case["segment"],
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        assert bridge.provider_segment_id(canonical) == case["expected"], case["name"]
    for case in contract["reconciliation_cases"]:
        canonical = json.dumps(
            case["segment"],
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        actual = bridge.provider_usage_needs_reconciliation(canonical)
        assert actual is case["expected"], case["name"]


def verify_provider_error_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "provider_error_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.classify_provider_error(
            case["status_code"],
            case["status"],
            case["code"],
            case["message"],
        )
        assert list(actual) == case["expected"], case["name"]


def verify_provider_retry_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "provider_retry_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["parse_cases"]:
        actual = bridge.parse_provider_retry_window(
            case["value"],
            case["now_unix_seconds"],
        )
        assert actual == case["expected"], case["name"]
    for case in contract["selection_cases"]:
        actual = bridge.select_provider_backoff_seconds(
            *case["headers"],
            case["fallback_seconds"],
            case["now_unix_seconds"],
        )
        assert actual == case["expected"], case["name"]


def verify_provider_runtime_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "provider_runtime_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["exception_cases"]:
        assert bridge.provider_exception_is_retryable(*case["facts"]) is case["expected"], case["name"]
    for case in contract["usage_cases"]:
        actual = bridge.provider_usage_has_billable_activity(
            json.dumps(case["usage"], ensure_ascii=False)
        )
        assert actual is case["expected"], case["name"]
    for case in contract["finish_cases"]:
        assert bridge.provider_finish_response_is_retryable(*case["facts"]) is case["expected"], case["name"]
    for case in contract["delay_cases"]:
        assert bridge.provider_retry_wait_seconds(case["attempt"]) == case["expected"], case


def verify_provider_tool_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "provider_tool_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = bridge.parse_pseudo_web_fetch(
            case["text"],
            case["round_index"],
            case["advertised_tool_names"],
            case["web_fetch_registered"],
        )
        assert (list(actual) if actual is not None else None) == case["expected"], case["name"]


def verify_provider_web_search_policy(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "provider_web_search_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["limit_cases"]:
        value_json = json.dumps(case["value"]) if "value" in case else None
        assert bridge.provider_web_search_max_uses(value_json) == case["expected"], case["name"]
    for case in contract["round_cases"]:
        value_json = json.dumps(case["server_value"]) if "server_value" in case else None
        actual = bridge.provider_web_search_round_metrics(
            value_json,
            case["tool_names"],
            case["annotation_types"],
        )
        assert list(actual) == case["expected"], case["name"]
    for case in contract["budget_cases"]:
        actual = bridge.provider_web_search_remaining_budget(
            case["remaining"],
            case["request_count"],
        )
        assert actual == case["expected"], case
    for case in contract["source_cases"]:
        actual = bridge.provider_web_search_source_urls(
            json.dumps(case["messages"], ensure_ascii=False)
        )
        assert actual == case["expected"], case["name"]
    for case in contract["outcome_cases"]:
        actual = bridge.provider_web_search_outcome_is_grounded(
            case["source_count"],
            case["citation_count"],
            case["text"],
        )
        assert actual is case["expected"], case


def verify_ai_reserve_estimates(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "ai_reserve_estimates.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["text_cases"]:
        assert bridge.ai_estimate_text_tokens(case["input"]) == case["expected"], case["name"]
    for case in contract["nested_cases"]:
        value_json = json.dumps(case["input"], ensure_ascii=False)
        assert bridge.ai_estimate_nested_tokens(value_json) == case["expected"], case["name"]
    for case in contract["message_cases"]:
        messages_json = json.dumps(case["input"], ensure_ascii=False)
        assert bridge.ai_estimate_message_tokens(messages_json) == case["expected"], case["name"]
    for case in contract["chat_cases"]:
        system_message_json = (
            json.dumps(case["system_message"], ensure_ascii=False)
            if case["system_message"] is not None
            else None
        )
        actual = bridge.ai_estimate_chat_reserve_credit_units(
            system_message_json,
            json.dumps(case["messages"], ensure_ascii=False),
            case["max_output_tokens"],
            case["extra_input_tokens"],
            case["model"],
        )
        assert actual == case["expected"], case["name"]
    for case in contract["vision_cases"]:
        actual = bridge.ai_estimate_vision_reserve_credit_units(
            case["prompt_text"],
            case["image_byte_length"],
            case["extra_input_tokens"],
            case["max_output_tokens"],
            case["model"],
        )
        assert actual == case["expected"], case["name"]
    for case in contract["transcription_cases"]:
        actual = bridge.ai_estimate_transcription_reserve_credit_units(
            case["audio_seconds"]
        )
        assert actual == case["expected"], case["name"]
    for case in contract["credit_cases"]:
        actual = bridge.ai_credit_units_from_usd_micros(case["usd_micros"])
        assert actual == case["expected"], case["name"]
    for case in contract["limit_cases"]:
        actual = bridge.ai_chat_output_token_limit(case["model"])
        assert actual == case["expected"], case["name"]
    assert (
        bridge.ai_estimate_firecrawl_reserve_credit_units()
        == contract["firecrawl_expected"]
    )


def verify_ai_pricing(bridge: ModuleType) -> None:
    path = Path(__file__).parents[1] / "contracts" / "ai_pricing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    for case in contract["cases"]:
        actual = json.loads(
            bridge.ai_calculate_billing_for_segments(
                json.dumps(case["segments"], ensure_ascii=False)
            )
        )
        expected = case["expected"]
        for key in (
            "raw_usd_micros",
            "raw_usd_micros_exact",
            "charged_credit_units",
            "pricing_complete",
        ):
            assert actual[key] == expected[key], (case["name"], key)
        if "pricing_basis" in expected:
            assert actual["segment_breakdown"][0]["pricing_basis"] == expected["pricing_basis"], case["name"]
        if "model_usd_micros" in expected:
            assert actual["model_breakdown"][0]["usd_micros"] == expected["model_usd_micros"], case["name"]
        if "tool_usd_micros" in expected:
            assert actual["tool_breakdown"][0]["usd_micros"] == expected["tool_usd_micros"], case["name"]


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        raise SystemExit("usage: rust_bridge_contract.py PATH_TO_EXTENSION")
    bridge = _load_bridge(Path(arguments[1]).resolve())
    assert bridge.migration_protocol_version() == 1
    verify_credit_units(bridge)
    verify_command_parsing(bridge)
    verify_command_normalization(bridge)
    verify_task_triggers(bridge)
    verify_price_queries(bridge)
    verify_market_context(bridge)
    verify_market_models(bridge)
    verify_satoshi(bridge)
    verify_devo(bridge)
    verify_rulo(bridge)
    verify_weather_selection(bridge)
    verify_polymarket_ranking(bridge)
    verify_hacker_news(bridge)
    verify_config_callbacks(bridge)
    verify_link_parsing(bridge)
    verify_admin_reports(bridge)
    verify_cache_policy(bridge)
    verify_message_state(bridge)
    verify_compaction_policy(bridge)
    verify_compaction_jobs(bridge)
    verify_compaction_queue(bridge)
    verify_message_aux_io(bridge)
    verify_message_history_io(bridge)
    verify_media_cache(bridge)
    verify_chat_admin_cache(bridge)
    verify_request_cache_io(bridge)
    verify_stale_cache_io(bridge)
    verify_redis_maintenance(bridge)
    verify_task_store_io(bridge)
    verify_billing_contracts(bridge)
    verify_ai_usage_policy(bridge)
    verify_provider_error_policy(bridge)
    verify_provider_retry_policy(bridge)
    verify_provider_runtime_policy(bridge)
    verify_provider_tool_policy(bridge)
    verify_provider_web_search_policy(bridge)
    verify_ai_reserve_estimates(bridge)
    verify_ai_pricing(bridge)
    verify_media_routing(bridge)
    verify_response_routing(bridge)
    verify_base_conversion(bridge)
    verify_random_selection(bridge)
    verify_random_reply(bridge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
