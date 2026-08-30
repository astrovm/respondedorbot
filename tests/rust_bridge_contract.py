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


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        raise SystemExit("usage: rust_bridge_contract.py PATH_TO_EXTENSION")
    bridge = _load_bridge(Path(arguments[1]).resolve())
    assert bridge.migration_protocol_version() == 1
    verify_credit_units(bridge)
    verify_command_parsing(bridge)
    verify_task_triggers(bridge)
    verify_price_queries(bridge)
    verify_market_context(bridge)
    verify_media_routing(bridge)
    verify_response_routing(bridge)
    verify_base_conversion(bridge)
    verify_random_selection(bridge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
