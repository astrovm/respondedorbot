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


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        raise SystemExit("usage: rust_bridge_contract.py PATH_TO_EXTENSION")
    bridge = _load_bridge(Path(arguments[1]).resolve())
    assert bridge.migration_protocol_version() == 1
    verify_credit_units(bridge)
    verify_command_parsing(bridge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
