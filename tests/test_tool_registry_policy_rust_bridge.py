from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from api.tools import registry


class _FakeRustToolRegistryPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object) -> object:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust tool-registry failure")
        return value

    def tool_parse_arguments(self, *arguments: object) -> str | None:
        value = self._result("arguments", '{"rust":true}', *arguments)
        return None if value is None else str(value)

    def tool_select_available(self, *arguments: object) -> list[int]:
        value = self._result("availability", [1], *arguments)
        assert isinstance(value, list)
        return value


def _schema(name: str) -> registry.ToolSchema:
    return registry.ToolSchema(name=name, description=name, parameters={})


def test_rust_tool_registry_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustToolRegistryPolicy()
    cache = [
        (_schema("python-first"), {"name": "python-first"}),
        (_schema("rust-selected"), {"name": "rust-selected"}),
    ]
    monkeypatch.setattr(registry, "_load_rust_tool_registry_policy", lambda: rust)
    monkeypatch.setattr(registry, "_get_schema_cache", lambda: cache)

    assert registry.parse_tool_call_arguments('{"python":true}') == {"rust": True}
    assert registry.get_all_tool_schemas() == [{"name": "rust-selected"}]
    assert rust.calls[0] == ("arguments", '{"python":true}')
    assert rust.calls[1][0] == "availability"
    assert json.loads(str(rust.calls[1][1])) == [
        {
            "environment_requirements_met": True,
            "context_requirements_met": True,
            "task_allowed": True,
        },
        {
            "environment_requirements_met": True,
            "context_requirements_met": True,
            "task_allowed": True,
        },
    ]
    assert rust.calls[1][2:] == (False, False)


def test_dict_arguments_preserve_identity_without_crossing_bridge(monkeypatch) -> None:
    rust = _FakeRustToolRegistryPolicy()
    monkeypatch.setattr(registry, "_load_rust_tool_registry_policy", lambda: rust)
    arguments: dict[str, Any] = {"value": 1}

    assert registry.parse_tool_call_arguments(arguments) is arguments
    assert rust.calls == []


def test_rust_tool_registry_failure_preserves_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustToolRegistryPolicy(fail=True)
    cache = [(_schema("available"), {"name": "available"})]
    monkeypatch.setattr(registry, "_load_rust_tool_registry_policy", lambda: rust)
    monkeypatch.setattr(registry, "_get_schema_cache", lambda: cache)

    with caplog.at_level(logging.ERROR, logger=registry.__name__):
        assert registry.parse_tool_call_arguments('{"python":true}') == {"python": True}
        assert registry.get_all_tool_schemas() == [{"name": "available"}]

    assert [call[0] for call in rust.calls] == ["arguments", "availability"]
    assert caplog.text.count("using Python fallback") == 2


def test_invalid_rust_schema_index_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustToolRegistryPolicy()
    rust.tool_select_available = lambda *_arguments: [99]  # type: ignore[method-assign]
    cache = [(_schema("available"), {"name": "available"})]
    monkeypatch.setattr(registry, "_load_rust_tool_registry_policy", lambda: rust)
    monkeypatch.setattr(registry, "_get_schema_cache", lambda: cache)

    with caplog.at_level(logging.ERROR, logger=registry.__name__):
        assert registry.get_all_tool_schemas() == [{"name": "available"}]

    assert "using Python fallback" in caplog.text


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_load_rust_tool_registry_policy", lambda: None)
    monkeypatch.delenv("MISSING", raising=False)
    path = Path(__file__).parents[1] / "contracts" / "tool_registry_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["argument_cases"]:
        actual = registry.parse_tool_call_arguments(case["raw"])
        if "expected_non_finite_key" in case:
            assert math.isnan(actual[case["expected_non_finite_key"]]), case["name"]
        else:
            assert actual == case["expected"], case["name"]

    for case in contract["availability_cases"]:
        tools = case["tools"]
        schemas = [
            registry.ToolSchema(
                name=f"tool-{index}",
                description="synthetic",
                parameters={},
                requires_env=[] if facts["environment_requirements_met"] else ["MISSING"],
                requires_context=(
                    [] if facts["context_requirements_met"] else ["missing_context"]
                ),
                task_allowed=facts["task_allowed"],
            )
            for index, facts in enumerate(tools)
        ]
        context = {} if case["context_provided"] else None
        if context is None and not case["task_mode"]:
            actual = list(range(len(schemas)))
        else:
            actual = [
                index
                for index, schema in enumerate(schemas)
                if registry._tool_is_available(schema, context)
                and (not case["task_mode"] or schema.task_allowed)
            ]
        assert actual == case["expected"], case["name"]
