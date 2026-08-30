from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from api.tools import runtime


class _FakeRustToolExecutionPolicy:
    def __init__(self, action: str = "skip_unregistered", *, fail: bool = False) -> None:
        self.action = action
        self.fail = fail
        self.calls: list[tuple[bool, bool]] = []

    def tool_execution_action(self, has_function: bool, registered: bool) -> str:
        self.calls.append((has_function, registered))
        if self.fail:
            raise ValueError("synthetic Rust tool-execution failure")
        return self.action


def _tool_call(name: str = "synthetic_tool") -> SimpleNamespace:
    return SimpleNamespace(
        id="synthetic-call",
        function=SimpleNamespace(name=name, arguments='{"value":1}'),
    )


def test_rust_tool_execution_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustToolExecutionPolicy("skip_unregistered")
    monkeypatch.setattr(runtime, "_load_rust_tool_execution_policy", lambda: rust)
    executions: list[tuple[object, ...]] = []
    logs: list[str] = []
    tool_runtime = runtime.ToolRuntime(
        execute_tool_fn=lambda *arguments: executions.append(arguments),
        tool_registry={"synthetic_tool": object()},
        print_fn=logs.append,
    )

    messages = tool_runtime.apply_tool_calls(
        SimpleNamespace(content="assistant"),
        [_tool_call()],
        [],
        {},
    )

    assert executions == []
    assert messages == [
        {"role": "assistant", "content": "assistant", "tool_calls": []}
    ]
    assert rust.calls == [(True, True)]
    assert "not registered" in logs[0]


def test_missing_function_is_classified_without_registry_lookup(monkeypatch) -> None:
    rust = _FakeRustToolExecutionPolicy("skip_missing_function")
    monkeypatch.setattr(runtime, "_load_rust_tool_execution_policy", lambda: rust)
    tool_runtime = runtime.ToolRuntime(
        tool_registry={"synthetic_tool": object()},
        print_fn=lambda _message: None,
    )

    messages = tool_runtime.apply_tool_calls(
        SimpleNamespace(content=""),
        [SimpleNamespace(id="missing", function=None)],
        [],
        {},
    )

    assert messages == [{"role": "assistant", "tool_calls": []}]
    assert rust.calls == [(False, False)]


def test_rust_tool_execution_failure_preserves_python_dispatch(monkeypatch, caplog) -> None:
    rust = _FakeRustToolExecutionPolicy(fail=True)
    monkeypatch.setattr(runtime, "_load_rust_tool_execution_policy", lambda: rust)
    result = SimpleNamespace(output="executed", metadata={})
    executions: list[tuple[object, ...]] = []

    def execute(*arguments):
        executions.append(arguments)
        return result

    tool_runtime = runtime.ToolRuntime(
        execute_tool_fn=execute,
        tool_registry={"synthetic_tool": object()},
        print_fn=lambda _message: None,
    )

    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        messages = tool_runtime.apply_tool_calls(
            SimpleNamespace(content="assistant"),
            [_tool_call()],
            [],
            {"chat_id": "synthetic-chat"},
        )

    assert len(executions) == 1
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "synthetic-call",
        "content": "executed",
    }
    assert "using Python fallback" in caplog.text


def test_invalid_rust_tool_execution_action_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustToolExecutionPolicy("invalid")
    monkeypatch.setattr(runtime, "_load_rust_tool_execution_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=runtime.__name__):
        actual = runtime._tool_execution_action(True, False)

    assert actual == "skip_unregistered"
    assert "using Python fallback" in caplog.text


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_load_rust_tool_execution_policy", lambda: None)
    path = Path(__file__).parents[1] / "contracts" / "tool_execution_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        actual = runtime._tool_execution_action(
            case["has_function"],
            case["registered"],
        )
        assert actual == case["expected"], case["name"]
