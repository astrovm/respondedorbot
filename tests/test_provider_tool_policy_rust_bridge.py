from __future__ import annotations

import logging

from api.providers import runtime as provider_runtime
from api.tools.runtime import ToolRuntime


class _FakeRustProviderToolPolicy:
    def __init__(
        self,
        result: tuple[str, str, str] | None,
        *,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def parse_pseudo_web_fetch(self, *arguments: object):
        self.calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust tool-policy failure")
        return self.result


def _runtime(*, registered: bool = True) -> provider_runtime.ProviderRuntime:
    runtime = object.__new__(provider_runtime.ProviderRuntime)
    registry = {"web_fetch": object()} if registered else {}
    runtime._tool_runtime = ToolRuntime(tool_registry=registry)
    return runtime


def test_rust_provider_tool_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderToolPolicy(
        ("rust_call_7", "web_fetch", "https://rust.example/result")
    )
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_tool_policy",
        lambda: rust,
    )

    actual = _runtime()._parse_pseudo_tool_call(
        "Python would reject this",
        6,
        [{"name": "calculate"}, {"name": "web_fetch"}],
    )

    assert actual is not None
    assert actual.id == "rust_call_7"
    assert actual.function.name == "web_fetch"
    assert actual.function.arguments == '{"url": "https://rust.example/result"}'
    assert rust.calls == [
        (
            "Python would reject this",
            6,
            ["calculate", "web_fetch"],
            True,
        )
    ]


def test_rust_provider_tool_policy_none_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderToolPolicy(None)
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_tool_policy",
        lambda: rust,
    )

    assert (
        _runtime()._parse_pseudo_tool_call(
            'web_fetch("https://python.example/would-match")',
            0,
            [{"name": "web_fetch"}],
        )
        is None
    )


def test_rust_provider_tool_policy_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderToolPolicy(None, fail=True)
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_tool_policy",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=provider_runtime.__name__):
        actual = _runtime()._parse_pseudo_tool_call(
            'web_fetch("https://python.example/fallback")',
            2,
            [{"name": "web_fetch"}],
        )

    assert actual is not None
    assert actual.id == "pseudo_call_3"
    assert actual.function.name == "web_fetch"
    assert actual.function.arguments == '{"url": "https://python.example/fallback"}'
    assert "using Python fallback" in caplog.text
