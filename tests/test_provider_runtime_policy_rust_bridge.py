from __future__ import annotations

import logging
from types import SimpleNamespace

from api.providers import runtime as provider_runtime


class _FakeRustProviderRuntimePolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object):
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust runtime policy failure")
        return value

    def provider_exception_is_retryable(self, *facts: object) -> bool:
        return bool(self._result("exception", True, *facts))

    def provider_usage_has_billable_activity(self, usage_json: str) -> bool:
        return bool(self._result("usage", False, usage_json))

    def provider_finish_response_is_retryable(self, *facts: object) -> bool:
        return bool(self._result("finish", True, *facts))

    def provider_retry_wait_seconds(self, attempt: int) -> int:
        return int(self._result("wait", 91, attempt))


def _runtime_with_usage(usage: dict[str, object]) -> provider_runtime.ProviderRuntime:
    runtime = object.__new__(provider_runtime.ProviderRuntime)
    runtime._deps = SimpleNamespace(extract_usage_map=lambda _response: usage)
    return runtime


def test_rust_provider_runtime_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderRuntimePolicy()
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_runtime_policy",
        lambda: rust,
    )
    runtime = _runtime_with_usage({"prompt_tokens": 10})
    response = SimpleNamespace()
    choice = SimpleNamespace(
        message=SimpleNamespace(content="already complete", tool_calls=[]),
        error=None,
    )

    assert provider_runtime._is_retryable_provider_error(Exception("permanent")) is True
    assert provider_runtime._retry_wait_seconds(0) == 91
    assert runtime._response_has_usage(response) is False
    assert runtime._is_retryable_finish_response(response, choice, None) is True
    assert [call[0] for call in rust.calls] == ["exception", "wait", "usage", "finish"]


def test_rust_provider_runtime_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderRuntimePolicy(fail=True)
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_runtime_policy",
        lambda: rust,
    )
    runtime = _runtime_with_usage({"prompt_tokens": 10})
    response = SimpleNamespace()
    choice = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]),
        error=None,
    )

    with caplog.at_level(logging.ERROR, logger=provider_runtime.__name__):
        assert provider_runtime._is_retryable_provider_error(Exception("permanent")) is False
        assert provider_runtime._retry_wait_seconds(0) == 1
        assert runtime._response_has_usage(response) is True
        assert runtime._is_retryable_finish_response(response, choice, None) is False

    assert "using Python fallback" in caplog.text
