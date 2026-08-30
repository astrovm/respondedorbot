from __future__ import annotations

import logging
from types import SimpleNamespace

from api.providers import errors


class _FakeRustProviderRetryPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.parse_calls: list[tuple[str | None, float]] = []
        self.selection_calls: list[tuple[object, ...]] = []

    def parse_provider_retry_window(
        self,
        value: str | None,
        now_unix_seconds: float,
    ) -> int | None:
        self.parse_calls.append((value, now_unix_seconds))
        if self.fail:
            raise ValueError("synthetic Rust retry parse failure")
        return 91

    def select_provider_backoff_seconds(
        self,
        retry_after: str | None,
        reset_requests: str | None,
        reset_tokens: str | None,
        reset: str | None,
        fallback_seconds: int | None,
        now_unix_seconds: float,
    ) -> int | None:
        self.selection_calls.append(
            (
                retry_after,
                reset_requests,
                reset_tokens,
                reset,
                fallback_seconds,
                now_unix_seconds,
            )
        )
        if self.fail:
            raise ValueError("synthetic Rust retry selection failure")
        return 92


def test_rust_provider_retry_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderRetryPolicy()
    monkeypatch.setattr(errors, "_load_rust_provider_retry_policy", lambda: rust)
    error = SimpleNamespace(
        headers={
            "Retry-After": "1m",
            "X-RateLimit-Reset-Requests": "2m",
        }
    )

    assert errors.parse_retry_window_seconds("3m") == 91
    assert errors.extract_rate_limit_backoff_seconds(error, 300) == 92
    assert rust.parse_calls[0][0] == "3m"
    assert isinstance(rust.parse_calls[0][1], float)
    assert rust.selection_calls[0][:5] == ("1m", "2m", None, None, 300)
    assert isinstance(rust.selection_calls[0][5], float)


def test_rust_provider_retry_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderRetryPolicy(fail=True)
    monkeypatch.setattr(errors, "_load_rust_provider_retry_policy", lambda: rust)
    error = SimpleNamespace(
        headers={
            "Retry-After": "invalid",
            "X-RateLimit-Reset-Requests": "3m",
        }
    )

    with caplog.at_level(logging.ERROR, logger=errors.__name__):
        assert errors.parse_retry_window_seconds("2m") == 120
        assert errors.extract_rate_limit_backoff_seconds(error, 300) == 180

    assert "using Python fallback" in caplog.text
