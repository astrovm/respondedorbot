from __future__ import annotations

import logging

from api.providers import errors


class _SyntheticProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        status: int | None = None,
        code: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.status = status
        self.code = code


class _FakeRustProviderErrorPolicy:
    def __init__(self, result: tuple[bool, bool], *, fail: bool = False) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[int | None, int | None, str, str]] = []

    def classify_provider_error(
        self,
        status_code: int | None,
        status: int | None,
        code: str,
        message: str,
    ) -> tuple[bool, bool]:
        self.calls.append((status_code, status, code, message))
        if self.fail:
            raise ValueError("synthetic Rust provider policy failure")
        return self.result


def test_rust_provider_error_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderErrorPolicy((False, True))
    monkeypatch.setattr(
        errors,
        "_load_rust_provider_error_policy",
        lambda: rust,
    )
    error = _SyntheticProviderError(
        "not a Python fallback match",
        status_code=500,
        code="server_error",
    )

    assert errors.is_rate_limit_error(error) is False
    assert errors.should_try_next_groq_account(error) is True
    assert rust.calls == [
        (500, None, "server_error", "not a Python fallback match"),
        (500, None, "server_error", "not a Python fallback match"),
    ]


def test_rust_provider_error_policy_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        errors,
        "_load_rust_provider_error_policy",
        lambda: _FakeRustProviderErrorPolicy((False, False), fail=True),
    )
    error = _SyntheticProviderError(
        "Error 429: rate limit; request_too_large",
    )

    with caplog.at_level(logging.ERROR, logger=errors.__name__):
        assert errors.is_rate_limit_error(error) is True
        assert errors.should_try_next_groq_account(error) is True

    assert "using Python fallback" in caplog.text
