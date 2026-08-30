from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from api.providers import support


class _FakeRustProviderUsageNormalization:
    def __init__(self, *, fail: bool = False, invalid: bool = False) -> None:
        self.fail = fail
        self.invalid = invalid
        self.calls: list[tuple[object, ...]] = []

    def provider_normalize_usage(self, *arguments):
        self.calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust provider-usage failure")
        if self.invalid:
            return ("", None, None, None, "")
        return (
            "rust/resolved-model",
            "rust/requested-model",
            "RustUpstream",
            "rust-tier",
            "rust-source",
        )


def _build_result() -> object:
    return support.build_usage_result(
        kind="chat",
        text="synthetic response",
        model="python/requested-model",
        response=SimpleNamespace(
            model="python/resolved-model",
            provider="PythonUpstream",
            service_tier="python-tier",
            usage={"prompt_tokens": 3},
        ),
        audio_seconds=None,
        cached=False,
        metadata={
            "provider": "python-source",
            "upstream_provider": "preexisting-upstream",
        },
        extract_usage=lambda response: response.usage,
    )


def test_rust_provider_usage_normalization_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderUsageNormalization()
    monkeypatch.setattr(
        support,
        "_load_rust_provider_usage_normalization",
        lambda: rust,
    )

    result = _build_result()

    assert result.model == "rust/resolved-model"
    assert result.source == "rust-source"
    assert result.metadata == {
        "provider": "python-source",
        "requested_model": "rust/requested-model",
        "upstream_provider": "preexisting-upstream",
        "service_tier": "rust-tier",
    }
    assert rust.calls == [
        (
            "python/requested-model",
            "python/resolved-model",
            "PythonUpstream",
            "python-tier",
            "python-source",
        )
    ]


def test_rust_provider_usage_failure_preserves_python_behavior(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderUsageNormalization(fail=True)
    monkeypatch.setattr(
        support,
        "_load_rust_provider_usage_normalization",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=support.__name__):
        result = _build_result()

    assert result.model == "python/resolved-model"
    assert result.source == "python-source"
    assert result.metadata["requested_model"] == "python/requested-model"
    assert result.metadata["upstream_provider"] == "preexisting-upstream"
    assert result.metadata["service_tier"] == "python-tier"
    assert "using Python fallback" in caplog.text


def test_invalid_rust_provider_usage_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustProviderUsageNormalization(invalid=True)
    monkeypatch.setattr(
        support,
        "_load_rust_provider_usage_normalization",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=support.__name__):
        result = _build_result()

    assert result.model == "python/resolved-model"
    assert result.source == "python-source"
    assert "using Python fallback" in caplog.text


def test_python_provider_usage_normalization_matches_shared_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        support,
        "_load_rust_provider_usage_normalization",
        lambda: None,
    )
    path = (
        Path(__file__).parents[1]
        / "contracts"
        / "provider_usage_normalization.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        actual = support._normalize_provider_usage(*case["input"])
        assert list(actual) == case["expected"], case["name"]
