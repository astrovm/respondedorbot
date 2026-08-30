from __future__ import annotations

import json
import logging

from api.billing import provider_usage
from api.billing.reconciliation import has_unresolved_provider_usage


class _FakeRustAiUsagePolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.identity_calls: list[str] = []
        self.reconciliation_calls: list[str] = []

    def provider_segment_id(self, segment_json: str) -> str:
        self.identity_calls.append(segment_json)
        if self.fail:
            raise ValueError("synthetic Rust identity failure")
        return "rust:authoritative"

    def provider_usage_needs_reconciliation(self, segment_json: str) -> bool:
        self.reconciliation_calls.append(segment_json)
        if self.fail:
            raise ValueError("synthetic Rust reconciliation failure")
        return True


def test_rust_provider_segment_identity_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustAiUsagePolicy()
    monkeypatch.setattr(provider_usage, "_load_rust_ai_usage_policy", lambda: rust)
    segment = {
        "metadata": {"provider_generation_id": "generation-1"},
        "source": "openrouter",
    }

    assert provider_usage.provider_segment_id(segment) == "rust:authoritative"
    assert rust.identity_calls == [
        '{"metadata": {"provider_generation_id": "generation-1"}, '
        '"source": "openrouter"}'
    ]


def test_rust_reconciliation_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustAiUsagePolicy()
    monkeypatch.setattr(provider_usage, "_load_rust_ai_usage_policy", lambda: rust)
    segment = {
        "source": "groq",
        "metadata": {},
        "usage": {"cost": 10},
    }

    assert has_unresolved_provider_usage([segment]) is True
    assert [json.loads(value) for value in rust.reconciliation_calls] == [segment]


def test_rust_ai_usage_policy_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAiUsagePolicy(fail=True)
    monkeypatch.setattr(provider_usage, "_load_rust_ai_usage_policy", lambda: rust)
    segment = {
        "source": "openrouter",
        "metadata": {
            "provider_generation_id": "generation-1",
            "stream_interrupted": True,
        },
        "usage": {},
    }

    with caplog.at_level(logging.ERROR, logger=provider_usage.__name__):
        assert provider_usage.provider_segment_id(segment) == "openrouter:generation-1"
        assert has_unresolved_provider_usage([segment]) is True

    assert "using Python fallback" in caplog.text
