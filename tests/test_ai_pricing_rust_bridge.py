from __future__ import annotations

import json
import logging

from api.ai import pricing


class _FakeRustAiPricing:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def ai_calculate_billing_for_segments(self, segments_json: str) -> str:
        self.calls.append(segments_json)
        if self.fail:
            raise ValueError("synthetic Rust AI pricing failure")
        return json.dumps({"implementation": "rust", "charged_credit_units": 91})


def test_rust_ai_pricing_is_authoritative_and_materializes_iterables(monkeypatch) -> None:
    rust = _FakeRustAiPricing()
    monkeypatch.setattr(pricing, "_load_rust_ai_pricing", lambda: rust)
    segments = (
        segment
        for segment in [
            {"kind": "chat", "usage": {"cost": "0.000001"}},
            None,
        ]
    )

    assert pricing.calculate_billing_for_segments(segments) == {
        "implementation": "rust",
        "charged_credit_units": 91,
    }
    assert json.loads(rust.calls[0]) == [
        {"kind": "chat", "usage": {"cost": "0.000001"}},
        None,
    ]


def test_rust_ai_pricing_failure_preserves_exact_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAiPricing(fail=True)
    monkeypatch.setattr(pricing, "_load_rust_ai_pricing", lambda: rust)
    segments = [
        {
            "kind": "chat",
            "model": "unknown/model",
            "usage": {"cost": "0.00000003"},
            "metadata": {"provider": "openrouter"},
        },
        {
            "kind": "chat",
            "model": "unknown/model",
            "usage": {"cost": "0.00000003"},
            "metadata": {"provider": "openrouter"},
        },
    ]

    with caplog.at_level(logging.ERROR, logger=pricing.__name__):
        result = pricing.calculate_billing_for_segments(segments)

    assert result["raw_usd_micros_exact"] == "0.06000000"
    assert result["charged_credit_units"] == 1
    assert result["pricing_complete"] is True
    assert "using Python fallback" in caplog.text
