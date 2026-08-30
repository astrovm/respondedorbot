from __future__ import annotations

import json

import pytest

from api import index
from api.i18n import use_locale


class _FakeRustMarketModel:
    def __init__(self, response: dict[str, str] | Exception) -> None:
        self.response = response
        self.input: tuple[str, int, float] | None = None

    def evaluate_market_model(self, model: str, elapsed_days: int, market_price: float) -> str:
        self.input = (model, elapsed_days, market_price)
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_rust_market_model_is_localized(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustMarketModel(
        {"value": "57869.18", "percentage": "13.60", "valuation": "cheap"}
    )
    monkeypatch.setattr(index, "_load_rust_market_model", lambda: rust)

    with use_locale("en"):
        result = index._market_model_command(
            "power_law", 5475, 50000.0, "market.powerlaw.result"
        )

    assert result == "power law estimates BTC at 57869.18 USD (13.60% undervalued)"
    assert rust.input == ("power_law", 5475, 50000.0)


def test_bridge_failure_uses_same_inputs_in_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustMarketModel(ValueError("synthetic failure"))
    monkeypatch.setattr(index, "_load_rust_market_model", lambda: rust)

    with use_locale("es"):
        result = index._market_model_command(
            "rainbow", 5470, 50000.0, "market.rainbow.result"
        )

    assert result == "segun rainbow chart btc deberia estar en 97886.11 usd (48.92% regalado gordo)"
    assert rust.input == ("rainbow", 5470, 50000.0)
    assert "using Python fallback" in caplog.text
