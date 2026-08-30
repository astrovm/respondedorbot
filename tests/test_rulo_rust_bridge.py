from __future__ import annotations

import json

import pytest

from api.i18n import use_locale
from api.markets import rulo


class _FakeRustRulo:
    def __init__(self, response: dict[str, object] | Exception) -> None:
        self.response = response
        self.input: dict[str, object] | None = None

    def evaluate_rulo(self, input_json: str) -> str:
        self.input = json.loads(input_json)
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def _provider_data() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        {
            "oficial": {"price": 1440},
            "blue": {"bid": 1430},
            "mep": {"al30": {"ci": {"price": 1459.73}}},
        },
        {
            "buenbit": {"totalAsk": 1.031},
            "xapo": {"totalAsk": 1.004},
        },
        {"buenbit": {"totalBid": 1458.44}},
    )


def test_rust_result_is_localized_and_excluded_exchanges_stay_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response: dict[str, object] = {
        "kind": "routes",
        "official": "1.440",
        "base_usd": "1.000",
        "base_ars": "1.440.000",
        "routes": [
            {
                "label": "USDT",
                "sell_price": "1.414,59",
                "difference": "-25,41",
                "percentage": "-1.76",
                "details": [
                    {"kind": "steps", "text": "USD→USDT BUENBIT, USDT→ARS BUENBIT"}
                ],
            }
        ],
    }
    rust = _FakeRustRulo(response)
    monkeypatch.setattr(rulo, "_load_rust_rulo_evaluator", lambda: rust)

    with use_locale("en"):
        result = rulo.build_rulo_message(*_provider_data())

    assert "Arbitrage from the official rate" in result
    assert "Steps: USD→USDT BUENBIT, USDT→ARS BUENBIT" in result
    assert rust.input is not None
    assert rust.input["usd_to_usdt"] == [{"exchange": "buenbit", "price": 1.031}]


def test_bridge_failure_uses_complete_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustRulo(ValueError("synthetic failure"))
    monkeypatch.setattr(rulo, "_load_rust_rulo_evaluator", lambda: rust)
    data = _provider_data()

    result = rulo.build_rulo_message(*data)

    assert result == rulo._build_rulo_message_python(*data)
    assert "using Python fallback" in caplog.text
