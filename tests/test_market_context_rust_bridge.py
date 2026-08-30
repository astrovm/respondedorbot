from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.markets import context


def test_python_fallback_matches_shared_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "market_context.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        assert context._format_market_info_python(case["input"]) == case["expected"], case[
            "name"
        ]


class _FakeRustFormatter:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.input: Any = None

    def format_market_info(self, market_json: str) -> str:
        self.input = json.loads(market_json)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_rust_formatter_receives_the_snapshot(monkeypatch) -> None:
    rust = _FakeRustFormatter("PRECIOS DE CRIPTOS:\n- BTC: 1 usd")
    monkeypatch.setattr(context, "_load_rust_market_context_formatter", lambda: rust)
    market = {"crypto": [{"symbol": "BTC", "price": 1}]}

    result = context.format_market_info(market)

    assert result == "PRECIOS DE CRIPTOS:\n- BTC: 1 usd"
    assert rust.input == market


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustFormatter(RuntimeError("synthetic bridge failure"))
    monkeypatch.setattr(context, "_load_rust_market_context_formatter", lambda: rust)

    result = context.format_market_info(
        {"dollar": [{"name": "Oficial", "price": 1000}]}
    )

    assert result == "DOLARES:\n- oficial: 1000"
    assert "using Python fallback" in caplog.text
