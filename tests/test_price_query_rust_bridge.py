from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.markets import price_commands


def _contract_result(
    result: price_commands.ParsedPriceQuery,
) -> dict[str, Any]:
    if isinstance(result, price_commands.AmountConversionRequest):
        return {
            "kind": "amount_conversion",
            "amount": result.amount,
            "source_symbol": result.source_symbol,
            "target_symbol": result.target_symbol,
            "target_parameter": result.target_parameter,
        }
    if isinstance(result, price_commands.UnsupportedPriceTimeframe):
        return {
            "kind": result.kind,
            "timeframe": result.timeframe,
        }
    return {
        "kind": result.kind,
        "query": result.query,
        "timeframe": result.timeframe,
        "target_symbol": result.target_symbol,
        "target_parameter": result.target_parameter,
        "conversion_requested": result.conversion_requested,
        "provider_scope": result.provider_scope,
    }


def test_python_fallback_matches_shared_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "price_queries.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        actual = price_commands._parse_price_query_python(
            case["input"],
            contract["valid_timeframes"],
        )
        assert _contract_result(actual) == case["expected"], case["name"]


class _FakeRustParser:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.input: tuple[str, list[str]] | None = None

    def parse_price_query(self, message_text: str, valid_timeframes_json: str) -> str:
        self.input = (message_text, json.loads(valid_timeframes_json))
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_rust_asset_query_becomes_typed_python_query(monkeypatch) -> None:
    rust = _FakeRustParser(
        {
            "kind": "assets",
            "query": "NVDA",
            "timeframe": "7d",
            "target_symbol": "EUR",
            "target_parameter": "EUR",
            "conversion_requested": True,
            "provider_scope": "stock",
        }
    )
    monkeypatch.setattr(price_commands, "_load_rust_price_query_parser", lambda: rust)

    result = price_commands.parse_price_query("stock:NVDA in EUR 7d", ["24h", "7d"])

    assert result == price_commands.AssetPriceQuery(
        kind="assets",
        query="NVDA",
        timeframe="7d",
        target_symbol="EUR",
        target_parameter="EUR",
        conversion_requested=True,
        provider_scope="stock",
    )
    assert rust.input == ("stock:NVDA in EUR 7d", ["24h", "7d"])


def test_rust_amount_conversion_becomes_existing_request_type(monkeypatch) -> None:
    rust = _FakeRustParser(
        {
            "kind": "amount_conversion",
            "amount": 2.5,
            "source_symbol": "BTC",
            "target_symbol": "SATS",
            "target_parameter": "BTC",
        }
    )
    monkeypatch.setattr(price_commands, "_load_rust_price_query_parser", lambda: rust)

    result = price_commands.parse_price_query("2.5 BTC in SATS", ["24h"])

    assert result == price_commands.AmountConversionRequest(2.5, "BTC", "SATS", "BTC")


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustParser(RuntimeError("synthetic bridge failure"))
    monkeypatch.setattr(price_commands, "_load_rust_price_query_parser", lambda: rust)

    result = price_commands.parse_price_query("btc 24h", ["24h"])

    assert result == price_commands.AssetPriceQuery(
        kind="assets",
        query="btc",
        timeframe="24h",
        target_symbol="USD",
        target_parameter="USD",
        conversion_requested=False,
        provider_scope=None,
    )
    assert "using Python fallback" in caplog.text
