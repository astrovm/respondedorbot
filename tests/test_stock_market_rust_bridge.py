from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

from api.markets import stocks


class _FakeRustStocks:
    def __init__(self) -> None:
        self.quote_outcome: object = None
        self.symbol_outcome: object = None
        self.query_outcome: object = None
        self.finviz_outcome: object = None
        self.calls: list[tuple[object, ...]] = []
        self.fail = False

    def _result(self, name: str, arguments: tuple[object, ...], outcome: object) -> object:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust stock failure")
        return outcome

    def stock_parse_yahoo_quote(self, *arguments: object) -> str:
        return json.dumps(self._result("quote", arguments, self.quote_outcome))

    def stock_select_yahoo_symbol(self, *arguments: object) -> object:
        return self._result("symbol", arguments, self.symbol_outcome)

    def stock_query_plan(self, *arguments: object) -> str:
        return json.dumps(self._result("query", arguments, self.query_outcome))

    def finviz_fetch(self, *arguments: object) -> str:
        return json.dumps(self._result("finviz", arguments, self.finviz_outcome))


def test_rust_yahoo_quote_parser_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustStocks()
    rust.quote_outcome = {
        "symbol": "EXM",
        "name": "Example",
        "price": 12.5,
        "currency": "EUR",
        "exchange": "Synthetic",
        "variation": -1.25,
    }
    monkeypatch.setattr(stocks, "_load_rust_stock_adapter", lambda: rust)
    response = {"data": {"provider": "payload-is-owned-by-rust"}}

    actual = stocks.fetch_yahoo_stock_quote(
        "EXM",
        cached_request=MagicMock(return_value=response),
        cache_ttl=300,
    )

    assert actual == stocks.StockQuote(
        "EXM", "Example", 12.5, "EUR", "Synthetic", -1.25
    )
    assert rust.calls == [
        ("quote", json.dumps(response, separators=(",", ":")), "EXM")
    ]


def test_rust_symbol_selection_and_query_plan_are_authoritative(monkeypatch) -> None:
    rust = _FakeRustStocks()
    rust.symbol_outcome = "EXM"
    rust.query_outcome = {
        "raw_query": "Example Holdings",
        "queries": [
            {
                "original": "Example Holdings",
                "normalized": "EXAMPLE HOLDINGS",
                "is_symbol": False,
            }
        ],
        "full_query_fallback": False,
        "needs_top_stocks": False,
    }
    monkeypatch.setattr(stocks, "_load_rust_stock_adapter", lambda: rust)
    cached = MagicMock(return_value={"data": {"quotes": []}})

    assert (
        stocks.search_yahoo_symbol(
            "Example Holdings", cached_request=cached, cache_ttl=300
        )
        == "EXM"
    )
    plan = stocks._stock_query_plan("Example Holdings")
    assert plan.queries == [
        stocks._StockQuery("Example Holdings", "EXAMPLE HOLDINGS", False)
    ]


def test_rust_finviz_fetch_is_authoritative_and_cached(monkeypatch) -> None:
    rust = _FakeRustStocks()
    rust.finviz_outcome = {"status": "success", "symbols": ["EXM", "ALT"]}
    monkeypatch.setattr(stocks, "_load_rust_stock_adapter", lambda: rust)
    redis_client = MagicMock()
    get_json = MagicMock(return_value=None)
    set_json = MagicMock()
    python_get = MagicMock(side_effect=AssertionError("Python HTTP must not run"))

    actual = stocks.fetch_top_stocks_by_market_cap(
        redis_factory=lambda: redis_client,
        redis_get_json=get_json,
        redis_set_json=set_json,
        http_get=python_get,
        cache_ttl=600,
    )

    assert actual == ["EXM", "ALT"]
    set_json.assert_called_once_with(
        redis_client,
        "market:stock_screener:mega_cap",
        ["EXM", "ALT"],
        ttl=600,
    )
    python_get.assert_not_called()


def test_invalid_rust_stock_results_use_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustStocks()
    rust.fail = True
    monkeypatch.setattr(stocks, "_load_rust_stock_adapter", lambda: rust)
    cached_response = {
        "data": {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "EXM",
                            "regularMarketPrice": 12,
                            "chartPreviousClose": 10,
                        }
                    }
                ]
            }
        }
    }

    with caplog.at_level(logging.ERROR, logger=stocks.__name__):
        quote = stocks.fetch_yahoo_stock_quote(
            "EXM",
            cached_request=MagicMock(return_value=cached_response),
            cache_ttl=300,
        )

    assert quote is not None
    assert quote.price == 12
    assert quote.variation == 20
    assert "using Python fallback" in caplog.text


def test_python_helpers_match_shared_query_and_finviz_contracts() -> None:
    path = Path(__file__).parents[1] / "contracts" / "stock_market.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["query_cases"]:
        assert json.loads(
            json.dumps(
                asdict(stocks._python_stock_query_plan(case["message"])),
            )
        ) == case["expected"], case["name"]
    for case in contract["finviz_cases"]:
        assert stocks._parse_finviz_symbols(case["html"]) == case["expected"], case[
            "name"
        ]
