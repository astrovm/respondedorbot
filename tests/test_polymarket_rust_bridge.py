from __future__ import annotations

import json

import pytest

from api.markets import polymarket
from api.markets.polymarket import MarketQuote


class _FakeRustRanker:
    def __init__(self, response: list[dict[str, object]] | Exception) -> None:
        self.response = response
        self.inputs: list[dict[str, object]] | None = None
        self.limit: int | None = None

    def rank_polymarket_outcomes(self, input_json: str, limit: int) -> str:
        self.inputs = json.loads(input_json)
        self.limit = limit
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_polymarket_ranking_uses_captured_live_prices_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustRanker(
        [
            {"title": "Second", "percentage": 75.0},
            {"title": "First", "percentage": 25.0},
        ]
    )
    monkeypatch.setattr(polymarket, "_load_rust_polymarket_ranker", lambda: rust)
    live_calls: list[str] = []
    live_prices = {"synthetic-token-a": 0.25, "synthetic-token-b": 0.75}

    def fetch_live(token_id: str) -> tuple[float, None]:
        live_calls.append(token_id)
        return live_prices[token_id], None

    result = polymarket._top_outcomes(
        [
            MarketQuote("First", 0.9, "synthetic-token-a"),
            MarketQuote("Second", 0.1, "synthetic-token-b"),
        ],
        limit=2,
        fetch_live=fetch_live,
    )

    assert result == [("Second", 75.0), ("First", 25.0)]
    assert live_calls == ["synthetic-token-a", "synthetic-token-b"]
    assert rust.limit == 2
    assert rust.inputs == [
        {
            "title": "First",
            "cached_probability": 0.9,
            "live_probability": 0.25,
        },
        {
            "title": "Second",
            "cached_probability": 0.1,
            "live_probability": 0.75,
        },
    ]


def test_polymarket_ranking_fallback_does_not_repeat_live_fetches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustRanker(ValueError("synthetic bridge failure"))
    monkeypatch.setattr(polymarket, "_load_rust_polymarket_ranker", lambda: rust)
    live_calls: list[str] = []

    def fetch_live(token_id: str) -> tuple[float, None]:
        live_calls.append(token_id)
        return 0.8, None

    result = polymarket._top_outcomes(
        [MarketQuote("Synthetic candidate", 0.2, "candidate")],
        limit=2,
        fetch_live=fetch_live,
    )

    assert result == [("Synthetic candidate", 80.0)]
    assert live_calls == ["candidate"]
