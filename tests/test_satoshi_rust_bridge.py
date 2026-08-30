from __future__ import annotations

import logging

import pytest

from api.markets import dollar


class _FakeRustSatoshiFormatter:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.input: tuple[float, float] | None = None

    def format_satoshi_quote(self, price_usd: float, price_ars: float) -> str:
        self.input = (price_usd, price_ars)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_rust_formats_prices_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = "synthetic satoshi quote"
    rust = _FakeRustSatoshiFormatter(expected)
    monkeypatch.setattr(dollar, "_load_rust_satoshi_formatter", lambda: rust)
    currencies: list[str] = []

    def get_price(currency: str) -> float:
        currencies.append(currency)
        return 50_000.0 if currency == "USD" else 10_000_000.0

    result = dollar.satoshi(get_btc_price=get_price, logger=logging.getLogger("tests.satoshi"))

    assert result == expected
    assert currencies == ["USD", "ARS"]
    assert rust.input == (50_000.0, 10_000_000.0)


def test_bridge_failure_uses_same_prices_in_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rust = _FakeRustSatoshiFormatter(ValueError("synthetic failure"))
    monkeypatch.setattr(dollar, "_load_rust_satoshi_formatter", lambda: rust)
    caplog.set_level(logging.WARNING, logger="tests.satoshi")
    currencies: list[str] = []

    def get_price(currency: str) -> float:
        currencies.append(currency)
        return 50_000.0 if currency == "USD" else 10_000_000.0

    result = dollar.satoshi(get_btc_price=get_price, logger=logging.getLogger("tests.satoshi"))

    assert result == dollar._format_satoshi_python(50_000.0, 10_000_000.0)
    assert currencies == ["USD", "ARS"]
    assert "using Python fallback" in caplog.text
