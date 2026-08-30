from __future__ import annotations

import json

import pytest

from api.markets import dollar


class _FakeRustDevo:
    def __init__(self) -> None:
        self.parse_response: tuple[str, float, float] | Exception = ("valid", 0.005, 100.0)
        self.calculate_response: dict[str, object] | Exception = {
            "profit": "62.68",
            "fee": "0.5",
            "official": "100",
            "usdt": "195",
            "card": "150",
        }
        self.calculate_input: tuple[float, ...] | None = None

    def parse_devo_input(self, _message_text: str) -> tuple[str, float, float]:
        if isinstance(self.parse_response, Exception):
            raise self.parse_response
        return self.parse_response

    def calculate_devo(self, *values: float) -> str:
        self.calculate_input = values
        if isinstance(self.calculate_response, Exception):
            raise self.calculate_response
        return json.dumps(self.calculate_response)


def _quotes() -> dict[str, object]:
    return {
        "data": {
            "oficial": {"price": 100.0},
            "tarjeta": {"price": 150.0},
            "cripto": {"usdt": {"ask": 200.0, "bid": 190.0}},
        }
    }


def test_rust_parser_and_calculator_share_one_provider_read(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustDevo()
    monkeypatch.setattr(dollar, "_load_rust_devo_calculator", lambda: rust)
    fetch_calls = 0

    def fetch() -> dict[str, object]:
        nonlocal fetch_calls
        fetch_calls += 1
        return _quotes()

    result = dollar.get_devo("0.5,100", fetch_dollars=fetch)

    assert "ganancia: 62.68%" in result
    assert fetch_calls == 1
    assert rust.calculate_input == (0.005, 100.0, 100.0, 150.0, 200.0, 190.0)


def test_bridge_failures_use_python_without_refetching(monkeypatch: pytest.MonkeyPatch) -> None:
    rust = _FakeRustDevo()
    rust.calculate_response = ValueError("synthetic failure")
    monkeypatch.setattr(dollar, "_load_rust_devo_calculator", lambda: rust)
    fetch_calls = 0

    def fetch() -> dict[str, object]:
        nonlocal fetch_calls
        fetch_calls += 1
        return _quotes()

    result = dollar.get_devo("0.5,100", fetch_dollars=fetch)

    assert "Ganarias 9402.5 ARS / 48.22 USDT" in result
    assert fetch_calls == 1


def test_unicode_parser_fallback_preserves_python_float_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustDevo()
    rust.parse_response = ValueError("unsupported Unicode")
    monkeypatch.setattr(dollar, "_load_rust_devo_calculator", lambda: rust)

    result = dollar.get_devo("０.５", fetch_dollars=_quotes)

    assert "ganancia: 62.68%" in result
