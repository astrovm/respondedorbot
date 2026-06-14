"""Normalized market records used by prompt formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CryptoQuote:
    symbol: str
    price: float
    change_24h: float | None = None
    dominance: float | None = None


@dataclass(frozen=True, slots=True)
class DollarQuote:
    label: str
    price: float
    bid: float | None = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    crypto: tuple[CryptoQuote, ...] = ()
    dollars: tuple[DollarQuote, ...] = ()


def _number(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        pass
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _crypto_quote(value: Any) -> CryptoQuote | None:
    row = _mapping(value)
    symbol = str(row.get("symbol") or row.get("name") or "").strip().upper()
    usd = _mapping(_mapping(row.get("quote")).get("USD"))
    price = _number(usd.get("price") if usd else row.get("price"))
    if not symbol or price is None:
        return None
    changes = _mapping(usd.get("changes"))
    return CryptoQuote(
        symbol=symbol,
        price=price,
        change_24h=_number(
            changes.get("24h") if usd else row.get("change_24h")
        ),
        dominance=_number(usd.get("dominance")) if usd else None,
    )


def _normalize_crypto(value: Any) -> tuple[CryptoQuote, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return ()
    quotes = (_crypto_quote(row) for row in value[:3])
    return tuple(quote for quote in quotes if quote is not None)


def _sequence_dollars(value: Sequence[Any]) -> tuple[DollarQuote, ...]:
    quotes: list[DollarQuote] = []
    for value_item in value:
        item = _mapping(value_item)
        label = str(item.get("name") or item.get("label") or "").strip().lower()
        price = _number(item.get("price"))
        if label and price is not None:
            quotes.append(DollarQuote(label=label, price=price))
    return tuple(quotes)


def _quote(
    label: str,
    value: Any,
    *,
    price_keys: tuple[str, ...] = ("price",),
) -> DollarQuote | None:
    row = _mapping(value)
    price = next(
        (
            parsed
            for key in price_keys
            if (parsed := _number(row.get(key))) is not None
        ),
        None,
    )
    if price is None:
        return None
    return DollarQuote(label=label, price=price, bid=_number(row.get("bid")))


def _mapping_dollars(value: Mapping[str, Any]) -> tuple[DollarQuote, ...]:
    mep = _mapping(_mapping(_mapping(value.get("mep")).get("al30")).get("ci"))
    crypto = _mapping(_mapping(value.get("cripto")).get("usdt"))
    candidates = (
        _quote("oficial", value.get("oficial")),
        _quote("blue", value.get("blue"), price_keys=("ask", "price")),
        _quote("mep al30 ci", mep),
        _quote("tarjeta", value.get("tarjeta")),
        _quote("usdt", crypto, price_keys=("ask",)),
    )
    return tuple(quote for quote in candidates if quote is not None)


def _normalize_dollars(value: Any) -> tuple[DollarQuote, ...]:
    if isinstance(value, Mapping):
        return _mapping_dollars(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return _sequence_dollars(value)
    return ()


def normalize_market_snapshot(market: Mapping[str, Any]) -> MarketSnapshot:
    return MarketSnapshot(
        crypto=_normalize_crypto(market.get("crypto")),
        dollars=_normalize_dollars(market.get("dollar")),
    )
