from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, Tuple, cast

from api.core.rust_bridge import load_rust_bridge

logger = logging.getLogger(__name__)

SUPPORTED_PRICE_SYMBOLS = {
    "ARS",
    "AUD",
    "BRL",
    "BTC",
    "BUSD",
    "CAD",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CZK",
    "DAI",
    "DKK",
    "ETH",
    "EUR",
    "GBP",
    "HKD",
    "ILS",
    "INR",
    "ISK",
    "JPY",
    "KRW",
    "MXN",
    "NZD",
    "PEN",
    "SATS",
    "SEK",
    "SGD",
    "TWD",
    "USD",
    "USDC",
    "USDT",
    "UYU",
    "XAU",
    "XMR",
}

STABLECOIN_SYMBOLS = (
    "BUSD",
    "DAI",
    "DOC",
    "EURT",
    "FDUSD",
    "FRAX",
    "GHO",
    "GUSD",
    "LUSD",
    "MAI",
    "MIM",
    "MIMATIC",
    "NUARS",
    "PAXG",
    "PYUSD",
    "RAI",
    "SUSD",
    "TUSD",
    "USDC",
    "USDD",
    "USDM",
    "USDP",
    "USDT",
    "UXD",
    "XAUT",
    "XSGD",
)

CONVERSION_PREPOSITIONS = ("in", "to", "a", "en")


@dataclass(frozen=True)
class AmountConversionRequest:
    amount: float
    source_symbol: str
    target_symbol: str
    target_parameter: str


@dataclass(frozen=True)
class UnsupportedPriceTimeframe:
    kind: Literal["unsupported_timeframe"]
    timeframe: str


@dataclass(frozen=True)
class AssetPriceQuery:
    kind: Literal["assets"]
    query: str
    timeframe: str | None
    target_symbol: str
    target_parameter: str
    conversion_requested: bool
    provider_scope: Literal["crypto", "stock"] | None


type ParsedPriceQuery = (
    AmountConversionRequest | UnsupportedPriceTimeframe | AssetPriceQuery
)


class _RustPriceQueryParser(Protocol):
    def parse_price_query(
        self,
        message_text: str,
        valid_timeframes_json: str,
    ) -> str: ...


def _load_rust_price_query_parser() -> Optional[_RustPriceQueryParser]:
    module = load_rust_bridge("RUST_PRICE_QUERY_PARSING_ENABLED")
    if module is None:
        return None
    return cast(_RustPriceQueryParser, module)


def normalize_price_symbol(value: str) -> str:
    return value.upper().replace(" ", "").lstrip("$")


def price_query_parameter(symbol: str) -> str:
    return "BTC" if symbol == "SATS" else symbol


def expand_price_tokens(tokens: Sequence[str]) -> list[str]:
    expanded = [normalize_price_symbol(token) for token in tokens]
    if "STABLES" in expanded or "STABLECOINS" in expanded:
        expanded.extend(STABLECOIN_SYMBOLS)
    return expanded


def parse_amount_conversion(text: str) -> Optional[AmountConversionRequest]:
    conversion_token_pattern = "|".join(CONVERSION_PREPOSITIONS)
    match = re.match(
        rf"^\s*([0-9]+(?:[\.,][0-9]+)?)\s+(\$?[a-zA-Z0-9]+)\s+(?:{conversion_token_pattern})\s+(\$?[a-zA-Z0-9]+)\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    amount_text, source_symbol, target_symbol = match.groups()
    normalized_target = normalize_price_symbol(target_symbol)
    return AmountConversionRequest(
        amount=float(amount_text.replace(",", ".")),
        source_symbol=normalize_price_symbol(source_symbol),
        target_symbol=normalized_target,
        target_parameter=price_query_parameter(normalized_target),
    )


def parse_conversion_only(text: str) -> Tuple[str, str, str]:
    conversion_token_pattern = "|".join(CONVERSION_PREPOSITIONS)
    match = re.match(
        rf"^\s*(?:{conversion_token_pattern})\s+(\$?[a-zA-Z0-9]+)\s*$",
        text,
        re.IGNORECASE,
    )
    if match:
        target = normalize_price_symbol(match.group(1))
        return "", target, price_query_parameter(target)

    split_parts = re.split(
        rf"\s+(?:{conversion_token_pattern})\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(split_parts) == 2:
        target = normalize_price_symbol(split_parts[1].strip())
        return split_parts[0].strip(), target, price_query_parameter(target)
    return text, "USD", "USD"


def _parse_price_timeframe(
    text: str,
    valid_timeframes: Sequence[str],
) -> tuple[str, str | None]:
    parts = text.strip().rsplit(None, 1)
    if parts and parts[-1].lower() in valid_timeframes:
        timeframe = parts[-1].lower()
        remaining = parts[0].strip() if len(parts) > 1 else ""
        return remaining, timeframe
    return text.strip(), None


def _unsupported_price_timeframe(text: str) -> str | None:
    if not text.strip():
        return None
    last_token = text.strip().rsplit(None, 1)[-1].lower()
    return last_token if re.fullmatch(r"\d+[hd]", last_token) else None


def _has_conversion_modifier(text: str) -> bool:
    return (
        re.search(
            r"(?:^|\s)(?:in|to|a|en)\s+\$?[a-zA-Z0-9]+\s*$",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def _parse_provider_scope(text: str) -> tuple[Literal["crypto", "stock"] | None, str]:
    match = re.match(r"^\s*(crypto|stock)\s*:\s*(.*?)\s*$", text, re.IGNORECASE)
    if not match:
        return None, text
    scope = match.group(1).lower()
    if scope not in ("crypto", "stock"):
        return None, text
    return cast(Literal["crypto", "stock"], scope), match.group(2)


def _parse_price_query_python(
    text: str,
    valid_timeframes: Sequence[str],
) -> ParsedPriceQuery:
    text, timeframe = _parse_price_timeframe(text, valid_timeframes)
    if timeframe is None:
        unsupported = _unsupported_price_timeframe(text)
        if unsupported is not None:
            return UnsupportedPriceTimeframe(
                kind="unsupported_timeframe",
                timeframe=unsupported,
            )

    conversion = parse_amount_conversion(text)
    if conversion is not None:
        return conversion

    conversion_requested = _has_conversion_modifier(text)
    query, target_symbol, target_parameter = parse_conversion_only(text)
    provider_scope, query = _parse_provider_scope(query)
    return AssetPriceQuery(
        kind="assets",
        query=query,
        timeframe=timeframe,
        target_symbol=target_symbol,
        target_parameter=target_parameter,
        conversion_requested=conversion_requested,
        provider_scope=provider_scope,
    )


def _price_query_from_rust(raw_result: str) -> ParsedPriceQuery:
    result = json.loads(raw_result)
    if not isinstance(result, Mapping):
        raise ValueError("Rust price query result is not a mapping")
    kind = result.get("kind")
    if kind == "unsupported_timeframe":
        return UnsupportedPriceTimeframe(
            kind="unsupported_timeframe",
            timeframe=str(result["timeframe"]),
        )
    if kind == "amount_conversion":
        return AmountConversionRequest(
            amount=float(result["amount"]),
            source_symbol=str(result["source_symbol"]),
            target_symbol=str(result["target_symbol"]),
            target_parameter=str(result["target_parameter"]),
        )
    if kind == "assets":
        raw_timeframe = result.get("timeframe")
        raw_scope = result.get("provider_scope")
        if raw_scope not in (None, "crypto", "stock"):
            raise ValueError("Rust price query result has an unknown provider scope")
        return AssetPriceQuery(
            kind="assets",
            query=str(result["query"]),
            timeframe=None if raw_timeframe is None else str(raw_timeframe),
            target_symbol=str(result["target_symbol"]),
            target_parameter=str(result["target_parameter"]),
            conversion_requested=bool(result["conversion_requested"]),
            provider_scope=cast(Literal["crypto", "stock"] | None, raw_scope),
        )
    raise ValueError("Rust price query result has an unknown kind")


def parse_price_query(
    text: str,
    valid_timeframes: Sequence[str],
) -> ParsedPriceQuery:
    rust = _load_rust_price_query_parser()
    if rust is not None:
        try:
            return _price_query_from_rust(
                rust.parse_price_query(
                    text,
                    json.dumps(list(valid_timeframes), separators=(",", ":")),
                )
            )
        except Exception as error:
            logger.warning(
                "Rust price query parsing failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _parse_price_query_python(text, valid_timeframes)


def find_coin_by_symbol_or_name(
    coins: Sequence[Mapping[str, Any]], token: str
) -> Optional[Mapping[str, Any]]:
    normalized = normalize_price_symbol(token)
    for coin in coins:
        symbol = normalize_price_symbol(str(coin.get("symbol", "")))
        name = normalize_price_symbol(str(coin.get("name", "")))
        if symbol == normalized or name == normalized:
            return coin
    return None
