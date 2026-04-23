from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

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


def normalize_price_symbol(value: str) -> str:
    return value.upper().replace(" ", "")


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
        rf"^\s*([0-9]+(?:[\.,][0-9]+)?)\s+([a-zA-Z0-9]+)\s+(?:{conversion_token_pattern})\s+([a-zA-Z0-9]+)\s*$",
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
        rf"^\s*(?:{conversion_token_pattern})\s+([a-zA-Z0-9]+)\s*$",
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
