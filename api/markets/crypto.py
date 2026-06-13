from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from api.markets.price_commands import (
    SUPPORTED_PRICE_SYMBOLS,
    expand_price_tokens,
    find_coin_by_symbol_or_name,
    parse_amount_conversion,
    parse_conversion_only,
    price_query_parameter,
)
from api.utils import fmt_num

PriceListFetcher = Callable[[str], dict[str, Any] | None]
QuoteFetcher = Callable[..., dict[str, Any] | None]


def get_prices(
    msg_text: str,
    *,
    change_fields: Mapping[str, str],
    fetch_prices: PriceListFetcher,
    fetch_quotes: QuoteFetcher,
) -> str | None:
    msg_text, timeframe = _parse_timeframe(msg_text, change_fields)
    if timeframe is None and msg_text.strip():
        last_token = msg_text.strip().rsplit(None, 1)[-1].lower()
        if re.fullmatch(r"\d+[hd]", last_token):
            valid = ", ".join(change_fields)
            return f"timeframe '{last_token}' no soportado, uso: {valid}"
    change_field = change_fields.get(
        timeframe or "24h", "percent_change_24h"
    )
    timeframe_label = timeframe or "24h"

    prices_number = 0
    convert_to = "USD"
    convert_parameter = "USD"

    conversion_request = parse_amount_conversion(msg_text)
    if conversion_request:
        amount = conversion_request.amount
        source_symbol = conversion_request.source_symbol
        convert_to = conversion_request.target_symbol
        if convert_to not in SUPPORTED_PRICE_SYMBOLS:
            return f"no laburo con {convert_to} gordo"

        convert_parameter = conversion_request.target_parameter
        prices = fetch_prices(convert_parameter)
        if not prices or "data" not in prices:
            return "no pude traer precios de crypto boludo"

        requested_asset = find_coin_by_symbol_or_name(
            prices["data"], source_symbol
        )
        if not requested_asset:
            source_parameter = price_query_parameter(source_symbol)
            reverse_prices = fetch_prices(source_parameter)
            if not reverse_prices or "data" not in reverse_prices:
                return "no pude traer precios de crypto boludo"

            target_asset = find_coin_by_symbol_or_name(
                reverse_prices["data"], convert_to
            )
            if not target_asset:
                return "no laburo con esos ponzis boludo"

            source_amount = amount / 100000000 if source_symbol == "SATS" else amount
            asset_price = target_asset["quote"][source_parameter]["price"]
            converted_value = source_amount / asset_price
            return (
                f"{fmt_num(amount, 8)} {source_symbol} = "
                f"{fmt_num(converted_value, 8)} {target_asset['symbol'].upper()}"
            )

        quote_price = requested_asset["quote"][convert_parameter]["price"]
        if convert_to == "SATS":
            quote_price *= 100000000
        converted_value = amount * quote_price
        return (
            f"{fmt_num(amount, 8)} {requested_asset['symbol'].upper()} = "
            f"{fmt_num(converted_value, 8)} {convert_to}"
        )

    msg_text, convert_to, convert_parameter = parse_conversion_only(msg_text)
    if convert_to not in SUPPORTED_PRICE_SYMBOLS:
        return f"no laburo con {convert_to} gordo"

    prices = fetch_prices(convert_parameter)
    if msg_text:
        for number in msg_text.upper().replace(" ", "").split(","):
            try:
                prices_number = max(prices_number, int(float(number)))
            except ValueError:
                continue

    if msg_text.upper().isupper():
        if not prices or "data" not in prices:
            return "no pude traer precios de crypto boludo"
        coins = expand_price_tokens(msg_text.split(","))
        new_prices = _select_listed_coins(prices["data"], coins, prices_number)

        if not new_prices:
            requested = _fallback_quote_tokens(coins)
            new_prices = _fetch_requested_quotes(
                requested,
                convert_parameter=convert_parameter,
                fetch_quotes=fetch_quotes,
            )
            if not new_prices and requested:
                return f"no encontre esos ponzis: {', '.join(requested)}"
            if not new_prices:
                return "no pude traer precios de crypto boludo"

        prices_number = len(new_prices)
        price_rows = new_prices
    else:
        price_rows = prices["data"] if prices and "data" in prices else []

    if prices_number < 1:
        prices_number = 10
    if not prices or "data" not in prices:
        return "no pude traer precios de crypto boludo"

    lines = []
    for coin in price_rows[:prices_number]:
        quote = coin["quote"][convert_parameter]
        display_price = float(quote["price"])
        if convert_to == "SATS":
            display_price *= 100000000

        decimals = f"{display_price:.12f}".split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))
        price = f"{display_price:.{zeros + 4}f}".rstrip("0").rstrip(".")
        percentage = f"{quote.get(change_field, 0):+.2f}".rstrip("0").rstrip(".")
        lines.append(
            f"{coin['symbol']}: {price} {convert_to} "
            f"({percentage}% {timeframe_label})"
        )
    return "\n".join(lines)


def _parse_timeframe(
    msg_text: str, valid: Mapping[str, str]
) -> tuple[str, str | None]:
    parts = msg_text.strip().rsplit(None, 1)
    if parts and parts[-1].lower() in valid:
        timeframe = parts[-1].lower()
        remaining = parts[0].strip() if len(parts) > 1 else ""
        return remaining, timeframe
    return msg_text.strip(), None


def _select_listed_coins(
    listed: list[dict[str, Any]],
    requested: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    requested_set = set(requested)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, coin in enumerate(listed):
        symbol = str(coin.get("symbol") or "").upper().replace(" ", "")
        name = str(coin.get("name") or "").upper().replace(" ", "")
        if symbol not in requested_set and name not in requested_set and index >= top_n:
            continue
        identity = str(coin.get("id") or symbol or name)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(coin)
    return selected


def _fallback_quote_tokens(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(
        token
        for token in tokens
        if token not in {"STABLES", "STABLECOINS"} and not token.isdigit()
    ))


def _iter_quote_rows(quote_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in (quote_data or {}).values():
        candidates = value if isinstance(value, list) else [value]
        rows.extend(candidate for candidate in candidates if isinstance(candidate, dict))
    return rows


def _has_price(coin: dict[str, Any], convert_parameter: str) -> bool:
    return (
        coin.get("quote", {})
        .get(convert_parameter, {})
        .get("price")
        is not None
    )


def _fetch_requested_quotes(
    requested: list[str],
    *,
    convert_parameter: str,
    fetch_quotes: QuoteFetcher,
) -> list[dict[str, Any]]:
    if not requested:
        return []

    symbol_rows = _iter_quote_rows(fetch_quotes(requested, convert_parameter))
    found = [coin for coin in symbol_rows if _has_price(coin, convert_parameter)]
    found_tokens = {
        str(coin.get(field) or "").upper().replace(" ", "")
        for coin in found
        for field in ("symbol", "name", "slug")
    }
    missing = [token for token in requested if token not in found_tokens]
    if missing:
        slug_rows = _iter_quote_rows(
            fetch_quotes(
                [token.lower() for token in missing],
                convert_parameter,
                by_slug=True,
            )
        )
        found.extend(
            coin for coin in slug_rows if _has_price(coin, convert_parameter)
        )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for coin in found:
        identity = str(coin.get("id") or coin.get("symbol") or coin.get("slug") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(coin)
    return unique
