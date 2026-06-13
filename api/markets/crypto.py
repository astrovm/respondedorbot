from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from api.markets.price_commands import (
    AmountConversionRequest,
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


@dataclass(frozen=True, slots=True)
class PriceDisplay:
    convert_to: str
    convert_parameter: str
    change_field: str
    timeframe_label: str


def get_prices(
    msg_text: str,
    *,
    change_fields: Mapping[str, str],
    fetch_prices: PriceListFetcher,
    fetch_quotes: QuoteFetcher,
) -> str | None:
    msg_text, timeframe = _parse_timeframe(msg_text, change_fields)
    timeframe_error = _unsupported_timeframe_error(
        msg_text,
        timeframe=timeframe,
        change_fields=change_fields,
    )
    if timeframe_error:
        return timeframe_error

    conversion_request = parse_amount_conversion(msg_text)
    if conversion_request:
        return _convert_amount(
            conversion_request,
            fetch_prices=fetch_prices,
        )

    msg_text, convert_to, convert_parameter = parse_conversion_only(msg_text)
    if convert_to not in SUPPORTED_PRICE_SYMBOLS:
        return f"no laburo con {convert_to} gordo"
    prices = fetch_prices(convert_parameter)
    listed = _price_data(prices)
    if listed is None:
        return "no pude traer precios de crypto boludo"

    price_rows, prices_number, selection_error = _select_price_rows(
        msg_text,
        listed=listed,
        convert_parameter=convert_parameter,
        fetch_quotes=fetch_quotes,
    )
    if selection_error:
        return selection_error
    display = PriceDisplay(
        convert_to=convert_to,
        convert_parameter=convert_parameter,
        change_field=change_fields.get(
            timeframe or "24h",
            "percent_change_24h",
        ),
        timeframe_label=timeframe or "24h",
    )
    return _format_price_rows(price_rows[:prices_number], display)


def _unsupported_timeframe_error(
    msg_text: str,
    *,
    timeframe: str | None,
    change_fields: Mapping[str, str],
) -> str | None:
    if timeframe is not None or not msg_text.strip():
        return None
    last_token = msg_text.strip().rsplit(None, 1)[-1].lower()
    if not re.fullmatch(r"\d+[hd]", last_token):
        return None
    valid = ", ".join(change_fields)
    return f"timeframe '{last_token}' no soportado, uso: {valid}"


def _price_data(
    prices: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not prices or not isinstance(prices.get("data"), list):
        return None
    return [item for item in prices["data"] if isinstance(item, dict)]


def _convert_amount(
    request: AmountConversionRequest,
    *,
    fetch_prices: PriceListFetcher,
) -> str:
    if request.target_symbol not in SUPPORTED_PRICE_SYMBOLS:
        return f"no laburo con {request.target_symbol} gordo"

    prices = _price_data(fetch_prices(request.target_parameter))
    if prices is None:
        return "no pude traer precios de crypto boludo"
    requested_asset = find_coin_by_symbol_or_name(prices, request.source_symbol)
    if requested_asset:
        quote_price = requested_asset["quote"][request.target_parameter]["price"]
        if request.target_symbol == "SATS":
            quote_price *= 100000000
        converted_value = request.amount * quote_price
        return (
            f"{fmt_num(request.amount, 8)} {requested_asset['symbol'].upper()} = "
            f"{fmt_num(converted_value, 8)} {request.target_symbol}"
        )

    source_parameter = price_query_parameter(request.source_symbol)
    reverse_prices = _price_data(fetch_prices(source_parameter))
    if reverse_prices is None:
        return "no pude traer precios de crypto boludo"
    target_asset = find_coin_by_symbol_or_name(
        reverse_prices,
        request.target_symbol,
    )
    if not target_asset:
        return "no laburo con esos ponzis boludo"
    source_amount = (
        request.amount / 100000000
        if request.source_symbol == "SATS"
        else request.amount
    )
    asset_price = target_asset["quote"][source_parameter]["price"]
    converted_value = source_amount / asset_price
    return (
        f"{fmt_num(request.amount, 8)} {request.source_symbol} = "
        f"{fmt_num(converted_value, 8)} {target_asset['symbol'].upper()}"
    )


def _requested_price_count(msg_text: str) -> int:
    result = 0
    for token in msg_text.upper().replace(" ", "").split(","):
        try:
            result = max(result, int(float(token)))
        except ValueError:
            continue
    return result


def _select_price_rows(
    msg_text: str,
    *,
    listed: list[dict[str, Any]],
    convert_parameter: str,
    fetch_quotes: QuoteFetcher,
) -> tuple[list[dict[str, Any]], int, str | None]:
    prices_number = _requested_price_count(msg_text)
    if not msg_text.upper().isupper():
        return listed, prices_number or 10, None

    coins = expand_price_tokens(msg_text.split(","))
    selected = _select_listed_coins(listed, coins, prices_number)
    requested = _fallback_quote_tokens(coins)
    if not selected:
        selected = _fetch_requested_quotes(
            requested,
            convert_parameter=convert_parameter,
            fetch_quotes=fetch_quotes,
        )
    if selected:
        return selected, len(selected), None
    if requested:
        return [], 0, f"no encontre esos ponzis: {', '.join(requested)}"
    return [], 0, "no pude traer precios de crypto boludo"


def _format_price_rows(
    rows: list[dict[str, Any]],
    display: PriceDisplay,
) -> str:
    lines = []
    for coin in rows:
        quote = coin["quote"][display.convert_parameter]
        display_price = float(quote["price"])
        if display.convert_to == "SATS":
            display_price *= 100000000

        decimals = f"{display_price:.12f}".split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))
        price = f"{display_price:.{zeros + 4}f}".rstrip("0").rstrip(".")
        percentage = (
            f"{quote.get(display.change_field, 0):+.2f}"
            .rstrip("0")
            .rstrip(".")
        )
        lines.append(
            f"{coin['symbol']}: {price} {display.convert_to} "
            f"({percentage}% {display.timeframe_label})"
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
