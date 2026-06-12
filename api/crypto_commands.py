from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from api.price_commands import (
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
        new_prices = []
        coins = expand_price_tokens(msg_text.split(","))
        if not prices or "data" not in prices:
            return "no pude traer precios de crypto boludo"

        for coin in prices["data"]:
            symbol = coin["symbol"].upper().replace(" ", "")
            name = coin["name"].upper().replace(" ", "")
            if symbol in coins or name in coins:
                new_prices.append(coin)
            elif prices_number > 0 and coin in prices["data"][:prices_number]:
                new_prices.append(coin)

        if not new_prices:
            found: list[dict[str, Any]] = []
            not_found: list[str] = []
            for coin_token in coins:
                token = coin_token.upper().replace(" ", "")
                quote_data = fetch_quotes([token], convert_parameter)
                if quote_data:
                    if _append_first_priced_quote(
                        found, quote_data, convert_parameter
                    ):
                        continue
                    quote_by_slug = fetch_quotes(
                        [token.lower()], convert_parameter, by_slug=True
                    )
                    if quote_by_slug and _append_first_priced_quote(
                        found, quote_by_slug, convert_parameter
                    ):
                        continue
                not_found.append(token)

            if not found and not_found:
                return f"no encontre esos ponzis: {', '.join(not_found)}"
            if not found:
                return "no pude traer precios de crypto boludo"
            new_prices = found

        prices_number = len(new_prices)
        prices["data"] = new_prices

    if prices_number < 1:
        prices_number = 10
    if not prices or "data" not in prices:
        return "no pude traer precios de crypto boludo"

    lines = []
    for coin in prices["data"][:prices_number]:
        quote = coin["quote"][convert_parameter]
        if convert_to == "SATS":
            quote["price"] *= 100000000

        decimals = f"{quote['price']:.12f}".split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))
        price = f"{quote['price']:.{zeros + 4}f}".rstrip("0").rstrip(".")
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


def _append_first_priced_quote(
    found: list[dict[str, Any]],
    quote_data: dict[str, Any],
    convert_parameter: str,
) -> bool:
    for coin_data in quote_data.values():
        price = (
            coin_data.get("quote", {})
            .get(convert_parameter, {})
            .get("price")
        )
        if price:
            found.append(coin_data)
            return True
    return False
