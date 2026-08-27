from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from api.i18n import tr
from api.markets.price_commands import (
    AmountConversionRequest,
    SUPPORTED_PRICE_SYMBOLS,
    expand_price_tokens,
    find_coin_by_symbol_or_name,
    parse_amount_conversion,
    parse_conversion_only,
    price_query_parameter,
)
from api.markets.stocks import StockQuote
from api.utils import fmt_num

PriceListFetcher = Callable[[str], dict[str, Any] | None]
QuoteFetcher = Callable[..., dict[str, Any] | None]
StockLookup = Callable[[str], list[tuple[str, StockQuote | None]] | None]


@dataclass(frozen=True, slots=True)
class PriceDisplay:
    convert_to: str
    convert_parameter: str
    change_field: str
    timeframe_label: str


@dataclass(frozen=True, slots=True)
class PriceSelection:
    rows: list[dict[str, Any]]
    count: int
    unresolved: list[str]


@dataclass(frozen=True, slots=True)
class MarketPriceResult:
    crypto_rows: list[dict[str, Any]]
    stock_quotes: list[StockQuote]
    unresolved: list[str]


def get_prices(
    msg_text: str,
    *,
    change_fields: Mapping[str, str],
    fetch_prices: PriceListFetcher,
    fetch_quotes: QuoteFetcher,
    lookup_stocks: StockLookup | None = None,
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

    conversion_requested = _has_conversion_modifier(msg_text)
    msg_text, convert_to, convert_parameter = parse_conversion_only(msg_text)
    if convert_to not in SUPPORTED_PRICE_SYMBOLS:
        return tr("market.crypto.unsupported_currency", symbol=convert_to)
    return _get_asset_prices(
        msg_text,
        convert_to=convert_to,
        convert_parameter=convert_parameter,
        timeframe=timeframe,
        conversion_requested=conversion_requested,
        change_fields=change_fields,
        fetch_prices=fetch_prices,
        fetch_quotes=fetch_quotes,
        lookup_stocks=lookup_stocks,
    )


def _get_asset_prices(
    msg_text: str,
    *,
    convert_to: str,
    convert_parameter: str,
    timeframe: str | None,
    conversion_requested: bool,
    change_fields: Mapping[str, str],
    fetch_prices: PriceListFetcher,
    fetch_quotes: QuoteFetcher,
    lookup_stocks: StockLookup | None,
) -> str:
    provider_scope, msg_text = _parse_provider_scope(msg_text)
    if provider_scope == "stock":
        return (
            _stock_modifier_error(msg_text)
            if _stock_modifiers_unsupported(timeframe, conversion_requested)
            else _format_stock_only(msg_text, lookup_stocks)
        )

    prices = fetch_prices(convert_parameter)
    listed = _price_data(prices)
    if listed is None:
        if provider_scope != "crypto":
            stock_only = _format_stock_only(msg_text, lookup_stocks, missing_error=False)
            if stock_only:
                return stock_only
        return tr("market.crypto.load_error")

    selection = _select_price_rows(
        msg_text,
        listed=listed,
        convert_parameter=convert_parameter,
        fetch_quotes=fetch_quotes,
    )
    stock_quotes: list[StockQuote] = []
    unresolved = selection.unresolved
    if provider_scope != "crypto" and lookup_stocks and unresolved:
        stock_quotes, unresolved = _lookup_stock_fallback(
            msg_text,
            selection=selection,
            lookup_stocks=lookup_stocks,
        )
    if stock_quotes and _stock_modifiers_unsupported(timeframe, conversion_requested):
        return _format_modifier_rejection(
            selection,
            stock_quotes=stock_quotes,
            convert_to=convert_to,
            convert_parameter=convert_parameter,
            timeframe=timeframe,
            change_fields=change_fields,
        )
    result = MarketPriceResult(selection.rows, stock_quotes, unresolved)
    if result.unresolved and not result.crypto_rows and not result.stock_quotes:
        return tr("market.crypto.missing", symbols=", ".join(result.unresolved))

    display = PriceDisplay(
        convert_to=convert_to,
        convert_parameter=convert_parameter,
        change_field=change_fields.get(
            timeframe or "24h",
            "percent_change_24h",
        ),
        timeframe_label=timeframe or "24h",
    )
    return _format_market_result(result, selection=selection, display=display)


def _format_market_result(
    result: MarketPriceResult,
    *,
    selection: PriceSelection,
    display: PriceDisplay,
) -> str:
    formatted_parts = [
        part
        for part in (
            _format_price_rows(result.crypto_rows[: selection.count], display),
            _format_stock_quotes(result.stock_quotes),
        )
        if part
    ]
    if result.unresolved:
        formatted_parts.append(tr("market.crypto.missing", symbols=", ".join(result.unresolved)))
    return "\n".join(formatted_parts)


def _parse_provider_scope(msg_text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*(crypto|stock)\s*:\s*(.*?)\s*$", msg_text, re.IGNORECASE)
    if not match:
        return None, msg_text
    return match.group(1).lower(), match.group(2)


def _has_conversion_modifier(msg_text: str) -> bool:
    return (
        re.search(r"(?:^|\s)(?:in|to|a|en)\s+\$?[a-zA-Z0-9]+\s*$", msg_text, re.IGNORECASE)
        is not None
    )


def _stock_modifiers_unsupported(timeframe: str | None, conversion_requested: bool) -> bool:
    return conversion_requested or timeframe not in (None, "24h")


def _stock_modifier_error(symbols: str) -> str:
    return tr("market.stock.modifiers_unsupported", symbols=symbols.upper())


def _format_modifier_rejection(
    selection: PriceSelection,
    *,
    stock_quotes: list[StockQuote],
    convert_to: str,
    convert_parameter: str,
    timeframe: str | None,
    change_fields: Mapping[str, str],
) -> str:
    error = _stock_modifier_error(", ".join(quote.symbol for quote in stock_quotes))
    if not selection.rows:
        return error
    display = PriceDisplay(
        convert_to=convert_to,
        convert_parameter=convert_parameter,
        change_field=change_fields.get(timeframe or "24h", "percent_change_24h"),
        timeframe_label=timeframe or "24h",
    )
    crypto = _format_price_rows(selection.rows[: selection.count], display)
    return f"{crypto}\n{error}"


def _format_stock_only(
    query: str,
    lookup_stocks: StockLookup | None,
    *,
    missing_error: bool = True,
) -> str:
    if not lookup_stocks or not query.strip():
        return tr("market.crypto.missing", symbols=query.upper()) if missing_error else ""
    resolved = lookup_stocks(query) or []
    quotes = [quote for _, quote in resolved if quote]
    missing = [item.upper() for item, quote in resolved if quote is None]
    parts = [_format_stock_quotes(quotes)] if quotes else []
    if missing_error and (missing or not quotes):
        missing_symbols = missing or [query.upper()]
        parts.append(tr("market.crypto.missing", symbols=", ".join(missing_symbols)))
    return "\n".join(parts)


def _lookup_stock_fallback(
    raw_query: str,
    *,
    selection: PriceSelection,
    lookup_stocks: StockLookup,
) -> tuple[list[StockQuote], list[str]]:
    stock_query = _stock_fallback_query(raw_query, selection)
    resolved = lookup_stocks(stock_query) or []
    quotes = [quote for _, quote in resolved if quote]
    if quotes:
        missing = [query.upper() for query, quote in resolved if quote is None]
        return quotes, missing
    return [], selection.unresolved


def _stock_fallback_query(raw_query: str, selection: PriceSelection) -> str:
    if not selection.rows and "," not in raw_query and len(selection.unresolved) > 1:
        return raw_query
    if "," not in raw_query:
        return ",".join(selection.unresolved)

    unresolved = set(selection.unresolved)
    stock_segments: list[str] = []
    for segment in (part.strip() for part in raw_query.split(",")):
        tokens = expand_price_tokens([token for token in segment.split() if token])
        if any(token in unresolved for token in tokens):
            stock_segments.append(segment)
    return ",".join(stock_segments) or ",".join(selection.unresolved)


def _format_stock_quotes(quotes: list[StockQuote]) -> str:
    lines: list[str] = []
    for quote in quotes:
        sign = "+" if quote.variation >= 0 else ""
        lines.append(
            f"{quote.symbol}: {quote.price:.2f} {quote.currency} ({sign}{quote.variation:.2f}% 24h)"
        )
    return "\n".join(lines)


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
    return tr("market.timeframe_invalid", timeframe=last_token, valid=valid)


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
        return tr("market.crypto.unsupported_currency", symbol=request.target_symbol)

    prices = _price_data(fetch_prices(request.target_parameter))
    if prices is None:
        return tr("market.crypto.load_error")
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
        return tr("market.crypto.load_error")
    target_asset = find_coin_by_symbol_or_name(
        reverse_prices,
        request.target_symbol,
    )
    if not target_asset:
        return tr("market.crypto.unsupported_pair")
    source_amount = (
        request.amount / 100000000 if request.source_symbol == "SATS" else request.amount
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
) -> PriceSelection:
    prices_number = _requested_price_count(msg_text)
    if not msg_text.upper().isupper():
        return PriceSelection(listed, prices_number or 10, [])

    raw_tokens = [token for token in re.split(r"[,\s]+", msg_text) if token]
    coins = expand_price_tokens(raw_tokens)
    explicit_requested = _fallback_quote_tokens(coins[: len(raw_tokens)])
    selected = _select_listed_coins(listed, coins, prices_number)
    requested = _fallback_quote_tokens(coins)
    matched_tokens = _matched_price_tokens(selected)
    missing = [token for token in requested if token not in matched_tokens]
    if missing:
        selected.extend(
            _fetch_requested_quotes(
                missing,
                convert_parameter=convert_parameter,
                fetch_quotes=fetch_quotes,
            )
        )
        selected = _unique_price_rows(selected)
    unresolved = [
        token for token in explicit_requested if token not in _matched_price_tokens(selected)
    ]
    if selected:
        return PriceSelection(selected, len(selected), unresolved)
    if explicit_requested:
        return PriceSelection([], 0, explicit_requested)
    return PriceSelection([], 0, [])


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
        percentage = f"{quote.get(display.change_field, 0):+.2f}".rstrip("0").rstrip(".")
        lines.append(
            f"{coin['symbol']}: {price} {display.convert_to} "
            f"({percentage}% {display.timeframe_label})"
        )
    return "\n".join(lines)


def _parse_timeframe(msg_text: str, valid: Mapping[str, str]) -> tuple[str, str | None]:
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
    return list(
        dict.fromkeys(
            token
            for token in tokens
            if token not in {"STABLES", "STABLECOINS"} and not token.isdigit()
        )
    )


def _iter_quote_rows(quote_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in (quote_data or {}).values():
        candidates = value if isinstance(value, list) else [value]
        rows.extend(candidate for candidate in candidates if isinstance(candidate, dict))
    return rows


def _has_price(coin: dict[str, Any], convert_parameter: str) -> bool:
    return coin.get("quote", {}).get(convert_parameter, {}).get("price") is not None


def _matched_price_tokens(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(coin.get(field) or "").upper().replace(" ", "")
        for coin in rows
        for field in ("symbol", "name", "slug")
    }


def _unique_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for coin in rows:
        identity = str(coin.get("id") or coin.get("symbol") or coin.get("slug") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(coin)
    return unique


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
        found.extend(coin for coin in slug_rows if _has_price(coin, convert_parameter))

    return _unique_price_rows(found)
