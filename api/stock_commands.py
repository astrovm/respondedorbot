from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import redis
from requests.exceptions import RequestException

from api.utils import fmt_num

CachedRequest = Callable[..., dict[str, Any] | None]
StockFetcher = Callable[[str], tuple[float, float] | None]
StockListFetcher = Callable[[], list[str]]
RedisFactory = Callable[[], redis.Redis | None]
RedisJsonGetter = Callable[[redis.Redis, str], Any]
RedisJsonSetter = Callable[..., bool]
HttpGetter = Callable[..., Any]

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FINVIZ_SCREENER_URL = "https://finviz.com/screener.ashx"


def fetch_yahoo_stock_price(
    symbol: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> tuple[float, float] | None:
    try:
        response = cached_request(
            YAHOO_CHART_URL.format(symbol=symbol),
            {"range": "5d", "interval": "1d"},
            {"User-Agent": "Mozilla/5.0"},
            cache_ttl,
        )
        if not response or "data" not in response:
            return None
        result = response["data"].get("chart", {}).get("result", [{}])[0]
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        closes = [close for close in quotes.get("close", []) if close is not None]
        if len(closes) < 2:
            return None
        previous_close, current = closes[-2:]
        if previous_close == 0:
            return None
        return current, ((current - previous_close) / previous_close) * 100
    except Exception:
        return None


def fetch_top_stocks_by_market_cap(
    *,
    redis_factory: RedisFactory,
    redis_get_json: RedisJsonGetter,
    redis_set_json: RedisJsonSetter,
    http_get: HttpGetter,
    cache_ttl: int,
) -> list[str]:
    redis_client = redis_factory()
    cache_key = "market:stock_screener:mega_cap"
    if redis_client:
        cached = redis_get_json(redis_client, cache_key)
        if isinstance(cached, list):
            return [str(symbol) for symbol in cached]

    try:
        response = http_get(
            FINVIZ_SCREENER_URL,
            params={"v": "152", "f": "cap_mega", "o": "-marketcap"},
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            },
            timeout=10,
        )
        response.raise_for_status()
        seen_companies: set[str] = set()
        result: list[str] = []
        pattern = r'data-boxover-ticker="([A-Z.]+)"\s+data-boxover-company="([^"]+)"'
        for match in re.finditer(pattern, response.text):
            symbol, company = match.group(1), match.group(2)
            if company not in seen_companies and len(result) < 10:
                seen_companies.add(company)
                result.append(symbol)
        if redis_client and result:
            redis_set_json(redis_client, cache_key, result, ttl=cache_ttl)
        return result
    except RequestException:
        return []


def get_oil_price(*, fetch_stock: StockFetcher) -> str:
    prices: dict[str, tuple[float, float]] = {}
    for name, symbol in {"Brent": "BZ=F", "WTI": "CL=F"}.items():
        parsed = fetch_stock(symbol)
        if parsed:
            prices[name] = parsed

    if not prices:
        return "no pude traer el precio del petróleo boludo"

    lines: list[str] = []
    for name in ("Brent", "WTI"):
        if name not in prices:
            continue
        price, variation = prices[name]
        sign = "+" if variation >= 0 else ""
        lines.append(
            f"{name}: {fmt_num(price, 2)} USD "
            f"({sign}{fmt_num(variation, 2)}% 24hs)"
        )
    return "\n".join(lines)


def get_stock_prices(
    msg_text: str,
    *,
    fetch_stock: StockFetcher,
    fetch_top_stocks: StockListFetcher,
) -> str:
    symbols = [symbol.strip() for symbol in str(msg_text or "").split() if symbol.strip()]
    if not symbols:
        symbols = fetch_top_stocks()
        if not symbols:
            return "no pude traer el top de acciones, probá de nuevo"

    lines: list[str] = []
    for symbol in symbols:
        normalized = symbol.upper()
        parsed = fetch_stock(normalized)
        if parsed:
            price, variation = parsed
            sign = "+" if variation >= 0 else ""
            lines.append(
                f"{normalized}: {price:.2f} USD ({sign}{variation:.2f}% 24h)"
            )
        else:
            lines.append(f"{normalized}: no se pudo obtener")
    return "\n".join(lines) if lines else "no se pudo obtener ninguna cotización"
