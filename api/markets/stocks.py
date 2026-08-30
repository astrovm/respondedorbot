from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import redis
from requests.exceptions import RequestException

from api.cache.service import CacheService
from api.core.config_runtime import ConfigRuntime
from api.core.logging import get_logger
from api.core.rust_bridge import load_rust_bridge
from api.i18n import tr
from api.services import http_client
from api.services.redis_helpers import redis_get_json, redis_set_json
from api.utils import fmt_num

CachedRequest = Callable[..., dict[str, Any] | None]
StockFetcher = Callable[[str], tuple[float, float] | None]
StockQuoteFetcher = Callable[[str], "StockQuote | None"]
StockSymbolResolver = Callable[[str], str | None]
StockListFetcher = Callable[[], list[str]]
RedisFactory = Callable[[], redis.Redis | None]
RedisJsonGetter = Callable[[redis.Redis, str], Any]
RedisJsonSetter = Callable[..., bool]
HttpGetter = Callable[..., Any]

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
FINVIZ_SCREENER_URL = "https://finviz.com/screener.ashx"
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StockQuote:
    symbol: str
    name: str
    price: float
    currency: str
    exchange: str
    variation: float


@dataclass(frozen=True, slots=True)
class _StockQuery:
    original: str
    normalized: str
    is_symbol: bool


@dataclass(frozen=True, slots=True)
class _StockQueryPlan:
    raw_query: str
    queries: list[_StockQuery]
    full_query_fallback: bool
    needs_top_stocks: bool


class _RustStockAdapter(Protocol):
    def stock_parse_yahoo_quote(
        self, response_json: str, fallback_symbol: str
    ) -> str: ...

    def stock_select_yahoo_symbol(self, response_json: str) -> str | None: ...

    def stock_query_plan(self, message: str) -> str: ...

    def finviz_fetch(self) -> str: ...


def _load_rust_stock_adapter() -> _RustStockAdapter | None:
    module = load_rust_bridge("RUST_STOCK_MARKET_ENABLED")
    if module is None:
        return None
    return cast(_RustStockAdapter, module)


def _stock_quote_from_mapping(value: Any) -> StockQuote | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Rust stock quote must be an object")
    text_fields = ("symbol", "name", "currency", "exchange")
    if not all(isinstance(value.get(field), str) for field in text_fields):
        raise ValueError("Rust stock quote text fields are invalid")
    price = value.get("price")
    variation = value.get("variation")
    if not isinstance(price, (int, float)) or isinstance(price, bool):
        raise ValueError("Rust stock price must be numeric")
    if not isinstance(variation, (int, float)) or isinstance(variation, bool):
        raise ValueError("Rust stock variation must be numeric")
    return StockQuote(
        symbol=cast(str, value["symbol"]),
        name=cast(str, value["name"]),
        price=float(price),
        currency=cast(str, value["currency"]),
        exchange=cast(str, value["exchange"]),
        variation=float(variation),
    )


def _python_stock_query_plan(message: str) -> _StockQueryPlan:
    raw_query = str(message or "").strip()
    full_query_fallback = False
    if "," in raw_query:
        originals = [part.strip() for part in raw_query.split(",") if part.strip()]
    else:
        originals = [part for part in raw_query.split() if part]
        full_query_fallback = len(originals) > 1
    queries = []
    for original in originals[:20]:
        normalized = original.upper().lstrip("$")
        queries.append(
            _StockQuery(
                original=original,
                normalized=normalized,
                is_symbol=re.fullmatch(r"[A-Z0-9.\^=\-]{1,30}", normalized)
                is not None,
            )
        )
    return _StockQueryPlan(
        raw_query=raw_query,
        queries=queries,
        full_query_fallback=full_query_fallback,
        needs_top_stocks=not queries,
    )


def _rust_stock_query_plan(raw_plan: str) -> _StockQueryPlan:
    plan = json.loads(raw_plan)
    if not isinstance(plan, Mapping):
        raise ValueError("Rust stock query plan must be an object")
    raw_query = plan.get("raw_query")
    raw_queries = plan.get("queries")
    full_query_fallback = plan.get("full_query_fallback")
    needs_top_stocks = plan.get("needs_top_stocks")
    if not isinstance(raw_query, str) or not isinstance(raw_queries, list):
        raise ValueError("Rust stock query plan fields are invalid")
    if not isinstance(full_query_fallback, bool) or not isinstance(
        needs_top_stocks, bool
    ):
        raise ValueError("Rust stock query plan flags are invalid")
    queries: list[_StockQuery] = []
    for raw_query_item in raw_queries:
        if not isinstance(raw_query_item, Mapping):
            raise ValueError("Rust stock query must be an object")
        original = raw_query_item.get("original")
        normalized = raw_query_item.get("normalized")
        is_symbol = raw_query_item.get("is_symbol")
        if not isinstance(original, str) or not isinstance(normalized, str):
            raise ValueError("Rust stock query text is invalid")
        if not isinstance(is_symbol, bool):
            raise ValueError("Rust stock query classification is invalid")
        queries.append(_StockQuery(original, normalized, is_symbol))
    return _StockQueryPlan(
        raw_query,
        queries,
        full_query_fallback,
        needs_top_stocks,
    )


def _stock_query_plan(message: str) -> _StockQueryPlan:
    rust = _load_rust_stock_adapter()
    if rust is not None:
        try:
            return _rust_stock_query_plan(rust.stock_query_plan(message))
        except Exception:
            logger.exception("Rust stock query planning failed; using Python fallback")
    return _python_stock_query_plan(message)


def _parse_finviz_symbols(html: str) -> list[str]:
    seen_companies: set[str] = set()
    result: list[str] = []
    pattern = r'data-boxover-ticker="([A-Z.]+)"\s+data-boxover-company="([^"]+)"'
    for match in re.finditer(pattern, html):
        symbol, company = match.group(1), match.group(2)
        if company not in seen_companies and len(result) < 10:
            seen_companies.add(company)
            result.append(symbol)
    return result


def _rust_finviz_symbols(rust: _RustStockAdapter) -> list[str]:
    outcome = json.loads(rust.finviz_fetch())
    if not isinstance(outcome, Mapping):
        raise ValueError("Rust Finviz outcome must be an object")
    status = outcome.get("status")
    if status == "success":
        symbols = outcome.get("symbols")
        if not isinstance(symbols, list) or not all(
            isinstance(symbol, str) for symbol in symbols
        ):
            raise ValueError("Rust Finviz symbols must be text")
        return symbols
    if status in {"http_error", "transport_error"}:
        return []
    raise ValueError(f"invalid Rust Finviz status: {status!r}")


def fetch_yahoo_stock_quote(
    symbol: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> StockQuote | None:
    try:
        response = cached_request(
            YAHOO_CHART_URL.format(symbol=symbol),
            {"range": "5d", "interval": "1d"},
            {"User-Agent": "Mozilla/5.0"},
            cache_ttl,
        )
        if not response or "data" not in response:
            return None
        rust = _load_rust_stock_adapter()
        if rust is not None:
            try:
                return _stock_quote_from_mapping(
                    json.loads(
                        rust.stock_parse_yahoo_quote(
                            json.dumps(
                                response,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            symbol,
                        )
                    )
                )
            except Exception:
                logger.exception("Rust Yahoo quote parser failed; using Python fallback")
        results = response["data"].get("chart", {}).get("result") or []
        if not results:
            return None
        result = results[0]
        meta = result.get("meta", {})
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        closes = [close for close in quotes.get("close", []) if close is not None]
        current = meta.get("regularMarketPrice")
        previous_close = meta.get("chartPreviousClose")
        if current is None and closes:
            current = closes[-1]
        if previous_close is None and len(closes) >= 2:
            previous_close = closes[-2]
        if current is None or previous_close in (None, 0):
            return None
        variation = ((float(current) - float(previous_close)) / float(previous_close)) * 100
        return StockQuote(
            symbol=str(meta.get("symbol") or symbol).upper(),
            name=str(meta.get("shortName") or meta.get("longName") or ""),
            price=float(current),
            currency=str(meta.get("currency") or "USD").upper(),
            exchange=str(meta.get("exchangeName") or ""),
            variation=variation,
        )
    except Exception:
        return None


def fetch_yahoo_stock_price(
    symbol: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> tuple[float, float] | None:
    quote = fetch_yahoo_stock_quote(
        symbol,
        cached_request=cached_request,
        cache_ttl=cache_ttl,
    )
    return (quote.price, quote.variation) if quote else None


def search_yahoo_symbol(
    query: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> str | None:
    try:
        normalized_query = query.strip().lstrip("$")
        search_queries = list(dict.fromkeys((normalized_query, normalized_query.replace(" ", ""))))
        for search_query in search_queries:
            response = cached_request(
                YAHOO_SEARCH_URL,
                {"q": search_query, "quotesCount": 5, "newsCount": 0},
                {"User-Agent": "Mozilla/5.0"},
                cache_ttl,
            )
            rust = _load_rust_stock_adapter()
            if rust is not None and response is not None:
                try:
                    resolved = rust.stock_select_yahoo_symbol(
                        json.dumps(
                            response,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    if resolved is not None and not isinstance(resolved, str):
                        raise ValueError("Rust Yahoo symbol must be text")
                    if resolved:
                        return resolved
                    continue
                except Exception:
                    logger.exception(
                        "Rust Yahoo symbol parser failed; using Python fallback"
                    )
            quotes = response.get("data", {}).get("quotes", []) if response else []
            for quote in quotes:
                if quote.get("quoteType") in {
                    "EQUITY",
                    "ETF",
                    "MUTUALFUND",
                    "INDEX",
                    "FUTURE",
                } and quote.get("symbol"):
                    return str(quote["symbol"])
    except Exception:
        return None
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

    rust = _load_rust_stock_adapter()
    if rust is not None:
        try:
            symbols = _rust_finviz_symbols(rust)
            if redis_client and symbols:
                redis_set_json(redis_client, cache_key, symbols, ttl=cache_ttl)
            return symbols
        except Exception:
            logger.exception("Rust Finviz adapter failed; using Python fallback")

    try:
        response = http_get(
            FINVIZ_SCREENER_URL,
            params={"v": "152", "f": "cap_mega", "o": "-marketcap"},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=10,
        )
        response.raise_for_status()
        result = _parse_finviz_symbols(response.text)
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
        return tr("market.stock.oil_error")

    lines: list[str] = []
    for name in ("Brent", "WTI"):
        if name not in prices:
            continue
        price, variation = prices[name]
        sign = "+" if variation >= 0 else ""
        lines.append(f"{name}: {fmt_num(price, 2)} USD ({sign}{fmt_num(variation, 2)}% 24hs)")
    return "\n".join(lines)


def get_stock_prices(
    msg_text: str,
    *,
    fetch_quote: StockQuoteFetcher,
    resolve_symbol: StockSymbolResolver,
    fetch_top_stocks: StockListFetcher,
) -> str:
    quotes = lookup_stock_quotes(
        msg_text,
        fetch_quote=fetch_quote,
        resolve_symbol=resolve_symbol,
        fetch_top_stocks=fetch_top_stocks,
    )
    if quotes is None:
        return tr("market.stock.top_error")

    lines: list[str] = []
    for query, quote in quotes:
        if quote:
            sign = "+" if quote.variation >= 0 else ""
            lines.append(
                f"{quote.symbol}: {quote.price:.2f} {quote.currency} "
                f"({sign}{quote.variation:.2f}% 24h)"
            )
        else:
            lines.append(tr("market.stock.not_found", query=query))
    return "\n".join(lines) if lines else tr("market.stock.none")


def lookup_stock_quotes(
    msg_text: str,
    *,
    fetch_quote: StockQuoteFetcher,
    resolve_symbol: StockSymbolResolver,
    fetch_top_stocks: StockListFetcher,
) -> list[tuple[str, StockQuote | None]] | None:
    """Resolve stock-like queries without applying user-facing formatting."""

    plan = _stock_query_plan(str(msg_text or ""))
    queries = plan.queries
    if plan.needs_top_stocks:
        top_stocks = fetch_top_stocks()
        if not top_stocks:
            return None
        queries = [
            _StockQuery(
                original=symbol,
                normalized=symbol.upper().lstrip("$"),
                is_symbol=True,
            )
            for symbol in top_stocks[:20]
        ]

    return _lookup_stock_quotes(
        plan.raw_query,
        queries,
        full_query_fallback=plan.full_query_fallback,
        fetch_quote=fetch_quote,
        resolve_symbol=resolve_symbol,
    )


def _lookup_stock_quotes(
    raw_query: str,
    queries: list[_StockQuery],
    *,
    full_query_fallback: bool,
    fetch_quote: StockQuoteFetcher,
    resolve_symbol: StockSymbolResolver,
) -> list[tuple[str, StockQuote | None]]:
    quotes: list[tuple[str, StockQuote | None]] = []
    for query in queries:
        quote = fetch_quote(query.normalized) if query.is_symbol else None
        quotes.append((query.original, quote))

    direct_quotes = [quote for _, quote in quotes if quote]
    if not full_query_fallback or len(direct_quotes) == len(quotes):
        return _resolve_missing_stock_quotes(
            quotes,
            fetch_quote=fetch_quote,
            resolve_symbol=resolve_symbol,
        )

    resolved = resolve_symbol(raw_query)
    direct_by_symbol = {quote.symbol.upper(): quote for quote in direct_quotes}
    full_quote = direct_by_symbol.get(str(resolved or "").upper())
    if full_quote is None and resolved:
        full_quote = fetch_quote(resolved)
    if full_quote and (not direct_quotes or full_quote.symbol.upper() not in direct_by_symbol):
        return [(raw_query, full_quote)]
    if not direct_quotes:
        return [(raw_query, None)]
    return _resolve_missing_stock_quotes(
        quotes,
        fetch_quote=fetch_quote,
        resolve_symbol=resolve_symbol,
    )


def _resolve_missing_stock_quotes(
    quotes: list[tuple[str, StockQuote | None]],
    *,
    fetch_quote: StockQuoteFetcher,
    resolve_symbol: StockSymbolResolver,
) -> list[tuple[str, StockQuote | None]]:
    resolved_quotes: list[tuple[str, StockQuote | None]] = []
    for query, quote in quotes:
        resolved_quote = quote
        if resolved_quote is None:
            resolved = resolve_symbol(query)
            resolved_quote = fetch_quote(resolved) if resolved else None
        resolved_quotes.append((query, resolved_quote))
    return resolved_quotes


class StockService:
    def __init__(
        self,
        *,
        cache: CacheService,
        config: ConfigRuntime,
        price_cache_ttl: int,
        screener_cache_ttl: int,
    ) -> None:
        self._cache = cache
        self._config = config
        self._price_cache_ttl = price_cache_ttl
        self._screener_cache_ttl = screener_cache_ttl

    def fetch_price(self, symbol: str) -> tuple[float, float] | None:
        return fetch_yahoo_stock_price(
            symbol,
            cached_request=self._cache.request,
            cache_ttl=self._price_cache_ttl,
        )

    def fetch_quote(self, symbol: str) -> StockQuote | None:
        return fetch_yahoo_stock_quote(
            symbol,
            cached_request=self._cache.request,
            cache_ttl=self._price_cache_ttl,
        )

    def resolve_symbol(self, query: str) -> str | None:
        return search_yahoo_symbol(
            query,
            cached_request=self._cache.request,
            cache_ttl=self._price_cache_ttl,
        )

    def fetch_top_stocks(self) -> list[str]:
        return fetch_top_stocks_by_market_cap(
            redis_factory=self._config.optional_redis,
            redis_get_json=redis_get_json,
            redis_set_json=redis_set_json,
            http_get=http_client.get,
            cache_ttl=self._screener_cache_ttl,
        )

    def get_oil_price(self) -> str:
        return get_oil_price(fetch_stock=self.fetch_price)

    def get_stock_prices(self, msg_text: str) -> str:
        return get_stock_prices(
            msg_text,
            fetch_quote=self.fetch_quote,
            resolve_symbol=self.resolve_symbol,
            fetch_top_stocks=self.fetch_top_stocks,
        )

    def lookup_quotes(self, msg_text: str) -> list[tuple[str, StockQuote | None]] | None:
        return lookup_stock_quotes(
            msg_text,
            fetch_quote=self.fetch_quote,
            resolve_symbol=self.resolve_symbol,
            fetch_top_stocks=self.fetch_top_stocks,
        )


__all__ = ["StockQuote", "StockService", "lookup_stock_quotes"]
