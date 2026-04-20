"""stock_prices tool — fetches stock prices from Yahoo Finance for agentic use."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

from api.tools.registry import ToolResult, register_tool

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _fetch_yahoo_price(symbol: str) -> Optional[Tuple[float, float]]:
    try:
        resp = requests.get(
            _YAHOO_CHART_URL.format(symbol=symbol),
            params={"range": "5d", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("chart", {}).get("result", [{}])[0]
        quotes = result.get("indicators", {}).get("quote", [{}])[0]
        closes = [c for c in quotes.get("close", []) if c is not None]
        if len(closes) < 2:
            return None
        prev_close = closes[-2]
        current = closes[-1]
        if prev_close == 0:
            return None
        change_pct = ((current - prev_close) / prev_close) * 100
        return current, change_pct
    except Exception:
        return None


def _execute_stock_prices(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    symbols_str = params.get("symbols", "")
    if isinstance(symbols_str, list):
        symbols_str = " ".join(symbols_str)
    symbols = [s.strip() for s in str(symbols_str).split() if s.strip()]
    if not symbols:
        return ToolResult(output="pasame los simbolos, ejemplo: aapl tsla googl")

    results: List[str] = []
    for sym in symbols:
        parsed = _fetch_yahoo_price(sym.upper())
        if parsed:
            price, var = parsed
            sign = "+" if var >= 0 else ""
            results.append(f"{sym.upper()}: ${price:.2f} ({sign}{var:.2f}% dia)")
        else:
            results.append(f"{sym.upper()}: no se pudo obtener")

    return ToolResult(output="\n".join(results))


register_tool(
    name="stock_prices",
    description="Get stock prices from Yahoo Finance. Pass ticker symbols like 'aapl tsla googl'. Returns current price and daily variation.",
    parameters={
        "type": "object",
        "properties": {
            "symbols": {
                "type": "string",
                "description": "Space-separated list of stock tickers, e.g. 'aapl tsla googl spy'",
            },
        },
        "required": ["symbols"],
    },
    executor=_execute_stock_prices,
)
