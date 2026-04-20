"""stock_prices tool — fetches stock prices from Stooq for agentic use."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

from api.tools.registry import ToolResult, register_tool


def fetch_stooq_price(symbol: str) -> Optional[Tuple[float, float]]:
    try:
        resp = requests.get(
            f"https://stooq.com/q/l/?s={symbol}&i=d", timeout=5
        )
        resp.raise_for_status()
        rows = [line.strip() for line in resp.text.splitlines() if line.strip()]
        if not rows:
            return None
        row = [field.strip() for field in rows[-1].split(",")]
        if len(row) < 7 or row[0].lower() == "symbol":
            return None
        open_price = row[3]
        close_price = row[6]
        if open_price in {"N/D", ""} or close_price in {"N/D", ""}:
            return None
        current = float(close_price)
        opening = float(open_price)
        if opening == 0:
            return None
        variation = ((current - opening) / opening) * 100
        return current, variation
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
        parsed = fetch_stooq_price(sym.upper())
        if parsed:
            price, var = parsed
            sign = "+" if var >= 0 else ""
            results.append(f"{sym.upper()}: ${price:.2f} ({sign}{var:.2f}% dia)")
        else:
            results.append(f"{sym.upper()}: no se pudo obtener")

    return ToolResult(output="\n".join(results))


register_tool(
    name="stock_prices",
    description="Get stock prices from Stooq. Pass ticker symbols like 'aapl tsla googl'. Returns current price and daily variation.",
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
