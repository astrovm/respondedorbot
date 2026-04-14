"""price_lookup tool — wraps get_prices for agentic use."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_price_lookup(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_prices_fn = context.get("get_prices")
    if get_prices_fn is None:
        return ToolResult(output="price lookup not available")
    symbols = params.get("symbols", "")
    if isinstance(symbols, list):
        symbols = " ".join(symbols)
    result = get_prices_fn(str(symbols))
    if result is None:
        return ToolResult(output="no se pudieron obtener los precios")
    return ToolResult(output=result)


register_tool(
    name="price_lookup",
    description="Get cryptocurrency and stock prices. Pass symbols like 'btc eth' or 'bitcoin ethereum'. Returns current prices with 24h changes.",
    parameters={
        "type": "object",
        "properties": {
            "symbols": {
                "type": "string",
                "description": "Space-separated list of crypto tickers or names, e.g. 'btc eth sol'",
            },
        },
        "required": ["symbols"],
    },
    executor=_execute_price_lookup,
)
