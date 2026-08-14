"""Fetch market prices through the shared stock service."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_stock_prices(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_stock_prices = context.get("get_stock_prices")
    if get_stock_prices is None:
        return ToolResult(output="stock price lookup not available")
    queries = params.get("queries", [])
    if isinstance(queries, str):
        queries = [queries]
    if not isinstance(queries, list) or not queries:
        return ToolResult(output="indicá al menos un símbolo o empresa")
    normalized = [str(query).strip() for query in queries if str(query).strip()]
    if not normalized:
        return ToolResult(output="indicá al menos un símbolo o empresa")
    return ToolResult(output=get_stock_prices(",".join(normalized[:20])))


register_tool(
    name="stock_prices",
    description=(
        "Get prices for stocks, ETFs, indexes, or futures from Yahoo Finance. "
        "Use exact Yahoo symbols when known; company names are also accepted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
                "description": "Yahoo symbols or company names.",
            },
        },
        "required": ["queries"],
    },
    executor=_execute_stock_prices,
    requires_context=["get_stock_prices"],
)
