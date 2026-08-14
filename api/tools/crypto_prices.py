"""Fetch cryptocurrency prices requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_crypto_prices(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_prices_fn = context.get("get_prices")
    if get_prices_fn is None:
        return ToolResult(output="crypto price lookup not available")
    assets = params.get("assets", [])
    if isinstance(assets, str):
        assets = [assets]
    if not isinstance(assets, list) or not assets:
        return ToolResult(output="indicá al menos una crypto")
    normalized = [str(asset).strip() for asset in assets if str(asset).strip()]
    if not normalized:
        return ToolResult(output="indicá al menos una crypto")

    query = ",".join(normalized[:20])
    convert = str(params.get("convert") or "USD").strip().upper()
    timeframe = str(params.get("timeframe") or "24h").strip().lower()
    query = f"{query} in {convert} {timeframe}"
    result = get_prices_fn(query)
    if result is None:
        return ToolResult(output="no se pudieron obtener los precios")
    return ToolResult(output=result)


register_tool(
    name="crypto_prices",
    description=(
        "Get cryptocurrency prices from CoinMarketCap by symbol or slug, "
        "with a quote currency and change timeframe."
    ),
    parameters={
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 20,
                "description": "CoinMarketCap symbols or slugs, such as BTC or bitcoin-cash.",
            },
            "convert": {
                "type": "string",
                "description": "Quote currency symbol, such as USD, EUR, ARS, or BTC.",
                "default": "USD",
            },
            "timeframe": {
                "type": "string",
                "enum": ["1h", "24h", "7d", "30d"],
                "default": "24h",
            },
        },
        "required": ["assets"],
    },
    executor=_execute_crypto_prices,
    requires_env=["COINMARKETCAP_KEY"],
    requires_context=["get_prices"],
)
