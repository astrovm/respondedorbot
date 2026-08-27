"""Fetch Argentine dollar rates requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.core.i18n import tr
from api.tools.registry import ToolResult, register_tool


def _execute_dollar_rates(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_rates = context.get("get_dollar_rates")
    if get_rates is None:
        return ToolResult(output=tr("tool.unavailable", tool="dollar rates"))

    result = get_rates(str(params.get("timeframe") or ""))
    if not result:
        return ToolResult(output=tr("tool.dollar.error"))
    return ToolResult(output=str(result))


register_tool(
    name="dollar_rates",
    description=(
        "Get current Argentine dollar exchange rates and their change over an "
        "optional historical timeframe."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["1h", "6h", "12h", "24h", "48h"],
                "description": "Optional comparison timeframe. Defaults to 24h.",
            },
        },
        "required": [],
    },
    executor=_execute_dollar_rates,
    requires_context=["get_dollar_rates"],
)
