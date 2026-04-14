"""dollar_lookup tool — wraps get_dollar_rates for agentic use."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_dollar_lookup(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_dollar_rates_fn = context.get("get_dollar_rates")
    if get_dollar_rates_fn is None:
        return ToolResult(output="dollar lookup not available")
    result = get_dollar_rates_fn("")
    if result is None:
        return ToolResult(output="no se pudieron obtener las cotizaciones del dolar")
    return ToolResult(output=result)


register_tool(
    name="dollar_lookup",
    description="Get current USD/ARS exchange rates (dolar blue, dolar oficial, dolar MEP, etc). No parameters needed.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    executor=_execute_dollar_lookup,
)
