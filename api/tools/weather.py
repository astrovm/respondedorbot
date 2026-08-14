"""Fetch current Buenos Aires weather requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.ai.prompt_context import format_weather_info
from api.tools.registry import ToolResult, register_tool


def _execute_weather(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    del params
    get_weather = context.get("get_weather_context")
    if get_weather is None:
        return ToolResult(output="weather lookup not available")

    weather = get_weather()
    if not weather:
        return ToolResult(output="no se pudo obtener el clima de Buenos Aires")
    return ToolResult(output=format_weather_info(weather))


register_tool(
    name="weather",
    description="Get the current weather in Buenos Aires, Argentina.",
    parameters={"type": "object", "properties": {}, "required": []},
    executor=_execute_weather,
    requires_context=["get_weather_context"],
)
