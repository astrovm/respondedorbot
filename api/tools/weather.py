"""Fetch current weather for a location requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.ai.prompt_context import format_weather_info
from api.tools.registry import ToolResult, register_tool


def _execute_weather(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_weather = context.get("get_weather_context")
    if get_weather is None:
        return ToolResult(output="weather lookup not available")

    location = str(params.get("location") or "").strip()
    if not location:
        return ToolResult(output="indicá una ciudad o ubicación")
    weather = get_weather(location)
    if not weather:
        return ToolResult(output=f"no se pudo obtener el clima de {location}")
    return ToolResult(output=format_weather_info(weather))


register_tool(
    name="weather",
    description="Get the current weather for any city or location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or location, including region or country when ambiguous.",
            }
        },
        "required": ["location"],
    },
    executor=_execute_weather,
    requires_context=["get_weather_context"],
)
