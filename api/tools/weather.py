"""Fetch current weather for a location requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.ai.prompt_context import format_weather_info
from api.core.i18n import tr
from api.tools.registry import ToolResult, register_tool


def _execute_weather(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    get_weather = context.get("get_weather_context")
    if get_weather is None:
        return ToolResult(output=tr("tool.unavailable", tool="weather"))

    location = str(params.get("location") or "").strip()
    if not location:
        return ToolResult(output=tr("tool.weather.required"))
    weather = get_weather(location)
    if not weather:
        return ToolResult(output=tr("weather.load_error", location=location))
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
