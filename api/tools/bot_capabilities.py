"""Return the bot capability catalog when requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool


def _execute_bot_capabilities(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    del params
    get_capabilities = context.get("get_bot_capabilities")
    if get_capabilities is None:
        return ToolResult(output="bot capability catalog not available")
    return ToolResult(output=str(get_capabilities()))


register_tool(
    name="bot_capabilities",
    description=(
        "Get the authoritative list of bot features and commands. Use when a "
        "user asks what the bot can do or which command to use."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    executor=_execute_bot_capabilities,
    requires_context=["get_bot_capabilities"],
)
