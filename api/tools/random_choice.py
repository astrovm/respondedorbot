"""Use the shared /random command implementation from an AI tool call."""

from __future__ import annotations

from typing import Any, Dict

from api.core.i18n import tr
from api.tools.registry import ToolResult, register_tool


def _execute_random_choice(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    select_random = context.get("select_random")
    if select_random is None:
        return ToolResult(output=tr("tool.unavailable", tool="random choice"))
    request = str(params.get("request") or "").strip()
    if not request:
        return ToolResult(output=tr("tool.random.required"))
    return ToolResult(output=select_random(request))


register_tool(
    name="random_choice",
    description=(
        "Choose one item at random or generate a random integer. "
        "This has the same behavior as the /random command."
    ),
    parameters={
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "Comma-separated options, such as 'option-alpha, option-beta', "
                    "or an inclusive integer range, such as '1-10'."
                ),
            },
        },
        "required": ["request"],
    },
    executor=_execute_random_choice,
    requires_context=["select_random"],
)
