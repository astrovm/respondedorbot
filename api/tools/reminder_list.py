"""reminder_list tool — list pending reminders for a chat."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import list_reminders


def _execute_reminder_list(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    chat_id = str(context.get("chat_id", ""))
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    reminders = list_reminders(chat_id)
    if not reminders:
        return ToolResult(output="no hay recordatorios pendientes")

    lines = []
    for r in reminders:
        lines.append(f"- {r['text']} ({r['next_run']})")
    return ToolResult(output="\n".join(lines))


register_tool(
    name="reminder_list",
    description="List all pending reminders for the current chat.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    executor=_execute_reminder_list,
)
