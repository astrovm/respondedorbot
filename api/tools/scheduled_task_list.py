"""scheduled_task_list tool — list recurring scheduled tasks for a chat."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import list_scheduled_tasks


def _execute_scheduled_task_list(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    chat_id = str(context.get("chat_id", ""))
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    tasks = list_scheduled_tasks(chat_id)
    if not tasks:
        return ToolResult(output="no hay tareas programadas")

    lines = []
    for t in tasks:
        interval = t.get("interval_seconds", 0)
        if interval >= 86400:
            freq = f"cada {interval // 86400}d"
        elif interval >= 3600:
            freq = f"cada {interval // 3600}h"
        else:
            freq = f"cada {interval // 60}m"
        lines.append(f"- [{t['id']}] {t['prompt']} ({freq})")

    return ToolResult(output="\n".join(lines))


register_tool(
    name="scheduled_task_list",
    description="List all recurring scheduled tasks for the current chat.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    executor=_execute_scheduled_task_list,
)
