"""task_list tool — list all tasks for a chat."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import list_tasks


def _execute_task_list(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    chat_id = str(context.get("chat_id", ""))
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    tasks = list_tasks(chat_id)
    if not tasks:
        return ToolResult(output="no hay tareas")

    lines = []
    for t in tasks:
        interval = t.get("interval_seconds")
        if interval:
            if interval >= 86400:
                freq = f"cada {interval // 86400}d"
            elif interval >= 3600:
                freq = f"cada {interval // 3600}h"
            else:
                freq = f"cada {interval // 60}m"
            lines.append(f"- [{t['id']}] {t['text']} ({freq})")
        else:
            lines.append(f"- [{t['id']}] {t['text']} ({t['next_run']})")

    return ToolResult(output="\n".join(lines))


register_tool(
    name="task_list",
    description="List all tasks (one-shot and recurring) for the current chat.",
    parameters={"type": "object", "properties": {}, "required": []},
    executor=_execute_task_list,
)
