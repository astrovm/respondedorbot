"""task_list tool — list all tasks for a chat."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import (
    format_task_summary,
    list_tasks,
)


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
        lines.append(format_task_summary(t))

    return ToolResult(output="\n".join(lines))


register_tool(
    name="task_list",
    description="List all tasks (one-shot and recurring) for the current chat.",
    parameters={"type": "object", "properties": {}, "required": []},
    executor=_execute_task_list,
)
