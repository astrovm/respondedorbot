"""task_list tool — list all tasks for a chat."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import list_tasks, format_interval


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
        owner = f"@{t['user_name']}" if t.get("user_name") else ""
        owner_bit = f" ({owner})" if owner else ""
        next_run = t.get("next_run", "")
        if interval:
            freq = format_interval(interval)
            lines.append(
                f"[{t['id']}] {t['text']}{owner_bit} - {freq}, prox: {next_run}"
            )
        else:
            lines.append(f"[{t['id']}] {t['text']}{owner_bit} - {next_run}")

    return ToolResult(output="\n".join(lines))


register_tool(
    name="task_list",
    description="List all tasks (one-shot and recurring) for the current chat.",
    parameters={"type": "object", "properties": {}, "required": []},
    executor=_execute_task_list,
)
