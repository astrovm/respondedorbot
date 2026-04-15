"""task_cancel tool — cancel a task by id."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import cancel_task


def _execute_task_cancel(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    task_id = params.get("task_id", "")
    if not task_id:
        return ToolResult(output="necesito el id de la tarea, usa /tareas para verlas")

    cancel_task(task_id)
    return ToolResult(output=f"tarea {task_id} cancelada")


register_tool(
    name="task_cancel",
    description="Cancel a task by its ID. Use task_list or /tareas to get the ID.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel",
            },
        },
        "required": ["task_id"],
    },
    executor=_execute_task_cancel,
)
