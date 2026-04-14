"""scheduled_task_cancel tool — cancel a recurring scheduled task."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import cancel_scheduled_task


def _execute_scheduled_task_cancel(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    task_id = params.get("task_id", "")
    if not task_id:
        return ToolResult(output="necesito el id de la tarea, usá /tareas para verlas")

    success = cancel_scheduled_task(task_id)
    if success:
        return ToolResult(output=f"tarea {task_id} cancelada")
    return ToolResult(output=f"no se pudo cancelar la tarea {task_id}")


register_tool(
    name="scheduled_task_cancel",
    description="Cancel a recurring scheduled task by its ID. Use scheduled_task_list to get the task ID first.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task to cancel (from /tareas output, format: xxxxxxxx)",
            },
        },
        "required": ["task_id"],
    },
    executor=_execute_scheduled_task_cancel,
)
