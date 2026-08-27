"""Cancel a scheduled task requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.core.i18n import tr
from api.tools.registry import ToolResult, register_tool
from api.tasks.scheduler import cancel_task, list_tasks


def _execute_task_cancel(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    task_id = params.get("task_id", "")
    if not task_id:
        return ToolResult(output=tr("task.cancel.required"))
    chat_id = str(context.get("chat_id", ""))
    if not chat_id:
        return ToolResult(output=tr("task.no_chat"))
    if not any(str(task.get("id")) == str(task_id) for task in list_tasks(chat_id)):
        return ToolResult(output=tr("task.cancel.not_found"))

    cancel_task(task_id)
    return ToolResult(output=tr("task.cancel.done", task_id=task_id))


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
    requires_context=["chat_id"],
)
