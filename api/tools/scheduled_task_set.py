"""scheduled_task_set tool — create a recurring scheduled task."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import schedule_recurring_task


def _describe_interval(seconds: int) -> str:
    if seconds >= 86400:
        days = seconds // 86400
        return f"cada {days} dia{'s' if days != 1 else ''}"
    elif seconds >= 3600:
        hours = seconds // 3600
        return f"cada {hours} hora{'s' if hours != 1 else ''}"
    else:
        minutes = seconds // 60
        return f"cada {minutes} minuto{'s' if minutes != 1 else ''}"


def _execute_scheduled_task_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    prompt = params.get("prompt", "")
    interval_seconds = params.get("interval_seconds")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))

    if not prompt:
        return ToolResult(output="no se que tarea programar, pasame el prompt")
    if interval_seconds is None:
        return ToolResult(
            output="necesito interval_seconds (segundos entre ejecuciones)"
        )
    if not isinstance(interval_seconds, int) or interval_seconds < 300:
        return ToolResult(output="el intervalo minimo es 300 segundos (5 min)")
    if interval_seconds > 86400 * 7:
        return ToolResult(output="el intervalo maximo es 7 dias")
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    task_id = schedule_recurring_task(chat_id, prompt, interval_seconds, user_name)
    if task_id is None:
        return ToolResult(output="no se pudo crear la tarea programada")

    desc = _describe_interval(interval_seconds)
    return ToolResult(
        output=f"listo, tarea programada {desc}: {prompt}",
        metadata={"task_id": task_id, "interval_seconds": interval_seconds},
    )


register_tool(
    name="scheduled_task_set",
    description=(
        "Create a recurring scheduled task. The bot executes the prompt "
        "automatically at the specified interval and sends the result. "
        "Use for things like 'send me news about X daily'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt to execute each time the task fires",
            },
            "interval_seconds": {
                "type": "integer",
                "description": (
                    "Interval in seconds between executions. "
                    "Examples: 300 (5 min), 3600 (1 hour), 86400 (1 day), 604800 (1 week). "
                    "Min: 300 seconds. Max: 604800 seconds (7 days)."
                ),
            },
        },
        "required": ["prompt", "interval_seconds"],
    },
    executor=_execute_scheduled_task_set,
)
