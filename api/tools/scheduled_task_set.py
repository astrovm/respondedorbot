"""scheduled_task_set tool — create a recurring scheduled task."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import schedule_recurring_task, parse_interval


def _execute_scheduled_task_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    prompt = params.get("prompt", "")
    interval_str = params.get("interval", "")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))

    if not prompt:
        return ToolResult(output="no se que tarea programar, pasame el prompt")
    if not interval_str:
        return ToolResult(
            output="no me dijiste la frecuencia (ej: 'diario', 'cada 6 horas', 'semanal')"
        )
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    interval_seconds = parse_interval(str(interval_str))
    if interval_seconds is None:
        return ToolResult(
            output=f"no entendi la frecuencia '{interval_str}', proba con 'diario', 'cada 2 horas', 'semanal'"
        )
    if interval_seconds < 300:
        return ToolResult(output="el intervalo minimo es 5 minutos, no seas ansioso")

    task_id = schedule_recurring_task(chat_id, prompt, interval_seconds, user_name)
    if task_id is None:
        return ToolResult(output="no se pudo crear la tarea programada")

    if interval_seconds >= 86400:
        days = interval_seconds // 86400
        desc = f"cada {days} dia{'s' if days != 1 else ''}"
    elif interval_seconds >= 3600:
        hours = interval_seconds // 3600
        desc = f"cada {hours} hora{'s' if hours != 1 else ''}"
    else:
        minutes = interval_seconds // 60
        desc = f"cada {minutes} minuto{'s' if minutes != 1 else ''}"

    return ToolResult(
        output=f"listo, tarea programada {desc}: {prompt}",
        metadata={"task_id": task_id, "interval_seconds": interval_seconds},
    )


register_tool(
    name="scheduled_task_set",
    description=(
        "Create a recurring scheduled task. The bot will execute the prompt "
        "automatically at the specified interval and send the result. "
        "Use for things like 'send me news about X daily' or 'check price of Y every 6 hours'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt to execute each time the task fires",
            },
            "interval": {
                "type": "string",
                "description": (
                    "How often to run, e.g. 'diario', 'cada 6 horas', "
                    "'semanal', 'every 30 min'"
                ),
            },
        },
        "required": ["prompt", "interval"],
    },
    executor=_execute_scheduled_task_set,
)
