"""task_set tool — create a one-shot or recurring task."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import schedule_task


def _describe_time(seconds: int, prefix: str = "") -> str:
    if seconds >= 86400:
        days = seconds // 86400
        return f"{prefix}{days} dia{'s' if days != 1 else ''}"
    elif seconds >= 3600:
        hours = seconds // 3600
        return f"{prefix}{hours} hora{'s' if hours != 1 else ''}"
    else:
        minutes = seconds // 60
        return f"{prefix}{minutes} minuto{'s' if minutes != 1 else ''}"


def _execute_task_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    text = params.get("text", "")
    delay_seconds = params.get("delay_seconds")
    interval_seconds = params.get("interval_seconds")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))

    if not text:
        return ToolResult(output="no se que tarea crear, pasame el texto")
    if delay_seconds is None and interval_seconds is None:
        return ToolResult(
            output="necesito delay_seconds (una vez) o interval_seconds (repetir)"
        )
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    if delay_seconds is not None:
        if not isinstance(delay_seconds, int) or delay_seconds < 1:
            return ToolResult(output="delay_seconds debe ser un entero positivo")
        if delay_seconds > 86400 * 30:
            return ToolResult(output="el maximo es 30 dias")

    if interval_seconds is not None:
        if not isinstance(interval_seconds, int) or interval_seconds < 300:
            return ToolResult(output="el intervalo minimo es 300 segundos (5 min)")
        if interval_seconds > 86400 * 7:
            return ToolResult(output="el intervalo maximo es 7 dias")

    task_id = schedule_task(
        chat_id,
        text,
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        user_name=user_name,
    )
    if task_id is None:
        return ToolResult(output="no se pudo crear la tarea")

    if interval_seconds:
        desc = _describe_time(interval_seconds, "cada ")
    else:
        desc = _describe_time(delay_seconds, "en ")

    return ToolResult(
        output=f"listo, tarea programada {desc}: {text}",
        metadata={"task_id": task_id},
    )


register_tool(
    name="task_set",
    description="Create a scheduled task. One-shot (delay_seconds) or recurring (interval_seconds).",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What the task should do or remind",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Delay in seconds for one-shot task. 60=1min, 3600=1h, 86400=1d. Max 2592000 (30d).",
            },
            "interval_seconds": {
                "type": "integer",
                "description": "Interval in seconds for recurring task. 300=5min, 3600=1h, 86400=1d, 604800=1w.",
            },
        },
        "required": ["text"],
    },
    executor=_execute_task_set,
)
