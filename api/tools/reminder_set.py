"""reminder_set tool — create a scheduled reminder."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import schedule_reminder


def _describe_delay(seconds: int) -> str:
    if seconds >= 86400:
        days = seconds // 86400
        return f"{days} dia{'s' if days != 1 else ''}"
    elif seconds >= 3600:
        hours = seconds // 3600
        return f"{hours} hora{'s' if hours != 1 else ''}"
    else:
        minutes = seconds // 60
        return f"{minutes} minuto{'s' if minutes != 1 else ''}"


def _execute_reminder_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    text = params.get("text", "")
    delay_seconds = params.get("delay_seconds")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))

    if not text:
        return ToolResult(output="no se que recordarte, pasame el texto")
    if delay_seconds is None:
        return ToolResult(output="necesito delay_seconds (segundos desde ahora)")
    if not isinstance(delay_seconds, int) or delay_seconds < 1:
        return ToolResult(output="delay_seconds debe ser un entero positivo")
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    if delay_seconds > 86400 * 30:
        return ToolResult(output="el maximo es 30 dias")

    reminder_id = schedule_reminder(chat_id, text, delay_seconds, user_name)
    if reminder_id is None:
        return ToolResult(output="no se pudo crear el recordatorio")

    time_desc = _describe_delay(delay_seconds)
    return ToolResult(
        output=f"listo, te acuerdo en {time_desc}: {text}",
        metadata={"reminder_id": reminder_id, "delay_seconds": delay_seconds},
    )


register_tool(
    name="reminder_set",
    description="Set a one-time reminder. The bot sends a message when the time arrives.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What to remind the user about",
            },
            "delay_seconds": {
                "type": "integer",
                "description": (
                    "Delay in seconds from now. "
                    "Examples: 60 (1 min), 1800 (30 min), 3600 (1 hour), 86400 (1 day). "
                    "Max: 2592000 seconds (30 days)."
                ),
            },
        },
        "required": ["text", "delay_seconds"],
    },
    executor=_execute_reminder_set,
)
