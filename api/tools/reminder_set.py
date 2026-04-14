"""reminder_set tool — create a scheduled reminder."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tools.reminder_scheduler import schedule_reminder, parse_delay


def _execute_reminder_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    text = params.get("text", "")
    delay_str = params.get("delay", "")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))

    if not text:
        return ToolResult(output="no se que recordarte, pasame el texto")
    if not delay_str:
        return ToolResult(
            output="no me dijiste cuando, pasame el tiempo (ej: '30 min', '2 horas')"
        )
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    delay_seconds = parse_delay(str(delay_str))
    if delay_seconds is None:
        return ToolResult(
            output=f"no entendi el tiempo '{delay_str}', proba con '30 min', '2 horas', '1 dia'"
        )

    reminder_id = schedule_reminder(chat_id, text, delay_seconds, user_name)
    if reminder_id is None:
        return ToolResult(output="no se pudo crear el recordatorio")

    minutes = delay_seconds // 60
    if minutes < 60:
        time_desc = f"{minutes} minuto{'s' if minutes != 1 else ''}"
    elif minutes < 1440:
        hours = minutes // 60
        time_desc = f"{hours} hora{'s' if hours != 1 else ''}"
    else:
        days = minutes // 1440
        time_desc = f"{days} dia{'s' if days != 1 else ''}"

    return ToolResult(
        output=f"listo, te acuerdo en {time_desc}: {text}",
        metadata={"reminder_id": reminder_id, "delay_seconds": delay_seconds},
    )


register_tool(
    name="reminder_set",
    description="Set a reminder for the user. The bot will send a message when the time arrives.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "What to remind the user about",
            },
            "delay": {
                "type": "string",
                "description": "When to remind, e.g. '30 min', '2 horas', '1 dia', '5 minutos'",
            },
        },
        "required": ["text", "delay"],
    },
    executor=_execute_reminder_set,
)
