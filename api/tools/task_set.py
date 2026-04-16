"""task_set tool — create a one-shot or recurring task."""

from __future__ import annotations

from typing import Any, Dict, Optional

from api.tools.registry import ToolResult, register_tool
from api.tools.task_scheduler import describe_trigger, format_interval, schedule_task
from api.services import credits_db

_SPANISH_TO_ENGLISH_WEEKDAY = {
    "lun": "mon",
    "mar": "tue",
    "mie": "wed",
    "jue": "thu",
    "vie": "fri",
    "sab": "sat",
    "dom": "sun",
}
_ENGLISH_WEEKDAYS = set(_SPANISH_TO_ENGLISH_WEEKDAY.values())


def _normalize_day_of_week(raw_value: Any) -> tuple[Optional[str], Optional[str]]:
    if raw_value in (None, ""):
        return None, None

    tokens = []
    for part in str(raw_value).split(","):
        token = part.strip().lower()
        if not token:
            continue
        normalized = _SPANISH_TO_ENGLISH_WEEKDAY.get(token, token)
        if normalized not in _ENGLISH_WEEKDAYS:
            return None, f"day_of_week invalido: {token}"
        tokens.append(normalized)

    if not tokens:
        return None, "day_of_week invalido"

    return ",".join(tokens), None


def _validate_cron_trigger(trigger_config: Dict[str, Any]) -> Optional[str]:
    hour = trigger_config.get("hour")
    if hour is None:
        return "hour es requerido para trigger cron"
    if not isinstance(hour, int) or hour < 0 or hour > 23:
        return "hour debe ser 0-23"

    minute = trigger_config.get("minute")
    if minute is None:
        return "minute es requerido para trigger cron"
    if not isinstance(minute, int) or minute < 0 or minute > 59:
        return "minute debe ser 0-59"

    normalized_day_of_week, day_of_week_error = _normalize_day_of_week(
        trigger_config.get("day_of_week")
    )
    if day_of_week_error:
        return day_of_week_error
    if normalized_day_of_week:
        trigger_config["day_of_week"] = normalized_day_of_week

    return None


def _execute_task_set(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    text = params.get("text", "")
    delay_seconds = params.get("delay_seconds")
    interval_seconds = params.get("interval_seconds")
    trigger_config = params.get("trigger_config")
    chat_id = str(context.get("chat_id", ""))
    user_name = str(context.get("user_name", ""))
    user_id = context.get("user_id")
    timezone_offset = int(context.get("timezone_offset", -3))

    if not text:
        return ToolResult(output="no se que tarea crear, pasame el texto")
    if delay_seconds is None and interval_seconds is None and not trigger_config:
        return ToolResult(
            output="necesito usar algun parametro de tiempo: delay_seconds (una vez), interval_seconds (repetir), o trigger_config."
        )
    if not chat_id:
        return ToolResult(output="no se en que chat estoy")

    if user_id:
        try:
            if credits_db.is_configured():
                balance = credits_db.get_balance("user", int(user_id))
                if balance <= 0:
                    return ToolResult(output="no tenes creditos, recargá primero")
        except Exception:
            pass

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

    if trigger_config:
        trigger_type = trigger_config.get("type")
        if trigger_type not in ("interval", "cron"):
            return ToolResult(output="trigger_config.type debe ser 'interval' o 'cron'")

        if trigger_type == "interval":
            days = trigger_config.get("days")
            if days is None:
                return ToolResult(output="days es requerido para trigger interval")
            if not isinstance(days, int) or days < 1:
                return ToolResult(output="days debe ser un entero positivo")
            if days > 90:
                return ToolResult(output="el maximo son 90 dias")

        if trigger_type == "cron":
            cron_error = _validate_cron_trigger(trigger_config)
            if cron_error:
                return ToolResult(output=cron_error)

    task_id = schedule_task(
        chat_id,
        text,
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        user_name=user_name,
        user_id=user_id,
        trigger_config=trigger_config,
        timezone_offset=timezone_offset,
    )
    if task_id is None:
        return ToolResult(output="no se pudo crear la tarea")

    if trigger_config:
        desc = describe_trigger(trigger_config)
    elif interval_seconds:
        desc = format_interval(interval_seconds)
    else:
        desc = format_interval(delay_seconds, "en ")

    return ToolResult(
        output=f"listo, tarea programada {desc}: {text}",
        metadata={"task_id": task_id},
    )


register_tool(
    name="task_set",
    description="Create a scheduled task. Supports delay_seconds, interval_seconds, or trigger_config (interval/cron).",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Literal future instruction the bot will execute later. Preserve original perspective, subject, and pronouns.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Delay in seconds for one-shot task. 60=1min, 3600=1h, 86400=1d. Max 2592000 (30d).",
            },
            "interval_seconds": {
                "type": "integer",
                "description": "Interval in seconds for recurring task. 300=5min, 3600=1h, 86400=1d, 604800=1w.",
            },
            "trigger_config": {
                "type": "object",
                "description": "Advanced trigger config with type=interval/cron. interval: {type:'interval', days:N}. cron: {type:'cron', hour:0-23, minute:0-59, day_of_week:'mon,wed' or 'lun,mie', day:1-31}",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["interval", "cron"],
                    },
                    "days": {
                        "type": "integer",
                        "description": "For interval type: number of days between runs",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "For cron type: hour (0-23)",
                    },
                    "minute": {
                        "type": "integer",
                        "description": "For cron type: minute (0-59)",
                    },
                    "day_of_week": {
                        "type": "string",
                        "description": "For cron type: days of week in English or Spanish abbreviations (mon,wed,fri or lun,mie,vie)",
                    },
                    "day": {
                        "type": "integer",
                        "description": "For cron type: day of month (1-31)",
                    },
                },
            },
        },
        "required": ["text"],
    },
    executor=_execute_task_set,
)
