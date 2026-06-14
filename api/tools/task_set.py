"""task_set tool — create a one-shot or recurring task."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tasks.scheduler import (
    describe_task_trigger,
    format_interval,
    get_scheduler_runtime_status,
    schedule_task,
)
from api.tasks.models import (
    DelayTrigger,
    IntervalTrigger,
    ScheduledTaskRequest,
    TaskTrigger,
    parse_task_trigger,
)
from api.services import credits_db


def _task_set_precondition_error(
    *,
    text: Any,
    chat_id: str,
    user_id: Any,
) -> str | None:
    if not text:
        return "no se que tarea crear, pasame el texto"
    if not chat_id:
        return "no se en que chat estoy"

    runtime_status = get_scheduler_runtime_status()
    if not runtime_status.get("ready"):
        reason = runtime_status.get("reason", "runtime unavailable")
        return f"no se pudo crear la tarea: {reason}"

    if user_id:
        try:
            if (
                credits_db.is_configured()
                and credits_db.get_balance("user", int(user_id)) <= 0
            ):
                return "no tenes creditos, recargá primero"
        except Exception:
            pass
    return None


def _trigger_description(trigger: TaskTrigger) -> str:
    if isinstance(trigger, DelayTrigger):
        return format_interval(trigger.seconds, "en ")
    if isinstance(trigger, IntervalTrigger):
        return format_interval(trigger.seconds)
    return describe_task_trigger(trigger)


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

    precondition_error = _task_set_precondition_error(
        text=text,
        chat_id=chat_id,
        user_id=user_id,
    )
    if precondition_error:
        return ToolResult(output=precondition_error)

    parsed = parse_task_trigger(
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        trigger_config=trigger_config,
    )
    if parsed.error or parsed.trigger is None:
        return ToolResult(output=parsed.error or "trigger invalido")

    task_id = schedule_task(
        ScheduledTaskRequest(
            chat_id=chat_id,
            text=str(text),
            trigger=parsed.trigger,
            user_name=user_name,
            user_id=int(user_id) if user_id is not None else None,
            timezone_offset=timezone_offset,
        )
    )
    if task_id is None:
        return ToolResult(output="no se pudo crear la tarea")

    return ToolResult(
        output=(
            f"listo, tarea programada "
            f"{_trigger_description(parsed.trigger)}: {text}"
        ),
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
                "description": "Content-only future instruction the bot will execute later. Preserve perspective, subject, and pronouns, but exclude scheduling/time expressions that belong in delay_seconds, interval_seconds, or trigger_config.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Delay in seconds for one-shot task. 60=1min, 3600=1h, 86400=1d. Max 315360000 (10y).",
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
    task_allowed=False,
)
