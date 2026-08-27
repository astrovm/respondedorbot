"""Create one-shot and recurring tasks requested by the AI."""

from __future__ import annotations

from typing import Any, Dict

from api.tools.registry import ToolResult, register_tool
from api.tasks.scheduler import (
    describe_task_trigger,
    estimate_task_reserve_credits,
    format_interval,
    get_scheduler_runtime_status,
    schedule_task,
)
from api.tasks.credits import task_credit_precondition_error
from api.tasks.models import (
    DelayTrigger,
    IntervalTrigger,
    ScheduledTaskRequest,
    TaskTrigger,
    parse_task_trigger,
)
from api.services import credits_db
from api.core.i18n import current_locale, tr


def _task_set_precondition_error(
    *,
    text: Any,
    chat_id: str,
) -> str | None:
    if not text:
        return tr("task.no_text")
    if not chat_id:
        return tr("task.no_chat")

    runtime_status = get_scheduler_runtime_status()
    if not runtime_status.get("ready"):
        reason = runtime_status.get("reason", "runtime unavailable")
        return tr("task.create_runtime", reason=reason)

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
    locale = str(context.get("locale") or current_locale())

    precondition_error = _task_set_precondition_error(
        text=text,
        chat_id=chat_id,
    )
    if precondition_error:
        return ToolResult(output=precondition_error)

    parsed = parse_task_trigger(
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        trigger_config=trigger_config,
    )
    if parsed.error or parsed.trigger is None:
        return ToolResult(output=parsed.error or tr("task.invalid_trigger"))

    required_credits = estimate_task_reserve_credits(str(text))
    if required_credits is None:
        return ToolResult(output=tr("task.cost_error"))
    credit_error = task_credit_precondition_error(
        credits_db_service=credits_db,
        user_id=user_id,
        required_credit_units=required_credits,
    )
    if credit_error:
        return ToolResult(output=credit_error)

    task_id = schedule_task(
        ScheduledTaskRequest(
            chat_id=chat_id,
            text=str(text),
            trigger=parsed.trigger,
            user_name=user_name,
            user_id=int(user_id) if user_id is not None else None,
            timezone_offset=timezone_offset,
            locale=locale,
        )
    )
    if task_id is None:
        return ToolResult(output=tr("task.create_error"))

    return ToolResult(
        output=tr(
            "task.created",
            schedule=_trigger_description(parsed.trigger),
            text=text,
        ),
        metadata={"task_id": task_id},
    )


register_tool(
    name="task_set",
    description=(
        "Create a scheduled task. Put only the future instruction in text and "
        "preserve its subject, perspective, and pronouns. Put time or frequency "
        "only in delay_seconds, interval_seconds, or trigger_config. For example, "
        "'tomorrow remind me to pay' uses text='remind me to pay' plus a delay; "
        "'every day at 20:30 tell me the score' uses text='tell me the score' "
        "plus a cron trigger. Choose a reasonable hour if the user omits one."
    ),
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
