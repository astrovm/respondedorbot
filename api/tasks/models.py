"""Typed task triggers shared by tool parsing and scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from api.core.i18n import tr

_SPANISH_TO_ENGLISH_WEEKDAY = {
    "lun": "mon",
    "mar": "tue",
    "mie": "wed",
    "jue": "thu",
    "vie": "fri",
    "sab": "sat",
    "dom": "sun",
}
_ENGLISH_WEEKDAYS = frozenset(_SPANISH_TO_ENGLISH_WEEKDAY.values())


@dataclass(frozen=True, slots=True)
class DelayTrigger:
    kind: Literal["delay"]
    seconds: int


@dataclass(frozen=True, slots=True)
class IntervalTrigger:
    kind: Literal["interval_seconds"]
    seconds: int


@dataclass(frozen=True, slots=True)
class DayIntervalTrigger:
    kind: Literal["interval_days"]
    days: int


@dataclass(frozen=True, slots=True)
class CronTrigger:
    kind: Literal["cron"]
    hour: int
    minute: int
    weekdays: tuple[str, ...] = ()
    day: int | None = None


type TaskTrigger = DelayTrigger | IntervalTrigger | DayIntervalTrigger | CronTrigger


@dataclass(frozen=True, slots=True)
class ScheduledTaskRequest:
    chat_id: str
    text: str
    trigger: TaskTrigger
    user_name: str = ""
    user_id: int | None = None
    timezone_offset: int = -3
    locale: str = "es"


@dataclass(frozen=True, slots=True)
class TriggerParseResult:
    trigger: TaskTrigger | None = None
    error: str | None = None


class TriggerValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BoundedIntField:
    key: str
    minimum: int
    maximum: int
    missing_error: str
    range_error: str


def _required_bounded_int(
    config: Mapping[str, Any],
    field: BoundedIntField,
) -> int:
    value = config.get(field.key)
    if value is None:
        raise TriggerValidationError(field.missing_error)
    if not isinstance(value, int) or not field.minimum <= value <= field.maximum:
        raise TriggerValidationError(field.range_error)
    return value


def _parse_weekdays(value: Any) -> tuple[tuple[str, ...], str | None]:
    if value in (None, ""):
        return (), None
    weekdays: list[str] = []
    for part in str(value).split(","):
        token = part.strip().lower()
        if not token:
            continue
        normalized = _SPANISH_TO_ENGLISH_WEEKDAY.get(token, token)
        if normalized not in _ENGLISH_WEEKDAYS:
            return (), tr("task.trigger.weekday", value=token)
        weekdays.append(normalized)
    return (tuple(weekdays), None) if weekdays else ((), tr("task.trigger.weekday_empty"))


def _parse_cron(config: Mapping[str, Any]) -> TriggerParseResult:
    try:
        hour = _required_bounded_int(
            config,
            BoundedIntField(
                key="hour",
                minimum=0,
                maximum=23,
                missing_error=tr("task.trigger.hour_required"),
                range_error=tr("task.trigger.hour_range"),
            ),
        )
        minute = _required_bounded_int(
            config,
            BoundedIntField(
                key="minute",
                minimum=0,
                maximum=59,
                missing_error=tr("task.trigger.minute_required"),
                range_error=tr("task.trigger.minute_range"),
            ),
        )
        weekdays, error = _parse_weekdays(config.get("day_of_week"))
        if error:
            raise TriggerValidationError(error)
        raw_day = config.get("day")
        if raw_day is not None and (not isinstance(raw_day, int) or not 1 <= raw_day <= 31):
            raise TriggerValidationError(tr("task.trigger.day_range"))
        return TriggerParseResult(
            trigger=CronTrigger(
                kind="cron",
                hour=hour,
                minute=minute,
                weekdays=weekdays,
                day=raw_day,
            )
        )
    except TriggerValidationError as error:
        return TriggerParseResult(error=str(error))


def _parse_delay(value: Any) -> TriggerParseResult:
    if not isinstance(value, int) or value < 1:
        return TriggerParseResult(error=tr("task.trigger.delay_positive"))
    if value > 86400 * 3650:
        return TriggerParseResult(error=tr("task.trigger.delay_max"))
    return TriggerParseResult(trigger=DelayTrigger(kind="delay", seconds=value))


def _parse_interval_seconds(value: Any) -> TriggerParseResult:
    if not isinstance(value, int) or value < 300:
        return TriggerParseResult(error=tr("task.trigger.interval_min"))
    if value > 86400 * 7:
        return TriggerParseResult(error=tr("task.trigger.interval_max"))
    return TriggerParseResult(trigger=IntervalTrigger(kind="interval_seconds", seconds=value))


def _parse_interval_days(config: Mapping[str, Any]) -> TriggerParseResult:
    days = config.get("days")
    if days is None:
        return TriggerParseResult(error=tr("task.trigger.days_required"))
    if not isinstance(days, int) or days < 1:
        return TriggerParseResult(error=tr("task.trigger.days_positive"))
    if days > 90:
        return TriggerParseResult(error=tr("task.trigger.days_max"))
    return TriggerParseResult(trigger=DayIntervalTrigger(kind="interval_days", days=days))


def parse_task_trigger(
    *,
    delay_seconds: Any = None,
    interval_seconds: Any = None,
    trigger_config: Any = None,
) -> TriggerParseResult:
    if delay_seconds is not None:
        return _parse_delay(delay_seconds)
    if interval_seconds is not None:
        return _parse_interval_seconds(interval_seconds)
    if not isinstance(trigger_config, Mapping):
        return TriggerParseResult(error=tr("task.trigger.required"))
    trigger_type = trigger_config.get("type")
    if trigger_type == "cron":
        return _parse_cron(trigger_config)
    if trigger_type == "interval":
        return _parse_interval_days(trigger_config)
    return TriggerParseResult(error=tr("task.trigger.type"))


def trigger_config(trigger: TaskTrigger) -> dict[str, Any] | None:
    if isinstance(trigger, CronTrigger):
        config: dict[str, Any] = {
            "type": "cron",
            "hour": trigger.hour,
            "minute": trigger.minute,
        }
        if trigger.weekdays:
            config["day_of_week"] = ",".join(trigger.weekdays)
        if trigger.day is not None:
            config["day"] = trigger.day
        return config
    if isinstance(trigger, DayIntervalTrigger):
        return {"type": "interval", "days": trigger.days}
    return None
