"""Typed task triggers shared by tool parsing and scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

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


type TaskTrigger = (
    DelayTrigger | IntervalTrigger | DayIntervalTrigger | CronTrigger
)


@dataclass(frozen=True, slots=True)
class ScheduledTaskRequest:
    chat_id: str
    text: str
    trigger: TaskTrigger
    user_name: str = ""
    user_id: int | None = None
    timezone_offset: int = -3


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
            return (), f"day_of_week invalido: {token}"
        weekdays.append(normalized)
    return (
        (tuple(weekdays), None)
        if weekdays
        else ((), "day_of_week invalido")
    )


def _parse_cron(config: Mapping[str, Any]) -> TriggerParseResult:
    try:
        hour = _required_bounded_int(
            config,
            BoundedIntField(
                key="hour",
                minimum=0,
                maximum=23,
                missing_error="hour es requerido para trigger cron",
                range_error="hour debe ser 0-23",
            ),
        )
        minute = _required_bounded_int(
            config,
            BoundedIntField(
                key="minute",
                minimum=0,
                maximum=59,
                missing_error="minute es requerido para trigger cron",
                range_error="minute debe ser 0-59",
            ),
        )
        weekdays, error = _parse_weekdays(config.get("day_of_week"))
        if error:
            raise TriggerValidationError(error)
        raw_day = config.get("day")
        if raw_day is not None and (
            not isinstance(raw_day, int) or not 1 <= raw_day <= 31
        ):
            raise TriggerValidationError("day debe ser 1-31")
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
        return TriggerParseResult(
            error="delay_seconds debe ser un entero positivo"
        )
    if value > 86400 * 3650:
        return TriggerParseResult(error="el maximo es 10 años")
    return TriggerParseResult(
        trigger=DelayTrigger(kind="delay", seconds=value)
    )


def _parse_interval_seconds(value: Any) -> TriggerParseResult:
    if not isinstance(value, int) or value < 300:
        return TriggerParseResult(
            error="el intervalo minimo es 300 segundos (5 min)"
        )
    if value > 86400 * 7:
        return TriggerParseResult(error="el intervalo maximo es 7 dias")
    return TriggerParseResult(
        trigger=IntervalTrigger(kind="interval_seconds", seconds=value)
    )


def _parse_interval_days(config: Mapping[str, Any]) -> TriggerParseResult:
    days = config.get("days")
    if days is None:
        return TriggerParseResult(error="days es requerido para trigger interval")
    if not isinstance(days, int) or days < 1:
        return TriggerParseResult(error="days debe ser un entero positivo")
    if days > 90:
        return TriggerParseResult(error="el maximo son 90 dias")
    return TriggerParseResult(
        trigger=DayIntervalTrigger(kind="interval_days", days=days)
    )


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
        return TriggerParseResult(
            error=(
                "necesito usar algun parametro de tiempo: delay_seconds (una vez), "
                "interval_seconds (repetir), o trigger_config."
            )
        )
    trigger_type = trigger_config.get("type")
    if trigger_type == "cron":
        return _parse_cron(trigger_config)
    if trigger_type == "interval":
        return _parse_interval_days(trigger_config)
    return TriggerParseResult(
        error="trigger_config.type debe ser 'interval' o 'cron'"
    )


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
