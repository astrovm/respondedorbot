"""Typed task triggers shared by tool parsing and scheduling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, cast

from api.core.rust_bridge import load_rust_bridge
from api.i18n import tr

logger = logging.getLogger(__name__)

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


class _RustTaskTriggerParser(Protocol):
    def parse_task_trigger(self, input_json: str) -> str: ...


def _load_rust_task_trigger_parser() -> Optional[_RustTaskTriggerParser]:
    module = load_rust_bridge("RUST_TASK_TRIGGERS_ENABLED")
    if module is None:
        return None
    return cast(_RustTaskTriggerParser, module)


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


def _parse_task_trigger_python(
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


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_RUST_TRIGGER_ERROR_CODES = frozenset(
    {
        "required",
        "type",
        "delay_positive",
        "delay_max",
        "interval_min",
        "interval_max",
        "days_required",
        "days_positive",
        "days_max",
        "hour_required",
        "hour_range",
        "minute_required",
        "minute_range",
        "weekday",
        "weekday_empty",
        "day_range",
    }
)


def _rust_integer_input(value: Any) -> dict[str, Any]:
    if value is None:
        return {"state": "missing"}
    if not isinstance(value, int):
        return {"state": "invalid"}
    if value < _I64_MIN:
        return {"state": "below_range"}
    if value > _I64_MAX:
        return {"state": "above_range"}
    return {"state": "value", "value": int(value)}


def _rust_trigger_input(
    delay_seconds: Any,
    interval_seconds: Any,
    trigger_config: Any,
) -> str:
    if not isinstance(trigger_config, Mapping):
        config: dict[str, Any] = {"kind": "missing"}
    elif trigger_config.get("type") == "cron":
        raw_weekdays = trigger_config.get("day_of_week")
        config = {
            "kind": "cron",
            "hour": _rust_integer_input(trigger_config.get("hour")),
            "minute": _rust_integer_input(trigger_config.get("minute")),
            "weekdays": None if raw_weekdays is None else str(raw_weekdays),
            "day": _rust_integer_input(trigger_config.get("day")),
        }
    elif trigger_config.get("type") == "interval":
        config = {
            "kind": "interval_days",
            "days": _rust_integer_input(trigger_config.get("days")),
        }
    else:
        config = {"kind": "unsupported"}

    return json.dumps(
        {
            "delay_seconds": _rust_integer_input(delay_seconds),
            "interval_seconds": _rust_integer_input(interval_seconds),
            "config": config,
        },
        separators=(",", ":"),
    )


def _trigger_from_rust(value: Any) -> TaskTrigger:
    if not isinstance(value, Mapping):
        raise ValueError("Rust task trigger result is not a mapping")
    kind = value.get("kind")
    if kind == "delay":
        return DelayTrigger(kind="delay", seconds=int(value["seconds"]))
    if kind == "interval_seconds":
        return IntervalTrigger(kind="interval_seconds", seconds=int(value["seconds"]))
    if kind == "interval_days":
        return DayIntervalTrigger(kind="interval_days", days=int(value["days"]))
    if kind == "cron":
        raw_weekdays = value.get("weekdays")
        if not isinstance(raw_weekdays, list):
            raise ValueError("Rust cron weekdays are not a list")
        raw_day = value.get("day")
        return CronTrigger(
            kind="cron",
            hour=int(value["hour"]),
            minute=int(value["minute"]),
            weekdays=tuple(str(day) for day in raw_weekdays),
            day=None if raw_day is None else int(raw_day),
        )
    raise ValueError("Rust task trigger result has an unknown kind")


def _error_from_rust(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("Rust task trigger error is not a mapping")
    code = value.get("code")
    if not isinstance(code, str) or code not in _RUST_TRIGGER_ERROR_CODES:
        raise ValueError("Rust task trigger result has an unknown error code")
    if code == "weekday":
        raw_value = value.get("value")
        if not isinstance(raw_value, str):
            raise ValueError("Rust weekday error has no value")
        return tr("task.trigger.weekday", value=raw_value)
    return tr(f"task.trigger.{code}")


def _parse_task_trigger_rust(
    rust: _RustTaskTriggerParser,
    *,
    delay_seconds: Any,
    interval_seconds: Any,
    trigger_config: Any,
) -> TriggerParseResult:
    raw_result = rust.parse_task_trigger(
        _rust_trigger_input(delay_seconds, interval_seconds, trigger_config)
    )
    result = json.loads(raw_result)
    if not isinstance(result, Mapping):
        raise ValueError("Rust task trigger response is not a mapping")
    trigger = result.get("trigger")
    error = result.get("error")
    if trigger is not None and error is None:
        return TriggerParseResult(trigger=_trigger_from_rust(trigger))
    if error is not None and trigger is None:
        return TriggerParseResult(error=_error_from_rust(error))
    raise ValueError("Rust task trigger response has an invalid result shape")


def parse_task_trigger(
    *,
    delay_seconds: Any = None,
    interval_seconds: Any = None,
    trigger_config: Any = None,
) -> TriggerParseResult:
    rust = _load_rust_task_trigger_parser()
    if rust is not None:
        try:
            return _parse_task_trigger_rust(
                rust,
                delay_seconds=delay_seconds,
                interval_seconds=interval_seconds,
                trigger_config=trigger_config,
            )
        except Exception as error:
            logger.warning(
                "Rust task trigger parsing failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _parse_task_trigger_python(
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        trigger_config=trigger_config,
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
