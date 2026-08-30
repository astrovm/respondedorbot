"""Task scheduler using APScheduler with Redis job store.

Unified scheduler for one-shot and recurring tasks per chat.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Hashable, Iterator
from datetime import datetime, timedelta, timezone, UTC
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, cast

from api.core.rust_bridge import load_rust_bridge
from api.core.rust_redis import redis_endpoint_from_env
from api.core.logging import get_logger
from api.i18n import current_locale, tr
from api.bot.general_commands import gen_random
from api.services import credits_db as credits_db_service
from api.tasks.executor import (
    build_task_executor,
    TaskExecutor,
)
from api.tasks.models import (
    CronTrigger,
    DayIntervalTrigger,
    DelayTrigger,
    IntervalTrigger,
    ScheduledTaskRequest,
    TaskTrigger,
    parse_task_trigger,
    trigger_config,
)

logger = get_logger(__name__)

_scheduler_instance: Optional[Any] = None
_redis_client: Optional[Any] = None
_task_executor: Optional[Any] = None

TASK_REDIS_PREFIX = "task:data:"
TASK_CHAT_INDEX_PREFIX = "task:chat:"
TASK_INDEX_TTL = 86400 * 3650
TASK_SCHEMA_VERSION = 1

_MINUTE = 60
_HOUR = 3600
_DAY = 86400
_ENGLISH_TO_SPANISH_WEEKDAY = {
    "mon": "lun",
    "tue": "mar",
    "wed": "mie",
    "thu": "jue",
    "fri": "vie",
    "sat": "sab",
    "sun": "dom",
}


class _RustTaskStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def setex(self, key: str, ttl: int, value: str) -> bool: ...
    def delete(self, key: str) -> int: ...
    def zadd(self, key: str, member: str, score: float) -> int: ...
    def expire(self, key: str, ttl: int) -> bool: ...
    def zrem(self, key: str, members: list[str]) -> int: ...
    def scan(self, pattern: str) -> str: ...
    def zrange(self, key: str) -> str: ...
    def mget(self, keys: list[str]) -> str: ...


class _RustTaskStoreModule(Protocol):
    def RedisTaskStore(
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _RustTaskStore: ...


def _load_rust_task_store() -> _RustTaskStoreModule | None:
    module = load_rust_bridge("RUST_TASK_STORE_IO_ENABLED")
    if module is None:
        return None
    return cast(_RustTaskStoreModule, module)


@lru_cache(maxsize=8)
def _cached_rust_task_store(
    module: Hashable,
    host: str,
    port: int,
    password: str | None,
) -> _RustTaskStore:
    return cast(_RustTaskStoreModule, module).RedisTaskStore(host, port, password)


class _RustTaskRedisClient:
    """Expose the scheduler's narrow Redis subset through the Rust adapter."""

    def __init__(self, store: _RustTaskStore) -> None:
        self._store = store

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        return self._store.setex(key, ttl, value)

    def delete(self, key: str) -> int:
        return self._store.delete(key)

    def zadd(self, key: str, mapping: Mapping[str, float]) -> int:
        return sum(
            self._store.zadd(key, str(member), float(score))
            for member, score in mapping.items()
        )

    def expire(self, key: str, ttl: int) -> bool:
        return self._store.expire(key, ttl)

    def zrem(self, key: str, *members: str) -> int:
        return self._store.zrem(key, list(members))

    def scan_iter(self, pattern: str) -> Iterator[str]:
        loaded = json.loads(self._store.scan(pattern))
        if not isinstance(loaded, list):
            raise ValueError("Rust task-store scan result must be a list")
        return iter(str(key) for key in loaded)

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        if (start, end) != (0, -1):
            raise ValueError("Rust task-store only supports complete index reads")
        loaded = json.loads(self._store.zrange(key))
        if not isinstance(loaded, list):
            raise ValueError("Rust task-store sorted-set result must be a list")
        return [str(member) for member in loaded]

    def mget(self, keys: list[str]) -> list[str | None]:
        loaded = json.loads(self._store.mget(keys))
        if not isinstance(loaded, list):
            raise ValueError("Rust task-store multi-get result must be a list")
        return [None if value is None else str(value) for value in loaded]


def _task_redis_client(redis_factory: Callable[[], Any]) -> Any:
    module = _load_rust_task_store()
    if module is None:
        return redis_factory()
    host, port, password = redis_endpoint_from_env()
    store = _cached_rust_task_store(
        cast(Hashable, module),
        host,
        port,
        password,
    )
    return _RustTaskRedisClient(store)


def init_scheduler(
    redis_factory: Callable[[], Any],
    task_executor_deps: Dict[str, Any],
) -> None:
    global _redis_client, _task_executor
    _redis_client = _task_redis_client(redis_factory)
    _task_executor = build_task_executor(**task_executor_deps)
    status = get_scheduler_runtime_status()
    logger.info(
        "runtime ready: scheduler=%s redis=%s executor=%s reason=%s",
        status["scheduler"],
        status["redis"],
        status["executor"],
        status["reason"] or "ok",
    )


def _ensure_runtime_deps() -> None:
    if _redis_client is not None and _task_executor is not None:
        return

    try:
        from api.index import app_runtime

        init_scheduler(
            redis_factory=app_runtime.config.redis,
            task_executor_deps={
                "ask_ai": app_runtime.ai.ask,
                "send_msg": app_runtime.telegram.send_message,
                "admin_report": app_runtime.admin.report,
                "credits_db_service": credits_db_service,
                "gen_random_fn": gen_random,
                "build_insufficient_credits_message_fn": (
                    app_runtime.billing.build_insufficient_message
                ),
                "estimate_ai_base_reserve_credits": (app_runtime.estimate_ai_base_reserve_credits),
            },
        )
    except Exception as error:
        logger.error("failed to initialize runtime deps: %s", error)


def _get_task_executor() -> Any:
    _ensure_runtime_deps()
    return _task_executor


def estimate_task_reserve_credits(text: str) -> Optional[int]:
    """Estimate the personal credits required for a future task execution."""

    executor = _get_task_executor()
    if executor is None:
        return None
    return int(executor.estimate_required_credits(text))


def set_task_executor(executor: Any) -> None:
    global _task_executor
    _task_executor = executor


def set_redis_client(client: Any) -> None:
    global _redis_client
    _redis_client = client


def get_scheduler_runtime_status() -> Dict[str, Any]:
    scheduler = get_scheduler()
    redis_client = _get_redis()
    executor = _get_task_executor()

    if scheduler is None:
        return {
            "ready": False,
            "reason": "scheduler unavailable",
            "scheduler": False,
            "redis": redis_client is not None,
            "executor": executor is not None,
        }

    if redis_client is None:
        return {
            "ready": False,
            "reason": "storage unavailable",
            "scheduler": True,
            "redis": False,
            "executor": executor is not None,
        }

    if executor is None:
        return {
            "ready": False,
            "reason": "task executor unavailable",
            "scheduler": True,
            "redis": True,
            "executor": False,
        }

    return {
        "ready": True,
        "reason": "",
        "scheduler": True,
        "redis": True,
        "executor": True,
    }


def format_interval(seconds: int, prefix: str = "cada ") -> str:
    if seconds >= _DAY:
        val = seconds // _DAY
        unit = tr("task.day" if val == 1 else "task.days")
    elif seconds >= _HOUR:
        val = seconds // _HOUR
        unit = tr("task.hour" if val == 1 else "task.hours")
    else:
        val = seconds // _MINUTE
        unit = tr("task.minute" if val == 1 else "task.minutes")
    key = "task.in" if prefix.strip() == "en" else "task.every"
    return tr(key, value=val, unit=unit)


def describe_trigger(trigger_config: Optional[Dict[str, Any]]) -> str:
    if not trigger_config:
        return ""
    parsed = parse_task_trigger(trigger_config=trigger_config)
    return describe_task_trigger(parsed.trigger) if parsed.trigger else ""


def describe_task_trigger(trigger: TaskTrigger) -> str:
    if isinstance(trigger, CronTrigger):
        time_text = f"{trigger.hour:02d}:{trigger.minute:02d}"
        if trigger.weekdays:
            weekdays = ", ".join(
                day if current_locale() == "en" else _ENGLISH_TO_SPANISH_WEEKDAY.get(day, day)
                for day in trigger.weekdays
            )
            return tr("task.weekdays", weekdays=weekdays, time=time_text)
        if trigger.day is not None:
            return tr("task.monthly", day=trigger.day, time=time_text)
        return tr("task.daily", time=time_text)
    if isinstance(trigger, DayIntervalTrigger):
        return tr("task.every", value=trigger.days, unit=tr("task.days"))
    if isinstance(trigger, IntervalTrigger):
        return format_interval(trigger.seconds)
    return format_interval(trigger.seconds, "en ")


def format_task_summary(task: Dict[str, Any], *, prefix: str = "") -> str:
    interval = task.get("interval_seconds")
    trigger_config = task.get("trigger_config")
    owner_bit = _owner_display(task.get("user_name", ""))
    next_run = task.get("next_run", "")
    task_text = _no_mention(task.get("text", ""))

    if interval:
        freq = format_interval(interval)
        return (
            f"{prefix}[{task['id']}] {task_text}{owner_bit} - {freq}, "
            f"{tr('task.next', time=next_run)}"
        )

    if trigger_config:
        freq = describe_trigger(trigger_config)
        return (
            f"{prefix}[{task['id']}] {task_text}{owner_bit} - {freq}, "
            f"{tr('task.next', time=next_run)}"
        )

    return f"{prefix}[{task['id']}] {task_text}{owner_bit} - {next_run}"


def _no_mention(text: str) -> str:
    # Preserve the visible username without triggering a Telegram mention.
    return text.replace("@", "@\u200b")


def _owner_display(user_name: str) -> str:
    if not user_name:
        return ""
    return f" ({_no_mention(user_name)})"


def _coerce_timezone_offset(value: Any, default: int = -3) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _format_run_time(raw: str, timezone_offset: int = -3) -> str:
    if not raw or raw == "unknown":
        return raw
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        local = dt.astimezone(timezone(timedelta(hours=timezone_offset)))
        return local.strftime("%d/%m %H:%M")
    except ValueError, TypeError:
        return raw


def get_scheduler() -> Any:
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.jobstores.redis import RedisJobStore

        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.environ.get("REDIS_PASSWORD", "")

        jobstores = {
            "default": RedisJobStore(
                host=host,
                port=port,
                db=1,
                password=password or None,
            )
        }
        job_defaults = {
            "misfire_grace_time": 300,
            "coalesce": True,
            "max_instances": 100,
        }
        _scheduler_instance = BackgroundScheduler(
            jobstores=jobstores,
            job_defaults=job_defaults,
        )
        _scheduler_instance.start()
        logger.info("started with %d jobstores", len(jobstores))
        return _scheduler_instance
    except Exception as e:
        logger.error("failed to initialize APScheduler: %s", e)
        return None


def shutdown_scheduler() -> None:
    global _scheduler_instance
    if _scheduler_instance is not None:
        _scheduler_instance.shutdown(wait=False)
        _scheduler_instance = None


def _get_redis() -> Any:
    _ensure_runtime_deps()
    return _redis_client


def _task_index_key(chat_id: str) -> str:
    return f"{TASK_CHAT_INDEX_PREFIX}{chat_id}"


def _task_index_marker_key(chat_id: str) -> str:
    return f"{_task_index_key(chat_id)}:indexed"


def _task_score(data: Mapping[str, Any]) -> float:
    raw_run_date = data.get("next_run_at") or data.get("run_date")
    if raw_run_date:
        try:
            return datetime.fromisoformat(str(raw_run_date).replace("Z", "+00:00")).timestamp()
        except TypeError, ValueError:
            pass
    return 0.0


def _index_task(redis_client: Any, data: Mapping[str, Any]) -> None:
    chat_id = str(data.get("chat_id") or "")
    task_id = str(data.get("id") or "")
    if not chat_id or not task_id:
        return
    index_key = _task_index_key(chat_id)
    redis_client.zadd(index_key, {task_id: _task_score(data)})
    redis_client.expire(index_key, TASK_INDEX_TTL)
    redis_client.setex(_task_index_marker_key(chat_id), TASK_INDEX_TTL, "1")


def _decode_task(raw: Any) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _iso_utc(value: Any) -> str | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _job_next_run(scheduler: Any, task_id: str) -> str | None:
    if scheduler is None:
        return None
    try:
        job = scheduler.get_job(f"task_{task_id}")
        return _iso_utc(job.next_run_time) if job is not None else None
    except Exception:
        return None


def _canonical_task_data(
    data: Mapping[str, Any],
    *,
    next_run_at: str | None = None,
    schedule_anchor_at: str | None = None,
) -> Dict[str, Any]:
    canonical = dict(data)
    canonical["schema_version"] = TASK_SCHEMA_VERSION
    canonical["schedule_anchor_at"] = (
        _iso_utc(schedule_anchor_at)
        or _iso_utc(canonical.get("schedule_anchor_at"))
        or _iso_utc(canonical.get("run_date"))
    )
    canonical["next_run_at"] = (
        _iso_utc(next_run_at)
        or _iso_utc(canonical.get("next_run_at"))
        or _iso_utc(canonical.get("run_date"))
    )
    canonical.setdefault("last_execution_id", None)
    return canonical


def _is_recurring(data: Mapping[str, Any]) -> bool:
    return bool(data.get("interval_seconds") or data.get("trigger_config"))


def backfill_canonical_task_records() -> Dict[str, int]:
    """Copy APScheduler next-run state into every readable task record.

    This is safe while Python remains the sole scheduler owner. Missing
    recurring jobs are reported instead of receiving an invented schedule.
    """

    redis_client = _get_redis()
    scheduler = get_scheduler()
    status = {"scanned": 0, "updated": 0, "unmatched": 0, "invalid": 0}
    if redis_client is None or scheduler is None:
        return status

    try:
        keys = list(redis_client.scan_iter(f"{TASK_REDIS_PREFIX}*"))
    except Exception as error:
        logger.error("failed to scan task records for canonical backfill: %s", error)
        return status

    for raw_key in keys:
        key = raw_key if isinstance(raw_key, str) else raw_key.decode("utf-8")
        status["scanned"] += 1
        try:
            data = _decode_task(redis_client.get(key))
            if data is None:
                status["invalid"] += 1
                continue
            task_id = str(data.get("id") or "")
            if not task_id:
                status["invalid"] += 1
                continue
            next_run_at = _job_next_run(scheduler, task_id)
            if next_run_at is None and _is_recurring(data):
                status["unmatched"] += 1
                continue
            canonical = _canonical_task_data(data, next_run_at=next_run_at)
            if canonical != data:
                redis_client.setex(key, TASK_INDEX_TTL, json.dumps(canonical))
                _index_task(redis_client, canonical)
                status["updated"] += 1
        except Exception as error:
            status["invalid"] += 1
            logger.warning("failed to canonicalize task record %s: %s", key, error)

    if status["unmatched"] or status["invalid"]:
        logger.warning(
            "task canonical backfill incomplete: scanned=%d updated=%d unmatched=%d invalid=%d",
            status["scanned"],
            status["updated"],
            status["unmatched"],
            status["invalid"],
        )
    else:
        logger.info(
            "task canonical backfill complete: scanned=%d updated=%d",
            status["scanned"],
            status["updated"],
        )
    return status


class _CanonicalTaskMissingNext(ValueError):
    pass


def _utc_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_scheduler_trigger(data: Mapping[str, Any], next_run_at: datetime) -> Any:
    interval_seconds = data.get("interval_seconds")
    trigger_config_value = data.get("trigger_config")
    if interval_seconds is not None:
        parsed = parse_task_trigger(interval_seconds=interval_seconds)
    elif trigger_config_value is not None:
        parsed = parse_task_trigger(trigger_config=trigger_config_value)
    else:
        parsed = parse_task_trigger(delay_seconds=1)
    if parsed.trigger is None:
        raise ValueError("canonical task trigger is invalid")
    raw_anchor = data.get("schedule_anchor_at") or data.get("next_run_at")
    return _apscheduler_trigger(
        parsed.trigger,
        _coerce_timezone_offset(data.get("timezone_offset")),
        next_run_at if isinstance(parsed.trigger, DelayTrigger) else None,
        _utc_datetime(raw_anchor),
    )


def _rebuild_canonical_task_job(
    redis_client: Any,
    scheduler: Any,
    key: str,
) -> str:
    data = _decode_task(redis_client.get(key))
    if data is None or data.get("schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("task record is not canonical schema version 1")
    task_id = str(data.get("id") or "")
    if not task_id:
        raise ValueError("canonical task id is missing")
    raw_next_run = data.get("next_run_at")
    if not raw_next_run:
        raise _CanonicalTaskMissingNext("canonical next run is missing")
    next_run_at = _utc_datetime(raw_next_run)
    _add_scheduled_job(
        scheduler,
        task_id,
        _canonical_scheduler_trigger(data, next_run_at),
        next_run_at=next_run_at,
    )
    _index_task(redis_client, data)
    return f"task_{task_id}"


def _remove_orphaned_task_jobs(scheduler: Any, expected_jobs: set[str]) -> int:
    removed = 0
    for job in scheduler.get_jobs():
        job_id = str(getattr(job, "id", ""))
        if job_id.startswith("task_") and job_id not in expected_jobs:
            scheduler.remove_job(job_id)
            removed += 1
    return removed


def rebuild_scheduler_from_canonical_records() -> Dict[str, int]:
    """Make APScheduler an execution cache of canonical task records.

    The language-neutral Redis records remain authoritative. Every valid record
    receives its stored ``next_run_at`` exactly, and orphaned APScheduler task
    jobs are removed so a restart cannot resurrect a deleted task.
    """

    redis_client = _get_redis()
    scheduler = get_scheduler()
    status = {
        "scanned": 0,
        "rebuilt": 0,
        "removed": 0,
        "missing_next": 0,
        "invalid": 0,
    }
    if redis_client is None or scheduler is None:
        return status

    try:
        keys = list(redis_client.scan_iter(f"{TASK_REDIS_PREFIX}*"))
    except Exception as error:
        logger.error("failed to scan canonical task records: %s", error)
        return status

    expected_jobs: set[str] = set()
    for raw_key in keys:
        key = raw_key if isinstance(raw_key, str) else raw_key.decode("utf-8")
        status["scanned"] += 1
        try:
            expected_jobs.add(_rebuild_canonical_task_job(redis_client, scheduler, key))
            status["rebuilt"] += 1
        except _CanonicalTaskMissingNext:
            status["missing_next"] += 1
        except Exception as error:
            status["invalid"] += 1
            logger.warning("failed to rebuild canonical task %s: %s", key, error)

    try:
        status["removed"] = _remove_orphaned_task_jobs(scheduler, expected_jobs)
    except Exception as error:
        status["invalid"] += 1
        logger.warning("failed to remove orphaned task jobs: %s", error)

    logger.info(
        "canonical task rebuild: scanned=%d rebuilt=%d removed=%d missing_next=%d invalid=%d",
        status["scanned"],
        status["rebuilt"],
        status["removed"],
        status["missing_next"],
        status["invalid"],
    )
    return status


def _delete_task(
    redis_key: str,
    task_id: str,
    redis_client: Any = None,
    *,
    chat_id: str = "",
) -> None:
    client = redis_client if redis_client is not None else _get_redis()
    if client is not None:
        try:
            client.delete(redis_key)
            if chat_id:
                client.zrem(_task_index_key(chat_id), task_id)
        except Exception:
            pass
    scheduler = get_scheduler()
    if scheduler is not None:
        try:
            scheduler.remove_job(f"task_{task_id}")
        except Exception:
            pass


def _fire_task(task_id: str) -> None:
    logger.info("firing task %s", task_id)

    redis_client = _get_redis()
    if redis_client is None:
        logger.warning("no redis, cannot fire %s", task_id)
        return

    key = f"{TASK_REDIS_PREFIX}{task_id}"
    raw = redis_client.get(key)
    if not raw:
        logger.warning("no data for %s in redis", task_id)
        return

    data = _decode_task(raw)
    if data is None:
        logger.warning("invalid JSON for %s", task_id)
        return

    executor = _get_task_executor()
    if executor is None:
        return

    executor._pool.submit(_execute_and_cleanup, executor, data, key, task_id)


def _execute_and_cleanup(
    executor: TaskExecutor, data: Mapping[str, Any], key: str, task_id: str
) -> None:
    if executor.execute(data):
        redis_client = _get_redis()
        if redis_client is not None:
            _delete_task(
                key,
                task_id,
                redis_client,
                chat_id=str(data.get("chat_id") or ""),
            )


def _run_date(trigger: TaskTrigger, created_at: datetime | None = None) -> datetime | None:
    if not isinstance(trigger, DelayTrigger):
        return None
    created_at = created_at or datetime.now(UTC)
    return created_at + timedelta(seconds=trigger.seconds)


def _apscheduler_trigger(
    trigger: TaskTrigger,
    timezone_offset: int,
    run_date: datetime | None,
    created_at: datetime,
) -> Any:
    if isinstance(trigger, CronTrigger):
        from apscheduler.triggers.cron import (  # type: ignore[import-untyped]
            CronTrigger as APSCronTrigger,
        )

        kwargs: Dict[str, Any] = {
            "timezone": timezone(timedelta(hours=timezone_offset)),
            "hour": trigger.hour,
            "minute": trigger.minute,
        }
        if trigger.weekdays:
            kwargs["day_of_week"] = ",".join(trigger.weekdays)
        if trigger.day is not None:
            kwargs["day"] = trigger.day
        return APSCronTrigger(**kwargs)
    if isinstance(trigger, (DayIntervalTrigger, IntervalTrigger)):
        from apscheduler.triggers.interval import (  # type: ignore[import-untyped]
            IntervalTrigger as APSIntervalTrigger,
        )

        if isinstance(trigger, DayIntervalTrigger):
            return APSIntervalTrigger(
                days=trigger.days,
                start_date=created_at + timedelta(days=trigger.days),
            )
        if isinstance(trigger, IntervalTrigger):
            return APSIntervalTrigger(
                seconds=trigger.seconds,
                start_date=created_at + timedelta(seconds=trigger.seconds),
            )
    if run_date is not None:
        from apscheduler.triggers.date import DateTrigger  # type: ignore[import-untyped]

        return DateTrigger(run_date=run_date)
    raise ValueError("task trigger has no schedulable occurrence")


def _task_payload(
    task_id: str,
    request: ScheduledTaskRequest,
    run_date: datetime | None,
    created_at: datetime,
    next_run_at: datetime,
) -> Dict[str, Any]:
    interval_seconds = (
        request.trigger.seconds if isinstance(request.trigger, IntervalTrigger) else None
    )
    return _canonical_task_data({
        "id": task_id,
        "chat_id": request.chat_id,
        "text": request.text,
        "user_name": request.user_name,
        "user_id": request.user_id,
        "interval_seconds": interval_seconds,
        "run_date": run_date.isoformat() if run_date else None,
        "trigger_config": trigger_config(request.trigger),
        "timezone_offset": _coerce_timezone_offset(request.timezone_offset),
        "locale": request.locale,
    }, schedule_anchor_at=created_at.isoformat(), next_run_at=next_run_at.isoformat())


def _add_scheduled_job(
    scheduler: Any,
    task_id: str,
    scheduler_trigger: Any,
    *,
    next_run_at: datetime | None = None,
) -> None:
    common = {
        "id": f"task_{task_id}",
        "args": [task_id],
        "replace_existing": True,
    }
    if next_run_at is not None:
        common["next_run_time"] = next_run_at
    scheduler.add_job(_fire_task, scheduler_trigger, **common)


def schedule_task(request: ScheduledTaskRequest) -> Optional[str]:
    scheduler = get_scheduler()
    redis_client = _get_redis()
    if scheduler is None or redis_client is None:
        if redis_client is None:
            logger.warning("no redis, cannot schedule task")
        return None

    task_id = str(uuid.uuid4())[:8]
    created_at = datetime.now(UTC)
    run_date = _run_date(request.trigger, created_at)
    try:
        scheduler_trigger = _apscheduler_trigger(
            request.trigger,
            request.timezone_offset,
            run_date,
            created_at,
        )
        next_run_at = scheduler_trigger.get_next_fire_time(None, created_at)
        if next_run_at is None:
            return None
    except Exception as e:
        logger.error("failed to construct task trigger: %s", e)
        return None
    redis_key = f"{TASK_REDIS_PREFIX}{task_id}"
    data = _task_payload(task_id, request, run_date, created_at, next_run_at)
    try:
        redis_client.setex(redis_key, TASK_INDEX_TTL, json.dumps(data))
        _index_task(redis_client, data)
    except Exception as e:
        logger.error("failed to persist task %s: %s", task_id, e)
        try:
            redis_client.delete(redis_key)
            redis_client.zrem(_task_index_key(request.chat_id), task_id)
        except Exception:
            pass
        return None

    try:
        _add_scheduled_job(
            scheduler,
            task_id,
            scheduler_trigger,
        )
    except Exception as e:
        logger.error("add_job failed for %s: %s", task_id, e)
        if redis_client is not None:
            try:
                redis_client.delete(redis_key)
            except Exception:
                pass
        return None

    return task_id


def _migrate_chat_task_index(redis_client: Any, chat_id: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for key_bytes in redis_client.scan_iter(f"{TASK_REDIS_PREFIX}*"):
        key = key_bytes if isinstance(key_bytes, str) else key_bytes.decode("utf-8")
        data = _decode_task(redis_client.get(key))
        if data is None or str(data.get("chat_id")) != chat_id:
            continue
        tasks.append(data)
        _index_task(redis_client, data)
    redis_client.setex(_task_index_marker_key(chat_id), TASK_INDEX_TTL, "1")
    return tasks


def _load_indexed_tasks(redis_client: Any, chat_id: str) -> List[Dict[str, Any]]:
    index_key = _task_index_key(chat_id)
    raw_ids = redis_client.zrange(index_key, 0, -1)
    task_ids = [item if isinstance(item, str) else item.decode("utf-8") for item in raw_ids]
    if not task_ids:
        if redis_client.get(_task_index_marker_key(chat_id)):
            return []
        return _migrate_chat_task_index(redis_client, chat_id)

    raw_tasks = redis_client.mget([f"{TASK_REDIS_PREFIX}{task_id}" for task_id in task_ids])
    tasks: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    for task_id, raw in zip(task_ids, raw_tasks):
        data = _decode_task(raw)
        if data is None:
            missing_ids.append(task_id)
            continue
        tasks.append(data)
    if missing_ids:
        redis_client.zrem(index_key, *missing_ids)
    return tasks


def list_tasks(chat_id: str) -> List[Dict[str, Any]]:
    redis_client = _get_redis()
    if redis_client is None:
        return []

    scheduler = get_scheduler()
    normalized_chat_id = str(chat_id)
    results = []
    try:
        for data in _load_indexed_tasks(redis_client, normalized_chat_id):
            task_id = data.get("id", "")
            interval = data.get("interval_seconds")
            timezone_offset = _coerce_timezone_offset(data.get("timezone_offset"), -3)

            next_run = data.get("next_run_at") or data.get("run_date") or "unknown"
            if scheduler is not None:
                scheduler_next_run = _job_next_run(scheduler, str(task_id))
                if scheduler_next_run:
                    next_run = scheduler_next_run

            results.append(
                {
                    "id": task_id,
                    "text": data.get("text", ""),
                    "user_name": data.get("user_name", ""),
                    "user_id": data.get("user_id"),
                    "interval_seconds": interval,
                    "trigger_config": data.get("trigger_config"),
                    "next_run": _format_run_time(next_run, timezone_offset),
                }
            )
    except Exception as e:
        logger.error("list_tasks error: %s", e)

    return results


def cancel_task(task_id: str) -> bool:
    scheduler = get_scheduler()
    if scheduler is not None:
        try:
            scheduler.remove_job(f"task_{task_id}")
        except Exception:
            pass

    redis_client = _get_redis()
    if redis_client is not None:
        try:
            redis_key = f"{TASK_REDIS_PREFIX}{task_id}"
            data = _decode_task(redis_client.get(redis_key))
            _delete_task(
                redis_key,
                task_id,
                redis_client,
                chat_id=str(data.get("chat_id") or "") if data else "",
            )
        except Exception:
            pass

    return True
