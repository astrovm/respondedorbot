"""Task scheduler using APScheduler with Redis job store.

Unified scheduler for one-shot and recurring tasks per chat.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone, UTC
from typing import Any, Callable, Dict, List, Mapping, Optional

from api.core.logging import get_logger
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


def init_scheduler(
    redis_factory: Callable[[], Any],
    task_executor_deps: Dict[str, Any],
) -> None:
    global _redis_client, _task_executor
    _redis_client = redis_factory()
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
                "estimate_ai_base_reserve_credits": (
                    app_runtime.estimate_ai_base_reserve_credits
                ),
            },
        )
    except Exception as error:
        logger.error("failed to initialize runtime deps: %s", error)


def _get_task_executor() -> Any:
    _ensure_runtime_deps()
    return _task_executor


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
        unit = f"dia{'s' if val != 1 else ''}"
    elif seconds >= _HOUR:
        val = seconds // _HOUR
        unit = f"hora{'s' if val != 1 else ''}"
    else:
        val = seconds // _MINUTE
        unit = f"minuto{'s' if val != 1 else ''}"
    return f"{prefix}{val} {unit}"


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
                _ENGLISH_TO_SPANISH_WEEKDAY.get(day, day)
                for day in trigger.weekdays
            )
            return f"los {weekdays} a las {time_text}"
        if trigger.day is not None:
            return f"el dia {trigger.day} de cada mes a las {time_text}"
        return f"todos los dias a las {time_text}"
    if isinstance(trigger, DayIntervalTrigger):
        return f"cada {trigger.days} dias"
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
            f"{prefix}[{task['id']}] {task_text}{owner_bit} - {freq}, prox: {next_run}"
        )

    if trigger_config:
        freq = describe_trigger(trigger_config)
        return (
            f"{prefix}[{task['id']}] {task_text}{owner_bit} - {freq}, prox: {next_run}"
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
    except (TypeError, ValueError):
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
    except (ValueError, TypeError):
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
    raw_run_date = data.get("run_date")
    if raw_run_date:
        try:
            return datetime.fromisoformat(str(raw_run_date).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
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
    except (json.JSONDecodeError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


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


def _run_date(trigger: TaskTrigger) -> datetime | None:
    if not isinstance(trigger, DelayTrigger):
        return None
    return datetime.fromtimestamp(
        datetime.now(UTC).timestamp() + trigger.seconds,
        tz=UTC,
    )


def _task_payload(
    task_id: str,
    request: ScheduledTaskRequest,
    run_date: datetime | None,
) -> Dict[str, Any]:
    interval_seconds = (
        request.trigger.seconds
        if isinstance(request.trigger, IntervalTrigger)
        else None
    )
    return {
        "id": task_id,
        "chat_id": request.chat_id,
        "text": request.text,
        "user_name": request.user_name,
        "user_id": request.user_id,
        "interval_seconds": interval_seconds,
        "run_date": run_date.isoformat() if run_date else None,
        "trigger_config": trigger_config(request.trigger),
        "timezone_offset": _coerce_timezone_offset(request.timezone_offset),
    }


def _add_scheduled_job(
    scheduler: Any,
    task_id: str,
    trigger: TaskTrigger,
    timezone_offset: int,
    run_date: datetime | None,
) -> None:
    common = {
        "id": f"task_{task_id}",
        "args": [task_id],
        "replace_existing": True,
    }
    if isinstance(trigger, CronTrigger):
        kwargs: Dict[str, Any] = {
            "timezone": timezone(timedelta(hours=timezone_offset)),
            "hour": trigger.hour,
            "minute": trigger.minute,
        }
        if trigger.weekdays:
            kwargs["day_of_week"] = ",".join(trigger.weekdays)
        if trigger.day is not None:
            kwargs["day"] = trigger.day
        scheduler.add_job(_fire_task, "cron", **common, **kwargs)
    elif isinstance(trigger, DayIntervalTrigger):
        scheduler.add_job(_fire_task, "interval", days=trigger.days, **common)
    elif isinstance(trigger, IntervalTrigger):
        scheduler.add_job(
            _fire_task,
            "interval",
            seconds=trigger.seconds,
            **common,
        )
    elif run_date is not None:
        scheduler.add_job(_fire_task, "date", run_date=run_date, **common)


def schedule_task(request: ScheduledTaskRequest) -> Optional[str]:
    scheduler = get_scheduler()
    if scheduler is None:
        return None

    redis_client = _get_redis()
    if redis_client is None:
        logger.warning("no redis, cannot schedule task")
        return None

    task_id = str(uuid.uuid4())[:8]
    run_date = _run_date(request.trigger)
    redis_key = f"{TASK_REDIS_PREFIX}{task_id}"
    data = _task_payload(task_id, request, run_date)
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
            request.trigger,
            request.timezone_offset,
            run_date,
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
    task_ids = [
        item if isinstance(item, str) else item.decode("utf-8")
        for item in raw_ids
    ]
    if not task_ids:
        if redis_client.get(_task_index_marker_key(chat_id)):
            return []
        return _migrate_chat_task_index(redis_client, chat_id)

    raw_tasks = redis_client.mget(
        [f"{TASK_REDIS_PREFIX}{task_id}" for task_id in task_ids]
    )
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

            next_run = data.get("run_date") or "unknown"
            if scheduler is not None:
                job_id = f"task_{task_id}"
                try:
                    job = scheduler.get_job(job_id)
                    if job and job.next_run_time:
                        next_run = str(job.next_run_time)
                except Exception:
                    pass

            results.append(
                {
                    "id": task_id,
                    "text": data.get("text", ""),
                    "user_name": data.get("user_name", ""),
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
