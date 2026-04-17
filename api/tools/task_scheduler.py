"""Task scheduler using APScheduler with Redis job store.

Unified scheduler for one-shot and recurring tasks per chat.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from api.task_executor import (
    _strip_response_marker as _task_executor_strip_response_marker,
    build_task_executor,
)

_scheduler_instance: Optional[Any] = None
_redis_client: Optional[Any] = None
_task_executor: Optional[Any] = None

TASK_REDIS_PREFIX = "task:data:"

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


def _ensure_runtime_deps() -> None:
    if _redis_client is not None and _task_executor is not None:
        return

    try:
        from api import index as _index

        init_scheduler(
            redis_factory=_index.config_redis,
            task_executor_deps={
                "ask_ai": _index.ask_ai,
                "send_msg": _index.send_msg,
                "admin_report": _index.admin_report,
                "credits_db_service": _index.credits_db_service,
                "gen_random_fn": _index.gen_random,
                "build_insufficient_credits_message_fn": (
                    _index.build_insufficient_credits_message
                ),
            },
        )
    except Exception as error:
        print(f"task_scheduler: failed to initialize runtime deps: {error}")


def _get_task_executor() -> Any:
    _ensure_runtime_deps()
    return _task_executor


def set_task_executor(executor: Any) -> None:
    global _task_executor
    _task_executor = executor


def set_redis_client(client: Any) -> None:
    global _redis_client
    _redis_client = client


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

    trigger_type = trigger_config.get("type", "interval")

    if trigger_type == "cron":
        hour = trigger_config.get("hour")
        minute = trigger_config.get("minute", 0)
        day_of_week = trigger_config.get("day_of_week")
        day = trigger_config.get("day")
        time_str = f"{hour:02d}:{minute:02d}" if hour is not None else "a alguna hora"

        if day_of_week:
            weekday_tokens = []
            for token in str(day_of_week).split(","):
                normalized = token.strip().lower()
                if normalized:
                    weekday_tokens.append(
                        _ENGLISH_TO_SPANISH_WEEKDAY.get(normalized, normalized)
                    )
            weekdays_text = ", ".join(weekday_tokens) or str(day_of_week)
            return f"los {weekdays_text} a las {time_str}"
        if day:
            return f"el dia {day} de cada mes a las {time_str}"
        return f"todos los dias a las {time_str}"

    if trigger_type == "interval":
        days = trigger_config.get("days", 0)
        if days > 0:
            return f"cada {days} dias"

    return ""


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
            dt = dt.replace(tzinfo=timezone.utc)
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
            "max_instances": 3,
        }
        _scheduler_instance = BackgroundScheduler(
            jobstores=jobstores,
            job_defaults=job_defaults,
        )
        _scheduler_instance.start()
        print(f"task_scheduler: started with {len(jobstores)} jobstores")
        return _scheduler_instance
    except Exception as e:
        print(f"task_scheduler: failed to initialize APScheduler: {e}")
        return None


def shutdown_scheduler() -> None:
    global _scheduler_instance
    if _scheduler_instance is not None:
        _scheduler_instance.shutdown(wait=False)
        _scheduler_instance = None


def _strip_response_marker(response: str) -> str:
    return _task_executor_strip_response_marker(response)


def _get_redis() -> Any:
    _ensure_runtime_deps()
    return _redis_client


def _delete_task(redis_key: str, task_id: str, redis_client: Any = None) -> None:
    client = redis_client if redis_client is not None else _get_redis()
    if client is not None:
        try:
            client.delete(redis_key)
        except Exception:
            pass
    scheduler = get_scheduler()
    if scheduler is not None:
        try:
            scheduler.remove_job(f"task_{task_id}")
        except Exception:
            pass


def _fire_task(task_id: str) -> None:
    print(f"task_scheduler: firing task {task_id}")

    redis_client = _get_redis()
    if redis_client is None:
        print(f"task_scheduler: no redis, cannot fire {task_id}")
        return

    key = f"{TASK_REDIS_PREFIX}{task_id}"
    raw = redis_client.get(key)
    if not raw:
        print(f"task_scheduler: no data for {task_id} in redis")
        return

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        print(f"task_scheduler: invalid JSON for {task_id}")
        return

    if _get_task_executor() is not None and _get_task_executor().execute(data):
        _delete_task(key, task_id, redis_client)


def schedule_task(
    chat_id: str,
    text: str,
    delay_seconds: Optional[int] = None,
    interval_seconds: Optional[int] = None,
    user_name: str = "",
    user_id: Optional[int] = None,
    trigger_config: Optional[Dict[str, Any]] = None,
    timezone_offset: int = -3,
) -> Optional[str]:
    if delay_seconds is None and interval_seconds is None and not trigger_config:
        return None

    scheduler = get_scheduler()
    if scheduler is None:
        return None

    task_id = str(uuid.uuid4())[:8]

    run_date = None
    is_recurring = bool(interval_seconds or trigger_config)

    if delay_seconds is not None:
        run_date = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + delay_seconds,
            tz=timezone.utc,
        )

    redis_key = f"{TASK_REDIS_PREFIX}{task_id}"
    redis_client = _get_redis()
    if redis_client is not None:
        data = {
            "id": task_id,
            "chat_id": str(chat_id),
            "text": text,
            "user_name": user_name,
            "user_id": user_id,
            "interval_seconds": interval_seconds,
            "run_date": run_date.isoformat() if run_date else None,
            "trigger_config": trigger_config,
            "timezone_offset": _coerce_timezone_offset(timezone_offset),
        }
        ttl = 86400 * 90 if is_recurring else 86400 * 30
        redis_client.setex(redis_key, ttl, json.dumps(data))

    try:
        if trigger_config and trigger_config.get("type") == "cron":
            tz = timezone(timedelta(hours=timezone_offset))

            cron_kwargs: Dict[str, Any] = {"timezone": tz}
            for key, conv in (
                ("hour", int),
                ("minute", int),
                ("day", int),
                ("day_of_week", str),
            ):
                if key in trigger_config:
                    cron_kwargs[key] = conv(trigger_config[key])

            scheduler.add_job(
                _fire_task,
                "cron",
                id=f"task_{task_id}",
                args=[task_id],
                replace_existing=True,
                **cron_kwargs,
            )
        elif trigger_config and trigger_config.get("type") == "interval":
            days = int(trigger_config.get("days", 1))
            scheduler.add_job(
                _fire_task,
                "interval",
                days=max(days, 1),
                id=f"task_{task_id}",
                args=[task_id],
                replace_existing=True,
            )
        elif interval_seconds:
            scheduler.add_job(
                _fire_task,
                "interval",
                seconds=interval_seconds,
                id=f"task_{task_id}",
                args=[task_id],
                replace_existing=True,
            )
        elif run_date:
            scheduler.add_job(
                _fire_task,
                "date",
                run_date=run_date,
                id=f"task_{task_id}",
                args=[task_id],
                replace_existing=True,
            )
    except Exception as e:
        print(f"task_scheduler: add_job failed for {task_id}: {e}")
        if redis_client is not None:
            try:
                redis_client.delete(redis_key)
            except Exception:
                pass
        return None

    return task_id


def list_tasks(chat_id: str) -> List[Dict[str, Any]]:
    redis_client = _get_redis()
    if redis_client is None:
        return []

    scheduler = get_scheduler()

    results = []
    try:
        for key_bytes in redis_client.scan_iter(f"{TASK_REDIS_PREFIX}*"):
            key = key_bytes if isinstance(key_bytes, str) else key_bytes.decode("utf-8")
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if str(data.get("chat_id")) != str(chat_id):
                continue

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
        print(f"list_tasks error: {e}")

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
            redis_client.delete(f"{TASK_REDIS_PREFIX}{task_id}")
        except Exception:
            pass

    return True
