"""Task scheduler using APScheduler with Redis job store.

Unified scheduler for one-shot and recurring tasks per chat.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from api.ai_billing import AIMessageBilling

_scheduler_instance: Optional[Any] = None

TASK_REDIS_PREFIX = "task:data:"

_MINUTE = 60
_HOUR = 3600
_DAY = 86400


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


def _no_mention(text: str) -> str:
    return text.replace("@", "@\u200b")


def _owner_display(user_name: str) -> str:
    if not user_name:
        return ""
    return f" ({_no_mention(user_name)})"


def _format_run_time(raw: str) -> str:
    if not raw or raw == "unknown":
        return raw
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(timezone(timedelta(hours=-3)))
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
            "misfire_grace_time": timedelta(seconds=300),
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


def _get_redis() -> Any:
    from api.index import config_redis

    try:
        return config_redis()
    except Exception:
        return None


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


def _strip_response_marker(response: str) -> str:
    marker = "[[AI_FALLBACK]]"
    if response.startswith(marker):
        return response[len(marker) :].lstrip()
    return response


def _fire_task(task_id: str) -> None:
    from api.index import (
        ask_ai,
        admin_report,
        build_insufficient_credits_message,
        credits_db_service,
        gen_random,
        send_msg,
    )

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

    chat_id = str(data.get("chat_id", ""))
    text = str(data.get("text", ""))
    user_name = str(data.get("user_name", ""))
    interval = data.get("interval_seconds")
    trigger_cfg = data.get("trigger_config")
    user_id = data.get("user_id")

    if not chat_id or not text:
        print(f"task_scheduler: {task_id} missing chat_id or text")
        return

    if not user_name:
        print(f"task_scheduler: {task_id} missing user_name, skipping")
        return

    display = user_name

    task_message = {"from": {"id": user_id}} if user_id else {}
    billing = AIMessageBilling(
        credits_db_service=credits_db_service,
        admin_reporter=admin_report,
        gen_random_fn=gen_random,
        build_insufficient_credits_message_fn=build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda uid: None,
        command="task",
        chat_id=chat_id,
        chat_type="private",
        user_id=user_id,
        numeric_chat_id=None,
        message=task_message,
    )

    messages = [{"role": "user", "content": text}]
    response_meta: Dict[str, Any] = {}
    is_fallback = False

    reserve_meta, reserve_error = billing.reserve_ai_credits(
        "task_ai",
        1000,
        metadata={"task_id": task_id, "chat_id": chat_id},
    )
    if reserve_error:
        print(f"task_scheduler: {task_id} no credits, skipping: {reserve_error}")
        if not interval and not trigger_cfg:
            _delete_task(key, task_id, redis_client)
        return

    try:
        print(f"task_scheduler: {task_id} calling ask_ai...")
        response = ask_ai(
            messages,
            response_meta=response_meta,
            enable_web_search=True,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
        )
        is_fallback = response.startswith("[[AI_FALLBACK]]")
        if response:
            response = _strip_response_marker(response)
            send_msg(chat_id, f"{display}, tarea programada: {response}")
            print(f"task_scheduler: {task_id} completed successfully")
    except Exception as e:
        print(f"task_scheduler: {task_id} ask_ai failed: {e}")
        admin_report(f"task_scheduler {task_id} ask_ai error", e, {"chat_id": chat_id})
        is_fallback = True
    else:
        if is_fallback:
            billing.refund_reserved_ai_credits(reserve_meta, reason="task_fallback")
        else:
            segments = list(response_meta.get("billing_segments") or [])
            billing.settle_reserved_ai_credits(
                reserve_meta,
                segments,
                reason="task_success",
            )

    if not interval and not trigger_cfg:
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
                    "next_run": _format_run_time(next_run),
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
