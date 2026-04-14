"""Reminder scheduler using APScheduler with Redis job store.

Manages per-chat reminders that fire and send messages via the bot.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_scheduler_instance: Optional[Any] = None

REMINDER_REDIS_PREFIX = "reminder:data:"


def _get_redis_url() -> str:
    import os

    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD", "")
    if password:
        return f"redis://:{password}@{host}:{port}/1"
    return f"redis://{host}:{port}/1"


def get_scheduler() -> Any:
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.jobstores.redis import RedisJobStore

        jobstores = {"default": RedisJobStore(url=_get_redis_url())}
        _scheduler_instance = BackgroundScheduler(jobstores=jobstores)
        _scheduler_instance.start()
        return _scheduler_instance
    except Exception as e:
        print(f"reminder_scheduler: failed to initialize APScheduler: {e}")
        return None


def shutdown_scheduler() -> None:
    global _scheduler_instance
    if _scheduler_instance is not None:
        _scheduler_instance.shutdown(wait=False)
        _scheduler_instance = None


def _store_reminder_meta(
    redis_client: Any,
    reminder_id: str,
    chat_id: str,
    text: str,
    user_name: str,
) -> None:
    key = f"{REMINDER_REDIS_PREFIX}{reminder_id}"
    data = json.dumps(
        {
            "id": reminder_id,
            "chat_id": str(chat_id),
            "text": text,
            "user_name": user_name,
        }
    )
    redis_client.setex(key, 86400 * 30, data)


def _get_redis() -> Any:
    from api.index import config_redis

    try:
        return config_redis()
    except Exception:
        return None


def _fire_reminder(reminder_id: str) -> None:
    redis_client = _get_redis()
    if redis_client is None:
        print(f"reminder_scheduler: no redis, cannot fire {reminder_id}")
        return

    key = f"{REMINDER_REDIS_PREFIX}{reminder_id}"
    raw = redis_client.get(key)
    if not raw:
        print(f"reminder_scheduler: no data for {reminder_id}")
        return

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return

    chat_id = str(data.get("chat_id", ""))
    text = str(data.get("text", ""))
    user_name = str(data.get("user_name", ""))

    if not chat_id or not text:
        return

    from api.index import send_msg

    display = f"@{user_name}" if user_name else "che"
    message = f"{display}, te acorduerdo: {text}"
    try:
        send_msg(chat_id, message)
    except Exception as e:
        print(f"reminder_scheduler: failed to send reminder: {e}")
    finally:
        try:
            redis_client.delete(key)
        except Exception:
            pass


def schedule_reminder(
    chat_id: str,
    text: str,
    delay_seconds: int,
    user_name: str = "",
) -> Optional[str]:
    if delay_seconds < 1:
        return None
    if delay_seconds > 86400 * 30:
        return None

    scheduler = get_scheduler()
    if scheduler is None:
        return None

    reminder_id = str(uuid.uuid4())[:8]

    redis_client = _get_redis()
    if redis_client is not None:
        _store_reminder_meta(redis_client, reminder_id, chat_id, text, user_name)

    run_date = datetime.now(timezone.utc).timestamp() + delay_seconds
    run_date_dt = datetime.fromtimestamp(run_date, tz=timezone.utc)

    scheduler.add_job(
        _fire_reminder,
        "date",
        run_date=run_date_dt,
        id=f"reminder_{reminder_id}",
        args=[reminder_id],
        replace_existing=True,
    )

    return reminder_id


def list_reminders(chat_id: str) -> List[Dict[str, Any]]:
    scheduler = get_scheduler()
    if scheduler is None:
        return []

    redis_client = _get_redis()
    if redis_client is None:
        return []

    results = []
    prefix = f"{REMINDER_REDIS_PREFIX}"
    try:
        for key_bytes in redis_client.scan_iter(f"{prefix}*"):
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

            reminder_id = data.get("id", "")
            job_id = f"reminder_{reminder_id}"
            try:
                job = scheduler.get_job(job_id)
            except Exception:
                job = None
            if job is None:
                continue

            next_run = job.next_run_time
            results.append(
                {
                    "id": reminder_id,
                    "text": data.get("text", ""),
                    "user_name": data.get("user_name", ""),
                    "next_run": str(next_run) if next_run else "unknown",
                }
            )
    except Exception as e:
        print(f"list_reminders error: {e}")

    return results


def parse_delay(text: str) -> Optional[int]:
    """Parse a natural language delay string into seconds.

    Supports: '5 min', '2 horas', '30 minutos', '1 dia', '1h', '30m', '2d'
    """
    import re

    text = text.lower().strip()
    if not text:
        return None

    patterns = [
        (r"(\d+)\s*(?:dia|dias|días|día)\b", 86400),
        (r"(\d+)\s*(?:hora|horas|h)\b", 3600),
        (r"(\d+)\s*(?:minuto|minutos|min|m)\b", 60),
        (r"(\d+)\s*(?:segundo|segundos|seg|s)\b", 1),
    ]

    total = 0
    matched = False
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, text):
            try:
                total += int(match.group(1)) * multiplier
                matched = True
            except (ValueError, TypeError):
                pass

    if matched and total > 0:
        return total

    try:
        num = int(text)
        if 1 <= num <= 1440:
            return num * 60
    except ValueError:
        pass

    return None
