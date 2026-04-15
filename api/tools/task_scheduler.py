"""Task scheduler using APScheduler with Redis job store.

Unified scheduler for one-shot and recurring tasks per chat.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_scheduler_instance: Optional[Any] = None

TASK_REDIS_PREFIX = "task:data:"


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
        _scheduler_instance = BackgroundScheduler(jobstores=jobstores)
        _scheduler_instance.start()
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


def _fire_task(task_id: str) -> None:
    redis_client = _get_redis()
    if redis_client is None:
        print(f"task_scheduler: no redis, cannot fire {task_id}")
        return

    key = f"{TASK_REDIS_PREFIX}{task_id}"
    raw = redis_client.get(key)
    if not raw:
        return

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return

    chat_id = str(data.get("chat_id", ""))
    text = str(data.get("text", ""))
    user_name = str(data.get("user_name", ""))
    interval = data.get("interval_seconds")

    if not chat_id or not text:
        return

    from api.index import send_msg

    display = f"@{user_name}" if user_name else "che"

    if interval:
        try:
            from api.index import ask_ai

            response = ask_ai(
                [{"role": "user", "content": text}],
                response_meta={},
                enable_web_search=True,
                chat_id=chat_id,
                user_name=user_name,
            )
            if response:
                send_msg(chat_id, f"{display}, tarea programada: {response}")
        except Exception as e:
            print(f"task_scheduler: recurring task {task_id} failed: {e}")
    else:
        try:
            send_msg(chat_id, f"{display}, te recuerdo: {text}")
        except Exception as e:
            print(f"task_scheduler: one-shot task {task_id} failed: {e}")
        finally:
            try:
                redis_client.delete(key)
            except Exception:
                pass


def schedule_task(
    chat_id: str,
    text: str,
    delay_seconds: Optional[int] = None,
    interval_seconds: Optional[int] = None,
    user_name: str = "",
) -> Optional[str]:
    if not delay_seconds and not interval_seconds:
        return None

    scheduler = get_scheduler()
    if scheduler is None:
        return None

    task_id = str(uuid.uuid4())[:8]

    redis_client = _get_redis()
    if redis_client is not None:
        data = json.dumps(
            {
                "id": task_id,
                "chat_id": str(chat_id),
                "text": text,
                "user_name": user_name,
                "interval_seconds": interval_seconds,
            }
        )
        ttl = 86400 * 90 if interval_seconds else 86400 * 30
        redis_client.setex(f"{TASK_REDIS_PREFIX}{task_id}", ttl, data)

    if interval_seconds:
        scheduler.add_job(
            _fire_task,
            "interval",
            seconds=interval_seconds,
            id=f"task_{task_id}",
            args=[task_id],
            replace_existing=True,
        )
    elif delay_seconds:
        run_date = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + delay_seconds,
            tz=timezone.utc,
        )
        scheduler.add_job(
            _fire_task,
            "date",
            run_date=run_date,
            id=f"task_{task_id}",
            args=[task_id],
            replace_existing=True,
        )

    return task_id


def list_tasks(chat_id: str) -> List[Dict[str, Any]]:
    scheduler = get_scheduler()
    if scheduler is None:
        return []

    redis_client = _get_redis()
    if redis_client is None:
        return []

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
            job_id = f"task_{task_id}"
            try:
                job = scheduler.get_job(job_id)
            except Exception:
                job = None
            if job is None:
                continue

            next_run = job.next_run_time
            interval = data.get("interval_seconds")
            results.append(
                {
                    "id": task_id,
                    "text": data.get("text", ""),
                    "user_name": data.get("user_name", ""),
                    "interval_seconds": interval,
                    "next_run": str(next_run) if next_run else "unknown",
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
