"""Webhook retry, idempotency, and persisted reservation helpers."""

from __future__ import annotations

from contextvars import ContextVar, Token
from os import environ
import re
import threading
import time
from typing import Any, Mapping, Optional, Tuple

import redis

from api.services.redis_helpers import redis_get_json, redis_setex_json
from api.utils import now_utc_iso


_webhook_request_started_at: ContextVar[Optional[float]] = ContextVar(
    "webhook_request_started_at", default=None
)
_webhook_operation_key: ContextVar[Optional[str]] = ContextVar(
    "webhook_operation_key", default=None
)
_webhook_redis_client: ContextVar[Optional[redis.Redis]] = ContextVar(
    "webhook_redis_client", default=None
)
_webhook_force_paid_retry: ContextVar[bool] = ContextVar(
    "webhook_force_paid_retry", default=False
)

WEBHOOK_IDEMPOTENCY_PREFIX = "webhook:idempotency"


class ForceWebhookRetry(RuntimeError):
    """Abort current webhook processing so Telegram retries the same update."""


WebhookContextTokens = Tuple[Token, Token, Token, Token]
WebhookLockHeartbeat = Tuple[threading.Event, Optional[threading.Thread]]


def _get_webhook_max_runtime_seconds() -> float:
    raw_value = str(environ.get("WEBHOOK_MAX_RUNTIME_SECONDS") or "60").strip()
    try:
        return max(1.0, float(raw_value))
    except (TypeError, ValueError):
        return 60.0


def _get_webhook_retry_safety_margin_seconds() -> float:
    raw_value = str(environ.get("WEBHOOK_RETRY_SAFETY_MARGIN_SECONDS") or "30").strip()
    try:
        return max(1.0, float(raw_value))
    except (TypeError, ValueError):
        return 30.0


def _get_webhook_idempotency_ttl_seconds() -> int:
    raw_value = str(environ.get("WEBHOOK_IDEMPOTENCY_TTL_SECONDS") or "").strip()
    if raw_value:
        try:
            return max(10, int(raw_value))
        except (TypeError, ValueError):
            pass
    return max(180, int(_get_webhook_max_runtime_seconds()) + 60)


def _get_webhook_force_paid_retry_ttl_seconds() -> int:
    raw_value = str(environ.get("WEBHOOK_FORCE_PAID_RETRY_TTL_SECONDS") or "").strip()
    if raw_value:
        try:
            return max(10, int(raw_value))
        except (TypeError, ValueError):
            pass
    return max(
        300,
        _get_webhook_idempotency_ttl_seconds(),
        int(_get_webhook_max_runtime_seconds()) + 120,
    )


def _get_webhook_lock_refresh_interval_seconds() -> float:
    ttl_seconds = float(_get_webhook_idempotency_ttl_seconds())
    return max(5.0, min(30.0, ttl_seconds / 3.0))


def _set_webhook_request_context(
    *,
    request_started_at: float,
    operation_key: Optional[str],
    redis_client: Optional[redis.Redis],
    force_paid_retry: bool,
) -> WebhookContextTokens:
    return (
        _webhook_request_started_at.set(request_started_at),
        _webhook_operation_key.set(operation_key),
        _webhook_redis_client.set(redis_client),
        _webhook_force_paid_retry.set(bool(force_paid_retry)),
    )


def _restore_webhook_request_context(tokens: WebhookContextTokens) -> None:
    started_token, operation_token, redis_token, force_paid_token = tokens
    _webhook_request_started_at.reset(started_token)
    _webhook_operation_key.reset(operation_token)
    _webhook_redis_client.reset(redis_token)
    _webhook_force_paid_retry.reset(force_paid_token)


def _get_webhook_time_remaining_seconds() -> Optional[float]:
    started_at = _webhook_request_started_at.get()
    if started_at is None:
        return None
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    return max(0.0, _get_webhook_max_runtime_seconds() - elapsed_seconds)


def _webhook_processing_lock_key(operation_key: str) -> str:
    return f"{WEBHOOK_IDEMPOTENCY_PREFIX}:{operation_key}:lock"


def _webhook_completed_key(operation_key: str) -> str:
    return f"{WEBHOOK_IDEMPOTENCY_PREFIX}:{operation_key}:completed"


def _webhook_force_paid_retry_key(operation_key: str) -> str:
    return f"{WEBHOOK_IDEMPOTENCY_PREFIX}:{operation_key}:force_paid_retry"


def _webhook_reservation_key(operation_key: str, usage_tag: str) -> str:
    safe_usage_tag = re.sub(r"[^a-zA-Z0-9:_-]", "_", str(usage_tag or "ai_usage"))
    return f"{WEBHOOK_IDEMPOTENCY_PREFIX}:{operation_key}:reservation:{safe_usage_tag}"


def _extract_webhook_operation_key(request_json: Mapping[str, Any]) -> Optional[str]:
    callback_query = request_json.get("callback_query")
    if isinstance(callback_query, Mapping):
        callback_id = str(callback_query.get("id") or "").strip()
        if callback_id:
            return f"callback:{callback_id}"

    pre_checkout_query = request_json.get("pre_checkout_query")
    if isinstance(pre_checkout_query, Mapping):
        query_id = str(pre_checkout_query.get("id") or "").strip()
        if query_id:
            return f"pre_checkout:{query_id}"

    message = request_json.get("message")
    if isinstance(message, Mapping):
        chat = message.get("chat") or {}
        if isinstance(chat, Mapping):
            chat_id = chat.get("id")
            message_id = message.get("message_id")
            if chat_id is not None and message_id is not None:
                return f"message:{chat_id}:{message_id}"

    return None


def _mark_webhook_completed(redis_client: redis.Redis, operation_key: str) -> None:
    redis_setex_json(
        redis_client,
        _webhook_completed_key(operation_key),
        _get_webhook_idempotency_ttl_seconds(),
        {"status": "completed", "completed_at": now_utc_iso()},
    )


def _mark_webhook_force_paid_retry_pending(
    redis_client: redis.Redis,
    operation_key: str,
    *,
    reason: str,
) -> None:
    redis_setex_json(
        redis_client,
        _webhook_force_paid_retry_key(operation_key),
        _get_webhook_force_paid_retry_ttl_seconds(),
        {
            "status": "force_paid_retry_pending",
            "reason": reason,
            "created_at": now_utc_iso(),
        },
    )


def _has_webhook_force_paid_retry_pending(
    redis_client: Optional[redis.Redis],
    operation_key: Optional[str],
) -> bool:
    if redis_client is None or not operation_key:
        return False
    return bool(redis_get_json(redis_client, _webhook_force_paid_retry_key(operation_key)))


def _clear_webhook_force_paid_retry_pending(
    redis_client: Optional[redis.Redis],
    operation_key: Optional[str],
) -> None:
    if redis_client is None or not operation_key:
        return
    try:
        redis_client.delete(_webhook_force_paid_retry_key(operation_key))
    except Exception:
        return


def _acquire_webhook_processing_lock(
    redis_client: redis.Redis,
    operation_key: str,
    owner_token: str,
) -> str:
    completed = redis_get_json(redis_client, _webhook_completed_key(operation_key))
    if completed:
        return "completed"

    try:
        acquired = bool(
            redis_client.set(
                _webhook_processing_lock_key(operation_key),
                owner_token,
                ex=_get_webhook_idempotency_ttl_seconds(),
                nx=True,
            )
        )
    except Exception:
        return "unavailable"

    if acquired:
        return "acquired"

    completed = redis_get_json(redis_client, _webhook_completed_key(operation_key))
    if completed:
        return "completed"
    return "in_flight"


def _release_webhook_processing_lock(
    redis_client: Optional[redis.Redis],
    operation_key: Optional[str],
    owner_token: Optional[str],
) -> None:
    if redis_client is None or not operation_key or not owner_token:
        return
    lock_key = _webhook_processing_lock_key(operation_key)
    try:
        current_owner = redis_client.get(lock_key)
        if current_owner == owner_token:
            redis_client.delete(lock_key)
    except Exception:
        return


def _refresh_webhook_processing_lock(
    redis_client: Optional[redis.Redis],
    operation_key: Optional[str],
    owner_token: Optional[str],
) -> bool:
    if redis_client is None or not operation_key or not owner_token:
        return False
    lock_key = _webhook_processing_lock_key(operation_key)
    try:
        current_owner = redis_client.get(lock_key)
        if current_owner != owner_token:
            return False
        redis_client.expire(lock_key, _get_webhook_idempotency_ttl_seconds())
        return True
    except Exception:
        return False


def _start_webhook_processing_lock_heartbeat(
    redis_client: Optional[redis.Redis],
    operation_key: Optional[str],
    owner_token: Optional[str],
) -> WebhookLockHeartbeat:
    stop_event = threading.Event()
    if redis_client is None or not operation_key or not owner_token:
        return stop_event, None

    interval_seconds = _get_webhook_lock_refresh_interval_seconds()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_seconds):
            if not _refresh_webhook_processing_lock(
                redis_client,
                operation_key,
                owner_token,
            ):
                break

    thread = threading.Thread(
        target=_heartbeat,
        name=f"webhook-lock-heartbeat:{operation_key}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _stop_webhook_processing_lock_heartbeat(heartbeat: WebhookLockHeartbeat) -> None:
    stop_event, thread = heartbeat
    stop_event.set()
    if thread is not None:
        thread.join(timeout=0.2)


def _load_persisted_webhook_reservation(usage_tag: str) -> Optional[Mapping[str, Any]]:
    operation_key = _webhook_operation_key.get()
    redis_client = _webhook_redis_client.get()
    if redis_client is None or not operation_key:
        return None
    cached = redis_get_json(redis_client, _webhook_reservation_key(operation_key, usage_tag))
    if not isinstance(cached, Mapping):
        return None
    return cached


def _persist_webhook_reservation(usage_tag: str, reservation: Mapping[str, Any]) -> None:
    operation_key = _webhook_operation_key.get()
    redis_client = _webhook_redis_client.get()
    if redis_client is None or not operation_key:
        return
    redis_setex_json(
        redis_client,
        _webhook_reservation_key(operation_key, usage_tag),
        _get_webhook_force_paid_retry_ttl_seconds(),
        dict(reservation),
    )


def _clear_persisted_webhook_reservation(usage_tag: str) -> None:
    operation_key = _webhook_operation_key.get()
    redis_client = _webhook_redis_client.get()
    if redis_client is None or not operation_key:
        return
    try:
        redis_client.delete(_webhook_reservation_key(operation_key, usage_tag))
    except Exception:
        return


def _maybe_abort_webhook_for_paid_retry(
    *,
    label: str,
    scope: str,
    account: str,
    free_account: str,
    has_paid_fallback: bool,
    reason: str,
) -> None:
    if scope != "compound" or account != free_account or not has_paid_fallback:
        return
    operation_key = _webhook_operation_key.get()
    redis_client = _webhook_redis_client.get()
    if redis_client is None or not operation_key:
        return
    remaining_seconds = _get_webhook_time_remaining_seconds()
    if remaining_seconds is None:
        return
    margin_seconds = _get_webhook_retry_safety_margin_seconds()
    if remaining_seconds > margin_seconds:
        return
    _mark_webhook_force_paid_retry_pending(
        redis_client,
        operation_key,
        reason=reason,
    )
    print(
        f"{label} aborting for paid retry due to low remaining time "
        f"on operation={operation_key} account={account} scope={scope} "
        f"remaining_seconds={remaining_seconds:.3f} margin_seconds={margin_seconds:.3f}"
    )
    raise ForceWebhookRetry(reason)


def _mark_webhook_paid_retry_preferred(
    *,
    scope: str,
    account: str,
    free_account: str,
    has_paid_fallback: bool,
    reason: str,
) -> None:
    if scope != "compound" or account != free_account or not has_paid_fallback:
        return
    operation_key = _webhook_operation_key.get()
    redis_client = _webhook_redis_client.get()
    if redis_client is None or not operation_key:
        return
    _mark_webhook_force_paid_retry_pending(
        redis_client,
        operation_key,
        reason=reason,
    )
