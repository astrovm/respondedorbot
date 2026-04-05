from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Optional

import redis

from api.config import config_redis
from api.services import credits_db

CHAT_STATE_TTL = 30 * 24 * 60 * 60
GIPHY_STALE_TTL = 7 * 24 * 60 * 60
REQUEST_CACHE_HISTORY_TTL = 3 * 24 * 60 * 60
REQUEST_CACHE_MIN_TTL = 60
LAST_SUCCESS_MIN_TTL = 24 * 60 * 60
AI_LEDGER_RETENTION_DAYS = 30
REDIS_MAXMEMORY = str(os.environ.get("REDIS_MAXMEMORY") or "256mb").strip() or "256mb"
REDIS_MAXMEMORY_POLICY = (
    str(os.environ.get("REDIS_MAXMEMORY_POLICY") or "allkeys-lru").strip()
    or "allkeys-lru"
)

_LEGACY_REQUEST_CACHE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_REQUEST_CACHE_HISTORY_KEY_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}[0-9a-f]{64}$"
)


def request_cache_key(request_hash: str) -> str:
    return f"request_cache:{request_hash}"


def request_cache_history_key(hour_key: str, request_hash: str) -> str:
    return f"request_cache_history:{hour_key}:{request_hash}"


def request_cache_ttl(expiration_time: int) -> int:
    return max(REQUEST_CACHE_MIN_TTL, int(expiration_time or 0))


def last_success_ttl(ttl: int, stale_grace: int) -> int:
    return int(ttl or 0) + int(stale_grace or 0)


def iter_legacy_cache_keys(redis_client: redis.Redis) -> Iterable[str]:
    for raw_key in redis_client.scan_iter():
        key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
        if _LEGACY_REQUEST_CACHE_KEY_RE.fullmatch(key):
            yield key
            continue
        if _LEGACY_REQUEST_CACHE_HISTORY_KEY_RE.fullmatch(key):
            yield key


def prune_redis_growth(
    redis_client: redis.Redis,
    *,
    legacy_cache_keys: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    expired_keys = 0
    deleted_keys = 0

    ttl_targets = (
        ("giphy_pool_stale:*", GIPHY_STALE_TTL),
        ("chat_history:*", CHAT_STATE_TTL),
        ("chat_message_ids:*", CHAT_STATE_TTL),
    )
    for pattern, ttl in ttl_targets:
        for raw_key in redis_client.scan_iter(match=pattern):
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            try:
                current_ttl = int(redis_client.ttl(key))
            except Exception:
                continue
            if current_ttl == -1 and redis_client.expire(key, ttl):
                expired_keys += 1

    stale_keys = list(legacy_cache_keys or iter_legacy_cache_keys(redis_client))
    if stale_keys:
        deleted_keys = int(redis_client.delete(*stale_keys) or 0)

    return {
        "expired_keys": expired_keys,
        "deleted_keys": deleted_keys,
    }


def apply_redis_memory_policy(redis_client: redis.Redis) -> Dict[str, Any]:
    redis_client.config_set("maxmemory", REDIS_MAXMEMORY)
    redis_client.config_set("maxmemory-policy", REDIS_MAXMEMORY_POLICY)
    config = redis_client.config_get("maxmemory", "maxmemory-policy")
    return {
        "maxmemory": config.get("maxmemory"),
        "maxmemory_policy": config.get("maxmemory-policy"),
    }


def run_maintenance() -> Dict[str, Any]:
    redis_client = config_redis()
    redis_config = apply_redis_memory_policy(redis_client)
    redis_cleanup = prune_redis_growth(redis_client)
    if credits_db.is_configured():
        ledger_cleanup = credits_db.purge_expired_ai_ledger_events(
            AI_LEDGER_RETENTION_DAYS
        )
    else:
        ledger_cleanup = {"skipped": True, "reason": "postgres not configured"}
    return {
        "redis": {**redis_cleanup, **redis_config},
        "ledger": ledger_cleanup,
    }
