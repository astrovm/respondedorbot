"""Shared Redis JSON helpers."""

from __future__ import annotations

import json
from typing import Any, Optional

import redis

__all__ = ["redis_get_json", "redis_setex_json", "redis_set_json"]


def redis_get_json(redis_client: redis.Redis, key: str) -> Optional[Any]:
    """Fetch ``key`` from Redis and decode JSON into Python objects."""
    try:
        data = redis_client.get(key)
        if not data:
            return None
        return json.loads(str(data))
    except Exception:
        return None


def redis_setex_json(redis_client: redis.Redis, key: str, ttl: int, value: Any) -> bool:
    """Store JSON under ``key`` with a TTL, returning success boolean."""
    try:
        return bool(redis_client.setex(key, ttl, json.dumps(value)))
    except Exception:
        return False


def redis_set_json(redis_client: redis.Redis, key: str, value: Any) -> bool:
    """Store JSON under ``key`` without TTL; return success boolean."""
    try:
        return bool(redis_client.set(key, json.dumps(value)))
    except Exception:
        return False
