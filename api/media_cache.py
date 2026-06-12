from __future__ import annotations

from collections.abc import Callable
from logging import Logger

import redis

RedisFactory = Callable[[], redis.Redis]


def get_cached_media(
    prefix: str,
    file_id: str,
    *,
    redis_factory: RedisFactory,
    logger: Logger,
) -> str | None:
    cache_key = f"{prefix}:{file_id}"
    try:
        cached_value = redis_factory().get(cache_key)
        return str(cached_value) if cached_value else None
    except Exception:
        logger.exception("Error getting cached %s", prefix)
        return None


def cache_media(
    prefix: str,
    file_id: str,
    text: str,
    ttl: int,
    *,
    redis_factory: RedisFactory,
    logger: Logger,
) -> None:
    cache_key = f"{prefix}:{file_id}"
    try:
        redis_factory().setex(cache_key, ttl, text)
    except Exception:
        logger.exception("Error caching %s", prefix)
