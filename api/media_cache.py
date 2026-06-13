from __future__ import annotations

from collections.abc import Callable
from logging import Logger

import redis

from api.config_runtime import ConfigRuntime

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


class MediaCacheService:
    def __init__(
        self,
        *,
        config: ConfigRuntime,
        logger: Logger,
        default_ttl: int,
    ) -> None:
        self._config = config
        self._logger = logger
        self._default_ttl = default_ttl

    def get(self, prefix: str, file_id: str) -> str | None:
        return get_cached_media(
            prefix,
            file_id,
            redis_factory=self._config.redis,
            logger=self._logger,
        )

    def set(
        self,
        prefix: str,
        file_id: str,
        text: str,
        ttl: int | None = None,
    ) -> None:
        cache_media(
            prefix,
            file_id,
            text,
            ttl if ttl is not None else self._default_ttl,
            redis_factory=self._config.redis,
            logger=self._logger,
        )

    def get_transcription(self, file_id: str) -> str | None:
        return self.get("audio_transcription", file_id)

    def cache_transcription(
        self,
        file_id: str,
        text: str,
        ttl: int | None = None,
    ) -> None:
        self.set(
            "audio_transcription",
            file_id,
            text,
            ttl if ttl is not None else self._default_ttl,
        )

    def get_description(self, file_id: str) -> str | None:
        return self.get("image_description", file_id)

    def cache_description(
        self,
        file_id: str,
        description: str,
        ttl: int | None = None,
    ) -> None:
        self.set(
            "image_description",
            file_id,
            description,
            ttl if ttl is not None else self._default_ttl,
        )


__all__ = ["MediaCacheService"]
