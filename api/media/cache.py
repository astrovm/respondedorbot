"""Redis cache for expensive media transcription and image descriptions."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Protocol, cast

import redis

from api.core.config_runtime import ConfigRuntime
from api.core.rust_bridge import load_rust_bridge
from api.core.rust_redis import redis_endpoint_from_env

RedisFactory = Callable[[], redis.Redis]


class _RustMediaCache(Protocol):
    def redis_media_cache_get(
        self,
        host: str,
        port: int,
        password: str | None,
        prefix: str,
        file_id: str,
    ) -> str | None: ...

    def redis_media_cache_set(
        self,
        host: str,
        port: int,
        password: str | None,
        prefix: str,
        file_id: str,
        text: str,
        ttl: int,
    ) -> None: ...


def _load_rust_media_cache() -> _RustMediaCache | None:
    module = load_rust_bridge("RUST_MEDIA_CACHE_ENABLED")
    if module is None:
        return None
    return cast(_RustMediaCache, module)


def get_cached_media(
    prefix: str,
    file_id: str,
    *,
    redis_factory: RedisFactory,
    logger: Logger,
) -> str | None:
    """Read one cached media result without making Redis failures fatal."""

    rust = _load_rust_media_cache()
    if rust is not None:
        try:
            host, port, password = redis_endpoint_from_env()
            return rust.redis_media_cache_get(host, port, password, prefix, file_id)
        except Exception:
            logger.exception("Error getting cached %s through Rust", prefix)
            return None

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
    """Store one media result; processing can continue if Redis is down."""

    rust = _load_rust_media_cache()
    if rust is not None:
        try:
            host, port, password = redis_endpoint_from_env()
            rust.redis_media_cache_set(
                host,
                port,
                password,
                prefix,
                file_id,
                text,
                ttl,
            )
        except Exception:
            logger.exception("Error caching %s through Rust", prefix)
        return

    cache_key = f"{prefix}:{file_id}"
    try:
        redis_factory().setex(cache_key, ttl, text)
    except Exception:
        logger.exception("Error caching %s", prefix)


class MediaCacheService:
    """Name and cache transcription/description entries consistently."""

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
