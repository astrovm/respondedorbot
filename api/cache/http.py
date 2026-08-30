"""Cache external HTTP responses with stale-data recovery."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Hashable, Mapping
from functools import lru_cache
from logging import Logger
from typing import Any, Protocol, cast

import redis

from api.core.rust_bridge import load_rust_bridge
from api.core.rust_redis import redis_endpoint_from_env
from api.services.maintenance import (
    REQUEST_CACHE_HISTORY_TTL,
    request_cache_history_key,
    request_cache_key,
    request_cache_ttl,
)

RedisFactory = Callable[[], redis.Redis]
RedisJsonGetter = Callable[[redis.Redis, str], Any]
RedisJsonSetter = Callable[..., bool]
HistoryGetter = Callable[[int, str, redis.Redis], Any]
HttpGetter = Callable[..., Any]
AdminReporter = Callable[[str, Exception | None, dict[str, Any] | None], None]


class _RustJsonCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, *, ex: int | None = None) -> bool: ...


class _RustJsonCacheModule(Protocol):
    def RedisJsonCache(
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _RustJsonCache: ...


def _load_rust_request_cache() -> _RustJsonCacheModule | None:
    module = load_rust_bridge("RUST_REQUEST_CACHE_IO_ENABLED")
    if module is None:
        return None
    return cast(_RustJsonCacheModule, module)


@lru_cache(maxsize=8)
def _cached_rust_request_cache(
    module: Hashable,
    host: str,
    port: int,
    password: str | None,
) -> _RustJsonCache:
    return cast(_RustJsonCacheModule, module).RedisJsonCache(host, port, password)


def _request_cache_client(redis_factory: RedisFactory) -> redis.Redis:
    module = _load_rust_request_cache()
    if module is None:
        return redis_factory()
    host, port, password = redis_endpoint_from_env()
    cache = _cached_rust_request_cache(
        cast(Hashable, module),
        host,
        port,
        password,
    )
    return cast(redis.Redis, cache)


def cached_request(
    api_url: str,
    parameters: Mapping[str, Any] | None,
    headers: Mapping[str, Any] | None,
    expiration_time: int,
    *,
    hourly_cache: bool,
    history_hours: int | bool,
    verify_ssl: bool,
    redis_factory: RedisFactory,
    redis_get_json: RedisJsonGetter,
    redis_set_json: RedisJsonSetter,
    get_history: HistoryGetter,
    http_get: HttpGetter,
    admin_report: AdminReporter,
    logger: Logger,
) -> dict[str, Any] | None:
    """Cache an outbound JSON HTTP request by payload and TTL."""
    try:
        arguments = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
        }
        request_hash = hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()

        redis_client = _request_cache_client(redis_factory)
        redis_response = redis_get_json(
            redis_client, request_cache_key(request_hash)
        )
        cache_history = (
            get_history(int(history_hours), request_hash, redis_client)
            if history_hours
            else None
        )
        timestamp = int(time.time())

        def make_request() -> dict[str, Any]:
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    response = http_get(
                        api_url,
                        params=parameters,
                        headers=headers,
                        timeout=5,
                        verify=verify_ssl,
                    )
                    response.raise_for_status()
                    redis_value = {
                        "timestamp": timestamp,
                        "data": json.loads(response.text),
                    }
                    redis_set_json(
                        redis_client,
                        request_cache_key(request_hash),
                        redis_value,
                        ttl=request_cache_ttl(expiration_time),
                    )
                    if hourly_cache:
                        current_hour = time.strftime("%Y-%m-%d-%H")
                        hourly_key = request_cache_history_key(
                            current_hour, request_hash
                        )
                        if redis_client.get(hourly_key) is None:
                            redis_set_json(
                                redis_client,
                                hourly_key,
                                redis_value,
                                ttl=REQUEST_CACHE_HISTORY_TTL,
                            )
                    if cache_history is not None:
                        redis_value["history"] = cache_history
                    return redis_value
                except Exception as error:
                    last_error = error
                    if attempt == 0:
                        time.sleep(0.5)
            raise last_error or RuntimeError("request failed")

        if redis_response is None:
            try:
                return make_request()
            except Exception as error:
                logger.warning(
                    "cache request error url=%s error=%s", api_url, error
                )
                return None

        cached_data = cast(dict[str, Any], redis_response)
        cache_age = timestamp - int(cached_data["timestamp"])
        if cache_history is not None:
            cached_data["history"] = cache_history

        if cache_age <= expiration_time:
            return cached_data

        try:
            return make_request()
        except Exception as error:
            logger.warning("cache update error url=%s error=%s", api_url, error)
            return cached_data
    except Exception as error:
        error_context = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
            "expiration_time": expiration_time,
        }
        error_message = f"Error in cached_requests: {error!s}"
        print(error_message)
        admin_report(error_message, error, error_context)
        return None
