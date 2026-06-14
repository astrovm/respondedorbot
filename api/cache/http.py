"""Cache external HTTP responses with stale-data recovery."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from logging import Logger
from typing import Any, cast

import redis

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

        redis_client = redis_factory()
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
