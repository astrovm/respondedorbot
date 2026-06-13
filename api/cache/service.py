"""HTTP response caching used by market and external-data services."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from logging import Logger
from typing import Any, Dict, Mapping, Optional, Union

import redis

from api.admin.service import AdminService
from api.cache.http import cached_request
from api.core.config_runtime import ConfigRuntime
from api.services import http_client
from api.services.maintenance import request_cache_history_key
from api.services.redis_helpers import redis_get_json, redis_set_json


class CacheService:
    """Combine HTTP fetching, Redis storage, and stale-history lookup.

    Other services only describe the request and cache lifetime. This class
    owns the mechanics of reading Redis, making the HTTP call, and reporting
    failures.
    """

    def __init__(
        self,
        *,
        config: ConfigRuntime,
        admin: AdminService,
        logger: Logger,
    ) -> None:
        self._config = config
        self._admin = admin
        self._logger = logger

    def get_history(
        self,
        hours_ago: int,
        request_hash: str,
        redis_client: redis.Redis,
    ) -> Optional[Dict[str, Any]]:
        """Load an hourly snapshot used for comparisons and stale fallback."""

        timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime(
            "%Y-%m-%d-%H"
        )
        cached_data = redis_client.get(
            request_cache_history_key(timestamp, request_hash)
        )
        if cached_data is None:
            return None
        cache_history = json.loads(cached_data)
        if isinstance(cache_history, dict) and "timestamp" in cache_history:
            return cache_history
        return None

    def request(
        self,
        api_url: str,
        parameters: Optional[Mapping[str, Any]],
        headers: Optional[Mapping[str, Any]],
        expiration_time: int,
        hourly_cache: bool = False,
        get_history: Union[int, bool] = False,
        verify_ssl: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch JSON-like data while reusing a valid Redis cache entry."""

        return cached_request(
            api_url,
            parameters,
            headers,
            expiration_time,
            hourly_cache=hourly_cache,
            history_hours=get_history,
            verify_ssl=verify_ssl,
            redis_factory=self._config.redis,
            redis_get_json=redis_get_json,
            redis_set_json=redis_set_json,
            get_history=self.get_history,
            http_get=http_client.get,
            admin_report=self._admin.report,
            logger=self._logger,
        )


__all__ = ["CacheService"]
