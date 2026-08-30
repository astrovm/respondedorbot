"""Serve stale cache values while one worker refreshes them."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Literal, Optional, Protocol, cast

from api.core.rust_bridge import load_rust_bridge
from api.core.rust_redis import redis_endpoint_from_env

logger = logging.getLogger(__name__)

CacheStatus = Literal["fresh", "stale", "miss"]


class _RustCachePolicy(Protocol):
    def evaluate_cache_policy(
        self,
        cached_timestamp: int | None,
        now: int,
        ttl: int,
        stale_grace: int,
    ) -> str: ...


class _RustStaleCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def setex(self, key: str, ttl: int, value: str) -> bool: ...
    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool: ...


class _RustStaleCacheModule(Protocol):
    def RedisJsonCache(
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _RustStaleCache: ...


def _load_rust_cache_policy() -> _RustCachePolicy | None:
    module = load_rust_bridge("RUST_CACHE_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustCachePolicy, module)


def _load_rust_stale_cache() -> _RustStaleCacheModule | None:
    module = load_rust_bridge("RUST_STALE_CACHE_IO_ENABLED")
    if module is None:
        return None
    return cast(_RustStaleCacheModule, module)


@lru_cache(maxsize=8)
def _cached_rust_stale_cache(
    module: Hashable,
    host: str,
    port: int,
    password: str | None,
) -> _RustStaleCache:
    return cast(_RustStaleCacheModule, module).RedisJsonCache(host, port, password)


def _cache_decision(timestamp: int, now: int, ttl: int, stale_grace: int) -> str:
    rust = _load_rust_cache_policy()
    if rust is not None:
        try:
            return rust.evaluate_cache_policy(timestamp, now, ttl, stale_grace)
        except Exception:
            logger.exception("Rust stale-cache policy failed; using Python")
    age = now - timestamp
    if age <= ttl:
        return "fresh"
    if age <= ttl + stale_grace:
        return "stale"
    return "refresh_inline"


@dataclass(frozen=True)
class StaleCacheResult:
    value: Any
    status: CacheStatus


class StaleCache:
    def __init__(self, *, redis_client: Any, now: Callable[[], int] | None = None):
        module = _load_rust_stale_cache()
        if module is None:
            self._redis_client = redis_client
        else:
            host, port, password = redis_endpoint_from_env()
            self._redis_client = _cached_rust_stale_cache(
                cast(Hashable, module),
                host,
                port,
                password,
            )
        self._now = now or (lambda: int(time.time()))

    def get(
        self,
        *,
        key: str,
        lock_key: str,
        ttl: int,
        stale_grace: int,
        refresh: Callable[[], Any],
        schedule_refresh: Callable[[Callable[[], None]], None],
    ) -> StaleCacheResult:
        cached = self._load(key)
        now = self._now()
        if cached is not None:
            decision = _cache_decision(
                int(cached.get("timestamp", 0)),
                now,
                ttl,
                stale_grace,
            )
            value = cached.get("value")
            if decision == "fresh":
                return StaleCacheResult(value=value, status="fresh")
            if decision == "stale":
                # Serve stale immediately; one lock holder refreshes in background.
                if self._acquire_lock(lock_key, ttl):
                    schedule_refresh(
                        lambda: self._refresh_and_store(
                            key,
                            ttl + stale_grace,
                            refresh,
                        )
                    )
                return StaleCacheResult(value=value, status="stale")

        value = refresh()
        self._store(key, ttl + stale_grace, value)
        return StaleCacheResult(value=value, status="miss")

    def _load(self, key: str) -> Optional[dict[str, Any]]:
        raw = self._redis_client.get(key)
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            loaded = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("stale_cache: invalid payload key=%s", key)
            return None
        return loaded if isinstance(loaded, dict) else None

    def _store(self, key: str, ttl: int, value: Any) -> None:
        payload = {"timestamp": self._now(), "value": value}
        self._redis_client.setex(key, ttl, json.dumps(payload))

    def _acquire_lock(self, lock_key: str, ttl: int) -> bool:
        lock_ttl = min(30, ttl)
        return bool(self._redis_client.set(lock_key, "1", nx=True, ex=lock_ttl))

    def _refresh_and_store(self, key: str, ttl: int, refresh: Callable[[], Any]) -> None:
        try:
            self._store(key, ttl, refresh())
        except Exception:
            logger.exception("stale_cache: background refresh failed key=%s", key)
