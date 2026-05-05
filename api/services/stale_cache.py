from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

CacheStatus = Literal["fresh", "stale", "miss"]


@dataclass(frozen=True)
class StaleCacheResult:
    value: Any
    status: CacheStatus


class StaleCache:
    def __init__(self, *, redis_client: Any, now: Callable[[], int] | None = None):
        self._redis_client = redis_client
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
            age = now - int(cached.get("timestamp", 0))
            value = cached.get("value")
            if age <= ttl:
                return StaleCacheResult(value=value, status="fresh")
            if age <= ttl + stale_grace:
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
        return bool(self._redis_client.set(lock_key, "1", nx=True, ex=ttl))

    def _refresh_and_store(self, key: str, ttl: int, refresh: Callable[[], Any]) -> None:
        try:
            self._store(key, ttl, refresh())
        except Exception:
            logger.exception("stale_cache: background refresh failed key=%s", key)
