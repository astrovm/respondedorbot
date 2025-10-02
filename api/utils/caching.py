"""Lightweight helpers to work with in-memory caches that mirror Redis."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import time

__all__ = [
    "now_utc_iso",
    "update_local_cache",
    "local_cache_get",
]


def now_utc_iso() -> str:
    """Return the current UTC timestamp rendered as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def update_local_cache(
    cache_store: Dict[str, Any],
    value: Any,
    ttl: int,
    stale_grace: int,
    fetched_at: Optional[str] = None,
) -> None:
    """Populate a cache entry with its metadata in a consistent format."""

    now = time.time()
    cache_store["value"] = value
    cache_store["expires_at"] = now + max(ttl, 0)
    cache_store["stale_until"] = cache_store["expires_at"] + max(stale_grace, 0)
    cache_store["meta"] = {"fetched_at": fetched_at or now_utc_iso()}


def local_cache_get(
    cache_store: Dict[str, Any], allow_stale: bool = False
) -> Tuple[Optional[Any], bool, Dict[str, Any]]:
    """Return ``(value, is_fresh, meta)`` respecting TTL and stale windows."""

    value = cache_store.get("value")
    meta = cache_store.get("meta") or {}
    if value is None:
        return None, False, {}

    now = time.time()
    expires_at = cache_store.get("expires_at", 0.0)
    if now <= expires_at:
        return value, True, meta

    stale_until = cache_store.get("stale_until", 0.0)
    if allow_stale and now <= stale_until:
        return value, False, meta

    if now > stale_until:
        cache_store["value"] = None
        cache_store["meta"] = {}
    return None, False, {}
