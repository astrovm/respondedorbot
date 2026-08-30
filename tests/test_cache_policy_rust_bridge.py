from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from api.services import maintenance, stale_cache


@pytest.fixture(autouse=True)
def _use_python_stale_cache_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cache-policy tests focused on the pure decision boundary."""

    monkeypatch.setattr(stale_cache, "_load_rust_stale_cache", lambda: None)


class _FakeRustCachePolicy:
    def __init__(self, *, decision: str = "stale", fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail
        self.calls: list[tuple[int | None, int, int, int]] = []

    def _check(self) -> None:
        if self.fail:
            raise ValueError("synthetic cache-policy failure")

    def evaluate_cache_policy(
        self,
        timestamp: int | None,
        now: int,
        ttl: int,
        stale_grace: int,
    ) -> str:
        self._check()
        self.calls.append((timestamp, now, ttl, stale_grace))
        return self.decision

    def request_cache_key(self, request_hash: str) -> str:
        self._check()
        return f"rust-request:{request_hash}"

    def request_cache_history_key(self, hour_key: str, request_hash: str) -> str:
        self._check()
        return f"rust-history:{hour_key}:{request_hash}"

    def request_cache_ttl(self, expiration_time: int) -> int:
        self._check()
        return expiration_time + 1

    def last_success_ttl(self, ttl: int, stale_grace: int) -> int:
        self._check()
        return ttl + stale_grace + 1


class _FakeRedis:
    def __init__(self, cached: dict[str, Any]) -> None:
        self.cached = json.dumps(cached)
        self.lock_calls: list[tuple[str, str, bool, int]] = []
        self.stores: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str:
        return self.cached

    def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        self.lock_calls.append((key, value, nx, ex))
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.stores.append((key, ttl, value))


def test_stale_cache_uses_rust_decision_without_duplicating_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustCachePolicy()
    redis_client = _FakeRedis({"timestamp": 80, "value": "cached"})
    scheduled: list[Callable[[], None]] = []
    refresh_calls = 0

    def refresh() -> str:
        nonlocal refresh_calls
        refresh_calls += 1
        return "new"

    monkeypatch.setattr(stale_cache, "_load_rust_cache_policy", lambda: rust)
    result = stale_cache.StaleCache(redis_client=redis_client, now=lambda: 100).get(
        key="value",
        lock_key="lock",
        ttl=10,
        stale_grace=60,
        refresh=refresh,
        schedule_refresh=scheduled.append,
    )

    assert result == stale_cache.StaleCacheResult(value="cached", status="stale")
    assert rust.calls == [(80, 100, 10, 60)]
    assert redis_client.lock_calls == [("lock", "1", True, 10)]
    assert len(scheduled) == 1
    assert refresh_calls == 0
    scheduled[0]()
    assert refresh_calls == 1
    assert len(redis_client.stores) == 1


def test_stale_cache_falls_back_after_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = _FakeRedis({"timestamp": 95, "value": "cached"})
    monkeypatch.setattr(
        stale_cache,
        "_load_rust_cache_policy",
        lambda: _FakeRustCachePolicy(fail=True),
    )

    result = stale_cache.StaleCache(redis_client=redis_client, now=lambda: 100).get(
        key="value",
        lock_key="lock",
        ttl=10,
        stale_grace=60,
        refresh=lambda: "new",
        schedule_refresh=lambda callback: callback(),
    )

    assert result == stale_cache.StaleCacheResult(value="cached", status="fresh")
    assert redis_client.lock_calls == []
    assert redis_client.stores == []


def test_maintenance_cache_helpers_use_rust_and_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustCachePolicy()
    monkeypatch.setattr(maintenance, "_load_rust_cache_policy", lambda: rust)

    assert maintenance.request_cache_key("abc") == "rust-request:abc"
    assert maintenance.request_cache_history_key("hour", "abc") == "rust-history:hour:abc"
    assert maintenance.request_cache_ttl(10) == 11
    assert maintenance.last_success_ttl(10, 60) == 71

    failing = _FakeRustCachePolicy(fail=True)
    monkeypatch.setattr(maintenance, "_load_rust_cache_policy", lambda: failing)
    assert maintenance.request_cache_key("abc") == "request_cache:abc"
    assert maintenance.request_cache_history_key("hour", "abc") == "request_cache_history:hour:abc"
    assert maintenance.request_cache_ttl(10) == 60
    assert maintenance.last_success_ttl(10, 60) == 70
