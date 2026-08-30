from __future__ import annotations

import json

import pytest

from api.services import stale_cache


class _FailingPythonRedis:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"Python Redis owner must not run: {name}")


class _FakeRustCache:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.values: dict[str, str] = {}
        self.gets: list[str] = []
        self.setex_calls: list[tuple[str, int, str]] = []
        self.lock_calls: list[tuple[str, str, bool, int | None]] = []

    def get(self, key: str) -> str | None:
        if self.fail:
            raise ValueError("synthetic Rust Redis read failure")
        self.gets.append(key)
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        if self.fail:
            raise ValueError("synthetic Rust Redis write failure")
        self.setex_calls.append((key, ttl, value))
        self.values[key] = value
        return True

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        if self.fail:
            raise ValueError("synthetic Rust Redis lock failure")
        self.lock_calls.append((key, value, nx, ex))
        return True


class _FakeRustModule:
    def __init__(self, cache: _FakeRustCache) -> None:
        self.cache = cache
        self.endpoints: list[tuple[str, int, str | None]] = []

    def RedisJsonCache(
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _FakeRustCache:
        self.endpoints.append((host, port, password))
        return self.cache


def test_stale_cache_rust_path_owns_values_locks_and_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_cache = _FakeRustCache()
    rust_cache.values["rates"] = json.dumps({"timestamp": 100, "value": "old"})
    module = _FakeRustModule(rust_cache)
    stale_cache._cached_rust_stale_cache.cache_clear()
    monkeypatch.setattr(stale_cache, "_load_rust_stale_cache", lambda: module)
    monkeypatch.setattr(stale_cache, "_load_rust_cache_policy", lambda: None)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    scheduled: list[object] = []

    cache = stale_cache.StaleCache(redis_client=_FailingPythonRedis(), now=lambda: 120)
    result = cache.get(
        key="rates",
        lock_key="rates:lock",
        ttl=10,
        stale_grace=60,
        refresh=lambda: "new",
        schedule_refresh=lambda callback: scheduled.append(callback),
    )

    assert result == stale_cache.StaleCacheResult(value="old", status="stale")
    assert module.endpoints == [("redis.internal", 6380, "synthetic-password")]
    assert rust_cache.gets == ["rates"]
    assert rust_cache.lock_calls == [("rates:lock", "1", True, 10)]
    assert len(scheduled) == 1
    callback = scheduled[0]
    assert callable(callback)
    callback()
    assert len(rust_cache.setex_calls) == 1
    key, ttl, payload = rust_cache.setex_calls[0]
    assert (key, ttl) == ("rates", 70)
    assert json.loads(payload) == {"timestamp": 120, "value": "new"}


def test_stale_cache_rust_error_does_not_fall_through_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeRustModule(_FakeRustCache(fail=True))
    stale_cache._cached_rust_stale_cache.cache_clear()
    monkeypatch.setattr(stale_cache, "_load_rust_stale_cache", lambda: module)

    cache = stale_cache.StaleCache(redis_client=_FailingPythonRedis(), now=lambda: 100)
    with pytest.raises(ValueError, match="synthetic Rust Redis read failure"):
        cache.get(
            key="rates",
            lock_key="rates:lock",
            ttl=10,
            stale_grace=60,
            refresh=lambda: "new",
            schedule_refresh=lambda callback: callback(),
        )
