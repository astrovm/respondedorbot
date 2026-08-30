from __future__ import annotations

import json
from logging import Logger

import pytest

from api.cache import http
from api.services.maintenance import REQUEST_CACHE_HISTORY_TTL
from api.services.redis_helpers import redis_get_json, redis_set_json


class _FakeRustCache:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.values: dict[str, str] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, str, int | None]] = []

    def get(self, key: str) -> str | None:
        if self.fail:
            raise ValueError("synthetic Rust Redis read failure")
        self.gets.append(key)
        return self.values.get(key)

    def set(self, key: str, value: str, *, ex: int | None = None) -> bool:
        if self.fail:
            raise ValueError("synthetic Rust Redis write failure")
        self.sets.append((key, value, ex))
        self.values[key] = value
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


class _Response:
    text = '{"ok": true}'

    def raise_for_status(self) -> None:
        return None


def test_request_cache_rust_path_is_the_only_redis_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _FakeRustCache()
    module = _FakeRustModule(cache)
    http._cached_rust_request_cache.cache_clear()
    monkeypatch.setattr(http, "_load_rust_request_cache", lambda: module)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")

    result = http.cached_request(
        "https://example.test/data",
        None,
        None,
        300,
        hourly_cache=True,
        history_hours=False,
        verify_ssl=True,
        redis_factory=lambda: pytest.fail("Python Redis owner must not run"),
        redis_get_json=redis_get_json,
        redis_set_json=redis_set_json,
        get_history=lambda *_arguments: None,
        http_get=lambda *_arguments, **_keywords: _Response(),
        admin_report=lambda *_arguments: None,
        logger=Logger("test-request-cache"),
    )

    assert result is not None
    assert result["data"] == {"ok": True}
    assert module.endpoints == [("redis.internal", 6380, "synthetic-password")]
    assert len(cache.gets) == 2
    assert cache.gets[0].startswith("request_cache:")
    assert cache.gets[1].startswith("request_cache_history:")
    assert len(cache.sets) == 2
    current_key, current_value, current_ttl = cache.sets[0]
    hourly_key, hourly_value, hourly_ttl = cache.sets[1]
    assert current_key.startswith("request_cache:")
    assert json.loads(current_value)["data"] == {"ok": True}
    assert current_ttl == 300
    assert hourly_key.startswith("request_cache_history:")
    assert json.loads(hourly_value)["data"] == {"ok": True}
    assert hourly_ttl == REQUEST_CACHE_HISTORY_TTL


def test_request_cache_rust_error_does_not_fall_through_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _FakeRustCache(fail=True)
    module = _FakeRustModule(cache)
    http._cached_rust_request_cache.cache_clear()
    monkeypatch.setattr(http, "_load_rust_request_cache", lambda: module)

    result = http.cached_request(
        "https://example.test/data",
        None,
        None,
        60,
        hourly_cache=False,
        history_hours=False,
        verify_ssl=True,
        redis_factory=lambda: pytest.fail("Python Redis owner must not run"),
        redis_get_json=redis_get_json,
        redis_set_json=redis_set_json,
        get_history=lambda *_arguments: None,
        http_get=lambda *_arguments, **_keywords: _Response(),
        admin_report=lambda *_arguments: None,
        logger=Logger("test-request-cache-errors"),
    )

    assert result is not None
    assert result["data"] == {"ok": True}
