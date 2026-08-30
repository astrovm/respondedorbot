from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import redis

from api.bot import chat_settings


class _FakeRustChatAdminCache:
    def __init__(self, cached: bool | None, *, fail: bool = False) -> None:
        self.cached = cached
        self.fail = fail
        self.gets: list[tuple[object, ...]] = []
        self.sets: list[tuple[object, ...]] = []

    def redis_chat_admin_get(self, *arguments: object) -> bool | None:
        self.gets.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust Redis read failure")
        return self.cached

    def redis_chat_admin_set(self, *arguments: object) -> None:
        self.sets.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust Redis write failure")


def _unexpected_python_redis() -> redis.Redis | None:
    pytest.fail("Python Redis owner must not run")


def _unused_log(_message: str, _extra: Mapping[str, Any] | None) -> None:
    return None


def test_rust_cached_admin_result_skips_python_redis_and_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustChatAdminCache(True)
    telegram_calls = 0

    def telegram_request(*_args: object, **_kwargs: object) -> tuple[None, None]:
        nonlocal telegram_calls
        telegram_calls += 1
        return None, None

    monkeypatch.setattr(chat_settings, "_load_rust_chat_admin_cache", lambda: rust)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")

    assert chat_settings.is_chat_admin(
        "chat-1",
        42,
        redis_client=None,
        optional_redis_client=_unexpected_python_redis,
        telegram_request=telegram_request,
        log_event=_unused_log,
    )
    assert telegram_calls == 0
    assert rust.gets == [
        ("redis.internal", 6380, "synthetic-password", "chat-1", "42")
    ]
    assert rust.sets == []


def test_rust_cache_miss_fetches_telegram_and_caches_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustChatAdminCache(None)
    monkeypatch.setattr(chat_settings, "_load_rust_chat_admin_cache", lambda: rust)

    def telegram_request(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], None]:
        return {"ok": True, "result": {"status": "administrator"}}, None

    assert chat_settings.is_chat_admin(
        "chat-1",
        42,
        redis_client=None,
        optional_redis_client=_unexpected_python_redis,
        telegram_request=telegram_request,
        log_event=_unused_log,
    )
    assert rust.sets == [
        ("localhost", 6379, None, "chat-1", "42", True, 300)
    ]


def test_rust_cache_outage_keeps_telegram_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustChatAdminCache(None, fail=True)
    monkeypatch.setattr(chat_settings, "_load_rust_chat_admin_cache", lambda: rust)

    def telegram_request(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], None]:
        return {"ok": True, "result": {"status": "member"}}, None

    assert not chat_settings.is_chat_admin(
        "chat-1",
        42,
        redis_client=None,
        optional_redis_client=_unexpected_python_redis,
        telegram_request=telegram_request,
        log_event=_unused_log,
    )
    assert len(rust.gets) == 1
    assert len(rust.sets) == 1
