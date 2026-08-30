from __future__ import annotations

from logging import Logger

import pytest
import redis

from api.media import cache


class _FakeRustMediaCache:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.gets: list[tuple[object, ...]] = []
        self.sets: list[tuple[object, ...]] = []

    def redis_media_cache_get(self, *arguments: object) -> str | None:
        if self.fail:
            raise ValueError("synthetic Rust Redis read failure")
        self.gets.append(arguments)
        return "rust-cached"

    def redis_media_cache_set(self, *arguments: object) -> None:
        if self.fail:
            raise ValueError("synthetic Rust Redis write failure")
        self.sets.append(arguments)


def test_media_cache_rust_path_is_the_only_redis_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMediaCache()
    python_redis_calls = 0

    def python_redis() -> redis.Redis:
        nonlocal python_redis_calls
        python_redis_calls += 1
        raise AssertionError("Python Redis owner must not run")

    monkeypatch.setattr(cache, "_load_rust_media_cache", lambda: rust)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    logger = Logger("test-media-cache")

    assert (
        cache.get_cached_media(
            "audio_transcription",
            "file",
            redis_factory=python_redis,
            logger=logger,
        )
        == "rust-cached"
    )
    cache.cache_media(
        "image_description",
        "image",
        "description",
        3600,
        redis_factory=python_redis,
        logger=logger,
    )

    assert python_redis_calls == 0
    assert rust.gets == [
        ("redis.internal", 6380, "synthetic-password", "audio_transcription", "file")
    ]
    assert rust.sets == [
        (
            "redis.internal",
            6380,
            "synthetic-password",
            "image_description",
            "image",
            "description",
            3600,
        )
    ]


def test_media_cache_rust_errors_keep_existing_nonfatal_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cache,
        "_load_rust_media_cache",
        lambda: _FakeRustMediaCache(fail=True),
    )

    def python_redis() -> redis.Redis:
        pytest.fail("Python Redis owner must not run")

    logger = Logger("test-media-cache-errors")

    assert (
        cache.get_cached_media(
            "audio_transcription",
            "file",
            redis_factory=python_redis,
            logger=logger,
        )
        is None
    )
    cache.cache_media(
        "image_description",
        "image",
        "description",
        3600,
        redis_factory=python_redis,
        logger=logger,
    )
