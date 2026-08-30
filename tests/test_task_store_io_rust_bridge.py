from __future__ import annotations

import json

import pytest

from api.tasks import scheduler


class _FakeRustTaskStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _call(self, name: str, *arguments: object) -> None:
        if self.fail:
            raise ValueError(f"synthetic Rust task-store {name} failure")
        self.calls.append((name, *arguments))

    def get(self, key: str) -> str | None:
        self._call("get", key)
        return "payload"

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self._call("setex", key, ttl, value)
        return True

    def delete(self, key: str) -> int:
        self._call("delete", key)
        return 1

    def zadd(self, key: str, member: str, score: float) -> int:
        self._call("zadd", key, member, score)
        return 1

    def expire(self, key: str, ttl: int) -> bool:
        self._call("expire", key, ttl)
        return True

    def zrem(self, key: str, members: list[str]) -> int:
        self._call("zrem", key, members)
        return len(members)

    def scan(self, pattern: str) -> str:
        self._call("scan", pattern)
        return json.dumps(["task:data:t1"])

    def zrange(self, key: str) -> str:
        self._call("zrange", key)
        return json.dumps(["t1"])

    def mget(self, keys: list[str]) -> str:
        self._call("mget", keys)
        return json.dumps(["payload", None])


class _FakeRustModule:
    def __init__(self, store: _FakeRustTaskStore) -> None:
        self.store = store
        self.endpoints: list[tuple[str, int, str | None]] = []

    def RedisTaskStore(
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _FakeRustTaskStore:
        self.endpoints.append((host, port, password))
        return self.store


@pytest.fixture(autouse=True)
def _restore_scheduler_globals() -> object:
    redis_client = scheduler._redis_client
    task_executor = scheduler._task_executor
    yield
    scheduler._redis_client = redis_client
    scheduler._task_executor = task_executor
    scheduler._cached_rust_task_store.cache_clear()


def test_task_store_rust_path_is_the_only_redis_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeRustTaskStore()
    module = _FakeRustModule(store)
    monkeypatch.setattr(scheduler, "_load_rust_task_store", lambda: module)
    monkeypatch.setattr(scheduler, "build_task_executor", lambda **_deps: object())
    monkeypatch.setattr(scheduler, "get_scheduler", lambda: object())
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")

    scheduler.init_scheduler(
        lambda: pytest.fail("Python Redis owner must not run"),
        {},
    )
    client = scheduler._redis_client
    assert isinstance(client, scheduler._RustTaskRedisClient)

    assert client.get("task:data:t1") == "payload"
    assert client.setex("task:data:t1", 60, "payload") is True
    assert client.delete("task:data:t1") == 1
    assert client.zadd("task:chat:c1", {"t1": 42.5}) == 1
    assert client.expire("task:chat:c1", 60) is True
    assert client.zrem("task:chat:c1", "t1", "t2") == 2
    assert list(client.scan_iter("task:data:*")) == ["task:data:t1"]
    assert client.zrange("task:chat:c1", 0, -1) == ["t1"]
    assert client.mget(["task:data:t1", "task:data:t2"]) == ["payload", None]

    assert module.endpoints == [("redis.internal", 6380, "synthetic-password")]
    assert store.calls == [
        ("get", "task:data:t1"),
        ("setex", "task:data:t1", 60, "payload"),
        ("delete", "task:data:t1"),
        ("zadd", "task:chat:c1", "t1", 42.5),
        ("expire", "task:chat:c1", 60),
        ("zrem", "task:chat:c1", ["t1", "t2"]),
        ("scan", "task:data:*"),
        ("zrange", "task:chat:c1"),
        ("mget", ["task:data:t1", "task:data:t2"]),
    ]


def test_task_store_rust_error_does_not_fall_through_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeRustModule(_FakeRustTaskStore(fail=True))
    monkeypatch.setattr(scheduler, "_load_rust_task_store", lambda: module)
    client = scheduler._task_redis_client(
        lambda: pytest.fail("Python Redis owner must not run")
    )

    with pytest.raises(ValueError, match="synthetic Rust task-store get failure"):
        client.get("task:data:t1")
