from __future__ import annotations

import json
from collections.abc import Callable
from logging import Logger
from typing import Any

import pytest

from api.memory import background
from api.memory.compaction import CompactionPlan


class _FakeRustQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}
        self.locks: list[tuple[str, str, int]] = []
        self.releases: list[tuple[str, str]] = []
        self.quarantined: list[tuple[str, str, str]] = []
        self.fail_exists = False

    def job_exists(self, chat_id: str) -> bool:
        if self.fail_exists:
            raise ValueError("synthetic Rust Redis failure")
        return chat_id in self.jobs

    def insert_job(self, chat_id: str, payload: str) -> bool:
        if chat_id in self.jobs:
            return False
        self.jobs[chat_id] = payload
        return True

    def list_jobs(self) -> str:
        return json.dumps(
            [
                {"chat_id": chat_id, "payload": payload}
                for chat_id, payload in sorted(self.jobs.items())
            ]
        )

    def replace_job(self, chat_id: str, payload: str) -> None:
        self.jobs[chat_id] = payload

    def delete_job(self, chat_id: str) -> bool:
        return self.jobs.pop(chat_id, None) is not None

    def acquire_lock(self, chat_id: str, token: str, ttl_seconds: int) -> bool:
        self.locks.append((chat_id, token, ttl_seconds))
        return True

    def release_lock(self, chat_id: str, token: str) -> bool:
        self.releases.append((chat_id, token))
        return True

    def quarantine_job(
        self,
        chat_id: str,
        dead_job_id: str,
        dead_payload: str,
    ) -> bool:
        self.jobs.pop(chat_id, None)
        self.quarantined.append((chat_id, dead_job_id, dead_payload))
        return True


class _FakeRustModule:
    def __init__(self, queue: _FakeRustQueue) -> None:
        self.queue = queue
        self.endpoints: list[tuple[str, int, str | None]] = []

    def RedisCompactionQueue(  # noqa: N802
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _FakeRustQueue:
        self.endpoints.append((host, port, password))
        return self.queue


class _Billing:
    user_id = 42
    message = {"message_id": 99}

    def __init__(self) -> None:
        self.reservations = 0

    def reserve_background_ai_credits(
        self,
        usage_tag: str,
        _reserve_units: int,
        *,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], None]:
        self.reservations += 1
        return (
            {
                "reserved_credit_units": 3,
                "source": "user",
                "usage_tag": usage_tag,
                "metadata": metadata,
            },
            None,
        )


def _queue(
    monkeypatch: pytest.MonkeyPatch,
    rust_queue: _FakeRustQueue,
    redis_factory: Callable[[], Any],
) -> background.DurableCompactionQueue:
    module = _FakeRustModule(rust_queue)
    monkeypatch.setattr(background, "_load_rust_compaction_queue", lambda: module)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    queue = background.DurableCompactionQueue(
        redis_factory=redis_factory,
        compact=lambda _messages, _summary: ("summary", 1),
        get_summary=lambda _client, _chat_id: None,
        get_marker=lambda _client, _chat_id: None,
        save_result=lambda _client, _chat_id, _summary, _marker: None,
        estimate_reserve=lambda _plan: 3,
        settle_reservation=lambda **_kwargs: {"applied": True},
        record_provider_usage=lambda **_kwargs: True,
        list_provider_usage=lambda **_kwargs: [],
        logger=Logger("test-compaction-queue"),
    )
    assert module.endpoints == [("redis.internal", 6380, "synthetic-password")]
    return queue


def _unexpected_python_redis() -> Any:
    pytest.fail("Python must not own compaction-queue Redis operations")


def test_rust_queue_exclusively_checks_and_inserts_durable_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_queue = _FakeRustQueue()
    queue = _queue(monkeypatch, rust_queue, _unexpected_python_redis)
    billing = _Billing()
    plan = CompactionPlan(
        chat_id="chat-1",
        messages=[{"id": "message-1", "content": "synthetic"}],
        prior_summary=None,
        expected_marker=None,
        target_marker="message-1",
    )

    assert queue.enqueue(plan, billing)
    assert billing.reservations == 1
    stored = json.loads(rust_queue.jobs["chat-1"])
    assert stored["schema_version"] == 1
    assert stored["target_marker"] == "message-1"


def test_rust_queue_atomically_quarantines_undecodable_job_without_python_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_queue = _FakeRustQueue()
    rust_queue.jobs["chat-1"] = "not-json"
    queue = _queue(monkeypatch, rust_queue, _unexpected_python_redis)

    assert queue.run_pending_once() == 0
    assert rust_queue.jobs == {}
    assert len(rust_queue.quarantined) == 1
    chat_id, dead_job_id, dead_payload = rust_queue.quarantined[0]
    assert chat_id == "chat-1"
    assert dead_job_id.startswith("chat-1:")
    assert json.loads(dead_payload)["reason"] == "undecodable"


def test_rust_queue_error_does_not_fall_through_to_python_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_queue = _FakeRustQueue()
    rust_queue.fail_exists = True
    queue = _queue(monkeypatch, rust_queue, _unexpected_python_redis)
    billing = _Billing()
    plan = CompactionPlan(
        chat_id="chat-1",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="message-1",
    )

    assert not queue.enqueue(plan, billing)
    assert billing.reservations == 0
