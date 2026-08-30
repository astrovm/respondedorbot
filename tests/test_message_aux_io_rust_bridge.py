from __future__ import annotations

import json
from typing import Any

import pytest
import redis

from api.memory import state


class _FakeRustMessageState:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.members: dict[str, dict[str, str]] = {}
        self.calls: list[tuple[object, ...]] = []

    def get_value(self, key: str) -> str | None:
        self.calls.append(("get", key))
        return self.values.get(key)

    def set_value(self, key: str, value: str, ttl_seconds: int) -> None:
        self.calls.append(("set", key, value, ttl_seconds))
        self.values[key] = value

    def save_compaction_result(
        self,
        summary_key: str,
        marker_key: str,
        summary: str,
        marker: str,
        ttl_seconds: int,
    ) -> None:
        self.calls.append(
            ("compaction", summary_key, marker_key, summary, marker, ttl_seconds)
        )
        self.values[summary_key] = summary
        self.values[marker_key] = marker

    def save_chat_member(
        self,
        key: str,
        user_id: str,
        payload: str,
        ttl_seconds: int,
    ) -> None:
        self.calls.append(("member-set", key, user_id, payload, ttl_seconds))
        self.members.setdefault(key, {})[user_id] = payload

    def get_chat_members(self, key: str) -> str:
        self.calls.append(("members-get", key))
        return json.dumps(
            [
                {"user_id": user_id, "payload": payload}
                for user_id, payload in sorted(self.members.get(key, {}).items())
            ]
        )


class _FakeRustModule:
    def __init__(self, message_state: _FakeRustMessageState) -> None:
        self.message_state = message_state
        self.endpoints: list[tuple[str, int, str | None]] = []

    def RedisMessageState(  # noqa: N802
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _FakeRustMessageState:
        self.endpoints.append((host, port, password))
        return self.message_state


class _UnexpectedRedis:
    def __getattr__(self, name: str) -> Any:
        pytest.fail(f"Python Redis owner must not run: {name}")


def test_rust_exclusively_owns_auxiliary_message_state_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMessageState()
    module = _FakeRustModule(rust)
    monkeypatch.setattr(state, "_load_rust_message_state_io", lambda: module)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    python_redis = _UnexpectedRedis()
    reporter_calls: list[str] = []

    state.save_chat_summary(python_redis, "chat", "summary")
    assert state.get_chat_summary(python_redis, "chat") == "summary"
    state.save_user_chat_summary(python_redis, "chat", "user-summary")
    assert state.get_user_chat_summary(python_redis, "chat") == "user-summary"
    state.save_chat_compaction_result(python_redis, "chat", "new-summary", "42")
    assert state.get_chat_compacted_until(python_redis, "chat") == "42"
    state.save_bot_message_metadata(
        python_redis,
        "chat",
        9,
        {"source": "synthetic"},
        admin_reporter=lambda message, _error, _extra: reporter_calls.append(message),
        ttl=120,
    )
    assert state.get_bot_message_metadata(
        python_redis,
        "chat",
        9,
        admin_reporter=lambda message, _error, _extra: reporter_calls.append(message),
        decode_redis_value=lambda _value: pytest.fail("Python decoder must not run"),
    ) == {"source": "synthetic"}
    state.save_chat_member(python_redis, "chat", "7", "Ana", "ana")
    members = state.get_chat_members(python_redis, "chat")
    assert members[0]["user_id"] == "7"
    assert members[0]["first_name"] == "Ana"
    assert members[0]["username"] == "ana"
    assert members[0]["last_seen"] > 0

    assert reporter_calls == []
    assert module.endpoints
    assert set(module.endpoints) == {
        ("redis.internal", 6380, "synthetic-password")
    }


def test_rust_member_read_failure_keeps_existing_empty_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRust(_FakeRustMessageState):
        def get_chat_members(self, key: str) -> str:
            raise redis.RedisError(f"synthetic read failure for {key}")

    rust = _FailingRust()
    monkeypatch.setattr(
        state,
        "_load_rust_message_state_io",
        lambda: _FakeRustModule(rust),
    )
    assert state.get_chat_members(_UnexpectedRedis(), "chat") == []
