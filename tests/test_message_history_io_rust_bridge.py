from __future__ import annotations

import json
from typing import Any

import pytest

from api.memory import state


class _FakeRustMessageHistory:
    def __init__(self) -> None:
        self.saved: list[tuple[object, ...]] = []
        self.history = [
            json.dumps({"id": "2", "text": "later", "timestamp": 20}),
            json.dumps({"id": "1", "text": "earlier", "timestamp": 10, "role": "user"}),
        ]
        self.rows = [
            {
                "key": "chatmsg:chat:2",
                "fields": {
                    "message_id": "2",
                    "text": "wallet later",
                    "timestamp": "20",
                    "reply_to_message_id": "",
                },
            },
            {
                "key": "chatmsg:chat:1",
                "fields": {
                    "message_id": "1",
                    "text": "wallet earlier",
                    "timestamp": "10",
                    "reply_to_message_id": "99",
                },
            },
        ]

    def save_message(self, *arguments: object) -> bool:
        self.saved.append(arguments)
        return True

    def get_history_entries(self, chat_id: str, max_messages: int) -> str:
        assert chat_id == "chat"
        return json.dumps(self.history[:max_messages])

    def fetch_messages(self, chat_id: str, limit: int) -> str:
        assert chat_id == "chat"
        return json.dumps(self.rows[:limit])

    def search_messages(self, chat_id: str, query_text: str, limit: int) -> str:
        assert (chat_id, query_text, limit) == ("chat", "wallet", 10)
        return json.dumps(self.rows)


class _FakeRustModule:
    def __init__(self, history: _FakeRustMessageHistory) -> None:
        self.history = history
        self.endpoints: list[tuple[str, int, str | None]] = []

    def RedisMessageState(  # noqa: N802
        self,
        host: str,
        port: int,
        password: str | None,
    ) -> _FakeRustMessageHistory:
        self.endpoints.append((host, port, password))
        return self.history


class _UnexpectedRedis:
    def __getattr__(self, name: str) -> Any:
        pytest.fail(f"Python Redis owner must not run: {name}")


def test_rust_exclusively_owns_message_history_and_search_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMessageHistory()
    module = _FakeRustModule(rust)
    monkeypatch.setattr(state, "_load_rust_message_history_io", lambda: module)
    monkeypatch.setattr(state, "_load_rust_message_state", lambda: None)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    python_redis = _UnexpectedRedis()
    reports: list[str] = []

    def reporter(message: str, _error: Exception | None, _extra: Any) -> None:
        reports.append(message)

    state.save_message_to_redis(
        "chat",
        "3",
        "synthetic message",
        python_redis,
        admin_reporter=reporter,
        user_id="7",
        username="synthetic-user",
    )
    assert rust.saved
    assert rust.saved[0][0:3] == ("chat", "3", "synthetic message")
    assert state.get_chat_history(
        "chat",
        python_redis,
        admin_reporter=reporter,
        max_messages=2,
    ) == [
        {"id": "1", "text": "earlier", "timestamp": 10, "role": "user"},
        {"id": "2", "text": "later", "timestamp": 20, "role": "user"},
    ]
    compactable = state.fetch_chat_messages_for_compaction(
        python_redis,
        "chat",
        limit=2,
        admin_reporter=reporter,
    )
    assert [row["id"] for row in compactable] == ["1", "2"]
    searched = state.search_chat_history(
        python_redis,
        "chat",
        "wallet",
        reply_to_message_id="99",
        limit=2,
        admin_reporter=reporter,
    )
    assert [row["message_id"] for row in searched] == ["1", "2"]
    assert reports == []
    assert module.endpoints == [("redis.internal", 6380, "synthetic-password")]


def test_rust_history_error_reports_without_python_fallthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRust(_FakeRustMessageHistory):
        def get_history_entries(self, chat_id: str, max_messages: int) -> str:
            raise ValueError(f"synthetic Rust read failure {chat_id} {max_messages}")

    module = _FakeRustModule(_FailingRust())
    monkeypatch.setattr(state, "_load_rust_message_history_io", lambda: module)
    reports: list[str] = []
    assert state.get_chat_history(
        "chat",
        _UnexpectedRedis(),
        admin_reporter=lambda message, _error, _extra: reports.append(message),
    ) == []
    assert len(reports) == 1
