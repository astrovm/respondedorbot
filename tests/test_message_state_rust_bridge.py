from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from api.memory import state


class _FakeRustMessageState:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Any, ...]] = []

    def prepare_message_write(self, *arguments: Any) -> str:
        if self.fail:
            raise ValueError("synthetic message-state failure")
        self.calls.append(arguments)
        return json.dumps(
            {
                "keys": {
                    "history": "rust:history",
                    "order": "rust:order",
                    "legacy_ids": "rust:legacy",
                    "sequence": "rust:sequence",
                    "search_document": "rust:search",
                },
                "message_id": "rust-message",
                "history_entry": "rust-entry",
                "chat_id": "rust-chat",
                "role": "rust-role",
                "user_id": "rust-user",
                "username": "rust-name",
                "text": "rust-text",
                "timestamp": 123,
                "reply_to_message_id": "rust-reply",
                "mentions_bot": "1",
            }
        )

    def escape_message_search_text(self, query_text: str) -> str:
        if self.fail:
            raise ValueError("synthetic search escape failure")
        return f"rust-text:{query_text}"

    def escape_message_search_tag(self, value: str) -> str:
        if self.fail:
            raise ValueError("synthetic tag escape failure")
        return f"rust-tag:{value}"

    def rank_message_search_results(
        self,
        candidates_json: str,
        search_text: str,
        reply_to_message_id: str | None,
        excluded_message_ids: list[str],
        limit: int,
    ) -> str:
        if self.fail:
            raise ValueError("synthetic search ranking failure")
        candidates = json.loads(candidates_json)
        assert search_text
        assert reply_to_message_id == "reply"
        assert excluded_message_ids == ["excluded"]
        assert limit == 2
        assert len(candidates) == 2
        return json.dumps(
            [
                {"index": 1, "reply_score": 1, "overlap_score": 2},
                {"index": 0, "reply_score": 0, "overlap_score": 1},
            ]
        )


def test_message_write_uses_one_rust_plan_and_one_atomic_redis_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMessageState()
    redis_client = MagicMock()
    monkeypatch.setattr(state, "_load_rust_message_state", lambda: rust)
    monkeypatch.setattr("api.memory.state.time.time", lambda: 100)

    state.save_message_to_redis(
        "chat",
        "message",
        "text",
        redis_client,
        admin_reporter=MagicMock(),
        role="user",
        user_id="7",
        username="astro",
        reply_to_message_id="previous",
        mentions_bot=True,
    )

    assert rust.calls == [
        ("chat", "message", "text", 100, "user", "7", "astro", "previous", True)
    ]
    eval_arguments = redis_client.eval.call_args.args
    assert eval_arguments[1:7] == (
        5,
        "rust:history",
        "rust:order",
        "rust:legacy",
        "rust:sequence",
        "rust:search",
    )
    assert eval_arguments[7] == "rust-message"
    assert eval_arguments[8] == "rust-entry"
    assert eval_arguments[11:] == (
        "rust-chat",
        "rust-role",
        "rust-user",
        "rust-name",
        "rust-text",
        123,
        "rust-reply",
        "1",
    )
    redis_client.eval.assert_called_once()


def test_message_write_falls_back_to_legacy_compatible_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        state,
        "_load_rust_message_state",
        lambda: _FakeRustMessageState(fail=True),
    )
    plan = state._prepare_message_write(
        "chat",
        "bot_2",
        "text",
        100,
        None,
        None,
        None,
        None,
        False,
    )

    assert plan["keys"]["history"] == "chat_history:chat"
    assert plan["role"] == "assistant"
    assert json.loads(plan["history_entry"]) == {
        "id": "bot_2",
        "text": "text",
        "timestamp": 100,
        "role": "assistant",
    }


def test_search_helpers_use_rust_without_losing_original_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMessageState()
    monkeypatch.setattr(state, "_load_rust_message_state", lambda: rust)
    rows = [
        {"message_id": "1", "text": "first", "timestamp": 1, "custom": "a"},
        {
            "message_id": "2",
            "text": "second",
            "reply_to_message_id": "reply",
            "timestamp": 2,
            "custom": "b",
        },
    ]

    assert state._escape_search_text("hello") == "rust-text:hello"
    assert state._escape_tag_value("-1") == "rust-tag:-1"
    ranked = state._rank_search_results(
        rows,
        "query",
        "reply",
        {"excluded"},
        2,
    )

    assert [row["custom"] for row in ranked] == ["b", "a"]
    assert ranked[0]["_reply_score"] == 1
    assert ranked[1]["_overlap_score"] == 1


def test_search_helpers_fall_back_after_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        state,
        "_load_rust_message_state",
        lambda: _FakeRustMessageState(fail=True),
    )
    rows = [
        {"message_id": "1", "text": "wallet", "timestamp": 1},
        {"message_id": "2", "text": "wallet error", "timestamp": 2},
    ]

    assert state._escape_search_text("wallet, @bot") == "wallet \\@bot"
    assert state._escape_tag_value("-1") == "\\-1"
    ranked = state._rank_search_results(rows, "wallet error", None, set(), 2)
    assert [row["message_id"] for row in ranked] == ["2", "1"]
