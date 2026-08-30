from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from api.bot import streaming


class _FakeRustTelegramStreamPlanning:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object) -> object:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust Telegram stream-planning failure")
        return value

    def telegram_stream_should_edit(self, *arguments: object) -> bool:
        return bool(self._result("should_edit", True, *arguments))

    def telegram_stream_plan_feed(self, *arguments: object) -> tuple[str, str]:
        result = self._result("feed", ("rust-buffer", "send"), *arguments)
        assert isinstance(result, tuple)
        return result

    def telegram_stream_plan_finalize(self, *arguments: object) -> tuple[str, str]:
        result = self._result("finalize", ("rust-final", "none"), *arguments)
        assert isinstance(result, tuple)
        return result


def _streamer(
    sent: list[tuple[str, str, Optional[str]]],
    edits: list[tuple[str, str, str]],
    *,
    min_edit_interval_ms: float = 300.0,
    min_chars_between_edits: int = 15,
) -> streaming.TelegramMessageStreamer:
    def send_message(
        chat_id: str,
        text: str,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[int]:
        sent.append((chat_id, text, reply_to_message_id))
        return 321

    def edit_message(chat_id: str, text: str, message_id: str) -> None:
        edits.append((chat_id, text, message_id))

    return streaming.TelegramMessageStreamer(
        "synthetic-chat",
        send_message,
        edit_message,
        min_edit_interval_ms=min_edit_interval_ms,
        min_chars_between_edits=min_chars_between_edits,
        reply_to_message_id="synthetic-reply",
    )


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        streaming,
        "_load_rust_telegram_stream_planning",
        lambda: None,
    )
    path = Path(__file__).parents[1] / "contracts" / "telegram_stream_planning.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["should_edit_cases"]:
        (
            done,
            has_message_id,
            now,
            last_edit,
            buffer_chars,
            sent_chars,
            min_interval,
            min_chars,
        ) = case["facts"]
        streamer = _streamer([], [])
        streamer._done = done
        streamer._message_id = "1" if has_message_id else None
        streamer._last_edit_time = last_edit
        streamer._buffer = "a" * buffer_chars
        streamer._sent_text = "a" * sent_chars
        streamer._min_interval = min_interval
        streamer._min_chars = min_chars
        monkeypatch.setattr(streaming.time, "time", lambda value=now: value)
        assert streamer._should_edit() is case["expected"], case["name"]

    for case in contract["feed_cases"]:
        (
            done,
            has_message_id,
            send_attempted,
            buffer,
            sent_text,
            token,
            now,
            last_edit,
            min_interval,
            min_chars,
        ) = case["facts"]
        streamer = _streamer([], [])
        streamer._done = done
        streamer._message_id = "1" if has_message_id else None
        streamer._send_attempted = send_attempted
        streamer._buffer = buffer
        streamer._sent_text = sent_text
        streamer._last_edit_time = last_edit
        streamer._min_interval = min_interval
        streamer._min_chars = min_chars
        assert list(streamer._plan_feed(token, now)) == case["expected"], case["name"]

    for case in contract["finalize_cases"]:
        buffer, sent_text, has_message_id, final_text = case["facts"]
        streamer = _streamer([], [])
        streamer._buffer = buffer
        streamer._sent_text = sent_text
        streamer._message_id = "1" if has_message_id else None
        assert list(streamer._plan_finalize(final_text)) == case["expected"], case["name"]


def test_rust_telegram_stream_planning_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustTelegramStreamPlanning()
    monkeypatch.setattr(
        streaming,
        "_load_rust_telegram_stream_planning",
        lambda: rust,
    )
    monkeypatch.setattr(streaming.time, "time", lambda: 5.0)
    sent: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []
    streamer = _streamer(sent, edits)

    streamer.feed("python-token")

    assert sent == [("synthetic-chat", "rust-buffer", "synthetic-reply")]
    assert edits == []
    assert streamer.message_id == "321"
    assert rust.calls[0] == (
        "feed",
        False,
        False,
        False,
        "",
        "",
        "python-token",
        0.0,
        0.0,
        0.3,
        15,
    )

    streamer._buffer = "short"
    streamer._sent_text = "short"
    assert streamer._should_edit() is True
    assert rust.calls[1][0] == "should_edit"

    assert streamer.finalize("python-final") == "rust-final"
    assert sent == [("synthetic-chat", "rust-buffer", "synthetic-reply")]
    assert edits == []
    assert rust.calls[2] == (
        "finalize",
        "short",
        "short",
        True,
        "python-final",
    )


def test_rust_telegram_stream_planning_failure_preserves_python_behavior(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustTelegramStreamPlanning(fail=True)
    monkeypatch.setattr(
        streaming,
        "_load_rust_telegram_stream_planning",
        lambda: rust,
    )
    monkeypatch.setattr(streaming.time, "time", lambda: 5.0)
    sent: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []
    streamer = _streamer(
        sent,
        edits,
        min_edit_interval_ms=0.0,
        min_chars_between_edits=1,
    )

    with caplog.at_level(logging.ERROR, logger=streaming.__name__):
        streamer.feed("hello")
        streamer.feed(" world")
        final_text = streamer.finalize("final")

    assert final_text == "final"
    assert sent == [("synthetic-chat", "hello", "synthetic-reply")]
    assert edits == [
        ("synthetic-chat", "hello world", "321"),
        ("synthetic-chat", "final", "321"),
    ]
    assert [call[0] for call in rust.calls] == ["feed", "feed", "finalize"]
    assert caplog.text.count("using Python fallback") == 3


def test_invalid_rust_action_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustTelegramStreamPlanning()
    rust.telegram_stream_plan_feed = lambda *_arguments: ("ignored", "invalid")  # type: ignore[method-assign]
    monkeypatch.setattr(
        streaming,
        "_load_rust_telegram_stream_planning",
        lambda: rust,
    )
    sent: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []
    streamer = _streamer(sent, edits)

    with caplog.at_level(logging.ERROR, logger=streaming.__name__):
        streamer.feed("hello")

    assert sent == [("synthetic-chat", "hello", "synthetic-reply")]
    assert "using Python fallback" in caplog.text
