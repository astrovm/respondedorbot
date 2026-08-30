from __future__ import annotations

import json
import logging
from pathlib import Path

from api.bot import chat_context, message_content


class _FakeRustTelegramInput:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.content: object = {}
        self.calls: list[tuple[object, ...]] = []

    def _call(self, name: str, arguments: tuple[object, ...], result: object) -> object:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Telegram input failure")
        return result

    def telegram_extract_message_content(self, *arguments: object) -> str:
        return json.dumps(self._call("content", arguments, self.content))

    def telegram_is_group_chat_type(self, *arguments: object) -> bool:
        return bool(self._call("group", arguments, True))

    def telegram_normalize_numeric_id(self, *arguments: object) -> int:
        return int(self._call("numeric", arguments, 42))

    def telegram_extract_user_id(self, *arguments: object) -> int:
        return int(self._call("user_id", arguments, 77))

    def telegram_format_user_identity(self, *arguments: object) -> str:
        return str(self._call("identity", arguments, "Rust User"))


def test_rust_message_content_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustTelegramInput()
    rust.content = {
        "text": "typed text",
        "photo_file_id": "photo",
        "audio_file_id": "audio",
    }
    monkeypatch.setattr(message_content, "load_rust_telegram_input", lambda: rust)
    message = {"provider": "payload-owned-by-rust"}

    assert message_content.extract_message_content(message) == (
        "typed text",
        "photo",
        "audio",
    )
    assert rust.calls == [
        ("content", json.dumps(message, separators=(",", ":")))
    ]


def test_invalid_rust_message_content_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustTelegramInput(fail=True)
    monkeypatch.setattr(message_content, "load_rust_telegram_input", lambda: rust)
    message = {"text": " fallback ", "voice": {"file_id": "voice"}}

    with caplog.at_level(logging.ERROR, logger=message_content.__name__):
        actual = message_content.extract_message_content(message)

    assert actual == ("fallback", None, "voice")
    assert "using Python fallback" in caplog.text


def test_rust_chat_context_helpers_are_authoritative(monkeypatch) -> None:
    rust = _FakeRustTelegramInput()
    monkeypatch.setattr(chat_context, "load_rust_telegram_input", lambda: rust)

    assert chat_context.is_group_chat_type("private") is True
    assert chat_context.extract_numeric_chat_id("not-numeric") == 42
    assert chat_context.extract_user_id({"from": {}}) == 77
    assert chat_context.format_user_identity({}) == "Rust User"


def test_python_fallback_helpers_match_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(message_content, "load_rust_telegram_input", lambda: None)
    monkeypatch.setattr(chat_context, "load_rust_telegram_input", lambda: None)
    path = Path(__file__).parents[1] / "contracts" / "telegram_input.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["content_cases"]:
        text, photo, audio = message_content.extract_message_content(case["message"])
        assert {
            "text": text,
            "photo_file_id": photo,
            "audio_file_id": audio,
        } == case["expected"], case["name"]
    for case in contract["numeric_id_cases"]:
        assert chat_context.extract_numeric_chat_id(case["value"]) == case["expected"]
    for case in contract["user_cases"]:
        assert (
            chat_context.extract_user_id({"from": case["user"]})
            == case["expected_id"]
        )
        assert (
            chat_context.format_user_identity(case["user"])
            == case["expected_identity"]
        )
    for case in contract["group_cases"]:
        assert (
            chat_context.is_group_chat_type(case["chat_type"]) is case["expected"]
        )
