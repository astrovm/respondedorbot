from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.bot import command_registry


def _handler(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    path = Path(__file__).parents[1] / "contracts" / "media_routing.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    commands = {"/known": (_handler, False, False)}

    for case in contract["cases"]:
        username = case["bot_username"]
        if username is None:
            monkeypatch.delenv("TELEGRAM_USERNAME", raising=False)
        else:
            monkeypatch.setenv("TELEGRAM_USERNAME", username)
        reply_username = case["reply_username"]
        message = {"chat": {"type": case["chat_type"]}}
        if reply_username is not None:
            message["reply_to_message"] = {"from": {"username": reply_username}}

        actual = command_registry._should_auto_process_media_python(
            commands,
            "/known" if case["known_command"] else "",
            case["message_text"],
            message,
        )

        assert actual is case["expected"], case["name"]


class _FakeRustRouter:
    def __init__(self, response: bool | Exception) -> None:
        self.response = response
        self.input: tuple[Any, ...] | None = None

    def should_auto_process_media(self, *values: Any) -> bool:
        self.input = values
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_rust_router_receives_normalized_message_facts(monkeypatch) -> None:
    rust = _FakeRustRouter(True)
    monkeypatch.setattr(command_registry, "_load_rust_media_router", lambda: rust)
    monkeypatch.setenv("TELEGRAM_USERNAME", " testbot ")
    commands = {"/ask": (_handler, True, True)}

    result = command_registry.should_auto_process_media(
        commands,
        "/ask",
        "hola @TESTBOT",
        {
            "chat": {"type": "group"},
            "reply_to_message": {"from": {"username": "testbot"}},
        },
    )

    assert result is True
    assert rust.input == ("group", True, "hola @TESTBOT", "testbot", "testbot")


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustRouter(RuntimeError("synthetic bridge failure"))
    monkeypatch.setattr(command_registry, "_load_rust_media_router", lambda: rust)
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    result = command_registry.should_auto_process_media(
        {},
        "",
        "hola @testbot",
        {"chat": {"type": "group"}},
    )

    assert result is True
    assert "using Python fallback" in caplog.text
