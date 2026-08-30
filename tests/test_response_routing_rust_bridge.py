from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from api.bot import command_registry


def _handler(*_args: Any, **_kwargs: Any) -> None:
    return None


class _FakeRustRouter:
    def __init__(self, responses: list[str] | Exception) -> None:
        self.responses = responses
        self.inputs: list[dict[str, Any]] = []

    def evaluate_response_routing(self, input_json: str) -> str:
        self.inputs.append(json.loads(input_json))
        if isinstance(self.responses, Exception):
            raise self.responses
        if not self.responses:
            raise RuntimeError("no synthetic response remains")
        return self.responses.pop(0)


def test_rust_router_loads_config_then_samples_random_once(monkeypatch) -> None:
    rust = _FakeRustRouter(
        ["needs_trigger_words", "needs_random_sample", "respond"]
    )
    monkeypatch.setattr(command_registry, "_load_rust_response_router", lambda: rust)
    monkeypatch.setattr(command_registry.random, "random", MagicMock(return_value=0.05))
    load_config = MagicMock(return_value={"trigger_words": ["gordo"]})

    result = command_registry.should_gordo_respond(
        {},
        "hola",
        "hola gordo",
        {"chat": {"type": "group"}},
        {"ai_random_replies": True},
        None,
        load_bot_config_fn=load_config,
    )

    assert result is True
    load_config.assert_called_once_with()
    command_registry.random.random.assert_called_once_with()
    assert [value["trigger_words"] for value in rust.inputs] == [
        None,
        ["gordo"],
        ["gordo"],
    ]
    assert [value["random_sample"] for value in rust.inputs] == [None, None, 0.05]


def test_early_rust_ignore_needs_no_config_or_random(monkeypatch) -> None:
    rust = _FakeRustRouter(["ignore"])
    monkeypatch.setattr(command_registry, "_load_rust_response_router", lambda: rust)
    random_sample = MagicMock()
    monkeypatch.setattr(command_registry.random, "random", random_sample)
    load_config = MagicMock()

    result = command_registry.should_gordo_respond(
        {},
        "hola",
        "hola",
        {
            "chat": {"type": "group"},
            "reply_to_message": {
                "from": {"username": "testbot"},
                "text": "https://fxtwitter.com/example/status/1",
            },
        },
        {"ignore_link_fix_followups": True},
        None,
        load_bot_config_fn=load_config,
    )

    assert result is False
    load_config.assert_not_called()
    random_sample.assert_not_called()


def test_value_error_loading_config_uses_default_trigger_words(monkeypatch) -> None:
    rust = _FakeRustRouter(["needs_trigger_words", "ignore"])
    monkeypatch.setattr(command_registry, "_load_rust_response_router", lambda: rust)
    load_config = MagicMock(side_effect=ValueError("synthetic invalid config"))

    result = command_registry.should_gordo_respond(
        {},
        "hola",
        "ordinary",
        {"chat": {"type": "group"}},
        {"ai_random_replies": True},
        None,
        load_bot_config_fn=load_config,
    )

    assert result is False
    assert rust.inputs[1]["trigger_words"] == ["bot", "assistant"]


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustRouter(RuntimeError("synthetic bridge failure"))
    monkeypatch.setattr(command_registry, "_load_rust_response_router", lambda: rust)
    load_config = MagicMock(return_value={"trigger_words": []})

    result = command_registry.should_gordo_respond(
        {},
        "hola",
        "hola",
        {"chat": {"type": "private"}},
        {"ai_random_replies": False},
        None,
        load_bot_config_fn=load_config,
    )

    assert result is True
    load_config.assert_called_once_with()
    assert "using Python fallback" in caplog.text
