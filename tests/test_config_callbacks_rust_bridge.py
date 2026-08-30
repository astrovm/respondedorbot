from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from api.bot import callbacks
from api.bot.callbacks import CallbackConfigDeps, ConfigUpdateContext


class _FakeRustConfigCallbacks:
    def __init__(self, evaluation: dict[str, Any] | Exception) -> None:
        self.evaluation = evaluation
        self.input: dict[str, Any] | None = None

    def evaluate_config_callback(self, input_json: str) -> str:
        self.input = json.loads(input_json)
        if isinstance(self.evaluation, Exception):
            raise self.evaluation
        return json.dumps(self.evaluation)


def _context(set_chat_config: MagicMock) -> ConfigUpdateContext:
    return ConfigUpdateContext(
        chat_id="synthetic-chat",
        callback_id="synthetic-callback",
        deps=CallbackConfigDeps(
            set_chat_config=set_chat_config,
            coerce_bool=lambda value, *, default: value if isinstance(value, bool) else default,
            guard_callback=MagicMock(return_value=False),
            log_event=MagicMock(),
            timezone_offset_min=-12,
            timezone_offset_max=14,
        ),
    )


def test_config_callback_rust_update_executes_one_persistence_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustConfigCallbacks(
        {"kind": "set_toggle", "field": "ai_random_replies", "value": False}
    )
    monkeypatch.setattr(callbacks, "_load_rust_config_callbacks", lambda: rust)
    set_chat_config = MagicMock(return_value={"ai_random_replies": False})
    context = _context(set_chat_config)

    result = callbacks.update_callback_config(
        {"ai_random_replies": True},
        "random",
        "toggle",
        context=context,
    )

    assert result == ({"ai_random_replies": False}, False)
    set_chat_config.assert_called_once_with(
        "synthetic-chat",
        ai_random_replies=False,
    )
    assert rust.input is not None
    assert rust.input["current_toggle"] is True


def test_config_callback_rust_guard_preserves_callback_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustConfigCallbacks({"kind": "guard_current"})
    monkeypatch.setattr(callbacks, "_load_rust_config_callbacks", lambda: rust)
    context = _context(MagicMock())
    config = {"timezone_offset": -3}

    result = callbacks.update_callback_config(
        config,
        "timezone",
        "current",
        context=context,
    )

    assert result == (config, True)
    cast(MagicMock, context.deps.guard_callback).assert_called_once_with(
        "synthetic-callback", True
    )


def test_config_callback_falls_back_after_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustConfigCallbacks(ValueError("synthetic bridge failure"))
    monkeypatch.setattr(callbacks, "_load_rust_config_callbacks", lambda: rust)
    set_chat_config = MagicMock(return_value={"language": "en"})

    result = callbacks.update_callback_config(
        {"language": "es"},
        "language",
        "en",
        context=_context(set_chat_config),
    )

    assert result == ({"language": "en"}, False)
    set_chat_config.assert_called_once_with("synthetic-chat", language="en")
