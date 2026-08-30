from __future__ import annotations

import json
from unittest.mock import MagicMock

from api.bot.callbacks import _callback_context


class _Bridge:
    def __init__(self, result: object) -> None:
        self.result = result

    def telegram_parse_callback_context(self, callback_json: str) -> str:
        assert json.loads(callback_json)["data"] == "cfg:language:en"
        if isinstance(self.result, Exception):
            raise self.result
        return json.dumps(self.result)


def _deps() -> MagicMock:
    deps = MagicMock()
    deps.guard_callback.return_value = False
    return deps


def test_callback_context_uses_validated_rust_result(monkeypatch) -> None:
    bridge = _Bridge(
        {
            "kind": "context",
            "context": {
                "callback_id": "callback-1",
                "data": "cfg:language:en",
                "chat_id": "-10042",
                "chat_type": "supergroup",
                "message_id": 7,
                "user_id": 99,
                "user_language_code": "es",
                "route": "config",
            },
        }
    )
    monkeypatch.setattr("api.bot.callbacks.load_rust_telegram_callbacks", lambda: bridge)
    deps = _deps()
    user = {"id": 99, "language_code": "es"}

    context = _callback_context(
        {
            "id": "callback-1",
            "data": "cfg:language:en",
            "message": {
                "message_id": "7",
                "chat": {"id": -10042, "type": "supergroup"},
            },
            "from": user,
        },
        deps,
    )

    assert context is not None
    assert context.callback_id == "callback-1"
    assert context.chat_id == "-10042"
    assert context.message_id == 7
    assert context.user is user
    deps.guard_callback.assert_not_called()


def test_callback_context_applies_rust_guard_outcome(monkeypatch) -> None:
    bridge = _Bridge({"kind": "guard"})
    monkeypatch.setattr("api.bot.callbacks.load_rust_telegram_callbacks", lambda: bridge)
    deps = _deps()

    assert (
        _callback_context(
            {
                "id": "callback-1",
                "data": "cfg:language:en",
            },
            deps,
        )
        is None
    )
    deps.guard_callback.assert_called_once_with("callback-1", True)


def test_callback_context_falls_back_when_bridge_result_is_invalid(monkeypatch) -> None:
    bridge = _Bridge({"kind": "context", "context": {"data": 7}})
    monkeypatch.setattr("api.bot.callbacks.load_rust_telegram_callbacks", lambda: bridge)
    deps = _deps()

    context = _callback_context(
        {
            "id": "callback-1",
            "data": "cfg:language:en",
            "message": {
                "message_id": 7,
                "chat": {"id": -10042, "type": "supergroup"},
            },
            "from": {"id": 99},
        },
        deps,
    )

    assert context is not None
    assert context.data == "cfg:language:en"
    assert context.message_id == 7
