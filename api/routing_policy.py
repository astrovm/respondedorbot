from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Mapping, Optional

from api.command_registry import CommandTuple


@dataclass(frozen=True)
class RoutingPolicy:
    base_policy: Callable[..., bool]
    has_ai_credits_for_random_reply: Callable[[Mapping[str, Any]], bool]
    load_bot_config_fn: Callable[[], Mapping[str, Any]]

    def should_respond(
        self,
        commands: Mapping[str, CommandTuple],
        command: str,
        message_text: str,
        message: Mapping[str, Any],
        chat_config: Mapping[str, Any],
        reply_metadata: Optional[Mapping[str, Any]],
    ) -> bool:
        should_respond = self.base_policy(
            commands,
            command,
            message_text,
            message,
            chat_config,
            reply_metadata,
            load_bot_config_fn=self.load_bot_config_fn,
        )
        if not should_respond:
            return False

        chat = message.get("chat") or {}
        chat_type = str(chat.get("type") or "")
        if chat_type == "private" or command in commands:
            return True

        bot_username = str(environ.get("TELEGRAM_USERNAME") or "").strip()
        if bot_username and f"@{bot_username}" in message_text.lower():
            return True

        reply = message.get("reply_to_message") or {}
        if (
            isinstance(reply, Mapping)
            and str((reply.get("from") or {}).get("username") or "") == bot_username
        ):
            return True

        return self.has_ai_credits_for_random_reply(message)


__all__ = ["RoutingPolicy"]
