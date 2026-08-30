"""Temporary bridge protocol for typed Telegram input parsing."""

from __future__ import annotations

from typing import Protocol, cast

from api.core.rust_bridge import load_rust_bridge


class RustTelegramInput(Protocol):
    def telegram_extract_message_content(self, message_json: str) -> str: ...

    def telegram_is_group_chat_type(self, chat_type: str | None) -> bool: ...

    def telegram_normalize_numeric_id(self, value_json: str) -> int | None: ...

    def telegram_extract_user_id(self, message_json: str) -> int | None: ...

    def telegram_format_user_identity(self, user_json: str) -> str: ...


def load_rust_telegram_input() -> RustTelegramInput | None:
    module = load_rust_bridge("RUST_TELEGRAM_INPUT_ENABLED")
    if module is None:
        return None
    return cast(RustTelegramInput, module)


__all__ = ["RustTelegramInput", "load_rust_telegram_input"]
