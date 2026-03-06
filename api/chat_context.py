"""Shared Telegram chat and user identity helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def is_group_chat_type(chat_type: Optional[str]) -> bool:
    """Return True for Telegram group and supergroup chats."""

    return str(chat_type) in {"group", "supergroup"}


def extract_numeric_chat_id(chat_id: Any) -> Optional[int]:
    """Normalize a Telegram chat id to int when possible."""

    try:
        return int(chat_id)
    except (TypeError, ValueError):
        return None


def extract_user_id(message: Mapping[str, Any]) -> Optional[int]:
    """Read a Telegram sender id from a message-like mapping."""

    user = message.get("from") if message else None
    if not isinstance(user, Mapping):
        return None
    try:
        return int(user.get("id"))
    except (TypeError, ValueError):
        return None


def format_user_identity(user: Mapping[str, Any]) -> str:
    """Build a short display name for a Telegram user."""

    first_name = "" if user.get("first_name") is None else str(user.get("first_name", ""))
    username = "" if user.get("username") is None else str(user.get("username", ""))
    return first_name + (f" ({username})" if username else "")

