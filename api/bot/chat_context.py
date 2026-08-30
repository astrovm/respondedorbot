"""Shared Telegram chat and user identity helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from api.core.logging import get_logger
from api.core.rust_telegram_input import load_rust_telegram_input


logger = get_logger(__name__)


def is_group_chat_type(chat_type: Optional[str]) -> bool:
    """Return True for Telegram group and supergroup chats."""

    rust = load_rust_telegram_input()
    if rust is not None and (chat_type is None or isinstance(chat_type, str)):
        try:
            return rust.telegram_is_group_chat_type(chat_type)
        except Exception:
            logger.exception(
                "Rust Telegram chat type parser failed; using Python fallback"
            )
    return str(chat_type) in {"group", "supergroup"}


def extract_numeric_chat_id(chat_id: Any) -> Optional[int]:
    """Normalize a Telegram chat id to int when possible."""

    rust = load_rust_telegram_input()
    if rust is not None:
        try:
            result = rust.telegram_normalize_numeric_id(
                json.dumps(chat_id, ensure_ascii=False, separators=(",", ":"))
            )
            if result is not None and (
                not isinstance(result, int) or isinstance(result, bool)
            ):
                raise ValueError("Rust Telegram chat id must be an integer")
            return result
        except Exception:
            logger.exception("Rust Telegram chat id parser failed; using Python fallback")
    try:
        return int(chat_id)
    except (TypeError, ValueError):
        return None


def extract_user_id(message: Mapping[str, Any]) -> Optional[int]:
    """Read a Telegram sender id from a message-like mapping."""

    rust = load_rust_telegram_input()
    if rust is not None:
        try:
            result = rust.telegram_extract_user_id(
                json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            )
            if result is not None and (
                not isinstance(result, int) or isinstance(result, bool)
            ):
                raise ValueError("Rust Telegram user id must be an integer")
            return result
        except Exception:
            logger.exception("Rust Telegram user id parser failed; using Python fallback")
    user = message.get("from") if message else None
    if not isinstance(user, Mapping):
        return None
    user_id = user.get("id")
    if user_id is None:
        return None
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


def format_user_identity(user: Mapping[str, Any]) -> str:
    """Build a short display name for a Telegram user."""

    rust = load_rust_telegram_input()
    if rust is not None:
        try:
            result = rust.telegram_format_user_identity(
                json.dumps(user, ensure_ascii=False, separators=(",", ":"))
            )
            if not isinstance(result, str):
                raise ValueError("Rust Telegram user identity must be text")
            return result
        except Exception:
            logger.exception(
                "Rust Telegram user identity parser failed; using Python fallback"
            )
    first_name = (
        "" if user.get("first_name") is None else str(user.get("first_name", ""))
    )
    username = "" if user.get("username") is None else str(user.get("username", ""))
    return first_name + (f" ({username})" if username else "")
