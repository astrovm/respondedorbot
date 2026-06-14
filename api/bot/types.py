"""Typed Telegram payloads accepted at the application boundary."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class TelegramUser(TypedDict):
    id: int
    username: NotRequired[str]
    first_name: NotRequired[str]
    last_name: NotRequired[str]


class TelegramChat(TypedDict):
    id: int | str
    type: str
    title: NotRequired[str]


TelegramMessage = TypedDict(
    "TelegramMessage",
    {
        "message_id": int,
        "chat": TelegramChat,
        "from": TelegramUser,
        "text": str,
        "caption": str,
        "reply_to_message": dict[str, Any],
        "photo": list[dict[str, Any]],
        "voice": dict[str, Any],
        "audio": dict[str, Any],
        "video": dict[str, Any],
        "video_note": dict[str, Any],
        "sticker": dict[str, Any],
        "poll": dict[str, Any],
        "successful_payment": dict[str, Any],
    },
    total=False,
)


TelegramCallbackQuery = TypedDict(
    "TelegramCallbackQuery",
    {
        "id": str,
        "data": str,
        "from": TelegramUser,
        "message": TelegramMessage,
    },
    total=False,
)


class TelegramUpdate(TypedDict, total=False):
    message: TelegramMessage
    callback_query: TelegramCallbackQuery
    pre_checkout_query: dict[str, Any]
