"""Chat history, reply context, and bot metadata helpers."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Union, cast

import redis


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
DecodeRedisValue = Callable[[Any], Optional[str]]
ExtractMessageText = Callable[[Dict[str, Any]], str]


BOT_MESSAGE_META_PREFIX = "bot_message_meta:"
BOT_MESSAGE_META_TTL = 3 * 24 * 60 * 60


def truncate_text(text: Optional[str], max_length: int = 512) -> str:
    """Truncate text to max_length and add ellipsis if needed."""

    if text is None:
        return ""
    if max_length <= 0:
        return ""
    if max_length <= 3:
        return "." * max_length
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def save_message_to_redis(
    chat_id: str,
    message_id: str,
    text: str,
    redis_client: redis.Redis,
    *,
    admin_reporter: AdminReporter,
) -> None:
    """Persist a chat message while deduplicating message ids."""

    try:
        chat_history_key = f"chat_history:{chat_id}"
        message_ids_key = f"chat_message_ids:{chat_id}"

        if redis_client.sismember(message_ids_key, message_id):
            return

        history_entry = json.dumps(
            {
                "id": message_id,
                "text": truncate_text(text),
                "timestamp": int(time.time()),
            }
        )

        pipe = redis_client.pipeline()
        pipe.lpush(chat_history_key, history_entry)
        pipe.sadd(message_ids_key, message_id)
        pipe.ltrim(chat_history_key, 0, 31)
        pipe.lrange(chat_history_key, 0, -1)
        results = pipe.execute()

        message_entries = results[-1]
        valid_ids = set()
        for entry in message_entries:
            try:
                msg = json.loads(entry)
                valid_ids.add(msg["id"])
            except (json.JSONDecodeError, KeyError):
                continue

        try:
            current_ids_set = redis_client.smembers(message_ids_key)
            current_ids = list(cast(Set[str], current_ids_set)) if current_ids_set else []
            to_remove = [entry_id for entry_id in current_ids if entry_id not in valid_ids]
        except Exception:
            to_remove = []

        if to_remove:
            redis_client.srem(message_ids_key, *to_remove)

    except Exception as error:
        admin_reporter(
            f"Redis save message error: {error}",
            error,
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text_length": len(text),
            },
        )


def get_chat_history(
    chat_id: str,
    redis_client: redis.Redis,
    *,
    admin_reporter: AdminReporter,
    max_messages: int = 8,
) -> List[Dict[str, Any]]:
    """Return the latest chat history in chronological order."""

    try:
        chat_history_key = f"chat_history:{chat_id}"
        history: List[str] = cast(
            List[str], redis_client.lrange(chat_history_key, 0, max_messages - 1)
        )
        if not history:
            return []

        messages: List[Dict[str, Any]] = []
        for entry in history:
            try:
                msg = json.loads(entry)
                is_bot = str(msg["id"]).startswith("bot_")
                msg["role"] = "assistant" if is_bot else "user"
                messages.append(msg)
            except json.JSONDecodeError as decode_error:
                admin_reporter(
                    f"JSON decode error in chat history: {decode_error}",
                    decode_error,
                    {"chat_id": chat_id, "entry": entry},
                )

        return list(reversed(messages))
    except Exception as error:
        admin_reporter(
            f"Error retrieving chat history: {error}",
            error,
            {"chat_id": chat_id, "max_messages": max_messages},
        )
        return []


def bot_message_meta_key(chat_id: str, message_id: Union[str, int]) -> str:
    """Return the Redis key for persisted bot message metadata."""

    return f"{BOT_MESSAGE_META_PREFIX}{chat_id}:{message_id}"


def save_bot_message_metadata(
    redis_client: redis.Redis,
    chat_id: str,
    message_id: Union[str, int],
    metadata: Mapping[str, Any],
    *,
    admin_reporter: AdminReporter,
    ttl: int = BOT_MESSAGE_META_TTL,
) -> None:
    """Persist lightweight metadata about a sent bot message."""

    try:
        redis_client.setex(
            bot_message_meta_key(chat_id, message_id),
            ttl,
            json.dumps(dict(metadata)),
        )
    except Exception as error:
        admin_reporter(
            "Error saving bot message metadata",
            error,
            {"chat_id": chat_id, "message_id": message_id},
        )


def get_bot_message_metadata(
    redis_client: redis.Redis,
    chat_id: str,
    message_id: Union[str, int],
    *,
    admin_reporter: AdminReporter,
    decode_redis_value: DecodeRedisValue,
) -> Optional[Dict[str, Any]]:
    """Load persisted bot message metadata when available."""

    try:
        raw_value = redis_client.get(bot_message_meta_key(chat_id, message_id))
        serialized = decode_redis_value(raw_value)
        if serialized:
            try:
                loaded = json.loads(serialized)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                return loaded
    except Exception as error:
        admin_reporter(
            "Error loading bot message metadata",
            error,
            {"chat_id": chat_id, "message_id": message_id},
        )
    return None


def format_user_identity(user: Mapping[str, Any]) -> str:
    """Build a display name for a Telegram user."""

    first_name = "" if user.get("first_name") is None else str(user.get("first_name", ""))
    username = "" if user.get("username") is None else str(user.get("username", ""))
    return first_name + (f" ({username})" if username else "")


def describe_replied_message(
    reply_msg: Mapping[str, Any],
    *,
    extract_message_text_fn: ExtractMessageText,
) -> Optional[str]:
    """Generate a short description for a replied-to message."""

    reply_text = extract_message_text_fn(cast(Dict[str, Any], reply_msg))
    if reply_text:
        return reply_text
    if reply_msg.get("photo"):
        return "una foto sin texto"
    if reply_msg.get("sticker"):
        sticker = cast(Mapping[str, Any], reply_msg.get("sticker", {}))
        emoji_char = sticker.get("emoji")
        if emoji_char:
            return f"un sticker {emoji_char}"
    if reply_msg.get("voice"):
        return "un audio de voz"
    if reply_msg.get("audio"):
        return "un archivo de audio"
    if reply_msg.get("video"):
        return "un video"
    if reply_msg.get("document"):
        return "un archivo adjunto"
    return None


def build_reply_context_text(
    message: Mapping[str, Any],
    *,
    extract_message_text_fn: ExtractMessageText,
) -> Optional[str]:
    """Return contextual text describing the message being replied to."""

    reply_msg = message.get("reply_to_message") if message else None
    if not isinstance(reply_msg, Mapping):
        return None

    reply_description = describe_replied_message(
        reply_msg, extract_message_text_fn=extract_message_text_fn
    )
    if not reply_description:
        return None

    reply_user = format_user_identity(cast(Mapping[str, Any], reply_msg.get("from", {}))).strip()
    if reply_user:
        return f"{reply_user}: {reply_description}"
    return reply_description


def format_user_message(
    message: Dict[str, Any],
    message_text: str,
    reply_context: Optional[str] = None,
) -> str:
    """Format a user message with author and reply context."""

    formatted_user = format_user_identity(message.get("from", {}))
    if reply_context:
        if formatted_user:
            return f"{formatted_user} (en respuesta a {reply_context}): {message_text}"
        return f"(en respuesta a {reply_context}): {message_text}"
    return f"{formatted_user}: {message_text}"


__all__ = [
    "BOT_MESSAGE_META_TTL",
    "build_reply_context_text",
    "format_user_message",
    "get_bot_message_metadata",
    "get_chat_history",
    "save_bot_message_metadata",
    "save_message_to_redis",
    "truncate_text",
]
