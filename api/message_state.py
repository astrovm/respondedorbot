"""Chat history, reply context, and bot metadata helpers."""

from __future__ import annotations

import json
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Union,
    cast,
)

import redis

from api.chat_context import format_user_identity
from api.services.maintenance import CHAT_STATE_TTL

AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
DecodeRedisValue = Callable[[Any], Optional[str]]
ExtractMessageText = Callable[[Dict[str, Any]], str]


BOT_MESSAGE_META_PREFIX = "bot_message_meta:"
BOT_MESSAGE_META_TTL = 3 * 24 * 60 * 60
CHAT_HISTORY_MAX_MESSAGES = 100
CHAT_SUMMARY_TTL = CHAT_STATE_TTL
CHAT_SEARCH_INDEX = "idx:chat_messages"

_SEARCH_INDEX_READY = False


def truncate_text(text: Optional[str], max_length: int = 1024) -> str:
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


def _search_doc_key(chat_id: str, message_id: str) -> str:
    return f"chatmsg:{chat_id}:{message_id}"


def _summary_key(chat_id: str) -> str:
    return f"chat_summary:{chat_id}"


def _user_summary_key(chat_id: str) -> str:
    return f"chat_user_summary:{chat_id}"


def _compacted_until_key(chat_id: str) -> str:
    return f"chat_compacted_until:{chat_id}"


def _decode_redis_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _ensure_search_index(redis_client: redis.Redis) -> None:
    global _SEARCH_INDEX_READY
    if _SEARCH_INDEX_READY:
        return
    try:
        redis_client.execute_command(
            "FT.CREATE",
            CHAT_SEARCH_INDEX,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            "chatmsg:",
            "SCHEMA",
            "chat_id",
            "TAG",
            "role",
            "TAG",
            "user_id",
            "TAG",
            "reply_to_message_id",
            "TAG",
            "mentions_bot",
            "TAG",
            "username",
            "TEXT",
            "text",
            "TEXT",
            "timestamp",
            "NUMERIC",
            "SORTABLE",
        )
    except Exception as error:
        if "Index already exists" not in str(error):
            raise
    _SEARCH_INDEX_READY = True


def _escape_search_text(query_text: str) -> str:
    tokens = [re.sub(r"[^\w@.-]", "", token) for token in str(query_text or "").split()]
    tokens = [token.replace("@", "\\@") for token in tokens if token]
    return " ".join(tokens)


def _escape_tag_value(value: str) -> str:
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", str(value or ""))


def _parse_search_result_row(key: Any, fields: Any) -> Dict[str, Any]:
    parsed_fields = list(fields or [])
    data: Dict[str, Any] = {
        "key": _decode_redis_text(key) or "",
    }
    for idx in range(0, len(parsed_fields), 2):
        field = _decode_redis_text(parsed_fields[idx]) or ""
        value = _decode_redis_text(parsed_fields[idx + 1]) if idx + 1 < len(parsed_fields) else None
        if field:
            data[field] = value
    return data


def get_chat_summary(redis_client: redis.Redis, chat_id: str) -> Optional[str]:
    return _decode_redis_text(redis_client.get(_summary_key(chat_id)))


def save_chat_summary(redis_client: redis.Redis, chat_id: str, summary: str) -> None:
    redis_client.setex(_summary_key(chat_id), CHAT_SUMMARY_TTL, summary)


def get_user_chat_summary(redis_client: redis.Redis, chat_id: str) -> Optional[str]:
    return _decode_redis_text(redis_client.get(_user_summary_key(chat_id)))


def save_user_chat_summary(redis_client: redis.Redis, chat_id: str, summary: str) -> None:
    redis_client.setex(_user_summary_key(chat_id), CHAT_SUMMARY_TTL, summary)


def get_chat_compacted_until(redis_client: redis.Redis, chat_id: str) -> Optional[str]:
    return _decode_redis_text(redis_client.get(_compacted_until_key(chat_id)))


def save_chat_compacted_until(redis_client: redis.Redis, chat_id: str, marker: str) -> None:
    redis_client.setex(_compacted_until_key(chat_id), CHAT_SUMMARY_TTL, marker)


def fetch_chat_messages_for_compaction(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    limit: int = 500,
    admin_reporter: Optional[AdminReporter] = None,
) -> List[Dict[str, Any]]:
    try:
        _ensure_search_index(redis_client)
        query = f"@chat_id:{{{_escape_tag_value(chat_id)}}}"
        raw = redis_client.execute_command(
            "FT.SEARCH",
            CHAT_SEARCH_INDEX,
            query,
            "DIALECT",
            "2",
            "SORTBY",
            "timestamp",
            "ASC",
            "LIMIT",
            "0",
            str(limit),
        )
        if not isinstance(raw, list) or len(raw) <= 1:
            return []
        rows: List[Dict[str, Any]] = []
        for idx in range(1, len(raw), 2):
            row = _parse_search_result_row(raw[idx], raw[idx + 1] if idx + 1 < len(raw) else [])
            row["timestamp"] = int(row.get("timestamp") or 0)
            rows.append(row)
        return rows
    except Exception as error:
        if admin_reporter is not None:
            admin_reporter(
                f"Error fetching chat messages for compaction: {error}",
                error,
                {"chat_id": chat_id, "limit": limit},
            )
        return []


def save_message_to_redis(
    chat_id: str,
    message_id: str,
    text: str,
    redis_client: redis.Redis,
    *,
    admin_reporter: AdminReporter,
    role: Optional[str] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    mentions_bot: bool = False,
) -> None:
    """Persist a chat message while deduplicating message ids."""

    try:
        chat_history_key = f"chat_history:{chat_id}"
        message_ids_key = f"chat_message_ids:{chat_id}"

        if redis_client.sismember(message_ids_key, message_id):
            return

        effective_role = role or ("assistant" if str(message_id).startswith("bot_") else "user")
        history_entry = json.dumps(
            {
                "id": message_id,
                "text": truncate_text(text),
                "timestamp": int(time.time()),
                "role": effective_role,
            }
        )

        pipe = redis_client.pipeline()
        try:
            _ensure_search_index(redis_client)
        except Exception:
            pass
        pipe.lpush(chat_history_key, history_entry)
        pipe.sadd(message_ids_key, message_id)
        pipe.ltrim(chat_history_key, 0, max(0, CHAT_HISTORY_MAX_MESSAGES * 2 - 1))
        pipe.expire(chat_history_key, CHAT_STATE_TTL)
        pipe.expire(message_ids_key, CHAT_STATE_TTL)
        pipe.hset(
            _search_doc_key(chat_id, message_id),
            mapping={
                "chat_id": chat_id,
                "message_id": message_id,
                "role": effective_role,
                "user_id": str(user_id or ""),
                "username": str(username or ""),
                "text": truncate_text(text),
                "timestamp": int(time.time()),
                "reply_to_message_id": str(reply_to_message_id or ""),
                "mentions_bot": "1" if mentions_bot else "0",
            },
        )
        pipe.expire(_search_doc_key(chat_id, message_id), CHAT_STATE_TTL)
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
            current_ids = (
                list(cast(Set[str], current_ids_set)) if current_ids_set else []
            )
            to_remove = [
                entry_id for entry_id in current_ids if entry_id not in valid_ids
            ]
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
    max_messages: int = CHAT_HISTORY_MAX_MESSAGES,
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
                if "role" not in msg:
                    is_bot = str(msg["id"]).startswith("bot_")
                    msg["role"] = "assistant" if is_bot else "user"
                messages.append(msg)
            except json.JSONDecodeError as decode_error:
                admin_reporter(
                    f"JSON decode error in chat history: {decode_error}",
                    decode_error,
                    {"chat_id": chat_id, "entry": entry},
                )

        return sorted(messages, key=lambda m: m.get("timestamp", 0))
    except Exception as error:
        admin_reporter(
            f"Error retrieving chat history: {error}",
            error,
            {"chat_id": chat_id, "max_messages": max_messages},
        )
        return []


def search_chat_history(
    redis_client: redis.Redis,
    chat_id: str,
    query_text: str,
    *,
    reply_to_message_id: Optional[str] = None,
    limit: int = 8,
    exclude_message_ids: Optional[Set[str]] = None,
    admin_reporter: Optional[AdminReporter] = None,
) -> List[Dict[str, Any]]:
    search_text = _escape_search_text(query_text)
    if not search_text:
        return []
    try:
        _ensure_search_index(redis_client)
        query = f"@chat_id:{{{_escape_tag_value(chat_id)}}} {search_text}"
        raw = redis_client.execute_command(
            "FT.SEARCH",
            CHAT_SEARCH_INDEX,
            query,
            "DIALECT",
            "2",
            "SORTBY",
            "timestamp",
            "DESC",
            "LIMIT",
            "0",
            str(max(limit * 3, 10)),
        )
        if not isinstance(raw, list) or len(raw) <= 1:
            return []
        results: List[Dict[str, Any]] = []
        excluded = exclude_message_ids or set()
        query_tokens = set(search_text.lower().split())
        for idx in range(1, len(raw), 2):
            row = _parse_search_result_row(raw[idx], raw[idx + 1] if idx + 1 < len(raw) else [])
            message_id = str(row.get("message_id") or "")
            if message_id and message_id in excluded:
                continue
            row["timestamp"] = int(row.get("timestamp") or 0)
            row["_reply_score"] = 1 if reply_to_message_id and row.get("reply_to_message_id") == reply_to_message_id else 0
            text_tokens = set(str(row.get("text") or "").lower().split())
            row["_overlap_score"] = len(query_tokens & text_tokens)
            results.append(row)
        results.sort(
            key=lambda item: (
                int(item.get("_reply_score") or 0),
                int(item.get("_overlap_score") or 0),
                int(item.get("timestamp") or 0),
            ),
            reverse=True,
        )
        return results[:limit]
    except Exception as error:
        if admin_reporter is not None:
            admin_reporter(
                f"Error searching chat history: {error}",
                error,
                {"chat_id": chat_id, "query_text": query_text, "limit": limit},
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

    reply_user = format_user_identity(
        cast(Mapping[str, Any], reply_msg.get("from", {}))
    ).strip()
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
    "CHAT_HISTORY_MAX_MESSAGES",
    "build_reply_context_text",
    "format_user_message",
    "fetch_chat_messages_for_compaction",
    "get_bot_message_metadata",
    "get_chat_compacted_until",
    "get_chat_history",
    "get_chat_summary",
    "get_user_chat_summary",
    "save_chat_compacted_until",
    "save_chat_summary",
    "save_user_chat_summary",
    "save_bot_message_metadata",
    "save_message_to_redis",
    "search_chat_history",
    "truncate_text",
]
