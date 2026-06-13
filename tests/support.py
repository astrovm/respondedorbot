from typing import Any, Dict, Mapping, Optional

import json  # noqa: F401
import os  # noqa: F401
import random  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
from datetime import datetime, timedelta, timezone  # noqa: F401

import pytest  # noqa: F401
import redis
import requests  # noqa: F401
from unittest.mock import ANY, MagicMock, patch  # noqa: F401

from api.core import config as config_module  # noqa: F401
from api import index
from api.ai.pipeline import remove_gordo_prefix  # noqa: F401
from api.bot.chat_settings import (  # noqa: F401
    CHAT_ADMIN_STATUS_TTL,
    CHAT_CONFIG_DEFAULTS,
    build_config_keyboard as _build_config_keyboard,
    build_config_text as _build_config_text,
    decode_redis_value as _decode_redis_value,
    get_chat_config as _get_chat_config,
    is_chat_admin as _is_chat_admin,
    set_chat_config as _set_chat_config,
)
from api.bot.command_registry import (
    parse_command as _parse_command,
    should_auto_process_media as _should_auto_process_media,
    should_gordo_respond as _should_gordo_respond,
)
from api.memory.state import (
    build_reply_context_text as _build_reply_context_text,
    format_user_message as _format_user_message,
    get_bot_message_metadata as _get_bot_message_metadata,
    save_bot_message_metadata as _save_bot_message_metadata,
    truncate_text as _truncate_text,
)
from api.billing.ai import AIMessageBilling  # noqa: F401
from api.links.service import LinkService
from api.services import bcra as bcra_service  # noqa: F401


def make_link_service(**overrides: Any) -> LinkService:
    dependencies = {
        "optional_redis_client": index.app_runtime.config.optional_redis,
        "hash_cache_key": index._hash_cache_key,
        "request_fn": index.request_with_ssl_fallback,
        "redis_get_json": index.redis_get_json,
        "redis_setex_json": index.redis_setex_json,
        "extract_video_id": index.extract_youtube_video_id,
        "fetch_transcript": index.get_youtube_transcript_context,
        "logger": index._logger,
        "format_log_context": index.format_log_context,
        "metadata_ttl": index.TTL_LINK_METADATA,
        "metadata_max_bytes": index.LINK_METADATA_MAX_BYTES,
        "max_links": index.MAX_LINKS_IN_MESSAGE,
    }
    dependencies.update(overrides)
    return LinkService(**dependencies)


link_service = make_link_service()

convert_to_command = index.convert_to_command
config_redis = index.app_runtime.config.redis
check_provider_available = index.app_runtime.providers.is_scope_available
extract_message_text = index.extract_message_text
complete_with_providers = index.app_runtime.providers.complete
get_provider_backoff_remaining = index.app_runtime.providers.get_backoff_remaining
handle_ai_response = index.app_runtime.responses.handle
handle_msg = index.app_runtime.handle_message
handle_config_command = index.handle_config_command
handle_callback_query = index.app_runtime.handle_callback_query
TTL_MEDIA_CACHE = index.TTL_MEDIA_CACHE
get_rulo = index.app_runtime.dollar.get_rulo
get_oil_price = index.app_runtime.stocks.get_oil_price


parse_command = _parse_command


def format_user_message(
    message: Dict[str, Any],
    message_text: str,
    reply_context: Optional[str] = None,
) -> str:
    return _format_user_message(message, message_text, reply_context)


def build_reply_context_text(message: Mapping[str, Any]) -> Optional[str]:
    return _build_reply_context_text(
        message,
        extract_message_text_fn=index.extract_message_text,
    )


def should_gordo_respond(
    commands,
    command,
    message_text,
    message,
    chat_config,
    reply_metadata,
):
    return _should_gordo_respond(
        commands,
        command,
        message_text,
        message,
        chat_config,
        reply_metadata,
        load_bot_config_fn=index.app_runtime.config.load_bot_config,
    )


def should_auto_process_media(commands, command, message_text, message):
    return _should_auto_process_media(commands, command, message_text, message)


truncate_text = _truncate_text


def get_chat_config(redis_client: redis.Redis, chat_id: str):
    return _get_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=index.chat_config_db_service,
        admin_reporter=index.app_runtime.admin.report,
        log_event=index._log_config_event,
    )


def set_chat_config(redis_client: redis.Redis, chat_id: str, **updates: Any):
    return _set_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=index.chat_config_db_service,
        admin_reporter=index.app_runtime.admin.report,
        log_event=index._log_config_event,
        **updates,
    )


def build_config_text(config: Mapping[str, Any]) -> str:
    return _build_config_text(config)


def build_config_keyboard(config: Mapping[str, Any]) -> Dict[str, Any]:
    return _build_config_keyboard(config)


def is_chat_admin(
    chat_id: str,
    user_id,
    *,
    redis_client: Optional[redis.Redis] = None,
):
    return _is_chat_admin(
        chat_id,
        user_id,
        redis_client=redis_client,
        optional_redis_client=index.app_runtime.config.optional_redis,
        telegram_request=index._telegram_request,
        log_event=index._log_config_event,
        redis_get_json_fn=index.redis_get_json,
        redis_setex_json_fn=index.redis_setex_json,
    )


def save_bot_message_metadata(
    redis_client: redis.Redis,
    chat_id: str,
    message_id,
    metadata: Mapping[str, Any],
    *,
    ttl: int = index.BOT_MESSAGE_META_TTL,
):
    return _save_bot_message_metadata(
        redis_client,
        chat_id,
        message_id,
        metadata,
        admin_reporter=index.app_runtime.admin.report,
        ttl=ttl,
    )


def get_bot_message_metadata(redis_client: redis.Redis, chat_id: str, message_id):
    return _get_bot_message_metadata(
        redis_client,
        chat_id,
        message_id,
        admin_reporter=index.app_runtime.admin.report,
        decode_redis_value=_decode_redis_value,
    )


RAW_TOOL_LEAKS = (
    'web_fetch(',
    '<｜｜DSML｜｜tool_calls>',
    '"tool_calls"',
    '"function_call"',
    '"arguments":',
)


def assert_no_raw_tool_syntax(text: str) -> None:
    for leak in RAW_TOOL_LEAKS:
        assert leak not in text


def make_mock_config_redis():
    mock_redis = MagicMock()
    return mock_redis


def make_ai_message_billing(
    *,
    command: str = "/ask",
    chat_id: str = "557",
    chat_type: str = "private",
    user_id: int = 101,
    numeric_chat_id: int = 557,
    **overrides: Any,
) -> AIMessageBilling:
    kwargs = {
        "credits_db_service": MagicMock(),
        "admin_reporter": MagicMock(),
        "gen_random_fn": lambda _: "random",
        "build_insufficient_credits_message_fn": lambda **_: "insufficient",
        "maybe_grant_onboarding_credits_fn": lambda _user_id: None,
        "command": command,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "user_id": user_id,
        "numeric_chat_id": numeric_chat_id,
        "message": {"from": {"first_name": "Ana"}},
    }
    kwargs.update(overrides)
    return AIMessageBilling(**kwargs)


__all__ = [name for name in globals() if not name.startswith("_")] + [
    "_decode_redis_value"
]
