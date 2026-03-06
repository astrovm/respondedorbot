from unittest.mock import ANY, MagicMock, patch
import json
import os
import random
import re
import requests
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, cast

import pytest
import redis
from flask import Flask, request

from api import config as config_module
from api import index
from api.agent import AGENT_THOUGHT_CHAR_LIMIT, AGENT_THOUGHT_DISPLAY_LIMIT
from api.ai_pipeline import remove_gordo_prefix
from api.chat_settings import (
    CHAT_ADMIN_STATUS_TTL,
    CHAT_CONFIG_DEFAULTS,
    build_config_keyboard as _build_config_keyboard,
    build_config_text as _build_config_text,
    decode_redis_value as _decode_redis_value,
    get_chat_config as _get_chat_config,
    is_chat_admin as _is_chat_admin,
    set_chat_config as _set_chat_config,
)
from api.command_registry import (
    parse_command as _parse_command,
    should_auto_process_media as _should_auto_process_media,
    should_gordo_respond as _should_gordo_respond,
)
from api.message_state import (
    build_reply_context_text as _build_reply_context_text,
    format_user_message as _format_user_message,
    get_bot_message_metadata as _get_bot_message_metadata,
    save_bot_message_metadata as _save_bot_message_metadata,
    truncate_text as _truncate_text,
)
from api.services import bcra as bcra_service


app = Flask(__name__)

responder = index.responder
convert_to_command = index.convert_to_command
config_redis = index.config_redis
check_global_rate_limit = index.check_global_rate_limit
extract_message_text = index.extract_message_text
fetch_url_content = index.fetch_url_content
web_search = index.web_search
search_command = index.search_command
parse_tool_call = index.parse_tool_call
execute_tool = index.execute_tool
complete_with_providers = index.complete_with_providers
get_groq_ai_response = index.get_groq_ai_response
get_groq_compound_response = index.get_groq_compound_response
get_provider_backoff_remaining = index.get_provider_backoff_remaining
handle_ai_response = index.handle_ai_response
handle_msg = index.handle_msg
replace_links = index.replace_links
handle_config_command = index.handle_config_command
handle_callback_query = index.handle_callback_query
ensure_callback_updates_enabled = index.ensure_callback_updates_enabled
TTL_MEDIA_CACHE = index.TTL_MEDIA_CACHE
get_rulo = index.get_rulo
is_repetitive_thought = index.is_repetitive_thought
find_repetitive_recent_thought = index.find_repetitive_recent_thought
build_agent_retry_prompt = index.build_agent_retry_prompt
build_agent_fallback_entry = index.build_agent_fallback_entry
summarize_recent_agent_topics = index.summarize_recent_agent_topics
agent_sections_are_valid = index.agent_sections_are_valid
get_agent_retry_hint = index.get_agent_retry_hint
should_force_web_search = index.should_force_web_search
should_search_previous_query = index.should_search_previous_query
should_use_groq_compound_tools = index.should_use_groq_compound_tools
get_groq_compound_enabled_tools = index.get_groq_compound_enabled_tools


def parse_command(message_text: str, bot_name: str):
    return _parse_command(message_text, bot_name)


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
        load_bot_config_fn=index.load_bot_config,
    )


def should_auto_process_media(commands, command, message_text, message):
    return _should_auto_process_media(commands, command, message_text, message)


def truncate_text(text: Optional[str], max_length: int = 512) -> str:
    return _truncate_text(text, max_length)


def get_chat_config(redis_client: redis.Redis, chat_id: str):
    return _get_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=index.chat_config_db_service,
        admin_reporter=index.admin_report,
        log_event=index._log_config_event,
    )


def set_chat_config(redis_client: redis.Redis, chat_id: str, **updates: Any):
    return _set_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=index.chat_config_db_service,
        admin_reporter=index.admin_report,
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
        optional_redis_client=index._optional_redis_client,
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
        admin_reporter=index.admin_report,
        ttl=ttl,
    )


def get_bot_message_metadata(redis_client: redis.Redis, chat_id: str, message_id):
    return _get_bot_message_metadata(
        redis_client,
        chat_id,
        message_id,
        admin_reporter=index.admin_report,
        decode_redis_value=_decode_redis_value,
    )


__all__ = [name for name in globals() if not name.startswith("_")] + ["_decode_redis_value"]
