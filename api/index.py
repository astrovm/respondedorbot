from contextvars import ContextVar, Token
from cryptography.fernet import Fernet
from datetime import datetime, timedelta, timezone, date
from flask import Flask, Request, request
from html.parser import HTMLParser
from html import unescape
from math import log
from openai import OpenAI
from os import environ
from PIL import Image
from requests.exceptions import RequestException, SSLError
from typing import (
    Dict,
    List,
    Tuple,
    Callable,
    Union,
    Optional,
    Any,
    cast,
    Set,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    NamedTuple,
    TYPE_CHECKING,
    TypedDict,
    Literal,
)
import ast
import base64
import emoji
import hashlib
import io
import json
import random
import re
import redis
import requests
import time
import traceback
from pykakasi import kakasi
from openpyxl import load_workbook
from decimal import Decimal
import unicodedata
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urlunparse
from functools import lru_cache

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputParam
else:
    ResponseInputParam = Any  # type: ignore[assignment]

from api.utils import (
    fmt_num,
    fmt_signed_pct,
    local_cache_get,
    now_utc_iso,
    parse_date_string,
    parse_monetary_number,
    to_ddmmyy,
    to_es_number,
    update_local_cache,
)
from api.services.redis_helpers import (
    redis_get_json,
    redis_set_json,
    redis_setex_json,
)
from api.config import (
    config_redis as _config_config_redis,
    configure as configure_app_config,
    load_bot_config as _config_load_bot_config,
)
from api.ai_billing import (
    build_insufficient_credits_message as _billing_build_insufficient_credits_message,
    build_topup_keyboard as _billing_build_topup_keyboard,
    extract_numeric_chat_id as _billing_extract_numeric_chat_id,
    extract_user_id as _billing_extract_user_id,
    format_balance_command as _billing_format_balance_command,
    get_ai_billing_pack as _billing_get_ai_billing_pack,
    get_ai_billing_packs as _billing_get_ai_billing_packs,
    get_ai_credits_per_response as _billing_get_ai_credits_per_response,
    get_ai_onboarding_credits as _billing_get_ai_onboarding_credits,
    maybe_grant_onboarding_credits as _billing_maybe_grant_onboarding_credits,
    parse_topup_payload as _billing_parse_topup_payload,
)
from api.ai_pipeline import (
    clean_duplicate_response as _ai_clean_duplicate_response,
    handle_ai_response as _ai_handle_response,
    remove_gordo_prefix as _ai_remove_gordo_prefix,
)
from api.chat_settings import (
    CHAT_ADMIN_STATUS_TTL,
    CHAT_CONFIG_DEFAULTS,
    build_config_keyboard as _chat_build_config_keyboard,
    build_config_text as _chat_build_config_text,
    coerce_bool as _chat_coerce_bool,
    decode_redis_value as _chat_decode_redis_value,
    get_chat_config as _chat_get_chat_config,
    is_chat_admin as _chat_is_chat_admin,
    is_group_chat_type as _chat_is_group_chat_type,
    report_unauthorized_config_attempt as _chat_report_unauthorized_config_attempt,
    set_chat_config as _chat_set_chat_config,
)
from api.command_registry import (
    build_command_registry as _build_command_registry,
    parse_command as _command_parse_command,
    should_auto_process_media as _command_should_auto_process_media,
    should_gordo_respond as _command_should_gordo_respond,
)
from api.message_handler import MessageHandlerDeps, handle_msg as _handle_msg_impl
from api.message_state import (
    BOT_MESSAGE_META_TTL,
    build_reply_context_text as _state_build_reply_context_text,
    format_user_message as _state_format_user_message,
    get_bot_message_metadata as _state_get_bot_message_metadata,
    get_chat_history as _state_get_chat_history,
    save_bot_message_metadata as _state_save_bot_message_metadata,
    save_message_to_redis as _state_save_message_to_redis,
    truncate_text as _state_truncate_text,
)
from api.services import bcra as bcra_service
from api.services import chat_config_db as chat_config_db_service
from api.services import credits_db as credits_db_service
from api.agent import (
    AGENT_EMPTY_RESPONSE_FALLBACK,
    AGENT_LOOP_FALLBACK_PREFIX,
    AGENT_RECENT_THOUGHT_WINDOW,
    AGENT_REPETITION_ESCALATION_HINT,
    AGENT_REPETITION_RETRY_LIMIT,
    AGENT_REQUIRED_SECTIONS,
    AGENT_THOUGHTS_KEY,
    AGENT_THOUGHT_CHAR_LIMIT,
    AGENT_THOUGHT_DISPLAY_LIMIT,
    MAX_AGENT_THOUGHTS,
    build_agent_fallback_entry,
    build_agent_retry_messages,
    build_agent_retry_prompt,
    build_agent_thoughts_context_message,
    configure as configure_agent_memory,
    ensure_agent_response_text,
    extract_agent_keywords,
    extract_agent_section_content,
    find_repetitive_recent_thought,
    format_agent_thoughts,
    get_agent_retry_hint,
    get_agent_text_features,
    get_agent_thoughts,
    agent_sections_are_valid,
    is_empty_agent_thought_text,
    is_loop_fallback_text,
    is_repetitive_thought,
    normalize_agent_text,
    request_agent_response,
    save_agent_thought,
    summarize_recent_agent_topics,
)
from api.utils.http import request_with_ssl_fallback
from api.utils.links import (
    can_embed_url as _links_can_embed_url,
    is_social_frontend as _links_is_social_frontend,
    replace_links as _links_replace_links,
)


# TTL constants (seconds)
TTL_PRICE = 300  # 5 minutes
TTL_DOLLAR = 300  # 5 minutes
TTL_WEATHER = 1800  # 30 minutes
TTL_WEB_SEARCH = 300  # 5 minutes
TTL_WEB_FETCH = 300  # 5 minutes
TTL_POLYMARKET = 5  # 5 seconds
TTL_POLYMARKET_STREAM = 5  # 5 seconds for live price lookups
WEB_FETCH_MAX_BYTES = 250_000
WEB_FETCH_MIN_CHARS = 500
WEB_FETCH_MAX_CHARS = 8000
WEB_FETCH_DEFAULT_CHARS = 4000
TTL_MEDIA_CACHE = 7 * 24 * 60 * 60  # 7 days
TTL_HACKER_NEWS = 600  # 10 minutes
BA_TZ = timezone(timedelta(hours=-3))
GROQ_COMPOUND_DEFAULT_MODEL = "groq/compound"
GROQ_COMPOUND_DEFAULT_TOOLS = (
    "web_search",
    "code_interpreter",
    "visit_website",
    "browser_automation",
    "wolfram_alpha",
)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
AI_FALLBACK_MARKER = "[[AI_FALLBACK]]"


# Polymarket constants
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_ARGENTINA_ELECTION_SLUG = (
    "which-party-wins-most-seats-in-argentina-deputies-election"
)
POLYMARKET_ARGENTINA_SEATS_AFTER_SLUG = (
    "which-party-holds-the-most-seats-after-argentina-deputies-election"
)
POLYMARKET_PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"
POLYMARKET_STREAM_LOOKBACK_SECONDS = 60 * 30  # 30 minutes
POLYMARKET_STREAM_FIDELITY = 1  # minute buckets


FORCE_WEB_SEARCH_PATTERNS = [
    r"\bcuando\s+sale\b",
    r"\bfecha\b",
    r"\bestreno\b",
    r"\blanzamiento\b",
    r"\bultima(s)?\b",
    r"\bultimo(s)?\b",
    r"\bnoticia(s)?\b",
    r"\bnovedad(es)?\b",
    r"\bhoy\b",
    r"\b20(24|25|26)\b",
]

FORCE_WEB_SEARCH_PREVIOUS_PATTERNS = [
    r"\bbuscalo\b",
    r"\bbusca\s+eso\b",
    r"\bbusca\s+esto\b",
]


def should_use_groq_compound_tools() -> bool:
    """Return True when Groq Compound built-in tools are enabled."""

    return bool(environ.get("GROQ_API_KEY"))


def get_groq_compound_enabled_tools() -> List[str]:
    """Return the enabled Groq Compound tool list, sanitized by allowlist."""

    return list(GROQ_COMPOUND_DEFAULT_TOOLS)


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by removing accents and lowercasing for matching."""

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).lower()
    return normalized


def should_force_web_search(text: str) -> bool:
    """Return True when queries look like date/release/latest-news requests."""

    if not text:
        return False

    normalized = normalize_text_for_matching(text)

    return any(re.search(pattern, normalized) for pattern in FORCE_WEB_SEARCH_PATTERNS)


def should_search_previous_query(text: str) -> bool:
    """Return True when the user asks to search the previous topic."""

    if not text:
        return False

    normalized = normalize_text_for_matching(text)
    return any(
        re.search(pattern, normalized)
        for pattern in FORCE_WEB_SEARCH_PREVIOUS_PATTERNS
    )


def extract_user_message_from_context(text: str) -> str:
    """Extract the user-provided message from a context block."""

    if not text:
        return ""

    marker = "MENSAJE:"
    start_index = text.find(marker)
    if start_index == -1:
        return text.strip()

    message_block = text[start_index + len(marker) :]
    end_marker = "INSTRUCCIONES:"
    end_index = message_block.find(end_marker)
    if end_index != -1:
        message_block = message_block[:end_index]
    return message_block.strip()


def _fetch_polymarket_live_price(token_id: str) -> Optional[Tuple[float, Optional[int]]]:
    """Return the latest price and timestamp for a Polymarket CLOB token."""

    if not token_id:
        return None

    now = int(time.time())
    start_ts = max(0, now - POLYMARKET_STREAM_LOOKBACK_SECONDS)
    earliest_ts = max(0, start_ts - POLYMARKET_STREAM_LOOKBACK_SECONDS)

    response = cached_requests(
        POLYMARKET_PRICES_HISTORY_URL,
        {
            "startTs": start_ts,
            "market": token_id,
            "earliestTimestamp": earliest_ts,
            "fidelity": POLYMARKET_STREAM_FIDELITY,
        },
        None,
        TTL_POLYMARKET_STREAM,
        verify_ssl=False,
    )

    if not response or "data" not in response:
        return None

    history_data = response.get("data")

    if not isinstance(history_data, dict):
        return None

    history = history_data.get("history")

    if not isinstance(history, list) or not history:
        return None

    latest_entry = history[-1]
    price = latest_entry.get("p")
    timestamp = latest_entry.get("t")

    try:
        price_value = float(price)
    except (TypeError, ValueError):
        return None

    entry_timestamp: Optional[int]
    try:
        entry_timestamp = int(timestamp) if timestamp is not None else None
    except (TypeError, ValueError):
        entry_timestamp = None

    return price_value, entry_timestamp


def _fetch_criptoya_dollar_data(*, hourly_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Retrieve dollar rates from CriptoYa with shared cache semantics."""

    return cached_requests(
        "https://criptoya.com/api/dolar",
        None,
        None,
        TTL_DOLLAR,
        hourly_cache,
    )


def _find_previous_user_text(
    messages: List[Dict[str, Any]], latest_user_index: int
) -> str:
    for msg in reversed(messages[:latest_user_index]):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _run_forced_web_search(
    *,
    query: str,
    messages: List[Dict[str, Any]],
    system_message: Dict[str, Any],
    compound_system_message: Optional[Dict[str, Any]] = None,
) -> str:
    if not compound_system_message:
        return "la búsqueda web no está disponible en este momento"

    compound_messages = [{"role": "user", "content": query}]
    compound_response = get_groq_compound_response(
        compound_system_message, compound_messages
    )
    if compound_response:
        return compound_response

    return "no pude completar la búsqueda web ahora"


def config_redis(host=None, port=None, password=None):
    configure_app_config(admin_reporter=globals().get("admin_report"))
    return _config_config_redis(host=host, port=port, password=password)


def load_bot_config() -> Dict[str, Any]:
    configure_app_config(admin_reporter=globals().get("admin_report"))
    return _config_load_bot_config()


def _optional_redis_client(**kwargs: Any) -> Optional[redis.Redis]:
    """Return a Redis client when available, otherwise ``None``."""

    try:
        return config_redis(**kwargs)
    except Exception:
        return None


def _hash_cache_key(prefix: str, payload: Mapping[str, Any]) -> str:
    """Return a stable cache key composed of *prefix* and a SHA-256 hash."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


configure_agent_memory(redis_factory=config_redis, tz=BA_TZ)


_T = TypeVar("_T")




bcra_api_get = bcra_service.bcra_api_get
bcra_list_variables = bcra_service.bcra_list_variables
bcra_fetch_latest_variables = bcra_service.bcra_fetch_latest_variables
bcra_get_value_for_date = bcra_service.bcra_get_value_for_date
cache_bcra_variables = bcra_service.cache_bcra_variables
cache_currency_band_limits = bcra_service.cache_currency_band_limits
cache_mayorista_missing = bcra_service.cache_mayorista_missing
get_cached_bcra_variables = bcra_service.get_cached_bcra_variables
get_or_refresh_bcra_variables = bcra_service.get_or_refresh_bcra_variables
get_latest_itcrm_details = bcra_service.get_latest_itcrm_details
get_latest_itcrm_value = bcra_service.get_latest_itcrm_value
get_latest_itcrm_value_and_date = bcra_service.get_latest_itcrm_value_and_date
get_country_risk_summary = bcra_service.get_country_risk_summary
_parse_currency_band_rows = bcra_service._parse_currency_band_rows


def fetch_currency_band_limits() -> Optional[Dict[str, Any]]:
    """Proxy fetch helper to allow patching API dependencies in tests."""

    return bcra_service.fetch_currency_band_limits(
        list_variables_fn=bcra_list_variables,
        api_get_fn=bcra_api_get,
    )


def get_currency_band_limits() -> Optional[Dict[str, Any]]:
    """Proxy to BCRA service allowing tests to patch the fetcher."""

    return bcra_service.get_currency_band_limits(fetcher=fetch_currency_band_limits)


def format_bcra_variables(variables: Dict[str, Any]) -> str:
    """Proxy format helper so tests can patch dependent callables."""

    return bcra_service.format_bcra_variables(
        variables,
        band_fetcher=get_currency_band_limits,
        itcrm_getter=get_latest_itcrm_value_and_date,
        country_risk_getter=get_country_risk_summary,
    )


def calculate_tcrm_100(
    target_date: Optional[Union[str, datetime, date]] = None,
) -> Optional[float]:
    """Proxy calculator ensuring Redis helpers are patchable in tests."""

    return bcra_service.calculate_tcrm_100(
        target_date,
        config_redis_fn=config_redis,
        redis_get_json_fn=redis_get_json,
        redis_set_json_fn=redis_set_json,
        redis_setex_json_fn=redis_setex_json,
        bcra_get_value_for_date_fn=bcra_get_value_for_date,
        cache_mayorista_missing_fn=cache_mayorista_missing,
        itcrm_getter_fn=get_latest_itcrm_value_and_date,
    )


def get_cached_tcrm_100(
    hours_ago: int = 24, expiration_time: int = 300
) -> Tuple[Optional[float], Optional[float]]:
    """Proxy cache helper exposing injectable dependencies for tests."""

    return bcra_service.get_cached_tcrm_100(
        hours_ago,
        expiration_time,
        config_redis_fn=config_redis,
        redis_get_json_fn=redis_get_json,
        redis_set_json_fn=redis_set_json,
        redis_setex_json_fn=redis_setex_json,
        calculate_tcrm_fn=calculate_tcrm_100,
        get_latest_itcrm_fn=get_latest_itcrm_value_and_date,
        bcra_get_value_for_date_fn=bcra_get_value_for_date,
        cache_mayorista_missing_fn=cache_mayorista_missing,
        get_cache_history_fn=get_cache_history,
    )


def get_agent_memory_context() -> Optional[Dict[str, Any]]:
    thoughts = get_agent_thoughts()
    return build_agent_thoughts_context_message(thoughts)


def show_agent_thoughts() -> str:
    thoughts = get_agent_thoughts()
    visible = thoughts[:AGENT_THOUGHT_DISPLAY_LIMIT]
    return format_agent_thoughts(visible)


def can_embed_url(url: str) -> bool:
    """Wrapper to allow tests to monkeypatch embed detection."""

    return _links_can_embed_url(url)


def is_social_frontend(host: str) -> bool:
    """Expose social frontend check while keeping implementation in utils."""

    return _links_is_social_frontend(host)


def replace_links(text: str) -> Tuple[str, bool, List[str]]:
    """Delegate to utils helper while keeping embed checker injectable in tests."""

    return _links_replace_links(text, embed_checker=can_embed_url)

# Provider backoff windows (seconds)
GROQ_RATE_LIMIT_BACKOFF_SECONDS = 600  # wait 10 minutes after a rate limit response


_provider_backoff_until: Dict[str, float] = {}
_ai_provider_request_count: ContextVar[int] = ContextVar(
    "ai_provider_request_count", default=0
)


def _reset_ai_provider_request_count() -> Token:
    return _ai_provider_request_count.set(0)


def _restore_ai_provider_request_count(token: Token) -> None:
    _ai_provider_request_count.reset(token)


def _increment_ai_provider_request_count() -> None:
    _ai_provider_request_count.set(int(_ai_provider_request_count.get() or 0) + 1)


def _get_ai_provider_request_count() -> int:
    return int(_ai_provider_request_count.get() or 0)


def _set_provider_backoff(provider: str, duration: Optional[int]) -> None:
    """Set or extend the backoff window for a provider."""

    if not provider:
        return

    duration = max(0, int(duration or 0))
    if duration == 0:
        return

    provider_key = provider.lower()
    new_until = time.time() + duration
    current_until = _provider_backoff_until.get(provider_key, 0.0)
    if new_until > current_until:
        _provider_backoff_until[provider_key] = new_until


def get_provider_backoff_remaining(provider: str) -> float:
    """Return seconds remaining on the provider backoff window."""

    if not provider:
        return 0.0

    provider_key = provider.lower()
    return max(0.0, _provider_backoff_until.get(provider_key, 0.0) - time.time())


def is_provider_backoff_active(provider: str) -> bool:
    """Check whether calls should be skipped for the provider due to rate limiting."""

    return get_provider_backoff_remaining(provider) > 0

HACKER_NEWS_RSS_URL = "https://hnrss.org/best"
HACKER_NEWS_CACHE_KEY = "context:hacker_news:best"
HACKER_NEWS_MAX_ITEMS = 5

TTL_RATE_GROQ_MINUTE = 120
TTL_RATE_GROQ_DAY = 2 * 24 * 60 * 60

GROQ_RATE_LIMITS = {
    "chat": {"rpm": 1000, "rpd": 500_000, "model": "moonshotai/kimi-k2-instruct-0905"},
    "compound": {"rpm": 200, "rpd": 20_000, "model": GROQ_COMPOUND_DEFAULT_MODEL},
    "vision": {"rpm": 1000, "rpd": 500_000, "model": GROQ_VISION_MODEL},
    "transcribe": {"rpm": 400, "rpd": 200_000, "model": GROQ_TRANSCRIBE_MODEL},
}


def _groq_rate_limit_minute_key(scope: str, bucket: Optional[int] = None) -> str:
    minute_bucket = int(bucket if bucket is not None else time.time() // 60)
    return f"rate_limit:groq:{scope}:minute:{minute_bucket}"


def _groq_rate_limit_day_key(scope: str, day_bucket: Optional[str] = None) -> str:
    utc_day = day_bucket or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"rate_limit:groq:{scope}:day:{utc_day}"


def _decode_rate_counter(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_groq_rate_limit_config(scope: str) -> Dict[str, Any]:
    return dict(GROQ_RATE_LIMITS.get(scope, GROQ_RATE_LIMITS["chat"]))


def _peek_groq_rate_limit(scope: str, redis_client: Optional[redis.Redis] = None) -> bool:
    redis_client = redis_client or _optional_redis_client()
    if redis_client is None:
        return True

    config = _get_groq_rate_limit_config(scope)
    try:
        minute_count = _decode_rate_counter(
            redis_client.get(_groq_rate_limit_minute_key(scope))
        )
        day_count = _decode_rate_counter(redis_client.get(_groq_rate_limit_day_key(scope)))
    except redis.RedisError:
        return True

    return minute_count < int(config["rpm"]) and day_count < int(config["rpd"])


def _consume_groq_rate_limit(scope: str, redis_client: Optional[redis.Redis] = None) -> bool:
    redis_client = redis_client or _optional_redis_client()
    if redis_client is None:
        return True

    config = _get_groq_rate_limit_config(scope)
    minute_key = _groq_rate_limit_minute_key(scope)
    day_key = _groq_rate_limit_day_key(scope)

    try:
        pipe = redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, TTL_RATE_GROQ_MINUTE, nx=True)
        pipe.incr(day_key)
        pipe.expire(day_key, TTL_RATE_GROQ_DAY, nx=True)
        results = pipe.execute()
        minute_count = _decode_rate_counter(results[0] if results else 0)
        day_count = _decode_rate_counter(results[2] if len(results) > 2 else 0)
    except redis.RedisError:
        return True

    return minute_count <= int(config["rpm"]) and day_count <= int(config["rpd"])

def run_agent_cycle() -> Dict[str, Any]:
    """Trigger the autonomous agent, persist its thought and return metadata."""

    recent_thoughts = get_agent_thoughts()
    recent_entry_texts: List[str] = []
    if recent_thoughts:
        for thought in recent_thoughts[:AGENT_RECENT_THOUGHT_WINDOW]:
            candidate = str(thought.get("text", "")).strip()
            if candidate:
                recent_entry_texts.append(candidate)

    last_entry_text = recent_entry_texts[0] if recent_entry_texts else None
    recent_topic_summaries = summarize_recent_agent_topics(
        recent_thoughts[:AGENT_RECENT_THOUGHT_WINDOW]
    )
    hacker_news_items = get_hacker_news_context(limit=3)

    agent_prompt = (
        "Estás operando en modo autónomo. Podés investigar, navegar y usar herramientas. "
        "Registrá en primera persona qué investigaste, qué encontraste y recién después el próximo paso. "
        'Devolvé la nota en dos secciones en mayúsculas: "HALLAZGOS:" con los datos concretos y "PRÓXIMO PASO:" con la acción puntual.'
    )
    if last_entry_text:
        agent_prompt += (
            "\n\nÚLTIMA MEMORIA GUARDADA:\n"
            f"{truncate_text(last_entry_text, 220)}\n"
            "Resolvé ese pendiente ahora mismo y deja asentado el resultado concreto antes de planear otra cosa."
        )
    if recent_topic_summaries:
        topics_lines = "\n".join(f"- {value}" for value in recent_topic_summaries)
        agent_prompt += (
            "\nEstos fueron los últimos temas que trabajaste:\n"
            f"{topics_lines}\n"
            "Solo repetí uno si apareció un dato nuevo y específico; si no, cambiá a otro interés del gordo."
        )
    if hacker_news_items:
        hn_lines = format_hacker_news_info(hacker_news_items, include_discussion=False)
        agent_prompt += (
            "\n\nHACKER NEWS HOY:\n"
            f"{hn_lines}\n"
            "Si alguna nota trae datos frescos que sumen, citá la fuente y metela en los hallazgos."
        )
    agent_prompt += (
        "\nIncluí datos específicos (números, titulares, fuentes) de lo que investigues y evitá repetir entradas previas. "
        "Si necesitás info fresca, llamá a la herramienta web_search con un query puntual y resumí el hallazgo. "
        "Si hace falta leer una nota puntual, llamá a fetch_url con la URL (incluí https://) y anotá lo relevante. "
        "Máximo 500 caracteres, sin saludar a nadie: es un apunte privado."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": agent_prompt,
                }
            ],
        }
    ]

    def generate_response(current_messages: List[Dict[str, Any]]) -> str:
        raw = ask_ai(current_messages)
        sanitized_response = sanitize_tool_artifacts(raw)
        return clean_duplicate_response(sanitized_response).strip()

    cleaned = request_agent_response(
        generate_response, messages, "Autonomous agent execution failed"
    )

    if not agent_sections_are_valid(cleaned):
        original_attempt = cleaned
        corrective_prompt = build_agent_retry_prompt(original_attempt)
        missing_sections = [
            header
            for header in AGENT_REQUIRED_SECTIONS
            if not extract_agent_section_content(original_attempt, header)
        ]
        if missing_sections:
            section_list = ", ".join(missing_sections)
            corrective_prompt += (
                f" La nota anterior no tenía contenido en: {section_list}. "
                "Respetá ambas secciones con información concreta."
            )
        retry_messages = build_agent_retry_messages(
            messages, original_attempt, corrective_prompt
        )
        cleaned = request_agent_response(
            generate_response,
            retry_messages,
            "Autonomous agent execution failed (structure retry)",
        )
        if not agent_sections_are_valid(cleaned):
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK

    matching_recent_text = find_repetitive_recent_thought(cleaned, recent_entry_texts)
    repetition_attempt = 0
    while matching_recent_text and repetition_attempt < AGENT_REPETITION_RETRY_LIMIT:
        corrective_prompt = build_agent_retry_prompt(matching_recent_text)
        if repetition_attempt == AGENT_REPETITION_RETRY_LIMIT - 1:
            corrective_prompt += " " + AGENT_REPETITION_ESCALATION_HINT

        retry_messages = build_agent_retry_messages(
            messages, cleaned, corrective_prompt
        )
        cleaned = request_agent_response(
            generate_response,
            retry_messages,
            "Autonomous agent execution failed (retry)",
        )

        if not agent_sections_are_valid(cleaned):
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK
            matching_recent_text = None
            break

        repetition_attempt += 1
        matching_recent_text = find_repetitive_recent_thought(
            cleaned, recent_entry_texts
        )

    if matching_recent_text:
        fallback = clean_duplicate_response(
            build_agent_fallback_entry(matching_recent_text)
        )
        fallback_entry = ensure_agent_response_text(fallback)

        comparison_texts: Iterable[str]
        if recent_entry_texts:
            filtered_recent_texts: List[str] = []
            skip_match = False
            for candidate_text in recent_entry_texts:
                if not skip_match and candidate_text == matching_recent_text:
                    skip_match = True
                    continue
                filtered_recent_texts.append(candidate_text)
            comparison_texts = filtered_recent_texts
        else:
            comparison_texts = recent_entry_texts

        if is_loop_fallback_text(fallback_entry) and find_repetitive_recent_thought(
            fallback_entry, comparison_texts
        ):
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK
        else:
            cleaned = fallback_entry

    if not agent_sections_are_valid(cleaned):
        cleaned = AGENT_EMPTY_RESPONSE_FALLBACK

    if is_empty_agent_thought_text(cleaned):
        fallback_text = ensure_agent_response_text(cleaned)
        return {"text": fallback_text, "persisted": False}

    entry = save_agent_thought(cleaned)
    if not entry:
        admin_report(
            "Autonomous agent could not persist thought",
            None,
            {"thought_preview": cleaned[:80]},
        )
        raise RuntimeError("Failed to persist autonomous agent thought")

    result: Dict[str, Any] = {
        "text": entry.get("text", cleaned),
        "persisted": True,
    }
    timestamp_value = entry.get("timestamp")
    if isinstance(timestamp_value, (int, float)):
        ts_int = int(timestamp_value)
        result["timestamp"] = ts_int
        result["iso_time"] = datetime.fromtimestamp(ts_int, tz=BA_TZ).isoformat()

    return result


def _get_cached_media(prefix: str, file_id: str) -> Optional[str]:
    """Retrieve a cached media payload stored under the given prefix."""
    cache_key = f"{prefix}:{file_id}"
    try:
        redis_client = config_redis()
        cached_value = redis_client.get(cache_key)
        if cached_value:
            return str(cached_value)
        return None
    except Exception as e:
        print(f"Error getting cached {prefix}: {e}")
        return None


def _cache_media(prefix: str, file_id: str, text: str, ttl: int) -> None:
    """Persist a media payload using the provided prefix and TTL."""
    cache_key = f"{prefix}:{file_id}"
    try:
        redis_client = config_redis()
        redis_client.setex(cache_key, ttl, text)
    except Exception as e:
        print(f"Error caching {prefix}: {e}")


def get_cached_transcription(file_id: str) -> Optional[str]:
    """Get cached audio transcription from Redis"""
    return _get_cached_media("audio_transcription", file_id)


def cache_transcription(file_id: str, text: str, ttl: int = TTL_MEDIA_CACHE) -> None:
    """Cache audio transcription in Redis (default 7 days)"""
    _cache_media("audio_transcription", file_id, text, ttl)


def get_cached_description(file_id: str) -> Optional[str]:
    """Get cached image description from Redis"""
    return _get_cached_media("image_description", file_id)


def cache_description(
    file_id: str, description: str, ttl: int = TTL_MEDIA_CACHE
) -> None:
    """Cache image description in Redis (default 7 days)"""
    _cache_media("image_description", file_id, description, ttl)


# get cached data from previous hour
def get_cache_history(hours_ago, request_hash, redis_client):
    # subtract hours to current date
    timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d-%H")
    # get previous api data from redis cache
    cached_data = redis_client.get(timestamp + request_hash)

    if cached_data is None:
        return None
    else:
        cache_history = json.loads(cached_data)
        if cache_history is not None and "timestamp" not in cache_history:
            cache_history = None
        return cache_history


def cached_requests(
    api_url,
    parameters,
    headers,
    expiration_time,
    hourly_cache=False,
    get_history=False,
    verify_ssl=True,
):
    """Cache any outbound HTTP request by payload and TTL."""
    try:
        arguments_dict = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
        }
        request_hash = hashlib.sha256(
            json.dumps(arguments_dict, sort_keys=True).encode()
        ).hexdigest()

        redis_client = config_redis()
        redis_response = redis_get_json(redis_client, request_hash)
        cache_history = (
            get_cache_history(get_history, request_hash, redis_client)
            if get_history
            else None
        )
        timestamp = int(time.time())

        def make_request():
            last_err: Optional[Exception] = None
            for attempt in range(2):  # try once, then one retry
                try:
                    response = requests.get(
                        api_url,
                        params=parameters,
                        headers=headers,
                        timeout=5,
                        verify=verify_ssl,
                    )
                    response.raise_for_status()
                    redis_value = {
                        "timestamp": timestamp,
                        "data": json.loads(response.text),
                    }
                    redis_set_json(redis_client, request_hash, redis_value)
                    if hourly_cache:
                        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
                        redis_set_json(
                            redis_client, current_hour + request_hash, redis_value
                        )
                    if cache_history is not None:
                        redis_value["history"] = cache_history
                    return redis_value
                except Exception as e:
                    last_err = e
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
            # If both attempts failed
            raise last_err if last_err else Exception("request failed")

        if redis_response is None:
            try:
                return make_request()
            except Exception as e:
                print(f"[CACHE] Error requesting {api_url}: {str(e)}")
                return None
        else:
            cached_data = cast(Dict[str, Any], redis_response)
            cache_age = timestamp - int(cached_data["timestamp"])

            if cache_history is not None:
                cached_data["history"] = cache_history

            if cache_age > expiration_time:
                try:
                    return make_request()
                except Exception as e:
                    print(f"[CACHE] Error updating cache for {api_url}: {str(e)}")
                    return cached_data
            else:
                return cached_data

    except Exception as e:
        error_context = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
            "expiration_time": expiration_time,
        }
        error_msg = f"Error in cached_requests: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return None


bcra_service.configure(
    cached_requests=cached_requests,
    redis_factory=lambda *args, **kwargs: config_redis(*args, **kwargs),
    cache_history=get_cache_history,
)


def gen_random(name: str) -> str:
    rand_res = random.randint(0, 1)
    rand_name = random.randint(0, 2)

    if rand_res:
        msg = "si"
    else:
        msg = "no"

    if rand_name == 1:
        msg = f"{msg} boludo"
    elif rand_name == 2:
        msg = f"{msg} {name}"

    return msg


def select_random(msg_text: str) -> str:
    try:
        values = [v.strip() for v in msg_text.split(",")]
        if len(values) >= 2:
            return random.choice(values)
    except ValueError:
        pass

    try:
        start, end = [int(v.strip()) for v in msg_text.split("-")]
        if start < end:
            return str(random.randint(start, end))
    except ValueError:
        pass

    return "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo"


def get_api_or_cache_prices(convert_to: str, limit: Optional[int] = None):
    # coinmarketcap api config
    api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {"start": "1", "limit": "100", "convert": convert_to}
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": environ.get("COINMARKETCAP_KEY"),
    }

    # Allow callers to limit the page size directly (dedupe across features)
    if isinstance(limit, int) and limit > 0:
        parameters["limit"] = str(limit)

    response = cached_requests(api_url, parameters, headers, TTL_PRICE)

    return response["data"] if response else None


def _fetch_polymarket_event(
    slug: str,
) -> Optional[Tuple[Dict[str, Any], Optional[int]]]:
    """Fetch a single Polymarket event by slug."""

    response = cached_requests(
        POLYMARKET_EVENTS_URL,
        {"slug": slug},
        None,
        TTL_POLYMARKET,
    )

    if not response or "data" not in response:
        return None

    events = response.get("data")

    if not isinstance(events, list) or not events:
        return None

    event = events[0]

    timestamp = response.get("timestamp")
    if not isinstance(timestamp, int):
        timestamp = None

    return event, timestamp


def _format_polymarket_event_section(
    event: Dict[str, Any],
    header: str,
    filter_prefixes: Sequence[str],
) -> Optional[Tuple[List[str], Optional[int]]]:
    """Return formatted lines and timestamp for a Polymarket event."""

    markets = event.get("markets") or []
    odds: List[Tuple[str, float]] = []

    latest_stream_timestamp: Optional[int] = None

    for market in markets:
        raw_outcomes = market.get("outcomes")
        raw_prices = market.get("outcomePrices")
        raw_token_ids = market.get("clobTokenIds")

        if not raw_outcomes or not raw_prices:
            continue

        try:
            outcomes = json.loads(raw_outcomes)
            prices = json.loads(raw_prices)
        except (TypeError, json.JSONDecodeError):
            continue

        try:
            token_ids = json.loads(raw_token_ids) if raw_token_ids else None
        except (TypeError, json.JSONDecodeError):
            token_ids = None

        if not outcomes or not prices:
            continue

        try:
            yes_index = outcomes.index("Yes")
        except ValueError:
            yes_index = 0

        if yes_index >= len(prices):
            continue

        yes_price: Optional[float] = None
        yes_timestamp: Optional[int] = None

        if token_ids and yes_index < len(token_ids):
            stream_result = _fetch_polymarket_live_price(token_ids[yes_index])
            if stream_result:
                yes_price, yes_timestamp = stream_result

        if yes_price is None:
            try:
                yes_price = float(prices[yes_index])
            except (TypeError, ValueError):
                continue

        if yes_timestamp is not None:
            latest_stream_timestamp = (
                yes_timestamp
                if latest_stream_timestamp is None
                else max(latest_stream_timestamp, yes_timestamp)
            )

        probability = max(0.0, min(yes_price, 1.0)) * 100
        title = (
            market.get("groupItemTitle")
            or market.get("question")
            or market.get("slug")
        )

        if not title:
            continue

        odds.append((title, probability))

    if not odds:
        return None

    odds.sort(key=lambda item: item[1], reverse=True)

    filtered_odds: List[Tuple[str, float]] = []
    for title, probability in odds:
        normalized_title = title.strip().upper()
        if any(normalized_title.startswith(prefix.upper()) for prefix in filter_prefixes):
            filtered_odds.append((title, probability))

    odds_to_display = filtered_odds or odds

    lines = [header, ""]

    for title, probability in odds_to_display:
        decimals = 2 if probability < 10 else 1
        lines.append(f"- {title}: {fmt_num(probability, decimals)}%")

    return lines, latest_stream_timestamp


def get_polymarket_argentina_election() -> str:
    """Return Polymarket probabilities for Argentina's 2025 deputies election."""

    sections: List[str] = []

    for slug, header, url in [
        (
            POLYMARKET_ARGENTINA_ELECTION_SLUG,
            "Polymarket - ¿Quién gana más bancas en Diputados 2025?",
            "https://polymarket.com/event/which-party-wins-most-seats-in-argentina-deputies-election",
        ),
        (
            POLYMARKET_ARGENTINA_SEATS_AFTER_SLUG,
            "Polymarket - ¿Quién queda con más bancas después de Diputados 2025?",
            "https://polymarket.com/event/which-party-holds-the-most-seats-after-argentina-deputies-election",
        ),
    ]:
        fetched_event = _fetch_polymarket_event(slug)
        if not fetched_event:
            continue

        event, response_timestamp = fetched_event
        formatted = _format_polymarket_event_section(event, header, ("LLA", "UP"))
        if not formatted:
            continue

        lines, latest_stream_timestamp = formatted

        timestamp = latest_stream_timestamp or response_timestamp
        if isinstance(timestamp, int):
            updated_at_utc = datetime.fromtimestamp(timestamp, timezone.utc)
            updated_at_ba = updated_at_utc.astimezone(BA_TZ)
            lines.extend(
                [
                    "",
                    f"Actualizado: {updated_at_ba.strftime('%Y-%m-%d %H:%M')} UTC-3",
                ]
            )

        lines.append(url)
        sections.append("\n".join(lines))

    if sections:
        return "\n\n".join(sections)

    return "No pude traer las probabilidades desde Polymarket"


def get_btc_price(convert_to: str = "USD") -> Optional[float]:
    """Return BTC price in the requested currency using CoinMarketCap cache.

    Uses the unified prices helper with limit=1 and extracts the first row.
    """
    try:
        data = get_api_or_cache_prices(convert_to)
        if not data or "data" not in data or not data["data"]:
            return None
        first = data["data"][0]
        quote = first.get("quote", {})
        price_info = quote.get(convert_to)
        if not price_info:
            return None
        return float(price_info.get("price"))
    except Exception:
        return None


# get crypto pices from coinmarketcap
def get_prices(msg_text: str) -> Optional[str]:
    prices_number = 0
    # when the user asks for sats we need to ask for btc to the api and convert later
    convert_to = "USD"
    # here we keep the currency that we'll request to the api
    convert_to_parameter = "USD"
    supported_symbols = {
        "ARS",
        "AUD",
        "BRL",
        "BTC",
        "BUSD",
        "CAD",
        "CHF",
        "CLP",
        "CNY",
        "COP",
        "CZK",
        "DAI",
        "DKK",
        "ETH",
        "EUR",
        "GBP",
        "HKD",
        "ILS",
        "INR",
        "ISK",
        "JPY",
        "KRW",
        "MXN",
        "NZD",
        "PEN",
        "SATS",
        "SEK",
        "SGD",
        "TWD",
        "USD",
        "USDC",
        "USDT",
        "UYU",
        "XAU",
        "XMR",
    }

    conversion_prepositions = ("in", "to", "a", "en")
    conversion_token_pattern = "|".join(conversion_prepositions)

    conversion_match = re.match(
        rf"^\s*([0-9]+(?:[\.,][0-9]+)?)\s+([a-zA-Z0-9]+)\s+(?:{conversion_token_pattern})\s+([a-zA-Z0-9]+)\s*$",
        msg_text,
        re.IGNORECASE,
    )

    # symmetric behavior: amount + source_asset + in + target_currency
    if conversion_match:
        amount_text, source_symbol, target_symbol = conversion_match.groups()
        amount = float(amount_text.replace(",", "."))
        source_symbol = source_symbol.upper()
        convert_to = target_symbol.upper()

        if convert_to not in supported_symbols:
            return f"no laburo con {convert_to} gordo"

        convert_to_parameter = "BTC" if convert_to == "SATS" else convert_to
        prices = get_api_or_cache_prices(convert_to_parameter)

        if not prices or "data" not in prices:
            return "no pude traer precios de crypto boludo"

        requested_asset = None
        normalized_source = source_symbol.replace(" ", "")
        for coin in prices["data"]:
            symbol = coin["symbol"].upper().replace(" ", "")
            name = coin["name"].upper().replace(" ", "")
            if symbol == normalized_source or name == normalized_source:
                requested_asset = coin
                break

        if not requested_asset:
            source_parameter = "BTC" if source_symbol == "SATS" else source_symbol
            reverse_prices = get_api_or_cache_prices(source_parameter)
            if not reverse_prices or "data" not in reverse_prices:
                return "no pude traer precios de crypto boludo"

            requested_target_asset = None
            normalized_target = convert_to.replace(" ", "")
            for coin in reverse_prices["data"]:
                symbol = coin["symbol"].upper().replace(" ", "")
                name = coin["name"].upper().replace(" ", "")
                if symbol == normalized_target or name == normalized_target:
                    requested_target_asset = coin
                    break

            if not requested_target_asset:
                return "no laburo con esos ponzis boludo"

            source_amount = amount
            if source_symbol == "SATS":
                source_amount = source_amount / 100000000

            asset_price_in_source = requested_target_asset["quote"][source_parameter]["price"]
            converted_value = source_amount / asset_price_in_source
            return (
                f"{fmt_num(amount, 8)} {source_symbol} = "
                f"{fmt_num(converted_value, 8)} {requested_target_asset['symbol'].upper()}"
            )

        quote_price = requested_asset["quote"][convert_to_parameter]["price"]
        if convert_to == "SATS":
            quote_price = quote_price * 100000000

        converted_value = amount * quote_price
        return (
            f"{fmt_num(amount, 8)} {requested_asset['symbol'].upper()} = "
            f"{fmt_num(converted_value, 8)} {convert_to}"
        )

    conversion_only_match = re.match(
        rf"^\s*(?:{conversion_token_pattern})\s+([a-zA-Z0-9]+)\s*$",
        msg_text,
        re.IGNORECASE,
    )
    if conversion_only_match:
        convert_to = conversion_only_match.group(1).upper()
        msg_text = ""
        if convert_to == "SATS":
            convert_to_parameter = "BTC"
        elif convert_to in supported_symbols:
            convert_to_parameter = convert_to
        else:
            return f"no laburo con {convert_to} gordo"
    else:
        split_parts = re.split(
            rf"\s+(?:{conversion_token_pattern})\s+",
            msg_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(split_parts) == 2:
            msg_text, convert_to = split_parts[0], split_parts[1].upper().strip()
            msg_text = msg_text.strip()
            if convert_to in supported_symbols:
                if convert_to == "SATS":
                    convert_to_parameter = "BTC"
                else:
                    convert_to_parameter = convert_to
            else:
                return f"no laburo con {convert_to} gordo"

    # get prices from api or cache
    prices = get_api_or_cache_prices(convert_to_parameter)

    if msg_text != "":
        numbers = msg_text.upper().replace(" ", "").split(",")

        for number in numbers:
            try:
                number = int(float(number))
                if number > prices_number:
                    prices_number = number
            except ValueError:
                # ignore items which aren't integers
                pass

    if msg_text.upper().isupper():
        new_prices = []
        coins = msg_text.upper().replace(" ", "").split(",")

        if "STABLES" in coins or "STABLECOINS" in coins:
            coins.extend(
                [
                    "BUSD",
                    "DAI",
                    "DOC",
                    "EURT",
                    "FDUSD",
                    "FRAX",
                    "GHO",
                    "GUSD",
                    "LUSD",
                    "MAI",
                    "MIM",
                    "MIMATIC",
                    "NUARS",
                    "PAXG",
                    "PYUSD",
                    "RAI",
                    "SUSD",
                    "TUSD",
                    "USDC",
                    "USDD",
                    "USDM",
                    "USDP",
                    "USDT",
                    "UXD",
                    "XAUT",
                    "XSGD",
                ]
            )

        if not prices or "data" not in prices:
            return "no pude traer precios de crypto boludo"

        for coin in prices["data"]:
            symbol = coin["symbol"].upper().replace(" ", "")
            name = coin["name"].upper().replace(" ", "")

            if symbol in coins or name in coins:
                new_prices.append(coin)
            elif (
                prices_number > 0
                and "data" in prices
                and coin in prices["data"][:prices_number]
            ):
                new_prices.append(coin)

        if not new_prices:
            return "no laburo con esos ponzis boludo"

        prices_number = len(new_prices)
        prices["data"] = new_prices

    # default number of prices
    if prices_number < 1:
        prices_number = 10

    # generate the message to answer the user
    msg = ""
    if not prices or "data" not in prices:
        return "no pude traer precios de crypto boludo"

    for coin in prices["data"][:prices_number]:
        if convert_to == "SATS":
            coin["quote"][convert_to_parameter]["price"] = (
                coin["quote"][convert_to_parameter]["price"] * 100000000
            )

        decimals = f"{coin['quote'][convert_to_parameter]['price']:.12f}".split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))

        ticker = coin["symbol"]
        price = f"{coin['quote'][convert_to_parameter]['price']:.{zeros+4}f}".rstrip(
            "0"
        ).rstrip(".")
        percentage = (
            f"{coin['quote'][convert_to_parameter]['percent_change_24h']:+.2f}".rstrip(
                "0"
            ).rstrip(".")
        )
        line = f"{ticker}: {price} {convert_to} ({percentage}% 24hs)"

        if (
            prices
            and "data" in prices
            and len(prices["data"]) > 0
            and prices["data"][0]["symbol"] == coin["symbol"]
        ):
            msg = line
        else:
            msg = f"{msg}\n{line}"

    return msg


def sort_dollar_rates(
    dollar_rates, tcrm_100: Optional[float] = None, tcrm_history: Optional[float] = None
):
    dollars = dollar_rates["data"]

    sorted_dollar_rates = [
        {
            "name": "Mayorista",
            "price": dollars["mayorista"]["price"],
            "history": dollars["mayorista"]["variation"],
        },
        {
            "name": "Oficial",
            "price": dollars["oficial"]["price"],
            "history": dollars["oficial"]["variation"],
        },
        {
            "name": "Tarjeta",
            "price": dollars["tarjeta"]["price"],
            "history": dollars["tarjeta"]["variation"],
        },
        {
            "name": "MEP",
            "price": dollars["mep"]["al30"]["ci"]["price"],
            "history": dollars["mep"]["al30"]["ci"]["variation"],
        },
        {
            "name": "CCL",
            "price": dollars["ccl"]["al30"]["ci"]["price"],
            "history": dollars["ccl"]["al30"]["ci"]["variation"],
        },
        {
            "name": "Blue",
            "price": dollars["blue"]["ask"],
            "history": dollars["blue"]["variation"],
        },
        {
            "name": "Bitcoin",
            "price": dollars["cripto"]["ccb"]["ask"],
            "history": dollars["cripto"]["ccb"]["variation"],
        },
        {
            "name": "USDC",
            "price": dollars["cripto"]["usdc"]["ask"],
            "history": dollars["cripto"]["usdc"]["variation"],
        },
        {
            "name": "USDT",
            "price": dollars["cripto"]["usdt"]["ask"],
            "history": dollars["cripto"]["usdt"]["variation"],
        },
    ]

    if tcrm_100 is not None:
        sorted_dollar_rates.append(
            {"name": "TCRM 100", "price": tcrm_100, "history": tcrm_history}
        )

    sorted_dollar_rates.sort(key=lambda x: x["price"])

    return sorted_dollar_rates


def format_dollar_rates(
    dollar_rates: List[Dict],
    hours_ago: int,
    band_limits: Optional[Dict[str, Any]] = None,
) -> str:
    rates = list(dollar_rates)

    if band_limits:
        band_entries: List[Dict[str, Any]] = []
        for label, key in (("Banda piso", "lower"), ("Banda techo", "upper")):
            value = band_limits.get(key)
            if not isinstance(value, (int, float)):
                continue
            history = band_limits.get(f"{key}_change_pct")
            band_entries.append(
                {
                    "name": label,
                    "price": float(value),
                    "history": history if isinstance(history, (int, float)) else None,
                }
            )
        if band_entries:
            rates.extend(band_entries)

    # Ensure we keep ascending order when band entries are appended
    rates.sort(key=lambda item: item.get("price", 0))

    msg_lines: List[str] = []
    for dollar in rates:
        price_formatted = fmt_num(dollar["price"], 2)
        line = f"{dollar['name']}: {price_formatted}"
        if dollar["history"] is not None:
            percentage = dollar["history"]
            formatted_percentage = fmt_signed_pct(percentage, 2)
            line += f" ({formatted_percentage}% {hours_ago}hs)"
        msg_lines.append(line)

    return "\n".join(msg_lines)


def get_dollar_rates() -> Optional[str]:
    dollars = _fetch_criptoya_dollar_data()

    tcrm_100, tcrm_history = get_cached_tcrm_100()

    sorted_dollar_rates = sort_dollar_rates(dollars, tcrm_100, tcrm_history)

    band_limits = get_currency_band_limits()

    return format_dollar_rates(sorted_dollar_rates, 24, band_limits)


def get_devo(msg_text: str) -> str:
    try:
        fee = 0
        compra = 0

        if "," in msg_text:
            numbers = msg_text.replace(" ", "").split(",")
            fee = float(numbers[0]) / 100
            if len(numbers) > 1:
                compra = float(numbers[1])
        else:
            fee = float(msg_text) / 100

        if fee != fee or fee > 1 or compra != compra or compra < 0:
            return "mandá bien los datos capo: fee entre 0 y 100, y monto de compra positivo"

        dollars = _fetch_criptoya_dollar_data()

        if not dollars or "data" not in dollars:
            return "no pude traer cotizaciones del dólar boludo"

        usdt_ask = float(dollars["data"]["cripto"]["usdt"]["ask"])
        usdt_bid = float(dollars["data"]["cripto"]["usdt"]["bid"])
        usdt = (usdt_ask + usdt_bid) / 2
        oficial = float(dollars["data"]["oficial"]["price"])
        tarjeta = float(dollars["data"]["tarjeta"]["price"])

        profit = -(fee * usdt + oficial - usdt) / tarjeta

        msg = f"""ganancia: {fmt_num(profit * 100, 2)}%

comisión: {fmt_num(fee * 100, 2)}%
oficial: {fmt_num(oficial, 2)}
usdt: {fmt_num(usdt, 2)}
tarjeta: {fmt_num(tarjeta, 2)}"""

        if compra > 0:
            compra_ars = compra * tarjeta
            compra_usdt = compra_ars / usdt
            ganancia_ars = compra_ars * profit
            ganancia_usdt = ganancia_ars / usdt
            msg = f"""{fmt_num(compra, 2)} USD Tarjeta = {fmt_num(compra_ars, 2)} ARS = {fmt_num(compra_usdt, 2)} USDT
Ganarias {fmt_num(ganancia_ars, 2)} ARS / {fmt_num(ganancia_usdt, 2)} USDT
Total: {fmt_num(compra_ars + ganancia_ars, 2)} ARS / {fmt_num(compra_usdt + ganancia_usdt, 2)} USDT

{msg}"""

        return msg
    except ValueError:
        return "uso: /devo <fee_porcentaje>[, <monto_compra>]"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _format_local_currency(value: float, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    if decimals:
        formatted = formatted.rstrip("0").rstrip(",")
    return formatted


def _format_local_signed(value: float, decimals: int = 2) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_local_currency(abs(value), decimals)}"


def _format_spread_line(
    label: str, sell_price: float, oficial_price: float, details: Sequence[str]
) -> str:
    diff = sell_price - oficial_price
    pct = (diff / oficial_price) * 100 if oficial_price else 0.0
    lines = [
        f"- {label}",
        f"  • Precio venta: {_format_local_currency(sell_price)} ARS/USD",
        f"  • Diferencia vs oficial: {_format_local_signed(diff)} ARS ({fmt_signed_pct(pct, 2)}%)",
    ]
    lines.extend(f"  • {detail}" for detail in details)
    return "\n".join(lines)


def get_rulo() -> str:
    cache_expiration_time = TTL_DOLLAR
    usd_amount = 1000.0
    amount_param = (
        f"{int(usd_amount)}" if isinstance(usd_amount, float) and usd_amount.is_integer() else str(usd_amount)
    )

    dollars = _fetch_criptoya_dollar_data()

    if not dollars or "data" not in dollars:
        return "error consiguiendo cotizaciones del dólar"

    data = dollars["data"]
    oficial_price = _safe_float(data.get("oficial", {}).get("price"))

    if not oficial_price or oficial_price <= 0:
        return "No pude conseguir el oficial para armar el rulo"

    oficial_cost_ars = oficial_price * usd_amount

    base_usd = _format_local_currency(usd_amount, 0)
    base_ars = _format_local_currency(oficial_cost_ars)

    lines: List[str] = [
        f"Rulos desde Oficial (precio oficial: {_format_local_currency(oficial_price)} ARS/USD)",
        f"Inversión base: {base_usd} USD → {base_ars} ARS",
        "",
    ]

    # Oficial -> MEP
    mep_best_price = _safe_float(data["mep"]["al30"]["ci"]["price"])
    mep_label = "MEP (AL30 CI)"

    if mep_best_price:
        mep_final_ars = mep_best_price * usd_amount
        mep_profit_ars = mep_final_ars - oficial_cost_ars
        mep_extra = [
            f"Resultado: {base_usd} USD → {_format_local_currency(mep_final_ars)} ARS",
            f"Ganancia: {_format_local_signed(mep_profit_ars)} ARS",
        ]
        lines.append(
            _format_spread_line(mep_label, mep_best_price, oficial_price, mep_extra)
        )

    # Oficial -> Blue
    blue_data = data.get("blue", {})
    blue_price = _safe_float(blue_data.get("bid")) or _safe_float(blue_data.get("price"))
    if blue_price:
        blue_final_ars = blue_price * usd_amount
        blue_profit_ars = blue_final_ars - oficial_cost_ars
        blue_extra = [
            f"Resultado: {base_usd} USD → {_format_local_currency(blue_final_ars)} ARS",
            f"Ganancia: {_format_local_signed(blue_profit_ars)} ARS",
        ]
        lines.append(
            _format_spread_line("Blue", blue_price, oficial_price, blue_extra)
        )

    # Oficial -> USDT -> ARS
    usd_usdt = cached_requests(
        f"https://criptoya.com/api/USDT/USD/{amount_param}",
        None,
        None,
        cache_expiration_time,
        True,
    )
    usdt_ars = cached_requests(
        f"https://criptoya.com/api/USDT/ARS/{amount_param}",
        None,
        None,
        cache_expiration_time,
        True,
    )

    best_usd_to_usdt: Optional[Tuple[str, float]] = None
    excluded_usd_to_usdt_exchanges = {"banexcoin", "xapo", "x4t"}
    excluded_usdt_to_ars_exchanges = {"okexp2p"}

    if usd_usdt and "data" in usd_usdt:
        for exchange, quote in usd_usdt["data"].items():
            if not isinstance(quote, Mapping):
                continue
            if exchange.lower() in excluded_usd_to_usdt_exchanges:
                continue
            ask = _safe_float(quote.get("totalAsk")) or _safe_float(quote.get("ask"))
            if not ask or ask <= 0:
                continue
            if best_usd_to_usdt is None or ask < best_usd_to_usdt[1]:
                best_usd_to_usdt = (exchange, ask)

    best_usdt_to_ars: Optional[Tuple[str, float]] = None
    if usdt_ars and "data" in usdt_ars:
        for exchange, quote in usdt_ars["data"].items():
            if not isinstance(quote, Mapping):
                continue
            if exchange.lower() in excluded_usdt_to_ars_exchanges:
                continue
            bid = _safe_float(quote.get("totalBid")) or _safe_float(quote.get("bid"))
            if not bid or bid <= 0:
                continue
            if best_usdt_to_ars is None or bid > best_usdt_to_ars[1]:
                best_usdt_to_ars = (exchange, bid)

    if best_usd_to_usdt and best_usdt_to_ars:
        usd_to_usdt_rate = best_usd_to_usdt[1]
        usdt_to_ars_rate = best_usdt_to_ars[1]
        usdt_obtained = usd_amount / usd_to_usdt_rate
        ars_obtained = usdt_obtained * usdt_to_ars_rate
        final_price = ars_obtained / usd_amount
        usdt_profit_ars = ars_obtained - oficial_cost_ars
        path_detail = (
            f"Tramos: USD→USDT {best_usd_to_usdt[0].upper()}, "
            f"USDT→ARS {best_usdt_to_ars[0].upper()}"
        )
        usdt_extra = [
            path_detail,
            (
                f"Resultado: {base_usd} USD → {_format_local_currency(usdt_obtained, 2)} USDT → "
                f"{_format_local_currency(ars_obtained)} ARS"
            ),
            f"Ganancia: {_format_local_signed(usdt_profit_ars)} ARS",
        ]
        lines.append(
            _format_spread_line("USDT", final_price, oficial_price, usdt_extra)
        )

    if len(lines) <= 2:
        return "No encontré ningún rulo potable"

    return "\n".join(lines)


def satoshi() -> str:
    """Calculate the value of 1 satoshi in USD and ARS"""
    try:
        btc_price_usd = get_btc_price("USD")
        btc_price_ars = get_btc_price("ARS")

        if btc_price_usd is None:
            return "no pude traer el precio de btc en usd"
        if btc_price_ars is None:
            return "no pude traer el precio de btc en ars"

        # Calculate satoshi value (1 BTC = 100,000,000 sats)
        sat_value_usd = btc_price_usd / 100_000_000
        sat_value_ars = btc_price_ars / 100_000_000

        # Calculate how many sats per unit
        sats_per_dollar = int(100_000_000 / btc_price_usd)
        sats_per_peso = 100_000_000 / btc_price_ars

        msg = f"""1 satoshi = ${sat_value_usd:.8f} USD
1 satoshi = ${sat_value_ars:.4f} ARS

$1 USD = {sats_per_dollar:,} sats
$1 ARS = {sats_per_peso:.3f} sats"""

        return msg
    except Exception:
        return "no pude conseguir el precio de btc boludo"


def handle_bcra_variables() -> str:
    """Handle BCRA economic variables command"""
    try:
        # Use unified cache/API helper
        variables = get_or_refresh_bcra_variables()

        if not variables:
            return "No pude obtener las variables del BCRA en este momento, probá más tarde"
        return format_bcra_variables(variables)

    except Exception as e:
        print(f"Error handling BCRA variables: {e}")
        return "error al obtener las variables del BCRA"


_DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = {
    "download": "no pude bajar el audio, mandalo de nuevo",
    "transcribe": "no pude sacar nada de ese audio, probá más tarde",
}


def _transcribe_audio_file(
    file_id: str, *, use_cache: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Wrapper for audio transcription with consistent error codes."""

    return transcribe_file_by_id(file_id, use_cache=use_cache)


def _transcription_error_message(
    error_code: Optional[str],
    *,
    download_message: Optional[str] = None,
    transcribe_message: Optional[str] = None,
) -> Optional[str]:
    """Map transcription error codes to user-facing messages."""

    if not error_code:
        return None

    if error_code == "download":
        return download_message or _DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["download"]

    return transcribe_message or _DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["transcribe"]


def _describe_replied_media(
    replied_msg: Mapping[str, Any],
    *,
    media_key: str,
    extract_file_id: Callable[[Any], Optional[str]],
    prompt: str,
    success_prefix: str,
    download_error: str,
    describe_error: str,
) -> Optional[str]:
    media = replied_msg.get(media_key)
    if not media:
        return None

    file_id = extract_file_id(media)
    if not file_id:
        return None

    description, error_code = describe_media_by_id(file_id, prompt)
    if description:
        return f"{success_prefix}{description}"

    if error_code == "download":
        return download_error

    return describe_error


def handle_transcribe_with_message(message: Dict) -> str:
    """Transcribe audio or describe image from replied message"""
    try:
        # Check if this is a reply to another message
        if "reply_to_message" not in message:
            return "respondeme un audio, imagen o sticker y te digo qué carajo hay ahí"

        replied_msg = message["reply_to_message"]

        _, photo_file_id, audio_file_id = extract_message_content(replied_msg)

        if audio_file_id:
            text, error_code = _transcribe_audio_file(audio_file_id, use_cache=True)
            if text:
                return f"🎵 te saqué esto del audio: {text}"
            error_message = _transcription_error_message(error_code)
            if error_message:
                return error_message
            return _DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["transcribe"]

        if photo_file_id:
            def _find_media_message(
                container: Mapping[str, Any], key: str
            ) -> Optional[Mapping[str, Any]]:
                current: Optional[Mapping[str, Any]] = container
                while isinstance(current, Mapping):
                    value = current.get(key)
                    if key == "photo":
                        if (
                            isinstance(value, Sequence)
                            and not isinstance(value, (str, bytes))
                            and value
                        ):
                            return current
                    elif value:
                        return current

                    next_msg = current.get("reply_to_message")
                    if isinstance(next_msg, Mapping):
                        current = next_msg
                    else:
                        break
                return None

            photo_source = _find_media_message(replied_msg, "photo")
            if photo_source:
                photo_response = _describe_replied_media(
                    photo_source,
                    media_key="photo",
                    extract_file_id=lambda media: media[-1]["file_id"]
                    if isinstance(media, Sequence)
                    and not isinstance(media, (str, bytes))
                    and media
                    else None,
                    prompt="Describe what you see in this image in detail.",
                    success_prefix="🖼️ en la imagen veo: ",
                    download_error="no pude bajar la imagen, mandala de nuevo",
                    describe_error="no pude sacar qué mierda tiene la imagen, probá más tarde",
                )
                if photo_response:
                    return photo_response

            sticker_source = _find_media_message(replied_msg, "sticker")
            if sticker_source:
                sticker_response = _describe_replied_media(
                    sticker_source,
                    media_key="sticker",
                    extract_file_id=lambda media: media.get("file_id")
                    if isinstance(media, Mapping)
                    else None,
                    prompt="Describe what you see in this sticker in detail.",
                    success_prefix="🎨 en el sticker veo: ",
                    download_error="no pude bajar el sticker, mandalo de nuevo",
                    describe_error="no pude sacar qué carajo tiene el sticker, probá más tarde",
                )
                if sticker_response:
                    return sticker_response

        return "ese mensaje no tiene audio, imagen ni sticker para laburar"

    except Exception as e:
        print(f"Error in handle_transcribe: {e}")
        return "se trabó el /transcribe, probá más tarde"


def handle_transcribe() -> str:
    """Return the marker command handled by the message processor."""
    return "el /transcribe se usa respondiendo a un audio, imagen o sticker"


def powerlaw() -> str:
    today = datetime.now(timezone.utc)
    since = datetime(day=4, month=1, year=2009).replace(tzinfo=timezone.utc)
    days_since = (today - since).days

    # Giovanni Santostasi Bitcoin Power Law model
    # Formula: 1.0117e-17 * (days since genesis block)^5.82
    value = 1.0117e-17 * (days_since**5.82)

    price = get_btc_price("USD")
    if price is None:
        return "no pude traer el precio de btc para calcular power law"

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% caro boludo"
    else:
        percentage_txt = f"{abs(percentage):.2f}% regalado gordo"

    msg = f"segun power law btc deberia estar en {value:.2f} usd ({percentage_txt})"
    return msg


def rainbow() -> str:
    today = datetime.now(timezone.utc)
    since = datetime(day=9, month=1, year=2009).replace(tzinfo=timezone.utc)
    days_since = (today - since).days
    value = 10 ** (2.66167155005961 * log(days_since) - 17.9183761889864)

    price = get_btc_price("USD")
    if price is None:
        return "no pude traer el precio de btc para calcular rainbow"

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% caro boludo"
    else:
        percentage_txt = f"{abs(percentage):.2f}% regalado gordo"

    msg = f"segun rainbow chart btc deberia estar en {value:.2f} usd ({percentage_txt})"
    return msg


def convert_base(msg_text: str) -> str:
    try:
        input_parts = msg_text.split(",")
        if len(input_parts) != 3:
            return "capo mandate algo como /convertbase 101, 2, 10 y te paso de binario a decimal"
        number_str, base_from_str, base_to_str = map(str.strip, input_parts)
        base_from, base_to = map(int, (base_from_str, base_to_str))

        if not all(c.isalnum() for c in number_str):
            return "el numero tiene que ser alfanumerico boludo"
        if not 2 <= base_from <= 36:
            return f"base origen '{base_from_str}' tiene que ser entre 2 y 36 gordo"
        if not 2 <= base_to <= 36:
            return f"base destino '{base_to_str}' tiene que ser entre 2 y 36 boludo"

        # Convert input to output base
        digits = []
        value = 0
        for digit in number_str:
            if digit.isdigit():
                digit_value = int(digit)
            else:
                digit_value = ord(digit.upper()) - ord("A") + 10
            value = value * base_from + digit_value
        while value > 0:
            digit_value = value % base_to
            if digit_value >= 10:
                digit = chr(digit_value - 10 + ord("A"))
            else:
                digit = str(digit_value)
            digits.append(digit)
            value //= base_to
        result = "".join(reversed(digits))

        return f"ahi tenes boludo, {number_str} en base {base_from} es {result} en base {base_to}"
    except ValueError:
        return "mandate numeros posta gordo, no me hagas perder el tiempo"


def get_timestamp() -> str:
    return f"{int(time.time())}"


JAPANESE_TEXT_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u3400-\u4DBF"
    r"\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF\U0002A700-\U0002B73F"
    r"\U0002B740-\U0002B81F\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF"
    r"\U0002F800-\U0002FA1F\U00030000-\U0003134F]"
)


def romanize_japanese(text: str) -> str:
    """Convert Japanese kana/kanji text to romaji when possible."""
    segments = kakasi().convert(text)
    return "".join(str(segment.get("hepburn") or segment.get("orig") or "") for segment in segments)


def is_japanese_text(text: str) -> bool:
    """Return True when the text includes Japanese scripts or CJK extensions."""
    return bool(JAPANESE_TEXT_RE.search(text))


def convert_to_command(msg_text: str) -> str:
    if not msg_text:
        return "y que queres que convierta boludo? mandate texto"

    # Convert emojis to their textual representation in Spanish with underscore delimiters
    emoji_text = emoji.demojize(msg_text, delimiters=("_", "_"), language="es")
    if is_japanese_text(emoji_text):
        romanized_text = romanize_japanese(emoji_text)
    else:
        romanized_text = emoji_text

    # Convert to uppercase and replace Ñ
    replaced_ni_text = re.sub(r"\bÑ\b", "ENIE", romanized_text.upper()).replace(
        "Ñ", "NI"
    )

    # Normalize the text and remove consecutive spaces
    single_spaced_text = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFD", replaced_ni_text)
        .encode("ascii", "ignore")
        .decode("utf-8"),
    )

    # Replace consecutive dots and specific punctuation marks
    translated_punctuation = re.sub(
        r"\.{3}", "_PUNTOSSUSPENSIVOS_", single_spaced_text
    ).translate(
        str.maketrans(
            {
                " ": "_",
                "\n": "_",
                "?": "_SIGNODEPREGUNTA_",
                "!": "_SIGNODEEXCLAMACION_",
                ".": "_PUNTO_",
            }
        )
    )

    # Remove non-alphanumeric characters and consecutive, trailing and leading underscores
    cleaned_text = re.sub(
        r"^_+|_+$",
        "",
        re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"_+", "_", translated_punctuation)),
    )

    # If there are no remaining characters after processing, return an error
    if not cleaned_text:
        return "no me mandes giladas boludo, tiene que tener letras o numeros"

    command = f"/{cleaned_text}"
    return command


def get_help() -> str:
    return """
esto es lo que sé hacer, boludo:

- /ask, /pregunta, /che, /gordo: te contesto cualquier gilada

- /bcra, /variables: te tiro las variables económicas del bcra

- /comando, /command algo: te convierto eso en comando de telegram

- /convertbase 101, 2, 10: te paso números entre bases

- /buscar algo: te busco en la web

- /agent: te muestro lo último que pensé con el agente autónomo

- /eleccion: odds actuales de Polymarket para Diputados 2025

- /devo 0.5, 100: te calculo el arbitraje entre tarjeta y crypto

- /rulo: te armo los rulos desde el oficial

- /dolar, /dollar, /usd: te tiro la posta del blue y todos los dólares

- /instance: te digo donde estoy corriendo

- /config: tocás la config del gordo y de los links

- /topup: cargás créditos ia con telegram stars por privado

- /balance: te muestro cuántos créditos ia te quedan

- /transfer 100: le pasás 100 créditos tuyos al grupo

- /prices, /precio, /precios, /presio, /presios, /bresio, /bresios, /brecio, /brecios: top 10 cryptos en usd
- /prices in btc: top 10 en btc
- /prices 20: top 20 en usd
- /prices 100 in eur: top 100 en eur
- /prices btc, eth, xmr: bitcoin, ethereum y monero en usd
- /prices dai in sats: dai en satoshis
- /prices stables: stablecoins en usd

- /random pizza, carne, sushi: elijo por vos
- /random 1-10: numero random del 1 al 10

- /powerlaw: te tiro el precio justo de btc según power law y si está caro o barato
- /rainbow: idem pero con el rainbow chart

- /satoshi, /sat, /sats: te digo cuanto vale un satoshi

- /time: timestamp unix actual

- /transcribe: te transcribo audio o describo imagen (responde a un mensaje)
"""


def get_instance_name() -> str:
    instance = environ.get("FRIENDLY_INSTANCE_NAME")
    return f"estoy corriendo en {instance} boludo"


def _telegram_request(
    endpoint: str,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 5,
    token: Optional[str] = None,
    log_errors: bool = True,
    expect_json: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Perform a Telegram Bot API request and return the parsed JSON payload."""

    resolved_token = token or environ.get("TELEGRAM_TOKEN")
    if not resolved_token:
        error_msg = "Telegram token not configured"
        if log_errors:
            print(error_msg)
        return None, error_msg

    url = f"https://api.telegram.org/bot{resolved_token}/{endpoint}"
    method_upper = method.upper()
    try:
        if method_upper == "GET" and json_payload is None:
            response = requests.get(url, params=params, timeout=timeout)
        elif method_upper == "POST" and params is None:
            response = requests.post(url, json=json_payload, timeout=timeout)
        else:
            response = requests.request(
                method_upper,
                url,
                params=params,
                json=json_payload,
                timeout=timeout,
            )
        response.raise_for_status()
        if not expect_json:
            return {}, None
    except requests.RequestException as error:
        if log_errors:
            print(f"Telegram request to {endpoint} failed: {error}")
        return None, str(error)

    try:
        payload = response.json()
    except ValueError as exc:
        if log_errors:
            print(f"Telegram request to {endpoint} returned invalid JSON: {exc}")
        return None, str(exc)

    if not isinstance(payload, dict):
        if log_errors:
            print(
                f"Telegram request to {endpoint} returned unexpected payload type"
            )
        return None, "unexpected response"

    if not payload.get("ok"):
        description = str(payload.get("description") or "telegram request failed")
        if log_errors:
            print(
                f"Telegram request to {endpoint} returned ok=false: {description}"
            )
        return payload, description

    return payload, None


def send_typing(token: str, chat_id: str) -> None:
    _telegram_request(
        "sendChatAction",
        method="GET",
        params={"chat_id": chat_id, "action": "typing"},
        token=token,
        log_errors=False,
        expect_json=False,
    )


def _message_has_domain_link(message: str, domain: str) -> bool:
    """Return True when *message* includes a URL matching *domain*."""

    if not message:
        return False

    normalized_domain = domain.lower().strip(".")
    if not normalized_domain:
        return False

    candidates = re.findall(
        r"(https?://[^\s<>()]+|www\.[^\s<>()]+|(?<!@)\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>()]*)?)",
        message,
        flags=re.IGNORECASE,
    )
    for candidate in candidates:
        cleaned_candidate = candidate.rstrip(".,!?;:)]}'\"")
        normalized_url = _normalize_http_url(cleaned_candidate)
        if not normalized_url:
            continue
        hostname = (urlparse(normalized_url).hostname or "").lower()
        if hostname == normalized_domain or hostname.endswith(f".{normalized_domain}"):
            return True

    return False


def send_msg(
    chat_id: str,
    msg: str,
    msg_id: str = "",
    buttons: Optional[List[str]] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    payload: Dict[str, Any] = {"chat_id": chat_id, "text": msg}
    if _message_has_domain_link(msg, "polymarket.com"):
        payload["disable_web_page_preview"] = True
    if msg_id:
        payload["reply_to_message_id"] = msg_id

    markup = reply_markup
    if markup is None and buttons:
        keyboard = [[{"text": "abrir en la app", "url": url}] for url in buttons]
        markup = {"inline_keyboard": keyboard}

    if markup is not None:
        payload["reply_markup"] = markup

    payload_response, error = _telegram_request(
        "sendMessage", method="POST", json_payload=payload
    )
    if error or not payload_response:
        return None

    result = payload_response.get("result")
    if isinstance(result, dict):
        message_id = result.get("message_id")
        if isinstance(message_id, int):
            return message_id

    return None


def delete_msg(chat_id: str, msg_id: str) -> None:
    """Delete a Telegram message"""
    _telegram_request(
        "deleteMessage",
        method="GET",
        params={"chat_id": chat_id, "message_id": msg_id},
        log_errors=False,
        expect_json=False,
    )


def admin_report(
    message: str,
    error: Optional[Exception] = None,
    extra_context: Optional[Dict] = None,
) -> None:
    """Enhanced admin reporting with optional error details and extra context"""
    admin_chat_id = environ.get("ADMIN_CHAT_ID")
    instance_name = environ.get("FRIENDLY_INSTANCE_NAME")

    # Basic error message
    formatted_message = f"reporte admin desde {instance_name}: {message}"

    # Add extra context if provided
    if extra_context:
        context_details = "\n\ncontexto adicional:"
        for key, value in extra_context.items():
            context_details += f"\n{key}: {value}"
        formatted_message += context_details

    # Add error details if provided
    if error:
        error_details = f"\n\ntipo de error: {type(error).__name__}"
        error_details += f"\nmensaje de error: {str(error)}"

        error_details += f"\n\ntraceback:\n{traceback.format_exc()}"

        formatted_message += error_details

    if admin_chat_id:
        send_msg(admin_chat_id, formatted_message)


def _chat_admin_cache_key(chat_id: str, user_id: Union[str, int]) -> str:
    return f"chat_admin:{chat_id}:{user_id}"


def is_chat_admin(
    chat_id: str,
    user_id: Optional[Union[str, int]],
    *,
    redis_client: Optional[redis.Redis] = None,
) -> bool:
    return _chat_is_chat_admin(
        chat_id,
        user_id,
        redis_client=redis_client,
        optional_redis_client=_optional_redis_client,
        telegram_request=_telegram_request,
        log_event=_log_config_event,
        redis_get_json_fn=redis_get_json,
        redis_setex_json_fn=redis_setex_json,
    )


def _report_unauthorized_config_attempt(
    chat_id: str,
    user: Mapping[str, Any],
    *,
    chat_type: Optional[str],
    action: str,
    callback_data: Optional[str] = None,
) -> None:
    _chat_report_unauthorized_config_attempt(
        chat_id,
        user,
        chat_type=chat_type,
        action=action,
        log_event=_log_config_event,
        callback_data=callback_data,
    )


bcra_service.configure(
    cached_requests=cached_requests,
    redis_factory=lambda *args, **kwargs: config_redis(*args, **kwargs),
    admin_reporter=admin_report,
    cache_history=get_cache_history,
)


configure_app_config(admin_reporter=admin_report)


def get_weather() -> dict:
    """Get current weather for Buenos Aires"""
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        parameters = {
            "latitude": -34.5429,
            "longitude": -58.7119,
            "hourly": "apparent_temperature,precipitation_probability,weather_code,cloud_cover,visibility",
            "timezone": "auto",
            "forecast_days": 2,
        }

        response = cached_requests(
            weather_url, parameters, None, TTL_WEATHER
        )  # Cache por 30 minutos
        if response and "data" in response:
            hourly = response["data"]["hourly"]

            # Get current time in Buenos Aires
            current_time = datetime.now(BA_TZ)

            # Find the current hour index by matching with timestamps
            current_index = None
            for i, timestamp in enumerate(hourly["time"]):
                forecast_time = datetime.fromisoformat(timestamp)
                if (
                    forecast_time.year == current_time.year
                    and forecast_time.month == current_time.month
                    and forecast_time.day == current_time.day
                    and forecast_time.hour == current_time.hour
                ):
                    current_index = i
                    break

            if current_index is None:
                return {}

            # Get current values
            return {
                "apparent_temperature": hourly["apparent_temperature"][current_index],
                "precipitation_probability": hourly["precipitation_probability"][
                    current_index
                ],
                "weather_code": hourly["weather_code"][current_index],
                "cloud_cover": hourly["cloud_cover"][current_index],
                "visibility": hourly["visibility"][current_index],
            }
        return {}
    except Exception as e:
        print(f"Error getting weather: {str(e)}")
        return {}


def normalize_web_search_output(
    tool_output: Any,
) -> Tuple[str, List[Mapping[str, Any]]]:
    """Extract query/results from web_search output into a normalized shape."""
    try:
        data = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
    except Exception:
        return "", []

    if not isinstance(data, Mapping):
        return "", []

    query = data.get("query", "")
    raw_results = data.get("results", [])
    normalized_results: List[Mapping[str, Any]] = []
    if isinstance(raw_results, Sequence):
        for item in raw_results:
            if isinstance(item, Mapping):
                normalized_results.append(cast(Mapping[str, Any], item))
    return query, normalized_results


def resolve_tool_calls(
    system_message: Dict[str, Any],
    messages: List[Dict[str, Any]],
    initial_response: Optional[str],
) -> Optional[str]:
    """Resolve tool calls from an initial model response, returning a final reply."""
    if not initial_response:
        return None

    tool_call = parse_tool_call(initial_response)
    if not tool_call:
        return None

    tool_name, tool_args = tool_call
    if tool_name.lower() == "web_search":
        query = str(tool_args.get("query", "")).strip()
        if not query:
            return "query vacío"
        if not should_use_groq_compound_tools():
            return "la búsqueda web no está disponible en este momento"
        compound_response = get_groq_compound_response(
            build_compound_system_message(),
            [{"role": "user", "content": query}],
        )
        if compound_response:
            return compound_response
        return "no pude completar la búsqueda web ahora"

    try:
        print(
            f"ask_ai: executing tool '{tool_name}' args={json.dumps(tool_args)[:200]}"
        )
        tool_output = execute_tool(tool_name, tool_args)
    except Exception as tool_err:
        tool_output = f"Error al ejecutar herramienta {tool_name}: {tool_err}"
        print(f"ask_ai: tool '{tool_name}' raised error: {tool_err}")

    # Feed tool result back into the conversation and get final answer
    tool_context = {
        "tool": tool_name,
        "args": tool_args,
        "result": tool_output,
    }
    messages = messages + [
        {
            "role": "assistant",
            "content": sanitize_tool_artifacts(initial_response),
        },
        {
            "role": "user",
            "content": f"RESULTADO DE HERRAMIENTA:\n{json.dumps(tool_context)[:4000]}",
        },
    ]

    last_tool_name = tool_name
    last_tool_output = tool_output

    attempts = 0
    final = None
    while attempts < 3:
        final = complete_with_providers(system_message, messages)
        if not final:
            print(
                "ask_ai: second pass returned None; falling back to tool-output formatting"
            )
            break
        print(
            f"ask_ai: final len={len(final)} preview='{final[:160].replace('\n',' ')}'"
        )
        next_tool_call = parse_tool_call(final)
        if not next_tool_call:
            return final

        tool_name, tool_args = next_tool_call
        try:
            print(
                f"ask_ai: executing tool '{tool_name}' args={json.dumps(tool_args)[:200]}"
            )
            tool_output = execute_tool(tool_name, tool_args)
        except Exception as tool_err:
            tool_output = f"Error al ejecutar herramienta {tool_name}: {tool_err}"
            print(f"ask_ai: tool '{tool_name}' raised error: {tool_err}")

        tool_context = {
            "tool": tool_name,
            "args": tool_args,
            "result": tool_output,
        }
        messages = messages + [
            {
                "role": "assistant",
                "content": sanitize_tool_artifacts(final),
            },
            {
                "role": "user",
                "content": f"RESULTADO DE HERRAMIENTA:\n{json.dumps(tool_context)[:4000]}",
            },
        ]
        last_tool_name = tool_name
        last_tool_output = tool_output
        attempts += 1

    # If the second pass (or subsequent passes) failed, synthesize a response from the last tool output
    try:
        if last_tool_name == "web_search":
            query, normalized_results = normalize_web_search_output(last_tool_output)
            return summarize_search_results(query, normalized_results)
        if last_tool_name == "fetch_url":
            data: Any = last_tool_output
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    return str(last_tool_output)[:1500]
            if not isinstance(data, dict):
                return str(last_tool_output)[:1500]
            url = str(data.get("url") or "").strip()
            error_msg = str(data.get("error") or "").strip()
            if error_msg:
                if url:
                    return f"no pude leer {url}: {error_msg}"
                return f"no pude leer la URL: {error_msg}"
            title = str(data.get("title") or "").strip()
            content = str(data.get("content") or "").strip()
            truncated_flag = bool(data.get("truncated"))
            lines: List[str] = []
            if title:
                lines.append(f"📄 {title}")
            if url:
                lines.append(url)
            if content:
                lines.append("")
                lines.append(content)
            if truncated_flag and content:
                lines.append("")
                lines.append("(texto recortado)")
            formatted = "\n".join(line for line in lines if line).strip()
            if formatted:
                return formatted
            if url:
                return f"leí {url} pero no encontré texto para mostrar"
            return "no había texto legible en la página"
        # Generic fallback for other tools
        return f"Resultado de {last_tool_name}:\n{str(last_tool_output)[:1500]}"
    except Exception:
        # If even formatting fails, return a safe generic message
        return "tuve un problema usando la herramienta, probá de nuevo más tarde"


def ask_ai(
    messages: List[Dict[str, Any]],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
) -> str:
    try:
        messages = list(messages or [])

        # Prepend autonomous agent memory so the model can reference past thoughts
        agent_memory = get_agent_memory_context()
        if agent_memory:
            messages = [agent_memory] + messages

        # Build context with market and weather data
        context_data = {
            "market": get_market_context(),
            "weather": get_weather_context(),
            "time": get_time_context(),
            "hacker_news": get_hacker_news_context(),
        }

        # Build system message with personality, context and tool instructions
        system_message = build_system_message(context_data, include_tools=True)
        compound_system_message = (
            build_compound_system_message() if should_use_groq_compound_tools() else None
        )

        # If we have an image, first describe it with the vision model then continue normal flow
        if image_data:
            print("Processing image with Groq vision model...")

            # Always use a description prompt for the vision model, not the user's question
            user_text = "Describe what you see in this image in detail."

            # Describe the image using Groq
            image_description = describe_image_groq(image_data, user_text, image_file_id)

            if image_description:
                # Add image description to the conversation context
                image_context = f"[Imagen: {image_description}]"

                # Modify the last message to include the image description
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = (
                            last_message["content"] + f"\n\n{image_context}"
                        )

                print(f"Image described, continuing with normal AI flow...")
            else:
                print("Failed to describe image, continuing without description...")

        # Continue with normal AI flow (for both image and text).
        latest_user_text = ""
        latest_user_message = ""
        latest_user_index: Optional[int] = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    latest_user_text = content
                    latest_user_message = extract_user_message_from_context(content)
                    latest_user_index = idx
                break

        if should_search_previous_query(latest_user_message):
            previous_query = ""
            if latest_user_index is not None:
                previous_query = _find_previous_user_text(messages, latest_user_index)
            previous_message = extract_user_message_from_context(previous_query)
            if not previous_message:
                return "Decime qué querés buscar o usá /buscar <tema>."
            print("ask_ai: previous query web_search triggered")
            return _run_forced_web_search(
                query=previous_message,
                messages=messages,
                system_message=system_message,
                compound_system_message=compound_system_message,
            )

        if should_force_web_search(latest_user_message):
            print("ask_ai: forced web_search heuristic triggered")
            return _run_forced_web_search(
                query=latest_user_message,
                messages=messages,
                system_message=system_message,
                compound_system_message=compound_system_message,
            )

        # First pass: get an initial response that might include a tool call.
        initial = complete_with_providers(system_message, messages)
        if initial:
            print(
                f"ask_ai: initial len={len(initial)} preview='{initial[:160].replace('\n',' ')}'"
            )

        tool_final = resolve_tool_calls(system_message, messages, initial)
        if tool_final:
            return tool_final

        # If no tool call or second pass failed, return the best we had
        if initial:
            return initial or get_fallback_response(messages)

        # Final fallback to random response if all AI providers fail
        return _mark_ai_fallback_response(get_fallback_response(messages))

    except Exception as e:
        error_context = {
            "messages_count": len(messages),
            "messages_preview": [msg.get("content", "")[:100] for msg in messages],
        }
        admin_report("Error in ask_ai", e, error_context)
        return _mark_ai_fallback_response(get_fallback_response(messages))


def complete_with_providers(
    system_message: Dict[str, Any], messages: List[Dict[str, Any]]
) -> Optional[str]:
    """Try Groq and return the first response."""

    response = get_groq_ai_response(system_message, messages)
    if response:
        print("complete_with_providers: got response from Groq")
        return response
    return None


class _ToolLine(NamedTuple):
    raw: str
    stripped: str
    normalized: str
    in_fence: bool
    is_fence: bool


def _prepare_tool_lines(text: str) -> List[_ToolLine]:
    in_fence = False
    prepared: List[_ToolLine] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        is_fence = stripped.startswith("```")
        normalized = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", stripped)
        normalized = normalized.strip().strip("`")
        if is_fence:
            normalized = ""
        prepared.append(
            _ToolLine(
                raw=raw,
                stripped=stripped,
                normalized=normalized,
                in_fence=in_fence,
                is_fence=is_fence,
            )
        )
        if is_fence:
            in_fence = not in_fence
    return prepared


def parse_tool_call(text: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Detect a tool call line like: [TOOL] web_search {"query": "..."}"""
    if not text:
        return None
    try:
        prepared = _prepare_tool_lines(text)
        i = 0
        while i < len(prepared):
            line = prepared[i]

            if line.is_fence or line.in_fence:
                i += 1
                continue

            marker_index = line.normalized.find("[TOOL]")
            if marker_index == -1:
                i += 1
                continue

            normalized = line.normalized[marker_index + len("[TOOL]") :].strip()
            if normalized.startswith(":"):
                normalized = normalized[1:].strip()

            name_source_index = i
            if not normalized:
                j = i + 1
                while j < len(prepared):
                    candidate = prepared[j]
                    if candidate.is_fence or candidate.in_fence:
                        j += 1
                        continue
                    if candidate.normalized:
                        normalized = candidate.normalized
                        name_source_index = j
                        break
                    j += 1
                if not normalized:
                    i += 1
                    continue

            parts = normalized.split(" ", 1)
            if not parts:
                i += 1
                continue

            name = parts[0].strip().strip(":")
            if not name:
                i += 1
                continue

            remainder = parts[1].strip() if len(parts) > 1 else ""
            if remainder.startswith(":"):
                remainder = remainder[1:].strip()

            json_candidate = remainder
            j = name_source_index + 1

            while "{" not in json_candidate and j < len(prepared):
                addition_line = prepared[j]
                if addition_line.is_fence or addition_line.in_fence:
                    j += 1
                    continue
                if addition_line.normalized:
                    json_candidate = (json_candidate + " " + addition_line.normalized).strip()
                j += 1

            if "{" not in json_candidate:
                i += 1
                continue

            open_count = json_candidate.count("{") - json_candidate.count("}")
            while open_count > 0 and j < len(prepared):
                addition_line = prepared[j]
                if addition_line.is_fence or addition_line.in_fence:
                    j += 1
                    continue
                if addition_line.normalized:
                    json_candidate = (json_candidate + " " + addition_line.normalized).strip()
                    open_count += addition_line.normalized.count("{") - addition_line.normalized.count("}")
                j += 1

            closing_index = json_candidate.rfind("}")
            if closing_index == -1:
                i += 1
                continue

            json_text = json_candidate[: closing_index + 1].strip().rstrip(",")

            args: Any = None
            try:
                args = json.loads(json_text)
            except Exception:
                try:
                    args = ast.literal_eval(json_text)
                except Exception:
                    print(
                        f"parse_tool_call: failed to parse args after [TOOL] {name}: '{json_text[:200]}'"
                    )
                    args = None

            if isinstance(args, dict) and name:
                print(
                    f"parse_tool_call: detected tool '{name}' with args keys={list(args.keys())}"
                )
                return name, args

            i += 1
    except Exception:
        return None
    return None


def _normalize_http_url(raw_url: str) -> Optional[str]:
    """Normalize raw URL strings to HTTP/HTTPS form without fragments."""

    if not raw_url:
        return None

    candidate = str(raw_url).strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    if not parsed.scheme:
        parsed = urlparse(f"https://{candidate}")

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    netloc = parsed.netloc
    if any(char.isspace() for char in netloc):
        return None

    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned)


class _VisibleTextExtractor(HTMLParser):
    """Extract visible text and title from HTML documents."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: List[str] = []
        self._skip_depth = 0
        self._title_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._in_title = True
            return
        if tag_lower in {
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "li",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag_lower == "title":
            self._in_title = False
            return
        if tag_lower in {
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        text = unescape(data)
        if self._in_title:
            self._title_parts.append(text)

        cleaned = text.strip()
        if not cleaned:
            return

        if self._buffer and not self._buffer[-1].endswith((" ", "\n")):
            self._buffer.append(" ")

        self._buffer.append(cleaned)

    def get_text(self) -> str:
        raw = "".join(self._buffer)
        collapsed_spaces = re.sub(r"[ \t]+", " ", raw)
        collapsed_lines = re.sub(r"\n\s*", "\n", collapsed_spaces)
        collapsed_lines = re.sub(r"\n{3,}", "\n\n", collapsed_lines)
        return collapsed_lines.strip()

    def get_title(self) -> Optional[str]:
        title = "".join(self._title_parts).strip()
        title = re.sub(r"\s+", " ", title)
        return title or None


def _extract_text_from_html(html_text: str) -> Tuple[Optional[str], str]:
    parser = _VisibleTextExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass
    return parser.get_title(), parser.get_text()


def fetch_url_content(
    raw_url: str, max_chars: Optional[Union[int, str]] = None
) -> Dict[str, Any]:
    """Fetch arbitrary HTTP/HTTPS URLs and return sanitized textual content."""

    normalized = _normalize_http_url(raw_url)
    if not normalized:
        return {"error": "url inválida"}

    try:
        requested_max = (
            int(max_chars) if max_chars is not None else WEB_FETCH_DEFAULT_CHARS
        )
    except (TypeError, ValueError):
        requested_max = WEB_FETCH_DEFAULT_CHARS

    requested_max = max(WEB_FETCH_MIN_CHARS, min(requested_max, WEB_FETCH_MAX_CHARS))

    cache_payload = {"url": normalized, "max": requested_max}
    cache_key = _hash_cache_key("fetch_url", cache_payload)
    redis_client = _optional_redis_client()
    if redis_client is not None:
        try:
            cached = redis_get_json(redis_client, cache_key)
            if isinstance(cached, dict):
                return cached
        except Exception:
            redis_client = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RespondedorBot/1.0;"
            " +https://github.com/gusgusf/RespondedorBot)"
        ),
        "Accept": "text/html,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }

    response = None
    try:
        response = requests.get(
            normalized,
            headers=headers,
            timeout=10,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
    except SSLError:
        return {"error": "no pude establecer conexión segura", "url": normalized}
    except RequestException as req_error:
        error_name = req_error.__class__.__name__
        return {
            "error": f"error de red ({error_name})",
            "url": normalized,
        }

    final_url = normalized
    content_type = ""
    encoding: Optional[str] = None
    apparent_encoding: Optional[str] = None
    status_code = None
    try:
        if isinstance(getattr(response, "url", None), str):
            maybe_url = _normalize_http_url(response.url)
            if maybe_url:
                final_url = maybe_url
        content_type = str(response.headers.get("Content-Type", "")).lower()
        encoding = response.encoding
        apparent_encoding = getattr(response, "apparent_encoding", None)
        status_code = getattr(response, "status_code", None)

        max_bytes = min(WEB_FETCH_MAX_BYTES, max(requested_max * 6, 20000))
        total = 0
        chunks: List[bytes] = []
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
        content_bytes = b"".join(chunks)
    finally:
        try:
            if response is not None:
                response.close()
        except Exception:
            pass

    if not content_bytes:
        result: Dict[str, Any] = {
            "url": final_url,
            "status": status_code,
            "content_type": content_type or "",
            "title": None,
            "content": "",
            "truncated": False,
            "char_count": 0,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "max_chars": requested_max,
        }
        if redis_client:
            try:
                redis_setex_json(redis_client, cache_key, TTL_WEB_FETCH, result)
            except Exception:
                pass
        return result

    if not encoding:
        encoding = apparent_encoding or "utf-8"

    try:
        text_body = content_bytes.decode(encoding or "utf-8", errors="replace")
    except Exception:
        text_body = content_bytes.decode("utf-8", errors="replace")

    textual_tokens = (
        "text",
        "json",
        "xml",
        "javascript",
        "markdown",
        "yaml",
        "csv",
        "x-www-form-urlencoded",
    )
    is_textual = not content_type or any(
        token in content_type for token in textual_tokens
    )
    sample_lower = content_bytes[:400].lower()
    if not is_textual:
        if b"<html" in sample_lower or b"<!doctype" in sample_lower:
            is_textual = True

    if not is_textual:
        return {
            "error": (
                f"el contenido ({content_type or 'desconocido'}) no es texto legible"
            ),
            "url": final_url,
            "status": status_code,
        }

    html_detected = "html" in content_type or "<html" in text_body[:400].lower()
    title: Optional[str] = None
    cleaned_text = text_body

    if html_detected:
        title, cleaned_text = _extract_text_from_html(text_body)
    elif "json" in content_type:
        try:
            parsed_json = json.loads(text_body)
            cleaned_text = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except Exception:
            cleaned_text = text_body
    else:
        cleaned_text = text_body

    cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    truncated_text = truncate_text(cleaned_text, requested_max)
    truncated_flag = len(cleaned_text) > requested_max

    result = {
        "url": final_url,
        "status": status_code,
        "content_type": content_type or "",
        "title": title,
        "content": truncated_text,
        "truncated": truncated_flag,
        "char_count": len(cleaned_text),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "max_chars": requested_max,
    }

    if redis_client:
        try:
            redis_setex_json(redis_client, cache_key, TTL_WEB_FETCH, result)
        except Exception:
            pass

    return result


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a named tool and return a plain-text result string."""
    name = name.lower()
    if name == "fetch_url":
        raw_url = args.get("url") or args.get("link") or args.get("href") or ""
        url = str(raw_url).strip()
        max_chars_arg = args.get("max_chars") or args.get("chars")
        result = fetch_url_content(url, max_chars=max_chars_arg)
        try:
            status = result.get("status") if isinstance(result, dict) else None
            err = result.get("error") if isinstance(result, dict) else None
            log_url = url or (result.get("url") if isinstance(result, dict) else "")
            print(
                "execute_tool:fetch_url: url='%s' status=%s error=%s"
                % (str(log_url)[:200], status, err)
            )
        except Exception:
            pass
        return json.dumps(result)
    return f"herramienta desconocida: {name}"


def web_search(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Simple web search using DDGS library."""
    query = query.strip()
    limit = max(1, min(int(limit), 15))

    cache_payload = {"q": query, "limit": limit}
    cache_key = _hash_cache_key("web_search", cache_payload)
    redis_client = _optional_redis_client()
    if redis_client is not None:
        try:
            cached = redis_get_json(redis_client, cache_key)
            if isinstance(cached, list):
                return cached
        except Exception:
            redis_client = None

    try:
        from ddgs import DDGS

        # Use DDGS for robust DuckDuckGo search
        ddgs = DDGS(timeout=8)
        # Light retry on transient errors
        raw_results: List[Dict[str, Any]] = []
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                raw_results = ddgs.text(
                    query=query, region="ar-es", safesearch="off", max_results=limit
                )
                break
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raw_results = []

        # Convert to our expected format
        results = []
        for result in raw_results:
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")[:500],
                }
            )

        # Store in cache for 5 minutes
        if redis_client is not None:
            try:
                redis_setex_json(redis_client, cache_key, TTL_WEB_SEARCH, results)
            except Exception:
                pass

        return results
    except Exception:
        return []


def format_search_results(
    query: str,
    results: Optional[Sequence[Mapping[str, Any]]],
    limit: int = 5,
) -> str:
    """Format DuckDuckGo search results with consistent truncation and fallbacks."""

    normalized_query = (query or "").strip()
    if not results:
        return "no encontré resultados ahora con duckduckgo"

    try:
        normalized_limit = int(limit)
    except (TypeError, ValueError):
        normalized_limit = 5
    if normalized_limit < 1:
        normalized_limit = 1

    limited_results = list(results)[:normalized_limit]
    if not limited_results:
        return "no encontré resultados ahora con duckduckgo"

    lines = [f"🔎 Resultados para: {normalized_query}"]
    for index, result in enumerate(limited_results, 1):
        title = str(result.get("title") or result.get("url") or "(sin título)")
        url = str(result.get("url") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"{index}. {title}\n{url}\n{snippet[:300]}")
        else:
            lines.append(f"{index}. {title}\n{url}")

    return "\n\n".join(lines)


def summarize_search_results(
    query: str,
    results: Optional[Sequence[Mapping[str, Any]]],
) -> str:
    """Summarize the top DuckDuckGo search result into 1-3 short sentences."""
    if not results:
        return "no encontré resultados ahora con duckduckgo"

    top_result = next((item for item in results if isinstance(item, Mapping)), None)
    if not top_result:
        return "no encontré resultados ahora con duckduckgo"

    title = str(top_result.get("title") or top_result.get("url") or "").strip()
    snippet = str(top_result.get("snippet") or "").strip()

    if not title and not snippet:
        return "no encontré resultados ahora con duckduckgo"

    parts: List[str] = []
    if snippet:
        trimmed_snippet = snippet[:280].rstrip()
        if trimmed_snippet and not trimmed_snippet.endswith((".", "!", "?")):
            trimmed_snippet += "."
        if trimmed_snippet:
            parts.append(trimmed_snippet)

    if not parts and title:
        parts.append(f"{title}.")

    return " ".join(parts).strip()


def sanitize_tool_artifacts(text: Optional[str]) -> str:
    """Remove any visible [TOOL] lines or code blocks that contain them from model output."""
    if not text:
        return ""
    prepared = _prepare_tool_lines(text)
    out_lines: List[str] = []
    block_lines: List[str] = []
    block_has_tool = False
    inside_block = False

    for line in prepared:
        if line.is_fence:
            if not line.in_fence:
                inside_block = True
                block_lines = [line.raw]
                block_has_tool = False
            else:
                block_lines.append(line.raw)
                if not block_has_tool:
                    out_lines.extend(block_lines)
                block_lines = []
                block_has_tool = False
                inside_block = False
            continue

        if inside_block:
            block_lines.append(line.raw)
            if "[TOOL]" in line.raw:
                block_has_tool = True
            continue

        if "[TOOL]" not in line.raw:
            out_lines.append(line.raw)

    if inside_block and block_lines and not block_has_tool:
        out_lines.extend(block_lines)

    return "\n".join(out_lines).strip()


def search_command(msg_text: Optional[str]) -> str:
    """/buscar command: perform a web search and return concise results"""
    q = (msg_text or "").strip()
    if not q:
        return "decime qué querés buscar capo"
    results = web_search(q, limit=10)
    return format_search_results(q, results, limit=10)


def get_hacker_news_context(limit: int = HACKER_NEWS_MAX_ITEMS) -> List[Dict[str, Any]]:
    """Return top Hacker News stories from the best RSS feed (cached)."""

    try:
        limit = int(limit)
    except Exception:
        limit = HACKER_NEWS_MAX_ITEMS

    if limit < 1:
        limit = 1
    if limit > HACKER_NEWS_MAX_ITEMS:
        limit = HACKER_NEWS_MAX_ITEMS

    redis_client = _optional_redis_client()

    cached_items: Optional[List[Dict[str, Any]]] = None
    if redis_client:
        cached = redis_get_json(redis_client, HACKER_NEWS_CACHE_KEY)
        if isinstance(cached, list):
            cached_items = cached
            if cached_items:
                return cached_items[:limit]

    try:
        response = request_with_ssl_fallback(HACKER_NEWS_RSS_URL, timeout=5)
        response.raise_for_status()
    except RequestException as request_error:
        print(f"Error fetching Hacker News RSS: {request_error}")
        return (cached_items or [])[:limit]

    response_text = response.text

    try:
        root = ET.fromstring(response_text)
        channel = root.find("channel")
        if channel is None:
            return (cached_items or [])[:limit]

        items: List[Dict[str, Any]] = []
        for item_el in channel.findall("item"):
            title_raw = item_el.findtext("title", "")
            title = unescape(str(title_raw or "")).strip()
            if not title:
                continue

            link = str(item_el.findtext("link", "") or "").strip()
            description = item_el.findtext("description", "") or ""

            points_val: Optional[int] = None
            points_match = re.search(r"Points:\s*(\d+)", description)
            if points_match:
                try:
                    points_val = int(points_match.group(1))
                except Exception:
                    points_val = None

            comments_val: Optional[int] = None
            comments_match = re.search(r"# Comments:\s*(\d+)", description)
            if comments_match:
                try:
                    comments_val = int(comments_match.group(1))
                except Exception:
                    comments_val = None

            comments_url_match = re.search(
                r"Comments URL: <a href=\"([^\"]+)\"", description
            )
            comments_url = (
                comments_url_match.group(1).strip() if comments_url_match else ""
            )

            items.append(
                {
                    "title": title,
                    "url": link,
                    "points": points_val,
                    "comments": comments_val,
                    "comments_url": comments_url,
                }
            )

            if len(items) >= HACKER_NEWS_MAX_ITEMS:
                break

        if redis_client and items:
            try:
                redis_setex_json(
                    redis_client, HACKER_NEWS_CACHE_KEY, TTL_HACKER_NEWS, items
                )
            except Exception:
                pass

        return items[:limit] if items else (cached_items or [])[:limit]
    except Exception as parse_error:
        print(f"Error parsing Hacker News RSS: {parse_error}")
        return (cached_items or [])[:limit]


def get_market_context() -> Dict:
    """Get crypto, dollar and BCRA market data"""
    market_data = {}

    try:
        # Get crypto prices (reuse unified helper and 5-minute cache)
        api_data = get_api_or_cache_prices("USD", limit=5)
        if api_data and "data" in api_data:
            market_data["crypto"] = clean_crypto_data(api_data["data"])
    except Exception as e:
        print(f"Error fetching crypto data: {e}")

    try:
        # Get dollar rates (reuse 5-minute cache)
        dollar_response = _fetch_criptoya_dollar_data(hourly_cache=False)
        if dollar_response and "data" in dollar_response:
            market_data["dollar"] = dollar_response["data"]
    except Exception as e:
        print(f"Error fetching dollar data: {e}")

    try:
        # Get BCRA variables
        bcra_variables = get_or_refresh_bcra_variables()
        if bcra_variables:
            market_data["bcra"] = bcra_variables
    except Exception as e:
        print(f"Error fetching BCRA data: {e}")

    return market_data


def get_weather_context() -> Optional[Dict]:
    """Get weather data with descriptions"""
    try:
        weather = get_weather()
        if weather:
            weather["description"] = get_weather_description(weather["weather_code"])
        return weather
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def get_time_context() -> Dict:
    """Get current time in Buenos Aires"""
    current_time = datetime.now(BA_TZ)
    return {"datetime": current_time, "formatted": current_time.strftime("%A %d/%m/%Y")}


def _invoke_provider(
    provider_name: str,
    *,
    attempt: Callable[[], Optional[str]],
    rate_limit_backoff: Optional[int] = None,
    label: Optional[str] = None,
    rate_limit_scope: Optional[str] = None,
) -> Optional[str]:
    """Execute a provider call with shared backoff and error handling."""

    display_name = label or provider_name.capitalize()
    if is_provider_backoff_active(provider_name):
        remaining = int(get_provider_backoff_remaining(provider_name))
        print(
            f"{display_name} backoff active ({remaining}s remaining), skipping API call"
        )
        return None

    if rate_limit_scope and not _consume_groq_rate_limit(rate_limit_scope):
        print(f"{display_name} local rate limit reached for scope={rate_limit_scope}, skipping API call")
        return None

    try:
        return attempt()
    except Exception as error:
        print(f"{display_name} error: {error}")
        if _is_rate_limit_error(error):
            _set_provider_backoff(provider_name, rate_limit_backoff)
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                f"{display_name} rate limit detected; backing off for {remaining}s"
            )
        return None


def _is_rate_limit_error(error: Exception) -> bool:
    """Detect whether the provided exception represents a rate limit."""

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    status = getattr(error, "status", None)
    if status == 429:
        return True

    message = str(getattr(error, "message", "") or error)
    lowered = message.lower()
    if "rate limit" in lowered:
        return True

    return "429" in lowered


def get_groq_ai_response(
    system_msg: Dict[str, Any], messages: List[Dict[str, Any]]
) -> Optional[str]:
    """First option using Groq AI"""
    provider_name = "groq"
    groq_api_key = environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Groq API key not configured")
        return None

    def _attempt() -> Optional[str]:
        print("Trying Groq AI as first option...")
        _increment_ai_provider_request_count()
        groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        response = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=cast(Any, [system_msg] + messages),
            max_tokens=1024,
        )

        if response and hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "stop":
                print("Groq AI response successful")
                return response.choices[0].message.content
        return None

    return _invoke_provider(
        provider_name,
        attempt=_attempt,
        rate_limit_backoff=GROQ_RATE_LIMIT_BACKOFF_SECONDS,
        label="Groq AI",
        rate_limit_scope="chat",
    )


def get_groq_compound_response(
    system_msg: Optional[Dict[str, Any]], messages: List[Dict[str, Any]]
) -> Optional[str]:
    """Use Groq Compound built-in tools for a single response."""

    provider_name = "groq"
    groq_api_key = environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Groq API key not configured")
        return None

    model = GROQ_COMPOUND_DEFAULT_MODEL
    enabled_tools = get_groq_compound_enabled_tools()

    def _attempt() -> Optional[str]:
        print("Trying Groq Compound tools...")
        _increment_ai_provider_request_count()
        groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            default_headers={"Groq-Model-Version": "latest"},
        )

        payload_messages = messages
        if system_msg:
            payload_messages = [system_msg] + messages

        response = groq_client.chat.completions.create(
            model=model,
            messages=cast(Any, payload_messages),
            max_tokens=1024,
            extra_body={
                "compound_custom": {
                    "tools": {"enabled_tools": enabled_tools},
                }
            },
        )

        if response and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            if content:
                return content
        return None

    return _invoke_provider(
        provider_name,
        attempt=_attempt,
        rate_limit_backoff=GROQ_RATE_LIMIT_BACKOFF_SECONDS,
        label="Groq Compound",
        rate_limit_scope="compound",
    )


def get_fallback_response(messages: List[Dict]) -> str:
    """Generate fallback random response"""
    first_name = ""
    if messages and len(messages) > 0:
        last_message = messages[-1]["content"]
        if "Usuario: " in last_message:
            first_name = last_message.split("Usuario: ")[1].split(" ")[0]
    return gen_random(first_name)


def _mark_ai_fallback_response(text: str) -> str:
    """Attach a private marker so billing logic can detect fallback responses."""
    return f"{AI_FALLBACK_MARKER}{text}"


def _strip_ai_fallback_marker(text: str) -> Tuple[str, bool]:
    """Remove internal fallback marker and return (clean_text, was_marked)."""
    if text.startswith(AI_FALLBACK_MARKER):
        return text[len(AI_FALLBACK_MARKER) :].lstrip(), True
    return text, False


def clean_crypto_data(cryptos: List[Dict]) -> List[Dict]:
    """Clean and format crypto data"""
    cleaned = []
    for crypto in cryptos:
        cleaned.append(
            {
                "name": crypto["name"],
                "symbol": crypto["symbol"],
                "slug": crypto["slug"],
                "supply": {
                    "max": crypto["max_supply"],
                    "circulating": crypto["circulating_supply"],
                    "total": crypto["total_supply"],
                    "infinite": crypto["infinite_supply"],
                },
                "quote": {
                    "USD": {
                        "price": crypto["quote"]["USD"]["price"],
                        "volume_24h": crypto["quote"]["USD"]["volume_24h"],
                        "changes": {
                            "1h": crypto["quote"]["USD"]["percent_change_1h"],
                            "24h": crypto["quote"]["USD"]["percent_change_24h"],
                            "7d": crypto["quote"]["USD"]["percent_change_7d"],
                            "30d": crypto["quote"]["USD"]["percent_change_30d"],
                        },
                        "market_cap": crypto["quote"]["USD"]["market_cap"],
                        "dominance": crypto["quote"]["USD"]["market_cap_dominance"],
                    }
                },
            }
        )
    return cleaned


def get_weather_description(code: int) -> str:
    """Get weather description from code"""
    descriptions = {
        0: "despejado",
        1: "mayormente despejado",
        2: "parcialmente nublado",
        3: "nublado",
        45: "neblina",
        48: "niebla",
        51: "llovizna leve",
        53: "llovizna moderada",
        55: "llovizna intensa",
        56: "llovizna helada leve",
        57: "llovizna helada intensa",
        61: "lluvia leve",
        63: "lluvia moderada",
        65: "lluvia intensa",
        66: "lluvia helada leve",
        67: "lluvia helada intensa",
        71: "nevada leve",
        73: "nevada moderada",
        75: "nevada intensa",
        77: "granizo",
        80: "lluvia leve intermitente",
        81: "lluvia moderada intermitente",
        82: "lluvia fuerte intermitente",
        85: "nevada leve intermitente",
        86: "nevada intensa intermitente",
        95: "tormenta",
        96: "tormenta con granizo leve",
        99: "tormenta con granizo intenso",
    }
    return descriptions.get(code, "clima raro")


def build_system_message(context: Dict, include_tools: bool = False) -> Dict[str, Any]:
    """Build system message with personality and context.

    include_tools: when True, advertise available tool calls and exact call syntax
    so the model can request a tool in plain text (provider-agnostic).
    """
    config = load_bot_config()
    market_info = format_market_info(context.get("market") or {})
    weather_source = context.get("weather")
    weather_info = format_weather_info(weather_source) if weather_source else ""
    news_info = format_hacker_news_info(context.get("hacker_news"))
    time_context = context.get("time") or {}
    formatted_time = str(time_context.get("formatted", "")).strip()

    # Build the complete system prompt with context
    base_prompt = config.get("system_prompt", "You are a helpful AI assistant.")

    contextual_info = f"""

FECHA ACTUAL:
{formatted_time}

CONTEXTO DEL MERCADO:
{market_info}

CLIMA EN BUENOS AIRES:
{weather_info}

NOTICIAS DE HACKER NEWS:
{news_info}

CONTEXTO POLITICO:
- Javier Milei (alias miller, javo, javito, javeto) le gano a Sergio Massa y es el presidente de Argentina desde el 10/12/2023 hasta el 10/12/2027
"""

    tools_section = ""
    if include_tools:
        tools_section = (
            "\n\nHERRAMIENTAS DISPONIBLES:\n"
            "- web_search: buscador web actual (devuelve hasta 10 resultados).\n"
            "- fetch_url: trae el texto plano de una URL http/https para citar fragmentos.\n"
            "\nCÓMO LLAMAR HERRAMIENTAS:\n"
            "Escribe exactamente una línea con el formato:\n"
            "[TOOL] <nombre> {JSON}\n"
            'Ejemplos:\n  [TOOL] web_search {"query": "inflación argentina hoy"}\n'
            '  [TOOL] fetch_url {"url": "https://example.com/noticia"}\n'
            "Luego espera la respuesta y continúa con tu contestación final.\n"
            "Usá herramientas solo si realmente ayudan (actualidad, datos frescos).\n"
            "Tras usar web_search, respondé directo al usuario con síntesis breve; no devuelvas lista cruda ni arranques con 'Encontré esto sobre...'."
        )

    print(f"build_system_message: include_tools={include_tools}")
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": base_prompt + contextual_info + tools_section,
            }
        ],
    }


def build_compound_system_message() -> Dict[str, Any]:
    """Build a minimal system message tailored for Groq Compound tools."""

    tool_hint = (
        "Respondé directo al usuario con síntesis breve."
        " Si necesitás info actualizada, usá las herramientas para buscar y confirmar."
    )

    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": tool_hint,
            }
        ],
    }


def format_hacker_news_info(
    news: Optional[Iterable[Dict[str, Any]]], include_discussion: bool = True
) -> str:
    """Format Hacker News context for system or agent prompts."""

    if not news:
        return "- sin datos por ahora"

    lines: List[str] = []
    for item in news:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "(sin título)").strip()
        url = str(item.get("url") or "").strip()

        stats: List[str] = []
        points_val = item.get("points")
        if isinstance(points_val, int):
            stats.append(f"{points_val} pts")
        comments_val = item.get("comments")
        if isinstance(comments_val, int):
            stats.append(f"{comments_val} coms")

        stats_text = f" ({', '.join(stats)})" if stats else ""
        entry = f"- {title}{stats_text}"

        if url:
            entry += f" → {url}"

        if include_discussion:
            hn_url = str(item.get("comments_url") or "").strip()
            if hn_url:
                entry += f" (HN: {hn_url})"

        lines.append(entry)

    return "\n".join(lines) if lines else "- sin datos por ahora"


def format_market_info(market: Dict) -> str:
    """Format market data for context"""
    info = []

    if "crypto" in market:
        info.append("PRECIOS DE CRIPTOS:")
        info.append(json.dumps(market["crypto"]))

    if "dollar" in market:
        info.append("DOLARES:")
        info.append(json.dumps(market["dollar"]))

    if "bcra" in market:
        info.append("VARIABLES ECONOMICAS BCRA:")
        info.append(json.dumps(market["bcra"]))

    return "\n".join(info)


def format_weather_info(weather: Dict) -> str:
    """Format weather data for context"""
    return f"""
- Temperatura aparente: {weather['apparent_temperature']}°C
- Probabilidad de lluvia: {weather['precipitation_probability']}%
- Estado: {weather['description']}
- Nubosidad: {weather['cloud_cover']}%
- Visibilidad: {weather['visibility']/1000:.1f}km
"""


def build_ai_messages(
    message: Dict,
    chat_history: List[Dict],
    message_text: str,
    reply_context: Optional[str] = None,
) -> List[Dict]:
    messages = []

    # Add chat history messages (which already includes replies)
    for msg in chat_history:
        messages.append(
            {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["text"],
                    }
                ],
            }
        )

    # Get user info and context
    first_name = message["from"]["first_name"]
    username = message["from"].get("username", "")
    chat_type = message["chat"]["type"]
    chat_title = message["chat"].get("title", "") if chat_type != "private" else ""
    current_time = datetime.now(BA_TZ)

    # Build context sections
    context_parts = [
        "CONTEXTO:",
        f"- Chat: {chat_type}" + (f" ({chat_title})" if chat_title else ""),
        f"- Usuario: {first_name}" + (f" ({username})" if username else ""),
        f"- Hora: {current_time.strftime('%H:%M')}",
    ]

    if reply_context:
        context_parts.extend(
            [
                "",
                "MENSAJE AL QUE RESPONDE:",
                truncate_text(reply_context),
            ]
        )

    context_parts.extend(
        [
            "",
            "MENSAJE:",
            truncate_text(message_text),
            "",
            "INSTRUCCIONES:",
            "- Mantené el personaje del gordo",
            "- Usá lenguaje coloquial argentino",
        ]
    )

    messages.append(
        {
            "role": "user",
            "content": "\n".join(context_parts),
        }
    )

    return messages[-8:]


def _noop_command() -> str:
    return ""


def _noop_param_command(_arg: str) -> str:
    return ""


def initialize_commands() -> Dict[str, Tuple[Callable, bool, bool]]:
    """Initialize command handlers with a deduplicated command registry."""

    return _build_command_registry(
        {
            "ask_ai": ask_ai,
            "show_agent_thoughts": show_agent_thoughts,
            "config_command": _noop_command,
            "convert_base": convert_base,
            "select_random": select_random,
            "get_prices": get_prices,
            "get_dollar_rates": get_dollar_rates,
            "get_polymarket_argentina_election": get_polymarket_argentina_election,
            "get_rulo": get_rulo,
            "get_devo": get_devo,
            "powerlaw": powerlaw,
            "rainbow": rainbow,
            "satoshi": satoshi,
            "get_timestamp": get_timestamp,
            "convert_to_command": convert_to_command,
            "search_command": search_command,
            "get_instance_name": get_instance_name,
            "get_help": get_help,
            "handle_transcribe": handle_transcribe,
            "handle_bcra_variables": handle_bcra_variables,
            "topup_command": _noop_command,
            "balance_command": _noop_command,
            "transfer_command": _noop_param_command,
        }
    )


def truncate_text(text: Optional[str], max_length: int = 512) -> str:
    return _state_truncate_text(text, max_length)


configure_agent_memory(
    redis_factory=config_redis,
    admin_reporter=admin_report,
    truncate_text=truncate_text,
    tz=BA_TZ,
)


def save_message_to_redis(
    chat_id: str, message_id: str, text: str, redis_client: redis.Redis
) -> None:
    _state_save_message_to_redis(
        chat_id,
        message_id,
        text,
        redis_client,
        admin_reporter=admin_report,
    )


def get_chat_history(
    chat_id: str, redis_client: redis.Redis, max_messages: int = 8
) -> List[Dict]:
    return _state_get_chat_history(
        chat_id,
        redis_client,
        admin_reporter=admin_report,
        max_messages=max_messages,
    )


def should_gordo_respond(
    commands: Dict[str, Tuple[Callable, bool, bool]],
    command: str,
    message_text: str,
    message: dict,
    chat_config: Mapping[str, Any],
    reply_metadata: Optional[Mapping[str, Any]],
) -> bool:
    return _command_should_gordo_respond(
        commands,
        command,
        message_text,
        message,
        chat_config,
        reply_metadata,
        load_bot_config_fn=load_bot_config,
    )


def should_auto_process_media(
    commands: Mapping[str, Tuple[Callable, bool, bool]],
    command: str,
    message_text: str,
    message: Mapping[str, Any],
) -> bool:
    return _command_should_auto_process_media(
        commands,
        command,
        message_text,
        message,
    )


def check_global_rate_limit(redis_client: redis.Redis) -> bool:
    """Check whether the main Groq chat lane still has minute/day budget."""

    return _peek_groq_rate_limit("chat", redis_client)


def get_ai_credits_per_response() -> int:
    return _billing_get_ai_credits_per_response()


def get_ai_onboarding_credits() -> int:
    return _billing_get_ai_onboarding_credits()


def get_ai_billing_packs() -> List[Dict[str, int]]:
    return _billing_get_ai_billing_packs()


def get_ai_billing_pack(pack_id: str) -> Optional[Dict[str, int]]:
    return _billing_get_ai_billing_pack(pack_id)


def build_topup_keyboard() -> Dict[str, Any]:
    return _billing_build_topup_keyboard()


def _parse_topup_payload(payload: str) -> Tuple[Optional[str], Optional[int]]:
    return _billing_parse_topup_payload(payload)


def build_insufficient_credits_message(
    *, chat_type: str, user_balance: int, chat_balance: int
) -> str:
    return _billing_build_insufficient_credits_message(
        chat_type=chat_type,
        user_balance=user_balance,
        chat_balance=chat_balance,
    )


def _extract_numeric_chat_id(chat_id: str) -> Optional[int]:
    return _billing_extract_numeric_chat_id(chat_id)


def _extract_user_id(message: Mapping[str, Any]) -> Optional[int]:
    return _billing_extract_user_id(message)


def _maybe_grant_onboarding_credits(user_id: Optional[int]) -> None:
    _billing_maybe_grant_onboarding_credits(
        credits_db_service,
        admin_report,
        user_id,
    )


def _format_balance_command(chat_type: str, user_id: int, chat_id: int) -> str:
    class _BalanceServiceAdapter:
        @staticmethod
        def get_balance(scope_type: str, scope_id: int) -> int:
            return _fetch_balance(cast(Literal["user", "chat"], scope_type), scope_id)

    return _billing_format_balance_command(
        _BalanceServiceAdapter(),
        chat_type=chat_type,
        user_id=user_id,
        chat_id=chat_id,
    )


def _fetch_balance(scope_type: Literal["user", "chat"], scope_id: int) -> int:
    return credits_db_service.get_balance(scope_type, int(scope_id))


def _message_handler_maybe_grant_onboarding(
    _service: Any,
    _reporter: Callable[..., None],
    user_id: Optional[int],
) -> None:
    _maybe_grant_onboarding_credits(user_id)


def _message_handler_format_balance_command(_service: Any, **kwargs: Any) -> str:
    return _format_balance_command(
        kwargs["chat_type"],
        kwargs["user_id"],
        kwargs["chat_id"],
    )


def _send_stars_invoice(
    *,
    chat_id: str,
    user_id: int,
    pack: Mapping[str, int],
) -> bool:
    payload = f"topup:{pack['id']}:{user_id}"
    request_payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "title": f"Pack IA {pack['credits']} créditos",
        "description": f"Recarga de {pack['credits']} créditos para mensajes IA",
        "payload": payload,
        "provider_token": "",
        "currency": "XTR",
        "prices": [{"label": f"{pack['credits']} créditos IA", "amount": pack["xtr"]}],
    }
    payload_response, error = _telegram_request(
        "sendInvoice",
        method="POST",
        json_payload=request_payload,
    )
    return error is None and bool(payload_response)


def extract_message_text(message: Dict) -> str:
    """Extract text content from different message types"""
    # Prioritize text, then caption, then poll question
    if "text" in message and message["text"]:
        return str(message["text"]).strip()
    if "caption" in message and message["caption"]:
        return str(message["caption"]).strip()
    if "poll" in message and isinstance(message["poll"], dict):
        return str(message["poll"].get("question", "")).strip()
    return ""


def extract_message_content(message: Dict) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract text, photo/sticker file_id, and audio file_id from message"""
    text = extract_message_text(message)

    # Extract photo or sticker (get highest resolution)
    photo_file_id = None

    # First, check for photo in the main message
    if "photo" in message and message["photo"]:
        # Telegram sends array of photos in different resolutions
        # Take the last one (highest resolution)
        photo_file_id = message["photo"][-1]["file_id"]

    # Check for sticker in the main message
    elif "sticker" in message and message["sticker"]:
        photo_file_id = message["sticker"]["file_id"]
        print(f"Found sticker: {photo_file_id}")

    # If no photo/sticker in main message, check in replied message
    elif "reply_to_message" in message and message["reply_to_message"]:
        replied_msg = message["reply_to_message"]
        if "photo" in replied_msg and replied_msg["photo"]:
            photo_file_id = replied_msg["photo"][-1]["file_id"]
            print(f"Found photo in quoted message: {photo_file_id}")
        elif "sticker" in replied_msg and replied_msg["sticker"]:
            photo_file_id = replied_msg["sticker"]["file_id"]
            print(f"Found sticker in quoted message: {photo_file_id}")

    # Extract audio/voice
    audio_file_id = None
    if "voice" in message and message["voice"]:
        audio_file_id = message["voice"]["file_id"]
    elif "audio" in message and message["audio"]:
        audio_file_id = message["audio"]["file_id"]
    # Also check for audio in replied message
    elif "reply_to_message" in message and message["reply_to_message"]:
        replied_msg = message["reply_to_message"]
        if "voice" in replied_msg and replied_msg["voice"]:
            audio_file_id = replied_msg["voice"]["file_id"]
            print(f"Found voice in quoted message: {audio_file_id}")
        elif "audio" in replied_msg and replied_msg["audio"]:
            audio_file_id = replied_msg["audio"]["file_id"]
            print(f"Found audio in quoted message: {audio_file_id}")

    return text, photo_file_id, audio_file_id


def download_telegram_file(file_id: str) -> Optional[bytes]:
    """Download file from Telegram"""
    try:
        token = environ.get("TELEGRAM_TOKEN")

        # Get file path from Telegram
        file_info_url = f"https://api.telegram.org/bot{token}/getFile"
        file_response = requests.get(
            file_info_url, params={"file_id": file_id}, timeout=10
        )
        file_response.raise_for_status()

        file_data = file_response.json()
        if not file_data.get("ok"):
            print(f"Error getting file info: {file_data}")
            return None

        file_path = file_data["result"]["file_path"]

        # Download actual file
        file_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        download_response = requests.get(file_url, timeout=30)
        download_response.raise_for_status()

        return download_response.content

    except Exception as e:
        print(f"Error downloading Telegram file: {e}")
        return None


def _get_groq_client() -> Optional[OpenAI]:
    groq_api_key = environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Groq API key not configured")
        return None
    return OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")


def _extract_response_text(response: Any) -> Optional[str]:
    if response is None:
        return None

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        output = response.get("output")
    else:
        output = getattr(response, "output", None)

    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            content = None
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"}:
                            chunk = block.get("text") or block.get("output_text")
                            if chunk:
                                parts.append(str(chunk))
                    else:
                        chunk = getattr(block, "text", None) or getattr(
                            block, "output_text", None
                        )
                        if chunk:
                            parts.append(str(chunk))
            else:
                if isinstance(item, dict):
                    chunk = item.get("text") or item.get("output_text")
                else:
                    chunk = getattr(item, "text", None) or getattr(
                        item, "output_text", None
                    )
                if chunk:
                    parts.append(str(chunk))

        if parts:
            return " ".join(part.strip() for part in parts if part.strip()).strip()

    return None


def describe_image_groq(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[str]:
    """Describe image using Groq vision models."""

    if file_id and use_cache:
        cached = get_cached_description(file_id)
        if cached:
            return str(cached)

    image_base64 = encode_image_to_base64(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"

    def _attempt() -> Optional[str]:
        groq_client = _get_groq_client()
        if groq_client is None:
            return None
        print("Describing image with Groq vision model...")
        input_payload = cast(
            ResponseInputParam,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )

        response = groq_client.responses.create(
            model=GROQ_VISION_MODEL,
            input=input_payload,
            max_output_tokens=512,
        )
        description = _extract_response_text(response)
        if description:
            print(f"Image description successful: {description[:100]}...")
        return description

    description = _invoke_provider(
        "groq",
        attempt=_attempt,
        rate_limit_backoff=GROQ_RATE_LIMIT_BACKOFF_SECONDS,
        label="Groq Vision",
        rate_limit_scope="vision",
    )

    if description and file_id:
        cache_description(file_id, description)

    return description


def transcribe_audio_groq(
    audio_data: bytes,
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[str]:
    """Transcribe audio using Groq Whisper."""

    if file_id and use_cache:
        cached = get_cached_transcription(file_id)
        if cached:
            return str(cached)

    def _attempt() -> Optional[str]:
        groq_client = _get_groq_client()
        if groq_client is None:
            return None
        print("Transcribing audio with Groq Whisper...")
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        response = groq_client.audio.transcriptions.create(
            model=GROQ_TRANSCRIBE_MODEL,
            file=audio_file,
        )
        transcription = None
        if isinstance(response, dict):
            transcription = response.get("text")
        else:
            transcription = getattr(response, "text", None)
        if transcription:
            print(f"Audio transcribed successfully: {transcription[:100]}...")
        return transcription

    transcription = _invoke_provider(
        "groq",
        attempt=_attempt,
        rate_limit_backoff=GROQ_RATE_LIMIT_BACKOFF_SECONDS,
        label="Groq Whisper",
        rate_limit_scope="transcribe",
    )

    if transcription and file_id:
        cache_transcription(file_id, transcription)

    return transcription


def _process_media_with_cache(
    *,
    file_id: str,
    use_cache: bool,
    cache_lookup: Optional[Callable[[str], Optional[str]]],
    processor: Callable[[bytes], Optional[str]],
    downloader: Optional[Callable[[str], Optional[bytes]]] = None,
    failure_code: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Shared helper for cached Telegram media download + processing."""

    try:
        if use_cache and cache_lookup:
            cached_value = cache_lookup(file_id)
            if cached_value:
                return str(cached_value), None

        media_fetcher = downloader or download_telegram_file
        media_bytes = media_fetcher(file_id)
        if not media_bytes:
            return None, "download"

        result = processor(media_bytes)
        if result:
            return result, None
        return None, failure_code
    except Exception as error:
        print(f"Error processing media {file_id}: {error}")
        return None, failure_code


def transcribe_file_by_id(
    file_id: str, use_cache: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch transcription for a Telegram file_id with cache and retries.

    Returns (text, error):
    - On success: (transcription, None)
    - If download failed: (None, "download")
    - If transcription failed: (None, "transcribe")
    """
    return _process_media_with_cache(
        file_id=file_id,
        use_cache=use_cache,
        cache_lookup=get_cached_transcription,
        processor=lambda media: transcribe_audio_groq(
            media, file_id, use_cache=use_cache
        ),
        failure_code="transcribe",
    )


def describe_media_by_id(
    file_id: str, prompt: str
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch description for an image/sticker by Telegram file_id using Groq vision.

    Returns (description, error):
    - On success: (description, None)
    - If download failed: (None, "download")
    - If description failed: (None, "describe")
    """
    def _processor(media: bytes) -> Optional[str]:
        resized = resize_image_if_needed(media)
        return describe_image_groq(resized, prompt, file_id)

    return _process_media_with_cache(
        file_id=file_id,
        use_cache=True,
        cache_lookup=get_cached_description,
        processor=_processor,
        failure_code="describe",
    )


def resize_image_if_needed(image_data: bytes, max_size: int = 512) -> bytes:
    """Resize image if it's too large for vision processing"""
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_data))
        original_size = image.size

        # Check if resize is needed
        if max(original_size) > max_size:

            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / original_size[0], max_size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

            # Resize image
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert back to bytes
            output_buffer = io.BytesIO()
            # Save as JPEG to ensure compatibility and smaller size
            if resized_image.mode in ("RGBA", "LA", "P"):
                # Convert to RGB for JPEG
                resized_image = resized_image.convert("RGB")
            resized_image.save(output_buffer, format="JPEG", quality=85, optimize=True)

            resized_data = output_buffer.getvalue()
            return resized_data
        else:
            return image_data

    except ImportError:
        print("WARNING: PIL not available, cannot resize image")
        return image_data
    except Exception as e:
        print(f"ERROR: Failed to resize image: {e}")
        return image_data


def encode_image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to base64 string for AI models"""

    base64_encoded = base64.b64encode(image_data).decode("utf-8")

    return base64_encoded


def parse_command(message_text: str, bot_name: str) -> Tuple[str, str]:
    return _command_parse_command(message_text, bot_name)


def _log_config_event(message: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    log_entry: Dict[str, Any] = {"scope": "config", "message": message}
    if extra:
        for key, value in extra.items():
            log_entry[key] = value
    print(json.dumps(log_entry, ensure_ascii=False, default=str))


def _decode_redis_value(value: Any) -> Optional[str]:
    return _chat_decode_redis_value(value)


def _is_group_chat_type(chat_type: Optional[str]) -> bool:
    return _chat_is_group_chat_type(chat_type)


def get_chat_config(redis_client: redis.Redis, chat_id: str) -> Dict[str, Any]:
    return _chat_get_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_report,
        log_event=_log_config_event,
    )


def set_chat_config(
    redis_client: redis.Redis, chat_id: str, **updates: Any
) -> Dict[str, Any]:
    return _chat_set_chat_config(
        redis_client,
        chat_id,
        chat_config_db_service=chat_config_db_service,
        admin_reporter=admin_report,
        log_event=_log_config_event,
        **updates,
    )


def _coerce_bool(value: Any, *, default: bool) -> bool:
    return _chat_coerce_bool(value, default=default)


def build_config_text(config: Mapping[str, Any]) -> str:
    return _chat_build_config_text(config)


def build_config_keyboard(config: Mapping[str, Any]) -> Dict[str, Any]:
    return _chat_build_config_keyboard(config)


_WEBHOOK_CALLBACKS_CHECKED = False


def ensure_callback_updates_enabled() -> None:
    """Ensure the webhook can deliver callback and payment updates."""

    global _WEBHOOK_CALLBACKS_CHECKED
    if _WEBHOOK_CALLBACKS_CHECKED:
        return

    token = environ.get("TELEGRAM_TOKEN")
    webhook_key = environ.get("WEBHOOK_AUTH_KEY")
    function_url = environ.get("FUNCTION_URL")

    if not token or not webhook_key or not function_url:
        _WEBHOOK_CALLBACKS_CHECKED = True
        return

    webhook_info = get_telegram_webhook_info(token)
    if webhook_info.get("error"):
        _WEBHOOK_CALLBACKS_CHECKED = True
        admin_report("falló la lectura del webhook para callbacks")
        return

    allowed_updates = webhook_info.get("allowed_updates")
    if isinstance(allowed_updates, list):
        updates = {str(value) for value in allowed_updates}
        required_updates = {"callback_query", "pre_checkout_query"}
        if not allowed_updates or required_updates.issubset(updates):
            _WEBHOOK_CALLBACKS_CHECKED = True
            return
    elif allowed_updates is None:
        _WEBHOOK_CALLBACKS_CHECKED = True
        return

    expected_url = f"{function_url}?key={webhook_key}"
    current_url = webhook_info.get("url")
    if current_url and current_url != expected_url:
        _WEBHOOK_CALLBACKS_CHECKED = True
        _log_config_event(
            "el webhook apunta a otra url; salteo update de callbacks",
            {"current_url": current_url},
        )
        return

    updated = set_telegram_webhook(function_url)
    if updated:
        _log_config_event(
            "Updated webhook to enable callbacks and pre-checkout updates",
            {"url": expected_url},
        )
        _WEBHOOK_CALLBACKS_CHECKED = True
    else:
        _WEBHOOK_CALLBACKS_CHECKED = True
        admin_report("falló update del webhook para callbacks y pre-checkout")


def handle_config_command(chat_id: str) -> Tuple[str, Dict[str, Any]]:
    ensure_callback_updates_enabled()
    redis_client = config_redis()
    config = get_chat_config(redis_client, chat_id)
    return build_config_text(config), build_config_keyboard(config)


def _answer_callback_query(
    callback_query_id: str,
    *,
    text: Optional[str] = None,
    show_alert: bool = False,
) -> None:
    payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
        if show_alert:
            payload["show_alert"] = True

    _telegram_request(
        "answerCallbackQuery",
        method="POST",
        json_payload=payload,
        log_errors=False,
        expect_json=False,
    )


def handle_topup_callback(callback_query: Dict[str, Any]) -> None:
    callback_data = str(callback_query.get("data") or "")
    callback_id = callback_query.get("id")
    message = cast(Dict[str, Any], callback_query.get("message") or {})
    chat = cast(Dict[str, Any], message.get("chat") or {})
    user = cast(Dict[str, Any], callback_query.get("from") or {})
    chat_id = chat.get("id")
    chat_type = str(chat.get("type", ""))

    if chat_id is None:
        if callback_id:
            _answer_callback_query(callback_id)
        return

    if not credits_db_service.is_configured():
        if callback_id:
            _answer_callback_query(
                callback_id,
                text="el cobro de ia está hecho pelota, avisale al admin",
                show_alert=True,
            )
        return

    if chat_type != "private":
        if callback_id:
            _answer_callback_query(
                callback_id,
                text="cargá por privado, maestro",
                show_alert=True,
            )
        return

    parts = callback_data.split(":", 1)
    pack = get_ai_billing_pack(parts[1] if len(parts) == 2 else "")
    if not pack:
        if callback_id:
            _answer_callback_query(
                callback_id,
                text="ese pack es fruta, elegí otro",
                show_alert=True,
            )
        return

    try:
        user_id = int(user.get("id"))
    except (TypeError, ValueError):
        if callback_id:
            _answer_callback_query(callback_id)
        return

    sent_ok = _send_stars_invoice(chat_id=str(chat_id), user_id=user_id, pack=pack)
    if callback_id:
        if sent_ok:
            _answer_callback_query(callback_id, text="listo, te dejé la factura")
        else:
            _answer_callback_query(
                callback_id,
                text="no pude armar la factura, probá de nuevo",
                show_alert=True,
            )


def _answer_pre_checkout_query(
    query_id: str,
    *,
    ok: bool,
    error_message: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"pre_checkout_query_id": query_id, "ok": bool(ok)}
    if not ok and error_message:
        payload["error_message"] = error_message
    _telegram_request(
        "answerPreCheckoutQuery",
        method="POST",
        json_payload=payload,
        log_errors=False,
        expect_json=False,
    )


def handle_pre_checkout_query(pre_checkout_query: Dict[str, Any]) -> None:
    query_id = pre_checkout_query.get("id")
    if not query_id:
        return

    if not credits_db_service.is_configured():
        _answer_pre_checkout_query(
            str(query_id),
            ok=False,
            error_message="el cobro de ia está hecho pelota, avisale al admin",
        )
        return

    payload = str(pre_checkout_query.get("invoice_payload") or "")
    currency = str(pre_checkout_query.get("currency") or "")
    from_user = cast(Dict[str, Any], pre_checkout_query.get("from") or {})
    pack_id, payload_user_id = _parse_topup_payload(payload)
    pack = get_ai_billing_pack(pack_id or "")

    try:
        user_id = int(from_user.get("id"))
    except (TypeError, ValueError):
        _answer_pre_checkout_query(
            str(query_id),
            ok=False,
            error_message="tu usuario vino medio roto para cobrar",
        )
        return

    try:
        total_amount = int(pre_checkout_query.get("total_amount"))
    except (TypeError, ValueError):
        total_amount = -1

    if (
        not pack
        or currency != "XTR"
        or int(pack["xtr"]) != total_amount
        or (payload_user_id is not None and payload_user_id != user_id)
    ):
        _answer_pre_checkout_query(
            str(query_id),
            ok=False,
            error_message="ese pago vino raro y no te lo pude validar",
        )
        return

    _answer_pre_checkout_query(str(query_id), ok=True)


def handle_successful_payment_message(message: Dict[str, Any]) -> str:
    chat = cast(Dict[str, Any], message.get("chat") or {})
    chat_id_raw = chat.get("id")
    if chat_id_raw is None:
        return "ok"
    chat_id = str(chat_id_raw)

    if not credits_db_service.is_configured():
        send_msg(chat_id, "el cobro de ia no está andando, avisale al admin")
        return "ok"

    user_id = _extract_user_id(message)
    if user_id is None:
        return "ok"

    successful_payment = cast(Dict[str, Any], message.get("successful_payment") or {})
    currency = str(successful_payment.get("currency") or "")
    payload = str(successful_payment.get("invoice_payload") or "")
    charge_id = str(successful_payment.get("telegram_payment_charge_id") or "")
    pack_id, payload_user_id = _parse_topup_payload(payload)
    pack = get_ai_billing_pack(pack_id or "")

    try:
        total_amount = int(successful_payment.get("total_amount"))
    except (TypeError, ValueError):
        total_amount = -1

    if (
        not charge_id
        or not pack
        or currency != "XTR"
        or total_amount != int(pack["xtr"])
        or (payload_user_id is not None and payload_user_id != user_id)
    ):
        send_msg(chat_id, "me cayó un pago raro y no lo pude validar, avisale al admin")
        admin_report(
            "Invalid successful payment payload",
            None,
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "currency": currency,
                "payload": payload,
                "total_amount": total_amount,
                "charge_id": charge_id,
            },
        )
        return "ok"

    try:
        payment_result = credits_db_service.record_star_payment(
            telegram_payment_charge_id=charge_id,
            user_id=user_id,
            pack_id=str(pack["id"]),
            xtr_amount=int(pack["xtr"]),
            credits_awarded=int(pack["credits"]),
            payload=payload,
        )
    except Exception as error:
        admin_report(
            "falló persistencia de pago exitoso",
            error,
            {"chat_id": chat_id, "user_id": user_id, "charge_id": charge_id},
        )
        send_msg(chat_id, "me entró la guita pero se trabó la acreditación, avisale al admin")
        return "ok"

    balance = int(payment_result.get("user_balance", 0))
    if payment_result.get("inserted"):
        send_msg(
            chat_id,
            (
                f"listo, te cargué {pack['credits']} créditos\n"
                f"ahora te quedaron {balance}\n"
                "si querés mandarle al grupo: /transfer <monto>"
            ),
        )
    else:
        send_msg(chat_id, f"ese pago ya estaba cargado, no rompas las bolas\nte quedaron {balance}")
    return "ok"


def edit_message(
    chat_id: str, message_id: int, text: str, reply_markup: Dict[str, Any]
) -> bool:
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "reply_markup": reply_markup,
    }
    payload_response, error = _telegram_request(
        "editMessageText", method="POST", json_payload=payload
    )
    return error is None and bool(payload_response)


def handle_callback_query(callback_query: Dict[str, Any]) -> None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    user = callback_query.get("from") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")

    if not callback_data or chat_id is None or message_id is None:
        if callback_id:
            _answer_callback_query(callback_id)
        return

    if str(callback_data).startswith("topup:"):
        handle_topup_callback(callback_query)
        return

    redis_client = config_redis()
    chat_id_str = str(chat_id)
    chat_type = str(chat.get("type", ""))

    is_config_callback = str(callback_data).startswith("cfg:")
    if is_config_callback and _is_group_chat_type(chat_type):
        if not is_chat_admin(chat_id_str, user.get("id"), redis_client=redis_client):
            denial_message = "solo los admins pueden tocar esta config, maestro"
            if callback_id:
                _answer_callback_query(callback_id)
            send_msg(chat_id_str, denial_message, str(message_id))
            _report_unauthorized_config_attempt(
                chat_id_str,
                user,
                chat_type=chat_type,
                action="callback:config",
                callback_data=str(callback_data),
            )
            return

    config = get_chat_config(redis_client, chat_id_str)

    try:
        _, action, value = callback_data.split(":", 2)
    except ValueError:
        if callback_id:
            _answer_callback_query(callback_id)
        return

    if action == "link" and value in {"reply", "delete", "off"}:
        config = set_chat_config(redis_client, chat_id_str, link_mode=value)
    elif action == "random":
        current = _coerce_bool(config.get("ai_random_replies"), default=True)
        config = set_chat_config(
            redis_client,
            chat_id_str,
            ai_random_replies=not current,
        )
    elif action == "followups":
        current = _coerce_bool(config.get("ai_command_followups"), default=True)
        config = set_chat_config(
            redis_client,
            chat_id_str,
            ai_command_followups=not current,
        )

    text = build_config_text(config)
    keyboard = build_config_keyboard(config)
    try:
        edit_succeeded = edit_message(
            chat_id_str, int(message_id), text, keyboard
        )
        if not edit_succeeded:
            _log_config_event(
                "Falling back to new config message",
                {"chat_id": chat_id_str, "message_id": message_id},
            )
            send_msg(chat_id_str, text, reply_markup=keyboard)
    finally:
        if callback_id:
            _answer_callback_query(callback_id)


def save_bot_message_metadata(
    redis_client: redis.Redis,
    chat_id: str,
    message_id: Union[str, int],
    metadata: Mapping[str, Any],
    ttl: int = BOT_MESSAGE_META_TTL,
) -> None:
    _state_save_bot_message_metadata(
        redis_client,
        chat_id,
        message_id,
        metadata,
        admin_reporter=admin_report,
        ttl=ttl,
    )


def get_bot_message_metadata(
    redis_client: redis.Redis, chat_id: str, message_id: Union[str, int]
) -> Optional[Dict[str, Any]]:
    return _state_get_bot_message_metadata(
        redis_client,
        chat_id,
        message_id,
        admin_reporter=admin_report,
        decode_redis_value=_decode_redis_value,
    )


def build_reply_context_text(message: Mapping[str, Any]) -> Optional[str]:
    return _state_build_reply_context_text(
        message,
        extract_message_text_fn=extract_message_text,
    )


def format_user_message(
    message: Dict,
    message_text: str,
    reply_context: Optional[str] = None,
) -> str:
    return _state_format_user_message(message, message_text, reply_context)


def handle_msg(message: Dict) -> str:
    return _handle_msg_impl(
        message,
        MessageHandlerDeps(
            config_redis=config_redis,
            get_chat_config=get_chat_config,
            initialize_commands=initialize_commands,
            parse_command=parse_command,
            should_auto_process_media=should_auto_process_media,
            extract_message_content=extract_message_content,
            replace_links=replace_links,
            send_msg=send_msg,
            delete_msg=delete_msg,
            admin_report=admin_report,
            get_bot_message_metadata=get_bot_message_metadata,
            save_bot_message_metadata=save_bot_message_metadata,
            build_reply_context_text=build_reply_context_text,
            should_gordo_respond=should_gordo_respond,
            format_user_message=format_user_message,
            save_message_to_redis=save_message_to_redis,
            get_chat_history=get_chat_history,
            build_ai_messages=build_ai_messages,
            handle_ai_response=handle_ai_response,
            ask_ai=ask_ai,
            gen_random=gen_random,
            build_insufficient_credits_message=build_insufficient_credits_message,
            get_ai_credits_per_response=get_ai_credits_per_response,
            build_topup_keyboard=build_topup_keyboard,
            credits_db_service=credits_db_service,
            is_group_chat_type=_is_group_chat_type,
            extract_user_id=_extract_user_id,
            extract_numeric_chat_id=_extract_numeric_chat_id,
            maybe_grant_onboarding_credits=_message_handler_maybe_grant_onboarding,
            format_balance_command=_message_handler_format_balance_command,
            handle_transcribe_with_message=handle_transcribe_with_message,
            check_global_rate_limit=check_global_rate_limit,
            handle_rate_limit=handle_rate_limit,
            handle_successful_payment_message=handle_successful_payment_message,
            handle_config_command=handle_config_command,
            ensure_callback_updates_enabled=ensure_callback_updates_enabled,
            is_chat_admin=is_chat_admin,
            report_unauthorized_config_attempt=_report_unauthorized_config_attempt,
            handle_transcribe=handle_transcribe,
            _transcribe_audio_file=_transcribe_audio_file,
            _transcription_error_message=_transcription_error_message,
            download_telegram_file=download_telegram_file,
            resize_image_if_needed=resize_image_if_needed,
            encode_image_to_base64=encode_image_to_base64,
        ),
    )


def handle_rate_limit(chat_id: str, message: Dict) -> str:
    """Handle rate limited responses"""
    token = environ.get("TELEGRAM_TOKEN")
    if token:
        send_typing(token, chat_id)
    time.sleep(random.uniform(0, 1))
    return gen_random(message["from"]["first_name"])


def remove_gordo_prefix(text: Optional[str]) -> str:
    return _ai_remove_gordo_prefix(text)


def clean_duplicate_response(response: str) -> str:
    return _ai_clean_duplicate_response(response)


def handle_ai_response(
    chat_id: str,
    handler_func: Callable,
    messages: List[Dict],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    context_texts: Optional[Sequence[Optional[str]]] = None,
    user_identity: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
) -> str:
    return _ai_handle_response(
        chat_id,
        handler_func,
        messages,
        image_data=image_data,
        image_file_id=image_file_id,
        context_texts=context_texts,
        user_identity=user_identity,
        response_meta=response_meta,
        send_typing_fn=send_typing,
        telegram_token=environ.get("TELEGRAM_TOKEN"),
        reset_request_count_fn=_reset_ai_provider_request_count,
        restore_request_count_fn=_restore_ai_provider_request_count,
        get_request_count_fn=_get_ai_provider_request_count,
        sanitize_tool_artifacts_fn=sanitize_tool_artifacts,
        strip_ai_fallback_marker_fn=_strip_ai_fallback_marker,
    )


def get_telegram_webhook_info(token: str) -> Dict[str, Union[str, dict]]:
    payload, error = _telegram_request(
        "getWebhookInfo", token=token, log_errors=False
    )
    if not payload:
        return {"error": error or "request failed"}

    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return {"error": "unexpected response"}


def set_telegram_webhook(webhook_url: str) -> bool:
    webhook_key = environ.get("WEBHOOK_AUTH_KEY")
    token = environ.get("TELEGRAM_TOKEN")
    secret_token = hashlib.sha256(Fernet.generate_key()).hexdigest()
    parameters = {
        "url": f"{webhook_url}?key={webhook_key}",
        "allowed_updates": '["message","callback_query","pre_checkout_query"]',
        "secret_token": secret_token,
        "max_connections": 8,
    }
    payload_response, error = _telegram_request(
        "setWebhook", params=parameters, token=token, expect_json=False
    )
    if error is not None:
        return False
    redis_client = config_redis()
    redis_response = redis_client.set("X-Telegram-Bot-Api-Secret-Token", secret_token)
    return bool(redis_response)


def verify_webhook() -> bool:
    token = environ.get("TELEGRAM_TOKEN")
    if not token:
        return False

    webhook_key = environ.get("WEBHOOK_AUTH_KEY")
    function_url = environ.get("FUNCTION_URL")

    if not function_url or not webhook_key:
        return False

    webhook_info = get_telegram_webhook_info(token)
    if "error" in webhook_info:
        return False

    expected_webhook_url = f"{function_url}?key={webhook_key}"
    current_webhook_url = webhook_info.get("url")

    return current_webhook_url == expected_webhook_url


def is_secret_token_valid(request: Request) -> bool:
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    redis_client = config_redis()
    redis_secret_token = redis_client.get("X-Telegram-Bot-Api-Secret-Token")
    return redis_secret_token == secret_token


def _arg_is_true(args: Mapping[str, Any], key: str) -> bool:
    value = args.get(key)
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def _handle_webhook_actions(args: Mapping[str, Any]) -> Optional[Tuple[str, int]]:
    if _arg_is_true(args, "check_webhook"):
        webhook_verified = verify_webhook()
        return (
            ("webhook joya", 200)
            if webhook_verified
            else ("el webhook está hecho pelota", 400)
        )

    if _arg_is_true(args, "update_webhook"):
        function_url = environ.get("FUNCTION_URL")
        if not function_url:
            return "no pude acomodar el webhook", 400
        updated = set_telegram_webhook(function_url)
        return ("webhook acomodado", 200) if updated else ("no pude acomodar el webhook", 400)

    return None


def _handle_control_actions(args: Mapping[str, Any]) -> Optional[Tuple[str, int]]:
    if _arg_is_true(args, "update_dollars"):
        get_dollar_rates()
        return "dólares actualizados", 200

    if _arg_is_true(args, "run_agent"):
        try:
            # Operational exception: autonomous run does not bill user/group AI credits.
            thought_result = run_agent_cycle()
            payload = json.dumps({"status": "ok", "thought": thought_result}, ensure_ascii=False)
            return payload, 200
        except Exception as agent_error:
            admin_report("falló ejecución del agente", agent_error)
            return "el agente se hizo mierda", 500

    return None


def process_request_parameters(request: Request) -> Tuple[str, int]:
    try:
        args = request.args

        webhook_response = _handle_webhook_actions(args)
        if webhook_response:
            return webhook_response

        control_response = _handle_control_actions(args)
        if control_response:
            return control_response

        # Validate secret token
        if not is_secret_token_valid(request):
            admin_report("token secreto inválido")
            return "token secreto inválido", 400

        # Process message
        request_json = request.get_json(silent=True)
        if not request_json:
            return "json inválido", 400
        callback_query = request_json.get("callback_query")
        if callback_query:
            handle_callback_query(callback_query)
            return "ok", 200
        pre_checkout_query = request_json.get("pre_checkout_query")
        if pre_checkout_query:
            handle_pre_checkout_query(cast(Dict[str, Any], pre_checkout_query))
            return "ok", 200

        message = request_json.get("message")
        if not message:
            return "sin mensaje", 200

        handle_msg(message)
        return "ok", 200

    except Exception as e:
        error_context = {
            "request_method": request.method,
            "request_args": dict(request.args),
            "request_path": request.path,
        }

        error_msg = f"Request processing error: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return "error procesando request", 500


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def responder() -> Tuple[str, int]:
    try:
        webhook_key = request.args.get("key")
        if not webhook_key:
            return "falta key", 200

        if webhook_key != environ.get("WEBHOOK_AUTH_KEY"):
            admin_report("intento con key inválida")
            return "key incorrecta", 400

        response_message, status_code = process_request_parameters(request)
        return response_message, status_code
    except Exception as e:
        error_context = {
            "request_method": request.method,
            "request_args": dict(request.args),
            "request_path": request.path,
        }

        error_msg = "error crítico en responder"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return "error crítico", 500
