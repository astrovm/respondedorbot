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
from api.services import bcra as bcra_service
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
    if compound_system_message:
        compound_messages = [{"role": "user", "content": query}]
        compound_response = get_groq_compound_response(None, compound_messages)
        if compound_response:
            return compound_response

    tool_name = "web_search"
    tool_args = {"query": query}
    try:
        print(
            f"ask_ai: executing forced tool '{tool_name}' args={json.dumps(tool_args)[:200]}"
        )
        tool_output = execute_tool(tool_name, tool_args)
    except Exception as tool_err:
        tool_output = f"Error al ejecutar herramienta {tool_name}: {tool_err}"
        print(f"ask_ai: forced tool '{tool_name}' raised error: {tool_err}")

    forced_query, forced_results = normalize_web_search_output(tool_output)
    tool_contexts = [
        {
            "tool": tool_name,
            "args": tool_args,
            "result": tool_output,
        }
    ]

    top_url = ""
    for result in forced_results or []:
        if isinstance(result, Mapping):
            top_url = str(result.get("url") or "").strip()
            if top_url:
                break

    if top_url:
        fetch_args = {"url": top_url, "max_chars": 1500}
        try:
            fetch_output = execute_tool("fetch_url", fetch_args)
            tool_contexts.append(
                {
                    "tool": "fetch_url",
                    "args": fetch_args,
                    "result": fetch_output,
                }
            )
        except Exception as fetch_err:
            print(f"ask_ai: forced fetch_url failed: {fetch_err}")

    next_messages = messages + [
        {
            "role": "user",
            "content": f"RESULTADO DE HERRAMIENTA:\n{json.dumps(tool_contexts)[:4000]}",
        },
    ]
    forced_final = complete_with_providers(system_message, next_messages)
    forced_tool_final = resolve_tool_calls(system_message, next_messages, forced_final)
    if forced_tool_final:
        return forced_tool_final
    if forced_final:
        return forced_final
    return summarize_search_results(forced_query, forced_results)


CHAT_CONFIG_KEY_PREFIX = "chat_config:"
CHAT_CONFIG_DEFAULTS = {
    "link_mode": "off",
    "ai_random_replies": True,
    "ai_command_followups": True,
}
CHAT_ADMIN_STATUS_TTL = 300
BOT_MESSAGE_META_PREFIX = "bot_message_meta:"
BOT_MESSAGE_META_TTL = 3 * 24 * 60 * 60  # 3 days


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
    """Return a stable cache key composed of *prefix* and an MD5 hash of *payload*."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
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
    """Delegate to utils helper while preserving legacy monkeypatch hook."""

    return _links_replace_links(text, embed_checker=can_embed_url)

# Provider backoff windows (seconds)
GROQ_RATE_LIMIT_BACKOFF_SECONDS = 600  # wait 10 minutes after a rate limit response
OPENROUTER_RATE_LIMIT_BACKOFF_SECONDS = 600


_provider_backoff_until: Dict[str, float] = {}


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

RATE_LIMIT_GLOBAL_MAX = 1024
RATE_LIMIT_CHAT_MAX = 128
TTL_RATE_GLOBAL = 3600  # 1 hour
TTL_RATE_CHAT = 600  # 10 minutes

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


# generic proxy for caching any request
def cached_requests(
    api_url,
    parameters,
    headers,
    expiration_time,
    hourly_cache=False,
    get_history=False,
    verify_ssl=True,
):
    """Generic proxy for caching any request"""
    try:
        arguments_dict = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
        }
        request_hash = hashlib.md5(
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
        # Keep call signature compatible with tests that patch this helper
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

    if "IN " in msg_text.upper():
        words = msg_text.upper().split()
        coins = [
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
        ]
        convert_to = words[-1]
        if convert_to in coins:
            if convert_to == "SATS":
                convert_to_parameter = "BTC"
            else:
                convert_to_parameter = convert_to
            msg_text = msg_text.upper().replace(f"IN {convert_to}", "").strip()
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
            return "Error getting crypto prices"

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
        return "Error getting crypto prices"

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
            return "Invalid input. Fee should be between 0 and 100, and purchase amount should be a positive number."

        dollars = _fetch_criptoya_dollar_data()

        if not dollars or "data" not in dollars:
            return "Error getting dollar rates"

        usdt_ask = float(dollars["data"]["cripto"]["usdt"]["ask"])
        usdt_bid = float(dollars["data"]["cripto"]["usdt"]["bid"])
        usdt = (usdt_ask + usdt_bid) / 2
        oficial = float(dollars["data"]["oficial"]["price"])
        tarjeta = float(dollars["data"]["tarjeta"]["price"])

        profit = -(fee * usdt + oficial - usdt) / tarjeta

        msg = f"""Profit: {fmt_num(profit * 100, 2)}%

Fee: {fmt_num(fee * 100, 2)}%
Oficial: {fmt_num(oficial, 2)}
USDT: {fmt_num(usdt, 2)}
Tarjeta: {fmt_num(tarjeta, 2)}"""

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
        return "Invalid input. Usage: /devo <fee_percentage>[, <purchase_amount>]"


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
        f"  • Spread vs oficial: {_format_local_signed(diff)} ARS ({fmt_signed_pct(pct, 2)}%)",
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
        return "Error consiguiendo cotizaciones del dólar"

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
            return "Error getting BTC USD price"
        if btc_price_ars is None:
            return "Error getting BTC ARS price"

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
        return "Error al obtener las variables del BCRA"


_DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = {
    "download": "No pude descargar el audio",
    "transcribe": "No pude transcribir el audio, intentá más tarde",
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
            return "Respondé a un mensaje con audio o imagen para transcribir/describir"

        replied_msg = message["reply_to_message"]

        _, photo_file_id, audio_file_id = extract_message_content(replied_msg)

        if audio_file_id:
            text, error_code = _transcribe_audio_file(audio_file_id, use_cache=True)
            if text:
                return f"🎵 Transcripción: {text}"
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
                    success_prefix="🖼️ Descripción: ",
                    download_error="No pude descargar la imagen",
                    describe_error="No pude describir la imagen, intentá más tarde",
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
                    success_prefix="🎨 Descripción del sticker: ",
                    download_error="No pude descargar el sticker",
                    describe_error="No pude describir el sticker, intentá más tarde",
                )
                if sticker_response:
                    return sticker_response

        return "El mensaje no contiene audio, imagen o sticker para transcribir/describir"

    except Exception as e:
        print(f"Error in handle_transcribe: {e}")
        return "Error procesando el comando, intentá más tarde"


def handle_transcribe() -> str:
    """Transcribe command wrapper - requires special handling in message processor"""
    return "El comando /transcribe debe usarse respondiendo a un mensaje con audio o imagen"


def powerlaw() -> str:
    today = datetime.now(timezone.utc)
    since = datetime(day=4, month=1, year=2009).replace(tzinfo=timezone.utc)
    days_since = (today - since).days

    # Giovanni Santostasi Bitcoin Power Law model
    # Formula: 1.0117e-17 * (days since genesis block)^5.82
    value = 1.0117e-17 * (days_since**5.82)

    price = get_btc_price("USD")
    if price is None:
        return "Error getting BTC price for power law calculation"

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
        return "Error getting BTC price for rainbow calculation"

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
    converter = kakasi()
    converter.setMode("H", "a")
    converter.setMode("K", "a")
    converter.setMode("J", "a")
    converter.setMode("r", "Hepburn")
    converter.setMode("s", True)
    converter.setMode("C", True)
    return converter.getConverter().do(text)


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
comandos disponibles boludo:

- /ask, /pregunta, /che, /gordo: te contesto cualquier gilada

- /bcra, /variables: te tiro las variables economicas del bcra

- /comando, /command algo: te lo convierto en comando de telegram

- /convertbase 101, 2, 10: te paso numeros entre bases (ej: binario 101 a decimal)

- /buscar algo: te busco en la web

- /agent: te muestro lo ultimo que penso el agente autonomo

- /eleccion: odds actuales de Polymarket para Diputados 2025

- /devo 0.5, 100: te calculo el arbitraje entre tarjeta y crypto (fee%, monto opcional)

- /rulo: te armo los rulos desde el oficial

- /dolar, /dollar, /usd: te tiro la posta del blue y todos los dolares

- /instance: te digo donde estoy corriendo

- /config: cambiá la config del gordo, link fixer y demases

- /prices, /precio, /precios, /presio, /presios, /bresio, /bresios, /brecio, /brecios: top 10 cryptos en usd
- /prices in btc: top 10 en btc
- /prices 20: top 20 en usd
- /prices 100 in eur: top 100 en eur
- /prices btc, eth, xmr: bitcoin, ethereum y monero en usd
- /prices dai in sats: dai en satoshis
- /prices stables: stablecoins en usd

- /random pizza, carne, sushi: elijo por vos
- /random 1-10: numero random del 1 al 10

- /powerlaw: te tiro el precio justo de btc segun power law y si esta caro o barato
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


def send_msg(
    chat_id: str,
    msg: str,
    msg_id: str = "",
    buttons: Optional[List[str]] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    payload: Dict[str, Any] = {"chat_id": chat_id, "text": msg}
    if "polymarket.com" in msg.lower():
        payload["disable_web_page_preview"] = True
    if msg_id:
        payload["reply_to_message_id"] = msg_id

    markup = reply_markup
    if markup is None and buttons:
        keyboard = [[{"text": "Open in app", "url": url}] for url in buttons]
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
    formatted_message = f"Admin report from {instance_name}: {message}"

    # Add extra context if provided
    if extra_context:
        context_details = "\n\nAdditional Context:"
        for key, value in extra_context.items():
            context_details += f"\n{key}: {value}"
        formatted_message += context_details

    # Add error details if provided
    if error:
        error_details = f"\n\nError Type: {type(error).__name__}"
        error_details += f"\nError Message: {str(error)}"

        error_details += f"\n\nTraceback:\n{traceback.format_exc()}"

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
    """Return ``True`` if *user_id* is an admin of the chat."""

    if not chat_id or user_id is None:
        return False

    redis_client = redis_client or _optional_redis_client()
    cache_key = _chat_admin_cache_key(chat_id, user_id)

    if redis_client:
        cached_value = redis_get_json(redis_client, cache_key)
        cached_flag = (
            cached_value.get("is_admin")
            if isinstance(cached_value, Mapping)
            else cached_value
        )
        if isinstance(cached_flag, bool):
            return cached_flag

    payload, error = _telegram_request(
        "getChatMember",
        method="GET",
        params={"chat_id": chat_id, "user_id": user_id},
        log_errors=False,
    )

    is_admin = False
    if payload and payload.get("ok"):
        result = payload.get("result") or {}
        status = str(result.get("status") or "").lower()
        is_admin = status in {"administrator", "creator"}
    else:
        _log_config_event(
            "Failed to verify chat admin status",
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "error": error or (payload or {}).get("description"),
            },
        )

    if redis_client:
        redis_setex_json(
            redis_client,
            cache_key,
            CHAT_ADMIN_STATUS_TTL,
            {"is_admin": is_admin},
        )

    return is_admin


def _report_unauthorized_config_attempt(
    chat_id: str,
    user: Mapping[str, Any],
    *,
    chat_type: Optional[str],
    action: str,
    callback_data: Optional[str] = None,
) -> None:
    context: Dict[str, Any] = {
        "chat_id": chat_id,
        "chat_type": chat_type,
        "user_id": user.get("id"),
        "username": user.get("username"),
        "action": action,
    }
    if callback_data:
        context["callback_data"] = callback_data

    _log_config_event("Unauthorized config attempt", context)


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

        # Create OpenAI client
        openrouter = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=environ.get("OPENROUTER_API_KEY"),
        )

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
        return get_fallback_response(messages)

    except Exception as e:
        error_context = {
            "messages_count": len(messages),
            "messages_preview": [msg.get("content", "")[:100] for msg in messages],
        }
        admin_report("Error in ask_ai", e, error_context)
        return get_fallback_response(messages)


def complete_with_providers(
    system_message: Dict[str, Any], messages: List[Dict[str, Any]]
) -> Optional[str]:
    """Try Groq, then OpenRouter and return the first response."""

    def _run_provider(
        *,
        name: str,
        backoff_key: str,
        attempt: Callable[[], Optional[str]],
        skip_reason: Optional[str] = None,
    ) -> Optional[str]:
        if skip_reason:
            print(skip_reason)
            return None

        if is_provider_backoff_active(backoff_key):
            remaining = int(get_provider_backoff_remaining(backoff_key))
            print(
                f"{name} backoff active ({remaining}s remaining), skipping {name} attempts"
            )
            return None

        response = attempt()
        if response:
            print(f"complete_with_providers: got response from {name}")
            return response

        return None

    openrouter_api_key = environ.get("OPENROUTER_API_KEY")

    providers: Sequence[Tuple[str, str, Callable[[], Optional[str]], Optional[str]]] = [
        (
            "Groq",
            "groq",
            lambda: get_groq_ai_response(system_message, messages),
            None,
        ),
        (
            "OpenRouter",
            "openrouter",
            lambda: get_ai_response(
                OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key),
                system_message,
                messages,
                provider_name="openrouter",
            )
            if openrouter_api_key
            else None,
            None if openrouter_api_key else "OpenRouter API key not configured",
        ),
    ]

    for name, backoff_key, attempt, skip_reason in providers:
        result = _run_provider(
            name=name,
            backoff_key=backoff_key,
            attempt=attempt,
            skip_reason=skip_reason,
        )
        if result:
            return result

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
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "query vacío"
        # Always use a fixed limit so the model can't choose it
        results = web_search(query, limit=10)
        try:
            print(f"execute_tool:web_search: q='{query}' results={len(results)}")
        except Exception:
            pass
        return json.dumps({"query": query, "results": results})
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
) -> Optional[str]:
    """Execute a provider call with shared backoff and error handling."""

    display_name = label or provider_name.capitalize()
    if is_provider_backoff_active(provider_name):
        remaining = int(get_provider_backoff_remaining(provider_name))
        print(
            f"{display_name} backoff active ({remaining}s remaining), skipping API call"
        )
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


def get_ai_response(
    client: OpenAI,
    system_msg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    provider_name: str = "openrouter",
    backoff_seconds: Optional[int] = None,
) -> Optional[str]:
    """Get AI response (text-only) from a generic OpenAI-compatible provider."""

    models = [
        "moonshotai/kimi-k2:free",
    ]

    rate_limit_backoff = (
        backoff_seconds
        if backoff_seconds is not None
        else OPENROUTER_RATE_LIMIT_BACKOFF_SECONDS
    )

    def _attempt() -> Optional[str]:
        print(f"Attempt 1/1 using model: {models[0]}")

        response = client.chat.completions.create(
            model=models[0],
            extra_body={
                "models": models[1:],
            },
            messages=cast(Any, [system_msg] + messages),
            max_tokens=1024,
        )

        if response and hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "stop":
                return response.choices[0].message.content
        return None

    return _invoke_provider(
        provider_name,
        attempt=_attempt,
        rate_limit_backoff=rate_limit_backoff,
        label=provider_name.capitalize(),
    )



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
    )


def get_fallback_response(messages: List[Dict]) -> str:
    """Generate fallback random response"""
    first_name = ""
    if messages and len(messages) > 0:
        last_message = messages[-1]["content"]
        if "Usuario: " in last_message:
            first_name = last_message.split("Usuario: ")[1].split(" ")[0]
    return gen_random(first_name)


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


def initialize_commands() -> Dict[str, Tuple[Callable, bool, bool]]:
    """
    Initialize command handlers with metadata.
    Returns dict of command name -> (handler_function, uses_ai, takes_params)
    """
    return {
        # AI-based commands
        "/ask": (ask_ai, True, True),
        "/pregunta": (ask_ai, True, True),
        "/che": (ask_ai, True, True),
        "/gordo": (ask_ai, True, True),
        "/agent": (show_agent_thoughts, False, False),
        # Regular commands
        "/config": (lambda: "", False, False),
        "/convertbase": (convert_base, False, True),
        "/random": (select_random, False, True),
        "/prices": (get_prices, False, True),
        "/precios": (get_prices, False, True),
        "/precio": (get_prices, False, True),
        "/presios": (get_prices, False, True),
        "/presio": (get_prices, False, True),
        "/bresio": (get_prices, False, True),
        "/bresios": (get_prices, False, True),
        "/brecio": (get_prices, False, True),
        "/brecios": (get_prices, False, True),
        "/dolar": (get_dollar_rates, False, False),
        "/dollar": (get_dollar_rates, False, False),
        "/usd": (get_dollar_rates, False, False),
        "/eleccion": (get_polymarket_argentina_election, False, False),
        "/rulo": (get_rulo, False, False),
        "/devo": (get_devo, False, True),
        "/powerlaw": (powerlaw, False, False),
        "/rainbow": (rainbow, False, False),
        "/satoshi": (satoshi, False, False),
        "/sat": (satoshi, False, False),
        "/sats": (satoshi, False, False),
        "/time": (get_timestamp, False, False),
        "/comando": (convert_to_command, False, True),
        "/command": (convert_to_command, False, True),
        "/buscar": (search_command, False, True),
        "/search": (search_command, False, True),
        "/instance": (get_instance_name, False, False),
        "/help": (get_help, False, False),
        "/transcribe": (handle_transcribe, False, False),
        "/bcra": (handle_bcra_variables, False, False),
        "/variables": (handle_bcra_variables, False, False),
    }


def truncate_text(text: Optional[str], max_length: int = 512) -> str:
    """Truncate text to max_length and add ellipsis if needed"""

    if text is None:
        return ""

    if max_length <= 0:
        return ""

    if max_length <= 3:
        return "." * max_length

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


configure_agent_memory(
    redis_factory=config_redis,
    admin_reporter=admin_report,
    truncate_text=truncate_text,
    tz=BA_TZ,
)


def save_message_to_redis(
    chat_id: str, message_id: str, text: str, redis_client: redis.Redis
) -> None:
    try:
        chat_history_key = f"chat_history:{chat_id}"
        message_ids_key = f"chat_message_ids:{chat_id}"

        # Check if the message ID already exists using a SET structure
        if redis_client.sismember(message_ids_key, message_id):
            # Message already exists, don't save again
            return

        # Message doesn't exist, save it
        history_entry = json.dumps(
            {
                "id": message_id,
                "text": truncate_text(text),
                "timestamp": int(time.time()),
            }
        )

        pipe = redis_client.pipeline()
        pipe.lpush(chat_history_key, history_entry)  # Add new message to history list
        pipe.sadd(
            message_ids_key, message_id
        )  # Add message ID to set for duplicate tracking
        pipe.ltrim(chat_history_key, 0, 31)  # Keep only last 32 messages

        # Keep the message_ids set in sync with the history
        # Get all message IDs from history (this is expensive but needed to maintain consistency)
        pipe.lrange(chat_history_key, 0, -1)
        results = pipe.execute()

        # Last result is the list of messages after trimming
        message_entries = results[-1]
        valid_ids = set()
        for entry in message_entries:
            try:
                msg = json.loads(entry)
                valid_ids.add(msg["id"])
            except (json.JSONDecodeError, KeyError):
                continue

        # Remove any IDs from the set that are no longer in the history
        try:
            current_ids_set = redis_client.smembers(message_ids_key)
            current_ids = (
                list(cast(Set[str], current_ids_set)) if current_ids_set else []
            )
            to_remove = [id for id in current_ids if id not in valid_ids]
        except Exception:
            current_ids = []
            to_remove = []
        if to_remove:
            redis_client.srem(message_ids_key, *to_remove)

    except Exception as e:
        error_context = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text_length": len(text),
        }
        error_msg = f"Redis save message error: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)


def get_chat_history(
    chat_id: str, redis_client: redis.Redis, max_messages: int = 8
) -> List[Dict]:
    try:
        chat_history_key = f"chat_history:{chat_id}"
        history: List[str] = cast(
            List[str], redis_client.lrange(chat_history_key, 0, max_messages - 1)
        )

        if not history:
            return []

        messages = []
        for entry in history:
            try:
                msg = json.loads(entry)
                # Add role based on if it's from the bot or user
                is_bot = msg["id"].startswith("bot_")
                msg["role"] = "assistant" if is_bot else "user"
                messages.append(msg)
            except json.JSONDecodeError as decode_error:
                error_context = {"chat_id": chat_id, "entry": entry}
                error_msg = f"JSON decode error in chat history: {str(decode_error)}"
                print(error_msg)
                admin_report(error_msg, decode_error, error_context)
                continue

        return list(reversed(messages))
    except Exception as e:
        error_context = {"chat_id": chat_id, "max_messages": max_messages}
        error_msg = f"Error retrieving chat history: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return []


def should_gordo_respond(
    commands: Dict[str, Tuple[Callable, bool, bool]],
    command: str,
    message_text: str,
    message: dict,
    chat_config: Mapping[str, Any],
    reply_metadata: Optional[Mapping[str, Any]],
) -> bool:
    """Decide if the bot should respond to a message"""
    # Get message context
    message_lower = message_text.lower()
    chat_type = message["chat"]["type"]
    bot_username = environ.get("TELEGRAM_USERNAME")
    bot_name = f"@{bot_username}"

    # Ignore replies to link replacement messages
    reply = message.get("reply_to_message", {})
    if reply.get("from", {}).get("username") == bot_username:
        reply_text = reply.get("text") or ""
        replacement_domains = (
            "fxtwitter.com",
            "fixupx.com",
            "fxbsky.app",
            "kkinstagram.com",
            "rxddit.com",
        )
        if any(domain in reply_text for domain in replacement_domains):
            return False

    # Response conditions
    is_command = command in commands
    is_private = chat_type == "private"
    is_mention = bot_name in message_lower
    is_reply = reply.get("from", {}).get("username", "") == bot_username

    if (
        is_reply
        and reply_metadata
        and reply_metadata.get("type") == "command"
        and not bool(reply_metadata.get("uses_ai", False))
        and not bool(chat_config.get("ai_command_followups", True))
    ):
        return False

    # Check trigger keywords with 10% chance
    try:
        config = load_bot_config()
        trigger_words = config.get("trigger_words", ["bot", "assistant"])
    except ValueError:
        trigger_words = ["bot", "assistant"]
    if bool(chat_config.get("ai_random_replies", True)):
        is_trigger = (
            any(word in message_lower for word in trigger_words)
            and random.random() < 0.1
        )
    else:
        is_trigger = False

    return (
        is_command
        or not command.startswith("/")
        and (is_trigger or is_private or is_mention or is_reply)
    )


def check_rate_limit(chat_id: str, redis_client: redis.Redis) -> bool:
    """
    Checkea si un chat_id o el bot global superó el rate limit
    Returns True si puede hacer requests, False si está limitado
    """
    try:
        pipe = redis_client.pipeline()

        # Check global rate limit (requests/hour)
        hour_key = "rate_limit:global:hour"
        pipe.incr(hour_key)
        pipe.expire(hour_key, TTL_RATE_GLOBAL, nx=True)

        # Check individual chat rate limit (requests/10 minutes)
        chat_key = f"rate_limit:chat:{chat_id}"
        pipe.incr(chat_key)
        pipe.expire(chat_key, TTL_RATE_CHAT, nx=True)

        # Execute all commands atomically
        results = pipe.execute()

        # Get the final counts (every 2nd index starting from 0)
        hour_count = results[0] or 0  # Convert None to 0
        chat_count = results[2] or 0  # Convert None to 0

        return hour_count <= RATE_LIMIT_GLOBAL_MAX and chat_count <= RATE_LIMIT_CHAT_MAX
    except redis.RedisError:
        return False


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

    groq_client = _get_groq_client()
    if groq_client is None:
        return None

    image_base64 = encode_image_to_base64(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"

    def _attempt() -> Optional[str]:
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

    groq_client = _get_groq_client()
    if groq_client is None:
        return None

    def _attempt() -> Optional[str]:
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
    """Parse command and message text from input"""
    # Handle empty or whitespace-only messages
    message_text = message_text.strip()
    if not message_text:
        return "", ""

    # Split into command and rest of message
    split_message = message_text.split(" ", 1)
    command = split_message[0].lower().replace(bot_name, "")

    # Normalize Hangul filler alias to /ask
    if command.startswith("/"):
        command_body = command[1:]
        if command_body and all(char == "\u3164" for char in command_body):
            command = "/ask"

    # Get message text and handle extra spaces
    if len(split_message) > 1:
        message_text = split_message[1].lstrip()  # Remove leading spaces only
    else:
        message_text = ""

    return command, message_text


def _chat_config_key(chat_id: str) -> str:
    return f"{CHAT_CONFIG_KEY_PREFIX}{chat_id}"


def _legacy_link_mode_key(chat_id: str) -> str:
    return f"link_mode:{chat_id}"


def _log_config_event(message: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    log_entry: Dict[str, Any] = {"scope": "config", "message": message}
    if extra:
        for key, value in extra.items():
            log_entry[key] = value
    print(json.dumps(log_entry, ensure_ascii=False, default=str))


def _decode_redis_value(value: Any) -> Optional[str]:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if value is not None:
        return str(value)
    return None


def _is_group_chat_type(chat_type: Optional[str]) -> bool:
    return str(chat_type) in {"group", "supergroup"}


def get_chat_config(redis_client: redis.Redis, chat_id: str) -> Dict[str, Any]:
    config = dict(CHAT_CONFIG_DEFAULTS)
    try:
        _log_config_event("Loading chat config", {"chat_id": chat_id})
        raw_value = redis_client.get(_chat_config_key(chat_id))
        raw_value_text = _decode_redis_value(raw_value)
        deserializable_value = raw_value_text
        _log_config_event(
            "Chat config raw value fetched",
            {
                "chat_id": chat_id,
                "raw_value": raw_value_text,
            },
        )
        if deserializable_value:
            try:
                loaded = json.loads(deserializable_value)
                if isinstance(loaded, dict):
                    for key, value in loaded.items():
                        if key in config:
                            config[key] = value
            except json.JSONDecodeError:
                pass
        else:
            legacy_key = _legacy_link_mode_key(chat_id)
            legacy_value = redis_client.get(legacy_key)
            legacy_value_text = _decode_redis_value(legacy_value)
            if legacy_value_text:
                _log_config_event(
                    "Using legacy link mode config",
                    {
                        "chat_id": chat_id,
                        "legacy_value": legacy_value_text,
                    },
                )
                config["link_mode"] = legacy_value_text
            else:
                _log_config_event(
                    "No stored chat config found; using defaults",
                    {"chat_id": chat_id},
                )
    except Exception as error:
        admin_report(
            "Error loading chat config",
            error,
            {"chat_id": chat_id},
        )
    return config


def set_chat_config(
    redis_client: redis.Redis, chat_id: str, **updates: Any
) -> Dict[str, Any]:
    config = get_chat_config(redis_client, chat_id)
    for key, value in updates.items():
        if key in config:
            config[key] = value

    try:
        _log_config_event(
            "Saving chat config",
            {"chat_id": chat_id, "updates": updates, "config": config},
        )
        redis_client.set(_chat_config_key(chat_id), json.dumps(config))
        link_mode = config.get("link_mode", "off")
        legacy_key = _legacy_link_mode_key(chat_id)
        if link_mode in {"reply", "delete"}:
            redis_client.set(legacy_key, link_mode)
            _log_config_event(
                "Persisted legacy link mode",
                {"chat_id": chat_id, "link_mode": link_mode},
            )
        else:
            redis_client.delete(legacy_key)
            _log_config_event("Cleared legacy link mode", {"chat_id": chat_id})
    except Exception as error:
        admin_report(
            "Error saving chat config",
            error,
            {"chat_id": chat_id, "updates": updates},
        )

    return config


def _build_config_choice_button(
    label: str, value: str, current: str, *, action: str
) -> Dict[str, str]:
    prefix = "✅" if current == value else "▫️"
    return {
        "text": f"{prefix} {label}",
        "callback_data": f"cfg:{action}:{value}",
    }


def _build_config_toggle_button(label: str, enabled: bool, action: str) -> Dict[str, str]:
    prefix = "✅" if enabled else "▫️"
    return {
        "text": f"{prefix} {label}",
        "callback_data": f"cfg:{action}:toggle",
    }


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Normalize truthy config values that might be stored as strings."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "no", "off", "disabled"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    if value is None:
        return default

    return default


def build_config_text(config: Mapping[str, Any]) -> str:
    link_mode = str(config.get("link_mode", "off"))
    random_enabled = _coerce_bool(config.get("ai_random_replies"), default=True)
    followups_enabled = _coerce_bool(
        config.get("ai_command_followups"), default=True
    )

    link_labels = {
        "delete": "Delete original message",
        "reply": "Reply to original message",
        "off": "Off",
    }

    lines = [
        "Gordo config:",
        "",
        f"Link fixer: {link_labels.get(link_mode, link_mode)}",
        f"Random AI replies: {'✅ enabled' if random_enabled else '▫️ disabled'}",
        "Follow-ups for non-AI commands: "
        f"{'✅ enabled' if followups_enabled else '▫️ disabled'}",
        "",
        "Use the buttons below to change the settings.",
    ]
    return "\n".join(lines)


def build_config_keyboard(config: Mapping[str, Any]) -> Dict[str, Any]:
    link_mode = str(config.get("link_mode", "off"))
    random_enabled = _coerce_bool(config.get("ai_random_replies"), default=True)
    followups_enabled = _coerce_bool(
        config.get("ai_command_followups"), default=True
    )

    keyboard = [
        [
            _build_config_choice_button(
                "Reply to original message", "reply", link_mode, action="link"
            ),
            _build_config_choice_button(
                "Delete original message", "delete", link_mode, action="link"
            ),
            _build_config_choice_button("Off", "off", link_mode, action="link"),
        ],
        [
            _build_config_toggle_button(
                "Random AI replies", random_enabled, action="random"
            ),
        ],
        [
            _build_config_toggle_button(
                "Follow-ups for non-AI commands",
                followups_enabled,
                action="followups",
            ),
        ],
    ]
    return {"inline_keyboard": keyboard}


_WEBHOOK_CALLBACKS_CHECKED = False


def ensure_callback_updates_enabled() -> None:
    """Ensure the Telegram webhook can deliver callback_query updates."""

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
        admin_report("Failed to fetch webhook info for callbacks")
        return

    allowed_updates = webhook_info.get("allowed_updates")
    if isinstance(allowed_updates, list):
        if not allowed_updates or "callback_query" in allowed_updates:
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
            "Webhook points to different URL; skipping callback update",
            {"current_url": current_url},
        )
        return

    updated = set_telegram_webhook(function_url)
    if updated:
        _log_config_event(
            "Updated webhook to enable callback queries",
            {"url": expected_url},
        )
        _WEBHOOK_CALLBACKS_CHECKED = True
    else:
        _WEBHOOK_CALLBACKS_CHECKED = True
        admin_report("Failed to update webhook for callback queries")


def handle_config_command(chat_id: str) -> Tuple[str, Dict[str, Any]]:
    ensure_callback_updates_enabled()
    redis_client = config_redis()
    config = get_chat_config(redis_client, chat_id)
    return build_config_text(config), build_config_keyboard(config)


def _answer_callback_query(callback_query_id: str) -> None:
    _telegram_request(
        "answerCallbackQuery",
        method="POST",
        json_payload={"callback_query_id": callback_query_id},
        log_errors=False,
        expect_json=False,
    )


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

    redis_client = config_redis()
    chat_id_str = str(chat_id)
    chat_type = str(chat.get("type", ""))

    is_config_callback = str(callback_data).startswith("cfg:")
    if is_config_callback and _is_group_chat_type(chat_type):
        if not is_chat_admin(chat_id_str, user.get("id"), redis_client=redis_client):
            denial_message = "Solo los admins pueden cambiar la config del gordo acá."
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


def _bot_message_meta_key(chat_id: str, message_id: Union[str, int]) -> str:
    return f"{BOT_MESSAGE_META_PREFIX}{chat_id}:{message_id}"


def save_bot_message_metadata(
    redis_client: redis.Redis,
    chat_id: str,
    message_id: Union[str, int],
    metadata: Mapping[str, Any],
    ttl: int = BOT_MESSAGE_META_TTL,
) -> None:
    try:
        redis_client.setex(
            _bot_message_meta_key(chat_id, message_id),
            ttl,
            json.dumps(dict(metadata)),
        )
    except Exception as error:
        admin_report(
            "Error saving bot message metadata",
            error,
            {"chat_id": chat_id, "message_id": message_id},
        )


def get_bot_message_metadata(
    redis_client: redis.Redis, chat_id: str, message_id: Union[str, int]
) -> Optional[Dict[str, Any]]:
    try:
        raw_value = redis_client.get(_bot_message_meta_key(chat_id, message_id))
        deserializable_value = _decode_redis_value(raw_value)
        if deserializable_value:
            try:
                loaded = json.loads(deserializable_value)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                pass
    except Exception as error:
        admin_report(
            "Error loading bot message metadata",
            error,
            {"chat_id": chat_id, "message_id": message_id},
        )
    return None


def _format_user_identity(user: Mapping[str, Any]) -> str:
    """Build a display name for a Telegram user"""

    first_name = user.get("first_name", "") if user else ""
    username = user.get("username", "") if user else ""

    first_name = "" if first_name is None else str(first_name)
    username = "" if username is None else str(username)

    return f"{first_name}" + (f" ({username})" if username else "")


def _describe_replied_message(reply_msg: Mapping[str, Any]) -> Optional[str]:
    """Generate a short description for a replied-to message"""

    reply_text = extract_message_text(cast(Dict[str, Any], reply_msg))
    if reply_text:
        return reply_text

    if reply_msg.get("photo"):
        return "una foto sin texto"
    if reply_msg.get("sticker"):
        sticker = reply_msg.get("sticker", {})
        emoji_char = cast(Dict[str, Any], sticker).get("emoji")
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


def build_reply_context_text(message: Mapping[str, Any]) -> Optional[str]:
    """Return contextual text describing the message being replied to"""

    reply_msg = message.get("reply_to_message") if message else None
    if not isinstance(reply_msg, Mapping):
        return None

    reply_description = _describe_replied_message(reply_msg)
    if not reply_description:
        return None

    reply_user = _format_user_identity(cast(Mapping[str, Any], reply_msg.get("from", {})))
    reply_user = reply_user.strip()

    if reply_user:
        return f"{reply_user}: {reply_description}"

    return reply_description


def format_user_message(
    message: Dict,
    message_text: str,
    reply_context: Optional[str] = None,
) -> str:
    """Format message with user info and optional reply context"""

    formatted_user = _format_user_identity(message.get("from", {}))

    if reply_context:
        if formatted_user:
            return f"{formatted_user} (en respuesta a {reply_context}): {message_text}"
        return f"(en respuesta a {reply_context}): {message_text}"

    return f"{formatted_user}: {message_text}"


def handle_msg(message: Dict) -> str:
    try:
        # Extract multimedia content
        message_text, photo_file_id, audio_file_id = extract_message_content(message)
        chat = cast(Dict[str, Any], message.get("chat", {}))
        message_id = str(message.get("message_id"))
        chat_id = str(chat.get("id"))
        chat_type = str(chat.get("type", ""))
        user_identity = _format_user_identity(message.get("from", {})).strip()

        # Process audio first if present (but not for /transcribe commands)
        if audio_file_id and not (
            message_text and message_text.strip().lower().startswith("/transcribe")
        ):
            print(f"Processing audio message: {audio_file_id}")
            transcription, err = _transcribe_audio_file(audio_file_id, use_cache=False)
            if transcription:
                message_text = transcription
                print(f"Audio transcribed: {message_text[:100]}...")
            else:
                message_text = _transcription_error_message(
                    err,
                    download_message="no pude bajar tu audio, mandalo de vuelta",
                    transcribe_message="mandame texto que no soy alexa, boludo",
                ) or "mandame texto que no soy alexa, boludo"

        # Download image if present
        image_base64 = None
        resized_image_data = None
        if photo_file_id:
            print(f"Processing image message: {photo_file_id}")
            image_data = download_telegram_file(photo_file_id)
            if image_data:
                # Resize image if needed for vision compatibility
                resized_image_data = resize_image_if_needed(image_data)
                image_base64 = encode_image_to_base64(resized_image_data)
                print(f"Image encoded to base64: {len(image_base64)} chars")
                if not message_text:
                    message_text = "que onda con esta foto"
            else:
                if not message_text:
                    message_text = "no pude ver tu foto, boludo"

        # Initialize Redis and commands
        redis_client = config_redis()
        chat_config = get_chat_config(redis_client, chat_id)

        link_mode = str(chat_config.get("link_mode", "off"))
        if link_mode != "off" and message_text and not message_text.startswith("/"):
            fixed_text, changed, original_links = replace_links(message_text)
            if changed:
                user_info = message.get("from", {})
                username = user_info.get("username")
                if username:
                    shared_by = f"@{username}"
                else:
                    name_parts = [
                        part
                        for part in (
                            user_info.get("first_name"),
                            user_info.get("last_name"),
                        )
                        if part
                    ]
                    shared_by = " ".join(name_parts)

                if shared_by:
                    fixed_text += f"\n\nShared by {shared_by}"
                reply_id = message.get("reply_to_message", {}).get("message_id")
                if reply_id is not None:
                    reply_id = str(reply_id)
                if link_mode == "delete":
                    delete_msg(chat_id, message_id)
                    if reply_id:
                        send_msg(chat_id, fixed_text, reply_id, original_links)
                    else:
                        send_msg(chat_id, fixed_text, buttons=original_links)
                else:
                    send_msg(
                        chat_id,
                        fixed_text,
                        reply_id or message_id,
                        original_links,
                    )
                return "ok"
            urls = re.findall(r"https?://\S+", message_text)
            if urls:
                return "ok"

        commands = initialize_commands()
        bot_name = f"@{environ.get('TELEGRAM_USERNAME')}"

        # Get command and message text
        command, sanitized_message_text = parse_command(message_text, bot_name)

        reply_metadata: Optional[Dict[str, Any]] = None
        if "reply_to_message" in message:
            reply_msg = message["reply_to_message"]
            if reply_msg.get("from", {}).get("username") == environ.get(
                "TELEGRAM_USERNAME"
            ):
                reply_id = reply_msg.get("message_id")
                if reply_id is not None:
                    reply_metadata = get_bot_message_metadata(
                        redis_client, chat_id, reply_id
                    )

        reply_context_text = build_reply_context_text(message)

        # Check if we should respond
        if not should_gordo_respond(
            commands, command, sanitized_message_text, message, chat_config, reply_metadata
        ):
            # Even if we don't respond, save the message for context
            if message_text:
                formatted_message = format_user_message(
                    message, message_text, reply_context_text
                )
                save_message_to_redis(
                    chat_id, message_id, formatted_message, redis_client
                )
            return "ok"

        # Handle /comando and /command with reply special case
        if (
            command in ["/comando", "/command"]
            and not sanitized_message_text
            and "reply_to_message" in message
        ):
            sanitized_message_text = extract_message_text(message["reply_to_message"])

        # If this is a reply to another message, save that message for context
        if "reply_to_message" in message:
            reply_msg = message["reply_to_message"]
            reply_text = extract_message_text(reply_msg)
            reply_id = str(reply_msg["message_id"])
            is_bot = reply_msg.get("from", {}).get("username", "") == environ.get(
                "TELEGRAM_USERNAME"
            )

            # Save all replied-to messages regardless of source
            if reply_text:
                if is_bot:
                    # For bot messages, just save the text directly
                    save_message_to_redis(
                        chat_id, f"bot_{reply_id}", reply_text, redis_client
                    )
                else:
                    # For user messages, format with user info
                    formatted_reply = format_user_message(reply_msg, reply_text)
                    save_message_to_redis(
                        chat_id, reply_id, formatted_reply, redis_client
                    )

        # Process command or conversation
        response_msg: Optional[str] = None
        response_markup: Optional[Dict[str, Any]] = None
        response_uses_ai = False
        response_command: Optional[str] = None

        if (
            command == "/config"
            and _is_group_chat_type(chat_type)
            and not is_chat_admin(
                chat_id, message.get("from", {}).get("id"), redis_client=redis_client
            )
        ):
            denial_message = "Solo los admins pueden cambiar la config del gordo acá."
            send_msg(chat_id, denial_message, message_id)
            _report_unauthorized_config_attempt(
                chat_id,
                message.get("from", {}),
                chat_type=chat_type,
                action="command:/config",
            )
            return "ok"

        if command == "/config":
            response_command = command
            response_msg, response_markup = handle_config_command(chat_id)
        elif command in commands:
            handler_func, uses_ai, takes_params = commands[command]
            response_command = command

            if uses_ai:
                if not check_rate_limit(chat_id, redis_client):
                    response_msg = handle_rate_limit(chat_id, message)
                else:
                    chat_history = get_chat_history(chat_id, redis_client)
                    messages = build_ai_messages(
                        message,
                        chat_history,
                        sanitized_message_text,
                        reply_context_text,
                    )
                    response_msg = handle_ai_response(
                        chat_id,
                        handler_func,
                        messages,
                        image_data=resized_image_data if photo_file_id else None,
                        image_file_id=photo_file_id or None,
                        context_texts=[reply_context_text],
                        user_identity=user_identity,
                    )
                    response_uses_ai = True
            else:
                if command == "/transcribe":
                    response_msg = handle_transcribe_with_message(message)
                else:
                    if takes_params:
                        response_msg = handler_func(sanitized_message_text)
                    else:
                        response_msg = handler_func()
        else:
            if not check_rate_limit(chat_id, redis_client):
                response_msg = handle_rate_limit(chat_id, message)
            else:
                chat_history = get_chat_history(chat_id, redis_client)
                messages = build_ai_messages(
                    message,
                    chat_history,
                    message_text,
                    reply_context_text,
                )
                response_msg = handle_ai_response(
                    chat_id,
                    ask_ai,
                    messages,
                    image_data=resized_image_data if photo_file_id else None,
                    image_file_id=photo_file_id or None,
                    context_texts=[reply_context_text],
                    user_identity=user_identity,
                )
                response_uses_ai = True

        # Only save messages AFTER we've generated a response
        if message_text:
            formatted_message = format_user_message(
                message, message_text, reply_context_text
            )
            save_message_to_redis(chat_id, message_id, formatted_message, redis_client)

        # Save and send response
        if response_msg:
            # Save bot response
            save_message_to_redis(
                chat_id, f"bot_{message_id}", response_msg, redis_client
            )
            sent_message_id = send_msg(
                chat_id,
                response_msg,
                message_id,
                reply_markup=response_markup,
            )
            if sent_message_id is not None:
                metadata: Dict[str, Any]
                if response_command:
                    metadata = {
                        "type": "command",
                        "command": response_command,
                        "uses_ai": response_uses_ai,
                    }
                else:
                    metadata = {"type": "ai"}
                save_bot_message_metadata(
                    redis_client, chat_id, sent_message_id, metadata
                )

        return "ok"

    except Exception as e:
        error_context = {
            "message_id": message.get("message_id"),
            "chat_id": message.get("chat", {}).get("id"),
            "user": message.get("from", {}).get("username", "Unknown"),
        }

        error_msg = f"Message handling error: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return "Error processing message"


def handle_rate_limit(chat_id: str, message: Dict) -> str:
    """Handle rate limited responses"""
    token = environ.get("TELEGRAM_TOKEN")
    if token:
        send_typing(token, chat_id)
    time.sleep(random.uniform(0, 1))
    return gen_random(message["from"]["first_name"])


GORDO_PREFIX_PATTERN = re.compile(r"^\s*gordo\b\s*:\s*", re.IGNORECASE)


def remove_gordo_prefix(text: Optional[str]) -> str:
    """Strip leading 'gordo:' persona prefix from each line of a response."""
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        cleaned_lines.append(GORDO_PREFIX_PATTERN.sub("", line, count=1))

    return "\n".join(cleaned_lines).strip()


def clean_duplicate_response(response: str) -> str:
    """Remove duplicate consecutive text in AI responses"""
    if not response:
        return response

    # Split by lines and remove consecutive duplicates
    lines = response.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and (not cleaned_lines or line != cleaned_lines[-1]):
            cleaned_lines.append(line)

    cleaned_response = "\n".join(cleaned_lines)

    # Also check for repeated sentences within the same line
    sentences = cleaned_response.split(". ")
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and (not cleaned_sentences or sentence != cleaned_sentences[-1]):
            cleaned_sentences.append(sentence)

    final_response = ". ".join(cleaned_sentences)

    # Clean up any double periods
    final_response = final_response.replace("..", ".")

    return final_response


def _strip_leading_context(
    response: str, contexts: Optional[Sequence[Optional[str]]]
) -> str:
    """Remove leading echoes of known context strings from an AI response."""

    if not response or not contexts:
        return response

    trimmed_response = response
    normalized_contexts = [
        str(context).strip()
        for context in contexts
        if context is not None and str(context).strip()
    ]

    if not normalized_contexts:
        return trimmed_response

    changed = True
    max_passes = len(normalized_contexts)
    passes = 0

    while changed and passes < max_passes:
        changed = False
        passes += 1
        for context in normalized_contexts:
            context_lower = context.lower()
            if trimmed_response.lower().startswith(context_lower):
                trimmed_response = trimmed_response[len(context) :].lstrip(" \t:-\n")
                changed = True
                break

    return trimmed_response


def _strip_user_identity_prefix(response: str, user_identity: Optional[str]) -> str:
    """Remove a leading '<user>:' prefix that sometimes leaks into completions."""

    if not response or not user_identity:
        return response

    normalized_identity = str(user_identity).strip()
    if not normalized_identity:
        return response

    pattern = re.compile(
        rf"^\s*{re.escape(normalized_identity)}\s*:\s*", flags=re.IGNORECASE
    )
    return pattern.sub("", response, count=1).lstrip()


def handle_ai_response(
    chat_id: str,
    handler_func: Callable,
    messages: List[Dict],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    context_texts: Optional[Sequence[Optional[str]]] = None,
    user_identity: Optional[str] = None,
) -> str:
    """Handle AI API responses"""
    token = environ.get("TELEGRAM_TOKEN")
    if token:
        send_typing(token, chat_id)
    time.sleep(random.uniform(0, 1))

    # Call handler with image if supported
    if (
        image_data
        and hasattr(handler_func, "__name__")
        and handler_func.__name__ == "ask_ai"
    ):
        print("handle_ai_response: calling ask_ai with image context")
        response = handler_func(
            messages, image_data=image_data, image_file_id=image_file_id
        )
    else:
        print(
            f"handle_ai_response: calling {getattr(handler_func,'__name__','<callable>')} (text-only)"
        )
        response = handler_func(messages)

    # Remove any internal tool call lines before further processing
    sanitized_response = sanitize_tool_artifacts(response)

    # Strip persona prefixes that sometimes leak into completions
    persona_stripped_response = remove_gordo_prefix(sanitized_response)

    # Remove echoes of the original user/context text
    context_stripped_response = _strip_leading_context(
        persona_stripped_response, context_texts
    )

    # Remove leading user prefixes like "nombre (alias):"
    prefix_stripped_response = _strip_user_identity_prefix(
        context_stripped_response, user_identity
    )

    # Clean any duplicate text
    cleaned_response = clean_duplicate_response(prefix_stripped_response)
    try:
        print(
            f"handle_ai_response: response len={len(cleaned_response)} preview='{cleaned_response[:160].replace('\n',' ')}'"
        )
    except Exception:
        pass

    if not cleaned_response.strip():
        print("handle_ai_response: empty sanitized response")
        return "no pude generar respuesta, intentá de nuevo"

    return cleaned_response


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
        "allowed_updates": '["message","callback_query"]',
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
        return ("Webhook checked", 200) if webhook_verified else ("Webhook check error", 400)

    if _arg_is_true(args, "update_webhook"):
        function_url = environ.get("FUNCTION_URL")
        if not function_url:
            return "Webhook update error", 400
        updated = set_telegram_webhook(function_url)
        return ("Webhook updated", 200) if updated else ("Webhook update error", 400)

    return None


def _handle_control_actions(args: Mapping[str, Any]) -> Optional[Tuple[str, int]]:
    if _arg_is_true(args, "update_dollars"):
        get_dollar_rates()
        return "Dollars updated", 200

    if _arg_is_true(args, "run_agent"):
        try:
            thought_result = run_agent_cycle()
            payload = json.dumps({"status": "ok", "thought": thought_result}, ensure_ascii=False)
            return payload, 200
        except Exception as agent_error:
            admin_report("Agent run failed", agent_error)
            return "Agent run failed", 500

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
            admin_report("Wrong secret token")
            return "Wrong secret token", 400

        # Process message
        request_json = request.get_json(silent=True)
        if not request_json:
            return "Invalid JSON", 400
        callback_query = request_json.get("callback_query")
        if callback_query:
            handle_callback_query(callback_query)
            return "Ok", 200

        message = request_json.get("message")
        if not message:
            return "No message", 200

        handle_msg(message)
        return "Ok", 200

    except Exception as e:
        error_context = {
            "request_method": request.method,
            "request_args": dict(request.args),
            "request_path": request.path,
        }

        error_msg = f"Request processing error: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return "Error processing request", 500


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def responder() -> Tuple[str, int]:
    try:
        webhook_key = request.args.get("key")
        if not webhook_key:
            return "No key", 200

        if webhook_key != environ.get("WEBHOOK_AUTH_KEY"):
            admin_report("Wrong key attempt")
            return "Wrong key", 400

        response_message, status_code = process_request_parameters(request)
        return response_message, status_code
    except Exception as e:
        error_context = {
            "request_method": request.method,
            "request_args": dict(request.args),
            "request_path": request.path,
        }

        error_msg = "Critical error in responder"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        return "Critical error", 500
