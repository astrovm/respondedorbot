from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone, date
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from html import unescape
from math import log
from openai import OpenAI
from groq import Groq as GroqClient
from os import environ
from PIL import Image
from requests.exceptions import RequestException
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
    Literal,
)
import base64
import concurrent.futures
import emoji
import hashlib
import io
import json
import random
import re
import redis
import requests
import subprocess
import tempfile
import time
import traceback
import wave
from api.provider_backoff import (
    mark_provider_cooldown,
    get_provider_cooldown_remaining as _get_cooldown_remaining,
    is_provider_cooled_down,
)
from pykakasi import kakasi
from mutagen import File as MutagenFile
import unicodedata
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputParam
else:
    ResponseInputParam = Any  # type: ignore[assignment]

from api.utils import (
    fmt_num,
    fmt_signed_pct,
    local_cache_get,
    update_local_cache,
)
from api.services.redis_helpers import (
    redis_get_json,
    redis_set_json,
    redis_setex_json,
)
from api.services.maintenance import (
    GIPHY_STALE_TTL,
    REQUEST_CACHE_HISTORY_TTL,
    request_cache_history_key,
    request_cache_key,
    request_cache_ttl,
)
from api.config import (
    config_redis as _config_config_redis,
    configure as configure_app_config,
    load_bot_config as _config_load_bot_config,
)
from api.ai_billing import (
    BalanceFormatter,
    build_insufficient_credits_message as _billing_build_insufficient_credits_message,
    build_topup_keyboard as _billing_build_topup_keyboard,
    extract_numeric_chat_id as _billing_extract_numeric_chat_id,
    extract_user_id as _billing_extract_user_id,
    get_ai_billing_pack as _billing_get_ai_billing_pack,
    get_ai_billing_packs as _billing_get_ai_billing_packs,
    get_ai_onboarding_credits as _billing_get_ai_onboarding_credits,
    maybe_grant_onboarding_credits as _billing_maybe_grant_onboarding_credits,
    parse_topup_payload as _billing_parse_topup_payload,
)
from api.ai_pricing import (
    CHAT_OUTPUT_TOKEN_LIMIT,
    AIUsageResult,
    VISION_OUTPUT_TOKEN_LIMIT,
    calculate_billing_for_segments,
    credit_units_from_usd_micros,
    estimate_chat_reserve_credits,
    estimate_message_tokens,
    estimate_vision_reserve_credits,
    ensure_mapping,
    MODEL_PRICING_USD_MICROS,
)
from api.agent_tools import fetch_url_content, normalize_http_url
from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
import api.tools.crypto_prices
import api.tools.stock_prices
import api.tools.calculate
import api.tools.web_fetch
import api.tools.task_set
import api.tools.task_list
import api.tools.task_cancel
from api.tools import get_all_tool_schemas
from api.tool_runtime import ToolRuntime
from api.tools.task_scheduler import (
    list_tasks as _task_list_tasks,
    cancel_task as _task_cancel_task,
    format_task_summary,
)
from api.ai_pipeline import (
    clean_duplicate_response as _ai_clean_duplicate_response,
    handle_ai_response as _ai_handle_response,
    remove_gordo_prefix as _ai_remove_gordo_prefix,
)
from api.chat_settings import (
    TIMEZONE_OFFSET_MAX,
    TIMEZONE_OFFSET_MIN,
    build_config_keyboard,
    build_config_text,
    coerce_bool,
    decode_redis_value,
    get_chat_config as _chat_get_chat_config,
    is_chat_admin as _chat_is_chat_admin,
    is_group_chat_type,
    report_unauthorized_config_attempt as _chat_report_unauthorized_config_attempt,
    set_chat_config as _chat_set_chat_config,
)
from api.credit_units import format_credit_units
from api.command_registry import (
    COMMAND_GROUPS,
    build_command_registry as _build_command_registry,
    parse_command as _command_parse_command,
    should_auto_process_media as _command_should_auto_process_media,
    should_gordo_respond as _command_should_gordo_respond,
)
from api.message_handler import (
    MessageAIDeps,
    MessageChatDeps,
    MessageHandlerDeps,
    MessageIODeps,
    MessageMediaDeps,
    MessageRoutingDeps,
    MessageStateDeps,
    build_message_handler_deps,
    handle_msg as _handle_msg_impl,
)
from api.ai_service import build_ai_service
from api.routing_policy import RoutingPolicy
from api.telegram_gateway import TelegramGateway
from api.telegram_bot_commands import update_bot_commands as _update_bot_commands
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
from api.random_replies import build_random_reply
from api.services import bcra as bcra_service
from api.services import chat_config_db as chat_config_db_service
from api.services import credits_db as credits_db_service
from api.utils.http import request_with_ssl_fallback
from api.utils.links import (
    can_embed_url as _links_can_embed_url,
    is_social_frontend as _links_is_social_frontend,
    replace_links as _links_replace_links,
    fetch_tweet_text as _links_fetch_tweet_text,
)
from api.utils.youtube_transcript import (
    extract_youtube_video_id,
    get_youtube_transcript_context,
)

# TTL constants (seconds)
TTL_PRICE = 300  # 5 minutes
TTL_DOLLAR = 300  # 5 minutes
TTL_WEATHER = 1800  # 30 minutes
TTL_LINK_METADATA = 300  # 5 minutes
TTL_POLYMARKET = 5  # 5 seconds
TTL_POLYMARKET_STREAM = 5  # 5 seconds for live price lookups
LINK_METADATA_MAX_BYTES = 64_000
MAX_LINKS_IN_MESSAGE = 3
TTL_MEDIA_CACHE = 7 * 24 * 60 * 60  # 7 days
TTL_HACKER_NEWS = 600  # 10 minutes

# Timeframe support for /prices (maps to CMC native fields)
_CMC_CHANGE_FIELD: Dict[str, str] = {
    "1h": "percent_change_1h",
    "24h": "percent_change_24h",
    "7d": "percent_change_7d",
    "30d": "percent_change_30d",
}

# Timeframe support for /usd (maps to Redis hourly snapshot lookback)
_DOLLAR_TIMEFRAME_HOURS: Dict[str, int] = {
    "1h": 1,
    "6h": 6,
    "12h": 12,
    "24h": 24,
    "48h": 48,
}


def _parse_timeframe(msg_text: str, valid: Mapping) -> Tuple[str, Optional[str]]:
    """Strip a trailing timeframe token from msg_text.

    Returns (cleaned_text, tf_key) where tf_key is None if no timeframe found.
    """
    parts = msg_text.strip().rsplit(None, 1)
    if parts and parts[-1].lower() in valid:
        tf = parts[-1].lower()
        remaining = parts[0].strip() if len(parts) > 1 else ""
        return remaining, tf
    return msg_text.strip(), None


BA_TZ = timezone(timedelta(hours=-3))
PRIMARY_CHAT_MODEL = "qwen/qwen3.6-plus"
SUMMARY_MODEL = "google/gemini-2.5-flash-lite"
SUMMARY_FALLBACK_MODEL = "minimax/minimax-m2.5:free"
GROQ_VISION_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TRANSCRIBE_MODEL = "groq/whisper-large-v3"
AI_FALLBACK_MARKER = "[[AI_FALLBACK]]"
OPENROUTER_WEB_SEARCH_MAX_RESULTS = 10
OPENROUTER_WEB_SEARCH_MAX_QUERIES = 3
OPENROUTER_VISION_MODEL_MAP = {
    GROQ_VISION_MODEL: "meta-llama/llama-4-scout",
}


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


MESSAGE_BLOCK_PATTERN = re.compile(
    r"(?ms)^MENSAJE:\n(?P<message>.*?)(?:\n\nINSTRUCCIONES:|\Z)"
)
MESSAGE_URL_PATTERN = re.compile(
    r"(?i)\b("
    r"(?:https?://|www\.)[^\s<>()]+"
    r"|"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s<>()]*)?"
    r")"
)


def _extract_message_block_from_prompt(text: str) -> str:
    """Extract the message block from the AI prompt wrapper."""

    raw_text = str(text or "")
    if not raw_text:
        return ""

    match = MESSAGE_BLOCK_PATTERN.search(raw_text)
    if not match:
        return raw_text.strip()

    return str(match.group("message") or "").strip()


def _fetch_polymarket_live_price(
    token_id: str,
) -> Optional[Tuple[float, Optional[int]]]:
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


def _fetch_criptoya_dollar_data(
    *, hourly_cache: bool = True, get_history: int = 0
) -> Optional[Dict[str, Any]]:
    """Retrieve dollar rates from CriptoYa with shared cache semantics."""

    return cached_requests(
        "https://criptoya.com/api/dolar",
        None,
        None,
        TTL_DOLLAR,
        hourly_cache,
        get_history,
    )


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


def can_embed_url(url: str) -> bool:
    """Wrapper to allow tests to monkeypatch embed detection."""

    def _metadata_sink(metadata: Dict[str, Any]) -> None:
        _cache_link_metadata(url, metadata)

    return _links_can_embed_url(url, metadata_sink=_metadata_sink)


def is_social_frontend(host: str) -> bool:
    """Expose social frontend check while keeping implementation in utils."""

    return _links_is_social_frontend(host)


def replace_links(text: str) -> Tuple[str, bool, List[str]]:
    """Delegate to utils helper while keeping embed checker injectable in tests."""

    return _links_replace_links(text, embed_checker=can_embed_url)


def fetch_tweet_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Fetch tweet content via oEmbed API and return formatted summary."""
    return _links_fetch_tweet_text(text)


# Provider backoff windows (seconds)
GROQ_BACKOFF_DEFAULT_SECONDS = 60

GROQ_FREE_ACCOUNT = "free"
GROQ_PAID_ACCOUNT = "paid"
GROQ_ACCOUNT_ORDER: Tuple[str, ...] = (GROQ_FREE_ACCOUNT, GROQ_PAID_ACCOUNT)

_ai_provider_request_count: ContextVar[int] = ContextVar(
    "ai_provider_request_count", default=0
)
_link_metadata_local_cache: Dict[str, Dict[str, Any]] = {}


def _reset_ai_provider_request_count() -> Token:
    return _ai_provider_request_count.set(0)


def _restore_ai_provider_request_count(token: Token) -> None:
    _ai_provider_request_count.reset(token)


def _increment_ai_provider_request_count() -> None:
    _ai_provider_request_count.set(int(_ai_provider_request_count.get() or 0) + 1)


def _get_ai_provider_request_count() -> int:
    return int(_ai_provider_request_count.get() or 0)


def _set_provider_backoff(provider: str, duration: Optional[int]) -> None:
    if not provider:
        return

    duration = max(0, int(duration or 0))
    if duration == 0:
        return

    mark_provider_cooldown(provider.lower(), float(duration))


def get_provider_backoff_remaining(provider: str) -> float:
    if not provider:
        return 0.0

    return _get_cooldown_remaining(provider.lower())


def is_provider_backoff_active(provider: str) -> bool:
    return is_provider_cooled_down(provider.lower()) if provider else False


HACKER_NEWS_RSS_URL = "https://hnrss.org/best"
HACKER_NEWS_CACHE_KEY = "context:hacker_news:best"
HACKER_NEWS_MAX_ITEMS = 5


def _get_groq_api_key(account: str) -> Optional[str]:
    env_var = "GROQ_FREE_API_KEY" if account == GROQ_FREE_ACCOUNT else "GROQ_API_KEY"
    value = environ.get(env_var)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _get_configured_groq_accounts() -> List[str]:
    return [account for account in GROQ_ACCOUNT_ORDER if _get_groq_api_key(account)]


def _get_openrouter_vision_model(model: str) -> Optional[str]:
    return OPENROUTER_VISION_MODEL_MAP.get(model)


def _get_openrouter_api_key() -> Optional[str]:
    value = environ.get("OPENROUTER_API_KEY")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _get_openrouter_base_url() -> Optional[str]:
    value = environ.get("CF_AIG_BASE_URL")
    if value is None:
        return "https://openrouter.ai/api/v1"

    value = str(value).strip()
    if not value:
        return "https://openrouter.ai/api/v1"
    if "gateway.ai.cloudflare.com" not in value:
        return "https://openrouter.ai/api/v1"

    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if not path:
        return "https://openrouter.ai/api/v1"

    base_path = path.rsplit("/", 1)[0]
    openrouter_path = f"{base_path}/openrouter" if base_path else "/openrouter"
    return urlunparse(parsed._replace(path=openrouter_path))


def _get_openrouter_client(
    *, default_headers: Optional[Mapping[str, str]] = None
) -> Optional[OpenAI]:
    openrouter_api_key = _get_openrouter_api_key()
    openrouter_base_url = _get_openrouter_base_url()
    if not openrouter_api_key or not openrouter_base_url:
        return None

    headers: Dict[str, str] = dict(default_headers) if default_headers else {}
    cf_aig_token = environ.get("CF_AIG_TOKEN")
    if cf_aig_token:
        headers["cf-aig-authorization"] = f"Bearer {cf_aig_token}"

    client_kwargs: Dict[str, Any] = {
        "api_key": openrouter_api_key,
        "base_url": openrouter_base_url,
    }
    if headers:
        client_kwargs["default_headers"] = headers
    return OpenAI(**client_kwargs)


def _build_openrouter_web_search_tool() -> Dict[str, Any]:
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "firecrawl",
            "max_results": OPENROUTER_WEB_SEARCH_MAX_RESULTS,
            "max_total_results": OPENROUTER_WEB_SEARCH_MAX_RESULTS
            * OPENROUTER_WEB_SEARCH_MAX_QUERIES,
        },
    }


def _fetch_urls_from_latest_message(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    latest_text = ""
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            latest_text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            latest_text = " ".join(parts).strip()
        break

    if not latest_text:
        return ""

    max_fetches = MAX_LINKS_IN_MESSAGE
    urls: List[str] = []
    seen: Set[str] = set()
    for match in MESSAGE_URL_PATTERN.finditer(latest_text):
        normalized = _normalize_detected_message_url(match.group(1))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= max_fetches:
            break

    if not urls:
        return ""

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(urls), 5)
    ) as executor:
        results = list(executor.map(fetch_url_content, urls))

    parts = []
    for url, result in zip(urls, results):
        error = str(result.get("error") or "").strip()
        if error:
            parts.append(f"URL: {url}\nerror: {error}")
            continue
        title = str(result.get("title") or "").strip()
        content = str(result.get("content") or "").strip()
        lines = [f"URL: {url}"]
        if title:
            lines.append(f"titulo: {title}")
        if content:
            lines.append(f"contenido: {content}")
        parts.append("\n".join(lines))

    if not parts:
        return ""
    return "CONTENIDO DE URLs OBTENIDO:\n\n" + "\n\n---\n\n".join(parts)


def _get_groq_accounts_for_scope() -> List[str]:
    return _get_configured_groq_accounts()


def _get_groq_backoff_key(account: str, scope: str) -> str:
    return f"groq:{account}:{scope}".lower()


def _extract_error_headers(error: Exception) -> Dict[str, str]:
    possible_headers = getattr(error, "headers", None)
    response = getattr(error, "response", None)
    if not possible_headers and response is not None:
        possible_headers = getattr(response, "headers", None)
    if not possible_headers:
        return {}
    headers: Dict[str, str] = {}
    if isinstance(possible_headers, Mapping):
        for key, value in possible_headers.items():
            headers[str(key).lower()] = str(value)
        return headers
    try:
        for key, value in dict(possible_headers).items():
            headers[str(key).lower()] = str(value)
    except Exception:
        return {}
    return headers


def _parse_retry_window_seconds(value: Optional[str]) -> Optional[int]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        numeric = float(raw_value)
        return max(0, int(numeric))
    except (TypeError, ValueError):
        pass

    normalized = raw_value.lower()
    match = re.fullmatch(r"(?P<amount>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)?", normalized)
    if match:
        amount = float(match.group("amount"))
        unit = match.group("unit") or "s"
        multiplier = {
            "ms": 0.001,
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
        }.get(unit, 1.0)
        return max(0, int(amount * multiplier))

    try:
        parsed_dt = parsedate_to_datetime(raw_value)
    except Exception:
        return None
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    return max(0, int((parsed_dt - datetime.now(timezone.utc)).total_seconds()))


def _extract_rate_limit_backoff_seconds(
    error: Exception,
    fallback_seconds: Optional[int] = None,
) -> Optional[int]:
    headers = _extract_error_headers(error)
    for header_name in (
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ):
        parsed = _parse_retry_window_seconds(headers.get(header_name))
        if parsed is not None:
            return parsed
    return fallback_seconds


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
    return _get_cached_media("audio_transcription", file_id)


def cache_transcription(file_id: str, text: str, ttl: int = TTL_MEDIA_CACHE) -> None:
    _cache_media("audio_transcription", file_id, text, ttl)


def get_cached_description(file_id: str) -> Optional[str]:
    return _get_cached_media("image_description", file_id)


def cache_description(
    file_id: str, description: str, ttl: int = TTL_MEDIA_CACHE
) -> None:
    _cache_media("image_description", file_id, description, ttl)


# get cached data from previous hour
def get_cache_history(hours_ago, request_hash, redis_client):
    timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d-%H")
    cached_data = redis_client.get(request_cache_history_key(timestamp, request_hash))
    if cached_data is None:
        return None
    cache_history = json.loads(cached_data)
    return (
        cache_history
        if cache_history is not None and "timestamp" in cache_history
        else None
    )


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
        redis_response = redis_get_json(redis_client, request_cache_key(request_hash))
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
                    redis_set_json(
                        redis_client,
                        request_cache_key(request_hash),
                        redis_value,
                        ttl=request_cache_ttl(expiration_time),
                    )
                    if hourly_cache:
                        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
                        hourly_key = request_cache_history_key(
                            current_hour, request_hash
                        )
                        if redis_client.get(hourly_key) is None:
                            redis_set_json(
                                redis_client,
                                hourly_key,
                                redis_value,
                                ttl=REQUEST_CACHE_HISTORY_TTL,
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


def get_api_or_cache_prices(
    convert_to: str, limit: Optional[int] = None, hourly_cache: bool = False
):
    api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {"start": "1", "limit": "100", "convert": convert_to}
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": environ.get("COINMARKETCAP_KEY"),
    }

    if isinstance(limit, int) and limit > 0:
        parameters["limit"] = str(limit)

    response = cached_requests(api_url, parameters, headers, TTL_PRICE, hourly_cache)

    return response["data"] if response else None


def _fetch_cmc_quotes(
    identifiers: List[str],
    convert_to: str = "USD",
    by_slug: bool = False,
) -> Optional[Dict[str, Any]]:
    api_url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    param_key = "slug" if by_slug else "symbol"
    parameters = {param_key: ",".join(identifiers), "convert": convert_to}
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": environ.get("COINMARKETCAP_KEY"),
    }
    response = cached_requests(api_url, parameters, headers, TTL_PRICE)
    return response["data"] if response else None


def refresh_price_caches() -> None:
    """Refresh all price caches and store hourly snapshots for change calculations."""
    _fetch_criptoya_dollar_data(hourly_cache=True)
    get_api_or_cache_prices("ARS", hourly_cache=True)
    get_api_or_cache_prices("USD", hourly_cache=True)
    try:
        get_oil_price()
    except Exception:
        pass


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
            market.get("groupItemTitle") or market.get("question") or market.get("slug")
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
        if any(
            normalized_title.startswith(prefix.upper()) for prefix in filter_prefixes
        ):
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


def get_prices(msg_text: str) -> Optional[str]:
    msg_text, tf = _parse_timeframe(msg_text, _CMC_CHANGE_FIELD)
    if tf is None and msg_text.strip():
        last_token = msg_text.strip().rsplit(None, 1)[-1].lower()
        if re.fullmatch(r"\d+[hd]", last_token):
            valid = ", ".join(_CMC_CHANGE_FIELD)
            return f"timeframe '{last_token}' no soportado, uso: {valid}"
    cmc_change_field = _CMC_CHANGE_FIELD.get(tf or "24h", "percent_change_24h")
    tf_label = tf or "24h"

    prices_number = 0
    convert_to = "USD"
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

            asset_price_in_source = requested_target_asset["quote"][source_parameter][
                "price"
            ]
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
            found = []
            not_found = []
            for coin_token in coins:
                token = coin_token.upper().replace(" ", "")
                quote_data = _fetch_cmc_quotes([token], convert_to_parameter)
                if quote_data:
                    for cid, cdata in quote_data.items():
                        price = (
                            cdata.get("quote", {})
                            .get(convert_to_parameter, {})
                            .get("price")
                        )
                        if price:
                            found.append(cdata)
                            break
                    else:
                        quote_by_slug = _fetch_cmc_quotes(
                            [token.lower()], convert_to_parameter, by_slug=True
                        )
                        if quote_by_slug:
                            for cid, cdata in quote_by_slug.items():
                                price = (
                                    cdata.get("quote", {})
                                    .get(convert_to_parameter, {})
                                    .get("price")
                                )
                                if price:
                                    found.append(cdata)
                                    break
                            else:
                                not_found.append(token)
                        else:
                            not_found.append(token)
                else:
                    not_found.append(token)

            if not found and not_found:
                return f"no encontre esos ponzis: {', '.join(not_found)}"

            if not found:
                return "no pude traer precios de crypto boludo"

            new_prices = found

        prices_number = len(new_prices)
        prices["data"] = new_prices

    if prices_number < 1:
        prices_number = 10

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
        price = f"{coin['quote'][convert_to_parameter]['price']:.{zeros + 4}f}".rstrip(
            "0"
        ).rstrip(".")
        percentage = f"{coin['quote'][convert_to_parameter].get(cmc_change_field, 0):+.2f}".rstrip(
            "0"
        ).rstrip(".")
        line = f"{ticker}: {price} {convert_to} ({percentage}% {tf_label})"

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
    dollar_rates,
    tcrm_100: Optional[float] = None,
    tcrm_history: Optional[float] = None,
    hours_ago: int = 24,
):
    dollars = dollar_rates["data"]

    # For 24h use CriptoYa's own variation (accurate, always present).
    # For other timeframes use Redis hourly snapshots when available.
    hist_dollars: Optional[Dict[str, Any]] = None
    if hours_ago != 24:
        history_entry = dollar_rates.get("history")
        if isinstance(history_entry, dict) and "data" in history_entry:
            hist_dollars = history_entry["data"]

    def _var(
        current_price: float, criptoya_variation, *hist_path: str
    ) -> Optional[float]:
        if hours_ago == 24:
            return criptoya_variation
        if hist_dollars is None:
            return None
        try:
            obj: Any = hist_dollars
            for key in hist_path:
                obj = obj[key]
            return _pct_change(current_price, float(obj))
        except (KeyError, TypeError, ValueError):
            return None

    sorted_dollar_rates = [
        {
            "name": "Mayorista",
            "price": dollars["mayorista"]["price"],
            "history": _var(
                dollars["mayorista"]["price"],
                dollars["mayorista"]["variation"],
                "mayorista",
                "price",
            ),
        },
        {
            "name": "Oficial",
            "price": dollars["oficial"]["price"],
            "history": _var(
                dollars["oficial"]["price"],
                dollars["oficial"]["variation"],
                "oficial",
                "price",
            ),
        },
        {
            "name": "Tarjeta",
            "price": dollars["tarjeta"]["price"],
            "history": _var(
                dollars["tarjeta"]["price"],
                dollars["tarjeta"]["variation"],
                "tarjeta",
                "price",
            ),
        },
        {
            "name": "MEP",
            "price": dollars["mep"]["al30"]["ci"]["price"],
            "history": _var(
                dollars["mep"]["al30"]["ci"]["price"],
                dollars["mep"]["al30"]["ci"]["variation"],
                "mep",
                "al30",
                "ci",
                "price",
            ),
        },
        {
            "name": "CCL",
            "price": dollars["ccl"]["al30"]["ci"]["price"],
            "history": _var(
                dollars["ccl"]["al30"]["ci"]["price"],
                dollars["ccl"]["al30"]["ci"]["variation"],
                "ccl",
                "al30",
                "ci",
                "price",
            ),
        },
        {
            "name": "Blue",
            "price": dollars["blue"]["ask"],
            "history": _var(
                dollars["blue"]["ask"], dollars["blue"]["variation"], "blue", "ask"
            ),
        },
        {
            "name": "Bitcoin",
            "price": dollars["cripto"]["ccb"]["ask"],
            "history": _var(
                dollars["cripto"]["ccb"]["ask"],
                dollars["cripto"]["ccb"]["variation"],
                "cripto",
                "ccb",
                "ask",
            ),
        },
        {
            "name": "USDC",
            "price": dollars["cripto"]["usdc"]["ask"],
            "history": _var(
                dollars["cripto"]["usdc"]["ask"],
                dollars["cripto"]["usdc"]["variation"],
                "cripto",
                "usdc",
                "ask",
            ),
        },
        {
            "name": "USDT",
            "price": dollars["cripto"]["usdt"]["ask"],
            "history": _var(
                dollars["cripto"]["usdt"]["ask"],
                dollars["cripto"]["usdt"]["variation"],
                "cripto",
                "usdt",
                "ask",
            ),
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

    if hours_ago != 24 and all(r.get("history") is None for r in rates):
        msg_lines.append(f"\n(sin datos historicos para {hours_ago}hs todavia)")

    return "\n".join(msg_lines)


def get_dollar_rates(msg_text: str = "") -> Optional[str]:
    _, tf = _parse_timeframe(msg_text, _DOLLAR_TIMEFRAME_HOURS)
    if tf is None and msg_text.strip():
        token = msg_text.strip().lower()
        if re.fullmatch(r"\d+[hd]", token):
            valid = ", ".join(_DOLLAR_TIMEFRAME_HOURS)
            return f"timeframe '{token}' no soportado, uso: {valid}"
    hours_ago = _DOLLAR_TIMEFRAME_HOURS.get(tf, 24) if tf else 24

    dollars = _fetch_criptoya_dollar_data(
        hourly_cache=True,
        get_history=hours_ago if hours_ago != 24 else 0,
    )

    tcrm_100, tcrm_history = get_cached_tcrm_100(hours_ago)

    sorted_dollar_rates = sort_dollar_rates(
        dollars, tcrm_100, tcrm_history, hours_ago=hours_ago
    )

    band_limits = get_currency_band_limits()

    # Band deltas are daily-only values from the BCRA API - hide them for non-24h
    if band_limits and hours_ago != 24:
        band_limits = {
            k: v for k, v in band_limits.items() if not k.endswith("_change_pct")
        }

    return format_dollar_rates(sorted_dollar_rates, hours_ago, band_limits)


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


def _pct_change(current: float, historical: float) -> Optional[float]:
    """Compute percentage change from historical to current. Returns None on bad input."""
    try:
        h = float(historical)
        if h != 0:
            return ((float(current) - h) / h) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        pass
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
        f"{int(usd_amount)}"
        if isinstance(usd_amount, float) and usd_amount.is_integer()
        else str(usd_amount)
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
    blue_price = _safe_float(blue_data.get("bid")) or _safe_float(
        blue_data.get("price")
    )
    if blue_price:
        blue_final_ars = blue_price * usd_amount
        blue_profit_ars = blue_final_ars - oficial_cost_ars
        blue_extra = [
            f"Resultado: {base_usd} USD → {_format_local_currency(blue_final_ars)} ARS",
            f"Ganancia: {_format_local_signed(blue_profit_ars)} ARS",
        ]
        lines.append(_format_spread_line("Blue", blue_price, oficial_price, blue_extra))

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

        sat_value_usd = btc_price_usd / 100_000_000
        sat_value_ars = btc_price_ars / 100_000_000
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
    try:
        variables = get_or_refresh_bcra_variables()

        if not variables:
            return "No pude obtener las variables del BCRA en este momento, probá más tarde"
        return format_bcra_variables(variables)

    except Exception as e:
        print(f"Error handling BCRA variables: {e}")
        return "error al obtener las variables del BCRA"


_DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = {
    "download": "no pude bajar el audio, mandalo de nuevo",
    "duration": "no pude medir la duración del audio",
    "transcribe": "no pude sacar nada de ese audio, probá más tarde",
}


def _transcribe_audio_file(
    file_id: str, *, use_cache: bool
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
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
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    media = replied_msg.get(media_key)
    if not media:
        return None, None

    file_id = extract_file_id(media)
    if not file_id:
        return None, None

    description, error_code, billing_segment = describe_media_by_id(file_id, prompt)
    if description:
        return f"{success_prefix}{description}", billing_segment

    if error_code == "download":
        return download_error, None

    return describe_error, None


def handle_transcribe_with_message_result(
    message: Dict,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio or describe image from replied message with billing metadata."""
    try:
        # Check if this is a reply to another message
        if "reply_to_message" not in message:
            return (
                "respondeme un audio, video, imagen o sticker y te digo qué carajo hay ahí",
                [],
            )

        replied_msg = message["reply_to_message"]

        _, photo_file_id, audio_file_id = extract_message_content(replied_msg)

        if audio_file_id:
            text, error_code, billing_segment = _transcribe_audio_file(
                audio_file_id, use_cache=True
            )
            if text:
                return (
                    f"🎵 te saqué esto del audio: {text}",
                    [billing_segment] if billing_segment else [],
                )
            error_message = _transcription_error_message(error_code)
            if error_message:
                return error_message, []
            return _DEFAULT_TRANSCRIPTION_ERROR_MESSAGES["transcribe"], []

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
                photo_response, billing_segment = _describe_replied_media(
                    photo_source,
                    media_key="photo",
                    extract_file_id=lambda media: (
                        media[-1]["file_id"]
                        if isinstance(media, Sequence)
                        and not isinstance(media, (str, bytes))
                        and media
                        else None
                    ),
                    prompt="Describe what you see in this image in detail.",
                    success_prefix="🖼️ en la imagen veo: ",
                    download_error="no pude bajar la imagen, mandala de nuevo",
                    describe_error="no pude sacar qué mierda tiene la imagen, probá más tarde",
                )
                if photo_response:
                    return photo_response, [billing_segment] if billing_segment else []

            sticker_source = _find_media_message(replied_msg, "sticker")
            if sticker_source:
                sticker_response, billing_segment = _describe_replied_media(
                    sticker_source,
                    media_key="sticker",
                    extract_file_id=lambda media: (
                        media.get("file_id") if isinstance(media, Mapping) else None
                    ),
                    prompt="Describe what you see in this sticker in detail.",
                    success_prefix="🎨 en el sticker veo: ",
                    download_error="no pude bajar el sticker, mandalo de nuevo",
                    describe_error="no pude sacar qué carajo tiene el sticker, probá más tarde",
                )
                if sticker_response:
                    return sticker_response, (
                        [billing_segment] if billing_segment else []
                    )

        return "ese mensaje no tiene audio, video, imagen ni sticker para laburar", []

    except Exception as e:
        print(f"Error in handle_transcribe: {e}")
        return "se trabó el /transcribe, probá más tarde", []


def handle_transcribe_with_message(message: Dict) -> str:
    response_text, _billing_segments = handle_transcribe_with_message_result(message)
    return response_text


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
    return "".join(
        str(segment.get("hepburn") or segment.get("orig") or "") for segment in segments
    )


def is_japanese_text(text: str) -> bool:
    """Return True when the text includes Japanese scripts or CJK extensions."""
    return bool(JAPANESE_TEXT_RE.search(text))


def convert_to_command(msg_text: str) -> str:
    if not msg_text:
        return "y que queres que convierta boludo? mandate texto"

    emoji_text = emoji.demojize(msg_text, delimiters=("_", "_"), language="es")
    if is_japanese_text(emoji_text):
        romanized_text = romanize_japanese(emoji_text)
    else:
        romanized_text = emoji_text

    replaced_ni_text = re.sub(r"\bÑ\b", "ENIE", romanized_text.upper()).replace(
        "Ñ", "NI"
    )

    single_spaced_text = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFD", replaced_ni_text)
        .encode("ascii", "ignore")
        .decode("utf-8"),
    )

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

    cleaned_text = re.sub(
        r"^_+|_+$",
        "",
        re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"_+", "_", translated_punctuation)),
    )

    if not cleaned_text:
        return "no me mandes giladas boludo, tiene que tener letras o numeros"

    command = f"/{cleaned_text}"
    return command


def _build_tasks_message(
    tasks: List[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not tasks:
        return "no hay tareas", None

    lines = []
    rows: List[List[Dict[str, str]]] = []

    for t in tasks:
        lines.append(format_task_summary(t, prefix="• "))
        rows.append(
            [{"text": f"borrar {t['id']}", "callback_data": f"task:del:{t['id']}"}]
        )

    return "\n".join(lines), {"inline_keyboard": rows}


def tasks_command(chat_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not chat_id:
        return "no se en que chat estoy", None

    tasks = _task_list_tasks(chat_id)
    return _build_tasks_message(tasks)


def get_help() -> str:
    return """
esto es lo que sé hacer, boludo:

- /ask, /pregunta, /che, /gordo: te contesto cualquier gilada

- /prices, /precio, /precios, /presio, /presios, /bresio, /bresios, /brecio, /brecios: top 10 cryptos en usd
- /prices in btc: top 10 en btc
- /prices 20: top 20 en usd
- /prices 100 in eur: top 100 en eur
- /prices btc, eth, xmr: bitcoin, ethereum y monero en usd
- /prices dai in sats: dai en satoshis
- /prices stables: stablecoins en usd
- /prices btc 7d: variacion de 7 dias (acepta 1h, 24h, 7d, 30d)
- /prices 10 1h: top 10 con variacion de 1 hora

- /dolar, /dollar, /usd: te tiro la posta del blue y todos los dólares
- /usd 1h: variacion de la ultima hora (acepta 1h, 6h, 12h, 24h, 48h)

- /petroleo, /oil: te paso el precio del Brent y del WTI

- /bcra, /variables: te tiro las variables económicas del bcra

- /eleccion: odds actuales de Polymarket para Diputados 2025

- /devo 0.5, 100: te calculo el arbitraje entre tarjeta y crypto

- /rulo: te armo los rulos desde el oficial

- /powerlaw: te tiro el precio justo de btc según power law y si está caro o barato
- /rainbow: idem pero con el rainbow chart

- /satoshi, /sat, /sats: te digo cuanto vale un satoshi

- /random pizza, carne, sushi: elijo por vos
- /random 1-10: numero random del 1 al 10

- /convertbase 101, 2, 10: te paso números entre bases

- /comando, /command algo: te convierto eso en comando de telegram

- /time: timestamp unix actual

- /transcribe, /describe: te transcribo audio o describo imagen (responde a un mensaje)

- /gm: te mando un gif de buenos días random
- /gn: te mando un gif de buenas noches random

- /config: tocás la config del gordo y de los links

- /topup: cargás créditos ia con telegram stars por privado

- /balance: te muestro cuántos créditos ia te quedan

- /transfer 1.5: le pasás 1.5 créditos tuyos al grupo

- /tareas, /tasks: listado de tareas programadas con botones para borrar (el gordo agenda recordatorios y tareas recurrentes cuando le pedis)

- /resumen, /summary: resumí la conversación (opcional: /resumen 50 o /resumen focus en crypto)

- /acciones, /stocks: precios de acciones [aapl tsla googl]
"""


def _parse_stooq_quote(raw_response: str) -> Optional[Tuple[float, float]]:
    rows = [line.strip() for line in raw_response.splitlines() if line.strip()]
    if not rows:
        return None
    row = [field.strip() for field in rows[-1].split(",")]
    if len(row) < 7 or row[0].lower() == "symbol":
        return None
    open_price = row[3]
    close_price = row[6]
    if open_price in {"N/D", ""} or close_price in {"N/D", ""}:
        return None
    current_value = float(close_price)
    open_value = float(open_price)
    if open_value == 0:
        return None
    variation = ((current_value - open_value) / open_value) * 100
    return current_value, variation


def get_oil_price() -> str:
    """Return latest Brent and WTI oil prices in USD with daily variation."""

    def parse_daily_rows(rows: List[str]) -> Optional[Tuple[float, float]]:
        data_rows = rows
        if rows and rows[0].lower().startswith("date"):
            data_rows = rows[1:]
        if len(data_rows) < 2:
            return None

        latest_row = [field.strip() for field in data_rows[-1].split(",")]
        previous_row = [field.strip() for field in data_rows[-2].split(",")]
        if len(latest_row) < 5 or len(previous_row) < 5:
            return None

        close_price = latest_row[4]
        previous_close = previous_row[4]
        if close_price in {"N/D", ""} or previous_close in {"N/D", ""}:
            return None

        current_value = float(close_price)
        previous_value = float(previous_close)
        if previous_value == 0:
            return None

        variation = ((current_value - previous_value) / previous_value) * 100
        return current_value, variation

    symbols = {
        "Brent": "cb.f",
        "WTI": "cl.f",
    }
    prices: Dict[str, Dict[str, float]] = {}

    for name, symbol in symbols.items():
        try:
            parsed = None

            daily_response = requests.get(
                f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=5
            )
            daily_response.raise_for_status()
            rows = [
                line.strip()
                for line in daily_response.text.splitlines()
                if line.strip()
            ]
            parsed = parse_daily_rows(rows)

            if not parsed:
                parsed = _parse_stooq_quote(daily_response.text)

            if not parsed:
                quote_response = requests.get(
                    f"https://stooq.com/q/l/?s={symbol}&i=d", timeout=5
                )
                quote_response.raise_for_status()
                parsed = _parse_stooq_quote(quote_response.text)

            if not parsed:
                continue
            current_value, variation = parsed
            prices[name] = {"price": current_value, "variation": variation}
        except Exception:
            continue

    if not prices:
        return "no pude traer el precio del petróleo boludo"

    lines = []
    for name in ("Brent", "WTI"):
        if name not in prices:
            continue
        price = fmt_num(prices[name]["price"], 2)
        percentage = fmt_num(prices[name]["variation"], 2)
        sign = "+" if prices[name]["variation"] >= 0 else ""
        lines.append(f"{name}: {price} USD ({sign}{percentage}% 24hs)")

    return "\n".join(lines)


def get_stock_prices(msg_text: str) -> str:
    symbols = [s.strip() for s in str(msg_text or "").split() if s.strip()]
    if not symbols:
        return "pasame los simbolos, ejemplo: /acciones aapl tsla googl"

    lines: List[str] = []
    for sym in symbols:
        try:
            resp = requests.get(
                f"https://stooq.com/q/l/?s={sym.upper()}&i=d", timeout=5
            )
            resp.raise_for_status()
            parsed = _parse_stooq_quote(resp.text)
            if parsed:
                price, var = parsed
                sign = "+" if var >= 0 else ""
                lines.append(f"{sym.upper()}: ${price:.2f} ({sign}{var:.2f}% dia)")
            else:
                lines.append(f"{sym.upper()}: no se pudo obtener")
        except Exception:
            lines.append(f"{sym.upper()}: no se pudo obtener")

    return "\n".join(lines) if lines else "no se pudo obtener ninguna cotización"


def summarize_conversation(
    chat_id: str,
    redis_client: Any,
    num_messages: int = 50,
    custom_instruction: Optional[str] = None,
) -> str:
    from api.message_state import get_chat_history

    history = get_chat_history(chat_id, redis_client, max_messages=num_messages)
    if not history:
        return "no hay mensajes en el chat para resumir"

    formatted = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p["text"])
            content = " ".join(parts)
        if content:
            sender = "bot" if role == "assistant" else "user"
            formatted.append(f"{sender}: {content}")

    instruction = custom_instruction or "summarize this conversation, capturing the key points, topics discussed, and conclusions. be concise. respond in the same language as the conversation."
    messages = [
        {"role": "system", "content": "sos un asistente que resume conversaciones de telegram."},
        {"role": "user", "content": f"{instruction}\n\nConversación:\n" + "\n".join(formatted)},
    ]

    response, _ = _call_summary_model(messages)
    if not response:
        return "no pude resumir la conversación"
    return response


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
            print(f"Telegram request to {endpoint} returned unexpected payload type")
        return None, "unexpected response"

    if not payload.get("ok"):
        description = str(payload.get("description") or "telegram request failed")
        if log_errors:
            print(f"Telegram request to {endpoint} returned ok=false: {description}")
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
    return telegram_gateway.send_message(chat_id, msg, msg_id, buttons, reply_markup)


def delete_msg(chat_id: str, msg_id: str) -> None:
    """Delete a Telegram message"""
    telegram_gateway.delete_message(chat_id, msg_id)


def send_animation(
    chat_id: str,
    animation_url: str,
    msg_id: str = "",
    caption: str = "",
) -> Optional[int]:
    """Send an animation (GIF) to a Telegram chat"""
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "animation": animation_url,
    }
    if msg_id:
        payload["reply_to_message_id"] = msg_id
    if caption:
        payload["caption"] = caption

    payload_response, error = _telegram_request(
        "sendAnimation", method="POST", json_payload=payload
    )
    if error or not payload_response:
        return None

    result = payload_response.get("result")
    if isinstance(result, dict):
        message_id = result.get("message_id")
        if isinstance(message_id, int):
            return message_id

    return None


# Giphy API constants
GIPHY_API_URL = "https://api.giphy.com/v1/gifs"
TTL_GIPHY_POOL = 86400  # 24 hours

GIPHY_GM_TERMS = [
    "good morning",
    "buenos dias",
    "morning coffee",
    "rise and shine",
]
GIPHY_GN_TERMS = [
    "good night",
    "buenas noches",
    "sweet dreams",
    "go to sleep",
]


def _fetch_giphy_pool(category: str) -> List[str]:
    """Fetch a pool of GIF URLs from GIPHY for a category. Returns list of URLs."""
    api_key = environ.get("GIPHY_API_KEY")
    if not api_key:
        return []

    terms = GIPHY_GM_TERMS if category == "gm" else GIPHY_GN_TERMS
    urls: List[str] = []

    # One call per term to cover all search terms
    for term in terms:
        try:
            params = {
                "api_key": api_key,
                "q": term,
                "limit": 25,
                "offset": random.randint(0, 100),
                "rating": "g",
            }
            response = requests.get(f"{GIPHY_API_URL}/search", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            for gif in data.get("data", []):
                url = gif.get("images", {}).get("original", {}).get("url")
                if url:
                    urls.append(url)
        except Exception as e:
            print(f"Error fetching Giphy pool for {category}: {e}")

    return urls


def _get_giphy_pool(category: str) -> List[str]:
    """Return cached GIF pool, refreshing from API if expired or empty.
    Falls back to stale pool if refresh fails."""
    pool_key = f"giphy_pool:{category}"
    stale_key = f"giphy_pool_stale:{category}"
    redis_client = _optional_redis_client()

    if redis_client:
        try:
            cached = redis_get_json(redis_client, pool_key)
            if cached is not None:
                return cached
        except Exception as e:
            print(f"Error reading Giphy pool from cache: {e}")

    urls = _fetch_giphy_pool(category)

    if urls and redis_client:
        try:
            redis_client.setex(pool_key, TTL_GIPHY_POOL, json.dumps(urls))
            redis_client.setex(stale_key, GIPHY_STALE_TTL, json.dumps(urls))
        except Exception as e:
            print(f"Error caching Giphy pool: {e}")
    elif not urls and redis_client:
        try:
            stale = redis_get_json(redis_client, stale_key)
            if stale is not None:
                print(f"Giphy API failed, using stale pool for {category}")
                return stale
        except Exception as e:
            print(f"Error reading stale Giphy pool: {e}")

    return urls


def _get_random_gif(category: str) -> Optional[str]:
    pool = _get_giphy_pool(category)
    if not pool:
        return None
    return random.choice(pool)


def get_good_morning() -> str:
    gif_url = _get_random_gif("gm")
    if gif_url:
        return gif_url
    return "buen día boludo"


def get_good_night() -> str:
    gif_url = _get_random_gif("gn")
    if gif_url:
        return gif_url
    return "buenas noches boludo"


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

        def _format_context_value(key: str, value: Any) -> Any:
            key_name = str(key or "").lower()
            if "credit_units" not in key_name:
                return value
            if isinstance(value, bool):
                return value
            try:
                units = int(value)
            except (TypeError, ValueError):
                return value
            return f"{format_credit_units(units)} créditos ({units} unidades)"

        context_details = "\n\ncontexto adicional:"
        for key, value in extra_context.items():
            context_details += f"\n{key}: {_format_context_value(key, value)}"
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


def ask_ai(
    messages: List[Dict[str, Any]],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
    enable_web_search: bool = True,
    chat_id: Optional[str] = None,
    user_name: Optional[str] = None,
    user_id: Optional[int] = None,
    timezone_offset: int = -3,
    task_mode: bool = False,
) -> str:
    try:
        messages = list(messages or [])

        if len(messages) > 8:
            keep = 5
            dropped = messages[: -keep]
            dropped_text = _format_messages_for_summary(dropped)
            summary, summary_cost = _compact_conversation(dropped_text)
            if summary:
                messages = [
                    {"role": "system", "content": summary}
                ] + messages[-keep:]
                if summary_cost > 0:
                    _append_billing_segment(response_meta, _make_summary_result(summary_cost))
            else:
                messages = messages[-keep:]

        context_data = {
            "market": get_market_context(),
            "weather": get_weather_context(),
            "time": get_time_context(),
            "hacker_news": get_hacker_news_context(),
        }

        tool_context: Dict[str, Any] = {
            "get_prices": get_prices,
        }
        if chat_id:
            tool_context["chat_id"] = chat_id
        tool_context["timezone_offset"] = timezone_offset
        if user_name:
            tool_context["user_name"] = user_name
        if user_id is not None:
            tool_context["user_id"] = user_id

        extra_tools = get_all_tool_schemas(tool_context, task_mode=task_mode)
        system_message = build_system_message(
            context_data,
            tools_active=bool(extra_tools),
            tool_schemas=extra_tools,
            task_mode=task_mode,
        )

        if image_data:
            print("Processing image with Groq vision model...")

            user_text = "Describe what you see in this image in detail."

            image_result = _describe_image_groq_result(
                image_data, user_text, image_file_id
            )
            image_description = image_result.text if image_result else None

            if image_description:
                _append_billing_segment(response_meta, image_result)
                image_context = f"[Imagen: {image_description}]"

                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = (
                            last_message["content"] + f"\n\n{image_context}"
                        )

                print(f"Image described, continuing with normal AI flow...")
            else:
                print("Failed to describe image, continuing without description...")

        fetched_contents = (
            _fetch_urls_from_latest_message(messages) if enable_web_search else ""
        )
        if fetched_contents:
            messages = list(messages) + [
                {"role": "system", "content": fetched_contents}
            ]

        response = complete_with_providers(
            system_message,
            messages,
            response_meta=response_meta,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools or None,
            tool_context=tool_context,
        )
        response = str(response or "")
        if response:
            print(
                f"ask_ai: response len={len(response)} preview='{response[:160].replace(chr(10), ' ')}'"
            )
            return response

        return _mark_ai_fallback_response(get_fallback_response(messages))

    except Exception as e:
        error_context = {
            "messages_count": len(messages),
            "messages_preview": [msg.get("content", "")[:100] for msg in messages],
        }
        admin_report("Error in ask_ai", e, error_context)
        return _mark_ai_fallback_response(get_fallback_response(messages))


def complete_with_providers(
    system_message: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    response_meta: Optional[Dict[str, Any]] = None,
    enable_web_search: bool = True,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Try OpenRouter chat models and return the first response."""

    result = _get_openrouter_ai_response_result(
        system_message,
        messages,
        enable_web_search=enable_web_search,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )
    if result:
        if response_meta is not None:
            _append_billing_segment(response_meta, result)
        print("complete_with_providers: got response from OpenRouter")
        return result.text
    return None


class _HtmlMetadataExtractor(HTMLParser):
    """Extract preview metadata from HTML documents."""

    def __init__(self) -> None:
        super().__init__()
        self._title_parts: List[str] = []
        self._in_title = False
        self.title: Optional[str] = None
        self.description: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attrs_map = {str(key).lower(): value for key, value in attrs}
        if tag_lower == "title":
            self._in_title = True
            return
        if tag_lower != "meta":
            return

        property_name = str(attrs_map.get("property") or "").strip().lower()
        meta_name = str(attrs_map.get("name") or "").strip().lower()
        content = str(attrs_map.get("content") or "").strip()
        if not content:
            return

        normalized_content = re.sub(r"\s+", " ", unescape(content)).strip()
        if not normalized_content:
            return

        if property_name in {"og:title", "twitter:title"} and not self.title:
            self.title = normalized_content
        elif (
            property_name in {"og:description", "twitter:description"}
            and not self.description
        ):
            self.description = normalized_content
        elif meta_name == "description" and not self.description:
            self.description = normalized_content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if not self._in_title:
            return
        text = re.sub(r"\s+", " ", unescape(data)).strip()
        if text:
            self._title_parts.append(text)

    def finalize(self) -> Tuple[Optional[str], Optional[str]]:
        title = self.title or re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()
        return title or None, self.description or None


def _extract_html_metadata(html_text: str) -> Tuple[Optional[str], Optional[str]]:
    parser = _HtmlMetadataExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass
    return parser.finalize()


def _truncate_link_metadata_text(
    text: Optional[str], limit: int = 280
) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _utf16_slice(text: str, offset: int, length: int) -> str:
    if not text or length <= 0:
        return ""

    utf16 = text.encode("utf-16-le")
    start = max(0, int(offset)) * 2
    end = max(start, start + max(0, int(length)) * 2)
    try:
        return utf16[start:end].decode("utf-16-le", errors="ignore")
    except Exception:
        return ""


def _normalize_detected_message_url(raw_url: str) -> Optional[str]:
    candidate = str(raw_url or "").strip().rstrip(".,;:!?)\"]}'")
    if not candidate:
        return None
    return normalize_http_url(candidate)


def _cache_link_metadata(raw_url: str, metadata: Mapping[str, Any]) -> None:
    normalized = normalize_http_url(raw_url)
    if not normalized:
        return

    cache_payload = {
        "url": str(metadata.get("url") or normalized).strip() or normalized,
        "status": metadata.get("status"),
        "content_type": str(metadata.get("content_type") or ""),
        "title": _truncate_link_metadata_text(metadata.get("title"), limit=160),
        "description": _truncate_link_metadata_text(
            metadata.get("description"), limit=280
        ),
    }
    cache_store = _link_metadata_local_cache.setdefault(normalized, {})
    update_local_cache(
        cache_store,
        cache_payload,
        TTL_LINK_METADATA,
        0,
    )


def _extract_urls_from_entity_list(
    source_text: str,
    entities: Any,
) -> List[str]:
    urls: List[str] = []
    if not source_text or not isinstance(entities, Sequence):
        return urls

    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        entity_type = str(entity.get("type") or "").strip().lower()
        if entity_type == "text_link":
            candidate = str(entity.get("url") or "").strip()
        elif entity_type == "url":
            try:
                offset = int(entity.get("offset") or 0)
                length = int(entity.get("length") or 0)
            except (TypeError, ValueError):
                continue
            candidate = _utf16_slice(source_text, offset, length).strip()
        else:
            continue

        normalized = _normalize_detected_message_url(candidate)
        if normalized:
            urls.append(normalized)
    return urls


def extract_message_urls(message: Mapping[str, Any]) -> List[str]:
    """Extract normalized URLs from Telegram message text/caption and entities."""

    candidates: List[str] = []
    for text_key, entities_key in (
        ("text", "entities"),
        ("caption", "caption_entities"),
    ):
        source_text = str(message.get(text_key) or "")
        if source_text:
            candidates.extend(
                _extract_urls_from_entity_list(source_text, message.get(entities_key))
            )
            for match in MESSAGE_URL_PATTERN.finditer(source_text):
                normalized = _normalize_detected_message_url(match.group(1))
                if normalized:
                    candidates.append(normalized)

    unique_urls: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_urls.append(candidate)
        if len(unique_urls) >= MAX_LINKS_IN_MESSAGE:
            break
    return unique_urls


def fetch_link_metadata(raw_url: str) -> Dict[str, Any]:
    """Fetch preview metadata for a URL and cache the result."""

    normalized = normalize_http_url(raw_url)
    if not normalized:
        return {"url": str(raw_url or "").strip(), "error": "url inválida"}

    local_cached, is_fresh, _ = local_cache_get(
        _link_metadata_local_cache.setdefault(normalized, {}),
        allow_stale=False,
    )
    if is_fresh and isinstance(local_cached, dict):
        return local_cached

    redis_client = _optional_redis_client()
    cache_key = _hash_cache_key("link_metadata", {"url": normalized})
    if redis_client is not None:
        try:
            cached = redis_get_json(redis_client, cache_key)
            if isinstance(cached, dict):
                _cache_link_metadata(normalized, cached)
                return cached
        except Exception:
            redis_client = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }

    response = None
    try:
        response = request_with_ssl_fallback(
            normalized,
            headers=headers,
            timeout=8,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
    except RequestException as error:
        return {
            "url": normalized,
            "error": error.__class__.__name__,
        }

    final_url = normalized
    content_type = ""
    status_code = getattr(response, "status_code", None)
    encoding: Optional[str] = None
    apparent_encoding: Optional[str] = None
    content_bytes = b""
    try:
        maybe_url = normalize_http_url(str(getattr(response, "url", "") or ""))
        if maybe_url:
            final_url = maybe_url
        content_type = str(response.headers.get("Content-Type", "")).lower()
        encoding = response.encoding
        apparent_encoding = getattr(response, "apparent_encoding", None)
        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= LINK_METADATA_MAX_BYTES:
                break
        content_bytes = b"".join(chunks)
    finally:
        try:
            response.close()
        except Exception:
            pass

    result: Dict[str, Any] = {
        "url": final_url,
        "status": status_code,
        "content_type": content_type or "",
        "title": None,
        "description": None,
    }

    if content_bytes:
        sample_lower = content_bytes[:400].lower()
        if (
            "html" in content_type
            or b"<html" in sample_lower
            or b"<!doctype" in sample_lower
        ):
            try:
                text_body = content_bytes.decode(
                    encoding or apparent_encoding or "utf-8", errors="replace"
                )
            except Exception:
                text_body = content_bytes.decode("utf-8", errors="replace")
            title, description = _extract_html_metadata(text_body)
            result["title"] = _truncate_link_metadata_text(title, limit=160)
            result["description"] = _truncate_link_metadata_text(description, limit=280)

    if redis_client is not None:
        try:
            redis_setex_json(redis_client, cache_key, TTL_LINK_METADATA, result)
        except Exception:
            pass

    _cache_link_metadata(normalized, result)

    return result


def build_message_links_context(message: Mapping[str, Any]) -> str:
    """Build an AI-only context block with link metadata from the message."""

    urls = extract_message_urls(message)
    if not urls:
        return ""

    print(f"build_message_links_context: extracted {len(urls)} url(s) urls={urls}")

    lines = ["LINKS DEL MENSAJE:"]
    transcript_parts: List[str] = []

    for index, url in enumerate(urls, 1):
        metadata = fetch_link_metadata(url)
        final_url = str(metadata.get("url") or url).strip() or url
        lines.append(f"{index}. {final_url}")
        title = _truncate_link_metadata_text(metadata.get("title"), limit=160)
        description = _truncate_link_metadata_text(
            metadata.get("description"), limit=280
        )
        if title:
            lines.append(f"titulo: {title}")
        if description:
            lines.append(f"descripcion: {description}")

        video_id = extract_youtube_video_id(final_url)
        if video_id:
            transcript = get_youtube_transcript_context(video_id)
            if transcript:
                transcript_parts.append(transcript)

    if transcript_parts:
        lines.append("")
        lines.extend(transcript_parts)

    return "\n".join(lines)


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
    """Get crypto and dollar market data for the system prompt."""
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
    attempt: Callable[[], Optional[Any]],
    rate_limit_backoff: Optional[int] = None,
    label: Optional[str] = None,
    backoff_key: Optional[str] = None,
) -> Optional[Any]:
    """Execute a provider call with shared backoff and error handling."""

    display_name = label or provider_name.capitalize()
    normalized_backoff_key = str(backoff_key or provider_name or "").lower()
    if normalized_backoff_key and is_provider_backoff_active(normalized_backoff_key):
        remaining = int(get_provider_backoff_remaining(normalized_backoff_key))
        print(
            f"{display_name} backoff active ({remaining}s remaining), skipping API call"
        )
        return None

    try:
        return attempt()
    except Exception as error:
        print(f"{display_name} error: {error}")
        if _is_rate_limit_error(error):
            backoff_seconds = _extract_rate_limit_backoff_seconds(
                error, rate_limit_backoff or GROQ_BACKOFF_DEFAULT_SECONDS
            )
            _set_provider_backoff(
                normalized_backoff_key or provider_name,
                backoff_seconds,
            )
            remaining = int(
                get_provider_backoff_remaining(normalized_backoff_key or provider_name)
            )
            print(f"{display_name} rate limit detected; backing off for {remaining}s")
        return None
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


def _should_try_next_groq_account_after_error(error: Exception) -> bool:
    """Return True when a Groq request should fall through to the next account."""

    status_code = getattr(error, "status_code", None)
    if status_code == 413:
        return True

    status = getattr(error, "status", None)
    if status == 413:
        return True

    code = str(getattr(error, "code", "") or "").strip().lower()
    if code == "request_too_large":
        return True

    message = str(getattr(error, "message", "") or error)
    lowered = message.lower()
    return "request_too_large" in lowered or "payload too large" in lowered


def _append_billing_segment(
    response_meta: Optional[Dict[str, Any]],
    result: Optional[AIUsageResult],
) -> None:
    if response_meta is None or result is None:
        return
    response_meta.setdefault("billing_segments", []).append(result.billing_segment())


def _log_groq_request_result(
    *,
    label: str,
    scope: str,
    account: str,
    token_count: int,
    audio_seconds: float,
    result: Optional[AIUsageResult],
) -> None:
    log_entry: Dict[str, Any] = {
        "scope": "groq_request",
        "label": label,
        "request_scope": scope,
        "account": account,
        "estimated_token_count": int(max(0, token_count)),
        "estimated_audio_seconds": float(max(0.0, audio_seconds)),
        "status": "success" if result else "empty",
    }

    if result is not None:
        billing = calculate_billing_for_segments([result.billing_segment()])
        log_entry.update(
            {
                "kind": result.kind,
                "model": result.model,
                "cached": bool(result.cached),
                "text_length": len(result.text or ""),
                "usage": ensure_mapping(result.usage) or {},
                "audio_seconds": result.audio_seconds,
                "metadata": dict(result.metadata or {}),
                "local_billing": {
                    "raw_usd_micros": billing.get("raw_usd_micros", 0),
                    "charged_credit_units": billing.get("charged_credit_units", 0),
                    "charged_credits_display": billing.get(
                        "charged_credits_display", "0.0"
                    ),
                    "model_breakdown": billing.get("model_breakdown", []),
                    "tool_breakdown": billing.get("tool_breakdown", []),
                    "unsupported_notes": billing.get("unsupported_notes", []),
                },
            }
        )

    print(json.dumps(log_entry, ensure_ascii=False, default=str))


def _extract_groq_usage_map(response: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response, dict):
        return ensure_mapping(response.get("usage"))
    return ensure_mapping(getattr(response, "usage", None))


def _extract_latest_user_query_info(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, Optional[int]]:
    latest_user_text = ""
    latest_user_message = ""
    latest_user_index: Optional[int] = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        latest_user_text = content
        latest_user_message = _extract_message_block_from_prompt(content)
        latest_user_index = idx
        break

    return latest_user_text, latest_user_message, latest_user_index


def estimate_ai_base_reserve_credits(
    messages: List[Dict[str, Any]],
    *,
    extra_input_tokens: int = 0,
) -> Tuple[int, Dict[str, Any]]:
    system_message: Optional[Dict[str, Any]] = None
    try:
        context_data = {
            "market": get_market_context(),
            "weather": get_weather_context(),
            "time": get_time_context(),
            "hacker_news": get_hacker_news_context(),
        }
        system_message = build_system_message(context_data)
    except Exception as error:
        print(
            f"estimate_ai_base_reserve_credits: failed to build system message: {error}"
        )

    reserve = estimate_chat_reserve_credits(
        system_message=system_message,
        messages=messages,
        max_output_tokens=CHAT_OUTPUT_TOKEN_LIMIT,
        extra_input_tokens=extra_input_tokens,
        reasoning=True,
    )

    return reserve, {}


def estimate_image_context_reserve_credits(image_data: bytes, prompt_text: str) -> int:
    return estimate_vision_reserve_credits(
        prompt_text=prompt_text,
        image_data=image_data,
        max_output_tokens=VISION_OUTPUT_TOKEN_LIMIT,
    )


def estimate_image_context_rate_limit_tokens(
    image_data: bytes, prompt_text: str
) -> int:
    image_base64 = encode_image_to_base64(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"
    input_payload = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": image_url},
            ],
        }
    ]
    return estimate_message_tokens(input_payload) + VISION_OUTPUT_TOKEN_LIMIT


def _build_groq_usage_result(
    *,
    kind: str,
    text: str,
    model: str,
    response: Any,
    audio_seconds: Optional[float] = None,
    cached: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AIUsageResult:
    return AIUsageResult(
        kind=kind,
        text=text,
        model=model,
        usage=_extract_groq_usage_map(response),
        audio_seconds=audio_seconds,
        cached=cached,
        metadata=dict(metadata or {}),
    )


_MAX_TOOL_ROUNDS = 5
_TOOL_RUNTIME = ToolRuntime()


def _get_openrouter_ai_response_result(
    system_msg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    enable_web_search: bool = True,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
) -> Optional[AIUsageResult]:
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=_get_openrouter_client,
            admin_report=admin_report,
            increment_request_count=_increment_ai_provider_request_count,
            build_web_search_tool=_build_openrouter_web_search_tool,
            build_usage_result=_build_groq_usage_result,
            extract_usage_map=_extract_groq_usage_map,
            primary_model=PRIMARY_CHAT_MODEL,
            max_tool_rounds=_MAX_TOOL_ROUNDS,
        ),
        _TOOL_RUNTIME,
    )
    return runtime.complete(
        system_msg,
        messages,
        enable_web_search=enable_web_search,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )


def _call_summary_model(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], int]:
    client = _get_openrouter_client()
    if client is None:
        return None, 0
    for model in (SUMMARY_MODEL, SUMMARY_FALLBACK_MODEL):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
            )
            if resp and resp.choices and resp.choices[0].message:
                text = str(resp.choices[0].message.content or "").strip()
                usage = resp.usage or {}
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                cost = _estimate_summary_cost_usd_micros(
                    input_tokens, output_tokens, model
                )
                return text, cost
        except Exception as e:
            print(f"summary model {model} error: {e}")
    return None, 0


def _format_messages_for_summary(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(f"{role}: {part['text']}")
    return "\n".join(parts)


def _compact_conversation(dropped_text: str) -> Tuple[str, int]:
    prompt = (
        "summarize this conversation concisely in one paragraph. "
        "capture the main topics, user questions, bot responses, "
        "and any conclusions or pending actions. "
        "skip greetings and casual chat. "
        "match the language of the conversation."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": dropped_text},
    ]
    result, cost = _call_summary_model(messages)
    if result:
        return f"Conversation history summary:\n{result}", cost
    lines = dropped_text.split("\n")
    truncated = "\n".join(lines[:20])
    return f"Earlier conversation (truncated):\n{truncated}", 0


def _estimate_summary_cost_usd_micros(
    input_tokens: int, output_tokens: int, model: str
) -> int:
    pricing = MODEL_PRICING_USD_MICROS.get(model, {})
    input_rate = pricing.get("input_per_million", 100_000)
    output_rate = pricing.get("output_per_million", 400_000)
    return (input_tokens * input_rate + output_tokens * output_rate) // 1_000_000


def _make_summary_result(cost_usd_micros: int) -> Dict[str, Any]:
    return {
        "kind": "summary",
        "text": "context compaction",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "billing": {
            "raw_usd_micros": cost_usd_micros,
            "charged_credit_units": credit_units_from_usd_micros(cost_usd_micros),
        },
    }


def get_fallback_response(messages: List[Dict]) -> str:
    """Generate fallback random response"""
    display_name = ""
    if messages:
        last_message = messages[-1]["content"]
        if "Usuario: " in last_message:
            display_name = last_message.split("Usuario: ")[1].split(" ")[0]
    return gen_random(display_name)


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


def build_system_message(
    context: Dict,
    *,
    tools_active: bool = False,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    task_mode: bool = False,
) -> Dict[str, Any]:
    """Build system message with personality and context."""
    config = load_bot_config()
    market_info = format_market_info(context.get("market") or {})
    weather_source = context.get("weather")
    weather_info = format_weather_info(weather_source) if weather_source else ""
    news_info = format_hacker_news_info(context.get("hacker_news"))
    time_context = context.get("time") or {}
    formatted_time = str(time_context.get("formatted", "")).strip()

    base_prompt = config.get("system_prompt", "")

    task_prefix = ""
    if task_mode:
        task_prefix = (
            "EJECUTANDO TAREA PROGRAMADA:\n"
            "Responde la siguiente instruccion y nada mas.\n"
            "No hagas preguntas, no ofrezcas seguimientos, no pidas confirmacion.\n"
            "Genera tu respuesta y terminá.\n\n"
        )

    tool_instruction = ""
    if tools_active and tool_schemas:
        summaries = []
        for entry in tool_schemas:
            fn = entry.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            summaries.append(f"- {name}: {desc}")
        tool_summaries = "\n".join(summaries) + "\n"
        tool_instruction = (
            f"\n\nHERRAMIENTAS DISPONIBLES:\n{tool_summaries}"
            "Llamalas directamente, sin pedir permiso ni narrar antes.\n"
            "No expliques que vas a hacer antes de usar una herramienta simple.\n"
            "Usa las herramientas exactamente como estan nombradas arriba.\n"
            "\n"
            "task_set detalles:\n"
            "- task_set.text debe contener solo el contenido a ejecutar despues.\n"
            "- no incluyas tiempo ni frecuencia en text si ya van en delay_seconds, interval_seconds o trigger_config.\n"
            "- no reescribas pronombres ni cambies sujeto al guardar la tarea.\n"
            "- si el usuario dice 'decime', 'avisame' o pide que el bot hable de si mismo, preserva eso en el contenido restante.\n"
            "- ejemplo: 'A las 20:30 todos los dias decime cuanta aura farmeaste hoy' -> text=\"decime cuanta aura farmeaste hoy\" y trigger_config con hour=20, minute=30.\n"
            "- ejemplo: 'deci fumareeemooss todos los dias a las 4:20 am' -> text=\"deci fumareeemooss\" y trigger_config con hour=4, minute=20.\n"
            "- ejemplo: 'mañana recordame pagar el alquiler' -> text=\"recordame pagar el alquiler\" y el tiempo va en el parametro de schedule.\n"
            "- trigger_config con type='interval' y days=N para cada N dias.\n"
            "- trigger_config con type='cron', hour, minute para horarios especificos.\n"
            "- cron puede tener day_of_week='lun,mie,vie' o 'mon,wed,fri'; se normaliza internamente. Tambien puede usar day=1 para primer dia del mes.\n"
            "- si no especificas hora para cron, elegi una hora razonable segun el contexto.\n"
        )

    contextual_info = f"""
{tool_instruction}
FECHA ACTUAL:
{formatted_time}

CONTEXTO DEL MERCADO:
{market_info}

CLIMA EN BUENOS AIRES:
{weather_info}

NOTICIAS DE HACKER NEWS:
{news_info}
"""

    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": task_prefix + base_prompt + contextual_info,
            }
        ],
    }


def format_hacker_news_info(
    news: Optional[Iterable[Dict[str, Any]]], include_discussion: bool = True
) -> str:
    """Format Hacker News context for system prompts."""

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
    """Format market data for context with compact summaries."""

    info: List[str] = []

    crypto_rows = market.get("crypto")
    if isinstance(crypto_rows, Sequence) and not isinstance(
        crypto_rows, (str, bytes, bytearray)
    ):
        crypto_lines: List[str] = []
        for crypto in list(crypto_rows)[:3]:
            if not isinstance(crypto, Mapping):
                continue
            symbol = (
                str(crypto.get("symbol") or crypto.get("name") or "").strip().upper()
            )
            usd_quote = (
                ensure_mapping((ensure_mapping(crypto.get("quote")) or {}).get("USD"))
                or {}
            )
            price = _safe_float(
                usd_quote.get("price") if usd_quote else crypto.get("price")
            )
            change_24h = _safe_float(
                (ensure_mapping(usd_quote.get("changes")) or {}).get("24h")
                if usd_quote
                else crypto.get("change_24h")
            )
            dominance = _safe_float(usd_quote.get("dominance")) if usd_quote else None
            if not symbol or price is None:
                continue
            line = f"- {symbol}: {fmt_num(price, 2)} usd"
            if change_24h is not None:
                line += f" ({fmt_signed_pct(change_24h, 2)} 24h)"
            if dominance is not None:
                line += f", dom {fmt_num(dominance, 1)}%"
            crypto_lines.append(line)
        if crypto_lines:
            info.append("PRECIOS DE CRIPTOS:")
            info.extend(crypto_lines)

    dollar_value = market.get("dollar")
    dollar_data = ensure_mapping(dollar_value)
    if (
        not dollar_data
        and isinstance(dollar_value, Sequence)
        and not isinstance(dollar_value, (str, bytes, bytearray))
    ):
        sequence_lines: List[str] = []
        for item in dollar_value:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("name") or item.get("label") or "").strip().lower()
            price = _safe_float(item.get("price"))
            if label and price is not None:
                sequence_lines.append(f"- {label}: {fmt_num(price, 2)}")
        if sequence_lines:
            info.append("DOLARES:")
            info.extend(sequence_lines)
    elif dollar_data:
        dollar_lines: List[str] = []

        oficial_price = _safe_float(
            (ensure_mapping(dollar_data.get("oficial")) or {}).get("price")
        )
        if oficial_price is not None:
            dollar_lines.append(f"- oficial: {fmt_num(oficial_price, 2)}")

        blue_data = ensure_mapping(dollar_data.get("blue")) or {}
        blue_ask = _safe_float(blue_data.get("ask") or blue_data.get("price"))
        blue_bid = _safe_float(blue_data.get("bid"))
        if blue_ask is not None:
            blue_line = f"- blue: {fmt_num(blue_ask, 2)}"
            if blue_bid is not None:
                blue_line += f" (bid {fmt_num(blue_bid, 2)})"
            dollar_lines.append(blue_line)

        mep_ci = ensure_mapping(
            (ensure_mapping(dollar_data.get("mep")) or {}).get("al30")
        )
        mep_ci = ensure_mapping((mep_ci or {}).get("ci")) or {}
        mep_price = _safe_float(mep_ci.get("price"))
        if mep_price is not None:
            dollar_lines.append(f"- mep al30 ci: {fmt_num(mep_price, 2)}")

        tarjeta_price = _safe_float(
            (ensure_mapping(dollar_data.get("tarjeta")) or {}).get("price")
        )
        if tarjeta_price is not None:
            dollar_lines.append(f"- tarjeta: {fmt_num(tarjeta_price, 2)}")

        usdt_data = (
            ensure_mapping(
                (ensure_mapping(dollar_data.get("cripto")) or {}).get("usdt")
            )
            or {}
        )
        usdt_ask = _safe_float(usdt_data.get("ask"))
        usdt_bid = _safe_float(usdt_data.get("bid"))
        if usdt_ask is not None:
            usdt_line = f"- usdt: {fmt_num(usdt_ask, 2)}"
            if usdt_bid is not None:
                usdt_line += f" (bid {fmt_num(usdt_bid, 2)})"
            dollar_lines.append(usdt_line)

        if dollar_lines:
            info.append("DOLARES:")
            info.extend(dollar_lines)

    return "\n".join(info)


def format_weather_info(weather: Dict) -> str:
    """Format weather data for context"""
    visibility_km = weather.get("visibility")
    visibility_str = (
        f"{visibility_km / 1000:.1f}km" if visibility_km is not None else "sin datos"
    )
    return (
        f"- Temperatura aparente: {weather.get('apparent_temperature', '?')}°C\n"
        f"- Probabilidad de lluvia: {weather.get('precipitation_probability', '?')}%\n"
        f"- Estado: {weather.get('description', 'sin datos')}\n"
        f"- Nubosidad: {weather.get('cloud_cover', '?')}%\n"
        f"- Visibilidad: {visibility_str}"
    )


def build_ai_messages(
    message: Dict,
    chat_history: List[Dict],
    message_text: str,
    reply_context: Optional[str] = None,
    enable_web_search: bool = True,
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
    sender = cast(Mapping[str, Any], message.get("from", {}))
    chat = cast(Mapping[str, Any], message.get("chat", {}))
    first_name = str(sender.get("first_name") or "Usuario")
    username = str(sender.get("username") or "")
    chat_type = str(chat.get("type") or "private")
    chat_title = str(chat.get("title") or "") if chat_type != "private" else ""
    current_time = datetime.now(BA_TZ)

    # Build context sections
    context_parts = [
        "CONTEXTO:",
        f"- Chat: {chat_type}" + (f" ({chat_title})" if chat_title else ""),
        f"- Usuario: {first_name}" + (f" ({username})" if username else ""),
        f"- Hora: {current_time.strftime('%H:%M')}",
    ]

    last_in_history_is_assistant = (
        bool(messages) and messages[-1].get("role") == "assistant"
    )
    if reply_context and not last_in_history_is_assistant:
        context_parts.extend(
            [
                "",
                "MENSAJE AL QUE RESPONDE:",
                truncate_text(reply_context),
            ]
        )

    link_context = build_message_links_context(message)
    if link_context:
        context_parts.extend(["", link_context])

    instructions = [
        "",
        "INSTRUCCIONES:",
        "- mantené el personaje del gordo",
        "- usá lenguaje coloquial argentino",
    ]
    if enable_web_search:
        instructions.append("- si no estás seguro de algo podes buscarlo en internet")

    context_parts.extend(
        [
            "",
            "MENSAJE:",
            truncate_text(message_text),
        ]
        + instructions
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
            "config_command": _noop_command,
            "convert_base": convert_base,
            "select_random": select_random,
            "get_prices": get_prices,
            "get_dollar_rates": get_dollar_rates,
            "get_oil_price": get_oil_price,
            "get_stock_prices": get_stock_prices,
            "get_polymarket_argentina_election": get_polymarket_argentina_election,
            "get_rulo": get_rulo,
            "get_devo": get_devo,
            "powerlaw": powerlaw,
            "rainbow": rainbow,
            "satoshi": satoshi,
            "get_timestamp": get_timestamp,
            "convert_to_command": convert_to_command,
            "get_instance_name": get_instance_name,
            "get_help": get_help,
            "handle_transcribe": handle_transcribe,
            "handle_bcra_variables": handle_bcra_variables,
            "topup_command": _noop_command,
            "balance_command": _noop_command,
            "printcredits_command": _noop_param_command,
            "creditlog_command": _noop_param_command,
            "transfer_command": _noop_param_command,
            "get_good_morning": get_good_morning,
            "get_good_night": get_good_night,
            "tasks_command": tasks_command,
            "summary_command": lambda: "",
        }
    )


def truncate_text(text: Optional[str], max_length: int = 512) -> str:
    return _state_truncate_text(text, max_length)


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
    chat_id: str, redis_client: redis.Redis, max_messages: int = 4
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
    return _ROUTING_POLICY.should_respond(
        commands,
        command,
        message_text,
        message,
        chat_config,
        reply_metadata,
    )


def _has_ai_credits_for_random_reply(message: Mapping[str, Any]) -> bool:
    if not credits_db_service.is_configured():
        return False

    user_id = _extract_user_id(message)
    if user_id is None:
        return False

    if _fetch_balance("user", user_id) > 0:
        return True

    chat = cast(Mapping[str, Any], message.get("chat") or {})
    if not is_group_chat_type(str(chat.get("type") or "")):
        return False

    chat_id = _extract_numeric_chat_id(str(chat.get("id") or ""))
    if chat_id is None:
        return False

    return _fetch_balance("chat", chat_id) > 0


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


def check_provider_available(scope: str) -> bool:
    """Check whether at least one Groq account for the scope is not in cooldown.

    Returns True if any account is available. Returns True if no accounts are
    configured (allowing the call to proceed to OpenRouter). Returns False only
    when all configured accounts are in active cooldown.
    """

    configured_accounts = _get_groq_accounts_for_scope()
    if not configured_accounts:
        return True

    for account in configured_accounts:
        if not is_provider_backoff_active(_get_groq_backoff_key(account, scope)):
            return True
    return False


def has_openrouter_fallback() -> bool:
    return _get_openrouter_api_key() is not None


# Billing helpers: expose ai_billing functions directly while keeping
# the small compatibility surface required by callers/tests.
# These are thin aliases that forward to implementations in api.ai_billing.
get_ai_onboarding_credits = _billing_get_ai_onboarding_credits
get_ai_billing_packs = _billing_get_ai_billing_packs
get_ai_billing_pack = _billing_get_ai_billing_pack
build_topup_keyboard = _billing_build_topup_keyboard
_parse_topup_payload = _billing_parse_topup_payload


def build_insufficient_credits_message(
    *, chat_type: str, user_balance: int, chat_balance: int
) -> str:
    return _billing_build_insufficient_credits_message(
        chat_type=chat_type,
        user_balance=user_balance,
        chat_balance=chat_balance,
    )


_extract_numeric_chat_id = _billing_extract_numeric_chat_id
_extract_user_id = _billing_extract_user_id


def _fetch_balance(scope_type: Literal["user", "chat"], scope_id: int) -> int:
    return credits_db_service.get_balance(scope_type, int(scope_id))


def _maybe_grant_onboarding_credits(user_id: Optional[int]) -> None:
    _billing_maybe_grant_onboarding_credits(credits_db_service, admin_report, user_id)


def _send_stars_invoice(
    *,
    chat_id: str,
    user_id: int,
    pack: Mapping[str, int],
) -> bool:
    payload = f"topup:{pack['id']}:{user_id}"
    pack_credits = format_credit_units(pack["credits"])
    request_payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "title": f"Pack IA {pack_credits} créditos",
        "description": f"Recarga de {pack_credits} créditos para mensajes IA",
        "payload": payload,
        "provider_token": "",
        "currency": "XTR",
        "prices": [{"label": f"{pack_credits} créditos IA", "amount": pack["xtr"]}],
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

    # Extract audio/voice/video
    audio_file_id = None
    if "voice" in message and message["voice"]:
        audio_file_id = message["voice"]["file_id"]
    elif "audio" in message and message["audio"]:
        audio_file_id = message["audio"]["file_id"]
    elif "video" in message and message["video"]:
        audio_file_id = message["video"]["file_id"]
        print(f"Found video: {audio_file_id}")
    elif "video_note" in message and message["video_note"]:
        audio_file_id = message["video_note"]["file_id"]
        print(f"Found video_note: {audio_file_id}")
    # Also check for audio/video in replied message
    elif "reply_to_message" in message and message["reply_to_message"]:
        replied_msg = message["reply_to_message"]
        if "voice" in replied_msg and replied_msg["voice"]:
            audio_file_id = replied_msg["voice"]["file_id"]
            print(f"Found voice in quoted message: {audio_file_id}")
        elif "audio" in replied_msg and replied_msg["audio"]:
            audio_file_id = replied_msg["audio"]["file_id"]
            print(f"Found audio in quoted message: {audio_file_id}")
        elif "video" in replied_msg and replied_msg["video"]:
            audio_file_id = replied_msg["video"]["file_id"]
            print(f"Found video in quoted message: {audio_file_id}")
        elif "video_note" in replied_msg and replied_msg["video_note"]:
            audio_file_id = replied_msg["video_note"]["file_id"]
            print(f"Found video_note in quoted message: {audio_file_id}")

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


def _get_groq_client(
    account: str,
    *,
    default_headers: Optional[Mapping[str, str]] = None,
) -> Optional[OpenAI]:
    groq_api_key = _get_groq_api_key(account)
    if not groq_api_key:
        print(f"Groq API key not configured for account={account}")
        return None

    cf_aig_token = environ.get("CF_AIG_TOKEN")
    cf_gateway_base_url = environ.get(
        "CF_AIG_BASE_URL", "https://api.groq.com/openai/v1"
    )

    headers: Dict[str, str] = dict(default_headers) if default_headers else {}
    if cf_aig_token:
        headers["cf-aig-authorization"] = f"Bearer {cf_aig_token}"

    client_kwargs: Dict[str, Any] = {
        "api_key": groq_api_key,
        "base_url": cf_gateway_base_url,
    }
    if headers:
        client_kwargs["default_headers"] = headers
    return OpenAI(**client_kwargs)


def _get_groq_native_client(
    account: str,
) -> Optional[GroqClient]:
    """Create native Groq client using groq library (not OpenAI SDK)."""
    groq_api_key = _get_groq_api_key(account)
    if not groq_api_key:
        print(f"Groq API key not configured for account={account}")
        return None

    return GroqClient(api_key=groq_api_key)


def _execute_groq_request_with_fallback(
    scope: str,
    *,
    label: str,
    token_count: int = 0,
    audio_seconds: float = 0.0,
    attempt: Callable[[str], Optional[AIUsageResult]],
) -> Optional[AIUsageResult]:
    configured_accounts = list(_get_groq_accounts_for_scope())
    if not configured_accounts:
        print("Groq API key not configured")
        return None

    for account in configured_accounts:
        last_error: Optional[Exception] = None

        def _wrapped_attempt() -> Optional[AIUsageResult]:
            nonlocal last_error
            try:
                return attempt(account)
            except Exception as error:
                last_error = error
                raise

        request_started_at = time.monotonic()
        result = cast(
            Optional[AIUsageResult],
            _invoke_provider(
                "groq",
                attempt=_wrapped_attempt,
                rate_limit_backoff=GROQ_BACKOFF_DEFAULT_SECONDS,
                label=f"{label} ({account})",
                backoff_key=_get_groq_backoff_key(account, scope),
            ),
        )
        request_elapsed_seconds = max(0.0, time.monotonic() - request_started_at)
        if result is not None:
            result.metadata.setdefault(
                "request_elapsed_seconds", request_elapsed_seconds
            )
        _log_groq_request_result(
            label=label,
            scope=scope,
            account=account,
            token_count=token_count,
            audio_seconds=audio_seconds,
            result=result,
        )
        if result:
            result.metadata.setdefault("groq_account", account)
            return result
        if last_error and _should_try_next_groq_account_after_error(last_error):
            print(
                f"{label} retrying with next account after recoverable error on account={account}"
            )
            continue
        if is_provider_backoff_active(_get_groq_backoff_key(account, scope)):
            continue
        break

    return None


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


def measure_audio_duration_seconds(audio_data: bytes) -> Optional[float]:
    """Measure audio duration from file bytes, returning None when it cannot be determined."""

    if not audio_data:
        return None

    try:
        parsed = MutagenFile(io.BytesIO(audio_data))
        info = getattr(parsed, "info", None)
        length = getattr(info, "length", None)
        if isinstance(length, (int, float)) and length > 0:
            return float(length)
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate > 0 and frame_count > 0:
                return float(frame_count) / float(frame_rate)
    except Exception:
        pass

    return None


def extract_audio_from_video(video_data: bytes) -> Optional[bytes]:
    """Extract audio track from video bytes using ffmpeg.

    Returns audio bytes in OGG/Opus format, or None on failure.
    """
    if not video_data:
        return None
    try:
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4") as vid_f,
            tempfile.NamedTemporaryFile(suffix=".ogg") as aud_f,
        ):
            vid_f.write(video_data)
            vid_f.flush()
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    vid_f.name,
                    "-vn",
                    "-acodec",
                    "libopus",
                    "-b:a",
                    "64k",
                    aud_f.name,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"ffmpeg failed: {result.stderr[:500]}")
                return None
            aud_f.seek(0)
            audio_bytes = aud_f.read()
            if len(audio_bytes) == 0:
                return None
            return audio_bytes
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None


def _describe_image_groq_result(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[AIUsageResult]:
    """Describe image using Groq vision models."""

    if file_id and use_cache:
        cached = get_cached_description(file_id)
        if cached:
            return AIUsageResult(
                kind="vision",
                text=str(cached),
                model=GROQ_VISION_MODEL,
                cached=True,
                metadata={"file_id": file_id, "cache_hit": True},
            )

    image_base64 = encode_image_to_base64(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"
    estimated_token_count = (
        estimate_message_tokens(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        )
        + VISION_OUTPUT_TOKEN_LIMIT
    )

    def _attempt(account: str) -> Optional[AIUsageResult]:
        groq_client = _get_groq_client(account)
        if groq_client is None:
            return None
        print(f"Describing image with Groq vision model using account={account}...")
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
            max_output_tokens=VISION_OUTPUT_TOKEN_LIMIT,
        )
        description = _extract_response_text(response)
        if description:
            print(f"Image description successful: {description[:100]}...")
            return _build_groq_usage_result(
                kind="vision",
                text=description,
                model=GROQ_VISION_MODEL,
                response=response,
                metadata={
                    "file_id": file_id,
                    "cache_hit": False,
                    "groq_account": account,
                },
            )
        return None

    result = _execute_groq_request_with_fallback(
        attempt=_attempt,
        scope="vision",
        label="Groq Vision",
        token_count=estimated_token_count,
    )

    if result is None:
        result = _describe_image_openrouter_result(image_data, user_text, file_id)

    if result and result.text and file_id:
        cache_description(file_id, result.text)

    return result


def _describe_image_openrouter_result(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
) -> Optional[AIUsageResult]:
    model = _get_openrouter_vision_model(GROQ_VISION_MODEL)
    if not model:
        return None

    client = _get_openrouter_client()
    if client is None:
        return None

    print("Trying OpenRouter vision as fallback...")
    _increment_ai_provider_request_count()
    image_base64 = encode_image_to_base64(image_data)
    image_url = f"data:image/jpeg;base64,{image_base64}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            ),
            max_tokens=VISION_OUTPUT_TOKEN_LIMIT,
        )
    except Exception:
        return None
    if response and hasattr(response, "choices") and response.choices:
        content = getattr(response.choices[0].message, "content", None)
        if content:
            return _build_groq_usage_result(
                kind="vision",
                text=str(content),
                model=model,
                response=response,
                metadata={"file_id": file_id, "provider": "openrouter"},
            )
    return None


def describe_image_groq(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[str]:
    result = _describe_image_groq_result(
        image_data,
        user_text,
        file_id,
        use_cache=use_cache,
    )
    if result:
        return result.text
    return None


OPENROUTER_TRANSCRIBE_MODEL = "google/gemini-3.1-flash-lite-preview"


def _transcribe_audio_openrouter_result(
    audio_data: bytes,
    file_id: Optional[str] = None,
) -> Optional[AIUsageResult]:
    """Transcribe audio using OpenRouter with Gemini (fallback for Groq)."""
    model = OPENROUTER_TRANSCRIBE_MODEL

    client = _get_openrouter_client()
    if client is None:
        print("OpenRouter transcription: no client available")
        return None

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    audio_format = "webm"
    if audio_data.startswith(b"\x1aE\xdf\xa3"):
        audio_format = "mp3"
    elif audio_data.startswith(b"ID3"):
        audio_format = "mp3"
    elif audio_data.startswith(b"OggS"):
        audio_format = "ogg"

    print(f"Transcribing audio with OpenRouter Gemini using model={model}...")
    _increment_ai_provider_request_count()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "format": audio_format,
                                    "data": audio_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Transcribe this audio exactly as spoken.",
                            },
                        ],
                    }
                ],
            ),
            max_tokens=4096,
        )
    except Exception as e:
        print(f"OpenRouter transcription error: {e}")
        return None

    if response and hasattr(response, "choices") and response.choices:
        content = getattr(response.choices[0].message, "content", None)
        if content:
            print(f"Audio transcribed successfully: {content[:100]}...")
            return _build_groq_usage_result(
                kind="transcribe",
                text=str(content),
                model=model,
                response=response,
                audio_seconds=0.0,
                metadata={
                    "file_id": file_id,
                    "cache_hit": False,
                    "provider": "openrouter",
                },
            )
    return None


def _transcribe_audio_result(
    audio_data: bytes,
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[AIUsageResult]:
    """Transcribe audio using Groq Whisper, falling back to OpenRouter."""

    if file_id and use_cache:
        cached = get_cached_transcription(file_id)
        if cached:
            return AIUsageResult(
                kind="transcribe",
                text=str(cached),
                model=GROQ_TRANSCRIBE_MODEL,
                cached=True,
                metadata={"file_id": file_id, "cache_hit": True},
            )

    measured_audio_seconds = measure_audio_duration_seconds(audio_data)

    def _attempt(account: str) -> Optional[AIUsageResult]:
        native_client = _get_groq_native_client(account)
        if native_client is None:
            return None
        print(f"Transcribing audio with Groq Whisper using account={account}...")

        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        response = native_client.audio.transcriptions.create(
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
            return _build_groq_usage_result(
                kind="transcribe",
                text=str(transcription),
                model=GROQ_TRANSCRIBE_MODEL,
                response=response,
                audio_seconds=measured_audio_seconds,
                metadata={
                    "file_id": file_id,
                    "cache_hit": False,
                    "groq_account": account,
                },
            )
        return None

    result = _execute_groq_request_with_fallback(
        attempt=_attempt,
        scope="transcribe",
        label="Groq Whisper",
        audio_seconds=measured_audio_seconds or 0.0,
    )

    if result is None:
        print("Groq transcription failed, trying OpenRouter fallback...")
        result = _transcribe_audio_openrouter_result(audio_data, file_id)

    if result and result.text and file_id:
        cache_transcription(file_id, result.text)

    return result


def transcribe_audio_groq(
    audio_data: bytes,
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[str]:
    result = _transcribe_audio_result(
        audio_data,
        file_id,
        use_cache=use_cache,
    )
    if result:
        return result.text
    return None


def _process_media_with_cache(
    *,
    file_id: str,
    use_cache: bool,
    cache_lookup: Optional[Callable[[str], Optional[str]]],
    processor: Callable[[bytes], Optional[AIUsageResult]],
    downloader: Optional[Callable[[str], Optional[bytes]]] = None,
    failure_code: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Shared helper for cached Telegram media download + processing."""

    try:
        if use_cache and cache_lookup:
            cached_value = cache_lookup(file_id)
            if cached_value:
                return str(cached_value), None, None

        media_fetcher = downloader or download_telegram_file
        media_bytes = media_fetcher(file_id)
        if not media_bytes:
            return None, "download", None

        result = processor(media_bytes)
        if result:
            if not result.audio_seconds and result.kind == "transcribe":
                result.audio_seconds = measure_audio_duration_seconds(media_bytes)
            return result.text, None, result.billing_segment()
        return None, failure_code, None
    except Exception as error:
        print(f"Error processing media {file_id}: {error}")
        return None, failure_code, None


def transcribe_file_by_id(
    file_id: str, use_cache: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Fetch transcription for a Telegram file_id with cache and retries.

    Returns (text, error):
    - On success: (transcription, None)
    - If download failed: (None, "download")
    - If transcription failed: (None, "transcribe")
    """
    try:
        if use_cache:
            cached_value = get_cached_transcription(file_id)
            if cached_value:
                return str(cached_value), None, None

        media_bytes = download_telegram_file(file_id)
        if not media_bytes:
            return None, "download", None

        duration_seconds = measure_audio_duration_seconds(media_bytes)
        if duration_seconds is None:
            extracted = extract_audio_from_video(media_bytes)
            if extracted:
                print("Extracted audio from video for transcription")
                media_bytes = extracted
                duration_seconds = measure_audio_duration_seconds(media_bytes)
            if duration_seconds is None:
                return None, "duration", None

        result = _transcribe_audio_result(media_bytes, file_id, use_cache=use_cache)
        if result:
            if not result.audio_seconds:
                result.audio_seconds = duration_seconds
            return result.text, None, result.billing_segment()
        return None, "transcribe", None
    except Exception as error:
        print(f"Error processing media {file_id}: {error}")
        return None, "transcribe", None


def describe_media_by_id(
    file_id: str, prompt: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Fetch description for an image/sticker by Telegram file_id using Groq vision.

    Returns (description, error):
    - On success: (description, None)
    - If download failed: (None, "download")
    - If description failed: (None, "describe")
    """

    def _processor(media: bytes) -> Optional[AIUsageResult]:
        resized = resize_image_if_needed(media)
        return _describe_image_groq_result(resized, prompt, file_id)

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


def build_routing_policy() -> RoutingPolicy:
    """Builder for the global routing policy (returns the configured instance)."""
    return RoutingPolicy(
        base_policy=_command_should_gordo_respond,
        has_ai_credits_for_random_reply=_has_ai_credits_for_random_reply,
        load_bot_config_fn=load_bot_config,
    )


# Module-level composed instances (composition root surface)
telegram_gateway = TelegramGateway(_telegram_request)
_ROUTING_POLICY = build_routing_policy()


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


def handle_config_command(
    chat_id: str, chat_type: str = ""
) -> Tuple[str, Dict[str, Any]]:
    redis_client = config_redis()
    config = get_chat_config(redis_client, chat_id)
    return build_config_text(config, chat_type), build_config_keyboard(
        config, chat_type
    )


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


def _billing_unavailable_alert_text() -> str:
    return "el cobro de ia está hecho pelota, avisale al admin"


def _billing_unavailable_message_text() -> str:
    return "el cobro de ia no está andando, avisale al admin"


def _billing_is_available() -> bool:
    return bool(credits_db_service.is_configured())


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

    if not _billing_is_available():
        if callback_id:
            _answer_callback_query(
                callback_id,
                text=_billing_unavailable_alert_text(),
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

    if not _billing_is_available():
        _answer_pre_checkout_query(
            str(query_id),
            ok=False,
            error_message=_billing_unavailable_alert_text(),
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

    if not _billing_is_available():
        send_msg(chat_id, _billing_unavailable_message_text())
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
        send_msg(
            chat_id, "me entró la guita pero se trabó la acreditación, avisale al admin"
        )
        return "ok"

    balance = int(payment_result.get("user_balance", 0))
    pack_credits = format_credit_units(pack["credits"])
    if payment_result.get("inserted"):
        send_msg(
            chat_id,
            (
                f"listo, te cargué {pack_credits} créditos\n"
                f"ahora te quedaron {format_credit_units(balance)}\n"
                "si querés mandarle al grupo: /transfer <monto>"
            ),
        )
    else:
        send_msg(
            chat_id,
            (
                "ese pago ya estaba cargado, no rompas las bolas\n"
                f"te quedaron {format_credit_units(balance)}"
            ),
        )
    return "ok"


def handle_task_callback(callback_query: Dict[str, Any]) -> None:
    callback_data = callback_query.get("data")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    user = callback_query.get("from") or {}

    if not callback_data or chat_id is None:
        if callback_id:
            _answer_callback_query(callback_id)
        return

    parts = str(callback_data).split(":", 2)
    if len(parts) != 3 or parts[0] != "task":
        if callback_id:
            _answer_callback_query(callback_id)
        return

    _, _, task_id = parts

    tasks = _task_list_tasks(str(chat_id))
    target_task = next((t for t in tasks if str(t.get("id")) == str(task_id)), None)
    if not target_task:
        if callback_id:
            _answer_callback_query(
                callback_id, text="esa tarea no existe", show_alert=True
            )
        return

    request_user_id = user.get("id")
    task_owner_id = target_task.get("user_id")
    is_owner = task_owner_id and str(request_user_id) == str(task_owner_id)

    chat_type = str(chat.get("type", ""))
    is_admin = False
    if is_group_chat_type(chat_type):
        redis_client = config_redis()
        is_admin = is_chat_admin(
            str(chat_id), request_user_id, redis_client=redis_client
        )
    else:
        is_admin = True

    if not is_owner and not is_admin:
        if callback_id:
            _answer_callback_query(
                callback_id,
                text="solo el creador o un admin pueden borrar esta tarea",
                show_alert=True,
            )
        return

    _task_cancel_task(task_id)

    if callback_id:
        _answer_callback_query(callback_id, text=f"tarea {task_id} borrada")

    if message_id:
        tasks = _task_list_tasks(str(chat_id))
        new_text, new_keyboard = _build_tasks_message(tasks)
        try:
            edit_message(str(chat_id), int(message_id), new_text, new_keyboard)
        except Exception:
            pass


def edit_message(
    chat_id: str,
    message_id: int,
    text: str,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
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

    if str(callback_data).startswith("task:"):
        handle_task_callback(callback_query)
        return

    redis_client = config_redis()
    chat_id_str = str(chat_id)
    chat_type = str(chat.get("type", ""))

    is_config_callback = str(callback_data).startswith("cfg:")
    if is_config_callback and is_group_chat_type(chat_type):
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
        current = coerce_bool(config.get("ai_random_replies"), default=True)
        config = set_chat_config(
            redis_client,
            chat_id_str,
            ai_random_replies=not current,
        )
    elif action == "followups":
        current = coerce_bool(config.get("ai_command_followups"), default=True)
        config = set_chat_config(
            redis_client,
            chat_id_str,
            ai_command_followups=not current,
        )
    elif action == "linkfixfollowups":
        current = coerce_bool(config.get("ignore_link_fix_followups"), default=True)
        config = set_chat_config(
            redis_client,
            chat_id_str,
            ignore_link_fix_followups=not current,
        )
    elif action == "timezone":
        if value == "current":
            if callback_id:
                _answer_callback_query(callback_id)
            return
        else:
            try:
                offset = max(TIMEZONE_OFFSET_MIN, min(int(value), TIMEZONE_OFFSET_MAX))
                config = set_chat_config(
                    redis_client,
                    chat_id_str,
                    timezone_offset=offset,
                )
            except ValueError:
                pass
    elif action == "creditless":
        try:
            current_limit = int(
                config.get(
                    "creditless_user_hourly_limit",
                    config.get("creditless_user_daily_limit", 5),
                )
            )
            if value == "none":
                limit = 0
            elif value == "decrease":
                limit = (
                    current_limit if current_limit < 0 else max(0, current_limit - 1)
                )
            elif value == "current":
                if callback_id:
                    _answer_callback_query(callback_id)
                return
            elif value == "increase":
                limit = current_limit if current_limit < 0 else current_limit + 1
            elif value == "unlimited":
                limit = -1
            else:
                limit = int(value)
                if limit < -1:
                    raise ValueError
            config = set_chat_config(
                redis_client,
                chat_id_str,
                creditless_user_hourly_limit=limit,
            )
        except ValueError:
            pass

    text = build_config_text(config, chat_type)
    keyboard = build_config_keyboard(config, chat_type)
    try:
        edit_succeeded = edit_message(chat_id_str, int(message_id), text, keyboard)
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
        decode_redis_value=decode_redis_value,
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


def _build_message_handler_deps() -> MessageHandlerDeps:
    ai_svc = build_ai_service(
        credits_db_service=credits_db_service,
        get_chat_history=get_chat_history,
        build_ai_messages=build_ai_messages,
        check_provider_available=check_provider_available,
        has_openrouter_fallback=has_openrouter_fallback,
        handle_rate_limit=handle_rate_limit,
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
    )
    return build_message_handler_deps(
        chat=MessageChatDeps(
            config_redis=config_redis,
            get_chat_config=get_chat_config,
            extract_user_id=_extract_user_id,
            extract_numeric_chat_id=_extract_numeric_chat_id,
        ),
        routing=MessageRoutingDeps(
            initialize_commands=initialize_commands,
            parse_command=parse_command,
            should_auto_process_media=should_auto_process_media,
            replace_links=replace_links,
            should_gordo_respond=should_gordo_respond,
            is_group_chat_type=is_group_chat_type,
        ),
        io=MessageIODeps(
            send_msg=send_msg,
            send_animation=send_animation,
            delete_msg=delete_msg,
            admin_report=admin_report,
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=get_bot_message_metadata,
            save_bot_message_metadata=save_bot_message_metadata,
            build_reply_context_text=build_reply_context_text,
            build_message_links_context=build_message_links_context,
            format_user_message=format_user_message,
            save_message_to_redis=save_message_to_redis,
        ),
        ai=MessageAIDeps(
            ai_service=ai_svc,
            balance_formatter=BalanceFormatter(credits_db_service),
            ask_ai=ask_ai,
            gen_random=gen_random,
            build_insufficient_credits_message=build_insufficient_credits_message,
            build_topup_keyboard=build_topup_keyboard,
            credits_db_service=credits_db_service,
            maybe_grant_onboarding_credits=lambda _svc, _rep, uid: (
                _maybe_grant_onboarding_credits(uid)
            ),
            handle_transcribe_with_message=handle_transcribe_with_message,
            handle_transcribe_with_message_result=handle_transcribe_with_message_result,
            check_provider_available=check_provider_available,
            has_openrouter_fallback=has_openrouter_fallback,
            handle_rate_limit=handle_rate_limit,
            handle_successful_payment_message=handle_successful_payment_message,
            handle_config_command=handle_config_command,
            is_chat_admin=is_chat_admin,
            report_unauthorized_config_attempt=_report_unauthorized_config_attempt,
            handle_transcribe=handle_transcribe,
            estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
            estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
        ),
        media=MessageMediaDeps(
            extract_message_content=extract_message_content,
            _transcribe_audio_file=_transcribe_audio_file,
            _transcription_error_message=_transcription_error_message,
            download_telegram_file=download_telegram_file,
            measure_audio_duration_seconds=measure_audio_duration_seconds,
            resize_image_if_needed=resize_image_if_needed,
            encode_image_to_base64=encode_image_to_base64,
        ),
    )


def handle_msg(message: Dict) -> str:
    return _handle_msg_impl(message, _build_message_handler_deps())


def handle_rate_limit(chat_id: str, message: Dict) -> str:
    """Handle rate limited responses"""
    token = environ.get("TELEGRAM_TOKEN")
    if token:
        send_typing(token, chat_id)
    time.sleep(random.uniform(0, 1))
    return build_random_reply(
        gen_random,
        cast(Mapping[str, Any], message.get("from") or {}),
    )


def handle_ai_response(
    chat_id: str,
    handler_func: Callable,
    messages: List[Dict],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    context_texts: Optional[Sequence[Optional[str]]] = None,
    user_identity: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    timezone_offset: int = -3,
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
        user_id=user_id,
        timezone_offset=timezone_offset,
        send_typing_fn=send_typing,
        telegram_token=environ.get("TELEGRAM_TOKEN"),
        reset_request_count_fn=_reset_ai_provider_request_count,
        restore_request_count_fn=_restore_ai_provider_request_count,
        get_request_count_fn=_get_ai_provider_request_count,
        strip_ai_fallback_marker_fn=_strip_ai_fallback_marker,
    )


def update_telegram_bot_commands() -> bool:
    """Update the bot's command menu in Telegram automatically.

    Builds command list from COMMAND_GROUPS and calls setMyCommands.
    """
    token = environ.get("TELEGRAM_TOKEN")
    if not token:
        print("TELEGRAM_TOKEN not set, cannot update bot commands")
        return False
    try:
        return _update_bot_commands(
            token=token,
            request_fn=_telegram_request,
            command_groups=COMMAND_GROUPS,
            logger=print,
        )
    except Exception as e:
        print(f"Exception updating bot commands: {e}")
        return False
