from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone, date, UTC
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
    Mapping,
    Sequence,
    TypeVar,
    TYPE_CHECKING,
    Literal,
    Iterator,
)
import atexit
import concurrent.futures
import hashlib
import io
import json
import random
import re
import redis
import requests
import time
from api.provider_backoff import (
    mark_provider_cooldown,
    get_provider_cooldown_remaining as _get_cooldown_remaining,
    is_provider_cooled_down,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputParam
else:
    ResponseInputParam = Any  # type: ignore[assignment]

from api.utils import (
    fmt_num,
)
from api.general_commands import (
    convert_base,
    convert_to_command,
    gen_random,
    get_instance_name,
    get_timestamp,
    is_japanese_text,
    romanize_japanese,
    select_random,
)
from api import giphy_commands
from api import media_cache
from api.cached_http import cached_request as _cached_request
from api import stock_commands
from api.prompt_context import (
    clean_crypto_data,
    format_hacker_news_info,
    format_weather_info,
    get_weather_description,
)
from api.prompt_context import build_ai_messages as _build_ai_messages
from api.message_content import (
    extract_message_content,
    extract_message_text,
    extract_poll_text as _extract_poll_text,
    sticker_vision_file_id as _sticker_vision_file_id,
)
from api.media_utils import extract_audio_from_video, measure_audio_duration_seconds
from api.crypto_commands import get_prices as _get_prices
from api import weather_context
from api import hacker_news
from api.admin_reporting import admin_report as _admin_report
from api.system_prompt import build_system_message as _build_system_message
from api import image_processing
from api.provider_errors import (
    extract_error_headers as _extract_error_headers,
    extract_rate_limit_backoff_seconds as _extract_rate_limit_backoff_seconds,
    is_rate_limit_error as _is_rate_limit_error,
    parse_retry_window_seconds as _parse_retry_window_seconds,
    should_try_next_groq_account as _should_try_next_groq_account_after_error,
)
from api import polymarket_commands
from api import dollar_runtime
from api import provider_config
from api import provider_support
from api.memory_compaction import IncrementalSummarySource
from api import memory_compaction
from api import billing_callbacks
from api import summary_runtime
from api import ai_request_runtime
from api import callback_runtime
from api import media_runtime
from api import response_runtime
from api import media_commands
from api.services.redis_helpers import (
    redis_get_json,
    redis_set_json,
    redis_setex_json,
)
from api.services import http_client
from api.services.stale_cache import StaleCache, StaleCacheResult
from api.services.maintenance import (
    REQUEST_CACHE_HISTORY_TTL,
    request_cache_history_key,
)
from api.config import (
    config_redis as _config_config_redis,
    configure as configure_app_config,
    load_bot_config as _config_load_bot_config,
)
from api.ai_billing import (
    AIBillingPack,
    BalanceFormatter,
    build_insufficient_credits_message as _billing_build_insufficient_credits_message,
    build_topup_keyboard as _billing_build_topup_keyboard,
    get_ai_billing_pack as _billing_get_ai_billing_pack,
    get_ai_billing_packs as _billing_get_ai_billing_packs,
    get_ai_onboarding_credits as _billing_get_ai_onboarding_credits,
    maybe_grant_onboarding_credits as _billing_maybe_grant_onboarding_credits,
    parse_topup_payload as _billing_parse_topup_payload,
)
from api.chat_context import (
    extract_numeric_chat_id as _billing_extract_numeric_chat_id,
    extract_user_id as _billing_extract_user_id,
)
from api.ai_pricing import (
    CHAT_OUTPUT_TOKEN_LIMIT,
    AIUsageResult,
    VISION_OUTPUT_TOKEN_LIMIT,
    calculate_billing_for_segments,
    estimate_chat_reserve_credits,
    estimate_message_tokens,
    estimate_vision_reserve_credits,
    ensure_mapping,
    MODEL_PRICING_USD_MICROS,
)
from api.agent_tools import fetch_url_content
from api.constants import ADMIN_CONFIG_DENIAL_MESSAGE, PROMPT_NO_MARKDOWN
from api.providers import OpenRouterProvider, ProviderChain
# Side-effect imports: modules register tools at import time via register_tool()
import api.tools.crypto_prices
import api.tools.calculate
import api.tools.web_fetch
import api.tools.task_set
import api.tools.get_chat_members
from api.tools import get_all_tool_schemas
from api.tool_runtime import ToolRuntime
from api.tools.task_scheduler import (
    list_tasks as _task_list_tasks,
    cancel_task as _task_cancel_task,
    format_task_summary,
)
from api.ai_pipeline import (
    _extract_user_name,
    handle_ai_response as _ai_handle_response,
)
from api.streaming import (
    consume_stream_to_telegram,
    set_streamed_response_metadata,
    stream_to_telegram,
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
from api.feature_catalog import render_ai_capabilities_prompt, render_help_text
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
from api.token_signals import handle_token_signal_callback
from api.price_commands import (
    SUPPORTED_PRICE_SYMBOLS,
    expand_price_tokens,
    find_coin_by_symbol_or_name,
    parse_amount_conversion,
    parse_conversion_only,
    price_query_parameter,
)
from .dollar_commands import sort_dollar_rates
from .rulo_commands import build_rulo_message
from .market_commands import format_market_info
from api.routing_policy import RoutingPolicy
from api.telegram_gateway import (
    TelegramGateway,
    _redact_telegram_tokens,
    _truncate_telegram_text,
    send_typing,
    telegram_request as _telegram_request,
)
from api.telegram_bot_commands import update_bot_commands as _update_bot_commands
from api.message_state import (
    BOT_MESSAGE_META_TTL,
    CHAT_HISTORY_MAX_MESSAGES,
    build_reply_context_text as _state_build_reply_context_text,
    format_user_message as _state_format_user_message,
    fetch_chat_messages_for_compaction as _state_fetch_chat_messages_for_compaction,
    get_bot_message_metadata as _state_get_bot_message_metadata,
    get_chat_compacted_until as _state_get_chat_compacted_until,
    get_chat_history as _state_get_chat_history,
    get_chat_summary as _state_get_chat_summary,
    save_chat_compacted_until as _state_save_chat_compacted_until,
    save_chat_member as _state_save_chat_member,
    save_chat_summary as _state_save_chat_summary,
    save_bot_message_metadata as _state_save_bot_message_metadata,
    save_message_to_redis as _state_save_message_to_redis,
    search_chat_history as _state_search_chat_history,
    truncate_text as _state_truncate_text,
)
from api.logging_config import get_logger
from api.logging_config import format_log_context
from api.link_service import LinkService
from api.random_replies import build_random_reply
from api.services import bcra as bcra_service
from api.services import chat_config_db as chat_config_db_service
from api.services import credits_db as credits_db_service
from api.utils.http import request_with_ssl_fallback
from api.utils.text import sanitize_summary_text
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
TTL_STOCK_SCREENER = 3600  # 1 hour
DOLLAR_FORMATTED_STALE_GRACE = 30 * 60
STABLE_AI_CONTEXT_TTL = 60

_logger = get_logger(__name__)
_BACKGROUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="cache-refresh",
)
atexit.register(_BACKGROUND_REFRESH_EXECUTOR.shutdown, wait=False)

_STABLE_AI_CONTEXT_CACHE: Dict[int, Tuple[int, Dict[str, Any]]] = {}

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


def _parse_timeframe(
    msg_text: str, valid: Mapping[str, Any]
) -> Tuple[str, Optional[str]]:
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


def make_chat_tz(offset: int = -3) -> timezone:
    return timezone(timedelta(hours=offset))
PRIMARY_CHAT_MODEL = "deepseek/deepseek-v4-flash"
SUMMARY_MODEL = "deepseek/deepseek-v4-flash"
SUMMARY_MAX_TOKENS = 2048
COMPACTION_THRESHOLD = 40
COMPACTION_KEEP = 25
COMPACTION_TRUNCATE_LINES = 25
COMPACTION_MAX_SUMMARY_MESSAGES = 200
VISION_MODEL = "google/gemini-3.1-flash-lite-preview"
GROQ_TRANSCRIBE_MODEL = "groq/whisper-large-v3"

OPENROUTER_WEB_SEARCH_MAX_RESULTS = 10
OPENROUTER_WEB_SEARCH_MAX_QUERIES = 3


MESSAGE_BLOCK_PATTERN = re.compile(
    r"(?ms)^MENSAJE:\n(?P<message>.*?)(?:\n\nINSTRUCCIONES:|\Z)"
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
    return polymarket_commands.fetch_live_price(
        token_id,
        cached_request=cached_requests,
        cache_ttl=TTL_POLYMARKET_STREAM,
    )


def _fetch_polymarket_live_prices(token_ids: Sequence[str]) -> Dict[str, float]:
    return polymarket_commands.fetch_live_prices(
        token_ids,
        http_post=http_client.post,
    )


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


def config_redis(
    host: Optional[str] = None,
    port: Optional[Union[int, str]] = None,
    password: Optional[str] = None,
) -> redis.Redis:
    configure_app_config(admin_reporter=globals().get("admin_report"))
    return _config_config_redis(host=host, port=port, password=password)


def load_bot_config() -> Dict[str, Any]:
    configure_app_config(admin_reporter=globals().get("admin_report"))
    return _config_load_bot_config()


def _optional_redis_client(**kwargs: Any) -> Optional[redis.Redis]:
    """Return a Redis client when available, otherwise ``None``."""

    try:
        return config_redis(**kwargs)
    except Exception as error:
        _logger.warning("optional Redis client unavailable: %s", error)
        return None


def _hash_cache_key(prefix: str, payload: Mapping[str, Any]) -> str:
    """Return a stable cache key composed of *prefix* and a SHA-256 hash."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


_link_service = LinkService(
    optional_redis_client=_optional_redis_client,
    hash_cache_key=_hash_cache_key,
    request_fn=request_with_ssl_fallback,
    redis_get_json=redis_get_json,
    redis_setex_json=redis_setex_json,
    extract_video_id=extract_youtube_video_id,
    fetch_transcript=get_youtube_transcript_context,
    logger=_logger,
    format_log_context=format_log_context,
    metadata_ttl=TTL_LINK_METADATA,
    metadata_max_bytes=LINK_METADATA_MAX_BYTES,
    max_links=MAX_LINKS_IN_MESSAGE,
)


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

    _configure_bcra_service()
    return bcra_service.fetch_currency_band_limits(
        list_variables_fn=bcra_list_variables,
        api_get_fn=bcra_api_get,
    )


def get_currency_band_limits() -> Optional[Dict[str, Any]]:
    """Proxy to BCRA service allowing tests to patch the fetcher."""

    _configure_bcra_service()
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


# Provider backoff windows (seconds)
GROQ_BACKOFF_DEFAULT_SECONDS = 60

GROQ_FREE_ACCOUNT = "free"
GROQ_PAID_ACCOUNT = "paid"
GROQ_ACCOUNT_ORDER: Tuple[str, ...] = (GROQ_FREE_ACCOUNT, GROQ_PAID_ACCOUNT)

_ai_provider_request_count: ContextVar[int] = ContextVar(
    "ai_provider_request_count", default=0
)
def _reset_ai_provider_request_count() -> Token[int]:
    return _ai_provider_request_count.set(0)


def _restore_ai_provider_request_count(token: Token[int]) -> None:
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
HACKER_NEWS_RSS_FALLBACK_URL = "https://news.ycombinator.com/rss"
HACKER_NEWS_CACHE_KEY = "context:hacker_news:best"
HACKER_NEWS_MAX_ITEMS = 5


def _get_groq_api_key(account: str) -> Optional[str]:
    return provider_config.get_groq_api_key(account, environment=environ)


def _get_configured_groq_accounts() -> List[str]:
    return provider_config.get_configured_groq_accounts(
        GROQ_ACCOUNT_ORDER,
        get_api_key=_get_groq_api_key,
    )


def _get_openrouter_api_key() -> Optional[str]:
    return provider_config.get_openrouter_api_key(environment=environ)


def _get_openrouter_base_url() -> Optional[str]:
    return provider_config.get_openrouter_base_url(environment=environ)


def _get_openrouter_client(
    *, default_headers: Optional[Mapping[str, str]] = None
) -> Optional[OpenAI]:
    return provider_config.build_openrouter_client(
        get_api_key=_get_openrouter_api_key,
        get_base_url=_get_openrouter_base_url,
        environment=environ,
        client_factory=OpenAI,
        default_headers=default_headers,
    )


def _build_openrouter_web_search_tool() -> Dict[str, Any]:
    return provider_config.build_web_search_tool(
        OPENROUTER_WEB_SEARCH_MAX_RESULTS,
        OPENROUTER_WEB_SEARCH_MAX_QUERIES,
    )


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
    for match in _link_service.url_pattern.finditer(latest_text):
        normalized = _link_service.normalize_detected_url(match.group(1))
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


def _get_cached_media(prefix: str, file_id: str) -> Optional[str]:
    return media_cache.get_cached_media(
        prefix,
        file_id,
        redis_factory=config_redis,
        logger=_logger,
    )


def _cache_media(prefix: str, file_id: str, text: str, ttl: int) -> None:
    media_cache.cache_media(
        prefix,
        file_id,
        text,
        ttl,
        redis_factory=config_redis,
        logger=_logger,
    )


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
def get_cache_history(
    hours_ago: int, request_hash: str, redis_client: redis.Redis
) -> Optional[Dict[str, Any]]:
    timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d-%H")
    cached_data = redis_client.get(request_cache_history_key(timestamp, request_hash))
    if cached_data is None:
        return None
    cache_history = json.loads(cached_data)
    return cache_history if isinstance(cache_history, dict) and "timestamp" in cache_history else None


def cached_requests(
    api_url: str,
    parameters: Optional[Mapping[str, Any]],
    headers: Optional[Mapping[str, Any]],
    expiration_time: int,
    hourly_cache: bool = False,
    get_history: Union[int, bool] = False,
    verify_ssl: bool = True,
) -> Optional[Dict[str, Any]]:
    return _cached_request(
        api_url,
        parameters,
        headers,
        expiration_time,
        hourly_cache=hourly_cache,
        history_hours=get_history,
        verify_ssl=verify_ssl,
        redis_factory=config_redis,
        redis_get_json=redis_get_json,
        redis_set_json=redis_set_json,
        get_history=get_cache_history,
        http_get=http_client.get,
        admin_report=admin_report,
        logger=_logger,
    )


def get_api_or_cache_prices(
    convert_to: str, limit: Optional[int] = None, hourly_cache: bool = False
) -> Optional[Dict[str, Any]]:
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
    jobs = [
        lambda: _fetch_criptoya_dollar_data(hourly_cache=True),
        lambda: get_api_or_cache_prices("ARS", hourly_cache=True),
        lambda: get_api_or_cache_prices("USD", hourly_cache=True),
        get_oil_price,
    ]
    futures = [_BACKGROUND_REFRESH_EXECUTOR.submit(job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as error:
            _logger.warning("refresh_price_caches: cache refresh failed: %s", error)


def _fetch_polymarket_event(
    slug: str,
) -> Optional[Tuple[Dict[str, Any], Optional[int]]]:
    return polymarket_commands.fetch_event(
        slug,
        cached_request=cached_requests,
        cache_ttl=TTL_POLYMARKET,
    )


def _format_polymarket_event_section(
    event: Dict[str, Any],
    header: str,
    filter_prefixes: Sequence[str],
) -> Optional[Tuple[List[str], Optional[int]]]:
    return polymarket_commands.format_event_section(
        event,
        header,
        filter_prefixes,
        fetch_live=_fetch_polymarket_live_price,
    )


def _polymarket_event_top_outcomes(
    event: Dict[str, Any],
    limit: int = 2,
    *,
    fetch_live: Optional[
        Callable[[str], Optional[Tuple[float, Optional[int]]]]
    ] = None,
) -> List[Tuple[str, float]]:
    return polymarket_commands.event_top_outcomes(
        event,
        limit,
        fetch_live=fetch_live,
    )


def _format_usd_compact(value: float) -> str:
    return polymarket_commands.format_usd_compact(value)


def _country_flag(country_code: str) -> str:
    return polymarket_commands.country_flag(country_code)


def _country_code_from_name(name: str) -> str:
    return polymarket_commands.country_code_from_name(name)


def _event_country_flag(event: Dict[str, Any]) -> str:
    return polymarket_commands.event_country_flag(event)


def _flagged_country_name(name: str) -> str:
    return polymarket_commands.flagged_country_name(name)


def get_polymarket_global_elections() -> str:
    return polymarket_commands.get_global_elections(
        cached_request=cached_requests,
        cache_ttl=TTL_POLYMARKET,
        get_top_outcomes=_polymarket_event_top_outcomes,
        fetch_live_prices=_fetch_polymarket_live_prices,
        get_event_flag=_event_country_flag,
        format_liquidity=_format_usd_compact,
    )


def get_polymarket_world_cup_games(timezone_offset: int = -3) -> str:
    return polymarket_commands.get_world_cup_games(
        timezone_offset,
        fetch_winner_event=_fetch_polymarket_event,
        cached_request=cached_requests,
        cache_ttl=TTL_POLYMARKET,
        get_top_outcomes=_polymarket_event_top_outcomes,
        fetch_live=_fetch_polymarket_live_price,
        format_country=_flagged_country_name,
        make_timezone=make_chat_tz,
    )


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
    except (KeyError, TypeError, ValueError):
        return None
    except Exception as error:
        _logger.exception("get_btc_price failed for convert_to=%s: %s", convert_to, error)
        return None


def get_prices(msg_text: str) -> Optional[str]:
    return _get_prices(
        msg_text,
        change_fields=_CMC_CHANGE_FIELD,
        fetch_prices=get_api_or_cache_prices,
        fetch_quotes=_fetch_cmc_quotes,
    )


def format_dollar_rates(
    dollar_rates: List[Dict[str, Any]],
    hours_ago: int,
    band_limits: Optional[Dict[str, Any]] = None,
) -> str:
    return dollar_runtime.format_dollar_rates(
        dollar_rates,
        hours_ago,
        band_limits,
    )


def get_dollar_rates(msg_text: str = "") -> Optional[str]:
    return dollar_runtime.get_dollar_rates(
        msg_text,
        timeframes=_DOLLAR_TIMEFRAME_HOURS,
        get_cache=_get_dollar_snapshot_cache,
        build_text=_build_dollar_rates_text,
        cache_ttl=TTL_DOLLAR,
        stale_grace=DOLLAR_FORMATTED_STALE_GRACE,
        schedule_refresh=_schedule_background_refresh,
        logger=_logger,
    )


def _build_dollar_rates_text(hours_ago: int) -> Optional[str]:
    return dollar_runtime.build_dollar_rates_text(
        hours_ago,
        fetch_dollars=_fetch_criptoya_dollar_data,
        get_tcrm=get_cached_tcrm_100,
        sort_rates=sort_dollar_rates,
        get_band_limits=get_currency_band_limits,
        format_rates=format_dollar_rates,
    )


_DOLLAR_SNAPSHOT_CACHE: Optional[StaleCache] = None


def _get_dollar_snapshot_cache() -> StaleCache:
    global _DOLLAR_SNAPSHOT_CACHE
    if _DOLLAR_SNAPSHOT_CACHE is None:
        _DOLLAR_SNAPSHOT_CACHE = StaleCache(redis_client=config_redis())
    return _DOLLAR_SNAPSHOT_CACHE


def _schedule_background_refresh(fn: Callable[[], None]) -> None:
    _BACKGROUND_REFRESH_EXECUTOR.submit(fn)


def get_devo(msg_text: str) -> str:
    return dollar_runtime.get_devo(
        msg_text,
        fetch_dollars=_fetch_criptoya_dollar_data,
    )


def get_rulo() -> str:
    return dollar_runtime.get_rulo(
        fetch_dollars=_fetch_criptoya_dollar_data,
        cached_request=cached_requests,
        cache_ttl=TTL_DOLLAR,
        build_message=build_rulo_message,
    )


def satoshi() -> str:
    return dollar_runtime.satoshi(get_btc_price=get_btc_price, logger=_logger)


def handle_bcra_variables() -> str:
    return dollar_runtime.handle_bcra_variables(
        get_variables=get_or_refresh_bcra_variables,
        format_variables=format_bcra_variables,
        logger=_logger,
    )


_DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = (
    media_commands.DEFAULT_TRANSCRIPTION_ERROR_MESSAGES
)


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
    return media_commands.transcription_error_message(
        error_code,
        download_message=download_message,
        transcribe_message=transcribe_message,
    )


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
    return media_commands.describe_replied_media(
        replied_msg,
        media_key=media_key,
        extract_file_id=extract_file_id,
        prompt=prompt,
        success_prefix=success_prefix,
        download_error=download_error,
        describe_error=describe_error,
        describe_media=describe_media_by_id,
        sanitize_text=sanitize_summary_text,
    )


def handle_transcribe_with_message_result(
    message: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    return media_commands.handle_transcribe_with_message_result(
        message,
        extract_message_content=extract_message_content,
        transcribe_audio_file=_transcribe_audio_file,
        error_message=_transcription_error_message,
        describe_media=_describe_replied_media,
        sticker_file_id=_sticker_vision_file_id,
        logger=_logger,
    )


def handle_transcribe_with_message(message: Dict[str, Any]) -> str:
    response_text, _billing_segments = handle_transcribe_with_message_result(message)
    return response_text


def handle_transcribe() -> str:
    """Return the marker command handled by the message processor."""
    return "el /transcribe se usa respondiendo a un audio, imagen o sticker"


def powerlaw() -> str:
    today = datetime.now(UTC)
    since = datetime(day=4, month=1, year=2009).replace(tzinfo=UTC)
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
    today = datetime.now(UTC)
    since = datetime(day=9, month=1, year=2009).replace(tzinfo=UTC)
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
    return render_help_text()


def get_oil_price() -> str:
    return stock_commands.get_oil_price(fetch_stock=_fetch_yahoo_stock_price)


def _fetch_yahoo_stock_price(symbol: str) -> Optional[Tuple[float, float]]:
    return stock_commands.fetch_yahoo_stock_price(
        symbol,
        cached_request=cached_requests,
        cache_ttl=TTL_PRICE,
    )


def _fetch_top_stocks_by_market_cap() -> List[str]:
    return stock_commands.fetch_top_stocks_by_market_cap(
        redis_factory=_optional_redis_client,
        redis_get_json=redis_get_json,
        redis_set_json=redis_set_json,
        http_get=http_client.get,
        cache_ttl=TTL_STOCK_SCREENER,
    )


def get_stock_prices(msg_text: str) -> str:
    return stock_commands.get_stock_prices(
        msg_text,
        fetch_stock=_fetch_yahoo_stock_price,
        fetch_top_stocks=_fetch_top_stocks_by_market_cap,
    )


def send_msg(
    chat_id: str,
    msg: str,
    msg_id: str = "",
    buttons: Optional[List[str]] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    return telegram_gateway.send_message(chat_id, msg, msg_id, buttons, reply_markup)


def send_photo(
    chat_id: str,
    photo: bytes,
    *,
    caption: str = "",
    msg_id: str = "",
    reply_markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    return telegram_gateway.send_photo(
        chat_id,
        photo,
        caption=caption,
        msg_id=msg_id,
        reply_markup=reply_markup,
    )


def send_video(
    chat_id: str,
    video: bytes,
    *,
    caption: str = "",
    msg_id: str = "",
    buttons: Optional[List[str]] = None,
) -> Optional[int]:
    return telegram_gateway.send_video(
        chat_id,
        video,
        caption=caption,
        msg_id=msg_id,
        buttons=buttons,
    )


def edit_photo(
    chat_id: str,
    message_id: str,
    photo: bytes,
    *,
    caption: str = "",
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    return telegram_gateway.edit_photo(
        chat_id,
        message_id,
        photo,
        caption=caption,
        reply_markup=reply_markup,
    )


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
    return telegram_gateway.send_animation(chat_id, animation_url, msg_id, caption)


def _fetch_giphy_pool(category: str) -> List[str]:
    return giphy_commands.fetch_giphy_pool(category, logger=_logger)


def _get_giphy_pool(category: str) -> List[str]:
    return giphy_commands.get_giphy_pool(
        category,
        redis_factory=_optional_redis_client,
        fetch_pool=_fetch_giphy_pool,
        logger=_logger,
    )


def _get_random_gif(category: str) -> Optional[str]:
    return giphy_commands.get_random_gif(category, get_pool=_get_giphy_pool)


def get_good_morning() -> str:
    return giphy_commands.get_good_morning(get_gif=_get_random_gif)


def get_good_night() -> str:
    return giphy_commands.get_good_night(get_gif=_get_random_gif)


def admin_report(
    message: str,
    error: Optional[Exception] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    _admin_report(
        message,
        error,
        extra_context,
        send_message=send_msg,
        redact=_redact_telegram_tokens,
    )


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


_BCRA_SERVICE_CONFIG_SIGNATURE: Optional[Tuple[int, int, int, int]] = None


def _configure_bcra_service() -> None:
    global _BCRA_SERVICE_CONFIG_SIGNATURE

    signature = (
        id(cached_requests),
        id(config_redis),
        id(admin_report),
        id(get_cache_history),
    )
    if signature == _BCRA_SERVICE_CONFIG_SIGNATURE:
        return

    bcra_service.configure(
        cached_requests=cached_requests,
        redis_factory=config_redis,
        admin_reporter=admin_report,
        cache_history=get_cache_history,
    )
    _BCRA_SERVICE_CONFIG_SIGNATURE = signature


_configure_bcra_service()


configure_app_config(admin_reporter=admin_report)


def get_weather() -> dict[str, Any]:
    return weather_context.get_weather(
        cached_request=cached_requests,
        cache_ttl=TTL_WEATHER,
        local_timezone=BA_TZ,
        datetime_type=datetime,
        logger=_logger,
    )

def _sanitize_bot_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    return ai_request_runtime.sanitize_bot_message(msg)


def _build_ai_request(
    messages: List[Dict[str, Any]],
    *,
    chat_id: Optional[str] = None,
    user_name: Optional[str] = None,
    user_id: Optional[int] = None,
    timezone_offset: int = -3,
    task_mode: bool = False,
    enable_web_search: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    return ai_request_runtime.build_ai_request(
        messages,
        chat_id=chat_id,
        user_name=user_name,
        user_id=user_id,
        timezone_offset=timezone_offset,
        task_mode=task_mode,
        enable_web_search=enable_web_search,
        sanitize_message=_sanitize_bot_message,
        get_context=_get_stable_ai_context,
        get_prices=get_prices,
        config_redis=config_redis,
        get_tool_schemas=get_all_tool_schemas,
        build_system_message=build_system_message,
        fetch_urls=_fetch_urls_from_latest_message,
    )


def _get_stable_ai_context(timezone_offset: int = -3) -> Dict[str, Any]:
    return ai_request_runtime.get_stable_ai_context(
        timezone_offset,
        cache=_STABLE_AI_CONTEXT_CACHE,
        ttl=STABLE_AI_CONTEXT_TTL,
        now=time.time,
        get_market_context=get_market_context,
        get_weather_context=get_weather_context,
        get_time_context=get_time_context,
        get_hacker_news_context=get_hacker_news_context,
    )


def _inject_image_context(
    messages: List[Dict[str, Any]],
    image_data: Optional[bytes],
    image_file_id: Optional[str],
    response_meta: Optional[Dict[str, Any]],
) -> None:
    ai_request_runtime.inject_image_context(
        messages,
        image_data,
        image_file_id,
        response_meta,
        describe_image=_describe_image_result,
        append_billing_segment=_append_billing_segment,
        logger=_logger,
    )


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
    return ai_request_runtime.ask_ai(
        messages,
        image_data=image_data,
        image_file_id=image_file_id,
        response_meta=response_meta,
        enable_web_search=enable_web_search,
        chat_id=chat_id,
        user_name=user_name,
        user_id=user_id,
        timezone_offset=timezone_offset,
        task_mode=task_mode,
        build_request=_build_ai_request,
        inject_image=_inject_image_context,
        complete=complete_with_providers,
        fallback=get_fallback_response,
        admin_report=admin_report,
        logger=_logger,
    )


def ask_ai_stream(
    messages: List[Dict[str, Any]],
    *,
    enable_web_search: bool = True,
    chat_id: Optional[str] = None,
    user_name: Optional[str] = None,
    user_id: Optional[int] = None,
    timezone_offset: int = -3,
) -> Iterator[Tuple[str, str]]:
    return ai_request_runtime.ask_ai_stream(
        messages,
        enable_web_search=enable_web_search,
        chat_id=chat_id,
        user_name=user_name,
        user_id=user_id,
        timezone_offset=timezone_offset,
        build_request=_build_ai_request,
        stream=stream_with_providers,
    )


def _build_provider_chain() -> ProviderChain:
    openrouter = OpenRouterProvider(
        get_client=_get_openrouter_client,
        admin_report=admin_report,
        increment_request_count=_increment_ai_provider_request_count,
        build_web_search_tool=_build_openrouter_web_search_tool,
        build_usage_result=_build_groq_usage_result,
        extract_usage_map=_extract_groq_usage_map,
        primary_model=PRIMARY_CHAT_MODEL,
        max_tool_rounds=_MAX_TOOL_ROUNDS,
        tool_runtime=_TOOL_RUNTIME,
    )
    return ProviderChain([openrouter])


_provider_chain: Optional[ProviderChain] = None


def get_provider_chain() -> ProviderChain:
    global _provider_chain
    if _provider_chain is None:
        _provider_chain = _build_provider_chain()
    return _provider_chain


def complete_with_providers(
    system_message: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    response_meta: Optional[Dict[str, Any]] = None,
    enable_web_search: bool = True,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    chain = get_provider_chain()
    provider_result = chain.complete(
        system_message,
        messages,
        enable_web_search=enable_web_search,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )
    if provider_result.result:
        if response_meta is not None:
            _append_billing_segment(response_meta, provider_result.result)
        _logger.info("provider response provider=%s", provider_result.provider_name)
        return provider_result.result.text
    return None


def stream_with_providers(
    system_message: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    enable_web_search: bool = True,
    extra_tools: Optional[List[Dict[str, Any]]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[str, str]]:
    chain = get_provider_chain()
    return chain.stream(
        system_message,
        messages,
        enable_web_search=enable_web_search,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )


def get_hacker_news_context(limit: int = HACKER_NEWS_MAX_ITEMS) -> List[Dict[str, Any]]:
    return hacker_news.get_hacker_news_context(
        limit,
        max_items=HACKER_NEWS_MAX_ITEMS,
        cache_key=HACKER_NEWS_CACHE_KEY,
        cache_ttl=TTL_HACKER_NEWS,
        primary_url=HACKER_NEWS_RSS_URL,
        fallback_url=HACKER_NEWS_RSS_FALLBACK_URL,
        redis_factory=_optional_redis_client,
        redis_get_json=redis_get_json,
        redis_setex_json=redis_setex_json,
        request_get=request_with_ssl_fallback,
        logger=_logger,
    )


def get_market_context() -> Dict[str, Any]:
    """Get crypto and dollar market data for the system prompt."""
    market_data = {}

    try:
        # Get crypto prices (reuse unified helper and 5-minute cache)
        api_data = get_api_or_cache_prices("USD", limit=5)
        if api_data and "data" in api_data:
            market_data["crypto"] = clean_crypto_data(api_data["data"])
    except Exception:
        _logger.exception("Error fetching crypto data")

    try:
        # Get dollar rates (reuse 5-minute cache)
        dollar_response = _fetch_criptoya_dollar_data(hourly_cache=False)
        if dollar_response and "data" in dollar_response:
            market_data["dollar"] = dollar_response["data"]
    except Exception:
        _logger.exception("Error fetching dollar data")

    return market_data


def get_weather_context() -> Optional[Dict[str, Any]]:
    return weather_context.get_weather_context(
        get_weather_data=get_weather,
        get_description=get_weather_description,
        logger=_logger,
    )


def get_time_context(timezone_offset: int = -3) -> Dict[str, Any]:
    """Get current time in the chat's configured timezone."""
    current_time = datetime.now(make_chat_tz(timezone_offset))
    return {"datetime": current_time, "formatted": current_time.strftime("%A %d/%m/%Y")}


def _invoke_provider(
    provider_name: str,
    *,
    attempt: Callable[[], Optional[Any]],
    rate_limit_backoff: Optional[int] = None,
    label: Optional[str] = None,
    backoff_key: Optional[str] = None,
) -> Optional[Any]:
    return provider_support.invoke_provider(
        provider_name,
        attempt=attempt,
        rate_limit_backoff=rate_limit_backoff,
        label=label,
        backoff_key=backoff_key,
        is_backoff_active=is_provider_backoff_active,
        get_backoff_remaining=get_provider_backoff_remaining,
        is_rate_limit_error=_is_rate_limit_error,
        extract_backoff_seconds=_extract_rate_limit_backoff_seconds,
        set_backoff=_set_provider_backoff,
        default_backoff=GROQ_BACKOFF_DEFAULT_SECONDS,
    )


def _append_billing_segment(
    response_meta: Optional[Dict[str, Any]],
    result: Optional[AIUsageResult],
) -> None:
    provider_support.append_billing_segment(response_meta, result)


def _log_groq_request_result(
    *,
    label: str,
    scope: str,
    account: str,
    token_count: int,
    audio_seconds: float,
    result: Optional[AIUsageResult],
) -> None:
    provider_support.log_groq_request_result(
        label=label,
        scope=scope,
        account=account,
        token_count=token_count,
        audio_seconds=audio_seconds,
        result=result,
        calculate_billing=calculate_billing_for_segments,
        ensure_mapping=ensure_mapping,
        logger=_logger,
    )


def _extract_groq_usage_map(response: Any) -> Optional[Dict[str, Any]]:
    return provider_support.extract_usage_map(
        response,
        ensure_mapping=ensure_mapping,
    )


def _extract_latest_user_query_info(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, Optional[int]]:
    return provider_support.extract_latest_user_query_info(
        messages,
        extract_message=_extract_message_block_from_prompt,
    )


def estimate_ai_base_reserve_credits(
    messages: List[Dict[str, Any]],
    *,
    extra_input_tokens: int = 0,
    timezone_offset: int = -3,
) -> Tuple[int, Dict[str, Any]]:
    system_message: Optional[Dict[str, Any]] = None
    try:
        context_data = {
            "market": get_market_context(),
            "weather": get_weather_context(),
            "time": get_time_context(timezone_offset),
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
        model=PRIMARY_CHAT_MODEL,
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
    image_url = f"data:image/webp;base64,{image_base64}"
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
    return provider_support.build_usage_result(
        kind=kind,
        text=text,
        model=model,
        response=response,
        audio_seconds=audio_seconds,
        cached=cached,
        metadata=metadata,
        extract_usage=_extract_groq_usage_map,
    )


_MAX_TOOL_ROUNDS = 5
_TOOL_RUNTIME = ToolRuntime()


_summary_logger = get_logger(__name__)


def _call_summary_model(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], int]:
    return summary_runtime.call_summary_model(
        messages,
        get_client=_get_openrouter_client,
        estimate_tokens=estimate_message_tokens,
        estimate_cost=_estimate_summary_cost_usd_micros,
        model=SUMMARY_MODEL,
        max_tokens=SUMMARY_MAX_TOKENS,
        logger=_summary_logger,
    )


def _load_bot_personality() -> str:
    try:
        system_prompt = load_bot_config().get("system_prompt", "")
        return system_prompt if isinstance(system_prompt, str) else ""
    except Exception:
        return ""


def _build_chat_messages(
    bot_personality: str,
    messages: List[Dict[str, Any]],
    prompt_text: str,
    prior_summary: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return summary_runtime.build_chat_messages(
        bot_personality,
        messages,
        prompt_text,
        prior_summary,
    )


def _compact_conversation(
    messages: List[Dict[str, Any]],
    prior_summary: Optional[str] = None,
) -> Tuple[str, int]:
    return summary_runtime.compact_conversation(
        messages,
        prior_summary,
        load_personality=_load_bot_personality,
        call_model=_call_summary_model,
        sanitize_text=sanitize_summary_text,
        no_markdown_prompt=PROMPT_NO_MARKDOWN,
        max_summary_messages=COMPACTION_MAX_SUMMARY_MESSAGES,
        truncate_lines=COMPACTION_TRUNCATE_LINES,
    )


def _build_summary_messages(
    source: IncrementalSummarySource,
    prompt_text: str,
) -> List[Dict[str, Any]]:
    return summary_runtime.build_summary_messages(
        source,
        prompt_text,
        load_personality=_load_bot_personality,
    )


def _build_summary_provider() -> OpenRouterProvider:
    return OpenRouterProvider(
        get_client=_get_openrouter_client,
        admin_report=admin_report,
        increment_request_count=_increment_ai_provider_request_count,
        build_web_search_tool=_build_openrouter_web_search_tool,
        build_usage_result=_build_groq_usage_result,
        extract_usage_map=_extract_groq_usage_map,
        primary_model=SUMMARY_MODEL,
        max_tool_rounds=0,
    )


def _wrap_provider_stream(
    provider_name: str, token_iter: Iterator[str]
) -> Iterator[Tuple[str, str]]:
    return summary_runtime.wrap_provider_stream(
        provider_name,
        token_iter,
        logger=_summary_logger,
    )


def stream_summary_command(
    chat_id: str,
    redis_client: redis.Redis,
    prompt_text: str,
) -> Tuple[Iterator[Tuple[str, str]], Optional[str]]:
    return summary_runtime.stream_summary_command(
        chat_id,
        redis_client,
        prompt_text,
        get_history=get_chat_history,
        prepare_memory=prepare_chat_memory,
        load_personality=_load_bot_personality,
        build_provider=_build_summary_provider,
        sanitize_text=sanitize_summary_text,
        max_tokens=SUMMARY_MAX_TOKENS,
        logger=_summary_logger,
    )


def _build_incremental_summary_source(
    history: List[Dict[str, Any]],
    existing_summary: Optional[str],
    compacted_until: Optional[str],
) -> IncrementalSummarySource:
    return memory_compaction.build_incremental_summary_source(
        history,
        existing_summary,
        compacted_until,
    )


def _resolve_compaction_params(
    threshold: Optional[int] = None,
    keep: Optional[int] = None,
) -> Tuple[int, int]:
    return memory_compaction.resolve_compaction_params(
        threshold,
        keep,
        default_threshold=COMPACTION_THRESHOLD,
        default_keep=COMPACTION_KEEP,
    )


def compact_chat_memory(
    redis_client: Optional[redis.Redis],
    chat_id: Optional[str],
    messages: List[Dict[str, Any]],
    existing_summary: Optional[str],
    compacted_until: Optional[str],
    compact_fn: Callable[[List[Dict[str, Any]], Optional[str]], Tuple[str, int]] = _compact_conversation,
    compaction_threshold: Optional[int] = None,
    compaction_keep: Optional[int] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str], int]:
    compaction_threshold, compaction_keep = _resolve_compaction_params(
        compaction_threshold, compaction_keep
    )
    return memory_compaction.compact_chat_memory(
        redis_client,
        chat_id,
        messages,
        existing_summary,
        compacted_until,
        compact_fn=compact_fn,
        compaction_threshold=compaction_threshold,
        compaction_keep=compaction_keep,
        build_source=_build_incremental_summary_source,
        save_summary=_state_save_chat_summary,
        save_marker=_state_save_chat_compacted_until,
    )


def prepare_chat_memory(
    redis_client: Optional[redis.Redis],
    chat_id: Optional[str],
    chat_history: List[Dict[str, Any]],
    query_text: str,
    reply_to_message_id: Optional[str] = None,
    compaction_threshold: Optional[int] = None,
    compaction_keep: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]], int]:
    compaction_threshold, compaction_keep = _resolve_compaction_params(
        compaction_threshold, compaction_keep
    )
    return memory_compaction.prepare_chat_memory(
        redis_client,
        chat_id,
        chat_history,
        query_text,
        reply_to_message_id=reply_to_message_id,
        compaction_threshold=compaction_threshold,
        compaction_keep=compaction_keep,
        get_summary=_state_get_chat_summary,
        get_marker=_state_get_chat_compacted_until,
        fetch_full_history=_state_fetch_chat_messages_for_compaction,
        compact_memory=compact_chat_memory,
        search_history=_state_search_chat_history,
        admin_report=admin_report,
    )


def _estimate_summary_cost_usd_micros(
    input_tokens: int, output_tokens: int, model: str
) -> int:
    return summary_runtime.estimate_summary_cost_usd_micros(
        input_tokens,
        output_tokens,
        model,
        pricing_by_model=MODEL_PRICING_USD_MICROS,
    )





def get_fallback_response(messages: List[Dict[str, Any]]) -> str:
    """Generate fallback random response"""
    display_name = ""
    if messages:
        last_message = messages[-1]["content"]
        if "Usuario: " in last_message:
            display_name = last_message.split("Usuario: ")[1].split(" ")[0]
    return gen_random(display_name)


def build_system_message(
    context: Dict[str, Any],
    *,
    tools_active: bool = False,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    task_mode: bool = False,
) -> Dict[str, Any]:
    return _build_system_message(
        context,
        tools_active=tools_active,
        tool_schemas=tool_schemas,
        task_mode=task_mode,
        load_config=load_bot_config,
        format_market=format_market_info,
        format_weather=format_weather_info,
        format_news=format_hacker_news_info,
        render_capabilities=render_ai_capabilities_prompt,
    )


def build_ai_messages(
    message: Dict[str, Any],
    chat_history: List[Dict[str, Any]],
    message_text: str,
    reply_context: Optional[str] = None,
    enable_web_search: bool = True,
    summary_text: Optional[str] = None,
    retrieved_messages: Optional[List[Dict[str, Any]]] = None,
    timezone_offset: int = -3,
) -> List[Dict[str, Any]]:
    return _build_ai_messages(
        message,
        chat_history,
        message_text,
        reply_context=reply_context,
        enable_web_search=enable_web_search,
        summary_text=summary_text,
        retrieved_messages=retrieved_messages,
        timezone_offset=timezone_offset,
        make_timezone=make_chat_tz,
        truncate_text=truncate_text,
        build_links_context=_link_service.build_context,
    )


def _noop_command() -> str:
    return ""


def _noop_param_command(_arg: str) -> str:
    return ""


def initialize_commands() -> Dict[str, Tuple[Callable[..., Any], bool, bool]]:
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
            "get_polymarket_global_elections": get_polymarket_global_elections,
            "get_polymarket_world_cup_games": get_polymarket_world_cup_games,
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
            "summary_command": _noop_command,
        }
    )


def truncate_text(text: Optional[str], max_length: int = 4096) -> str:
    return _state_truncate_text(text, max_length)


def save_message_to_redis(
    chat_id: str,
    message_id: str,
    text: str,
    redis_client: redis.Redis,
    *,
    role: Optional[str] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    mentions_bot: bool = False,
) -> None:
    _state_save_message_to_redis(
        chat_id,
        message_id,
        text,
        redis_client,
        admin_reporter=admin_report,
        role=role,
        user_id=user_id,
        username=username,
        reply_to_message_id=reply_to_message_id,
        mentions_bot=mentions_bot,
    )


def save_chat_member(
    redis_client: redis.Redis,
    chat_id: str,
    user_id: Optional[str],
    first_name: str,
    username: str,
) -> None:
    _state_save_chat_member(
        redis_client,
        chat_id,
        user_id,
        first_name,
        username,
    )


def get_chat_history(
    chat_id: str,
    redis_client: redis.Redis,
    max_messages: int = CHAT_HISTORY_MAX_MESSAGES,
) -> List[Dict[str, Any]]:
    return _state_get_chat_history(
        chat_id,
        redis_client,
        admin_reporter=admin_report,
        max_messages=max_messages,
    )


def get_chat_summary(redis_client: redis.Redis, chat_id: str) -> Optional[str]:
    return _state_get_chat_summary(redis_client, chat_id)


def search_chat_history(
    redis_client: redis.Redis,
    chat_id: str,
    query_text: str,
    *,
    reply_to_message_id: Optional[str] = None,
    limit: int = 5,
    exclude_message_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    return _state_search_chat_history(
        redis_client,
        chat_id,
        query_text,
        reply_to_message_id=reply_to_message_id,
        limit=limit,
        exclude_message_ids=exclude_message_ids,
        admin_reporter=admin_report,
    )


def should_gordo_respond(
    commands: Mapping[str, Tuple[Callable[..., Any], bool, bool]],
    command: str,
    message_text: str,
    message: Mapping[str, Any],
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
    commands: Mapping[str, Tuple[Callable[..., Any], bool, bool]],
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
    pack: AIBillingPack,
) -> bool:
    return billing_callbacks.send_stars_invoice(
        chat_id=chat_id,
        user_id=user_id,
        pack=pack,
        format_credits=format_credit_units,
        telegram_request=_telegram_request,
    )


def download_telegram_file(file_id: str) -> Optional[bytes]:
    """Download file from Telegram"""
    return telegram_gateway.download_file(file_id)


def _get_groq_client(
    account: str,
    *,
    default_headers: Optional[Mapping[str, str]] = None,
) -> Optional[OpenAI]:
    return provider_config.build_groq_openai_client(
        account,
        get_api_key=_get_groq_api_key,
        environment=environ,
        client_factory=OpenAI,
        default_headers=default_headers,
    )


def _get_groq_native_client(
    account: str,
) -> Optional[GroqClient]:
    return provider_config.build_groq_native_client(
        account,
        get_api_key=_get_groq_api_key,
        client_factory=GroqClient,
    )


def _execute_groq_request_with_fallback(
    scope: str,
    *,
    label: str,
    token_count: int = 0,
    audio_seconds: float = 0.0,
    attempt: Callable[[str], Optional[AIUsageResult]],
) -> Optional[AIUsageResult]:
    return media_runtime.execute_groq_request_with_fallback(
        scope,
        label=label,
        token_count=token_count,
        audio_seconds=audio_seconds,
        attempt=attempt,
        get_accounts=_get_groq_accounts_for_scope,
        invoke_provider=_invoke_provider,
        get_backoff_key=_get_groq_backoff_key,
        log_result=_log_groq_request_result,
        should_try_next_account=_should_try_next_groq_account_after_error,
        is_backoff_active=is_provider_backoff_active,
        default_backoff_seconds=GROQ_BACKOFF_DEFAULT_SECONDS,
    )


def _describe_image_result(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[AIUsageResult]:
    return media_runtime.describe_image_result(
        image_data,
        user_text,
        file_id,
        use_cache=use_cache,
        get_cached_description=get_cached_description,
        get_client=_get_openrouter_client,
        prepare_image=prepare_vision_image,
        encode_image=encode_image_to_base64,
        increment_request_count=_increment_ai_provider_request_count,
        build_usage_result=_build_groq_usage_result,
        cache_description=cache_description,
        admin_report=admin_report,
        logger=_logger,
        model=VISION_MODEL,
        max_tokens=VISION_OUTPUT_TOKEN_LIMIT,
        no_markdown_prompt=PROMPT_NO_MARKDOWN,
    )


def describe_image_groq(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[str]:
    result = _describe_image_result(
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
    return media_runtime.transcribe_audio_openrouter_result(
        audio_data,
        file_id,
        get_client=_get_openrouter_client,
        increment_request_count=_increment_ai_provider_request_count,
        build_usage_result=_build_groq_usage_result,
        model=OPENROUTER_TRANSCRIBE_MODEL,
    )


def _transcribe_audio_result(
    audio_data: bytes,
    file_id: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[AIUsageResult]:
    return media_runtime.transcribe_audio_result(
        audio_data,
        file_id,
        use_cache=use_cache,
        get_cached_transcription=get_cached_transcription,
        measure_duration=measure_audio_duration_seconds,
        get_native_client=_get_groq_native_client,
        build_usage_result=_build_groq_usage_result,
        execute_with_fallback=_execute_groq_request_with_fallback,
        openrouter_fallback=_transcribe_audio_openrouter_result,
        cache_transcription=cache_transcription,
        model=GROQ_TRANSCRIBE_MODEL,
    )


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
    return media_runtime.process_media_with_cache(
        file_id=file_id,
        use_cache=use_cache,
        cache_lookup=cache_lookup,
        processor=processor,
        downloader=downloader or download_telegram_file,
        measure_duration=measure_audio_duration_seconds,
        failure_code=failure_code,
        logger=_logger,
    )


def transcribe_file_by_id(
    file_id: str, use_cache: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    return media_runtime.transcribe_file_by_id(
        file_id,
        use_cache,
        get_cached_transcription=get_cached_transcription,
        download_file=download_telegram_file,
        measure_duration=measure_audio_duration_seconds,
        extract_audio=extract_audio_from_video,
        transcribe=_transcribe_audio_result,
        process_media=_process_media_with_cache,
        logger=_logger,
    )


def describe_media_by_id(
    file_id: str, prompt: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    return media_runtime.describe_media_by_id(
        file_id,
        prompt,
        get_cached_description=get_cached_description,
        download_file=download_telegram_file,
        describe_image=_describe_image_result,
        process_media=_process_media_with_cache,
    )


def resize_image_if_needed(image_data: bytes, max_size: int = 512) -> bytes:
    return image_processing.resize_image_if_needed(
        image_data,
        max_size,
        image_module=Image,
    )


def prepare_vision_image(
    image_data: bytes, max_size: int = 512
) -> Optional[Tuple[bytes, str]]:
    return image_processing.prepare_vision_image(
        image_data,
        max_size,
        image_module=Image,
        logger=_logger,
    )


def encode_image_to_base64(image_data: bytes) -> str:
    return image_processing.encode_image_to_base64(image_data)


def parse_command(message_text: str, bot_name: str) -> Tuple[str, str]:
    return _command_parse_command(message_text, bot_name)


def _log_config_event(message: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    log_entry: Dict[str, Any] = {"scope": "config", "message": message}
    if extra:
        for key, value in extra.items():
            log_entry[key] = value
    _logger.info("config event: %s", json.dumps(log_entry, ensure_ascii=False, default=str))


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


def _guard_callback(
    callback_id: Optional[str],
    condition: bool,
    *,
    text: Optional[str] = None,
    show_alert: bool = False,
) -> bool:
    if condition:
        if callback_id:
            _answer_callback_query(callback_id, text=text, show_alert=show_alert)
        return True
    return False


def _billing_unavailable_alert_text() -> str:
    return "el cobro de ia está hecho pelota, avisale al admin"


def _billing_unavailable_message_text() -> str:
    return billing_callbacks.billing_unavailable_message()


def _billing_is_available() -> bool:
    return bool(credits_db_service.is_configured())


def handle_topup_callback(callback_query: Dict[str, Any]) -> None:
    billing_callbacks.handle_topup_callback(
        callback_query,
        guard_callback=_guard_callback,
        billing_available=_billing_is_available,
        get_pack=get_ai_billing_pack,
        send_invoice=_send_stars_invoice,
        answer_callback=_answer_callback_query,
        unavailable_alert=_billing_unavailable_alert_text,
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
    billing_callbacks.handle_pre_checkout_query(
        pre_checkout_query,
        billing_available=_billing_is_available,
        answer_query=_answer_pre_checkout_query,
        unavailable_alert=_billing_unavailable_alert_text,
        parse_payload=_parse_topup_payload,
        get_pack=get_ai_billing_pack,
    )


def handle_successful_payment_message(message: Dict[str, Any]) -> str:
    return billing_callbacks.handle_successful_payment(
        message,
        billing_available=_billing_is_available,
        unavailable_message=_billing_unavailable_message_text,
        send_message=send_msg,
        extract_user_id=_extract_user_id,
        parse_payload=_parse_topup_payload,
        get_pack=get_ai_billing_pack,
        record_payment=credits_db_service.record_star_payment,
        admin_report=admin_report,
        format_credits=format_credit_units,
    )


def handle_task_callback(callback_query: Dict[str, Any]) -> None:
    callback_runtime.handle_task_callback(
        callback_query,
        guard_callback=_guard_callback,
        list_tasks=_task_list_tasks,
        cancel_task=_task_cancel_task,
        is_group_chat_type=is_group_chat_type,
        config_redis=config_redis,
        is_chat_admin=is_chat_admin,
        answer_callback=_answer_callback_query,
        build_tasks_message=_build_tasks_message,
        edit_message=edit_message,
        logger=_logger,
    )


def edit_message(
    chat_id: str,
    message_id: int,
    text: str,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": _truncate_telegram_text(text),
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    payload_response, error = _telegram_request(
        "editMessageText", method="POST", json_payload=payload
    )
    return error is None and bool(payload_response)


def handle_callback_query(callback_query: Dict[str, Any]) -> None:
    callback_runtime.handle_callback_query(
        callback_query,
        guard_callback=_guard_callback,
        handle_topup=handle_topup_callback,
        handle_task=handle_task_callback,
        handle_signal=handle_token_signal_callback,
        config_redis=config_redis,
        delete_msg=delete_msg,
        edit_photo=edit_photo,
        is_chat_admin=is_chat_admin,
        answer_callback=_answer_callback_query,
        admin_report=admin_report,
        is_group_chat_type=is_group_chat_type,
        send_msg=send_msg,
        report_unauthorized=_report_unauthorized_config_attempt,
        denial_message=ADMIN_CONFIG_DENIAL_MESSAGE,
        get_chat_config=get_chat_config,
        set_chat_config=set_chat_config,
        coerce_bool=coerce_bool,
        log_config_event=_log_config_event,
        timezone_offset_min=TIMEZONE_OFFSET_MIN,
        timezone_offset_max=TIMEZONE_OFFSET_MAX,
        build_config_text=build_config_text,
        build_config_keyboard=build_config_keyboard,
        edit_message=edit_message,
    )


def format_user_message(
    message: Dict[str, Any],
    message_text: str,
    reply_context: Optional[str] = None,
) -> str:
    return _state_format_user_message(message, message_text, reply_context)


def _build_message_handler_deps() -> MessageHandlerDeps:
    ai_svc = build_ai_service(
        credits_db_service=credits_db_service,
        get_chat_history=get_chat_history,
        prepare_chat_memory=prepare_chat_memory,
        build_ai_messages=build_ai_messages,
        check_provider_available=check_provider_available,
        has_openrouter_fallback=has_openrouter_fallback,
        handle_rate_limit=handle_rate_limit,
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
        stream_summary_command=stream_summary_command,
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
            link_service=_link_service,
            should_gordo_respond=should_gordo_respond,
            is_group_chat_type=is_group_chat_type,
        ),
        io=MessageIODeps(
            send_msg=send_msg,
            send_animation=send_animation,
            send_photo=send_photo,
            send_video=send_video,
            delete_msg=delete_msg,
            edit_message=_edit_message_for_stream,
            admin_report=admin_report,
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=lambda redis_client, chat_id, message_id: (
                _state_get_bot_message_metadata(
                    redis_client,
                    chat_id,
                    message_id,
                    admin_reporter=admin_report,
                    decode_redis_value=decode_redis_value,
                )
            ),
            save_bot_message_metadata=lambda redis_client, chat_id, message_id, metadata: (
                _state_save_bot_message_metadata(
                    redis_client,
                    chat_id,
                    message_id,
                    metadata,
                    admin_reporter=admin_report,
                )
            ),
            build_reply_context_text=lambda message: _state_build_reply_context_text(
                message,
                extract_message_text_fn=extract_message_text,
            ),
            format_user_message=_state_format_user_message,
            save_message_to_redis=save_message_to_redis,
            save_chat_member=save_chat_member,
        ),
        ai=MessageAIDeps(
            ai_service=ai_svc,
            balance_formatter=BalanceFormatter(credits_db_service),
            handle_ai_stream=handle_ai_stream_response,
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


def handle_msg(message: Dict[str, Any]) -> str:
    return _handle_msg_impl(message, _build_message_handler_deps())


def handle_rate_limit(chat_id: str, message: Dict[str, Any]) -> str:
    return response_runtime.handle_rate_limit(
        chat_id,
        message,
        telegram_token=environ.get("TELEGRAM_TOKEN"),
        send_typing=send_typing,
        sleep=time.sleep,
        random_delay=random.uniform,
        build_random_reply=build_random_reply,
        gen_random=gen_random,
    )


def handle_ai_response(
    chat_id: str,
    handler_func: Callable[..., str],
    messages: List[Dict[str, Any]],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    context_texts: Optional[Sequence[Optional[str]]] = None,
    user_identity: Optional[str] = None,
    response_meta: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    timezone_offset: int = -3,
    reply_to_message_id: Optional[str] = None,
) -> str:
    return response_runtime.handle_ai_response(
        chat_id,
        handler_func,
        messages,
        stream_handler=handle_ai_stream_response,
        image_data=image_data,
        image_file_id=image_file_id,
        context_texts=context_texts,
        user_identity=user_identity,
        response_meta=response_meta,
        user_id=user_id,
        timezone_offset=timezone_offset,
        reply_to_message_id=reply_to_message_id,
        extract_user_name=_extract_user_name,
        handle_response=_ai_handle_response,
        send_typing=send_typing,
        telegram_token=environ.get("TELEGRAM_TOKEN"),
        reset_request_count=_reset_ai_provider_request_count,
        restore_request_count=_restore_ai_provider_request_count,
        get_request_count=_get_ai_provider_request_count,
    )


def _send_message_for_stream(
    chat_id: str, text: str, reply_to_message_id: Optional[str] = None
) -> Optional[int]:
    return response_runtime.send_message_for_stream(
        chat_id,
        text,
        reply_to_message_id,
        send_message=send_msg,
        logger=_logger,
    )


def _edit_message_for_stream(chat_id: str, text: str, message_id: str) -> None:
    response_runtime.edit_message_for_stream(
        chat_id,
        text,
        message_id,
        edit_message=edit_message,
        logger=_logger,
    )


def handle_ai_stream_response(
    messages: List[Dict[str, Any]],
    *,
    response_meta: Optional[Dict[str, Any]] = None,
    chat_id: Optional[str] = None,
    user_id: Optional[int] = None,
    user_name: Optional[str] = None,
    timezone_offset: int = -3,
    reply_to_message_id: Optional[str] = None,
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
    **_: Any,
) -> str:
    return response_runtime.handle_ai_stream_response(
        messages,
        response_meta=response_meta,
        chat_id=chat_id,
        user_id=user_id,
        user_name=user_name,
        timezone_offset=timezone_offset,
        reply_to_message_id=reply_to_message_id,
        image_data=image_data,
        image_file_id=image_file_id,
        inject_image_context=_inject_image_context,
        telegram_token=environ.get("TELEGRAM_TOKEN"),
        send_typing=send_typing,
        ask_ai_stream=ask_ai_stream,
        consume_stream=consume_stream_to_telegram,
        send_stream_message=_send_message_for_stream,
        edit_stream_message=_edit_message_for_stream,
        ask_ai=ask_ai,
        gen_random=gen_random,
        set_stream_metadata=set_streamed_response_metadata,
    )


def update_telegram_bot_commands() -> bool:
    """Update the bot's command menu in Telegram automatically.

    Builds command list from COMMAND_GROUPS and calls setMyCommands.
    """
    token = environ.get("TELEGRAM_TOKEN")
    if not token:
        _logger.warning("telegram commands: TELEGRAM_TOKEN not set, cannot update")
        return False
    try:
        return _update_bot_commands(
            token=token,
            request_fn=_telegram_request,
            command_groups=COMMAND_GROUPS,
            logger=_logger.info,
        )
    except Exception as e:
        _logger.warning("telegram commands: exception updating: %s", e)
        return False
