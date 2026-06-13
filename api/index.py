from __future__ import annotations

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
from api.providers.backoff import (
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
from api.bot.general_commands import (
    convert_base,
    convert_to_command,
    gen_random,
    get_instance_name,
    get_timestamp,
    is_japanese_text,
    romanize_japanese,
    select_random,
)
from api.bot import giphy as giphy_commands
from api.bot.giphy import GiphyService
from api.media import cache as media_cache
from api.media.cache import MediaCacheService
from api.markets import stocks as stock_commands
from api.markets.stocks import StockService
from api.ai.prompt_context import (
    clean_crypto_data,
    format_hacker_news_info,
    format_weather_info,
    get_weather_description,
)
from api.ai.prompt_context import build_ai_messages as _build_ai_messages
from api.bot.message_content import (
    extract_message_content,
    extract_message_text,
    extract_poll_text as _extract_poll_text,
    sticker_vision_file_id as _sticker_vision_file_id,
)
from api.media.utils import extract_audio_from_video, measure_audio_duration_seconds
from api.markets import weather as weather_context
from api.markets import hacker_news
from api.admin.service import AdminService
from api.application import ApplicationRuntime
from api.cache.service import CacheService
from api.ai.system_prompt import build_system_message as _build_system_message
from api.media import images as image_processing
from api.media.images import ImageService
from api.providers.errors import (
    extract_error_headers as _extract_error_headers,
    extract_rate_limit_backoff_seconds as _extract_rate_limit_backoff_seconds,
    is_rate_limit_error as _is_rate_limit_error,
    parse_retry_window_seconds as _parse_retry_window_seconds,
    should_try_next_groq_account as _should_try_next_groq_account_after_error,
)
from api.providers.service import ProviderService
from api.markets import polymarket as polymarket_commands
from api.markets.polymarket import PolymarketService
from api.markets.price import PriceService
from api.markets import dollar as dollar_runtime
from api.markets.dollar import DollarService
from api.providers import config as provider_config
from api.providers import support as provider_support
from api.billing.service import BillingService
from api.memory.summary import SummaryService, SummaryServiceDeps
from api.ai.request_runtime import AIRequestService, AIRequestServiceDeps
from api.bot import callbacks as callback_runtime
from api.media.runtime import MediaService, MediaServiceDeps
from api.bot.responses import ResponseService, ResponseServiceDeps
from api.media import commands as media_commands
from api.services.redis_helpers import (
    redis_get_json,
    redis_set_json,
    redis_setex_json,
)
from api.services import http_client
from api.services.stale_cache import StaleCache, StaleCacheResult
from api.billing.ai import (
    BalanceFormatter,
)
from api.bot.chat_context import (
    extract_numeric_chat_id as _billing_extract_numeric_chat_id,
    extract_user_id as _billing_extract_user_id,
)
from api.ai.pricing import (
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
from api.links.agent_tools import fetch_url_content
from api.core.constants import ADMIN_CONFIG_DENIAL_MESSAGE, PROMPT_NO_MARKDOWN
from api.providers import OpenRouterProvider, ProviderChain
# Side-effect imports: modules register tools at import time via register_tool()
import api.tools.crypto_prices
import api.tools.calculate
import api.tools.web_fetch
import api.tools.task_set
import api.tools.get_chat_members
from api.tools import get_all_tool_schemas
from api.tools.runtime import ToolRuntime
from api.tasks.scheduler import (
    list_tasks as _task_list_tasks,
    cancel_task as _task_cancel_task,
    format_task_summary,
)
from api.ai.pipeline import (
    _extract_user_name,
    handle_ai_response as _ai_handle_response,
)
from api.bot.streaming import (
    consume_stream_to_telegram,
    set_streamed_response_metadata,
    stream_to_telegram,
)
from api.bot.chat_settings import (
    TIMEZONE_OFFSET_MAX,
    TIMEZONE_OFFSET_MIN,
    build_config_keyboard,
    build_config_text,
    coerce_bool,
    decode_redis_value,
    is_group_chat_type,
)
from api.bot.chat_config_service import build_chat_config_service
from api.storage.chat_config_repository import build_chat_config_repository
from api.billing.credit_units import format_credit_units
from api.bot.command_registry import (
    COMMAND_GROUPS,
    build_command_registry as _build_command_registry,
    parse_command as _command_parse_command,
    should_auto_process_media as _command_should_auto_process_media,
    should_gordo_respond as _command_should_gordo_respond,
)
from api.bot.feature_catalog import render_ai_capabilities_prompt, render_help_text
from api.bot.message_handler import (
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
from api.ai.service import build_ai_service
from api.markets.token_signals import handle_token_signal_callback
from api.markets.price_commands import (
    SUPPORTED_PRICE_SYMBOLS,
    expand_price_tokens,
    find_coin_by_symbol_or_name,
    parse_amount_conversion,
    parse_conversion_only,
    price_query_parameter,
)
from api.markets.dollar_commands import sort_dollar_rates
from api.markets.rulo import build_rulo_message
from api.markets.context import format_market_info
from api.bot.routing import RoutingPolicy
from api.bot.telegram import (
    TelegramGateway,
    _redact_telegram_tokens,
    _truncate_telegram_text,
    send_typing,
    telegram_request as _telegram_request,
)
from api.bot.telegram_commands import update_bot_commands as _update_bot_commands
from api.memory.state import (
    BOT_MESSAGE_META_TTL,
    CHAT_HISTORY_MAX_MESSAGES,
    MessageStateService,
)
from api.core.logging import get_logger
from api.core.logging import format_log_context
from api.core.config_runtime import ConfigRuntime
from api.links.service import LinkService
from api.ai.random_replies import build_random_reply
from api.services import bcra as bcra_service
from api.services.bcra import BCRAService
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
_config_runtime = ConfigRuntime(_logger)
telegram_gateway = TelegramGateway(_telegram_request)

# index.py is the composition root: it creates each long-lived service once,
# wires its dependencies, and later exposes the finished graph as app_runtime.
_media_cache_service = MediaCacheService(
    config=_config_runtime,
    logger=_logger,
    default_ttl=TTL_MEDIA_CACHE,
)
_image_service = ImageService(
    image_module=Image,
    logger=_logger,
)


def _log_config_event_adapter(
    message: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    _log_config_event(message, extra)


_admin_service = AdminService(
    telegram=telegram_gateway,
    config=_config_runtime,
    log_event=_log_config_event_adapter,
    redis_get_json=redis_get_json,
    redis_setex_json=redis_setex_json,
)
_config_runtime.set_admin_reporter(_admin_service.report)
_message_state_service = MessageStateService(
    admin_reporter=_admin_service.report,
    decode_redis_value=decode_redis_value,
    extract_message_text=extract_message_text,
)


_BACKGROUND_REFRESH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="cache-refresh",
)
atexit.register(_BACKGROUND_REFRESH_EXECUTOR.shutdown, wait=False)

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
OPENROUTER_TRANSCRIBE_MODEL = "google/gemini-3.1-flash-lite-preview"

OPENROUTER_WEB_SEARCH_MAX_RESULTS = 10
OPENROUTER_WEB_SEARCH_MAX_QUERIES = 3
_MAX_TOOL_ROUNDS = 5
_TOOL_RUNTIME = ToolRuntime()


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


def _hash_cache_key(prefix: str, payload: Mapping[str, Any]) -> str:
    """Return a stable cache key composed of *prefix* and a SHA-256 hash."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


_link_service = LinkService(
    optional_redis_client=_config_runtime.optional_redis,
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


_parse_currency_band_rows = bcra_service._parse_currency_band_rows


# Provider backoff windows (seconds)
GROQ_BACKOFF_DEFAULT_SECONDS = 60

GROQ_FREE_ACCOUNT = "free"
GROQ_PAID_ACCOUNT = "paid"
GROQ_ACCOUNT_ORDER: Tuple[str, ...] = (GROQ_FREE_ACCOUNT, GROQ_PAID_ACCOUNT)

HACKER_NEWS_RSS_URL = "https://hnrss.org/best"
HACKER_NEWS_RSS_FALLBACK_URL = "https://news.ycombinator.com/rss"
HACKER_NEWS_CACHE_KEY = "context:hacker_news:best"
HACKER_NEWS_MAX_ITEMS = 5


_provider_service = ProviderService(
    environment=environ,
    admin_report=_admin_service.report,
    logger=_logger,
    tool_runtime=_TOOL_RUNTIME,
    primary_model=PRIMARY_CHAT_MODEL,
    account_order=GROQ_ACCOUNT_ORDER,
    default_backoff_seconds=GROQ_BACKOFF_DEFAULT_SECONDS,
    web_search_max_results=OPENROUTER_WEB_SEARCH_MAX_RESULTS,
    web_search_max_queries=OPENROUTER_WEB_SEARCH_MAX_QUERIES,
    max_tool_rounds=_MAX_TOOL_ROUNDS,
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


def refresh_price_caches() -> None:
    """Refresh all price caches and store hourly snapshots for change calculations."""
    jobs = [
        lambda: _dollar_service.fetch_dollar_data(hourly_cache=True),
        lambda: _price_service.get_api_prices("ARS", hourly_cache=True),
        lambda: _price_service.get_api_prices("USD", hourly_cache=True),
        _stock_service.get_oil_price,
    ]
    futures = [_BACKGROUND_REFRESH_EXECUTOR.submit(job) for job in jobs]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as error:
            _logger.warning("refresh_price_caches: cache refresh failed: %s", error)


def _schedule_background_refresh(fn: Callable[[], None]) -> None:
    _BACKGROUND_REFRESH_EXECUTOR.submit(fn)


_DEFAULT_TRANSCRIPTION_ERROR_MESSAGES = (
    media_commands.DEFAULT_TRANSCRIPTION_ERROR_MESSAGES
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
        describe_media=_media_service.describe_media,
        sanitize_text=sanitize_summary_text,
    )


def handle_transcribe_with_message_result(
    message: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    return media_commands.handle_transcribe_with_message_result(
        message,
        extract_message_content=extract_message_content,
        transcribe_audio_file=_media_service.transcribe_file,
        error_message=media_commands.transcription_error_message,
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

    price = _price_service.get_btc_price("USD")
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

    price = _price_service.get_btc_price("USD")
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


def get_weather() -> dict[str, Any]:
    return weather_context.get_weather(
        cached_request=_cache_service.request,
        cache_ttl=TTL_WEATHER,
        local_timezone=BA_TZ,
        datetime_type=datetime,
        logger=_logger,
    )

def get_hacker_news_context(limit: int = HACKER_NEWS_MAX_ITEMS) -> List[Dict[str, Any]]:
    return hacker_news.get_hacker_news_context(
        limit,
        max_items=HACKER_NEWS_MAX_ITEMS,
        cache_key=HACKER_NEWS_CACHE_KEY,
        cache_ttl=TTL_HACKER_NEWS,
        primary_url=HACKER_NEWS_RSS_URL,
        fallback_url=HACKER_NEWS_RSS_FALLBACK_URL,
        redis_factory=_config_runtime.optional_redis,
        redis_get_json=redis_get_json,
        redis_setex_json=redis_setex_json,
        request_get=request_with_ssl_fallback,
        logger=_logger,
    )


def get_market_context() -> Dict[str, Any]:
    """Get crypto and dollar market data for the system prompt."""
    market_data = {}

    try:
        # Reuse command caches so prompt enrichment adds no duplicate API calls.
        api_data = _price_service.get_api_prices("USD", limit=5)
        if api_data and "data" in api_data:
            market_data["crypto"] = clean_crypto_data(api_data["data"])
    except Exception:
        _logger.exception("Error fetching crypto data")

    try:
        dollar_response = _dollar_service.fetch_dollar_data(hourly_cache=False)
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
    image_base64 = _image_service.encode(image_data)
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


_media_service = MediaService(
    MediaServiceDeps(
        cache=_media_cache_service,
        telegram=telegram_gateway,
        images=_image_service,
        get_openrouter_client=_provider_service.get_openrouter_client,
        get_groq_native_client=_provider_service.get_groq_native_client,
        get_groq_accounts=_provider_service.get_groq_accounts,
        invoke_provider=_provider_service.invoke,
        get_backoff_key=_provider_service.get_groq_backoff_key,
        log_result=_provider_service.log_groq_request_result,
        should_try_next_account=_should_try_next_groq_account_after_error,
        is_backoff_active=_provider_service.is_backoff_active,
        increment_request_count=_provider_service.increment_request_count,
        build_usage_result=_provider_service.build_usage_result,
        admin_report=_admin_service.report,
        measure_duration=measure_audio_duration_seconds,
        extract_audio=extract_audio_from_video,
        logger=_logger,
        vision_model=VISION_MODEL,
        vision_max_tokens=VISION_OUTPUT_TOKEN_LIMIT,
        transcribe_model=GROQ_TRANSCRIBE_MODEL,
        openrouter_transcribe_model=OPENROUTER_TRANSCRIBE_MODEL,
        no_markdown_prompt=PROMPT_NO_MARKDOWN,
        default_backoff_seconds=GROQ_BACKOFF_DEFAULT_SECONDS,
    )
)


_summary_logger = get_logger(__name__)


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
        load_config=_config_runtime.load_bot_config,
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
        truncate_text=_message_state_service.truncate_text,
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
            "ask_ai": _ai_request_service.ask,
            "config_command": _noop_command,
            "convert_base": convert_base,
            "select_random": select_random,
            "get_prices": _price_service.get_prices,
            "get_dollar_rates": _dollar_service.get_rates,
            "get_oil_price": _stock_service.get_oil_price,
            "get_stock_prices": _stock_service.get_stock_prices,
            "get_polymarket_global_elections": (
                _polymarket_service.get_global_elections
            ),
            "get_polymarket_world_cup_games": (
                _polymarket_service.get_world_cup_games
            ),
            "get_rulo": _dollar_service.get_rulo,
            "get_devo": _dollar_service.get_devo,
            "powerlaw": powerlaw,
            "rainbow": rainbow,
            "satoshi": _dollar_service.satoshi,
            "get_timestamp": get_timestamp,
            "convert_to_command": convert_to_command,
            "get_instance_name": get_instance_name,
            "get_help": get_help,
            "handle_transcribe": handle_transcribe,
            "handle_bcra_variables": _dollar_service.handle_bcra_variables,
            "topup_command": _noop_command,
            "balance_command": _noop_command,
            "printcredits_command": _noop_param_command,
            "creditlog_command": _noop_param_command,
            "transfer_command": _noop_param_command,
            "get_good_morning": _giphy_service.get_good_morning,
            "get_good_night": _giphy_service.get_good_night,
            "tasks_command": tasks_command,
            "summary_command": _noop_command,
        }
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

    if _billing_service.fetch_balance("user", user_id) > 0:
        return True

    chat = cast(Mapping[str, Any], message.get("chat") or {})
    if not is_group_chat_type(str(chat.get("type") or "")):
        return False

    chat_id = _extract_numeric_chat_id(str(chat.get("id") or ""))
    if chat_id is None:
        return False

    return _billing_service.fetch_balance("chat", chat_id) > 0


_extract_numeric_chat_id = _billing_extract_numeric_chat_id
_extract_user_id = _billing_extract_user_id


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
        load_bot_config_fn=_config_runtime.load_bot_config,
    )


# Business services share the lower-level config, cache, provider, and state
# objects created above. Handlers use these services instead of rebuilding
# integrations for every Telegram update.
_cache_service = CacheService(
    config=_config_runtime,
    admin=_admin_service,
    logger=_logger,
)
_price_service = PriceService(
    cache=_cache_service,
    environment=environ,
    logger=_logger,
    cache_ttl=TTL_PRICE,
)
_stock_service = StockService(
    cache=_cache_service,
    config=_config_runtime,
    price_cache_ttl=TTL_PRICE,
    screener_cache_ttl=TTL_STOCK_SCREENER,
)
_giphy_service = GiphyService(
    config=_config_runtime,
    logger=_logger,
)
_polymarket_service = PolymarketService(
    cache=_cache_service,
    cache_ttl=TTL_POLYMARKET,
    stream_cache_ttl=TTL_POLYMARKET_STREAM,
    make_timezone=make_chat_tz,
)
_bcra_service = BCRAService(
    cache=_cache_service,
    config=_config_runtime,
    admin=_admin_service,
)
_dollar_service = DollarService(
    cache=_cache_service,
    config=_config_runtime,
    logger=_logger,
    timeframes=_DOLLAR_TIMEFRAME_HOURS,
    cache_ttl=TTL_DOLLAR,
    stale_grace=DOLLAR_FORMATTED_STALE_GRACE,
    schedule_refresh=_schedule_background_refresh,
    get_tcrm=_bcra_service.get_cached_tcrm_100,
    get_band_limits=_bcra_service.get_currency_band_limits,
    get_btc_price=_price_service.get_btc_price,
    get_bcra_variables=_bcra_service.get_or_refresh_variables,
    format_bcra_variables=_bcra_service.format_variables,
)
_chat_config_service = build_chat_config_service(
    repository=build_chat_config_repository(),
    admin_reporter=_admin_service.report,
    log_event=_log_config_event,
)
_summary_service = SummaryService(
    SummaryServiceDeps(
        state=_message_state_service,
        config=_config_runtime,
        provider=_provider_service,
        estimate_tokens=estimate_message_tokens,
        sanitize_text=sanitize_summary_text,
        logger=_summary_logger,
        model=SUMMARY_MODEL,
        max_tokens=SUMMARY_MAX_TOKENS,
        compaction_threshold=COMPACTION_THRESHOLD,
        compaction_keep=COMPACTION_KEEP,
        max_summary_messages=COMPACTION_MAX_SUMMARY_MESSAGES,
        truncate_lines=COMPACTION_TRUNCATE_LINES,
        no_markdown_prompt=PROMPT_NO_MARKDOWN,
        pricing_by_model=MODEL_PRICING_USD_MICROS,
    )
)
_ai_request_service = AIRequestService(
    AIRequestServiceDeps(
        get_market_context=get_market_context,
        get_weather_context=get_weather_context,
        get_time_context=get_time_context,
        get_hacker_news_context=get_hacker_news_context,
        get_prices=_price_service.get_prices,
        config_redis=_config_runtime.redis,
        get_tool_schemas=get_all_tool_schemas,
        build_system_message=build_system_message,
        fetch_urls=_fetch_urls_from_latest_message,
        describe_image=_media_service.describe_image_result,
        append_billing_segment=_provider_service.append_billing_segment,
        complete=_provider_service.complete,
        stream=_provider_service.stream,
        fallback=get_fallback_response,
        admin_report=_admin_service.report,
        logger=_logger,
        stable_context_ttl=STABLE_AI_CONTEXT_TTL,
        now=time.time,
    )
)
_ROUTING_POLICY = build_routing_policy()


def handle_config_command(
    chat_id: str, chat_type: str = ""
) -> Tuple[str, Dict[str, Any]]:
    redis_client = _config_runtime.redis()
    config = _chat_config_service.get_chat_config(redis_client, chat_id)
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


_billing_service = BillingService(
    credits=credits_db_service,
    admin_report=_admin_service.report,
    telegram=telegram_gateway,
    telegram_request=_telegram_request,
    guard_callback=_guard_callback,
    answer_callback=_answer_callback_query,
    answer_pre_checkout=_answer_pre_checkout_query,
    extract_user_id=_extract_user_id,
)


def handle_task_callback(callback_query: Dict[str, Any]) -> None:
    callback_runtime.handle_task_callback(
        callback_query,
        deps=callback_runtime.TaskCallbackDeps(
            guard_callback=_guard_callback,
            list_tasks=_task_list_tasks,
            cancel_task=_task_cancel_task,
            is_group_chat_type=is_group_chat_type,
            config_redis=_config_runtime.redis,
            is_chat_admin=_admin_service.is_chat_admin,
            answer_callback=_answer_callback_query,
            build_tasks_message=_build_tasks_message,
            edit_message=edit_message,
            logger=_logger,
        ),
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


_response_service = ResponseService(
    ResponseServiceDeps(
        telegram=telegram_gateway,
        ai=_ai_request_service,
        providers=_provider_service,
        telegram_token=lambda: environ.get("TELEGRAM_TOKEN"),
        send_typing=send_typing,
        sleep=time.sleep,
        random_delay=random.uniform,
        build_random_reply=build_random_reply,
        gen_random=gen_random,
        extract_user_name=_extract_user_name,
        handle_response=_ai_handle_response,
        consume_stream=consume_stream_to_telegram,
        set_stream_metadata=set_streamed_response_metadata,
        edit_message=edit_message,
        logger=_logger,
    )
)


def handle_callback_query(callback_query: Dict[str, Any]) -> None:
    callback_runtime.handle_callback_query(
        callback_query,
        deps=callback_runtime.CallbackQueryDeps(
            guard_callback=_guard_callback,
            handle_topup=_billing_service.handle_topup_callback,
            handle_task=handle_task_callback,
            handle_signal=handle_token_signal_callback,
            config_redis=_config_runtime.redis,
            delete_msg=telegram_gateway.delete_message,
            edit_photo=telegram_gateway.edit_photo,
            is_chat_admin=_admin_service.is_chat_admin,
            answer_callback=_answer_callback_query,
            admin_report=_admin_service.report,
            is_group_chat_type=is_group_chat_type,
            send_msg=telegram_gateway.send_message,
            report_unauthorized=_admin_service.report_unauthorized_config_attempt,
            denial_message=ADMIN_CONFIG_DENIAL_MESSAGE,
            get_chat_config=_chat_config_service.get_chat_config,
            config=callback_runtime.CallbackConfigDeps(
                set_chat_config=_chat_config_service.set_chat_config,
                coerce_bool=coerce_bool,
                guard_callback=_guard_callback,
                log_event=_log_config_event,
                timezone_offset_min=TIMEZONE_OFFSET_MIN,
                timezone_offset_max=TIMEZONE_OFFSET_MAX,
            ),
            build_config_text=build_config_text,
            build_config_keyboard=build_config_keyboard,
            edit_message=edit_message,
        ),
    )


def _build_message_handler_deps() -> MessageHandlerDeps:
    ai_svc = build_ai_service(
        credits_db_service=credits_db_service,
        get_chat_history=_message_state_service.get_history,
        prepare_chat_memory=_summary_service.prepare_memory,
        build_ai_messages=build_ai_messages,
        check_provider_available=_provider_service.is_scope_available,
        has_openrouter_fallback=_provider_service.has_openrouter_fallback,
        handle_rate_limit=_response_service.handle_rate_limit,
        handle_ai_response=_response_service.handle,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
        stream_summary_command=_summary_service.stream_command,
    )
    return build_message_handler_deps(
        chat=MessageChatDeps(
            config_redis=_config_runtime.redis,
            get_chat_config=_chat_config_service.get_chat_config,
            extract_user_id=_extract_user_id,
            extract_numeric_chat_id=_extract_numeric_chat_id,
        ),
        routing=MessageRoutingDeps(
            initialize_commands=initialize_commands,
            parse_command=_command_parse_command,
            should_auto_process_media=_command_should_auto_process_media,
            link_service=_link_service,
            should_gordo_respond=should_gordo_respond,
            is_group_chat_type=is_group_chat_type,
        ),
        io=MessageIODeps(
            send_msg=telegram_gateway.send_message,
            send_animation=telegram_gateway.send_animation,
            send_photo=telegram_gateway.send_photo,
            send_video=telegram_gateway.send_video,
            delete_msg=telegram_gateway.delete_message,
            edit_message=_response_service.edit_stream_message,
            admin_report=_admin_service.report,
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=_message_state_service.get_bot_metadata,
            save_bot_message_metadata=_message_state_service.save_bot_metadata,
            build_reply_context_text=_message_state_service.build_reply_context,
            format_user_message=_message_state_service.format_user_message,
            save_message_to_redis=_message_state_service.save_message,
            save_chat_member=_message_state_service.save_chat_member,
        ),
        ai=MessageAIDeps(
            ai_service=ai_svc,
            balance_formatter=BalanceFormatter(credits_db_service),
            handle_ai_stream=_response_service.stream_handler,
            gen_random=gen_random,
            build_insufficient_credits_message=(
                _billing_service.build_insufficient_message
            ),
            build_topup_keyboard=_billing_service.build_topup_keyboard,
            credits_db_service=credits_db_service,
            maybe_grant_onboarding_credits=lambda _svc, _rep, uid: (
                _billing_service.maybe_grant_onboarding(uid)
            ),
            handle_transcribe_with_message=handle_transcribe_with_message,
            handle_transcribe_with_message_result=handle_transcribe_with_message_result,
            check_provider_available=_provider_service.is_scope_available,
            has_openrouter_fallback=_provider_service.has_openrouter_fallback,
            handle_rate_limit=_response_service.handle_rate_limit,
            handle_successful_payment_message=(
                _billing_service.handle_successful_payment
            ),
            handle_config_command=handle_config_command,
            is_chat_admin=_admin_service.is_chat_admin,
            report_unauthorized_config_attempt=(
                _admin_service.report_unauthorized_config_attempt
            ),
            handle_transcribe=handle_transcribe,
            estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
            estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
        ),
        media=MessageMediaDeps(
            extract_message_content=extract_message_content,
            _transcribe_audio_file=_media_service.transcribe_file,
            _transcription_error_message=(
                media_commands.transcription_error_message
            ),
            download_telegram_file=telegram_gateway.download_file,
            measure_audio_duration_seconds=measure_audio_duration_seconds,
            resize_image_if_needed=_image_service.resize,
            encode_image_to_base64=_image_service.encode,
        ),
    )


def handle_msg(message: Dict[str, Any]) -> str:
    return _handle_msg_impl(message, _build_message_handler_deps())


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


# This is the single toolbox consumed by the Telegram adapter. Keeping the
# assembled graph here makes dependencies visible without hiding global state
# inside individual service modules.
app_runtime = ApplicationRuntime(
    config=_config_runtime,
    telegram=telegram_gateway,
    admin=_admin_service,
    cache=_cache_service,
    stocks=_stock_service,
    giphy=_giphy_service,
    polymarket=_polymarket_service,
    dollar=_dollar_service,
    bcra=_bcra_service,
    prices=_price_service,
    providers=_provider_service,
    ai=_ai_request_service,
    state=_message_state_service,
    summary=_summary_service,
    responses=_response_service,
    billing=_billing_service,
    media_cache=_media_cache_service,
    media=_media_service,
    images=_image_service,
    handle_message=handle_msg,
    handle_callback_query=handle_callback_query,
    estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
)
