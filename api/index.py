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
from urllib3.exceptions import InsecureRequestWarning
import warnings
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
)
from collections import Counter
from difflib import SequenceMatcher
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
from openpyxl import load_workbook
from decimal import Decimal
import unicodedata
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urlunparse
from functools import lru_cache


# TTL constants (seconds)
TTL_PRICE = 300  # 5 minutes
TTL_DOLLAR = 300  # 5 minutes
TTL_BCRA = 300  # 5 minutes
TTL_WEATHER = 1800  # 30 minutes
TTL_WEB_SEARCH = 300  # 5 minutes
TTL_WEB_FETCH = 300  # 5 minutes
WEB_FETCH_MAX_BYTES = 250_000
WEB_FETCH_MIN_CHARS = 500
WEB_FETCH_MAX_CHARS = 8000
WEB_FETCH_DEFAULT_CHARS = 4000
TTL_MEDIA_CACHE = 7 * 24 * 60 * 60  # 7 days
TTL_HACKER_NEWS = 600  # 10 minutes
TTL_MAYORISTA_MISSING = 300  # 5 minutes sentinel for missing mayorista values

# Provider backoff windows (seconds)
GROQ_RATE_LIMIT_BACKOFF_SECONDS = 600  # wait 10 minutes after a rate limit response
OPENROUTER_RATE_LIMIT_BACKOFF_SECONDS = 600
CLOUDFLARE_RATE_LIMIT_BACKOFF_SECONDS = 600

CACHE_STALE_GRACE_BCRA = 6 * 3600  # allow showing last BCRA data up to 6h stale
CACHE_STALE_GRACE_BANDS = 3600  # currency bands fall back for 1h

_bcra_local_cache: Dict[str, Any] = {
    "value": None,
    "expires_at": 0.0,
    "stale_until": 0.0,
    "meta": {},
}
_bcra_failure_until: float = 0.0

_currency_band_local_cache: Dict[str, Any] = {
    "value": None,
    "expires_at": 0.0,
    "stale_until": 0.0,
    "meta": {},
}
_currency_band_failure_until: float = 0.0


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

# Autonomous agent thought logging
AGENT_THOUGHTS_KEY = "agent:thoughts"
MAX_AGENT_THOUGHTS = 10
AGENT_THOUGHT_DISPLAY_LIMIT = 5
AGENT_THOUGHT_CHAR_LIMIT = 500
AGENT_RECENT_THOUGHT_WINDOW = 5
AGENT_REQUIRED_SECTIONS = ("HALLAZGOS", "PRÓXIMO PASO")
AGENT_EMPTY_RESPONSE_FALLBACK = (
    "HALLAZGOS: no se me ocurrió nada nuevo, pintó el vacío.\n"
    "PRÓXIMO PASO: meter una búsqueda puntual para traer un dato real y salir de la fiaca."
)
AGENT_REPETITION_RETRY_LIMIT = 3
AGENT_LOOP_FALLBACK_PREFIX = "HALLAZGOS: registré que estaba en un loop repitiendo"
AGENT_REPETITION_ESCALATION_HINT = (
    "No escribas que estás trabado o en un loop. Ejecutá de inmediato una herramienta "
    "(web_search o fetch_url) con un tema distinto y registrá datos nuevos "
    "(números, titulares, precios). Si el tema anterior no se mueve, cambiá a otro interés fuerte del gordo."
)

# Common filler words that shouldn't count as keywords when checking topic overlap
AGENT_KEYWORD_STOPWORDS: Set[str] = {
    "ante",
    "aqui",
    "aquel",
    "aquella",
    "aquello",
    "asi",
    "busque",
    "cada",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "cuyo",
    "datos",
    "donde",
    "durante",
    "entre",
    "este",
    "esta",
    "estas",
    "esto",
    "estos",
    "gordo",
    "hallazgos",
    "hacer",
    "hice",
    "investigue",
    "investigando",
    "investigar",
    "luego",
    "mientras",
    "mismo",
    "mucha",
    "mucho",
    "nada",
    "para",
    "pendiente",
    "pero",
    "porque",
    "proximo",
    "queda",
    "seguir",
    "sigue",
    "sobre",
    "solo",
    "todas",
    "todos",
    "todavia",
    "tema",
    "teni",
    "tenemos",
    "tener",
    "tenes",
    "tenia",
    "teniendo",
    "tengo",
    "unas",
    "unos",
    "voy",
}

# Rate limiting and timezone constants
RATE_LIMIT_GLOBAL_MAX = 1024
RATE_LIMIT_CHAT_MAX = 128
TTL_RATE_GLOBAL = 3600  # 1 hour
TTL_RATE_CHAT = 600  # 10 minutes
BA_TZ = timezone(timedelta(hours=-3))

# Global variable to cache bot configuration
_bot_config = None


def fmt_num(value: float, decimals: int = 2) -> str:
    """Format a number with up to `decimals` decimals, trimming trailing zeros."""
    try:
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def fmt_signed_pct(value: float, decimals: int = 2) -> str:
    """Format a signed percentage with trimmed trailing zeros."""
    try:
        return f"{value:+.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


# Known alternative frontends that users may already provide
ALTERNATIVE_FRONTENDS = {
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kkinstagram.com",
    "rxddit.com",
    "vxtiktok.com",
}

# Original social domains that may need replacement
ORIGINAL_FRONTENDS = {
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
}


def is_social_frontend(host: str) -> bool:
    """Return True if host is an original or alternative social frontend"""
    host = host.lower()
    frontends = ALTERNATIVE_FRONTENDS | ORIGINAL_FRONTENDS
    return any(host == d or host.endswith(f".{d}") for d in frontends)


def load_bot_config():
    """Load bot configuration from environment variables"""
    global _bot_config

    if _bot_config is not None:
        return _bot_config

    # Get required configuration from environment variables
    system_prompt = environ.get("BOT_SYSTEM_PROMPT")
    trigger_words_str = environ.get("BOT_TRIGGER_WORDS")

    if not system_prompt:
        raise ValueError("BOT_SYSTEM_PROMPT environment variable is required")

    if not trigger_words_str:
        raise ValueError("BOT_TRIGGER_WORDS environment variable is required")

    trigger_words = [word.strip() for word in trigger_words_str.split(",")]

    _bot_config = {"trigger_words": trigger_words, "system_prompt": system_prompt}

    return _bot_config


def config_redis(host=None, port=None, password=None):
    try:
        host = host or environ.get("REDIS_HOST", "localhost")
        port = int(port or environ.get("REDIS_PORT", 6379))
        password = password or environ.get("REDIS_PASSWORD", None)
        redis_client = redis.Redis(
            host=host, port=port, password=password, decode_responses=True
        )
        # Test connection
        redis_client.ping()
        return redis_client
    except Exception as e:
        error_context = {
            "host": host,
            "port": port,
            "password": "***" if password else None,
        }
        error_msg = f"Redis connection error: {str(e)}"
        print(error_msg)
        admin_report(error_msg, e, error_context)
        raise  # Re-raise to prevent silent failure


# -----------------------------
# Small parsing helpers (reused)
# -----------------------------
def parse_date_string(value: str) -> Optional[datetime]:
    """Parse common date formats into a datetime (date at midnight).

    Accepts: DD/MM/YYYY, DD/MM/YY, YYYY-MM-DD, DD-MM-YYYY
    Returns None on failure.
    """
    try:
        s = (value or "").strip().split()[0]
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
    except Exception:
        pass
    return None


def parse_monetary_number(value: Union[str, float, int, Decimal]) -> Optional[float]:
    """Parse localized monetary strings like "1.234,56" to float.

    Returns float or None on failure. If already number-like, casts to float.
    """
    try:
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        s = str(value)
        return float(s.replace(".", "").replace(",", "."))
    except Exception:
        return None


# -----------------------------
# Lightweight local cache helpers
# -----------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_local_cache(
    cache_store: Dict[str, Any],
    value: Any,
    ttl: int,
    stale_grace: int,
    fetched_at: Optional[str] = None,
) -> None:
    now = time.time()
    cache_store["value"] = value
    cache_store["expires_at"] = now + max(ttl, 0)
    cache_store["stale_until"] = cache_store["expires_at"] + max(stale_grace, 0)
    cache_store["meta"] = {"fetched_at": fetched_at or _now_utc_iso()}


def _local_cache_get(
    cache_store: Dict[str, Any], allow_stale: bool = False
) -> Tuple[Optional[Any], bool, Dict[str, Any]]:
    value = cache_store.get("value")
    meta = cache_store.get("meta") or {}
    if value is None:
        return None, False, {}

    now = time.time()
    expires_at = cache_store.get("expires_at", 0.0)
    if now <= expires_at:
        return value, True, meta

    stale_until = cache_store.get("stale_until", 0.0)
    if allow_stale and now <= stale_until:
        return value, False, meta

    if now > stale_until:
        cache_store["value"] = None
        cache_store["meta"] = {}
    return None, False, {}


# -----------------------------
# BCRA v4.0 API helpers
# -----------------------------
def bcra_api_get(
    path: str, params: Optional[Dict[str, Any]] = None, ttl: int = TTL_BCRA
) -> Optional[Dict[str, Any]]:
    """Call BCRA Estadísticas Monetarias v4.0 API via cached_requests without SSL validation."""

    try:
        base_url = "https://api.bcra.gob.ar/estadisticas/v4.0"
        url = base_url + (path if path.startswith("/") else "/" + path)
        # Explicitly skip certificate validation per ops request; quiet the warning once
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        resp = cached_requests(
            url,
            params,
            None,
            ttl,
            hourly_cache=False,
            get_history=False,
            verify_ssl=False,
        )
        if not resp or "data" not in resp:
            return None
        data = resp["data"]
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def bcra_list_variables(
    category: Optional[str] = "Principales Variables",
) -> Optional[List[Dict[str, Any]]]:
    """Return list of variables. If `category` is provided, filter client-side."""
    # Server defaults limit to 1000; omit category/limit for compatibility
    params: Optional[Dict[str, Any]] = None
    if category:
        params = {"limit": "2000"}
    data = bcra_api_get("/monetarias", params)
    try:
        if not data:
            return None
        results = data.get("results")
        if not isinstance(results, list):
            return None
        if not category:
            return results

        # Filter by category locally to avoid server-side incompatibilities
        def norm(s: str) -> str:
            return (
                unicodedata.normalize("NFKD", s or "")
                .encode("ascii", "ignore")
                .decode("ascii")
            ).lower()

        cat = norm(category)
        return [r for r in results if cat in norm(str(r.get("categoria", "")))]
    except Exception:
        return None


def _parse_currency_band_rows(
    rows: Iterable[List[str]], *, today: Optional[date] = None
) -> Optional[Dict[str, Any]]:
    """Return latest band row parsed into floats and formatted date."""

    effective_today = today or datetime.now(BA_TZ).date()
    parsed_rows: List[Tuple[date, float, float]] = []

    for row in rows:
        if len(row) < 3:
            continue
        date_raw, lower_raw, upper_raw = row[0], row[1], row[2]
        dt = parse_date_string(date_raw)
        if not dt:
            continue
        lower = parse_monetary_number(lower_raw)
        upper = parse_monetary_number(upper_raw)
        if lower is None or upper is None:
            continue
        parsed_rows.append((dt.date(), float(lower), float(upper)))

    if not parsed_rows:
        return None

    parsed_rows.sort(key=lambda item: item[0])

    on_or_before = [item for item in parsed_rows if item[0] <= effective_today]
    if on_or_before:
        current = on_or_before[-1]
        previous = on_or_before[-2] if len(on_or_before) > 1 else None
    else:
        current = parsed_rows[-1]
        previous = parsed_rows[-2] if len(parsed_rows) > 1 else None

    current_date, lower_val, upper_val = current

    def pct_change(
        curr: float, prev: Optional[Tuple[date, float, float]], index: int
    ) -> Optional[float]:
        if prev is None:
            return None
        prev_value_raw = prev[index]
        if not isinstance(prev_value_raw, (int, float)):
            return None
        prev_value = float(prev_value_raw)
        if prev_value == 0:
            return None
        try:
            return ((curr - prev_value) / prev_value) * 100
        except Exception:
            return None

    lower_change = pct_change(lower_val, previous, 1)
    upper_change = pct_change(upper_val, previous, 2)

    date_iso = current_date.isoformat()
    result: Dict[str, Any] = {
        "date": to_ddmmyy(date_iso),
        "date_iso": date_iso,
        "lower": lower_val,
        "upper": upper_val,
    }

    if lower_change is not None:
        result["lower_change_pct"] = lower_change
    if upper_change is not None:
        result["upper_change_pct"] = upper_change

    return result


def _fetch_currency_band_series(var_id: int, limit: int = 200) -> Dict[date, float]:
    """Return a {date: value} map for the requested Principales Variables series."""

    data = bcra_api_get(f"/monetarias/{var_id}", {"limit": str(int(limit))})
    if not data:
        return {}

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return {}

    first = results[0]
    detalle = first.get("detalle") if isinstance(first, dict) else None
    if not isinstance(detalle, list):
        return {}

    series: Dict[date, float] = {}
    for entry in detalle:
        if not isinstance(entry, dict):
            continue
        fecha_raw = entry.get("fecha")
        valor_raw = entry.get("valor")
        if valor_raw is None:
            continue
        dt = parse_date_string(str(fecha_raw or ""))
        if not dt:
            continue
        try:
            series[dt.date()] = float(valor_raw)
        except Exception:
            continue

    return series


def _get_cached_currency_band_entry(
    allow_stale: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return cached currency band limits with freshness metadata."""

    redis_client: Optional[redis.Redis]
    try:
        redis_client = config_redis()
    except Exception:
        redis_client = None

    cache_key = "bcra_currency_band_limits"
    if redis_client is not None:
        cached = redis_get_json(redis_client, cache_key)
        if cached:
            payload: Dict[str, Any]
            if isinstance(cached, dict) and "data" in cached:
                payload = cast(Dict[str, Any], cached)
            else:
                payload = {"data": cached}
            fetched_at = cast(Optional[str], payload.get("fetched_at"))
            _update_local_cache(
                _currency_band_local_cache,
                cast(Dict[str, Any], payload["data"]),
                TTL_BCRA,
                CACHE_STALE_GRACE_BANDS,
                fetched_at,
            )
            return cast(Dict[str, Any], payload["data"]), {
                "is_fresh": True,
                "fetched_at": fetched_at,
            }

        if allow_stale:
            last_success = redis_get_json(redis_client, f"{cache_key}:last_success")
            if isinstance(last_success, dict) and "data" in last_success:
                fetched_at = cast(Optional[str], last_success.get("fetched_at"))
                _update_local_cache(
                    _currency_band_local_cache,
                    cast(Dict[str, Any], last_success["data"]),
                    0,
                    CACHE_STALE_GRACE_BANDS,
                    fetched_at,
                )
                return cast(Dict[str, Any], last_success["data"]), {
                    "is_fresh": False,
                    "fetched_at": fetched_at,
                }

    local_value, is_fresh, meta = _local_cache_get(
        _currency_band_local_cache, allow_stale=allow_stale
    )
    if local_value is not None:
        return cast(Dict[str, Any], local_value), {
            "is_fresh": is_fresh,
            "fetched_at": meta.get("fetched_at"),
        }

    return None, {"is_fresh": False, "fetched_at": None}


def cache_currency_band_limits(data: Dict[str, Any], ttl: int = TTL_BCRA) -> None:
    if not data:
        return

    fetched_at_iso = _now_utc_iso()
    payload = {"data": data, "fetched_at": fetched_at_iso}

    try:
        redis_client = config_redis()
        redis_setex_json(redis_client, "bcra_currency_band_limits", ttl, payload)
        try:
            redis_set_json(
                redis_client,
                "bcra_currency_band_limits:last_success",
                payload,
            )
        except Exception:
            pass
    except Exception:
        pass

    _update_local_cache(
        _currency_band_local_cache,
        data,
        ttl,
        CACHE_STALE_GRACE_BANDS,
        fetched_at_iso,
    )


def fetch_currency_band_limits() -> Optional[Dict[str, Any]]:
    """Fetch latest currency band limits (piso/techo) via Principales Variables API."""

    try:
        variables = bcra_list_variables("Principales Variables")
        if not variables:
            return None

        def norm(text: str) -> str:
            return (
                unicodedata.normalize("NFKD", text or "")
                .encode("ascii", "ignore")
                .decode("ascii")
            ).lower()

        lower_id: Optional[int] = None
        upper_id: Optional[int] = None

        for item in variables:
            if lower_id and upper_id:
                break
            if not isinstance(item, dict):
                continue
            desc = str(item.get("descripcion", ""))
            if not desc:
                continue
            normalized = norm(desc)
            if "bandas cambiarias" not in normalized:
                continue
            var_id = item.get("idVariable")
            if not isinstance(var_id, int):
                continue
            if "superior" in normalized and upper_id is None:
                upper_id = var_id
            elif "inferior" in normalized and lower_id is None:
                lower_id = var_id

        if lower_id is None or upper_id is None:
            return None

        lower_series = _fetch_currency_band_series(lower_id)
        upper_series = _fetch_currency_band_series(upper_id)
        if not lower_series or not upper_series:
            return None

        common_dates = sorted(set(lower_series.keys()) & set(upper_series.keys()))
        if not common_dates:
            return None

        # limit to reasonable history window to keep downstream parsing lean
        max_rows = 180
        selected_dates = common_dates[-max_rows:]
        rows: List[List[str]] = [
            [
                dt.isoformat(),
                lower_series[dt],
                upper_series[dt],
            ]
            for dt in selected_dates
        ]

        return _parse_currency_band_rows(rows)
    except Exception as exc:
        print(f"Error fetching BCRA currency bands via API: {exc}")
        return None


def get_currency_band_limits() -> Optional[Dict[str, Any]]:
    """Return cached currency band limits, fetching and caching if necessary."""

    global _currency_band_failure_until

    cached, meta = _get_cached_currency_band_entry(allow_stale=True)
    effective_today = datetime.now(BA_TZ).date()

    if cached and isinstance(cached, dict):
        cached_date = parse_date_string(str(cached.get("date", "")))
        if cached_date and cached_date.date() > effective_today:
            cached = None
            meta = {"is_fresh": False, "fetched_at": meta.get("fetched_at")}

    if cached and meta.get("is_fresh"):
        return cached

    fallback = cached
    now_ts = time.time()
    if now_ts < _currency_band_failure_until:
        return fallback

    fetched = fetch_currency_band_limits()
    if fetched:
        cache_currency_band_limits(fetched, TTL_BCRA)
        _currency_band_failure_until = 0.0
        return fetched

    _currency_band_failure_until = now_ts + min(TTL_BCRA, 120)
    return fallback


def to_es_number(n: Union[float, int]) -> str:
    try:
        s = f"{float(n):,.2f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(n)


def to_ddmmyy(date_iso: str) -> str:
    dt = parse_date_string(date_iso)
    if not dt:
        return str(date_iso)
    return f"{dt.day:02d}/{dt.month:02d}/{dt.year % 100:02d}"


def bcra_fetch_latest_variables() -> Optional[Dict[str, Dict[str, str]]]:
    """Fetch latest Principales Variables via BCRA API and map to {name: {value, date}}"""
    try:
        vars_list = bcra_list_variables("Principales Variables")
        if not vars_list:
            return None
        variables: Dict[str, Dict[str, str]] = {}
        for v in vars_list:
            name = str(v.get("descripcion", "")).strip()
            date_iso = str(v.get("ultFechaInformada", "")).strip()
            val = v.get("ultValorInformado", None)
            if not name or val is None or not date_iso:
                continue
            variables[name] = {"value": to_es_number(val), "date": to_ddmmyy(date_iso)}
        return variables
    except Exception as e:
        print(f"Error fetching BCRA variables via API: {e}")
        return None


def bcra_get_value_for_date(desc_substr: str, date_iso: str) -> Optional[float]:
    """Get numeric value for the variable whose description contains desc_substr on date_iso.

    - Looks through Principales Variables for a matching description (accent-insensitive)
    - Requests the series for that exact date (desde=hasta=date_iso)
    - Caches mayorista by date if applicable
    """
    try:
        vars_list = bcra_list_variables("Principales Variables")
        if not vars_list:
            return None

        def norm(s: str) -> str:
            s = (
                unicodedata.normalize("NFKD", s or "")
                .encode("ascii", "ignore")
                .decode("ascii")
            )
            return s.lower()

        target = norm(desc_substr)
        var_id: Optional[int] = None
        var_name: Optional[str] = None
        for v in vars_list:
            desc = str(v.get("descripcion", ""))
            if target in norm(desc):
                vid = v.get("idVariable")
                if isinstance(vid, int):
                    var_id = vid
                    var_name = desc
                    break
        if var_id is None:
            return None

        data = bcra_api_get(
            f"/monetarias/{var_id}",
            {"desde": date_iso, "hasta": date_iso, "limit": "1"},
        )
        if not data:
            return None
        results = data.get("results")
        if not isinstance(results, list) or not results:
            return None
        detalle = results[0].get("detalle")
        if not isinstance(detalle, list) or not detalle:
            return None
        row = detalle[0]
        valor = row.get("valor")
        if valor is None:
            return None
        try:
            val_f = float(valor)
        except Exception:
            return None

        # If this is mayorista, persist by date
        try:
            if var_name and re.search("tipo.*cambio.*mayorista", norm(var_name)):
                redis_client = config_redis()
                redis_set_json(
                    redis_client,
                    f"bcra_mayorista:{date_iso}",
                    {"value": val_f, "date": to_ddmmyy(date_iso)},
                )
        except Exception:
            pass
        return val_f
    except Exception:
        return None


def redis_get_json(redis_client: redis.Redis, key: str) -> Optional[Any]:
    """Get a JSON value from Redis, parsed into Python or None."""
    try:
        data = redis_client.get(key)
        if not data:
            return None
        return json.loads(str(data))
    except Exception:
        return None


def redis_setex_json(redis_client: redis.Redis, key: str, ttl: int, value: Any) -> bool:
    """Set a JSON value with TTL in Redis; returns True on success."""
    try:
        return bool(redis_client.setex(key, ttl, json.dumps(value)))
    except Exception:
        return False


def redis_set_json(redis_client: redis.Redis, key: str, value: Any) -> bool:
    """Set a JSON value (no TTL) in Redis; returns True on success."""
    try:
        return bool(redis_client.set(key, json.dumps(value)))
    except Exception:
        return False


def get_agent_thoughts(
    redis_client: Optional[redis.Redis] = None,
) -> List[Dict[str, Any]]:
    """Return persisted autonomous agent thoughts (newest first)."""
    if redis_client is None:
        try:
            redis_client = config_redis()
        except Exception:
            return []

    try:
        raw_items = cast(
            List[str],
            redis_client.lrange(AGENT_THOUGHTS_KEY, 0, MAX_AGENT_THOUGHTS - 1),
        )
    except Exception as redis_error:
        admin_report(
            "Error retrieving agent thoughts",
            redis_error,
            {"operation": "lrange", "key": AGENT_THOUGHTS_KEY},
        )
        return []

    thoughts: List[Dict[str, Any]] = []
    for raw in raw_items:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        text_value = str(payload.get("text", "")).strip()
        if not text_value:
            continue

        timestamp_value: Optional[int]
        timestamp_raw = payload.get("timestamp")
        if isinstance(timestamp_raw, (int, float)):
            timestamp_value = int(timestamp_raw)
        elif isinstance(timestamp_raw, str) and timestamp_raw.isdigit():
            timestamp_value = int(timestamp_raw)
        else:
            timestamp_value = None

        thought_entry: Dict[str, Any] = {"text": text_value}
        if timestamp_value is not None:
            thought_entry["timestamp"] = timestamp_value
        thoughts.append(thought_entry)

    return thoughts


def build_agent_thoughts_context_message(
    thoughts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a system message describing recent agent thoughts for the model."""

    lines: List[str] = []
    for thought in thoughts:
        text = str(thought.get("text", "")).strip()
        if not text:
            continue
        timestamp_value = thought.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            dt = datetime.fromtimestamp(int(timestamp_value), tz=BA_TZ)
            time_label = dt.strftime("%d/%m %H:%M")
            lines.append(f"- [{time_label}] {text}")
        else:
            lines.append(f"- {text}")

    if not lines:
        return None

    context_text = (
        "MEMORIA AUTÓNOMA (más reciente primero):\n"
        + "\n".join(lines)
        + "\nUsá esta memoria cuando charles con humanos o cuando generes nuevos pensamientos autónomos."
    )

    return {
        "role": "system",
        "content": [{"type": "text", "text": context_text}],
    }


def get_agent_memory_context() -> Optional[Dict[str, Any]]:
    """Fetch persisted thoughts and build a message for ask_ai."""

    thoughts = get_agent_thoughts()
    return build_agent_thoughts_context_message(thoughts)


def format_agent_thoughts(thoughts: List[Dict[str, Any]]) -> str:
    """Render thoughts for human consumption."""

    if not thoughts:
        return "todavía no tengo pensamientos guardados, dejame que labure un toque."

    lines = ["🧠 Pensamientos recientes del gordo autónomo:"]
    index = 1
    for thought in thoughts:
        text = str(thought.get("text", "")).strip()
        if not text:
            continue

        formatted_text = text.replace("\r\n", "\n").replace("\r", "\n")
        formatted_text = formatted_text.replace("\n", "\n   ")
        timestamp_value = thought.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            dt = datetime.fromtimestamp(int(timestamp_value), tz=BA_TZ)
            time_label = dt.strftime("%d/%m %H:%M")
            lines.append(f"{index}. [{time_label}] {formatted_text}")
        else:
            lines.append(f"{index}. {formatted_text}")
        index += 1

    if len(lines) == 1:
        return "todavía no tengo pensamientos guardados, dejame que labure un toque."

    return "\n".join(lines)


def normalize_agent_text(text: str) -> str:
    """Normalize agent text for similarity comparisons."""

    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"[^a-z0-9]+", " ", without_accents)
    return collapsed.strip()


@lru_cache(maxsize=1)
def _loop_fallback_marker() -> str:
    """Cache the normalized marker for loop fallback detection."""

    return normalize_agent_text(AGENT_LOOP_FALLBACK_PREFIX)


def is_loop_fallback_text(text: str) -> bool:
    """Return True when the text matches the loop fallback template."""

    normalized_marker = _loop_fallback_marker()
    if not normalized_marker:
        return False

    normalized_text = normalize_agent_text(text)
    if not normalized_text:
        return False

    return normalized_text.startswith(normalized_marker)


@lru_cache(maxsize=1)
def _empty_fallback_marker() -> str:
    """Cache the normalized marker for the empty-response fallback."""

    return normalize_agent_text(AGENT_EMPTY_RESPONSE_FALLBACK)


def is_empty_agent_thought_text(text: str) -> bool:
    """Return True when the agent text matches the empty fallback template."""

    sanitized = str(text or "").strip()
    if not sanitized:
        return True

    normalized_marker = _empty_fallback_marker()
    if not normalized_marker:
        return False

    normalized_text = normalize_agent_text(sanitized)
    return normalized_text == normalized_marker


def _extract_keywords_from_normalized(normalized_text: str) -> Set[str]:
    """Return keyword candidates from an already-normalized agent text."""

    keywords: Set[str] = set()
    for token in normalized_text.split():
        if len(token) < 4:
            continue
        if token in AGENT_KEYWORD_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        keywords.add(token)

    return keywords


def get_agent_text_features(text: str) -> Tuple[str, Set[str]]:
    """Return normalized text plus keywords for repetition checks."""

    normalized = normalize_agent_text(text)
    if not normalized:
        return "", set()

    return normalized, _extract_keywords_from_normalized(normalized)


def extract_agent_keywords(text: str) -> Set[str]:
    """Return relevant lowercase keywords to detect repeated topics."""

    _, keywords = get_agent_text_features(text)
    return keywords


def _agent_keywords_are_repetitive(
    new_keywords: Set[str], previous_keywords: Set[str]
) -> bool:
    """Decide if two keyword sets point to the same topic."""

    if not new_keywords or not previous_keywords:
        return False

    overlap = new_keywords & previous_keywords
    if len(overlap) >= 3:
        return True

    min_len = min(len(new_keywords), len(previous_keywords))
    if min_len <= 1:
        return False

    if len(overlap) >= 2 and min_len <= 5:
        return True

    overlap_ratio = len(overlap) / min_len
    return overlap_ratio >= 0.6


def _normalized_texts_are_repetitive(normalized_new: str, normalized_prev: str) -> bool:
    """Compare normalized agent texts to determine repetitive content."""

    if not normalized_new or not normalized_prev:
        return False

    if normalized_new == normalized_prev:
        return True

    similarity = SequenceMatcher(None, normalized_new, normalized_prev).ratio()
    if similarity >= 0.88:
        return True

    new_tokens = set(normalized_new.split())
    prev_tokens = set(normalized_prev.split())
    if not new_tokens or not prev_tokens:
        return False

    union_len = len(new_tokens | prev_tokens)
    if union_len == 0:
        return False

    overlap = len(new_tokens & prev_tokens) / union_len
    return overlap >= 0.75


def is_repetitive_thought(new_text: str, previous_text: Optional[str]) -> bool:
    """Return True when the new thought is effectively a repeat of the last one."""

    if not new_text or not previous_text:
        return False

    normalized_new, new_keywords = get_agent_text_features(new_text)
    normalized_prev, prev_keywords = get_agent_text_features(previous_text)

    if not normalized_new or not normalized_prev:
        return False

    if _normalized_texts_are_repetitive(normalized_new, normalized_prev):
        return True

    return _agent_keywords_are_repetitive(new_keywords, prev_keywords)


def find_repetitive_recent_thought(
    new_text: str, previous_texts: Iterable[str]
) -> Optional[str]:
    """Return the first recent thought that matches the new text too closely."""

    normalized_new, new_keywords = get_agent_text_features(new_text)
    if not normalized_new:
        return None

    for candidate in previous_texts:
        sanitized = str(candidate or "").strip()
        if not sanitized:
            continue
        normalized_prev, previous_keywords = get_agent_text_features(sanitized)
        if not normalized_prev:
            continue
        if _normalized_texts_are_repetitive(normalized_new, normalized_prev):
            return sanitized
        if _agent_keywords_are_repetitive(new_keywords, previous_keywords):
            return sanitized

    return None


def summarize_recent_agent_topics(
    thoughts: Iterable[Dict[str, Any]], limit: int = 4
) -> List[str]:
    """Extract short summaries of recent topics from stored thoughts."""

    summaries: List[str] = []
    seen: Set[str] = set()

    for thought in thoughts:
        if len(summaries) >= limit:
            break

        text = ""
        if isinstance(thought, dict):
            text = str(thought.get("text", "")).strip()
        else:
            text = str(thought or "").strip()

        if not text:
            continue

        section_content = extract_agent_section_content(text, "HALLAZGOS")
        snippet_source = section_content or text
        snippet_source = re.sub(r"\s+", " ", snippet_source).strip()
        if not snippet_source:
            continue

        first_sentence_parts = re.split(r"(?<=[.!?])\s+", snippet_source, maxsplit=1)
        first_sentence = first_sentence_parts[0].strip() if first_sentence_parts else ""
        snippet = truncate_text(first_sentence or snippet_source, 120)
        snippet_key = snippet.lower()

        if not snippet or snippet_key in seen:
            continue

        summaries.append(snippet)
        seen.add(snippet_key)

    return summaries


def extract_agent_section_content(text: str, header: str) -> Optional[str]:
    """Extract the content of a structured agent section."""

    sanitized = (text or "").strip()
    if not sanitized:
        return None

    def normalize_value(value: str) -> str:
        """Return a normalized representation without diacritics for matching."""

        normalized = unicodedata.normalize("NFD", value)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def normalize_with_mapping(value: str) -> Tuple[str, List[int]]:
        """Normalize text while tracking original indices for slicing."""

        normalized_chars: List[str] = []
        index_map: List[int] = []
        for idx, char in enumerate(value):
            for component in unicodedata.normalize("NFD", char):
                if not unicodedata.combining(component):
                    normalized_chars.append(component)
                    index_map.append(idx)
        return "".join(normalized_chars), index_map

    normalized_text, index_map = normalize_with_mapping(sanitized)

    normalized_sections = {
        value: normalize_value(value) for value in AGENT_REQUIRED_SECTIONS
    }
    normalized_header = normalized_sections.get(header, normalize_value(header))

    header_pattern = re.compile(
        rf"(?im)^{re.escape(normalized_header)}:\s*", re.UNICODE
    )
    match = header_pattern.search(normalized_text)
    if not match:
        return None

    def original_index(normalized_index: int) -> int:
        if normalized_index <= 0:
            return 0
        if normalized_index >= len(index_map):
            return len(sanitized)
        return index_map[normalized_index]

    start = original_index(match.end())
    other_headers_normalized = [
        normalized_sections[value]
        for value in AGENT_REQUIRED_SECTIONS
        if value != header
    ]
    end = len(sanitized)
    combined_headers = "|".join(
        re.escape(value) for value in other_headers_normalized if value
    )
    if combined_headers:
        next_pattern = re.compile(rf"(?im)^(?:{combined_headers}):\s*", re.UNICODE)
        next_match = next_pattern.search(normalized_text, match.end())
        if next_match:
            end = original_index(next_match.start())

    content = sanitized[start:end].strip()
    return content or None


def agent_sections_are_valid(text: str) -> bool:
    """Check that the agent response includes the required structured sections."""

    if not text:
        return False

    for header in AGENT_REQUIRED_SECTIONS:
        if not extract_agent_section_content(text, header):
            return False

    return True


def get_agent_retry_hint(
    previous_text: Optional[str], rng: Optional[random.Random] = None
) -> str:
    """Return a randomized thinking prompt so the agent explores nuevos datos."""

    normalized_previous = normalize_agent_text(previous_text or "")
    tokens = [token for token in normalized_previous.split() if token]

    filtered_tokens: List[str] = []
    for token in tokens:
        if token in AGENT_KEYWORD_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        filtered_tokens.append(token)

    keyword_counter = Counter(filtered_tokens)
    top_keywords: List[str] = [word for word, _ in keyword_counter.most_common(3)]

    if top_keywords:
        avoided_fragment = (
            "Marcá como prohibidos estos términos ya gastados: "
            + ", ".join(top_keywords)
            + ". "
        )
    else:
        avoided_fragment = "Arrancá desde cero sin reciclar la última búsqueda. "

    rng_instance = rng if rng is not None else random

    option_count = rng_instance.randint(3, 4)
    ordinal_words = ("primera", "segunda", "tercera", "cuarta", "quinta")
    ordinal_index = rng_instance.randint(0, option_count - 1)
    ordinal_word = ordinal_words[min(ordinal_index, len(ordinal_words) - 1)]

    brainstorming_templates = (
        "Anotá {n} búsquedas frescas en temas distintos. Ejecutá la {ordinal} con web_search.",
        "Hacé una mini lluvia de ideas con {n} queries nuevos y corré la {ordinal} usando web_search.",
        "Pensá en {n} consultas posibles que sorprendan al gordo y quedate con la {ordinal} para web_search.",
    )

    follow_up_templates = (
        "Traé números, fechas y citá la fuente puntual.",
        "Resumí el dato clave con cifras concretas y quién lo publicó.",
        "Documentá resultados verificables (monto, variación, protagonista) y la fuente exacta.",
    )

    brainstorming_prompt = rng_instance.choice(brainstorming_templates).format(
        n=option_count,
        ordinal=ordinal_word,
    )
    follow_up_prompt = rng_instance.choice(follow_up_templates)

    letter_choices = tuple("abcdefghijklmnñopqrstuvwxyz")
    chosen_letter = rng_instance.choice(letter_choices).upper()
    alternate_pool = [
        value for value in letter_choices if value.upper() != chosen_letter
    ]
    fallback_letter = (
        rng_instance.choice(alternate_pool).upper() if alternate_pool else chosen_letter
    )

    numeric_floor = rng_instance.randint(8, 24)
    numeric_multiplier = rng_instance.randint(3, 9)
    numeric_target = numeric_floor * numeric_multiplier

    constraint_templates = (
        'Sumá una restricción creativa: la búsqueda tiene que incluir un protagonista cuya inicial sea "{letter}" y una cifra cerca de {value}.',
        'Obligate a que la consulta nombre algo que empiece con "{letter}" y mencione un número alrededor de {value}.',
        'Forzá el query a combinar un actor que arranque con "{letter}" más un dato numérico aproximado a {value}.',
    )

    pivot_templates = (
        'Si web_search no trae novedad, generá otra lluvia de ideas reemplazando las palabras prohibidas por categorías nuevas y probá con inicial "{fallback}".',
        'Si la ejecución devuelve humo, descartá la idea y repetí el proceso con términos distintos que comiencen con "{fallback}".',
        'Si no aparecen datos frescos, reseteá las keywords vetadas y buscá una consulta distinta arrancando por "{fallback}".',
    )

    constraint_prompt = rng_instance.choice(constraint_templates).format(
        letter=chosen_letter, value=numeric_target
    )
    pivot_prompt = rng_instance.choice(pivot_templates).format(fallback=fallback_letter)

    return (
        avoided_fragment
        + brainstorming_prompt
        + " "
        + follow_up_prompt
        + " "
        + constraint_prompt
        + " "
        + pivot_prompt
    )


def build_agent_retry_prompt(
    previous_text: Optional[str], rng: Optional[random.Random] = None
) -> str:
    """Create a corrective prompt so the agent avoids looping."""

    preview = truncate_text(previous_text or "", 160)
    preview_single_line = preview.replace("\n", " ").strip()
    base_prompt = (
        "La última nota no sirvió: te repetiste igual que la memoria anterior o no respetaste la estructura obligatoria. "
        "Antes de escribir otra vez, completá el pendiente y contá resultados concretos. "
        "Si necesitás info fresca, llamá a la herramienta web_search con un query preciso y resumí lo que encontraste. Si ya cerraste ese tema, cambiá a otro interés fuerte del gordo en vez de seguir clavado en lo mismo. "
        + (
            f'Esto fue lo último guardado o la nota fallida: "{preview_single_line}". '
            if preview_single_line
            else ""
        )
        + "Escribí ahora una nota distinta con hechos puntuales y cerrala en dos secciones claras: "
        '"HALLAZGOS:" con los datos específicos que obtuviste y "PRÓXIMO PASO:" con la siguiente acción directa.'
    )

    hint = get_agent_retry_hint(previous_text, rng=rng)
    if hint:
        return f"{base_prompt} {hint}"
    return base_prompt


def build_agent_fallback_entry(previous_text: Optional[str]) -> str:
    """Generate a non-repetitive fallback entry when the agent keeps looping."""

    sanitized_previous = (previous_text or "").strip()
    normalized_previous = normalize_agent_text(sanitized_previous)
    fallback_marker = _loop_fallback_marker()

    preview = truncate_text(sanitized_previous, 120)
    preview_single_line = preview.replace("\n", " ").strip()

    include_fragment = True
    if (
        normalized_previous
        and fallback_marker
        and normalized_previous.startswith(fallback_marker)
    ):
        include_fragment = False

    loop_fragment = (
        f' "{preview_single_line}"' if include_fragment and preview_single_line else ""
    )
    return (
        f"{AGENT_LOOP_FALLBACK_PREFIX}{loop_fragment} sin generar avances reales.\n"
        "PRÓXIMO PASO: hacer una búsqueda web urgente, anotar los datos específicos que salgan y recién después planear el próximo paso."
    )


def ensure_agent_response_text(text: Optional[str]) -> str:
    """Return a trimmed agent response or fall back to a canned entry."""

    sanitized = str(text or "").strip()
    return sanitized or AGENT_EMPTY_RESPONSE_FALLBACK


def build_agent_retry_messages(
    base_messages: List[Dict[str, Any]],
    assistant_text: str,
    corrective_prompt: str,
) -> List[Dict[str, Any]]:
    """Compose retry messages while keeping the original conversation intact."""

    return base_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text or ""}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": corrective_prompt}],
        },
    ]


def request_agent_response(
    generator: Callable[[List[Dict[str, Any]]], str],
    messages: List[Dict[str, Any]],
    error_context: str,
) -> str:
    """Invoke the agent generator while reporting failures consistently."""

    try:
        response = generator(messages)
    except Exception as ai_error:
        admin_report(error_context, ai_error)
        raise

    return ensure_agent_response_text(response)


def save_agent_thought(
    thought_text: str, redis_client: Optional[redis.Redis] = None
) -> Optional[Dict[str, Any]]:
    """Persist a new agent thought, trimming to the configured limits."""

    sanitized = (thought_text or "").strip()
    if not sanitized:
        return None

    truncated = truncate_text(sanitized, AGENT_THOUGHT_CHAR_LIMIT)

    if redis_client is None:
        try:
            redis_client = config_redis()
        except Exception:
            return None

    timestamp_value = int(time.time())
    entry = {"text": truncated, "timestamp": timestamp_value}

    try:
        payload = json.dumps(entry, ensure_ascii=False)
        pipeline = redis_client.pipeline()
        pipeline.lpush(AGENT_THOUGHTS_KEY, payload)
        pipeline.ltrim(AGENT_THOUGHTS_KEY, 0, MAX_AGENT_THOUGHTS - 1)
        pipeline.execute()
    except Exception as redis_error:
        admin_report(
            "Error saving agent thought",
            redis_error,
            {"thought_preview": truncated[:80]},
        )
        return None

    return entry


def show_agent_thoughts() -> str:
    """Expose stored thoughts through the /agent command."""

    thoughts = get_agent_thoughts()
    visible_thoughts = thoughts[:AGENT_THOUGHT_DISPLAY_LIMIT]
    return format_agent_thoughts(visible_thoughts)


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


def get_cached_transcription(file_id: str) -> Optional[str]:
    """Get cached audio transcription from Redis"""
    try:
        redis_client = config_redis()
        cache_key = f"audio_transcription:{file_id}"
        cached_text = redis_client.get(cache_key)
        if cached_text:
            return str(cached_text)
        return None
    except Exception as e:
        print(f"Error getting cached transcription: {e}")
        return None


def cache_transcription(file_id: str, text: str, ttl: int = TTL_MEDIA_CACHE) -> None:
    """Cache audio transcription in Redis (default 7 days)"""
    try:
        redis_client = config_redis()
        cache_key = f"audio_transcription:{file_id}"
        redis_client.setex(cache_key, ttl, text)
    except Exception as e:
        print(f"Error caching transcription: {e}")


def get_cached_description(file_id: str) -> Optional[str]:
    """Get cached image description from Redis"""
    try:
        redis_client = config_redis()
        cache_key = f"image_description:{file_id}"
        cached_desc = redis_client.get(cache_key)
        if cached_desc:
            return str(cached_desc)
        return None
    except Exception as e:
        print(f"Error getting cached description: {e}")
        return None


def cache_description(
    file_id: str, description: str, ttl: int = TTL_MEDIA_CACHE
) -> None:
    """Cache image description in Redis (default 7 days)"""
    try:
        redis_client = config_redis()
        cache_key = f"image_description:{file_id}"
        redis_client.setex(cache_key, ttl, description)
    except Exception as e:
        print(f"Error caching description: {e}")


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
    cache_expiration_time = TTL_DOLLAR
    api_url = "https://criptoya.com/api/dolar"

    dollars = cached_requests(api_url, None, None, cache_expiration_time, True)

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

        cache_expiration_time = TTL_DOLLAR

        api_url = "https://criptoya.com/api/dolar"

        dollars = cached_requests(api_url, None, None, cache_expiration_time, True)

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


def _get_cached_bcra_cache_entry(
    allow_stale: bool = False,
) -> Tuple[Optional[Dict], Dict[str, Any]]:
    """Return cached BCRA variables along with freshness metadata."""

    redis_client: Optional[redis.Redis]
    try:
        redis_client = config_redis()
    except Exception as e:
        print(f"Error getting cached BCRA variables: {e}")
        redis_client = None

    cache_key = "bcra_variables"
    if redis_client is not None:
        cached = redis_get_json(redis_client, cache_key)
        if cached:
            payload: Dict[str, Any]
            if isinstance(cached, dict) and "data" in cached:
                payload = cast(Dict[str, Any], cached)
            else:
                payload = {"data": cached}
            fetched_at = cast(Optional[str], payload.get("fetched_at"))
            _update_local_cache(
                _bcra_local_cache,
                cast(Dict, payload["data"]),
                TTL_BCRA,
                CACHE_STALE_GRACE_BCRA,
                fetched_at,
            )
            return cast(Dict, payload["data"]), {
                "is_fresh": True,
                "fetched_at": fetched_at,
            }

        if allow_stale:
            last_success = redis_get_json(redis_client, f"{cache_key}:last_success")
            if isinstance(last_success, dict) and "data" in last_success:
                fetched_at = cast(Optional[str], last_success.get("fetched_at"))
                _update_local_cache(
                    _bcra_local_cache,
                    cast(Dict, last_success["data"]),
                    0,
                    CACHE_STALE_GRACE_BCRA,
                    fetched_at,
                )
                return cast(Dict, last_success["data"]), {
                    "is_fresh": False,
                    "fetched_at": fetched_at,
                }

    local_value, is_fresh, meta = _local_cache_get(
        _bcra_local_cache, allow_stale=allow_stale
    )
    if local_value is not None:
        return cast(Dict, local_value), {
            "is_fresh": is_fresh,
            "fetched_at": meta.get("fetched_at"),
        }

    return None, {"is_fresh": False, "fetched_at": None}


def get_cached_bcra_variables(allow_stale: bool = False) -> Optional[Dict]:
    """Get cached BCRA variables (optionally allowing stale fallback)."""

    value, _ = _get_cached_bcra_cache_entry(allow_stale=allow_stale)
    return value


def cache_bcra_variables(variables: Dict, ttl: int = TTL_BCRA) -> None:
    """Cache BCRA variables in Redis and the local fallback store."""

    if not variables:
        return

    fetched_at_iso = _now_utc_iso()
    payload = {"data": variables, "fetched_at": fetched_at_iso}

    try:
        redis_client = config_redis()
        cache_key = "bcra_variables"
        redis_setex_json(redis_client, cache_key, ttl, payload)
        try:
            redis_set_json(redis_client, f"{cache_key}:last_success", payload)
        except Exception:
            pass

        # Also persist the mayorista value by its own date for future lookups
        try:
            for key, data in (variables or {}).items():
                if re.search("tipo.*cambio.*mayorista", str(key).lower()):
                    value_num = parse_monetary_number(data.get("value", ""))
                    parsed_dt = parse_date_string(str(data.get("date", "")))
                    if value_num is None or not parsed_dt:
                        continue
                    date_key = parsed_dt.date().isoformat()
                    redis_set_json(
                        redis_client,
                        f"bcra_mayorista:{date_key}",
                        {"value": value_num, "date": str(data.get("date", ""))},
                    )
                    break
        except Exception:
            # Never break caching if enrichment fails
            pass
    except Exception as e:
        print(f"Error caching BCRA variables: {e}")

    _update_local_cache(
        _bcra_local_cache,
        variables,
        ttl,
        CACHE_STALE_GRACE_BCRA,
        fetched_at_iso,
    )


def _attach_bcra_meta(value: Optional[Dict], meta: Dict[str, Any]) -> Optional[Dict]:
    if not value:
        return None
    if meta.get("is_fresh"):
        return value

    enriched = dict(value)
    meta_payload: Dict[str, Any] = {"stale": True}
    fetched_at = meta.get("fetched_at")
    if fetched_at:
        meta_payload["fetched_at"] = fetched_at
    enriched["_meta"] = meta_payload
    return enriched


def get_or_refresh_bcra_variables() -> Optional[Dict]:
    """Return BCRA variables using cache or the BCRA API, propagating fetch errors."""

    global _bcra_failure_until

    cached, meta = _get_cached_bcra_cache_entry(allow_stale=True)
    if cached and meta.get("is_fresh"):
        return cached

    fallback = cached
    fallback_meta = meta

    now_ts = time.time()
    if now_ts < _bcra_failure_until:
        return _attach_bcra_meta(fallback, fallback_meta)

    variables = bcra_fetch_latest_variables()
    if variables:
        try:
            cache_bcra_variables(variables)
        except Exception as e:
            print(f"Error caching BCRA variables: {e}")
        _bcra_failure_until = 0.0
        return variables

    _bcra_failure_until = now_ts + min(TTL_BCRA, 120)
    return _attach_bcra_meta(fallback, fallback_meta)


def get_latest_itcrm_value() -> Optional[float]:
    """Return latest TCRM value using the cached value+date function."""
    try:
        details = get_latest_itcrm_value_and_date()
        return details[0] if details else None
    except Exception:
        return None


def _get_itcrm_value_for_date(target_dt: datetime) -> Optional[Tuple[float, datetime]]:
    """Return ITCRM value and sheet date for the closest entry on/before `target_dt`."""

    try:
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        response = requests.get(
            "https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/ITCRMSerie.xlsx",
            timeout=10,
            verify=False,
        )
        workbook = load_workbook(io.BytesIO(response.content), data_only=True)
        sheet_like = getattr(workbook, "active", None) or (
            workbook.worksheets[0] if getattr(workbook, "worksheets", None) else None
        )
        if sheet_like is None:
            return None
        sheet = cast(Any, sheet_like)

        def normalize_date(value: Any) -> Optional[datetime]:
            try:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, date):
                    return datetime.combine(value, datetime.min.time())
                if hasattr(value, "date"):
                    maybe_date = value.date()
                    if isinstance(maybe_date, date):
                        return datetime.combine(maybe_date, datetime.min.time())
            except Exception:
                pass

            parsed = parse_date_string(str(value or ""))
            if parsed:
                return parsed
            return None

        def normalize_value(cell_value: Any) -> Optional[float]:
            if cell_value is None:
                return None
            if isinstance(cell_value, (int, float, Decimal)):
                return float(cell_value)
            try:
                text_val = str(cell_value).strip()
            except Exception:
                return None
            if not text_val:
                return None
            try:
                return float(text_val)
            except ValueError:
                try:
                    normalized = text_val.replace(".", "").replace(",", ".")
                    return float(normalized)
                except Exception:
                    return None

        target_date = target_dt.date()
        for row in range(sheet.max_row, 0, -1):
            row_date_raw = sheet.cell(row=row, column=1).value
            row_value_raw = sheet.cell(row=row, column=2).value
            row_date_dt = normalize_date(row_date_raw)
            if row_date_dt is None:
                continue
            row_date_only = row_date_dt.date()
            if row_date_only > target_date:
                continue
            value = normalize_value(row_value_raw)
            if value is None:
                continue
            normalized_dt = datetime.combine(row_date_only, datetime.min.time())
            return value, normalized_dt

        return None
    except Exception as exc:
        print(f"Error finding ITCRM value for {target_dt.date()}: {exc}")
        return None


def cache_mayorista_missing(
    date_key: str, redis_client: Optional[redis.Redis] = None
) -> None:
    """Store a short-lived sentinel indicating mayorista is missing for date_key."""

    try:
        client = redis_client or config_redis()
        if not client:
            return
        payload = {"missing": True, "timestamp": int(time.time())}
        redis_setex_json(
            client,
            f"bcra_mayorista:{date_key}",
            TTL_MAYORISTA_MISSING,
            payload,
        )
    except Exception:
        pass


def calculate_tcrm_100(
    target_date: Optional[Union[str, datetime, date]] = None,
) -> Optional[float]:
    """Calculate nominal exchange rate that sets ITCRM to 100.

    When `target_date` is provided, the calculation is performed using the ITCRM row
    closest to (and not after) that date; otherwise, the latest available value is used.
    Uses the ITCRM sheet date to look up a pre-cached Dólar Mayorista value stored by
    date via `cache_bcra_variables`. If not found in cache, does not compute.
    """

    try:
        normalized_target: Optional[datetime] = None
        if target_date is not None:
            if isinstance(target_date, datetime):
                normalized_target = target_date
            elif isinstance(target_date, date):
                normalized_target = datetime.combine(target_date, datetime.min.time())
            else:
                normalized_target = parse_date_string(str(target_date))
            if normalized_target is None:
                return None

        itcrm_value: Optional[float]
        itcrm_dt: Optional[datetime]

        if normalized_target is None:
            details = get_latest_itcrm_value_and_date()
            if not details:
                return None
            raw_value, raw_date_str = details
            try:
                itcrm_value = float(raw_value)
            except Exception:
                return None
            parsed_dt = parse_date_string(raw_date_str)
            if not parsed_dt:
                return None
            itcrm_dt = parsed_dt
        else:
            lookup = _get_itcrm_value_for_date(normalized_target)
            if not lookup:
                return None
            value, matched_dt = lookup
            try:
                itcrm_value = float(value)
            except Exception:
                return None
            itcrm_dt = matched_dt

        if itcrm_dt is None:
            return None

        date_key = itcrm_dt.date().isoformat()

        # Lookup mayorista value by date from Redis
        wholesale_value: Optional[float] = None
        sentinel_present = False
        redis_client: Optional[redis.Redis] = None
        try:
            redis_client = config_redis()
            cached = redis_get_json(redis_client, f"bcra_mayorista:{date_key}")
            if isinstance(cached, dict):
                if cached.get("missing"):
                    sentinel_present = True
                elif "value" in cached:
                    wholesale_value = parse_monetary_number(
                        cached["value"]
                    )  # robust cast
        except Exception:
            redis_client = None
            wholesale_value = None

        if sentinel_present:
            return None

        if wholesale_value is None:
            # Try to fetch from BCRA API for this exact date
            fetched = bcra_get_value_for_date("tipo de cambio mayorista", date_key)
            if fetched is not None:
                wholesale_value = float(fetched)
            else:
                cache_mayorista_missing(date_key, redis_client)
                return None

        return wholesale_value * 100 / itcrm_value
    except Exception as e:
        print(f"Error calculating TCRM 100: {e}")
        return None


def get_cached_tcrm_100(
    hours_ago: int = 24, expiration_time: int = 300
) -> Tuple[Optional[float], Optional[float]]:
    """Get cached TCRM 100 value with optional historical change"""
    cache_key = "tcrm_100"

    try:
        redis_client = config_redis()
        redis_response = redis_get_json(redis_client, cache_key)
        history_data = (
            get_cache_history(hours_ago, cache_key, redis_client) if hours_ago else None
        )

        if history_data is None and hours_ago:
            try:
                history_dt = datetime.now() - timedelta(hours=hours_ago)
                history_prefix = history_dt.strftime("%Y-%m-%d-%H")
                history_key = history_prefix + cache_key
                existing_slot = redis_client.get(history_key)
                if not existing_slot:
                    backfill_value = calculate_tcrm_100(target_date=history_dt)
                    if backfill_value is not None:
                        history_timestamp = int(
                            history_dt.replace(
                                minute=0, second=0, microsecond=0
                            ).timestamp()
                        )
                        history_data = {
                            "timestamp": history_timestamp,
                            "data": backfill_value,
                        }
                        redis_set_json(redis_client, history_key, history_data)
            except Exception:
                history_data = None

        history_value = history_data["data"] if history_data else None
        timestamp = int(time.time())

        # Determine if we have a same-day mayorista value for the current ITCRM date
        same_day_ok = False
        skip_mayorista_fetch = False
        try:
            itcrm_cached = redis_get_json(redis_client, "latest_itcrm_details")
            itcrm_date_str = None
            if isinstance(itcrm_cached, dict) and "date" in itcrm_cached:
                itcrm_date_str = str(itcrm_cached.get("date", ""))
            else:
                details = get_latest_itcrm_value_and_date()
                if details:
                    itcrm_date_str = details[1]
            dt = parse_date_string(itcrm_date_str or "")
            if dt is not None:
                date_key = dt.date().isoformat()
                mayorista_cached = redis_get_json(
                    redis_client, f"bcra_mayorista:{date_key}"
                )
                if isinstance(mayorista_cached, dict):
                    if mayorista_cached.get("missing"):
                        skip_mayorista_fetch = True
                    elif "value" in mayorista_cached:
                        if parse_monetary_number(mayorista_cached["value"]) is not None:
                            same_day_ok = True
                # If not cached, attempt a one-shot API fetch to populate cache
                if not same_day_ok and not skip_mayorista_fetch:
                    fetched_val = bcra_get_value_for_date(
                        "tipo de cambio mayorista", date_key
                    )
                    if fetched_val is not None:
                        same_day_ok = True
                    else:
                        cache_mayorista_missing(date_key, redis_client)
                        skip_mayorista_fetch = True
        except Exception:
            same_day_ok = False

        def compute_and_store() -> Optional[float]:
            value = calculate_tcrm_100()
            if value is None:
                return None
            redis_value = {"timestamp": timestamp, "data": value}
            redis_set_json(redis_client, cache_key, redis_value)
            current_hour = datetime.now().strftime("%Y-%m-%d-%H")
            redis_set_json(redis_client, current_hour + cache_key, redis_value)
            return value

        if not same_day_ok:
            current_value = None
        elif redis_response is None:
            current_value = compute_and_store()
        else:
            cached_data = cast(Dict, redis_response)
            cache_age = timestamp - int(cached_data["timestamp"])
            if cache_age > expiration_time:
                current_value = compute_and_store()
            else:
                current_value = cached_data.get("data")

        change = None
        if current_value is not None and history_value:
            try:
                change = ((current_value / history_value) - 1) * 100
            except ZeroDivisionError:
                change = None

        return current_value, change
    except Exception as e:
        print(f"Error getting cached TCRM 100: {e}")
        return None, None


def get_latest_itcrm_details() -> Optional[Tuple[float, str]]:
    """Return latest TCRM value and its date (DD/MM/YY) from the spreadsheet.

    Simpler variant for display: no Redis caching and minimal logic.
    """
    try:
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        response = requests.get(
            "https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/ITCRMSerie.xlsx",
            timeout=10,
            verify=False,
        )
        workbook = load_workbook(io.BytesIO(response.content), data_only=True)
        sheet_like = getattr(workbook, "active", None) or (
            workbook.worksheets[0] if getattr(workbook, "worksheets", None) else None
        )
        if sheet_like is None:
            return None
        sheet = cast(Any, sheet_like)

        def parse_date_cell(v: Any) -> Optional[str]:
            # If it's a datetime/date, format directly
            try:
                if hasattr(v, "strftime"):
                    # Normalize to date then format as DD/MM/YY
                    try:
                        d = v.date() if hasattr(v, "date") else v
                        return f"{d.day:02d}/{d.month:02d}/{d.year % 100:02d}"
                    except Exception:
                        pass
            except Exception:
                pass
            # Try string formats
            try:
                s = str(v).strip()
                for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
                    try:
                        dt = datetime.strptime(s, fmt)
                        return f"{dt.day:02d}/{dt.month:02d}/{dt.year % 100:02d}"
                    except Exception:
                        continue
            except Exception:
                pass
            return None

        for row in range(sheet.max_row, 0, -1):
            date_cell = sheet.cell(row=row, column=1).value
            cell_value = sheet.cell(row=row, column=2).value
            if cell_value is None:
                continue
            # Convert to float robustly
            val: Optional[float] = None
            if isinstance(cell_value, (int, float, Decimal)):
                val = float(cell_value)
            else:
                text_val = str(cell_value)
                try:
                    val = float(text_val)
                except ValueError:
                    try:
                        normalized = text_val.replace(".", "").replace(",", ".")
                        val = float(normalized)
                    except Exception:
                        continue
            date_str = parse_date_cell(date_cell) or ""
            if val is not None:
                return val, date_str
        return None
    except Exception as e:
        print(f"Error fetching ITCRM details: {e}")
        return None


def get_latest_itcrm_value_and_date() -> Optional[Tuple[float, str]]:
    """Cached wrapper returning latest TCRM value and its date.

    Caches both value and date in Redis for 30 minutes.
    """
    try:
        redis_client = config_redis()
        cached = redis_get_json(redis_client, "latest_itcrm_details")
        if isinstance(cached, dict) and "value" in cached:
            try:
                return float(cached["value"]), str(cached.get("date", ""))
            except Exception:
                pass
    except Exception:
        pass

    details = get_latest_itcrm_details()
    if not details:
        return None
    value, date_str = details
    try:
        redis_client = config_redis()
        redis_setex_json(
            redis_client,
            "latest_itcrm_details",
            1800,
            {"value": value, "date": date_str},
        )
    except Exception:
        pass
    return value, date_str


def format_bcra_variables(variables: Dict) -> str:
    """Format BCRA variables for display (robust to naming changes)."""
    if not variables:
        return "No se pudieron obtener las variables del BCRA"

    def norm(s: str) -> str:
        try:
            import unicodedata as _ud

            return (
                _ud.normalize("NFKD", s or "")
                .encode("ascii", "ignore")
                .decode("ascii")
                .lower()
            )
        except Exception:
            return (s or "").lower()

    def format_value(value_str: str, is_percentage: bool = False) -> str:
        try:
            clean_value = (
                value_str.replace(".", "").replace(",", ".")
                if not is_percentage
                else value_str.replace(",", ".")
            )
            num = float(clean_value)
            if is_percentage:
                return f"{num:.1f}%" if num >= 10 else f"{num:.2f}%"
            elif num >= 1_000_000:
                return f"{num/1000:,.0f}".replace(",", ".")
            elif num >= 1000:
                return f"{num:,.0f}".replace(",", ".")
            else:
                return f"{num:.2f}".replace(".", ",")
        except Exception:
            return f"{value_str}%" if is_percentage else value_str

    # Patterns over normalized (accent-free, lowercase) names
    specs = [
        # Base monetaria
        (
            r"base\s*monetaria",
            lambda v: f"🏦 Base monetaria: ${format_value(v)} mill. pesos",
        ),
        # Inflación mensual: "Variación mensual del índice de precios al consumidor"
        (
            r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual",
            lambda v: f"📈 Inflación mensual: {format_value(v, True)}",
        ),
        # Inflación interanual: "Variación interanual del índice de precios al consumidor"
        (
            r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual",
            lambda v: f"📊 Inflación interanual: {format_value(v, True)}",
        ),
        # Inflación esperada (REM)
        (
            r"(mediana.*variacion.*interanual.*(12|doce).*meses.*(relevamiento.*expectativas.*mercado|rem)|inflacion.*esperada)",
            lambda v: f"🔮 Inflación esperada: {format_value(v, True)}",
        ),
        # Tasas
        (r"tamar", lambda v: f"📈 TAMAR: {format_value(v, True)}"),
        (r"badlar", lambda v: f"📊 BADLAR: {format_value(v, True)}"),
        # Tipo de cambio
        (
            r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor",
            lambda v: f"💵 Dólar minorista: ${v}",
        ),
        (r"tipo.*cambio.*mayorista", lambda v: f"💱 Dólar mayorista: ${v}"),
        # UVA/CER
        (r"unidad.*valor.*adquisitivo|\buva\b", lambda v: f"💰 UVA: ${v}"),
        (r"coeficiente.*estabilizacion.*referencia|\bcer\b", lambda v: f"📊 CER: {v}"),
        # Reservas
        (
            r"reservas.*internacionales",
            lambda v: f"🏛️ Reservas: USD {format_value(v)} millones",
        ),
    ]

    meta_info: Dict[str, Any] = {}
    if isinstance(variables, dict):
        candidate_meta = variables.get("_meta")
        if isinstance(candidate_meta, dict):
            meta_info = candidate_meta

    lines = ["📊 Variables principales BCRA\n"]
    latest_dt: Optional[datetime] = None
    for pattern, formatter in specs:
        compiled = re.compile(pattern)
        for key, data in variables.items():
            if str(key).startswith("_"):
                continue
            if compiled.search(norm(key)):
                value, date = data.get("value", ""), data.get("date", "")
                line = formatter(value)
                if date and date != value:
                    line += f" ({date.replace('/2025', '/25')})"
                lines.append(line)
                parsed_dt = parse_date_string(str(date))
                if parsed_dt and (latest_dt is None or parsed_dt > latest_dt):
                    latest_dt = parsed_dt
                break

    band_limits = get_currency_band_limits()
    if band_limits:
        lower = band_limits.get("lower")
        upper = band_limits.get("upper")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            date_label = band_limits.get("date")
            lower_text = fmt_num(float(lower), 2)
            upper_text = fmt_num(float(upper), 2)
            line = f"📏 Bandas cambiarias: piso ${lower_text} / techo ${upper_text}"
            if isinstance(date_label, str) and date_label:
                line += f" ({date_label})"
            lines.append(line)

    # Append current TCRM value with its sheet date (cached)
    try:
        details = get_latest_itcrm_value_and_date()
        if details:
            itcrm_value, date_str = details
            lines.append(
                f"📐 TCRM: {fmt_num(float(itcrm_value), 2)}"
                + (f" ({date_str})" if date_str else "")
            )
    except Exception:
        pass

    if meta_info.get("stale"):
        stale_msg = (
            "⚠️ No hay actualización nueva del BCRA, te muestro lo último que tengo."
        )
        if stale_msg not in lines:
            lines.append(stale_msg)

    if latest_dt:
        age_days = (datetime.now(BA_TZ).date() - latest_dt.date()).days
        if age_days >= 3:
            lines.append(
                f"⚠️ Datos del BCRA con {age_days} días de atraso, chequeá más tarde."
            )

    return "\n".join(lines)


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


def handle_transcribe_with_message(message: Dict) -> str:
    """Transcribe audio or describe image from replied message"""
    try:
        # Check if this is a reply to another message
        if "reply_to_message" not in message:
            return "Respondé a un mensaje con audio o imagen para transcribir/describir"

        replied_msg = message["reply_to_message"]

        # Check for audio in replied message
        if "voice" in replied_msg and replied_msg["voice"]:
            audio_file_id = replied_msg["voice"]["file_id"]
            text, err = transcribe_file_by_id(audio_file_id)
            if text:
                return f"🎵 Transcripción: {text}"
            return (
                "No pude descargar el audio"
                if err == "download"
                else "No pude transcribir el audio, intentá más tarde"
            )

        # Check for regular audio
        elif "audio" in replied_msg and replied_msg["audio"]:
            audio_file_id = replied_msg["audio"]["file_id"]
            text, err = transcribe_file_by_id(audio_file_id)
            if text:
                return f"🎵 Transcripción: {text}"
            return (
                "No pude descargar el audio"
                if err == "download"
                else "No pude transcribir el audio, intentá más tarde"
            )

        # Check for photo in replied message
        elif "photo" in replied_msg and replied_msg["photo"]:
            photo_file_id = replied_msg["photo"][-1]["file_id"]
            desc, err = describe_media_by_id(
                photo_file_id, "Describe what you see in this image in detail."
            )
            if desc:
                return f"🖼️ Descripción: {desc}"
            return (
                "No pude descargar la imagen"
                if err == "download"
                else "No pude describir la imagen, intentá más tarde"
            )

        # Check for sticker in replied message
        elif "sticker" in replied_msg and replied_msg["sticker"]:
            sticker_file_id = replied_msg["sticker"]["file_id"]
            desc, err = describe_media_by_id(
                sticker_file_id, "Describe what you see in this sticker in detail."
            )
            if desc:
                return f"🎨 Descripción del sticker: {desc}"
            return (
                "No pude descargar el sticker"
                if err == "download"
                else "No pude describir el sticker, intentá más tarde"
            )

        else:
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


def convert_to_command(msg_text: str) -> str:
    if not msg_text:
        return "y que queres que convierta boludo? mandate texto"

    # Convert emojis to their textual representation in Spanish with underscore delimiters
    emoji_text = emoji.demojize(msg_text, delimiters=("_", "_"), language="es")

    # Convert to uppercase and replace Ñ
    replaced_ni_text = re.sub(r"\bÑ\b", "ENIE", emoji_text.upper()).replace("Ñ", "NI")

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

- /devo 0.5, 100: te calculo el arbitraje entre tarjeta y crypto (fee%, monto opcional)

- /dolar, /dollar, /usd: te tiro la posta del blue y todos los dolares

- /instance: te digo donde estoy corriendo

- /links reply|delete|off: te arreglo los links de twitter/x/bsky/instagram/reddit/tiktok

- /prices, /precio, /precios, /presio, /presios: top 10 cryptos en usd
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


def send_typing(token: str, chat_id: str) -> None:
    parameters = {"chat_id": chat_id, "action": "typing"}
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    requests.get(url, params=parameters, timeout=5)


def send_msg(
    chat_id: str,
    msg: str,
    msg_id: str = "",
    buttons: Optional[List[str]] = None,
) -> None:
    token = environ.get("TELEGRAM_TOKEN")
    parameters = {"chat_id": chat_id, "text": msg}
    if msg_id != "":
        parameters["reply_to_message_id"] = msg_id
    if buttons:
        keyboard = [[{"text": "Open in app", "url": url}] for url in buttons]
        parameters["reply_markup"] = json.dumps({"inline_keyboard": keyboard})
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.get(url, params=parameters, timeout=5)


def delete_msg(chat_id: str, msg_id: str) -> None:
    """Delete a Telegram message"""
    token = environ.get("TELEGRAM_TOKEN")
    url = f"https://api.telegram.org/bot{token}/deleteMessage"
    parameters = {"chat_id": chat_id, "message_id": msg_id}
    requests.get(url, params=parameters, timeout=5)


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

        # If we have an image, first describe it with LLaVA then continue normal flow
        if image_data:
            print("Processing image with LLaVA model...")

            # Always use a description prompt for LLaVA, not the user's question
            user_text = "Describe what you see in this image in detail."

            # Describe the image using LLaVA
            image_description = describe_image_cloudflare(
                image_data, user_text, image_file_id
            )

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
        # First pass: get an initial response that might include a tool call.
        initial = complete_with_providers(system_message, messages)
        if initial:
            print(
                f"ask_ai: initial len={len(initial)} preview='{initial[:160].replace('\n',' ')}'"
            )

        # If the model asked to call a tool, execute and do a second pass
        tool_call = parse_tool_call(initial) if initial else None
        if tool_call:
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
                    "content": sanitize_tool_artifacts(initial),
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
                    tool_output = (
                        f"Error al ejecutar herramienta {tool_name}: {tool_err}"
                    )
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
                    # tool_output is JSON string with {query, results}
                    data = (
                        json.loads(last_tool_output)
                        if isinstance(last_tool_output, str)
                        else last_tool_output
                    )
                    query = data.get("query", "")
                    results = data.get("results", []) or []
                    if not results:
                        return f"no encontré resultados ahora con duckduckgo"
                    lines = [f"🔎 Resultados para: {query}"]
                    for i, r in enumerate(results[:5], 1):
                        title = r.get("title") or r.get("url") or "(sin título)"
                        url = r.get("url", "")
                        snippet = (r.get("snippet") or "").strip()
                        if snippet:
                            lines.append(f"{i}. {title}\n{url}\n{snippet[:300]}")
                        else:
                            lines.append(f"{i}. {title}\n{url}")
                    return "\n\n".join(lines)
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
                return (
                    "tuve un problema usando la herramienta, probá de nuevo más tarde"
                )

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
    """Try Groq, then OpenRouter, then Cloudflare and return the first response."""

    # Try Groq first
    if is_provider_backoff_active("groq"):
        remaining = int(get_provider_backoff_remaining("groq"))
        print(f"Groq backoff active ({remaining}s remaining), skipping Groq attempts")
    else:
        groq_response = get_groq_ai_response(system_message, messages)
        if groq_response:
            print("complete_with_providers: got response from Groq")
            return groq_response

    # Try OpenRouter second
    if is_provider_backoff_active("openrouter"):
        remaining = int(get_provider_backoff_remaining("openrouter"))
        print(
            "OpenRouter backoff active "
            f"({remaining}s remaining), skipping OpenRouter attempts"
        )
    else:
        openrouter_api_key = environ.get("OPENROUTER_API_KEY")
        if openrouter_api_key:
            openrouter = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            response = get_ai_response(
                openrouter, system_message, messages, provider_name="openrouter"
            )
            if response:
                print("complete_with_providers: got response from OpenRouter")
                return response
        else:
            print("OpenRouter API key not configured")

    # Fallback to Cloudflare Workers AI
    if is_provider_backoff_active("cloudflare"):
        remaining = int(get_provider_backoff_remaining("cloudflare"))
        print(
            "Cloudflare backoff active "
            f"({remaining}s remaining), skipping Cloudflare attempts"
        )
    else:
        cloudflare_response = get_cloudflare_ai_response(system_message, messages)
        if cloudflare_response:
            print("complete_with_providers: got response from Cloudflare AI")
            return cloudflare_response

    return None


def parse_tool_call(text: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Detect a tool call line like: [TOOL] web_search {"query": "..."}"""
    if not text:
        return None
    try:
        lines = text.splitlines()
        i = 0
        in_fence = False
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()

            # Toggle code fence context so [TOOL] inside code blocks is ignored
            if line.startswith("```"):
                in_fence = not in_fence
                i += 1
                continue

            if in_fence:
                i += 1
                continue

            # Normalize common list/code prefixes
            normalized = line
            normalized = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", normalized)
            normalized = normalized.strip().strip("`")

            marker_index = normalized.find("[TOOL]")
            if marker_index == -1:
                i += 1
                continue

            normalized = normalized[marker_index:]

            # Expected: [TOOL] name {json...}
            without_prefix = normalized[len("[TOOL]") :].strip()
            if without_prefix.startswith(":"):
                without_prefix = without_prefix[1:].strip()

            name_source_index = i
            if not without_prefix:
                j = i + 1
                while j < len(lines):
                    addition = lines[j].strip().strip("`")
                    addition = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", addition)
                    addition = addition.strip()
                    if addition:
                        without_prefix = addition
                        name_source_index = j
                        break
                    j += 1
                if not without_prefix:
                    i += 1
                    continue

            parts = without_prefix.split(" ", 1)
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

            # Gather JSON possibly spanning multiple lines
            json_candidate = remainder
            j = name_source_index + 1

            while "{" not in json_candidate and j < len(lines):
                addition = lines[j].strip().strip("`")
                addition = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", addition)
                addition = addition.strip()
                if addition:
                    json_candidate = (json_candidate + " " + addition).strip()
                j += 1

            if "{" not in json_candidate:
                i += 1
                continue

            open_count = json_candidate.count("{") - json_candidate.count("}")
            while open_count > 0 and j < len(lines):
                addition = lines[j].strip().strip("`")
                addition = re.sub(r"^(?:[-*+]|\d+\.)\s*", "", addition)
                addition = addition.strip()
                if addition:
                    json_candidate = (json_candidate + " " + addition).strip()
                    open_count += addition.count("{") - addition.count("}")
                j += 1

            closing_index = json_candidate.rfind("}")
            if closing_index == -1:
                i += 1
                continue

            json_text = json_candidate[: closing_index + 1].strip().rstrip(",")

            # Try parse as JSON first, then fallback to Python literal
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

    cache_key = None
    redis_client: Optional[redis.Redis] = None
    try:
        cache_payload = {"url": normalized, "max": requested_max}
        cache_key = (
            "fetch_url:"
            + hashlib.md5(
                json.dumps(cache_payload, sort_keys=True).encode()
            ).hexdigest()
        )
        redis_client = config_redis()
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
        if redis_client and cache_key:
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

    if redis_client and cache_key:
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

    # Try cache first
    cache_key = None
    redis_client = None
    try:
        # Cache key prefix: 'web_search'
        cache_key = f"web_search:{hashlib.md5(json.dumps({'q': query, 'limit': limit}).encode()).hexdigest()}"
        redis_client = config_redis()
        cached = redis_get_json(redis_client, cache_key)
        if isinstance(cached, list):
            return cached
    except Exception:
        # Cache unavailable or invalid; continue without failing
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
        try:
            if redis_client and cache_key:
                redis_setex_json(redis_client, cache_key, TTL_WEB_SEARCH, results)
        except Exception:
            pass

        return results
    except Exception:
        return []


def sanitize_tool_artifacts(text: Optional[str]) -> str:
    """Remove any visible [TOOL] lines or code blocks that contain them from model output."""
    if not text:
        return ""
    lines = text.splitlines()
    out_lines: List[str] = []
    in_fence = False
    block_lines: List[str] = []
    block_has_tool = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                # starting a code fence
                in_fence = True
                block_lines = [line]
                block_has_tool = False
            else:
                # closing a code fence
                block_lines.append(line)
                if not block_has_tool:
                    out_lines.extend(block_lines)
                # reset
                in_fence = False
                block_lines = []
                block_has_tool = False
            continue

        if in_fence:
            block_lines.append(line)
            if "[TOOL]" in line:
                block_has_tool = True
        else:
            # Outside code blocks: drop any line that contains [TOOL]
            if "[TOOL]" not in line:
                out_lines.append(line)

    # If an unterminated fence exists, keep it only if it didn't contain [TOOL]
    if in_fence and not block_has_tool:
        out_lines.extend(block_lines)

    return "\n".join(out_lines).strip()


def search_command(msg_text: Optional[str]) -> str:
    """/buscar command: perform a web search and return concise results"""
    q = (msg_text or "").strip()
    if not q:
        return "decime qué querés buscar capo"
    results = web_search(q, limit=10)
    if not results:
        return "no encontré resultados ahora con duckduckgo"
    lines = [f"🔎 Resultados para: {q}"]
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("url")
        url = r.get("url", "")
        snippet = (r.get("snippet") or "").strip()
        if snippet:
            lines.append(f"{i}. {title}\n{url}\n{snippet[:300]}")
        else:
            lines.append(f"{i}. {title}\n{url}")
    return "\n\n".join(lines[:10])


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

    redis_client: Optional[redis.Redis]
    try:
        redis_client = config_redis()
    except Exception:
        redis_client = None

    cached_items: Optional[List[Dict[str, Any]]] = None
    if redis_client:
        cached = redis_get_json(redis_client, HACKER_NEWS_CACHE_KEY)
        if isinstance(cached, list):
            cached_items = cached
            if cached_items:
                return cached_items[:limit]

    response_text: Optional[str] = None
    try:
        response = requests.get(HACKER_NEWS_RSS_URL, timeout=5)
        response.raise_for_status()
        response_text = response.text
    except SSLError:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InsecureRequestWarning)
                response = requests.get(HACKER_NEWS_RSS_URL, timeout=5, verify=False)
                response.raise_for_status()
                response_text = response.text
        except Exception as ssl_error:
            print(f"Error fetching Hacker News RSS (SSL fallback): {ssl_error}")
            return (cached_items or [])[:limit]
    except RequestException as request_error:
        print(f"Error fetching Hacker News RSS: {request_error}")
        return (cached_items or [])[:limit]

    if not response_text:
        return (cached_items or [])[:limit]

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
        dollar_response = cached_requests(
            "https://criptoya.com/api/dolar",
            None,
            None,
            TTL_DOLLAR,
        )
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


def get_ai_response(
    client: OpenAI,
    system_msg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    provider_name: str = "openrouter",
    backoff_seconds: Optional[int] = None,
) -> Optional[str]:
    """Get AI response (text-only) from a generic OpenAI-compatible provider."""

    if is_provider_backoff_active(provider_name):
        remaining = int(get_provider_backoff_remaining(provider_name))
        print(
            f"{provider_name.capitalize()} backoff active ({remaining}s remaining), skipping API call"
        )
        return None

    models = [
        "moonshotai/kimi-k2:free",
        "x-ai/grok-4-fast:free",
        "deepseek/deepseek-chat-v3.1:free",
    ]

    try:
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

    except Exception as e:
        print(f"API error: {e}")
        if _is_rate_limit_error(e):
            duration = (
                backoff_seconds
                if backoff_seconds is not None
                else OPENROUTER_RATE_LIMIT_BACKOFF_SECONDS
            )
            _set_provider_backoff(provider_name, duration)
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                f"{provider_name.capitalize()} rate limit detected; backing off for {remaining}s"
            )

    return None


def get_cloudflare_ai_response(
    system_msg: Dict[str, Any], messages: List[Dict[str, Any]]
) -> Optional[str]:
    """Fallback using Cloudflare Workers AI for text-only"""
    provider_name = "cloudflare"
    try:
        cloudflare_account_id = environ.get("CLOUDFLARE_ACCOUNT_ID")
        cloudflare_api_key = environ.get("CLOUDFLARE_API_KEY")

        if not cloudflare_account_id or not cloudflare_api_key:
            print("Cloudflare Workers AI credentials not configured")
            return None

        if is_provider_backoff_active(provider_name):
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                "Cloudflare backoff active "
                f"({remaining}s remaining), skipping API call"
            )
            return None

        print("Trying Cloudflare Workers AI as fallback...")
        cloudflare = OpenAI(
            api_key=cloudflare_api_key,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/v1",
        )

        final_messages = [system_msg] + messages

        response = cloudflare.chat.completions.create(
            model="@cf/mistralai/mistral-small-3.1-24b-instruct",
            messages=cast(Any, final_messages),
            max_tokens=1024,
        )

        if response and hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "stop":
                print("Cloudflare Workers AI response successful")
                return response.choices[0].message.content

    except Exception as e:
        print(f"Cloudflare Workers AI error: {e}")
        if _is_rate_limit_error(e):
            _set_provider_backoff(
                provider_name, CLOUDFLARE_RATE_LIMIT_BACKOFF_SECONDS
            )
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                "Cloudflare rate limit detected; backing off for "
                f"{remaining}s"
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
    try:
        groq_api_key = environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("Groq API key not configured")
            return None

        if is_provider_backoff_active(provider_name):
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                f"Groq backoff active ({remaining}s remaining), skipping API call"
            )
            return None

        print("Trying Groq AI as first option...")
        groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        final_messages = [system_msg] + messages

        response = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=cast(Any, final_messages),
            max_tokens=1024,
        )

        if response and hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "stop":
                print("Groq AI response successful")
                return response.choices[0].message.content

    except Exception as e:
        print(f"Groq AI error: {e}")
        if _is_rate_limit_error(e):
            _set_provider_backoff(provider_name, GROQ_RATE_LIMIT_BACKOFF_SECONDS)
            remaining = int(get_provider_backoff_remaining(provider_name))
            print(
                "Groq rate limit detected; backing off for "
                f"{remaining}s"
            )

    return None


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
            "Usá herramientas solo si realmente ayudan (actualidad, datos frescos)."
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
    message: Dict, chat_history: List[Dict], message_text: str
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
        "\nMENSAJE:",
        truncate_text(message_text),
        "\nINSTRUCCIONES:",
        "- Mantené el personaje del gordo",
        "- Usá lenguaje coloquial argentino",
    ]

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
        "/convertbase": (convert_base, False, True),
        "/random": (select_random, False, True),
        "/prices": (get_prices, False, True),
        "/precios": (get_prices, False, True),
        "/precio": (get_prices, False, True),
        "/presios": (get_prices, False, True),
        "/presio": (get_prices, False, True),
        "/dolar": (get_dollar_rates, False, False),
        "/dollar": (get_dollar_rates, False, False),
        "/usd": (get_dollar_rates, False, False),
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
        "/links": (lambda params: "", False, True),
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
            "vxtiktok.com",
        )
        if any(domain in reply_text for domain in replacement_domains):
            return False

    # Response conditions
    is_command = command in commands
    is_private = chat_type == "private"
    is_mention = bot_name in message_lower
    is_reply = reply.get("from", {}).get("username", "") == bot_username

    # Check trigger keywords with 10% chance
    config = load_bot_config()
    trigger_words = config.get("trigger_words", ["bot", "assistant"])
    is_trigger = (
        any(word in message_lower for word in trigger_words) and random.random() < 0.1
    )

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


def describe_image_cloudflare(
    image_data: bytes,
    user_text: str = "¿Qué ves en esta imagen?",
    file_id: Optional[str] = None,
) -> Optional[str]:
    """Describe image using Cloudflare Workers AI LLaVA model"""
    try:
        # Check cache first if file_id is provided
        if file_id:
            cached_description = get_cached_description(file_id)
            if cached_description:
                return cached_description

        cloudflare_account_id = environ.get("CLOUDFLARE_ACCOUNT_ID")
        cloudflare_api_key = environ.get("CLOUDFLARE_API_KEY")

        if not cloudflare_account_id or not cloudflare_api_key:
            print("Cloudflare Workers AI credentials not configured")
            return None

        url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
        headers = {
            "Authorization": f"Bearer {cloudflare_api_key}",
            "Content-Type": "application/json",
        }

        # Convert bytes to array of integers (0-255) as expected by LLaVA
        image_array = list(image_data)

        payload = {"prompt": user_text, "image": image_array, "max_tokens": 1024}

        print(f"Describing image with LLaVA model...")
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                # Try both possible response formats
                if "response" in result["result"]:
                    description = result["result"]["response"]
                elif "description" in result["result"]:
                    description = result["result"]["description"]
                else:
                    print(f"Unexpected LLaVA response format: {result}")
                    return None

                print(f"Image description successful: {description[:100]}...")

                # Cache the description if file_id is provided
                if file_id and description:
                    cache_description(file_id, description)

                return description
            else:
                print(f"Unexpected LLaVA response format: {result}")
        else:
            error_text = response.text
            print(f"LLaVA API error {response.status_code}: {error_text}")

            # Parse error details if available
            try:
                error_json = response.json()
                if "errors" in error_json and error_json["errors"]:
                    pass
            except:
                pass

    except Exception as e:
        print(f"Error describing image: {e}")

    return None


def transcribe_audio_cloudflare(
    audio_data: bytes, file_id: Optional[str] = None
) -> Optional[str]:
    """Transcribe audio using Cloudflare Workers AI Whisper with retry"""

    # Check cache first if file_id is provided
    if file_id:
        cached_transcription = get_cached_transcription(file_id)
        if cached_transcription:
            return cached_transcription

    cloudflare_account_id = environ.get("CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_key = environ.get("CLOUDFLARE_API_KEY")

    if not cloudflare_account_id or not cloudflare_api_key:
        print("Cloudflare credentials not configured for audio transcription")
        return None

    print("Transcribing audio with Cloudflare Whisper...")

    # Use direct API call to Cloudflare Workers AI for Whisper
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{cloudflare_account_id}/ai/run/@cf/openai/whisper-large-v3-turbo"
    )
    headers = {
        "Authorization": f"Bearer {cloudflare_api_key}",
        "Content-Type": "application/octet-stream",
    }

    # Retry logic for timeout issues
    for attempt in range(2):  # 2 attempts total (original + 1 retry)
        try:
            if attempt > 0:
                print(f"Retrying audio transcription (attempt {attempt + 1}/2)...")
                time.sleep(2)  # Wait 2 seconds before retry

            response = requests.post(url, headers=headers, data=audio_data, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get("success") and "result" in result:
                transcription = result["result"].get("text", "")
                print(f"Audio transcribed successfully: {transcription[:100]}...")

                # Cache the transcription if file_id is provided
                if file_id and transcription:
                    cache_transcription(file_id, transcription)

                return transcription

            print(f"Whisper transcription failed: {result}")
            return None

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if "timeout" in str(e).lower() or "408" in str(e):
                print(f"Audio transcription timeout on attempt {attempt + 1}/2: {e}")
                if attempt == 1:  # Last attempt
                    print("Audio transcription failed after all retries")
                    return None
                # Continue to retry
            else:
                # Non-timeout error, don't retry
                print(f"Error transcribing audio: {e}")
                return None
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    return None


def transcribe_file_by_id(
    file_id: str, use_cache: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch transcription for a Telegram file_id with cache and retries.

    Returns (text, error):
    - On success: (transcription, None)
    - If download failed: (None, "download")
    - If transcription failed: (None, "transcribe")
    """
    try:
        if use_cache:
            cached = get_cached_transcription(file_id)
            if cached:
                return str(cached), None

        audio_data = download_telegram_file(file_id)
        if not audio_data:
            return None, "download"

        text = transcribe_audio_cloudflare(audio_data, file_id)
        if text:
            return text, None
        return None, "transcribe"
    except Exception as e:
        print(f"Error in transcribe_file_by_id: {e}")
        return None, "transcribe"


def describe_media_by_id(
    file_id: str, prompt: str
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch description for an image/sticker by Telegram file_id using LLaVA.

    Returns (description, error):
    - On success: (description, None)
    - If download failed: (None, "download")
    - If description failed: (None, "describe")
    """
    try:
        cached = get_cached_description(file_id)
        if cached:
            return str(cached), None

        image_data = download_telegram_file(file_id)
        if not image_data:
            return None, "download"

        resized = resize_image_if_needed(image_data)
        desc = describe_image_cloudflare(resized, prompt, file_id)
        if desc:
            return desc, None
        return None, "describe"
    except Exception as e:
        print(f"Error in describe_media_by_id: {e}")
        return None, "describe"


def resize_image_if_needed(image_data: bytes, max_size: int = 512) -> bytes:
    """Resize image if it's too large for LLaVA processing"""
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

    # Get message text and handle extra spaces
    if len(split_message) > 1:
        message_text = split_message[1].lstrip()  # Remove leading spaces only
    else:
        message_text = ""

    return command, message_text


def can_embed_url(url: str) -> bool:
    """Return True if URL has metadata for embed previews"""
    headers = {"User-Agent": "TelegramBot (like TwitterBot)"}
    try:
        response = requests.get(url, allow_redirects=True, timeout=5, headers=headers)
    except SSLError:
        try:
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            response = requests.get(
                url, allow_redirects=True, timeout=5, verify=False, headers=headers
            )
        except RequestException as e:
            print(f"[EMBED] {url} request failed after SSL error: {e}")
            return False
    except RequestException as e:
        print(f"[EMBED] {url} request failed: {e}")
        return False
    if response.status_code >= 400:
        print(f"[EMBED] {url} returned status {response.status_code}")
        return False
    content_type = response.headers.get("Content-Type", "").lower()
    if content_type.startswith(("image/", "video/", "audio/")):
        print(f"[EMBED] {url} served direct media type {content_type}")
        return True
    if "text/html" not in content_type:
        print(f"[EMBED] {url} content-type {content_type} not embeddable")
        return False
    html = response.text[:20000]

    class MetaParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            if tag != "meta":
                return
            attrs_dict = dict(attrs)
            key = attrs_dict.get("property") or attrs_dict.get("name")
            if not key:
                return
            key_lower = key.lower()
            if key_lower.startswith("og:") or key_lower.startswith("twitter:"):
                self.tags.append((key, attrs_dict.get("content", "")))

    parser = MetaParser()
    parser.feed(html)
    meta_tags = parser.tags
    has_meta = bool(meta_tags)
    if not has_meta:
        print(f"[EMBED] {url} missing og/twitter meta tags")
    else:
        detail = ", ".join(f"{k}={v[:80]}" for k, v in meta_tags)
        print(f"[EMBED] {url} has embed metadata: {detail}")
    return has_meta


def url_is_embedable(url: str) -> bool:
    """Check if a URL is likely to produce an embed preview"""
    return can_embed_url(url)


def replace_links(text: str) -> Tuple[str, bool, List[str]]:
    """Replace social links with alternative frontends"""

    patterns = [
        (r"(https?://)(?:www\.)?twitter\.com([^\s]*)", r"\1fxtwitter.com\2"),
        (r"(https?://)(?:www\.)?x\.com([^\s]*)", r"\1fixupx.com\2"),
        (r"(https?://)(?:www\.)?xcancel\.com([^\s]*)", r"\1fixupx.com\2"),
        (r"(https?://)(?:www\.)?bsky\.app([^\s]*)", r"\1fxbsky.app\2"),
        (r"(https?://)(?:www\.)?instagram\.com([^\s]*)", r"\1kkinstagram.com\2"),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)reddit\.com([^\s]*)",
            r"\1\2rxddit.com\3",
        ),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)tiktok\.com([^\s]*)",
            r"\1\2vxtiktok.com\3",
        ),
    ]

    changed = False
    original_links: List[str] = []

    def make_sub(repl: str):
        def _sub(match: re.Match) -> str:
            original = match.group(0)
            replaced = match.expand(repl)
            parsed = urlparse(replaced)
            cleaned = parsed._replace(query="", fragment="")
            replaced_full = urlunparse(cleaned)
            if url_is_embedable(replaced_full):
                nonlocal changed
                changed = True
                parsed_original = urlparse(original)
                cleaned_original = parsed_original._replace(query="", fragment="")
                original_links.append(urlunparse(cleaned_original))
                print(f"[LINK] replacing {original} with {replaced_full}")
                return replaced_full
            print(f"[LINK] cannot embed {replaced_full}, keeping {original}")
            return original

        return _sub

    new_text = text
    for pattern, repl in patterns:
        new_text = re.sub(pattern, make_sub(repl), new_text, flags=re.IGNORECASE)

    url_pattern = re.compile(r"(https?://[^\s]+)")

    def strip_tracking(match: re.Match) -> str:
        url = match.group(0)
        parsed = urlparse(url)
        if is_social_frontend(parsed.netloc):
            cleaned = parsed._replace(query="", fragment="")
            return urlunparse(cleaned)
        return url

    new_text = url_pattern.sub(strip_tracking, new_text)

    return new_text, changed, original_links


def configure_links(chat_id: str, params: str) -> str:
    """Configure automatic link fixing for a chat"""
    redis_client = config_redis()
    key = f"link_mode:{chat_id}"
    mode = params.strip().lower()

    if mode in {"reply", "delete"}:
        redis_client.set(key, mode)
        return f"Link fixer set to {mode} mode"
    if mode == "off":
        redis_client.delete(key)
        return "Link fixer disabled"

    current = redis_client.get(key) or "off"
    return f"Usage: /links reply|delete|off (current: {current})"


def format_user_message(message: Dict, message_text: str) -> str:
    """Format message with user info"""
    username = message["from"].get("username", "")
    first_name = message["from"].get("first_name", "")

    # Handle None values
    first_name = "" if first_name is None else first_name
    username = "" if username is None else username

    formatted_user = f"{first_name}" + (f" ({username})" if username else "")
    return f"{formatted_user}: {message_text}"


def handle_msg(message: Dict) -> str:
    try:
        # Extract multimedia content
        message_text, photo_file_id, audio_file_id = extract_message_content(message)
        message_id = str(message["message_id"])
        chat_id = str(message["chat"]["id"])

        # Process audio first if present (but not for /transcribe commands)
        if audio_file_id and not (
            message_text and message_text.strip().lower().startswith("/transcribe")
        ):
            print(f"Processing audio message: {audio_file_id}")
            transcription, err = transcribe_file_by_id(audio_file_id, use_cache=False)
            if transcription:
                message_text = transcription
                print(f"Audio transcribed: {message_text[:100]}...")
            else:
                message_text = (
                    "no pude bajar tu audio, mandalo de vuelta"
                    if err == "download"
                    else "mandame texto que no soy alexa, boludo"
                )

        # Download image if present
        image_base64 = None
        resized_image_data = None
        if photo_file_id:
            print(f"Processing image message: {photo_file_id}")
            image_data = download_telegram_file(photo_file_id)
            if image_data:
                # Resize image if needed for LLaVA compatibility
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

        link_mode = redis_client.get(f"link_mode:{chat_id}")
        if link_mode and message_text and not message_text.startswith("/"):
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

        # Check if we should respond
        if not should_gordo_respond(commands, command, sanitized_message_text, message):
            # Even if we don't respond, save the message for context
            if message_text:
                formatted_message = format_user_message(message, message_text)
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
        if command in commands:
            handler_func, uses_ai, takes_params = commands[command]

            if uses_ai:
                if not check_rate_limit(chat_id, redis_client):
                    response_msg = handle_rate_limit(chat_id, message)
                else:
                    # Get chat history BEFORE saving the current message
                    chat_history = get_chat_history(chat_id, redis_client)
                    messages = build_ai_messages(
                        message, chat_history, sanitized_message_text
                    )
                    response_msg = handle_ai_response(
                        chat_id,
                        handler_func,
                        messages,
                        image_data=resized_image_data if photo_file_id else None,
                        image_file_id=photo_file_id or None,
                    )
            else:
                # Special handling for transcribe command which needs the full message
                if command == "/transcribe":
                    response_msg = handle_transcribe_with_message(message)
                elif command == "/links":
                    response_msg = configure_links(chat_id, sanitized_message_text)
                else:
                    if takes_params:
                        response_msg = handler_func(sanitized_message_text)
                    else:
                        response_msg = handler_func()
        else:
            if not check_rate_limit(chat_id, redis_client):
                response_msg = handle_rate_limit(chat_id, message)
            else:
                # Get chat history BEFORE saving the current message
                chat_history = get_chat_history(chat_id, redis_client)
                messages = build_ai_messages(message, chat_history, message_text)
                response_msg = handle_ai_response(
                    chat_id,
                    ask_ai,
                    messages,
                    image_data=resized_image_data if photo_file_id else None,
                    image_file_id=photo_file_id or None,
                )

        # Only save messages AFTER we've generated a response
        if message_text:
            formatted_message = format_user_message(message, message_text)
            save_message_to_redis(chat_id, message_id, formatted_message, redis_client)

        # Save and send response
        if response_msg:
            # Save bot response
            save_message_to_redis(
                chat_id, f"bot_{message_id}", response_msg, redis_client
            )
            send_msg(chat_id, response_msg, message_id)

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


def handle_ai_response(
    chat_id: str,
    handler_func: Callable,
    messages: List[Dict],
    image_data: Optional[bytes] = None,
    image_file_id: Optional[str] = None,
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

    # Clean any duplicate text
    cleaned_response = clean_duplicate_response(persona_stripped_response)
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
    request_url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    try:
        telegram_response = requests.get(request_url, timeout=5)
        telegram_response.raise_for_status()
    except RequestException as request_error:
        return {"error": str(request_error)}
    return telegram_response.json()["result"]


def set_telegram_webhook(webhook_url: str) -> bool:
    webhook_key = environ.get("WEBHOOK_AUTH_KEY")
    token = environ.get("TELEGRAM_TOKEN")
    secret_token = hashlib.sha256(Fernet.generate_key()).hexdigest()
    parameters = {
        "url": f"{webhook_url}?key={webhook_key}",
        "allowed_updates": '["message"]',
        "secret_token": secret_token,
        "max_connections": 8,
    }
    request_url = f"https://api.telegram.org/bot{token}/setWebhook"
    try:
        telegram_response = requests.get(request_url, params=parameters, timeout=5)
        telegram_response.raise_for_status()
    except RequestException:
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


def process_request_parameters(request: Request) -> Tuple[str, int]:
    try:
        # Handle webhook checks
        check_webhook = request.args.get("check_webhook") == "true"
        if check_webhook:
            webhook_verified = verify_webhook()
            return (
                ("Webhook checked", 200)
                if webhook_verified
                else ("Webhook check error", 400)
            )

        # Handle webhook updates
        update_webhook = request.args.get("update_webhook") == "true"
        if update_webhook:
            function_url = environ.get("FUNCTION_URL")
            if not function_url:
                return ("Webhook update error", 400)
            return (
                ("Webhook updated", 200)
                if set_telegram_webhook(function_url)
                else ("Webhook update error", 400)
            )

        # Handle dollar rates update
        update_dollars = request.args.get("update_dollars") == "true"
        if update_dollars:
            get_dollar_rates()
            return "Dollars updated", 200

        run_agent = request.args.get("run_agent") == "true"
        if run_agent:
            try:
                thought_result = run_agent_cycle()
                return (
                    json.dumps(
                        {"status": "ok", "thought": thought_result}, ensure_ascii=False
                    ),
                    200,
                )
            except Exception as agent_error:
                admin_report("Agent run failed", agent_error)
                return "Agent run failed", 500

        # Validate secret token
        if not is_secret_token_valid(request):
            admin_report("Wrong secret token")
            return "Wrong secret token", 400

        # Process message
        request_json = request.get_json(silent=True)
        if not request_json:
            return "Invalid JSON", 400
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
