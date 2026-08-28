"""Resolve current OpenRouter endpoint prices for routed providers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any
from urllib.parse import quote

from api.ai.pricing import ensure_mapping
from api.services import http_client


_CACHE_TTL_SECONDS = 60 * 60
_ERROR_CACHE_TTL_SECONDS = 60
_REQUEST_TIMEOUT_SECONDS = 3.0
_CACHE_LOCK = threading.Lock()
_MODEL_ENDPOINT_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}


def needs_published_provider_pricing(usage: Mapping[str, Any] | None) -> bool:
    """Return whether reported usage lacks a positive undiscounted provider cost."""

    usage_map = dict(usage or {})
    cost_details = ensure_mapping(usage_map.get("cost_details")) or {}
    if "upstream_inference_cost" in cost_details:
        return not _is_positive_decimal(cost_details.get("upstream_inference_cost"))
    return not _is_positive_decimal(usage_map.get("cost"))


def get_openrouter_provider_pricing(
    model: str,
    upstream_provider: str,
    *,
    base_url: str,
    request_get: Callable[..., Any] = http_client.get,
    now: datetime | None = None,
) -> dict[str, int] | None:
    """Fetch the current price for one concrete OpenRouter model endpoint."""

    normalized_model = str(model or "").strip()
    normalized_provider = str(upstream_provider or "").strip().casefold()
    if not normalized_model or not normalized_provider or normalized_model.startswith("~"):
        return None

    payload = _get_model_endpoints(
        normalized_model,
        base_url=base_url,
        request_get=request_get,
    )
    data = ensure_mapping(payload.get("data")) if payload else None
    endpoints = data.get("endpoints") if data else None
    if not isinstance(endpoints, list):
        return None

    matching_prices: list[dict[str, int]] = []
    for raw_endpoint in endpoints:
        endpoint = ensure_mapping(raw_endpoint) or {}
        provider_name = str(endpoint.get("provider_name") or "").strip().casefold()
        if provider_name != normalized_provider:
            continue
        pricing = _current_endpoint_pricing(
            ensure_mapping(endpoint.get("pricing")) or {},
            now=now or datetime.now(UTC),
        )
        normalized = _normalize_endpoint_pricing(pricing)
        if normalized:
            matching_prices.append(normalized)

    unique_prices = {tuple(sorted(item.items())) for item in matching_prices}
    if len(unique_prices) != 1:
        # Provider name alone cannot identify one price when it has several variants.
        return None
    return dict(unique_prices.pop())


def _get_model_endpoints(
    model: str,
    *,
    base_url: str,
    request_get: Callable[..., Any],
) -> dict[str, Any] | None:
    now_monotonic = time.monotonic()
    normalized_base_url = str(base_url or "").rstrip("/")
    cache_key = (normalized_base_url, model)
    with _CACHE_LOCK:
        cached = _MODEL_ENDPOINT_CACHE.get(cache_key)
    if cached is not None:
        cached_at, payload = cached
        ttl = _CACHE_TTL_SECONDS if payload is not None else _ERROR_CACHE_TTL_SECONDS
        if now_monotonic - cached_at < ttl:
            return payload

    endpoint_url = f"{normalized_base_url}/models/{quote(model, safe='/')}/endpoints"
    try:
        response = request_get(endpoint_url, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        raw_payload = response.json()
        payload = ensure_mapping(raw_payload)
    except Exception:
        payload = None
    with _CACHE_LOCK:
        _MODEL_ENDPOINT_CACHE[cache_key] = (now_monotonic, payload)
    return payload


def _current_endpoint_pricing(
    pricing: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    current = dict(pricing)
    overrides = pricing.get("overrides")
    if not isinstance(overrides, list):
        return current
    for raw_override in overrides:
        override = ensure_mapping(raw_override) or {}
        if _override_is_active(override, now=now):
            current.update(override)
            break
    return current


def _override_is_active(override: Mapping[str, Any], *, now: datetime) -> bool:
    utc_now = now.astimezone(UTC)
    days = override.get("utc_days")
    if isinstance(days, list):
        normalized_days = {str(day).strip().casefold() for day in days}
        if utc_now.strftime("%A").casefold() not in normalized_days:
            return False

    if "utc_start" not in override and "utc_end" not in override:
        return True
    try:
        start = int(override.get("utc_start") or 0)
        end = int(override.get("utc_end") or 0)
    except TypeError, ValueError:
        return False
    current = utc_now.hour * 100 + utc_now.minute
    if start == end:
        active = True
    elif end == 0:
        active = current >= start
    elif start < end:
        active = start <= current < end
    else:
        active = current >= start or current < end
    return active


def _normalize_endpoint_pricing(pricing: Mapping[str, Any]) -> dict[str, int]:
    field_map = {
        "prompt": "input_per_million",
        "completion": "output_per_million",
        "input_cache_read": "cached_input_per_million",
        "input_cache_write": "cache_write_per_million",
        "audio": "audio_input_per_million",
    }
    normalized: dict[str, int] = {}
    for source_key, target_key in field_map.items():
        value = _rate_per_token_to_micros_per_million(pricing.get(source_key))
        if value is not None:
            normalized[target_key] = value
    request_cost = _usd_to_micros(pricing.get("request"))
    if request_cost is not None:
        normalized["request_usd_micros"] = request_cost
    if "input_per_million" not in normalized or "output_per_million" not in normalized:
        return {}
    return normalized


def _rate_per_token_to_micros_per_million(value: Any) -> int | None:
    try:
        rate = Decimal(str(value)) * Decimal(1_000_000_000_000)
    except InvalidOperation, TypeError, ValueError:
        return None
    if rate < 0:
        return None
    return int(rate.to_integral_value(rounding=ROUND_CEILING))


def _usd_to_micros(value: Any) -> int | None:
    try:
        cost = Decimal(str(value)) * Decimal(1_000_000)
    except InvalidOperation, TypeError, ValueError:
        return None
    if cost < 0:
        return None
    return int(cost.to_integral_value(rounding=ROUND_CEILING))


def _is_positive_decimal(value: Any) -> bool:
    try:
        return Decimal(str(value)) > 0
    except InvalidOperation, TypeError, ValueError:
        return False


def clear_openrouter_pricing_cache() -> None:
    """Clear cached endpoint data. Intended for deterministic tests."""

    with _CACHE_LOCK:
        _MODEL_ENDPOINT_CACHE.clear()
