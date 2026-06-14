"""Normalize provider failures into retry and fallback decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from requests.exceptions import RequestException


def extract_error_headers(error: Exception) -> dict[str, str]:
    possible_headers = getattr(error, "headers", None)
    response = getattr(error, "response", None)
    if not possible_headers and response is not None:
        possible_headers = getattr(response, "headers", None)
    if not possible_headers:
        return {}
    if isinstance(possible_headers, Mapping):
        return {
            str(key).lower(): str(value)
            for key, value in possible_headers.items()
        }
    try:
        return {
            str(key).lower(): str(value)
            for key, value in dict(possible_headers).items()
        }
    except Exception:
        return {}


def parse_retry_window_seconds(value: str | None) -> int | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        return max(0, int(float(raw_value)))
    except (TypeError, ValueError):
        pass

    match = re.fullmatch(
        r"(?P<amount>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)?",
        raw_value.lower(),
    )
    if match:
        amount = float(match.group("amount"))
        multiplier = {
            "ms": 0.001,
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
        }.get(match.group("unit") or "s", 1.0)
        return max(0, int(amount * multiplier))

    try:
        parsed = parsedate_to_datetime(raw_value)
    except (IndexError, KeyError, RequestException, TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, int((parsed - datetime.now(UTC)).total_seconds()))


def extract_rate_limit_backoff_seconds(
    error: Exception,
    fallback_seconds: int | None = None,
) -> int | None:
    headers = extract_error_headers(error)
    for name in (
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ):
        parsed = parse_retry_window_seconds(headers.get(name))
        if parsed is not None:
            return parsed
    return fallback_seconds


def is_rate_limit_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 429:
        return True
    if getattr(error, "status", None) == 429:
        return True
    message = str(getattr(error, "message", "") or error).lower()
    return "rate limit" in message or "429" in message


def should_try_next_groq_account(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 413:
        return True
    if getattr(error, "status", None) == 413:
        return True
    if str(getattr(error, "code", "") or "").strip().lower() == "request_too_large":
        return True
    message = str(getattr(error, "message", "") or error).lower()
    return "request_too_large" in message or "payload too large" in message
