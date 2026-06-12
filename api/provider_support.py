from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from logging import Logger
from typing import Any

from api.ai_pricing import AIUsageResult


def invoke_provider(
    provider_name: str,
    *,
    attempt: Callable[[], Any | None],
    rate_limit_backoff: int | None,
    label: str | None,
    backoff_key: str | None,
    is_backoff_active: Callable[[str], bool],
    get_backoff_remaining: Callable[[str], float],
    is_rate_limit_error: Callable[[Exception], bool],
    extract_backoff_seconds: Callable[[Exception, int | None], int | None],
    set_backoff: Callable[[str, int | None], None],
    default_backoff: int,
) -> Any | None:
    display_name = label or provider_name.capitalize()
    normalized_key = str(backoff_key or provider_name or "").lower()
    if normalized_key and is_backoff_active(normalized_key):
        remaining = int(get_backoff_remaining(normalized_key))
        print(f"{display_name} backoff active ({remaining}s remaining), skipping API call")
        return None
    try:
        return attempt()
    except Exception as error:
        print(f"{display_name} error: {error}")
        if is_rate_limit_error(error):
            seconds = extract_backoff_seconds(
                error, rate_limit_backoff or default_backoff
            )
            set_backoff(normalized_key or provider_name, seconds)
            remaining = int(get_backoff_remaining(normalized_key or provider_name))
            print(f"{display_name} rate limit detected; backing off for {remaining}s")
        return None


def append_billing_segment(
    response_meta: dict[str, Any] | None,
    result: AIUsageResult | None,
) -> None:
    if response_meta is not None and result is not None:
        response_meta.setdefault("billing_segments", []).append(
            result.billing_segment()
        )


def log_groq_request_result(
    *,
    label: str,
    scope: str,
    account: str,
    token_count: int,
    audio_seconds: float,
    result: AIUsageResult | None,
    calculate_billing: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ensure_mapping: Callable[[Any], dict[str, Any] | None],
    logger: Logger,
) -> None:
    entry: dict[str, Any] = {
        "scope": "groq_request",
        "label": label,
        "request_scope": scope,
        "account": account,
        "estimated_token_count": int(max(0, token_count)),
        "estimated_audio_seconds": float(max(0.0, audio_seconds)),
        "status": "success" if result else "empty",
    }
    if result is not None:
        billing = calculate_billing([result.billing_segment()])
        entry.update(
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
    logger.info(
        "groq request: %s",
        json.dumps(entry, ensure_ascii=False, default=str),
    )


def extract_usage_map(
    response: Any,
    *,
    ensure_mapping: Callable[[Any], dict[str, Any] | None],
) -> dict[str, Any] | None:
    value = response.get("usage") if isinstance(response, dict) else getattr(
        response, "usage", None
    )
    return ensure_mapping(value)


def extract_latest_user_query_info(
    messages: Sequence[Mapping[str, Any]],
    *,
    extract_message: Callable[[str], str],
) -> tuple[str, str, int | None]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content, extract_message(content), index
    return "", "", None


def build_usage_result(
    *,
    kind: str,
    text: str,
    model: str,
    response: Any,
    audio_seconds: float | None,
    cached: bool,
    metadata: Mapping[str, Any] | None,
    extract_usage: Callable[[Any], dict[str, Any] | None],
) -> AIUsageResult:
    return AIUsageResult(
        kind=kind,
        text=text,
        model=model,
        usage=extract_usage(response),
        audio_seconds=audio_seconds,
        cached=cached,
        metadata=dict(metadata or {}),
    )
