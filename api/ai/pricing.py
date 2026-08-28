"""AI pricing, usage normalization, and credit calculations."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from api.billing.credit_units import CREDIT_SCALE, format_credit_units

PRICING_VERSION = "2026-08-28"
CREDIT_USD_MICROS = 10_000
BILLING_MARKUP_MULTIPLIER = 2.0
CREDIT_CEIL_DIVISOR_USD_MICROS = int(CREDIT_USD_MICROS / BILLING_MARKUP_MULTIPLIER)
CREDIT_UNIT_USD_MICROS = CREDIT_CEIL_DIVISOR_USD_MICROS // CREDIT_SCALE

CHAT_OUTPUT_TOKEN_LIMIT = 1024
REASONING_CHAT_OUTPUT_TOKEN_LIMIT = 8192
VISION_OUTPUT_TOKEN_LIMIT = 512
IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE = 1_200
SYSTEM_CONTEXT_EXTRA_TOKENS_ESTIMATE = 4_000
FIRECRAWL_STANDARD_PLAN_USD_MICROS = 83_000_000
FIRECRAWL_STANDARD_PLAN_CREDITS = 100_000
FIRECRAWL_USD_MICROS_PER_CREDIT = (
    FIRECRAWL_STANDARD_PLAN_USD_MICROS // FIRECRAWL_STANDARD_PLAN_CREDITS
)
FIRECRAWL_SEARCH_MAX_CREDITS = 2


MODEL_PRICING_USD_MICROS: Dict[str, Dict[str, int]] = {
    "whisper-large-v3": {
        "audio_per_hour": 111_000,
    },
    "groq/whisper-large-v3": {
        "audio_per_hour": 111_000,
    },
    "google/gemini-3.1-flash-lite-preview": {
        "input_per_million": 250_000,
        "cached_input_per_million": 25_000,
        "cache_write_per_million": 83_333,
        "audio_input_per_million": 500_000,
        "output_per_million": 1_500_000,
    },
    "deepseek/deepseek-v4-flash-0731": {
        "input_per_million": 30_000,
        "cached_input_per_million": 7_000,
        "output_per_million": 100_000,
    },
}

PROVIDER_MODEL_PRICING_USD_MICROS: Dict[tuple[str, str], Dict[str, int]] = {
    ("groq", "openai/gpt-oss-120b"): {
        "input_per_million": 150_000,
        "cached_input_per_million": 75_000,
        "output_per_million": 600_000,
    },
}


def chat_output_token_limit(model: str) -> int:
    """Return a larger budget only for chat models that use hidden reasoning."""

    if str(model or "").split(":", 1)[0] == "deepseek/deepseek-v4-flash-0731":
        return REASONING_CHAT_OUTPUT_TOKEN_LIMIT
    return CHAT_OUTPUT_TOKEN_LIMIT


@dataclass
class AIUsageResult:
    """Structured AI response with billing metadata."""

    kind: str
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    audio_seconds: Optional[float] = None
    cached: bool = False
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def billing_segment(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of SDK response fragments into plain dicts."""

    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        data = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        if data:
            return data
    return None


def estimate_text_tokens(text: Optional[str]) -> int:
    """Approximate token count from text length."""

    if not text:
        return 0
    return max(1, math.ceil(len(str(text)) / 4))


def estimate_nested_tokens(value: Any) -> int:
    """Approximate token count for nested chat/response payload values."""

    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_text_tokens(value)
    if isinstance(value, Mapping):
        total = 0
        for item in value.values():
            total += estimate_nested_tokens(item)
        return total
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        total = 0
        for item in value:
            total += estimate_nested_tokens(item)
        return total
    return estimate_text_tokens(str(value))


def estimate_message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Approximate token count for a chat message list."""

    total = 0
    for message in messages:
        total += estimate_nested_tokens(message.get("role"))
        total += estimate_nested_tokens(message.get("content"))
        total += estimate_nested_tokens(message.get("name"))
    return total


def estimate_chat_reserve_credits(
    *,
    system_message: Optional[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    max_output_tokens: Optional[int] = None,
    extra_input_tokens: int = 0,
    model: str = "deepseek/deepseek-v4-flash-0731",
) -> int:
    pricing = MODEL_PRICING_USD_MICROS.get(
        model, MODEL_PRICING_USD_MICROS["deepseek/deepseek-v4-flash-0731"]
    )
    output_token_limit = (
        chat_output_token_limit(model) if max_output_tokens is None else max_output_tokens
    )
    input_tokens = estimate_message_tokens(messages) + extra_input_tokens
    if system_message:
        input_tokens += estimate_message_tokens([system_message])
    usd_micros = (
        input_tokens * pricing["input_per_million"]
        + output_token_limit * pricing["output_per_million"]
    ) // 1_000_000
    return credit_units_from_usd_micros(usd_micros)


def estimate_vision_reserve_credits(
    *,
    prompt_text: str,
    image_data: Optional[bytes] = None,
    extra_input_tokens: int = 0,
    max_output_tokens: int = VISION_OUTPUT_TOKEN_LIMIT,
    model: str = "google/gemini-3.1-flash-lite-preview",
) -> int:
    pricing = MODEL_PRICING_USD_MICROS.get(
        model, MODEL_PRICING_USD_MICROS["google/gemini-3.1-flash-lite-preview"]
    )
    image_url = ""
    if image_data:
        image_base64 = base64.b64encode(image_data).decode("utf-8")
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
    input_tokens = estimate_message_tokens(input_payload) + extra_input_tokens
    usd_micros = (
        input_tokens * pricing["input_per_million"]
        + max_output_tokens * pricing["output_per_million"]
    ) // 1_000_000
    return max(1, credit_units_from_usd_micros(usd_micros))


def estimate_transcribe_reserve_credits(audio_seconds: float) -> int:
    usd_micros = _calculate_transcription_usd_micros(audio_seconds)
    if usd_micros <= 0:
        return 1
    return max(1, credit_units_from_usd_micros(usd_micros))


def estimate_firecrawl_reserve_credits() -> int:
    """Reserve the published maximum cost of one application web search."""

    return max(
        1,
        credit_units_from_usd_micros(
            FIRECRAWL_SEARCH_MAX_CREDITS * FIRECRAWL_USD_MICROS_PER_CREDIT
        ),
    )


def credit_units_from_usd_micros(usd_micros: int) -> int:
    """Convert raw USD micros into hundredths of credits with markup."""

    micros = max(0, int(usd_micros or 0))
    if micros == 0:
        return 0
    return (micros + CREDIT_UNIT_USD_MICROS - 1) // CREDIT_UNIT_USD_MICROS


def _calculate_transcription_usd_micros(audio_seconds: float) -> int:
    hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3"]["audio_per_hour"]
    seconds = max(0.0, float(audio_seconds or 0.0))
    if seconds <= 0:
        return 0
    seconds = max(10.0, seconds)
    return math.ceil(seconds * hourly_rate / 3600)


def _extract_token_usage(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    usage_map = dict(usage or {})
    prompt_tokens_details = ensure_mapping(usage_map.get("prompt_tokens_details")) or {}
    input_tokens = int(usage_map.get("input_tokens") or usage_map.get("prompt_tokens") or 0)
    input_cached_tokens = int(
        usage_map.get("input_cached_tokens") or prompt_tokens_details.get("cached_tokens") or 0
    )
    output_tokens = int(usage_map.get("output_tokens") or usage_map.get("completion_tokens") or 0)
    input_cached_tokens = max(0, min(input_tokens, input_cached_tokens))
    input_non_cached_tokens = max(0, input_tokens - input_cached_tokens)
    return {
        "input_tokens": max(0, input_tokens),
        "input_cached_tokens": input_cached_tokens,
        "input_non_cached_tokens": input_non_cached_tokens,
        "output_tokens": max(0, output_tokens),
    }


def _calculate_model_token_cost(
    model: str,
    usage: Optional[Mapping[str, Any]],
    *,
    provider: str = "",
) -> Dict[str, Any]:
    normalized_provider = str(provider or "").strip().lower()
    if not usage:
        return {
            "model": model,
            "usd_micros": 0,
            "input_tokens": 0,
            "input_cached_tokens": 0,
            "input_non_cached_tokens": 0,
            "output_tokens": 0,
            "_usd_micros_exact": Decimal(0),
            "_pricing_basis": "missing",
        }

    reported_cost = _reported_cost_usd_micros_exact(usage)
    if reported_cost is not None and reported_cost > 0:
        return {
            "model": model,
            "usd_micros": int(reported_cost.to_integral_value(rounding=ROUND_FLOOR)),
            **_extract_token_usage(usage),
            "_usd_micros_exact": reported_cost,
            "_pricing_basis": "provider_reported",
        }

    # OpenRouter reports the completed request cost. If it does not, a local
    # model price cannot identify the routed provider's actual rate.
    pricing = None
    if normalized_provider != "openrouter":
        pricing = PROVIDER_MODEL_PRICING_USD_MICROS.get(
            (normalized_provider, model)
        ) or MODEL_PRICING_USD_MICROS.get(model)
    if not pricing:
        return {
            "model": model,
            "usd_micros": 0,
            **_extract_token_usage(usage),
            "_usd_micros_exact": Decimal(0),
            "_pricing_basis": "missing",
        }

    tokens = _extract_token_usage(usage)
    prompt_token_details = ensure_mapping(usage.get("prompt_tokens_details")) or {}
    audio_input_tokens = max(0, int(prompt_token_details.get("audio_tokens") or 0))
    cache_write_tokens = max(0, int(prompt_token_details.get("cache_write_tokens") or 0))
    non_cached_tokens = tokens["input_non_cached_tokens"]
    audio_input_tokens = min(non_cached_tokens, audio_input_tokens)
    cache_write_tokens = min(
        non_cached_tokens - audio_input_tokens,
        cache_write_tokens,
    )
    regular_input_tokens = max(0, non_cached_tokens - audio_input_tokens - cache_write_tokens)
    cached_input_per_million = pricing.get(
        "cached_input_per_million",
        pricing.get("input_per_million", 0),
    )
    usd_micros_exact = Decimal(
        pricing.get("request_usd_micros", 0) * 1_000_000
        + regular_input_tokens * pricing.get("input_per_million", 0)
        + tokens["input_cached_tokens"] * cached_input_per_million
        + audio_input_tokens
        * pricing.get("audio_input_per_million", pricing.get("input_per_million", 0))
        + cache_write_tokens
        * pricing.get("cache_write_per_million", pricing.get("input_per_million", 0))
        + tokens["output_tokens"] * pricing.get("output_per_million", 0)
    ) / Decimal(1_000_000)
    return {
        "model": model,
        "usd_micros": int(usd_micros_exact.to_integral_value(rounding=ROUND_FLOOR)),
        **tokens,
        "_usd_micros_exact": usd_micros_exact,
        "_pricing_basis": "published_rate",
    }


def _reported_cost_usd_micros_exact(
    usage: Mapping[str, Any],
) -> Optional[Decimal]:
    """Return provider-price cost in USD micros without float rounding."""

    cost_details = ensure_mapping(usage.get("cost_details")) or {}
    if "upstream_inference_cost" in cost_details:
        raw_cost = cost_details.get("upstream_inference_cost")
        try:
            cost = Decimal(str(raw_cost)) * Decimal(1_000_000)
        except InvalidOperation, TypeError, ValueError:
            pass
        else:
            if cost > 0:
                return cost

    raw_cost = usage.get("cost")
    if raw_cost is not None:
        try:
            cost = Decimal(str(raw_cost)) * Decimal(1_000_000)
        except InvalidOperation, TypeError, ValueError:
            return None
        if cost > 0:
            return cost
    return None


def _credit_units_from_exact_usd_micros(total_usd_micros: Decimal) -> int:
    if total_usd_micros <= 0:
        return 0
    return int(
        (total_usd_micros / Decimal(CREDIT_UNIT_USD_MICROS)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def _calculate_firecrawl_cost(
    metadata: Mapping[str, Any],
) -> tuple[int, Optional[Dict[str, Any]]]:
    try:
        web_search_requests = int(metadata.get("web_search_requests") or 0)
    except TypeError, ValueError:
        web_search_requests = 0
    try:
        firecrawl_credits = max(0, int(metadata.get("firecrawl_credits_used") or 0))
    except TypeError, ValueError:
        firecrawl_credits = 0
    if firecrawl_credits <= 0:
        return 0, None
    usd_micros = firecrawl_credits * FIRECRAWL_USD_MICROS_PER_CREDIT
    return usd_micros, {
        "tool": "web_search",
        "count": web_search_requests,
        "usd_micros": usd_micros,
    }


def _standalone_firecrawl_breakdown(
    *,
    kind: str,
    model: str,
    provider: str,
    metadata: Mapping[str, Any],
    segment_index: int,
    tool_breakdown: List[Dict[str, Any]],
    segment_breakdown: List[Dict[str, Any]],
) -> int | None:
    usd_micros, tool_cost = _calculate_firecrawl_cost(metadata)
    if kind != "web_search" or tool_cost is None:
        return None
    tool_breakdown.append(tool_cost)
    segment_breakdown.append(
        {
            "segment_index": segment_index,
            "kind": kind,
            "model": model,
            "provider": provider or "firecrawl",
            "pricing_basis": "firecrawl_standard",
            "cost_complete": True,
            "usd_micros_exact": str(usd_micros),
        }
    )
    return usd_micros


def calculate_billing_for_segments(
    segments: Iterable[Optional[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Calculate raw and marked-up billing totals for AI usage segments."""

    total_usd_micros = Decimal(0)
    model_breakdown: List[Dict[str, Any]] = []
    tool_breakdown: List[Dict[str, Any]] = []
    segment_breakdown: List[Dict[str, Any]] = []
    unsupported_notes: List[str] = []

    present_segments = (segment for segment in segments if segment is not None)
    for segment_index, raw_segment in enumerate(present_segments):
        segment = dict(raw_segment or {})
        if str(segment.get("source") or "").strip().lower() == "cache":
            segment_breakdown.append(
                {
                    "segment_index": segment_index,
                    "kind": str(segment.get("kind") or ""),
                    "model": str(segment.get("model") or ""),
                    "provider": "internal",
                    "pricing_basis": "internal_cache",
                    "cost_complete": True,
                    "usd_micros_exact": "0",
                }
            )
            continue
        kind = str(segment.get("kind") or "")
        model = str(segment.get("model") or "")
        usage = ensure_mapping(segment.get("usage")) or {}
        audio_seconds = float(segment.get("audio_seconds") or 0.0)
        metadata = ensure_mapping(segment.get("metadata")) or {}
        provider = str(metadata.get("provider") or segment.get("source") or "").strip().lower()
        upstream_provider = str(metadata.get("upstream_provider") or "").strip().lower()
        reported_cost = _reported_cost_usd_micros_exact(usage)

        standalone_search = _standalone_firecrawl_breakdown(
            kind=kind,
            model=model,
            provider=provider,
            metadata=metadata,
            segment_index=segment_index,
            tool_breakdown=tool_breakdown,
            segment_breakdown=segment_breakdown,
        )
        if standalone_search is not None:
            total_usd_micros += Decimal(standalone_search)
            continue

        pricing = MODEL_PRICING_USD_MICROS.get(model) or {}
        if (
            kind == "transcribe"
            and "audio_per_hour" in pricing
            and not (provider == "openrouter" and reported_cost is not None and reported_cost > 0)
        ):
            usd_micros = _calculate_transcription_usd_micros(audio_seconds)
            total_usd_micros += Decimal(usd_micros)
            model_breakdown.append(
                {
                    "kind": kind,
                    "model": model or "whisper-large-v3",
                    "usd_micros": usd_micros,
                    "audio_seconds": audio_seconds,
                }
            )
            segment_breakdown.append(
                {
                    "segment_index": segment_index,
                    "kind": kind,
                    "model": model or "whisper-large-v3",
                    "provider": provider or "groq",
                    "pricing_basis": "published_rate",
                    "cost_complete": audio_seconds > 0,
                    "usd_micros_exact": str(usd_micros),
                }
            )
            if audio_seconds <= 0:
                unsupported_notes.append(
                    f"missing_usage_or_cost:segment={segment_index}:provider={provider or 'groq'}:model={model}"
                )
            continue

        model_cost = _calculate_model_token_cost(
            model,
            usage,
            provider=provider,
        )
        exact_model_cost = model_cost.pop("_usd_micros_exact")
        pricing_basis = str(model_cost.pop("_pricing_basis"))
        model_cost["kind"] = kind
        total_usd_micros += exact_model_cost
        model_breakdown.append(model_cost)

        search_usd_micros, tool_cost = _calculate_firecrawl_cost(metadata)
        total_usd_micros += Decimal(search_usd_micros)
        if tool_cost is not None:
            tool_breakdown.append(tool_cost)

        token_usage = _extract_token_usage(usage)
        has_token_usage = bool(token_usage["input_tokens"] or token_usage["output_tokens"])
        cost_complete = bool(
            (reported_cost is not None and reported_cost > 0)
            or (has_token_usage and pricing_basis == "published_rate")
        )
        if not cost_complete:
            unsupported_notes.append(
                f"missing_usage_or_cost:segment={segment_index}:provider={provider or 'unknown'}:model={model or 'unknown'}"
            )
        segment_breakdown.append(
            {
                "segment_index": segment_index,
                "kind": kind,
                "model": model,
                "provider": provider or "unknown",
                "upstream_provider": upstream_provider or None,
                "provider_request_id": metadata.get("provider_request_id"),
                "provider_generation_id": metadata.get("provider_generation_id"),
                "pricing_basis": pricing_basis,
                "tool_pricing_basis": "firecrawl_standard" if tool_cost else None,
                "cost_complete": cost_complete,
                "usd_micros_exact": format(exact_model_cost, "f"),
            }
        )

    charged_credit_units = _credit_units_from_exact_usd_micros(total_usd_micros)
    raw_usd_micros = int(total_usd_micros.to_integral_value(rounding=ROUND_FLOOR))
    return {
        "pricing_version": PRICING_VERSION,
        "markup_multiplier": BILLING_MARKUP_MULTIPLIER,
        "raw_usd_micros": raw_usd_micros,
        "raw_usd_micros_exact": format(total_usd_micros, "f"),
        "charged_credit_units": charged_credit_units,
        "charged_credits_display": format_credit_units(charged_credit_units),
        "model_breakdown": model_breakdown,
        "tool_breakdown": tool_breakdown,
        "segment_breakdown": segment_breakdown,
        "pricing_complete": bool(segment_breakdown) and not unsupported_notes,
        "unsupported_notes": unsupported_notes,
    }
