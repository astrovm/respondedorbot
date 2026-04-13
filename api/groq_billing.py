"""Groq pricing, usage normalization, and credit calculations."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from api.credit_units import format_credit_units


PRICING_VERSION = "2026-04-12"
CREDIT_USD_MICROS = 10_000
BILLING_MARKUP_MULTIPLIER = 2.0
CREDIT_CEIL_DIVISOR_USD_MICROS = int(CREDIT_USD_MICROS / BILLING_MARKUP_MULTIPLIER)
CREDIT_UNIT_USD_MICROS = CREDIT_CEIL_DIVISOR_USD_MICROS // 10

CHAT_OUTPUT_TOKEN_LIMIT = 256
VISION_OUTPUT_TOKEN_LIMIT = 256
IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE = 1_200
WEB_SEARCH_USD_MICROS_PER_REQUEST = 1_660


MODEL_PRICING_USD_MICROS: Dict[str, Dict[str, int]] = {
    # Groq pricing
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input_per_million": 110_000,
        "output_per_million": 340_000,
    },
    # OpenRouter pricing
    "meta-llama/llama-4-scout": {
        "input_per_million": 80_000,
        "output_per_million": 300_000,
    },
    "whisper-large-v3": {
        "audio_per_hour": 111_000,
    },
    "qwen/qwen3.6-plus": {
        "input_per_million": 325_000,
        "output_per_million": 1_950_000,
    },
}


MODEL_BILLING_ALIASES = {
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/whisper-large-v3": "whisper-large-v3",
    "qwen/qwen3.6-plus-04-02": "qwen/qwen3.6-plus",
}


def normalize_model_id_for_billing(model: str) -> str:
    normalized = str(model or "").strip()
    return MODEL_BILLING_ALIASES.get(normalized, normalized)


@dataclass
class GroqUsageResult:
    """Structured Groq response with billing metadata."""

    kind: str
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    audio_seconds: Optional[float] = None
    cached: bool = False
    source: str = "groq"
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
        data = {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
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
    max_output_tokens: int = CHAT_OUTPUT_TOKEN_LIMIT,
    extra_input_tokens: int = 0,
) -> int:
    pricing = MODEL_PRICING_USD_MICROS["qwen/qwen3.6-plus"]
    input_tokens = estimate_message_tokens(messages) + extra_input_tokens
    if system_message:
        input_tokens += estimate_message_tokens([system_message])
    usd_micros = (
        input_tokens * pricing["input_per_million"]
        + max_output_tokens * pricing["output_per_million"]
    ) // 1_000_000
    return credit_units_from_usd_micros(usd_micros)


def estimate_vision_reserve_credits(
    *,
    prompt_text: str,
    image_data: Optional[bytes] = None,
    max_output_tokens: int = VISION_OUTPUT_TOKEN_LIMIT,
) -> int:
    pricing = MODEL_PRICING_USD_MICROS["meta-llama/llama-4-scout-17b-16e-instruct"]
    image_url = ""
    if image_data:
        image_base64 = base64.b64encode(image_data).decode("utf-8")
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
    input_tokens = estimate_message_tokens(input_payload)
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


def credit_units_from_usd_micros(usd_micros: int) -> int:
    """Convert raw USD micros into tenths of credits with markup."""

    micros = max(0, int(usd_micros or 0))
    if micros == 0:
        return 0
    return (micros + CREDIT_UNIT_USD_MICROS - 1) // CREDIT_UNIT_USD_MICROS


def _calculate_transcription_usd_micros(audio_seconds: float) -> int:
    hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3"]["audio_per_hour"]
    seconds = max(0.0, float(audio_seconds or 0.0))
    if seconds <= 0:
        return 0
    return math.ceil(seconds * hourly_rate / 3600)


def _extract_token_usage(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    usage_map = dict(usage or {})
    prompt_tokens_details = ensure_mapping(usage_map.get("prompt_tokens_details")) or {}
    input_tokens = int(
        usage_map.get("input_tokens") or usage_map.get("prompt_tokens") or 0
    )
    input_cached_tokens = int(
        usage_map.get("input_cached_tokens")
        or prompt_tokens_details.get("cached_tokens")
        or 0
    )
    output_tokens = int(
        usage_map.get("output_tokens") or usage_map.get("completion_tokens") or 0
    )
    input_cached_tokens = max(0, min(input_tokens, input_cached_tokens))
    input_non_cached_tokens = max(0, input_tokens - input_cached_tokens)
    return {
        "input_tokens": max(0, input_tokens),
        "input_cached_tokens": input_cached_tokens,
        "input_non_cached_tokens": input_non_cached_tokens,
        "output_tokens": max(0, output_tokens),
    }


def _calculate_model_token_cost(
    model: str, usage: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    normalized_model = normalize_model_id_for_billing(model)
    pricing = MODEL_PRICING_USD_MICROS.get(normalized_model)
    if not pricing or not usage:
        return {
            "model": normalized_model,
            "usd_micros": 0,
            "input_tokens": 0,
            "input_cached_tokens": 0,
            "input_non_cached_tokens": 0,
            "output_tokens": 0,
        }

    tokens = _extract_token_usage(usage)
    cached_input_per_million = pricing.get(
        "cached_input_per_million",
        pricing.get("input_per_million", 0),
    )
    usd_micros = (
        tokens["input_non_cached_tokens"] * pricing.get("input_per_million", 0)
        + tokens["input_cached_tokens"] * cached_input_per_million
        + tokens["output_tokens"] * pricing.get("output_per_million", 0)
    ) // 1_000_000
    try:
        gateway_cost_usd = float(usage.get("cost") or 0)
        gateway_usd_micros = int(gateway_cost_usd * 1_000_000)
        if gateway_usd_micros > usd_micros:
            usd_micros = gateway_usd_micros
    except (TypeError, ValueError):
        pass
    return {
        "model": normalized_model,
        "usd_micros": int(usd_micros),
        **tokens,
    }


def calculate_billing_for_segments(
    segments: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Calculate raw and marked-up billing totals for Groq usage segments."""

    total_usd_micros = 0
    model_breakdown: List[Dict[str, Any]] = []
    tool_breakdown: List[Dict[str, Any]] = []
    unsupported_notes: List[str] = []

    for raw_segment in segments:
        segment = dict(raw_segment or {})
        if str(segment.get("source") or "").strip().lower() == "cache":
            continue
        kind = str(segment.get("kind") or "")
        model = str(segment.get("model") or "")
        usage = ensure_mapping(segment.get("usage")) or {}
        audio_seconds = float(segment.get("audio_seconds") or 0.0)

        if kind == "transcribe":
            usd_micros = _calculate_transcription_usd_micros(audio_seconds)
            total_usd_micros += usd_micros
            model_breakdown.append(
                {
                    "model": normalize_model_id_for_billing(
                        model or "whisper-large-v3"
                    ),
                    "usd_micros": usd_micros,
                    "audio_seconds": audio_seconds,
                }
            )
            continue

        model_cost = _calculate_model_token_cost(model, usage)
        total_usd_micros += int(model_cost["usd_micros"])
        model_breakdown.append(model_cost)

        metadata = ensure_mapping(segment.get("metadata")) or {}
        try:
            web_search_requests = int(metadata.get("web_search_requests") or 0)
        except (TypeError, ValueError):
            web_search_requests = 0
        if web_search_requests > 0:
            search_usd_micros = web_search_requests * WEB_SEARCH_USD_MICROS_PER_REQUEST
            total_usd_micros += search_usd_micros
            tool_breakdown.append(
                {
                    "tool": "web_search",
                    "count": web_search_requests,
                    "usd_micros": search_usd_micros,
                }
            )

    charged_credit_units = credit_units_from_usd_micros(int(total_usd_micros))
    return {
        "pricing_version": PRICING_VERSION,
        "markup_multiplier": BILLING_MARKUP_MULTIPLIER,
        "raw_usd_micros": int(total_usd_micros),
        "charged_credit_units": charged_credit_units,
        "charged_credits_display": format_credit_units(charged_credit_units),
        "model_breakdown": model_breakdown,
        "tool_breakdown": tool_breakdown,
        "unsupported_notes": unsupported_notes,
    }
