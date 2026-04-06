"""Groq pricing, usage normalization, and credit calculations."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from api.credit_units import format_credit_units


PRICING_VERSION = "2026-03-06"
CREDIT_USD_MICROS = 10_000
BILLING_MARKUP_MULTIPLIER = 2.0
CREDIT_CEIL_DIVISOR_USD_MICROS = int(CREDIT_USD_MICROS / BILLING_MARKUP_MULTIPLIER)
CREDIT_UNIT_USD_MICROS = CREDIT_CEIL_DIVISOR_USD_MICROS // 10

CHAT_OUTPUT_TOKEN_LIMIT = 256
VISION_OUTPUT_TOKEN_LIMIT = 256
IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE = 1_200
VISIT_WEBSITE_USD_MICROS = 1_000
CODE_EXECUTION_USD_MICROS_PER_HOUR = 180_000
BROWSER_AUTOMATION_USD_MICROS_PER_HOUR = 80_000
WEB_SEARCH_STANDARD_USD_MICROS = 5_000
WEB_SEARCH_PREMIUM_USD_MICROS = 8_000
GPT_OSS_120B_FALLBACK_MODEL = "openai/gpt-oss-120b"
MAX_UNDOCUMENTED_TIME_BASED_TOOL_SECONDS_PER_REQUEST = 120.0


MODEL_PRICING_USD_MICROS: Dict[str, Dict[str, int]] = {
    # Groq pricing
    "moonshotai/kimi-k2-instruct-0905": {
        "input_per_million": 1_000_000,
        "cached_input_per_million": 500_000,
        "output_per_million": 3_000_000,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input_per_million": 110_000,
        "output_per_million": 340_000,
    },
    # OpenRouter pricing
    "moonshotai/kimi-k2-0905": {
        "input_per_million": 400_000,
        "cached_input_per_million": 150_000,
        "output_per_million": 2_000_000,
    },
    "meta-llama/llama-4-scout": {
        "input_per_million": 80_000,
        "output_per_million": 300_000,
    },
    "whisper-large-v3": {
        "audio_per_hour": 111_000,
    },
    "openai/gpt-oss-120b": {
        "input_per_million": 150_000,
        "cached_input_per_million": 75_000,
        "output_per_million": 600_000,
    },
}


MODEL_BILLING_ALIASES = {
    "groq/moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/whisper-large-v3": "whisper-large-v3",
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
    usage_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)
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


def ensure_mapping_list(value: Any) -> List[Dict[str, Any]]:
    """Normalize list-like SDK fragments into a list of plain dicts."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        normalized_items: List[Dict[str, Any]] = []
        for key, item in value.items():
            item_map = ensure_mapping(item)
            if item_map is None:
                item_map = {"value": item}
            item_map.setdefault("model", key)
            normalized_items.append(item_map)
        return normalized_items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized_items = []
        for item in value:
            item_map = ensure_mapping(item)
            if item_map is not None:
                normalized_items.append(item_map)
        return normalized_items
    item_map = ensure_mapping(value)
    return [item_map] if item_map is not None else []


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
    pricing = MODEL_PRICING_USD_MICROS["moonshotai/kimi-k2-instruct-0905"]
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
    hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3"]["audio_per_hour"]
    seconds = max(0.0, float(audio_seconds or 0.0))
    if seconds <= 0:
        return 1
    usd_micros = math.ceil(seconds * hourly_rate / 3600)
    return max(1, credit_units_from_usd_micros(usd_micros))


def estimate_compound_reserve_credits(
    *,
    system_message: Optional[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    enabled_tools: Optional[Sequence[str]] = None,
    max_output_tokens: int = CHAT_OUTPUT_TOKEN_LIMIT,
) -> int:
    pricing = MODEL_PRICING_USD_MICROS[GPT_OSS_120B_FALLBACK_MODEL]
    input_tokens = estimate_message_tokens(messages)
    if system_message:
        input_tokens += estimate_message_tokens([system_message])
    usd_micros = (
        input_tokens * pricing["input_per_million"]
        + max_output_tokens * pricing["output_per_million"]
    ) // 1_000_000
    normalized_tools = {
        str(tool).strip().lower() for tool in enabled_tools or [] if str(tool).strip()
    }
    if "web_search" in normalized_tools:
        usd_micros += WEB_SEARCH_PREMIUM_USD_MICROS
    if "visit_website" in normalized_tools:
        usd_micros += VISIT_WEBSITE_USD_MICROS
    return credit_units_from_usd_micros(usd_micros)


def credit_units_from_usd_micros(usd_micros: int) -> int:
    """Convert raw USD micros into tenths of credits with markup."""

    micros = max(0, int(usd_micros or 0))
    if micros == 0:
        return 0
    return (micros + CREDIT_UNIT_USD_MICROS - 1) // CREDIT_UNIT_USD_MICROS


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
    return {
        "model": normalized_model,
        "usd_micros": int(usd_micros),
        **tokens,
    }


def _extract_tool_name(tool: Mapping[str, Any]) -> str:
    for key in ("tool", "name", "type", "id"):
        value = str(tool.get(key) or "").strip()
        if value:
            return value
    nested_tool = ensure_mapping(tool.get("tool"))
    if nested_tool:
        return _extract_tool_name(nested_tool)
    return ""


def _extract_tool_count(tool: Mapping[str, Any]) -> int:
    for key in ("request_count", "count", "requests", "executions"):
        value = tool.get(key)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 1


def _normalize_usage_breakdown(value: Any) -> List[Dict[str, Any]]:
    usage_breakdown = ensure_mapping(value)
    if usage_breakdown and isinstance(usage_breakdown.get("models"), Sequence):
        return ensure_mapping_list(usage_breakdown.get("models"))
    return ensure_mapping_list(value)


def _normalize_tool_kind(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized in {"search", "web_search", "web-search"}:
        return "web_search"
    if normalized in {"visit", "visit_website", "website_visit"}:
        return "visit_website"
    if normalized in {"python", "code_interpreter", "code_execution"}:
        return "code_execution"
    if normalized in {"browser", "browser_automation"}:
        return "browser_automation"
    return normalized


def _calculate_tool_cost(tool: Mapping[str, Any]) -> Dict[str, Any]:
    tool_name = _extract_tool_name(tool)
    normalized_name = _normalize_tool_kind(tool_name)
    count = _extract_tool_count(tool)
    usd_micros = 0
    note = ""

    if normalized_name == "web_search":
        search_type = str(
            tool.get("search_type")
            or tool.get("variant")
            or tool.get("mode")
            or tool.get("quality")
            or ""
        ).lower()
        unit = (
            WEB_SEARCH_STANDARD_USD_MICROS
            if "standard" in search_type or "basic" in search_type
            else WEB_SEARCH_PREMIUM_USD_MICROS
        )
        usd_micros = unit * count
    elif normalized_name == "visit_website":
        usd_micros = VISIT_WEBSITE_USD_MICROS * count
    elif normalized_name == "code_execution":
        note = "duration_not_documented_in_api_response"
    elif normalized_name == "browser_automation":
        note = "duration_not_documented_in_api_response"
    else:
        note = "unsupported_tool"

    return {
        "tool": tool_name or "unknown",
        "normalized_tool": normalized_name,
        "usd_micros": int(max(0, usd_micros)),
        "count": count,
        "note": note,
    }


def _coerce_clamped_seconds(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return 0.0
    return min(parsed, MAX_UNDOCUMENTED_TIME_BASED_TOOL_SECONDS_PER_REQUEST)


def _extract_time_based_tool_seconds(
    metadata: Optional[Mapping[str, Any]],
    usage: Optional[Mapping[str, Any]],
) -> Tuple[float, str]:
    metadata_map = ensure_mapping(metadata) or {}
    usage_map = ensure_mapping(usage) or {}

    request_elapsed_seconds = _coerce_clamped_seconds(
        metadata_map.get("request_elapsed_seconds")
    )
    if request_elapsed_seconds is not None:
        return request_elapsed_seconds, "estimated_from_request_elapsed_seconds"

    total_time_seconds = _coerce_clamped_seconds(usage_map.get("total_time"))
    if total_time_seconds is not None:
        return total_time_seconds, "estimated_from_usage_total_time"

    return (
        MAX_UNDOCUMENTED_TIME_BASED_TOOL_SECONDS_PER_REQUEST,
        "estimated_max_120_second_request_cap",
    )


def _estimate_time_based_tool_cost(normalized_tool: str, seconds: float) -> int:
    if normalized_tool == "code_execution":
        return int(seconds * CODE_EXECUTION_USD_MICROS_PER_HOUR / 3600)
    if normalized_tool == "browser_automation":
        return int(seconds * BROWSER_AUTOMATION_USD_MICROS_PER_HOUR / 3600)
    return 0


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
        metadata = ensure_mapping(segment.get("metadata")) or {}
        usage_breakdown = _normalize_usage_breakdown(segment.get("usage_breakdown"))
        executed_tools = ensure_mapping_list(segment.get("executed_tools"))
        audio_seconds = float(segment.get("audio_seconds") or 0.0)

        if kind == "transcribe":
            hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3"]["audio_per_hour"]
            usd_micros = int(audio_seconds * hourly_rate / 3600)
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

        if kind == "compound":
            breakdown_items = usage_breakdown
            if not breakdown_items and usage:
                breakdown_items = [
                    {
                        "model": GPT_OSS_120B_FALLBACK_MODEL,
                        **usage,
                        "note": "top_level_usage_fallback",
                    }
                ]
            if not breakdown_items and executed_tools:
                unsupported_notes.append("compound_missing_usage_breakdown")
            for item in breakdown_items:
                item_model = str(
                    item.get("model") or item.get("name") or GPT_OSS_120B_FALLBACK_MODEL
                )
                model_usage = ensure_mapping(item.get("usage")) or item
                model_cost = _calculate_model_token_cost(item_model, model_usage)
                if item.get("note"):
                    model_cost["note"] = item.get("note")
                total_usd_micros += int(model_cost["usd_micros"])
                model_breakdown.append(model_cost)
        else:
            model_cost = _calculate_model_token_cost(model, usage)
            total_usd_micros += int(model_cost["usd_micros"])
            model_breakdown.append(model_cost)

        segment_tool_breakdown: List[Dict[str, Any]] = []
        estimated_time_based_candidates: List[Tuple[int, int, str]] = []

        for tool in executed_tools:
            tool_cost = _calculate_tool_cost(tool)
            normalized_tool = str(tool_cost.get("normalized_tool") or "")
            note = str(tool_cost.get("note") or "")
            if (
                note == "duration_not_documented_in_api_response"
                and normalized_tool in {"code_execution", "browser_automation"}
            ):
                estimated_seconds, estimated_note = _extract_time_based_tool_seconds(
                    metadata,
                    usage,
                )
                estimated_time_based_candidates.append(
                    (
                        len(segment_tool_breakdown),
                        _estimate_time_based_tool_cost(
                            normalized_tool, estimated_seconds
                        ),
                        estimated_note,
                    )
                )
            segment_tool_breakdown.append(tool_cost)

        if estimated_time_based_candidates:
            selected_index, selected_cost, selected_note = max(
                estimated_time_based_candidates, key=lambda item: item[1]
            )
            for idx, _, _ in estimated_time_based_candidates:
                if idx == selected_index:
                    segment_tool_breakdown[idx]["usd_micros"] = selected_cost
                    segment_tool_breakdown[idx]["note"] = selected_note
                else:
                    segment_tool_breakdown[idx]["usd_micros"] = 0
                    segment_tool_breakdown[idx]["note"] = (
                        f"estimated_shared_cap_from:{selected_note}"
                    )

        for tool_cost in segment_tool_breakdown:
            tool_cost.pop("normalized_tool", None)
            total_usd_micros += int(tool_cost["usd_micros"])
            tool_breakdown.append(tool_cost)
            note = str(tool_cost.get("note") or "")
            if note and not note.startswith("estimated_"):
                unsupported_notes.append(f"{tool_cost['tool']}:{note}")

    return {
        "pricing_version": PRICING_VERSION,
        "markup_multiplier": BILLING_MARKUP_MULTIPLIER,
        "raw_usd_micros": int(total_usd_micros),
        "charged_credit_units": credit_units_from_usd_micros(int(total_usd_micros)),
        "charged_credits_display": format_credit_units(
            credit_units_from_usd_micros(int(total_usd_micros))
        ),
        "model_breakdown": model_breakdown,
        "tool_breakdown": tool_breakdown,
        "unsupported_notes": unsupported_notes,
    }
