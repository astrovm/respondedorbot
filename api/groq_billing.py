"""Groq pricing, usage normalization, and credit calculations."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PRICING_VERSION = "2026-03-06"
CREDIT_USD_MICROS = 10_000
BILLING_MARKUP_MULTIPLIER = 2.5
CREDIT_CEIL_DIVISOR_USD_MICROS = 4_000

CHAT_OUTPUT_TOKEN_LIMIT = 512
VISION_OUTPUT_TOKEN_LIMIT = 512
IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE = 1_200
VISIT_WEBSITE_USD_MICROS = 1_000
CODE_EXECUTION_USD_MICROS_PER_HOUR = 180_000
BROWSER_AUTOMATION_USD_MICROS_PER_HOUR = 80_000
WEB_SEARCH_STANDARD_USD_MICROS = 5_000
WEB_SEARCH_PREMIUM_USD_MICROS = 8_000
GPT_OSS_120B_FALLBACK_MODEL = "openai/gpt-oss-120b"


MODEL_PRICING_USD_MICROS: Dict[str, Dict[str, int]] = {
    "moonshotai/kimi-k2-instruct-0905": {
        "input_per_million": 1_000_000,
        "output_per_million": 3_000_000,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input_per_million": 110_000,
        "output_per_million": 340_000,
    },
    "whisper-large-v3-turbo": {
        "audio_per_hour": 40_000,
    },
    "openai/gpt-oss-120b": {
        "input_per_million": 150_000,
        "output_per_million": 600_000,
    },
}


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
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
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
    return credits_from_usd_micros(usd_micros)


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
    return max(1, credits_from_usd_micros(usd_micros))


def estimate_transcribe_reserve_credits(audio_seconds: float) -> int:
    hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3-turbo"]["audio_per_hour"]
    seconds = max(0.0, float(audio_seconds or 0.0))
    if seconds <= 0:
        return 1
    usd_micros = math.ceil(seconds * hourly_rate / 3600)
    return max(1, credits_from_usd_micros(usd_micros))


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
    normalized_tools = {str(tool).strip().lower() for tool in enabled_tools or [] if str(tool).strip()}
    if "web_search" in normalized_tools:
        usd_micros += WEB_SEARCH_PREMIUM_USD_MICROS
    if "visit_website" in normalized_tools:
        usd_micros += VISIT_WEBSITE_USD_MICROS
    return credits_from_usd_micros(usd_micros)


def credits_from_usd_micros(usd_micros: int) -> int:
    """Convert raw USD micros into whole credits with markup."""

    micros = max(0, int(usd_micros or 0))
    if micros == 0:
        return 0
    return (micros + CREDIT_CEIL_DIVISOR_USD_MICROS - 1) // CREDIT_CEIL_DIVISOR_USD_MICROS


def _extract_token_usage(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    usage_map = dict(usage or {})
    input_tokens = int(
        usage_map.get("input_tokens")
        or usage_map.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage_map.get("output_tokens")
        or usage_map.get("completion_tokens")
        or 0
    )
    return {
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
    }


def _calculate_model_token_cost(model: str, usage: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    pricing = MODEL_PRICING_USD_MICROS.get(model)
    if not pricing or not usage:
        return {
            "model": model,
            "usd_micros": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    tokens = _extract_token_usage(usage)
    usd_micros = (
        tokens["input_tokens"] * pricing.get("input_per_million", 0)
        + tokens["output_tokens"] * pricing.get("output_per_million", 0)
    ) // 1_000_000
    return {
        "model": model,
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


def _extract_tool_duration_seconds(tool: Mapping[str, Any]) -> float:
    for key in ("duration_seconds", "runtime_seconds", "elapsed_seconds", "billed_seconds"):
        value = tool.get(key)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    duration_ms = tool.get("duration_ms")
    try:
        if duration_ms is not None:
            return max(0.0, float(duration_ms) / 1000.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


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
    duration_seconds = _extract_tool_duration_seconds(tool)
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
        if duration_seconds > 0:
            usd_micros = int(duration_seconds * CODE_EXECUTION_USD_MICROS_PER_HOUR / 3600)
        else:
            note = "missing_duration"
    elif normalized_name == "browser_automation":
        if duration_seconds > 0:
            usd_micros = int(duration_seconds * BROWSER_AUTOMATION_USD_MICROS_PER_HOUR / 3600)
        else:
            note = "missing_duration"
    else:
        note = "unsupported_tool"

    return {
        "tool": tool_name or "unknown",
        "usd_micros": int(max(0, usd_micros)),
        "count": count,
        "duration_seconds": duration_seconds,
        "note": note,
    }


def calculate_billing_for_segments(segments: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Calculate raw and marked-up billing totals for Groq usage segments."""

    total_usd_micros = 0
    model_breakdown: List[Dict[str, Any]] = []
    tool_breakdown: List[Dict[str, Any]] = []
    unsupported_notes: List[str] = []

    for raw_segment in segments:
        segment = dict(raw_segment or {})
        kind = str(segment.get("kind") or "")
        model = str(segment.get("model") or "")
        usage = ensure_mapping(segment.get("usage")) or {}
        usage_breakdown = _normalize_usage_breakdown(segment.get("usage_breakdown"))
        executed_tools = ensure_mapping_list(segment.get("executed_tools"))
        audio_seconds = float(segment.get("audio_seconds") or 0.0)

        if kind == "transcribe":
            hourly_rate = MODEL_PRICING_USD_MICROS["whisper-large-v3-turbo"]["audio_per_hour"]
            usd_micros = int(audio_seconds * hourly_rate / 3600)
            total_usd_micros += usd_micros
            model_breakdown.append(
                {
                    "model": model or "whisper-large-v3-turbo",
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
                item_model = str(item.get("model") or item.get("name") or GPT_OSS_120B_FALLBACK_MODEL)
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

        for tool in executed_tools:
            tool_cost = _calculate_tool_cost(tool)
            total_usd_micros += int(tool_cost["usd_micros"])
            tool_breakdown.append(tool_cost)
            if tool_cost["note"]:
                unsupported_notes.append(f"{tool_cost['tool']}:{tool_cost['note']}")

    return {
        "pricing_version": PRICING_VERSION,
        "markup_multiplier": BILLING_MARKUP_MULTIPLIER,
        "raw_usd_micros": int(total_usd_micros),
        "charged_credits": credits_from_usd_micros(int(total_usd_micros)),
        "model_breakdown": model_breakdown,
        "tool_breakdown": tool_breakdown,
        "unsupported_notes": unsupported_notes,
    }
