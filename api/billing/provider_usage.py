"""Stable identifiers for durable provider billing segments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
import logging
from typing import Any, Protocol, cast

from api.core.rust_bridge import load_rust_bridge


logger = logging.getLogger(__name__)


class _RustAiUsagePolicy(Protocol):
    def provider_segment_id(self, segment_json: str) -> str: ...

    def provider_usage_needs_reconciliation(self, segment_json: str) -> bool: ...


def _load_rust_ai_usage_policy() -> _RustAiUsagePolicy | None:
    module = load_rust_bridge("RUST_AI_USAGE_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustAiUsagePolicy, module)


def _canonical_segment_json(segment: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(segment),
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )


def provider_segment_id(segment: Mapping[str, Any]) -> str:
    """Identify one provider call without collapsing separate retries."""

    canonical = _canonical_segment_json(segment)
    rust = _load_rust_ai_usage_policy()
    if rust is not None:
        try:
            return str(rust.provider_segment_id(canonical))
        except Exception:
            logger.exception("Rust provider segment identity failed; using Python fallback")

    metadata = segment.get("metadata")
    segment_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    provider_id = segment_metadata.get(
        "provider_generation_id"
    ) or segment_metadata.get("provider_request_id")
    if provider_id:
        return f"{segment.get('source', 'provider')}:{provider_id}"
    if segment_metadata.get("tool_rounds"):
        return ":".join(
            (
                str(segment.get("source") or "provider"),
                str(segment.get("kind") or "unknown"),
                str(segment.get("model") or "unknown"),
                str(segment_metadata["tool_rounds"]),
            )
        )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def provider_usage_needs_reconciliation(segment: Mapping[str, Any]) -> bool:
    """Return whether interrupted OpenRouter usage still lacks final cost."""

    rust = _load_rust_ai_usage_policy()
    if rust is not None:
        try:
            return bool(
                rust.provider_usage_needs_reconciliation(
                    _canonical_segment_json(segment)
                )
            )
        except Exception:
            logger.exception("Rust provider usage policy failed; using Python fallback")

    metadata = segment.get("metadata")
    usage = segment.get("usage")
    return bool(
        segment.get("source") == "openrouter"
        and isinstance(metadata, Mapping)
        and (
            metadata.get("stream_interrupted")
            or metadata.get("provider_usage_pending")
        )
        and not (
            isinstance(usage, Mapping)
            and _positive_number(usage.get("cost"))
        )
    )


__all__ = ["provider_segment_id", "provider_usage_needs_reconciliation"]
