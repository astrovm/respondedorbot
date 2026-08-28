"""Stable identifiers for durable provider billing segments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def provider_segment_id(segment: Mapping[str, Any]) -> str:
    """Identify one provider call without collapsing separate retries."""

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
    encoded = json.dumps(
        dict(segment),
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["provider_segment_id"]
