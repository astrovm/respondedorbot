"""Small authorization boundary shared by billing and provider code."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


AI_COST_AUTHORIZER_KEY = "_ai_cost_authorizer"
AI_SEGMENT_RECORDER_KEY = "_ai_segment_recorder"
AuthorizeAICost = Callable[[str, int, Mapping[str, Any]], str | None]


class AIAuthorizationDenied(RuntimeError):
    """Raised before an external call when its credit hold cannot be extended."""


def authorize_ai_cost(
    context: Mapping[str, Any] | None,
    kind: str,
    estimated_credit_units: int,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Run the request authorizer when billing attached one to the context."""

    authorizer = context.get(AI_COST_AUTHORIZER_KEY) if context else None
    if not callable(authorizer):
        return
    error = authorizer(
        str(kind),
        max(0, int(estimated_credit_units or 0)),
        dict(metadata or {}),
    )
    if error:
        raise AIAuthorizationDenied(str(error))


__all__ = [
    "AI_COST_AUTHORIZER_KEY",
    "AI_SEGMENT_RECORDER_KEY",
    "AIAuthorizationDenied",
    "AuthorizeAICost",
    "authorize_ai_cost",
]
