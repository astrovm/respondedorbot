"""Recover durable AI usage and finish interrupted settlements."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import os
from threading import Event, Lock, Thread
from typing import Any

import requests

from api.ai.pricing import calculate_billing_for_segments
from api.core.logging import get_logger


logger = get_logger(__name__)
_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
_REQUEST_TIMEOUT = (5.0, 20.0)
_DEFAULT_INTERVAL_SECONDS = 60
_DEFAULT_RETRY_WINDOW_SECONDS = 3_600
_DEFAULT_SAFETY_CREDIT_UNITS = 10
_DEFAULT_STALE_SECONDS = 300
_active_operation_counts: dict[str, int] = {}
_active_operation_lock = Lock()


def mark_ai_operation_active(operation_id: str) -> None:
    """Prevent reconciliation while provider work is still running."""

    normalized = str(operation_id or "").strip()
    if not normalized:
        return
    with _active_operation_lock:
        _active_operation_counts[normalized] = (
            _active_operation_counts.get(normalized, 0) + 1
        )


def mark_ai_operation_inactive(operation_id: str) -> None:
    """Allow reconciliation after the last live request has finished."""

    normalized = str(operation_id or "").strip()
    if not normalized:
        return
    with _active_operation_lock:
        remaining = _active_operation_counts.get(normalized, 0) - 1
        if remaining > 0:
            _active_operation_counts[normalized] = remaining
        else:
            _active_operation_counts.pop(normalized, None)


def is_ai_operation_active(operation_id: str) -> bool:
    normalized = str(operation_id or "").strip()
    with _active_operation_lock:
        return _active_operation_counts.get(normalized, 0) > 0


def _configured_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except TypeError, ValueError:
        return default


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except TypeError, ValueError:
        return False


def _needs_reconciliation(segment: Mapping[str, Any]) -> bool:
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


def has_unresolved_provider_usage(segments: list[Mapping[str, Any]]) -> bool:
    return any(_needs_reconciliation(segment) for segment in segments)


def fetch_openrouter_generation(generation_id: str) -> Mapping[str, Any] | None:
    """Return one finalized OpenRouter generation, or None while it is pending."""

    api_key = str(os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None
    response = requests.get(
        _GENERATION_URL,
        params={"id": str(generation_id)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_REQUEST_TIMEOUT,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    return dict(data) if isinstance(data, Mapping) else dict(payload)


def _reconciled_segment(
    segment: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any] | None:
    cost = generation.get("total_cost", generation.get("cost"))
    if not _positive_number(cost):
        return None
    usage = dict(segment.get("usage") or {})
    usage.update(
        {
            "cost": cost,
            "prompt_tokens": generation.get(
                "tokens_prompt", usage.get("prompt_tokens", 0)
            ),
            "completion_tokens": generation.get(
                "tokens_completion", usage.get("completion_tokens", 0)
            ),
        }
    )
    metadata = dict(segment.get("metadata") or {})
    metadata.update(
        {
            "stream_interrupted": False,
            "provider_usage_pending": False,
            "usage_reconciled": True,
            "upstream_provider": generation.get(
                "provider_name", metadata.get("upstream_provider")
            ),
        }
    )
    return {
        **dict(segment),
        "model": generation.get("model") or segment.get("model"),
        "usage": usage,
        "metadata": metadata,
    }


class AIBillingReconciler:
    """Periodically settle durable provider segments left by interrupted work."""

    def __init__(
        self,
        *,
        credits: Any,
        admin_report: Callable[[str, Exception | None, dict[str, Any] | None], None],
        get_generation: Callable[[str], Mapping[str, Any] | None] = (
            fetch_openrouter_generation
        ),
        interval_seconds: int | None = None,
        retry_window_seconds: int | None = None,
        safety_credit_units: int | None = None,
        stale_seconds: int | None = None,
    ) -> None:
        self._credits = credits
        self._admin_report = admin_report
        self._get_generation = get_generation
        self._interval_seconds = max(
            5,
            int(
                interval_seconds
                if interval_seconds is not None
                else _configured_int(
                    "AI_RECONCILIATION_INTERVAL_SECONDS",
                    _DEFAULT_INTERVAL_SECONDS,
                )
            ),
        )
        self._retry_window_seconds = max(
            60,
            int(
                retry_window_seconds
                if retry_window_seconds is not None
                else _configured_int(
                    "AI_RECONCILIATION_RETRY_SECONDS",
                    _DEFAULT_RETRY_WINDOW_SECONDS,
                )
            ),
        )
        self._safety_credit_units = max(
            0,
            int(
                safety_credit_units
                if safety_credit_units is not None
                else _configured_int(
                    "AI_RECONCILIATION_SAFETY_CREDIT_UNITS",
                    _DEFAULT_SAFETY_CREDIT_UNITS,
                )
            ),
        )
        self._stale_seconds = max(
            30,
            int(
                stale_seconds
                if stale_seconds is not None
                else _configured_int(
                    "AI_RECONCILIATION_STALE_SECONDS",
                    _DEFAULT_STALE_SECONDS,
                )
            ),
        )
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="ai-billing-reconciler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("AI billing reconciliation failed")
            self._stop.wait(self._interval_seconds)

    def run_once(self) -> dict[str, int]:
        if not self._credits.is_configured():
            return {"settled": 0, "pending": 0, "unresolved": 0}
        totals = {"settled": 0, "pending": 0, "unresolved": 0}
        for operation in self._credits.list_unsettled_ai_operations():
            outcome = self._reconcile_operation(operation)
            totals[outcome] += 1
        return totals

    def _reconcile_operation(self, operation: Mapping[str, Any]) -> str:
        operation_id = str(operation["operation_id"])
        if is_ai_operation_active(operation_id):
            return "pending"
        if self._age_seconds(operation.get("last_activity_at")) < self._stale_seconds:
            return "pending"
        entries = [dict(item) for item in operation.get("segments") or []]
        segments = [dict(item.get("segment") or {}) for item in entries]
        pending_indexes = [
            index for index, segment in enumerate(segments) if _needs_reconciliation(segment)
        ]
        for index in pending_indexes:
            metadata = dict(segments[index].get("metadata") or {})
            generation_id = metadata.get("provider_generation_id")
            if not generation_id:
                continue
            generation = self._get_generation(str(generation_id))
            if generation is None:
                continue
            reconciled = _reconciled_segment(segments[index], generation)
            if reconciled is None:
                continue
            segment_id = str(entries[index].get("segment_id") or "")
            self._credits.update_ai_provider_usage(
                operation_id,
                segment_id,
                reconciled,
            )
            logger.info(
                "AI provider generation recovered operation_id=%s generation_id=%s",
                operation_id,
                generation_id,
            )
            segments[index] = reconciled

        still_pending = any(_needs_reconciliation(segment) for segment in segments)
        expired = (
            self._age_seconds(operation.get("last_activity_at"))
            >= self._retry_window_seconds
        )
        if still_pending and not expired:
            return "pending"

        breakdown = calculate_billing_for_segments(segments)
        actual = int(breakdown.get("charged_credit_units", 0) or 0)
        if still_pending:
            authorized = int(operation.get("authorized_credit_units", 0) or 0)
            actual = min(authorized, actual + self._safety_credit_units)
            self._admin_report(
                "uso OpenRouter interrumpido no pudo reconciliarse",
                None,
                {
                    "operation_id": operation_id,
                    "authorized_credit_units": authorized,
                    "safety_credit_units": self._safety_credit_units,
                },
            )

        reserve_metadata = dict(operation.get("reserve_metadata") or {})
        settlement_metadata = {
            **reserve_metadata,
            "operation_id": operation_id,
            "reason": (
                "reconciliation_timeout" if still_pending else "recovered_provider_usage"
            ),
            "billing_segments": segments,
            "pricing_version": breakdown.get("pricing_version"),
            "raw_usd_micros": breakdown.get("raw_usd_micros", 0),
            "markup_multiplier": breakdown.get("markup_multiplier"),
            "model_breakdown": breakdown.get("model_breakdown", []),
            "tool_breakdown": breakdown.get("tool_breakdown", []),
            "segment_breakdown": breakdown.get("segment_breakdown", []),
            "pricing_complete": not still_pending and breakdown.get("pricing_complete", False),
            "reconciliation_unresolved": still_pending,
        }
        self._credits.settle_ai_operation_once(
            user_id=int(operation["user_id"]),
            chat_id=operation.get("chat_id"),
            operation_id=operation_id,
            actual_credit_units=actual,
            metadata=settlement_metadata,
        )
        logger.info(
            "AI operation reconciled operation_id=%s authorized_units=%d settled_units=%d unresolved=%s",
            operation_id,
            int(operation.get("authorized_credit_units", 0) or 0),
            actual,
            still_pending,
        )
        return "unresolved" if still_pending else "settled"

    @staticmethod
    def _age_seconds(created_at: Any) -> float:
        if not isinstance(created_at, datetime):
            return 0
        normalized = (
            created_at.replace(tzinfo=UTC)
            if created_at.tzinfo is None
            else created_at
        )
        return max(0.0, (datetime.now(UTC) - normalized).total_seconds())


__all__ = [
    "AIBillingReconciler",
    "fetch_openrouter_generation",
    "has_unresolved_provider_usage",
    "is_ai_operation_active",
    "mark_ai_operation_active",
    "mark_ai_operation_inactive",
]
