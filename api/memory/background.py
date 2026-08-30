"""Durable, user-funded conversation compaction jobs."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Protocol, cast
from uuid import uuid4

from api.ai.pricing import calculate_billing_for_segments, credit_units_from_usd_micros
from api.billing.credit_units import CREDIT_SCALE, rescale_credit_units
from api.billing.provider_usage import provider_segment_id
from api.billing.reconciliation import (
    mark_ai_operation_active,
    mark_ai_operation_inactive,
)
from api.core.rust_bridge import load_rust_bridge
from api.i18n import current_locale, normalize_locale, use_locale
from api.memory.compaction import CompactionPlan


_JOBS_KEY = "memory:compaction:jobs"
_DEAD_JOBS_KEY = "memory:compaction:dead_jobs"
_LOCK_PREFIX = "memory:compaction:lock:"
_LOCK_TTL_SECONDS = 60 * 60
_POLL_SECONDS = 2.0
_MAX_ATTEMPTS = 3
_COMPACTION_JOB_SCHEMA_VERSION = 1


class _RustCompactionPolicy(Protocol):
    def evaluate_compaction_policy(
        self,
        current_summary: str | None,
        current_marker: str | None,
        prior_summary: str | None,
        expected_marker: str | None,
        result_summary: str | None,
        target_marker: str,
    ) -> str: ...

    def compaction_job_is_due(self, next_attempt_at: float, now: float) -> bool: ...

    def compaction_retry_transition(
        self,
        attempts: int,
        now: float,
        has_billing_segment: bool,
    ) -> str: ...


class _RustCompactionJobs(Protocol):
    def normalize_compaction_job(self, payload: str) -> str: ...


def _load_rust_compaction_policy() -> _RustCompactionPolicy | None:
    module = load_rust_bridge("RUST_COMPACTION_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustCompactionPolicy, module)


def _load_rust_compaction_jobs() -> _RustCompactionJobs | None:
    module = load_rust_bridge("RUST_COMPACTION_JOBS_ENABLED")
    if module is None:
        return None
    return cast(_RustCompactionJobs, module)


def _normalize_compaction_job_payload(
    payload: str,
    decoded: Any,
) -> dict[str, Any]:
    rust = _load_rust_compaction_jobs()
    if rust is not None:
        try:
            normalized = json.loads(rust.normalize_compaction_job(payload))
            if isinstance(normalized, dict):
                decoded = normalized
        except Exception:
            pass
    if not isinstance(decoded, dict):
        raise TypeError("compaction job must be an object")
    version = int(decoded.get("schema_version", _COMPACTION_JOB_SCHEMA_VERSION))
    if version != _COMPACTION_JOB_SCHEMA_VERSION:
        raise ValueError(f"unsupported compaction job schema version {version}")
    return cast(dict[str, Any], decoded)


def _compaction_job_is_due(next_attempt_at: float, now: float) -> bool:
    rust = _load_rust_compaction_policy()
    if rust is not None:
        try:
            return bool(rust.compaction_job_is_due(next_attempt_at, now))
        except Exception:
            pass
    return next_attempt_at <= now


def _compaction_disposition(
    *,
    current_summary: str | None,
    current_marker: str | None,
    prior_summary: str | None,
    expected_marker: str | None,
    result_summary: str | None,
    target_marker: str,
) -> str:
    rust = _load_rust_compaction_policy()
    if rust is not None:
        try:
            return rust.evaluate_compaction_policy(
                current_summary,
                current_marker,
                prior_summary,
                expected_marker,
                result_summary,
                target_marker,
            )
        except Exception:
            pass
    if (
        result_summary
        and current_summary == result_summary
        and current_marker == target_marker
    ):
        return "settle_recovered_success"
    if current_summary != prior_summary or current_marker != expected_marker:
        return "settle_obsolete"
    return "generate_summary" if not result_summary else "save_and_settle"


def _compaction_retry_transition(
    attempts: int,
    now: float,
    has_billing_segment: bool,
) -> dict[str, Any]:
    rust = _load_rust_compaction_policy()
    if rust is not None:
        try:
            transition = json.loads(
                rust.compaction_retry_transition(
                    attempts,
                    now,
                    has_billing_segment,
                )
            )
            if isinstance(transition, dict):
                return cast(dict[str, Any], transition)
        except Exception:
            pass
    attempts += 1
    if attempts >= _MAX_ATTEMPTS:
        return {
            "attempts": attempts,
            "terminal": True,
            "next_attempt_at": None,
            "actual_credit_units": None if has_billing_segment else 0,
        }
    return {
        "attempts": attempts,
        "terminal": False,
        "next_attempt_at": now + 30 * (2 ** (attempts - 1)),
        "actual_credit_units": None,
    }


def _reservation_credit_scale(reservation: Mapping[str, Any]) -> Any:
    metadata = reservation.get("metadata")
    metadata_scale = metadata.get("credit_scale") if isinstance(metadata, Mapping) else None
    return reservation.get("credit_scale") or metadata_scale


def _reservation_settlement_id(reservation: Mapping[str, Any]) -> str:
    metadata = reservation.get("metadata")
    metadata_id = metadata.get("settlement_id") if isinstance(metadata, Mapping) else None
    return str(reservation.get("settlement_id") or metadata_id or "")


def _reservation_operation_id(reservation: Mapping[str, Any]) -> str:
    metadata = reservation.get("metadata")
    metadata_id = metadata.get("operation_id") if isinstance(metadata, Mapping) else None
    return str(reservation.get("operation_id") or metadata_id or "")


@contextmanager
def _active_operation(operation_id: str) -> Iterator[None]:
    if operation_id:
        mark_ai_operation_active(operation_id)
    try:
        yield
    finally:
        if operation_id:
            mark_ai_operation_inactive(operation_id)


@dataclass
class CompactionJob:
    chat_id: str
    messages: list[dict[str, Any]]
    prior_summary: str | None
    expected_marker: str | None
    target_marker: str
    reservation: dict[str, Any]
    user_id: int
    message_id: str | None
    locale: str = "es"
    attempts: int = 0
    next_attempt_at: float = 0.0
    result_summary: str | None = None
    result_cost_usd_micros: int = 0
    result_billing_segment: dict[str, Any] | None = None
    schema_version: int = _COMPACTION_JOB_SCHEMA_VERSION


class DurableCompactionQueue:
    """Keep compaction jobs in Redis and process them outside answer latency."""

    def __init__(
        self,
        *,
        redis_factory: Callable[[], Any],
        compact: Callable[
            [list[dict[str, Any]], str | None],
            tuple[str, int] | tuple[str, int, dict[str, Any] | None],
        ],
        get_summary: Callable[[Any, str], str | None],
        get_marker: Callable[[Any, str], str | None],
        save_result: Callable[[Any, str, str, str], None],
        estimate_reserve: Callable[[CompactionPlan], int],
        settle_reservation: Callable[..., Mapping[str, Any]],
        record_provider_usage: Callable[..., bool],
        list_provider_usage: Callable[..., list[dict[str, Any]]],
        logger: Any,
        admin_report: Callable[..., Any] | None = None,
    ) -> None:
        self._redis_factory = redis_factory
        self._compact = compact
        self._get_summary = get_summary
        self._get_marker = get_marker
        self._save_result = save_result
        self._estimate_reserve = estimate_reserve
        self._settle_reservation = settle_reservation
        self._record_provider_usage = record_provider_usage
        self._list_provider_usage = list_provider_usage
        self._logger = logger
        self._admin_report = admin_report
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue(self, plan: CompactionPlan, billing: Any) -> bool:
        """Reserve the payer's credits and persist one job per chat."""

        try:
            client = self._redis_factory()
            if client.hexists(_JOBS_KEY, plan.chat_id):
                return False
        except Exception as error:
            self._logger.warning("compaction: Redis unavailable before reserve: %s", error)
            return False

        reserve_units = max(1, self._estimate_reserve(plan))
        usage_tag = f"memory_compaction:{plan.chat_id}:{plan.target_marker}:{uuid4().hex}"
        reservation, error_message = billing.reserve_background_ai_credits(
            usage_tag,
            reserve_units,
            metadata={
                "target_marker": plan.target_marker,
                "message_count": len(plan.messages),
                "background": True,
            },
        )
        if error_message or not reservation:
            self._logger.info(
                "compaction: skipped unfunded job chat_id=%s target=%s",
                plan.chat_id,
                plan.target_marker,
            )
            return False

        raw_message_id = getattr(billing, "message", {}).get("message_id")
        job = CompactionJob(
            **asdict(plan),
            reservation={**dict(reservation), "credit_scale": CREDIT_SCALE},
            user_id=int(billing.user_id),
            message_id=str(raw_message_id) if raw_message_id is not None else None,
            locale=current_locale(),
        )
        try:
            stored = bool(
                client.hsetnx(
                    _JOBS_KEY,
                    plan.chat_id,
                    json.dumps(asdict(job), ensure_ascii=False),
                )
            )
        except Exception as error:
            self._logger.warning("compaction: failed to persist job: %s", error)
            stored = False
        if not stored:
            self._settle_reservation(
                user_id=int(billing.user_id),
                chat_id=reservation.get("chat_scope_id"),
                source=str(reservation.get("source") or "user"),
                reserved_credit_units=int(reservation.get("reserved_credit_units") or 0),
                actual_credit_units=0,
                usage_tag=str(reservation.get("usage_tag") or usage_tag),
                metadata={
                    "reason": "memory_compaction_enqueue_failed",
                    "operation_id": _reservation_operation_id(reservation),
                    "settlement_id": _reservation_settlement_id(reservation),
                    "credit_scale": CREDIT_SCALE,
                },
            )
        return stored

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="memory-compaction",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def run_pending_once(self) -> int:
        client = self._redis_factory()
        raw_jobs = client.hgetall(_JOBS_KEY) or {}
        processed = 0
        for raw_chat_id, raw_payload in raw_jobs.items():
            chat_id = self._text(raw_chat_id)
            payload = self._text(raw_payload)
            if not chat_id or not payload:
                continue
            try:
                decoded = json.loads(payload)
            except TypeError, json.JSONDecodeError:
                self._quarantine_job(
                    client,
                    chat_id=chat_id,
                    payload=payload,
                    reason="undecodable",
                )
                continue
            try:
                normalized = _normalize_compaction_job_payload(payload, decoded)
                job = CompactionJob(**normalized)
            except TypeError, ValueError:
                self._logger.exception("compaction: incompatible job chat_id=%s", chat_id)
                if self._settle_invalid_job(decoded, chat_id=chat_id):
                    client.hdel(_JOBS_KEY, chat_id)
                continue
            if not _compaction_job_is_due(job.next_attempt_at, time.time()):
                continue
            token = uuid4().hex
            if not client.set(f"{_LOCK_PREFIX}{chat_id}", token, nx=True, ex=_LOCK_TTL_SECONDS):
                continue
            with _active_operation(_reservation_operation_id(job.reservation)):
                try:
                    self._process(client, job)
                    client.hdel(_JOBS_KEY, chat_id)
                    processed += 1
                except Exception as error:
                    self._retry_or_refund(client, job, error)
                finally:
                    lock_key = f"{_LOCK_PREFIX}{chat_id}"
                    if self._text(client.get(lock_key)) == token:
                        client.delete(lock_key)
        return processed

    def _process(self, client: Any, job: CompactionJob) -> None:
        current_summary = self._get_summary(client, job.chat_id)
        current_marker = self._get_marker(client, job.chat_id) if current_summary else None
        if self._restore_provider_result(job):
            client.hset(
                _JOBS_KEY,
                job.chat_id,
                json.dumps(asdict(job), ensure_ascii=False),
            )
        disposition = _compaction_disposition(
            current_summary=current_summary,
            current_marker=current_marker,
            prior_summary=job.prior_summary,
            expected_marker=job.expected_marker,
            result_summary=job.result_summary,
            target_marker=job.target_marker,
        )
        if disposition == "settle_recovered_success":
            # The process stopped after saving memory but before deleting the
            # durable job. Finish its billing instead of refunding it.
            self._settle(job, reason="memory_compaction_success")
            return
        if disposition == "settle_obsolete":
            self._settle(
                job,
                actual_credit_units=(None if job.result_billing_segment is not None else 0),
                reason="memory_compaction_obsolete",
            )
            return

        if disposition == "generate_summary":
            with use_locale(normalize_locale(job.locale)):
                result = self._compact(job.messages, job.prior_summary)
            summary, cost = result[:2]
            billing_segment = result[2] if len(result) > 2 else None
            if not summary or (cost <= 0 and billing_segment is None):
                raise RuntimeError("summary provider did not produce billable output")
            job.result_summary = summary
            job.result_cost_usd_micros = cost
            job.result_billing_segment = billing_segment
            self._persist_provider_usage(job)
            client.hset(
                _JOBS_KEY,
                job.chat_id,
                json.dumps(asdict(job), ensure_ascii=False),
            )

        else:
            self._persist_provider_usage(job)
        result_summary = job.result_summary
        if not result_summary:
            raise RuntimeError("compaction policy selected save without a result")
        self._save_result(
            client,
            job.chat_id,
            result_summary,
            job.target_marker,
        )
        self._settle(job, reason="memory_compaction_success")
        self._logger.info(
            "compaction: completed chat_id=%s target=%s attempts=%d cost_usd_micros=%d",
            job.chat_id,
            job.target_marker,
            job.attempts + 1,
            job.result_cost_usd_micros,
        )

    def _retry_or_refund(self, client: Any, job: CompactionJob, error: Exception) -> None:
        transition = _compaction_retry_transition(
            job.attempts,
            time.time(),
            job.result_billing_segment is not None,
        )
        job.attempts = int(transition["attempts"])
        self._logger.warning(
            "compaction: attempt failed chat_id=%s attempt=%d/%d error=%s",
            job.chat_id,
            job.attempts,
            _MAX_ATTEMPTS,
            error,
        )
        if bool(transition["terminal"]):
            self._settle(
                job,
                actual_credit_units=transition["actual_credit_units"],
                reason="memory_compaction_failed",
            )
            client.hdel(_JOBS_KEY, job.chat_id)
            return
        job.next_attempt_at = float(transition["next_attempt_at"])
        client.hset(
            _JOBS_KEY,
            job.chat_id,
            json.dumps(asdict(job), ensure_ascii=False),
        )

    def _settle(
        self,
        job: CompactionJob,
        *,
        reason: str,
        actual_credit_units: int | None = None,
    ) -> None:
        reserved = rescale_credit_units(
            job.reservation.get("reserved_credit_units"),
            _reservation_credit_scale(job.reservation),
        )
        operation_id = self._persist_provider_usage(job)
        billing_segments = self._provider_segments(job, operation_id)
        billing = (
            calculate_billing_for_segments(billing_segments)
            if billing_segments
            else None
        )
        pricing_complete = billing is None or billing.get("pricing_complete") is True
        billed_credit_units = (
            int(billing["charged_credit_units"])
            if billing is not None
            else credit_units_from_usd_micros(job.result_cost_usd_micros)
        )
        actual = (
            max(0, int(actual_credit_units))
            if actual_credit_units is not None
            else max(reserved, billed_credit_units)
            if not pricing_complete
            else billed_credit_units
        )
        if not pricing_complete and self._admin_report is not None:
            self._admin_report(
                "compactación de memoria sin costo de proveedor verificable; se mantiene la reserva",
                None,
                {
                    "chat_id": job.chat_id,
                    "user_id": job.user_id,
                    "reserved_credit_units": reserved,
                    "billing_segment": job.result_billing_segment,
                },
            )
        usage_tag = str(job.reservation.get("usage_tag") or "")
        pricing_breakdown = billing or {}
        self._settle_reservation(
            user_id=job.user_id,
            chat_id=job.reservation.get("chat_scope_id"),
            source=str(job.reservation.get("source") or "user"),
            reserved_credit_units=reserved,
            actual_credit_units=actual,
            usage_tag=usage_tag,
            metadata={
                "reason": reason,
                "message_id": job.message_id,
                "settlement_id": _reservation_settlement_id(job.reservation),
                "operation_id": operation_id,
                "credit_scale": CREDIT_SCALE,
                "billing_segments": billing_segments,
                "pricing_version": pricing_breakdown.get("pricing_version"),
                "raw_usd_micros": pricing_breakdown.get("raw_usd_micros", 0),
                "markup_multiplier": pricing_breakdown.get("markup_multiplier"),
                "model_breakdown": pricing_breakdown.get("model_breakdown", []),
                "tool_breakdown": pricing_breakdown.get("tool_breakdown", []),
                "segment_breakdown": pricing_breakdown.get("segment_breakdown", []),
                "pricing_complete": pricing_complete,
            },
        )

    def _persist_provider_usage(self, job: CompactionJob) -> str:
        operation_id = _reservation_operation_id(job.reservation)
        if operation_id and job.result_billing_segment is not None:
            self._record_provider_usage(
                user_id=job.user_id,
                chat_id=job.reservation.get("chat_scope_id"),
                operation_id=operation_id,
                segment_id=provider_segment_id(job.result_billing_segment),
                segment=job.result_billing_segment,
            )
        return operation_id

    def _provider_segments(
        self,
        job: CompactionJob,
        operation_id: str,
    ) -> list[dict[str, Any]]:
        if operation_id:
            durable = self._list_provider_usage(
                user_id=job.user_id,
                operation_id=operation_id,
            )
            if durable:
                return durable
        return (
            [job.result_billing_segment]
            if job.result_billing_segment is not None
            else []
        )

    def _restore_provider_result(self, job: CompactionJob) -> bool:
        if job.result_summary:
            return False
        operation_id = _reservation_operation_id(job.reservation)
        if not operation_id:
            return False
        segments = self._list_provider_usage(
            user_id=job.user_id,
            operation_id=operation_id,
        )
        for segment in reversed(segments):
            summary = str(segment.get("text") or "").strip()
            if segment.get("kind") != "summary" or not summary:
                continue
            billing = calculate_billing_for_segments(segments)
            job.result_summary = summary
            job.result_billing_segment = segment
            job.result_cost_usd_micros = int(billing.get("raw_usd_micros", 0) or 0)
            return True
        return False

    def _settle_invalid_job(self, decoded: Any, *, chat_id: str) -> bool:
        if not isinstance(decoded, Mapping):
            return False
        reservation = decoded.get("reservation")
        if not isinstance(reservation, Mapping) or decoded.get("user_id") is None:
            return False
        usage_tag = str(reservation.get("usage_tag") or "")
        if not usage_tag:
            return False
        try:
            reserved = rescale_credit_units(
                reservation.get("reserved_credit_units"),
                _reservation_credit_scale(reservation),
            )
            self._settle_reservation(
                user_id=int(decoded["user_id"]),
                chat_id=reservation.get("chat_scope_id"),
                source=str(reservation.get("source") or "user"),
                reserved_credit_units=reserved,
                actual_credit_units=0,
                usage_tag=usage_tag,
                metadata={
                    "reason": "memory_compaction_incompatible_job",
                    "chat_id": chat_id,
                    "settlement_id": _reservation_settlement_id(reservation),
                    "operation_id": _reservation_operation_id(reservation),
                    "credit_scale": CREDIT_SCALE,
                },
            )
            return True
        except Exception as error:
            self._logger.warning(
                "compaction: failed to settle incompatible job chat_id=%s error=%s",
                chat_id,
                error,
            )
            return False

    def _quarantine_job(
        self,
        client: Any,
        *,
        chat_id: str,
        payload: str,
        reason: str,
    ) -> bool:
        """Move an unreadable job out of the active queue without losing it."""

        dead_job_id = f"{chat_id}:{uuid4().hex}"
        dead_payload = json.dumps(
            {
                "chat_id": chat_id,
                "payload": payload,
                "reason": reason,
                "quarantined_at": time.time(),
            },
            ensure_ascii=False,
        )
        try:
            client.hset(_DEAD_JOBS_KEY, dead_job_id, dead_payload)
            client.hdel(_JOBS_KEY, chat_id)
        except Exception as error:
            self._logger.warning(
                "compaction: failed to quarantine job chat_id=%s error=%s",
                chat_id,
                error,
            )
            return False
        self._logger.warning(
            "compaction: quarantined job chat_id=%s reason=%s dead_job_id=%s",
            chat_id,
            reason,
            dead_job_id,
        )
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_pending_once()
            except Exception as error:
                self._logger.warning("compaction: worker poll failed: %s", error)
            self._stop.wait(_POLL_SECONDS)

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")
