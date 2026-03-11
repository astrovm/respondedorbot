"""AI credits billing helpers used by commands and message flow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import environ
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from api.chat_context import (
    extract_numeric_chat_id,
    extract_user_id,
    is_group_chat_type,
)
from api.groq_billing import calculate_billing_for_segments


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


AI_BILLING_DEFAULT_PACKS = [
    {"id": "p50", "credits": 50, "xtr": 25},
    {"id": "p100", "credits": 100, "xtr": 50},
    {"id": "p250", "credits": 250, "xtr": 125},
    {"id": "p500", "credits": 500, "xtr": 250},
    {"id": "p1000", "credits": 1000, "xtr": 500},
    {"id": "p2500", "credits": 2500, "xtr": 1250},
]
def get_ai_credits_per_response() -> int:
    """Return credits charged per AI response."""

    raw_value = str(environ.get("AI_CREDITS_PER_RESPONSE") or "1").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def get_ai_onboarding_credits() -> int:
    """Return onboarding credits granted once per user."""

    raw_value = str(environ.get("AI_ONBOARDING_CREDITS") or "3").strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return 3
    return max(0, parsed)


def get_ai_billing_packs() -> List[Dict[str, int]]:
    """Load Stars billing packs from env or defaults."""

    raw_value = str(environ.get("AI_STARS_PACKS_JSON") or "").strip()
    if not raw_value:
        return list(AI_BILLING_DEFAULT_PACKS)

    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError:
        return list(AI_BILLING_DEFAULT_PACKS)

    if not isinstance(loaded, list):
        return list(AI_BILLING_DEFAULT_PACKS)

    normalized: List[Dict[str, int]] = []
    for item in loaded:
        if not isinstance(item, Mapping):
            continue
        pack_id = str(item.get("id", "")).strip()
        try:
            credits = int(item.get("credits"))  # type: ignore[arg-type]
            xtr = int(item.get("xtr"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not pack_id or credits <= 0 or xtr <= 0:
            continue
        normalized.append({"id": pack_id, "credits": credits, "xtr": xtr})

    return normalized or list(AI_BILLING_DEFAULT_PACKS)


def get_ai_billing_pack(pack_id: str) -> Optional[Dict[str, int]]:
    """Return the pack dict matching pack_id."""

    for pack in get_ai_billing_packs():
        if str(pack.get("id")) == str(pack_id):
            return pack
    return None


def build_topup_keyboard() -> Dict[str, Any]:
    """Build inline keyboard with top-up packs."""

    rows: List[List[Dict[str, str]]] = []
    for pack in get_ai_billing_packs():
        pack_id = str(pack["id"])
        rows.append(
            [
                {
                    "text": f"{pack['credits']} créditos - {pack['xtr']} ⭐",
                    "callback_data": f"topup:{pack_id}",
                }
            ]
        )
    return {"inline_keyboard": rows}


def parse_topup_payload(payload: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse invoice payload for top-up purchases."""

    if not payload:
        return None, None
    parts = str(payload).split(":")
    if len(parts) < 2 or parts[0] != "topup":
        return None, None

    user_id: Optional[int] = None
    if len(parts) >= 3:
        try:
            user_id = int(parts[2])
        except (TypeError, ValueError):
            user_id = None
    return parts[1], user_id


def build_insufficient_credits_message(
    *,
    chat_type: str,
    user_balance: int,
    chat_balance: int,
) -> str:
    """Build a user-facing paywall message when no credits are available."""

    if is_group_chat_type(chat_type):
        return (
            "se quedaron secos de créditos ia en este grupo, boludo.\n"
            f"- lo tuyo: {user_balance}\n"
            f"- lo del grupo: {chat_balance}\n"
            "metele /topup por privado y si querés pasá saldo al grupo con /transfer <monto>\n"
            "si querés ver bien la miseria, mandá /balance"
        )

    return (
        "te quedaste seco de créditos ia, boludo.\n"
        f"saldo: {user_balance}\n"
        "metele /topup si querés que siga laburando"
    )
def maybe_grant_onboarding_credits(
    credits_db_service: Any,
    admin_reporter: AdminReporter,
    user_id: Optional[int],
) -> None:
    """Grant onboarding credits when configured and not yet granted."""

    if user_id is None:
        return

    onboarding_credits = get_ai_onboarding_credits()
    if onboarding_credits <= 0:
        return

    try:
        credits_db_service.grant_onboarding_if_needed(user_id, onboarding_credits)
    except Exception as error:
        admin_reporter(
            "falló la acreditación de onboarding",
            error,
            {"user_id": user_id},
        )


def format_balance_command(
    credits_db_service: Any,
    *,
    chat_type: str,
    user_id: int,
    chat_id: int,
) -> str:
    """Format the /balance response for private and group chats."""

    user_balance = credits_db_service.get_balance("user", int(user_id))
    if is_group_chat_type(chat_type):
        chat_balance = credits_db_service.get_balance("chat", int(chat_id))
        return (
            "saldos ia, maestro:\n"
            f"- lo tuyo: {user_balance}\n"
            f"- lo del grupo: {chat_balance}\n"
            "si no alcanza lo tuyo, manoteo del grupo\n"
            "si querés cargar más: /topup por privado\n"
            "si querés pasarle al grupo: /transfer <monto>"
        )

    return f"tenés {user_balance} créditos ia\nsi querés cargar más mandale /topup"


@dataclass
class AIMessageBilling:
    """Charge/refund helper for a single handled message."""

    credits_db_service: Any
    admin_reporter: AdminReporter
    gen_random_fn: Callable[[str], str]
    build_insufficient_credits_message_fn: Callable[..., str]
    maybe_grant_onboarding_credits_fn: Callable[[Optional[int]], None]
    get_ai_credits_per_response_fn: Callable[[], int]
    command: str
    chat_id: str
    chat_type: str
    user_id: Optional[int]
    numeric_chat_id: Optional[int]
    message: Mapping[str, Any]
    onboarding_checked: bool = False
    billing_not_configured_message: str = "el cobro de ia no está andando, avisale al admin"
    billing_missing_scope_message: str = "no te pude sacar bien el usuario o el chat para cobrar, qué quilombo"
    billing_charge_error_message: str = "se trabó el cobro de ia, probá de nuevo"
    charge_errors: List[str] = field(default_factory=list)

    def _resolve_ai_charge_context(self) -> Tuple[Optional[int], Optional[str]]:
        if not self.credits_db_service.is_configured():
            return None, self.billing_not_configured_message
        if self.user_id is None or self.numeric_chat_id is None:
            return None, self.billing_missing_scope_message
        return (
            self.numeric_chat_id if is_group_chat_type(self.chat_type) else None,
            None,
        )

    def _build_insufficient_credits_reply(self, charge_result: Mapping[str, Any]) -> str:
        random_name = str(self.message.get("from", {}).get("first_name") or "boludo")
        random_response = self.gen_random_fn(random_name)
        credits_message = self.build_insufficient_credits_message_fn(
            chat_type=self.chat_type,
            user_balance=int(charge_result.get("user_balance", 0)),
            chat_balance=int(charge_result.get("chat_balance", 0)),
        )
        return f"{random_response}\n\n{credits_message}"

    def _ensure_onboarding_checked(self) -> None:
        if not self.onboarding_checked:
            self.maybe_grant_onboarding_credits_fn(self.user_id)
            self.onboarding_checked = True

    def _build_charge_metadata(
        self,
        *,
        usage_tag: str,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "command": self.command,
            "usage_tag": usage_tag,
        }
        if extra:
            metadata.update(dict(extra))
        return metadata

    def reserve_ai_credits(
        self,
        usage_tag: str,
        estimated_credits: int,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Reserve a worst-case number of credits for a billable interaction."""

        chat_scope_id, context_error = self._resolve_ai_charge_context()
        if context_error:
            return None, context_error
        if self.user_id is None:
            return None, self.billing_missing_scope_message

        self._ensure_onboarding_checked()

        reserve_amount = max(0, int(estimated_credits or 0))
        reserve_metadata = self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={"reserved_credits": reserve_amount, **dict(metadata or {})},
        )

        try:
            charge_result = self.credits_db_service.charge_ai_credits(
                user_id=self.user_id,
                chat_id=chat_scope_id,
                amount=reserve_amount,
                event_type="ai_reserve",
                metadata=reserve_metadata,
            )
        except Exception as error:
            self.admin_reporter(
                "Error reserving IA credits",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "command": self.command,
                    "usage_tag": usage_tag,
                    "estimated_credits": reserve_amount,
                },
            )
            return None, self.billing_charge_error_message

        if not charge_result.get("ok"):
            return None, self._build_insufficient_credits_reply(charge_result)

        return {
            "reserved_credits": reserve_amount,
            "chat_scope_id": chat_scope_id,
            "source": str(charge_result.get("source") or "user"),
            "usage_tag": usage_tag,
            "metadata": reserve_metadata,
        }, None

    def settle_reserved_ai_credits(
        self,
        reservation_meta: Optional[Mapping[str, Any]],
        billing_segments: Optional[List[Mapping[str, Any]]],
        *,
        reason: str,
    ) -> None:
        """Settle a reservation using actual Groq usage, charging extra only if needed."""

        self.settle_reserved_ai_credits_batch(
            [reservation_meta] if reservation_meta else [],
            billing_segments,
            reason=reason,
        )

    def _build_settlement_metadata(
        self,
        *,
        usage_tag: str,
        usage_tags: Sequence[str],
        reserved_credits_total: int,
        settled_credits: int,
        refunded_credits: int,
        extra_charged_credits: int,
        debt_applied_credits: int,
        reason: str,
        breakdown: Mapping[str, Any],
        billing_segments: Sequence[Mapping[str, Any]],
        missing_usage_billing: bool,
    ) -> Dict[str, Any]:
        return self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={
                "reason": reason,
                "message_id": self.message.get("message_id"),
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "command": self.command,
                "usage_tags": list(usage_tags),
                "reserved_credits_total": reserved_credits_total,
                "settled_credits": settled_credits,
                "refunded_credits": refunded_credits,
                "extra_charged_credits": extra_charged_credits,
                "debt_applied_credits": debt_applied_credits,
                "pricing_version": breakdown.get("pricing_version"),
                "raw_usd_micros": breakdown.get("raw_usd_micros", 0),
                "markup_multiplier": breakdown.get("markup_multiplier"),
                "model_breakdown": breakdown.get("model_breakdown", []),
                "tool_breakdown": breakdown.get("tool_breakdown", []),
                "unsupported_notes": breakdown.get("unsupported_notes", []),
                "billing_segments": list(billing_segments or []),
                "missing_usage_billing": bool(missing_usage_billing),
            },
        )

    def _record_ai_settlement_result(
        self,
        *,
        chat_scope_id: Optional[int],
        settlement_metadata: Mapping[str, Any],
    ) -> None:
        if self.user_id is None:
            return
        try:
            self.credits_db_service.record_ai_settlement_result(
                user_id=self.user_id,
                chat_id=chat_scope_id,
                actor_user_id=self.user_id,
                metadata=settlement_metadata,
            )
        except Exception as error:
            self.admin_reporter(
                "falló registrar resultado de liquidación IA",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "command": self.command,
                },
            )

    def _settle_single_reservation(
        self,
        reservation_meta: Mapping[str, Any],
        billing_segments: Optional[List[Mapping[str, Any]]],
        *,
        reason: str,
    ) -> None:
        """Settle one reservation preserving legacy behavior where needed."""

        if not reservation_meta or self.user_id is None:
            return

        reserved_credits = int(reservation_meta.get("reserved_credits", 0) or 0)
        usage_tag = str(reservation_meta.get("usage_tag") or "ai_usage")
        usage_tags = [usage_tag]
        if not billing_segments:
            breakdown = {
                "pricing_version": None,
                "markup_multiplier": None,
                "raw_usd_micros": 0,
                "model_breakdown": [],
                "tool_breakdown": [],
                "unsupported_notes": ["missing_billing_segments_reserve_retained"],
            }
            settlement_metadata = self._build_settlement_metadata(
                usage_tag=usage_tag,
                usage_tags=usage_tags,
                reserved_credits_total=reserved_credits,
                settled_credits=reserved_credits,
                refunded_credits=0,
                extra_charged_credits=0,
                debt_applied_credits=0,
                reason=reason,
                breakdown=breakdown,
                billing_segments=list(billing_segments or []),
                missing_usage_billing=True,
            )
            self._record_ai_settlement_result(
                chat_scope_id=reservation_meta.get("chat_scope_id"),
                settlement_metadata=settlement_metadata,
            )
            self.admin_reporter(
                "respuesta IA exitosa sin usage billing; se mantiene cobro por reserva (sin reintegro)",
                None,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "reserved_credits": reserved_credits,
                },
            )
            return
        breakdown = calculate_billing_for_segments(billing_segments or [])
        actual_credits = int(breakdown.get("charged_credits", 0) or 0)
        refunded_credits = 0
        extra_charged_credits = 0
        debt_applied_credits = 0
        chat_scope_id = reservation_meta.get("chat_scope_id")
        source = "chat" if str(reservation_meta.get("source") or "user") == "chat" else "user"

        if actual_credits < reserved_credits:
            refunded_credits = reserved_credits - actual_credits
            try:
                self.credits_db_service.refund_ai_charge(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=refunded_credits,
                    source=source,
                    event_type="ai_refund",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={
                            "reason": reason,
                            "reserved_credits_total": reserved_credits,
                            "settled_credits": actual_credits,
                            "refunded_credits": refunded_credits,
                        },
                    ),
                )
            except Exception as error:
                self.admin_reporter(
                    "falló el reintegro de liquidación IA",
                    error,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits,
                        "actual_credits": actual_credits,
                        "reason": reason,
                    },
                )
                refunded_credits = 0

        elif actual_credits > reserved_credits:
            try:
                extra_charge = self.credits_db_service.charge_ai_credits(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=actual_credits - reserved_credits,
                    event_type="ai_settlement_charge",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={
                            "reason": reason,
                            "reserved_credits_total": reserved_credits,
                            "settled_credits": actual_credits,
                            "extra_charged_credits": actual_credits - reserved_credits,
                        },
                    ),
                )
            except Exception as error:
                self.admin_reporter(
                    "falló el ajuste de liquidación IA",
                    error,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits,
                        "actual_credits": actual_credits,
                        "reason": reason,
                    },
                )
                extra_charge = {"ok": False}

            if not extra_charge.get("ok"):
                self.admin_reporter(
                    "la liquidación IA superó la reserva y no pudo cobrar ajuste",
                    None,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits,
                        "actual_credits": actual_credits,
                        "reason": reason,
                        "billing_segments": list(billing_segments or []),
                    },
                )
                try:
                    self.credits_db_service.apply_ai_debt(
                        user_id=self.user_id,
                        chat_id=chat_scope_id,
                        amount=actual_credits - reserved_credits,
                        source=source,
                        event_type="ai_settlement_debt",
                        metadata=self._build_charge_metadata(
                            usage_tag=usage_tag,
                            extra={
                                "reason": reason,
                                "reserved_credits_total": reserved_credits,
                                "settled_credits": actual_credits,
                                "debt_applied_credits": actual_credits - reserved_credits,
                            },
                        ),
                    )
                    debt_applied_credits = actual_credits - reserved_credits
                except Exception as error:
                    self.admin_reporter(
                        "falló registrar deuda de liquidación IA",
                        error,
                        {
                            "chat_id": self.chat_id,
                            "user_id": self.user_id,
                            "reserved_credits": reserved_credits,
                            "actual_credits": actual_credits,
                            "reason": reason,
                        },
                    )
            else:
                extra_charged_credits = actual_credits - reserved_credits

        settlement_metadata = self._build_settlement_metadata(
            usage_tag=usage_tag,
            usage_tags=usage_tags,
            reserved_credits_total=reserved_credits,
            settled_credits=actual_credits,
            refunded_credits=refunded_credits,
            extra_charged_credits=extra_charged_credits,
            debt_applied_credits=debt_applied_credits,
            reason=reason,
            breakdown=breakdown,
            billing_segments=list(billing_segments or []),
            missing_usage_billing=False,
        )
        self._record_ai_settlement_result(
            chat_scope_id=chat_scope_id,
            settlement_metadata=settlement_metadata,
        )

    def settle_reserved_ai_credits_batch(
        self,
        reservation_metas: Iterable[Optional[Mapping[str, Any]]],
        billing_segments: Optional[List[Mapping[str, Any]]],
        *,
        reason: str,
    ) -> None:
        reservations = [dict(item) for item in reservation_metas if item]
        if not reservations or self.user_id is None:
            return

        source_values = {str(item.get("source") or "user") for item in reservations}
        chat_scope_values = {item.get("chat_scope_id") for item in reservations}
        if len(source_values) > 1 or len(chat_scope_values) > 1:
            self.admin_reporter(
                "liquidación IA batch con reservas incompatibles; vuelvo a liquidación individual",
                None,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "reservation_count": len(reservations),
                },
            )
            for index, reservation in enumerate(reservations):
                self._settle_single_reservation(
                    reservation,
                    billing_segments if index == 0 else [],
                    reason=reason,
                )
            return

        if len(reservations) == 1:
            self._settle_single_reservation(
                reservations[0],
                billing_segments,
                reason=reason,
            )
            return

        reserved_credits_total = sum(
            int(item.get("reserved_credits", 0) or 0) for item in reservations
        )
        usage_tags = [str(item.get("usage_tag") or "ai_usage") for item in reservations]
        usage_tag = usage_tags[0] if len(set(usage_tags)) == 1 else "ai_usage_batch"
        chat_scope_id = reservations[0].get("chat_scope_id")
        source = "chat" if str(reservations[0].get("source") or "user") == "chat" else "user"

        if not billing_segments:
            breakdown = {
                "pricing_version": None,
                "markup_multiplier": None,
                "raw_usd_micros": 0,
                "model_breakdown": [],
                "tool_breakdown": [],
                "unsupported_notes": ["missing_billing_segments_reserve_retained"],
            }
            settlement_metadata = self._build_settlement_metadata(
                usage_tag=usage_tag,
                usage_tags=usage_tags,
                reserved_credits_total=reserved_credits_total,
                settled_credits=reserved_credits_total,
                refunded_credits=0,
                extra_charged_credits=0,
                debt_applied_credits=0,
                reason=reason,
                breakdown=breakdown,
                billing_segments=list(billing_segments or []),
                missing_usage_billing=True,
            )
            self._record_ai_settlement_result(
                chat_scope_id=chat_scope_id,
                settlement_metadata=settlement_metadata,
            )
            self.admin_reporter(
                "respuesta IA exitosa sin usage billing; se mantiene cobro por reserva (sin reintegro)",
                None,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "reserved_credits": reserved_credits_total,
                },
            )
            return

        breakdown = calculate_billing_for_segments(billing_segments or [])
        actual_credits = int(breakdown.get("charged_credits", 0) or 0)
        refunded_credits = 0
        extra_charged_credits = 0
        debt_applied_credits = 0

        if actual_credits < reserved_credits_total:
            refunded_credits = reserved_credits_total - actual_credits
            try:
                self.credits_db_service.refund_ai_charge(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=refunded_credits,
                    source=source,
                    event_type="ai_refund",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={
                            "reason": reason,
                            "reserved_credits_total": reserved_credits_total,
                            "settled_credits": actual_credits,
                            "refunded_credits": refunded_credits,
                            "usage_tags": list(usage_tags),
                        },
                    ),
                )
            except Exception as error:
                self.admin_reporter(
                    "falló el reintegro batch de liquidación IA",
                    error,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits_total,
                        "actual_credits": actual_credits,
                        "reason": reason,
                    },
                )
                refunded_credits = 0

        elif actual_credits > reserved_credits_total:
            extra_amount = actual_credits - reserved_credits_total
            try:
                extra_charge = self.credits_db_service.charge_ai_credits(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=extra_amount,
                    event_type="ai_settlement_charge",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={
                            "reason": reason,
                            "reserved_credits_total": reserved_credits_total,
                            "settled_credits": actual_credits,
                            "extra_charged_credits": extra_amount,
                            "usage_tags": list(usage_tags),
                        },
                    ),
                )
            except Exception as error:
                self.admin_reporter(
                    "falló el ajuste batch de liquidación IA",
                    error,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits_total,
                        "actual_credits": actual_credits,
                        "reason": reason,
                    },
                )
                extra_charge = {"ok": False}

            if not extra_charge.get("ok"):
                self.admin_reporter(
                    "la liquidación IA batch superó la reserva y no pudo cobrar ajuste",
                    None,
                    {
                        "chat_id": self.chat_id,
                        "user_id": self.user_id,
                        "reserved_credits": reserved_credits_total,
                        "actual_credits": actual_credits,
                        "reason": reason,
                        "billing_segments": list(billing_segments or []),
                    },
                )
                try:
                    self.credits_db_service.apply_ai_debt(
                        user_id=self.user_id,
                        chat_id=chat_scope_id,
                        amount=extra_amount,
                        source=source,
                        event_type="ai_settlement_debt",
                        metadata=self._build_charge_metadata(
                            usage_tag=usage_tag,
                            extra={
                                "reason": reason,
                                "reserved_credits_total": reserved_credits_total,
                                "settled_credits": actual_credits,
                                "debt_applied_credits": extra_amount,
                                "usage_tags": list(usage_tags),
                            },
                        ),
                    )
                    debt_applied_credits = extra_amount
                except Exception as error:
                    self.admin_reporter(
                        "falló registrar deuda batch de liquidación IA",
                        error,
                        {
                            "chat_id": self.chat_id,
                            "user_id": self.user_id,
                            "reserved_credits": reserved_credits_total,
                            "actual_credits": actual_credits,
                            "reason": reason,
                        },
                    )
            else:
                extra_charged_credits = extra_amount

        settlement_metadata = self._build_settlement_metadata(
            usage_tag=usage_tag,
            usage_tags=usage_tags,
            reserved_credits_total=reserved_credits_total,
            settled_credits=actual_credits,
            refunded_credits=refunded_credits,
            extra_charged_credits=extra_charged_credits,
            debt_applied_credits=debt_applied_credits,
            reason=reason,
            breakdown=breakdown,
            billing_segments=list(billing_segments or []),
            missing_usage_billing=False,
        )
        self._record_ai_settlement_result(
            chat_scope_id=chat_scope_id,
            settlement_metadata=settlement_metadata,
        )

    def refund_reserved_ai_credits(
        self,
        reservation_meta: Optional[Mapping[str, Any]],
        *,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Refund a full reservation when the interaction does not produce a billable result."""

        if not reservation_meta or self.user_id is None:
            return

        reserved_credits = int(reservation_meta.get("reserved_credits", 0) or 0)
        source = "chat" if str(reservation_meta.get("source") or "user") == "chat" else "user"
        usage_tag = str(reservation_meta.get("usage_tag") or "ai_usage")
        refund_metadata = self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={
                "reason": reason,
                "reserved_credits": reserved_credits,
                "settled_credits": 0,
                "refunded_credits": reserved_credits,
                **dict(metadata or {}),
            },
        )

        try:
            self.credits_db_service.refund_ai_charge(
                user_id=self.user_id,
                chat_id=reservation_meta.get("chat_scope_id"),
                amount=reserved_credits,
                source=source,
                event_type="ai_refund",
                metadata=refund_metadata,
            )
        except Exception as refund_error:
            self.admin_reporter(
                "falló el reintegro de créditos IA",
                refund_error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "command": self.command,
                },
            )

    def charge_one_ai_request(
        self, usage_tag: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Charge one AI request, returning charge metadata or a user-facing error."""

        chat_scope_id, context_error = self._resolve_ai_charge_context()
        if context_error:
            return None, context_error
        if self.user_id is None:
            return None, self.billing_missing_scope_message

        return self.reserve_ai_credits(
            usage_tag,
            self.get_ai_credits_per_response_fn(),
        )

    def refund_ai_charge_meta(
        self,
        charge_meta: Optional[Mapping[str, Any]],
        reason: str,
    ) -> None:
        """Refund a previously applied AI charge."""

        if not charge_meta or self.user_id is None:
            return

        self.refund_reserved_ai_credits(
            {
                **dict(charge_meta),
                "reserved_credits": int(charge_meta.get("credits_cost", charge_meta.get("reserved_credits", 1))),
            },
            reason=reason,
        )

    @staticmethod
    def is_transcribe_success_response(text: Optional[str]) -> bool:
        """Return True when a media transcription/description completed successfully."""

        if not text:
            return False
        success_prefixes = (
            "🎵 te saqué esto del audio: ",
            "🖼️ en la imagen veo: ",
            "🎨 en el sticker veo: ",
        )
        return text.startswith(success_prefixes)


__all__ = [
    "AIMessageBilling",
    "build_insufficient_credits_message",
    "build_topup_keyboard",
    "extract_numeric_chat_id",
    "extract_user_id",
    "format_balance_command",
    "get_ai_billing_pack",
    "get_ai_billing_packs",
    "get_ai_credits_per_response",
    "get_ai_onboarding_credits",
    "maybe_grant_onboarding_credits",
    "parse_topup_payload",
]
