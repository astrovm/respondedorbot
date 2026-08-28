"""AI credits billing helpers used by commands and message flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

from api.i18n import tr
from api.bot.chat_context import (
    is_group_chat_type,
)
from api.billing.credit_units import (
    CREDIT_SCALE,
    format_credit_units,
    rescale_credit_units,
    whole_credits_to_units,
)
from api.ai.pricing import calculate_billing_for_segments
from api.ai.random_replies import build_random_reply

AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


@dataclass(frozen=True, slots=True)
class BatchReservation:
    items: list[dict[str, Any]]
    reserved_credit_units: int
    usage_tags: list[str]
    usage_tag: str
    chat_scope_id: Optional[int]
    source: str


@dataclass(frozen=True, slots=True)
class SettlementAdjustment:
    settled_credit_units: int
    refunded_credit_units: int = 0
    extra_charged_credit_units: int = 0
    debt_applied_credit_units: int = 0
    extra_payer_scope: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BatchSettlementRecord:
    reason: str
    breakdown: Mapping[str, Any]
    billing_segments: Sequence[Mapping[str, Any]]
    missing_usage_billing: bool = False


class AIBillingPack(TypedDict):
    id: str
    credits: int
    xtr: int


AI_BILLING_DEFAULT_PACKS: List[AIBillingPack] = [
    {"id": "p50", "credits": whole_credits_to_units(50), "xtr": 25},
    {"id": "p100", "credits": whole_credits_to_units(100), "xtr": 50},
    {"id": "p250", "credits": whole_credits_to_units(250), "xtr": 125},
    {"id": "p500", "credits": whole_credits_to_units(500), "xtr": 250},
    {"id": "p1000", "credits": whole_credits_to_units(1000), "xtr": 500},
    {"id": "p2500", "credits": whole_credits_to_units(2500), "xtr": 1250},
]


def get_ai_onboarding_credits() -> int:
    """Return onboarding credit units granted once per user."""
    return whole_credits_to_units(3)


def get_ai_billing_packs() -> List[AIBillingPack]:
    """Return Stars billing packs."""
    return list(AI_BILLING_DEFAULT_PACKS)


def get_ai_billing_pack(pack_id: str) -> Optional[AIBillingPack]:
    """Return the pack dict matching pack_id."""

    for pack in get_ai_billing_packs():
        if str(pack.get("id")) == str(pack_id):
            return pack
    return None


def _billing_summary_int(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def _billing_is_complete(summary: Mapping[str, Any]) -> bool:
    return summary.get("pricing_complete") is True


def build_topup_keyboard() -> Dict[str, Any]:
    """Build inline keyboard with top-up packs."""

    rows: List[List[Dict[str, str]]] = []
    for pack in get_ai_billing_packs():
        pack_id = str(pack["id"])
        rows.append(
            [
                {
                    "text": tr(
                        "topup.pack_button",
                        credits=format_credit_units(pack["credits"]),
                        stars=pack["xtr"],
                    ),
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
        except TypeError, ValueError:
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
        return tr(
            "billing.insufficient_group",
            user=format_credit_units(user_balance),
            group=format_credit_units(chat_balance),
        )

    return tr(
        "billing.insufficient_private",
        balance=format_credit_units(user_balance),
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
        return tr(
            "balance.group_full",
            user=format_credit_units(user_balance),
            group=format_credit_units(chat_balance),
        )

    return (
        tr("balance.user", balance=format_credit_units(user_balance))
        + "\n"
        + tr("balance.private_topup")
    )


@dataclass(frozen=True)
class BalanceFormatter:
    credits_db_service: Any

    def format(self, *, chat_type: str, user_id: int, chat_id: int) -> str:
        return format_balance_command(
            self.credits_db_service,
            chat_type=chat_type,
            user_id=user_id,
            chat_id=chat_id,
        )


@dataclass
class AIMessageBilling:
    """Charge/refund helper for a single handled message."""

    credits_db_service: Any
    admin_reporter: AdminReporter
    gen_random_fn: Callable[[str], str]
    build_insufficient_credits_message_fn: Callable[..., str]
    maybe_grant_onboarding_credits_fn: Callable[[Optional[int]], None]
    command: str
    chat_id: str
    chat_type: str
    user_id: Optional[int]
    numeric_chat_id: Optional[int]
    message: Mapping[str, Any]
    redis_client: Any = None
    creditless_user_hourly_limit: int = 0
    onboarding_checked: bool = False
    billing_not_configured_message: str = field(default_factory=lambda: tr("billing.unavailable"))
    billing_missing_scope_message: str = field(default_factory=lambda: tr("billing.missing_scope"))
    billing_charge_error_message: str = field(default_factory=lambda: tr("billing.charge_error"))
    charge_errors: List[str] = field(default_factory=list)
    load_persisted_reservation_fn: Callable[[str], Optional[Mapping[str, Any]]] = (
        lambda _usage_tag: None
    )
    persist_reservation_fn: Callable[[str, Mapping[str, Any]], None] = (
        lambda _usage_tag, _reservation: None
    )
    clear_persisted_reservation_fn: Callable[[str], None] = lambda _usage_tag: None

    def _resolve_ai_charge_context(self) -> Tuple[Optional[int], Optional[str]]:
        if not self.credits_db_service.is_configured():
            return None, self.billing_not_configured_message
        if self.user_id is None:
            return None, self.billing_missing_scope_message
        if is_group_chat_type(self.chat_type) and self.numeric_chat_id is None:
            return None, self.billing_missing_scope_message
        return (
            self.numeric_chat_id if is_group_chat_type(self.chat_type) else None,
            None,
        )

    def _check_creditless_cap(
        self,
        *,
        chat_scope_id: Optional[int],
        reserve_amount: int,
        usage_tag: str,
    ) -> Optional[str]:
        """After a chat-sourced charge, enforce per-user hourly cap via Redis counter.

        Returns an error string if the cap was hit (charge already refunded), else None.
        """
        key = self._creditless_cap_key(chat_scope_id)
        if key is None:
            return None

        count = self.redis_client.incr(key)
        if count == 1:
            self.redis_client.expire(key, 3600)

        if count > self.creditless_user_hourly_limit:
            try:
                self.credits_db_service.refund_ai_charge(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=reserve_amount,
                    source="chat",
                    event_type="ai_refund",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={"reason": "creditless_hourly_cap"},
                    ),
                )
            except Exception as err:
                self.admin_reporter(
                    "falló refund por limite creditless",
                    err,
                    {"chat_id": self.chat_id, "user_id": self.user_id},
                )
            return tr(
                "billing.group_cap",
                limit=self.creditless_user_hourly_limit,
            )
        return None

    def _creditless_cap_key(self, chat_scope_id: Optional[int]) -> Optional[str]:
        if (
            self.creditless_user_hourly_limit < 0
            or self.redis_client is None
            or self.user_id is None
            or chat_scope_id is None
        ):
            return None
        return f"creditless_cap:{self.chat_id}:{self.user_id}"

    def _rollback_creditless_cap(self, reservation_meta: Optional[Mapping[str, Any]]) -> None:
        if not reservation_meta:
            return
        if str(reservation_meta.get("source") or "user") != "chat":
            return

        key = self._creditless_cap_key(reservation_meta.get("chat_scope_id"))
        if key is None:
            return

        try:
            self.redis_client.decr(key)
        except Exception as error:
            self.admin_reporter(
                "falló rollback de limite creditless",
                error,
                {"chat_id": self.chat_id, "user_id": self.user_id},
            )

    def _build_insufficient_credits_reply(self, charge_result: Mapping[str, Any]) -> str:
        random_response = build_random_reply(
            self.gen_random_fn,
            cast(Mapping[str, Any], self.message.get("from") or {}),
        )
        credits_message = self.build_insufficient_credits_message_fn(
            chat_type=self.chat_type,
            user_balance=int(charge_result.get("user_balance_credit_units", 0)),
            chat_balance=int(charge_result.get("chat_balance_credit_units", 0)),
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
            "settlement_id": self._settlement_id(usage_tag),
            "message_id": self.message.get("message_id"),
            "origin_chat_id": self.chat_id,
            "credit_scale": CREDIT_SCALE,
        }
        if extra:
            metadata.update(dict(extra))
        return metadata

    def _settlement_id(self, usage_tag: str) -> str:
        message_id = self.message.get("message_id")
        return ":".join(
            (
                str(self.user_id or "unknown"),
                self.chat_id,
                str(message_id if message_id is not None else "unknown"),
                str(usage_tag or "ai_usage"),
            )
        )

    def reserve_ai_credits(
        self,
        usage_tag: str,
        estimated_credit_units: int,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Reserve a worst-case number of credits for a billable interaction."""

        return self._reserve_ai_credits(
            usage_tag,
            estimated_credit_units,
            metadata=metadata,
            enforce_message_cap=True,
        )

    def reserve_background_ai_credits(
        self,
        usage_tag: str,
        estimated_credit_units: int,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Reserve maintenance work without counting another user message."""

        return self._reserve_ai_credits(
            usage_tag,
            estimated_credit_units,
            metadata=metadata,
            enforce_message_cap=False,
        )

    def _reserve_ai_credits(
        self,
        usage_tag: str,
        estimated_credit_units: int,
        *,
        metadata: Optional[Mapping[str, Any]],
        enforce_message_cap: bool,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

        chat_scope_id, context_error = self._resolve_ai_charge_context()
        if context_error:
            return None, context_error
        if self.user_id is None:
            return None, self.billing_missing_scope_message

        persisted_reservation = self.load_persisted_reservation_fn(usage_tag)
        if persisted_reservation:
            raw_metadata = persisted_reservation.get("metadata")
            persisted_metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            persisted_scale = persisted_reservation.get(
                "credit_scale",
                persisted_metadata.get("credit_scale"),
            )
            return {
                "reserved_credit_units": rescale_credit_units(
                    persisted_reservation.get("reserved_credit_units", 0),
                    persisted_scale,
                ),
                "chat_scope_id": persisted_reservation.get("chat_scope_id", chat_scope_id),
                "source": str(persisted_reservation.get("source") or "user"),
                "usage_tag": str(persisted_reservation.get("usage_tag") or usage_tag),
                "metadata": persisted_metadata,
                "credit_scale": CREDIT_SCALE,
            }, None

        self._ensure_onboarding_checked()

        reserve_amount = max(0, int(estimated_credit_units or 0))
        reserve_metadata = self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={"reserved_credit_units": reserve_amount, **dict(metadata or {})},
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
                    "estimated_credit_units": reserve_amount,
                },
            )
            return None, self.billing_charge_error_message

        if not charge_result.get("ok"):
            return None, self._build_insufficient_credits_reply(charge_result)

        source = str(charge_result.get("source") or "user")
        reservation_payload = {
            "reserved_credit_units": reserve_amount,
            "chat_scope_id": chat_scope_id,
            "source": source,
            "usage_tag": usage_tag,
            "metadata": reserve_metadata,
            "credit_scale": CREDIT_SCALE,
        }

        if source == "chat" and enforce_message_cap:
            cap_error = self._check_creditless_cap(
                chat_scope_id=chat_scope_id,
                reserve_amount=reserve_amount,
                usage_tag=usage_tag,
            )
            if cap_error:
                return None, cap_error

        self.persist_reservation_fn(usage_tag, reservation_payload)
        return reservation_payload, None

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
        reserved_credit_units_total: int,
        settled_credit_units: int,
        refunded_credit_units: int,
        extra_charged_credit_units: int,
        debt_applied_credit_units: int,
        payer_breakdown: Sequence[Mapping[str, Any]],
        reason: str,
        breakdown: Mapping[str, Any],
        billing_segments: Sequence[Mapping[str, Any]],
        missing_usage_billing: bool,
        billing_zero_usage_fallback: bool,
    ) -> Dict[str, Any]:
        normalized_payers = [
            {
                "scope": "chat" if item.get("scope") == "chat" else "user",
                "credit_units": max(0, int(item.get("credit_units") or 0)),
            }
            for item in payer_breakdown
            if int(item.get("credit_units") or 0) > 0
        ]
        payer_scopes = {str(item["scope"]) for item in normalized_payers}
        payer_scope = (
            next(iter(payer_scopes))
            if len(payer_scopes) == 1
            else "mixed"
            if payer_scopes
            else "user"
        )
        settlement_ids = [self._settlement_id(tag) for tag in usage_tags]
        return self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={
                "reason": reason,
                "message_id": self.message.get("message_id"),
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "command": self.command,
                "usage_tags": list(usage_tags),
                "reserved_credit_units_total": reserved_credit_units_total,
                "settled_credit_units": settled_credit_units,
                "refunded_credit_units": refunded_credit_units,
                "extra_charged_credit_units": extra_charged_credit_units,
                "debt_applied_credit_units": debt_applied_credit_units,
                "charged_credit_units_total": (
                    reserved_credit_units_total
                    - refunded_credit_units
                    + extra_charged_credit_units
                    + debt_applied_credit_units
                ),
                "payer_scope": payer_scope,
                "payer_breakdown": normalized_payers,
                "settlement_ids": settlement_ids,
                "pricing_version": breakdown.get("pricing_version"),
                "raw_usd_micros": breakdown.get("raw_usd_micros", 0),
                "raw_usd_micros_exact": breakdown.get("raw_usd_micros_exact", "0"),
                "markup_multiplier": breakdown.get("markup_multiplier"),
                "model_breakdown": breakdown.get("model_breakdown", []),
                "tool_breakdown": breakdown.get("tool_breakdown", []),
                "segment_breakdown": breakdown.get("segment_breakdown", []),
                "pricing_complete": breakdown.get("pricing_complete", False),
                "unsupported_notes": breakdown.get("unsupported_notes", []),
                "billing_segments": list(billing_segments or []),
                "missing_usage_billing": bool(missing_usage_billing),
                "billing_zero_usage_fallback": bool(billing_zero_usage_fallback),
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
        for attempt in range(3):
            try:
                self.credits_db_service.record_ai_settlement_result(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    actor_user_id=self.user_id,
                    metadata=settlement_metadata,
                )
                return
            except Exception as error:
                if attempt == 2:
                    self.admin_reporter(
                        "falló registrar resultado de liquidación IA",
                        error,
                        {
                            "chat_id": self.chat_id,
                            "user_id": self.user_id,
                            "command": self.command,
                            "settlement_id": settlement_metadata.get("settlement_id"),
                        },
                    )

    @staticmethod
    def _payer_breakdown_for_settlement(
        *,
        source: str,
        reserved_credit_units: int,
        refunded_credit_units: int,
        debt_applied_credit_units: int,
        extra_charged_credit_units: int,
        extra_payer_scope: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        breakdown = [
            {
                "scope": source,
                "credit_units": (
                    reserved_credit_units - refunded_credit_units + debt_applied_credit_units
                ),
            }
        ]
        if extra_charged_credit_units > 0:
            breakdown.append(
                {
                    "scope": extra_payer_scope or source,
                    "credit_units": extra_charged_credit_units,
                }
            )
        return breakdown

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

        reserved_credit_units = int(reservation_meta.get("reserved_credit_units", 0) or 0)
        usage_tag = str(reservation_meta.get("usage_tag") or "ai_usage")
        usage_tags = [usage_tag]
        source = "chat" if str(reservation_meta.get("source") or "user") == "chat" else "user"
        if billing_segments is None:
            # Missing usage is ambiguous; keep the reserve rather than make the call free.
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
                reserved_credit_units_total=reserved_credit_units,
                settled_credit_units=reserved_credit_units,
                refunded_credit_units=0,
                extra_charged_credit_units=0,
                debt_applied_credit_units=0,
                payer_breakdown=[{"scope": source, "credit_units": reserved_credit_units}],
                reason=reason,
                breakdown=breakdown,
                billing_segments=list(billing_segments or []),
                missing_usage_billing=True,
                billing_zero_usage_fallback=False,
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
                    "reserved_credit_units": reserved_credit_units,
                },
            )
            self.clear_persisted_reservation_fn(usage_tag)
            return
        breakdown = calculate_billing_for_segments(billing_segments or [])
        actual_credit_units = _billing_summary_int(breakdown, "charged_credit_units")
        raw_usd_micros = _billing_summary_int(breakdown, "raw_usd_micros")
        refunded_credit_units = 0
        extra_charged_credit_units = 0
        debt_applied_credit_units = 0
        extra_payer_scope: Optional[str] = None
        chat_scope_id = reservation_meta.get("chat_scope_id")
        if not _billing_is_complete(breakdown):
            # Missing provider evidence is ambiguous; zero must not imply a free call.
            actual_credit_units = max(actual_credit_units, reserved_credit_units)
            self.admin_reporter(
                "liquidación IA sin costo de proveedor verificable; se mantiene la reserva",
                None,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "reserved_credit_units": reserved_credit_units,
                    "unsupported_notes": breakdown.get("unsupported_notes", []),
                    "billing_segments": list(billing_segments or []),
                },
            )
        if actual_credit_units < reserved_credit_units:
            refunded_credit_units = reserved_credit_units - actual_credit_units
            try:
                self.credits_db_service.refund_ai_charge(
                    user_id=self.user_id,
                    chat_id=chat_scope_id,
                    amount=refunded_credit_units,
                    source=source,
                    event_type="ai_refund",
                    metadata=self._build_charge_metadata(
                        usage_tag=usage_tag,
                        extra={
                            "reason": reason,
                            "reserved_credit_units_total": reserved_credit_units,
                            "settled_credit_units": actual_credit_units,
                            "refunded_credit_units": refunded_credit_units,
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
                        "reserved_credit_units": reserved_credit_units,
                        "actual_credit_units": actual_credit_units,
                        "reason": reason,
                    },
                )
                refunded_credit_units = 0

        elif actual_credit_units > reserved_credit_units:
            extra_amount = actual_credit_units - reserved_credit_units
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
                            "reserved_credit_units_total": reserved_credit_units,
                            "settled_credit_units": actual_credit_units,
                            "extra_charged_credit_units": extra_amount,
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
                        "reserved_credit_units": reserved_credit_units,
                        "actual_credit_units": actual_credit_units,
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
                        "reserved_credit_units": reserved_credit_units,
                        "actual_credit_units": actual_credit_units,
                        "reason": reason,
                        "billing_segments": list(billing_segments or []),
                    },
                )
                try:
                    # Record unpaid overage so settlement remains auditable.
                    self.credits_db_service.apply_ai_debt(
                        user_id=self.user_id,
                        chat_id=chat_scope_id,
                        source=source,
                        amount=extra_amount,
                        event_type="ai_settlement_debt",
                        metadata=self._build_charge_metadata(
                            usage_tag=usage_tag,
                            extra={
                                "reason": reason,
                                "reserved_credit_units_total": reserved_credit_units,
                                "settled_credit_units": actual_credit_units,
                                "debt_applied_credit_units": extra_amount,
                            },
                        ),
                    )
                    debt_applied_credit_units = extra_amount
                except Exception as error:
                    self.admin_reporter(
                        "falló registrar deuda de liquidación IA",
                        error,
                        {
                            "chat_id": self.chat_id,
                            "user_id": self.user_id,
                            "reserved_credit_units": reserved_credit_units,
                            "actual_credit_units": actual_credit_units,
                            "reason": reason,
                        },
                    )
            else:
                extra_charged_credit_units = extra_amount
                extra_payer_scope = "chat" if extra_charge.get("source") == "chat" else "user"

        settlement_metadata = self._build_settlement_metadata(
            usage_tag=usage_tag,
            usage_tags=usage_tags,
            reserved_credit_units_total=reserved_credit_units,
            settled_credit_units=actual_credit_units,
            refunded_credit_units=refunded_credit_units,
            extra_charged_credit_units=extra_charged_credit_units,
            debt_applied_credit_units=debt_applied_credit_units,
            payer_breakdown=self._payer_breakdown_for_settlement(
                source=source,
                reserved_credit_units=reserved_credit_units,
                refunded_credit_units=refunded_credit_units,
                debt_applied_credit_units=debt_applied_credit_units,
                extra_charged_credit_units=extra_charged_credit_units,
                extra_payer_scope=extra_payer_scope,
            ),
            reason=reason,
            breakdown=breakdown,
            billing_segments=list(billing_segments or []),
            missing_usage_billing=False,
            billing_zero_usage_fallback=raw_usd_micros == 0,
        )
        self._record_ai_settlement_result(
            chat_scope_id=chat_scope_id,
            settlement_metadata=settlement_metadata,
        )
        self.clear_persisted_reservation_fn(usage_tag)

    @staticmethod
    def _build_batch_reservation(
        reservations: list[dict[str, Any]],
    ) -> BatchReservation:
        usage_tags = [str(item.get("usage_tag") or "ai_usage") for item in reservations]
        return BatchReservation(
            items=reservations,
            reserved_credit_units=sum(
                int(item.get("reserved_credit_units", 0) or 0) for item in reservations
            ),
            usage_tags=usage_tags,
            usage_tag=(usage_tags[0] if len(set(usage_tags)) == 1 else "ai_usage_batch"),
            chat_scope_id=reservations[0].get("chat_scope_id"),
            source=("chat" if str(reservations[0].get("source") or "user") == "chat" else "user"),
        )

    @staticmethod
    def _batch_has_mixed_accounts(reservations: Sequence[Mapping[str, Any]]) -> bool:
        sources = {str(item.get("source") or "user") for item in reservations}
        scopes = {item.get("chat_scope_id") for item in reservations}
        return len(sources) > 1 or len(scopes) > 1

    def _settle_reservations_individually(
        self,
        reservations: Sequence[Mapping[str, Any]],
        billing_segments: Optional[List[Mapping[str, Any]]],
        *,
        reason: str,
    ) -> None:
        print(
            "settle_batch: mixed credit accounts, falling back to individual "
            f"settlement (count={len(reservations)})"
        )
        for index, reservation in enumerate(reservations):
            self._settle_single_reservation(
                reservation,
                billing_segments if index == 0 else [],
                reason=reason,
            )

    def _clear_batch_reservations(self, batch: BatchReservation) -> None:
        for usage_tag in batch.usage_tags:
            self.clear_persisted_reservation_fn(usage_tag)

    @staticmethod
    def _missing_usage_breakdown() -> dict[str, Any]:
        return {
            "pricing_version": None,
            "markup_multiplier": None,
            "raw_usd_micros": 0,
            "model_breakdown": [],
            "tool_breakdown": [],
            "unsupported_notes": ["missing_billing_segments_reserve_retained"],
        }

    def _record_batch_settlement(
        self,
        batch: BatchReservation,
        adjustment: SettlementAdjustment,
        record: BatchSettlementRecord,
    ) -> None:
        metadata = self._build_settlement_metadata(
            usage_tag=batch.usage_tag,
            usage_tags=batch.usage_tags,
            reserved_credit_units_total=batch.reserved_credit_units,
            settled_credit_units=adjustment.settled_credit_units,
            refunded_credit_units=adjustment.refunded_credit_units,
            extra_charged_credit_units=adjustment.extra_charged_credit_units,
            debt_applied_credit_units=adjustment.debt_applied_credit_units,
            payer_breakdown=self._payer_breakdown_for_settlement(
                source=batch.source,
                reserved_credit_units=batch.reserved_credit_units,
                refunded_credit_units=adjustment.refunded_credit_units,
                debt_applied_credit_units=adjustment.debt_applied_credit_units,
                extra_charged_credit_units=adjustment.extra_charged_credit_units,
                extra_payer_scope=adjustment.extra_payer_scope,
            ),
            reason=record.reason,
            breakdown=record.breakdown,
            billing_segments=record.billing_segments,
            missing_usage_billing=record.missing_usage_billing,
            billing_zero_usage_fallback=(
                not record.missing_usage_billing
                and _billing_summary_int(record.breakdown, "raw_usd_micros") == 0
            ),
        )
        self._record_ai_settlement_result(
            chat_scope_id=batch.chat_scope_id,
            settlement_metadata=metadata,
        )
        self._clear_batch_reservations(batch)

    def _retain_batch_reserve(
        self,
        batch: BatchReservation,
        *,
        reason: str,
    ) -> None:
        self._record_batch_settlement(
            batch,
            SettlementAdjustment(batch.reserved_credit_units),
            BatchSettlementRecord(
                reason=reason,
                breakdown=self._missing_usage_breakdown(),
                billing_segments=[],
                missing_usage_billing=True,
            ),
        )
        self.admin_reporter(
            "respuesta IA exitosa sin usage billing; se mantiene cobro por reserva (sin reintegro)",
            None,
            {
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "reason": reason,
                "reserved_credit_units": batch.reserved_credit_units,
            },
        )

    def _batch_adjustment_metadata(
        self,
        batch: BatchReservation,
        *,
        reason: str,
        settled_credit_units: int,
        amount_key: str,
        amount: int,
    ) -> dict[str, Any]:
        return self._build_charge_metadata(
            usage_tag=batch.usage_tag,
            extra={
                "reason": reason,
                "reserved_credit_units_total": batch.reserved_credit_units,
                "settled_credit_units": settled_credit_units,
                amount_key: amount,
                "usage_tags": list(batch.usage_tags),
            },
        )

    def _refund_batch_overreserve(
        self,
        batch: BatchReservation,
        *,
        actual_credit_units: int,
        reason: str,
    ) -> int:
        refund = batch.reserved_credit_units - actual_credit_units
        try:
            self.credits_db_service.refund_ai_charge(
                user_id=self.user_id,
                chat_id=batch.chat_scope_id,
                amount=refund,
                source=batch.source,
                event_type="ai_refund",
                metadata=self._batch_adjustment_metadata(
                    batch,
                    reason=reason,
                    settled_credit_units=actual_credit_units,
                    amount_key="refunded_credit_units",
                    amount=refund,
                ),
            )
        except Exception as error:
            self.admin_reporter(
                "falló el reintegro batch de liquidación IA",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reserved_credit_units": batch.reserved_credit_units,
                    "actual_credit_units": actual_credit_units,
                    "reason": reason,
                },
            )
            return 0
        return refund

    def _apply_batch_debt(
        self,
        batch: BatchReservation,
        *,
        actual_credit_units: int,
        extra_amount: int,
        reason: str,
    ) -> int:
        try:
            self.credits_db_service.apply_ai_debt(
                user_id=self.user_id,
                chat_id=batch.chat_scope_id,
                amount=extra_amount,
                source=batch.source,
                event_type="ai_settlement_debt",
                metadata=self._batch_adjustment_metadata(
                    batch,
                    reason=reason,
                    settled_credit_units=actual_credit_units,
                    amount_key="debt_applied_credit_units",
                    amount=extra_amount,
                ),
            )
        except Exception as error:
            self.admin_reporter(
                "falló registrar deuda batch de liquidación IA",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reserved_credit_units": batch.reserved_credit_units,
                    "actual_credit_units": actual_credit_units,
                    "reason": reason,
                },
            )
            return 0
        return extra_amount

    def _charge_batch_overage(
        self,
        batch: BatchReservation,
        billing_segments: Sequence[Mapping[str, Any]],
        *,
        actual_credit_units: int,
        reason: str,
    ) -> tuple[int, int, Optional[str]]:
        extra_amount = actual_credit_units - batch.reserved_credit_units
        try:
            charge = self.credits_db_service.charge_ai_credits(
                user_id=self.user_id,
                chat_id=batch.chat_scope_id,
                amount=extra_amount,
                event_type="ai_settlement_charge",
                metadata=self._batch_adjustment_metadata(
                    batch,
                    reason=reason,
                    settled_credit_units=actual_credit_units,
                    amount_key="extra_charged_credit_units",
                    amount=extra_amount,
                ),
            )
        except Exception as error:
            self.admin_reporter(
                "falló el ajuste batch de liquidación IA",
                error,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reserved_credit_units": batch.reserved_credit_units,
                    "actual_credit_units": actual_credit_units,
                    "reason": reason,
                },
            )
            charge = {"ok": False}

        if charge.get("ok"):
            payer_scope = "chat" if charge.get("source") == "chat" else "user"
            return extra_amount, 0, payer_scope
        self.admin_reporter(
            "la liquidación IA batch superó la reserva y no pudo cobrar ajuste",
            None,
            {
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "reserved_credit_units": batch.reserved_credit_units,
                "actual_credit_units": actual_credit_units,
                "reason": reason,
                "billing_segments": list(billing_segments),
            },
        )
        debt = self._apply_batch_debt(
            batch,
            actual_credit_units=actual_credit_units,
            extra_amount=extra_amount,
            reason=reason,
        )
        return 0, debt, None

    def _calculate_batch_adjustment(
        self,
        batch: BatchReservation,
        billing_segments: Sequence[Mapping[str, Any]],
        *,
        reason: str,
    ) -> tuple[Mapping[str, Any], SettlementAdjustment]:
        breakdown = calculate_billing_for_segments(billing_segments)
        actual = _billing_summary_int(breakdown, "charged_credit_units")
        if not _billing_is_complete(breakdown):
            actual = max(actual, batch.reserved_credit_units)
            self.admin_reporter(
                "liquidación IA batch sin costo de proveedor verificable; se mantiene la reserva",
                None,
                {
                    "chat_id": self.chat_id,
                    "user_id": self.user_id,
                    "reason": reason,
                    "reserved_credit_units": batch.reserved_credit_units,
                    "unsupported_notes": breakdown.get("unsupported_notes", []),
                    "billing_segments": list(billing_segments),
                },
            )

        if actual < batch.reserved_credit_units:
            refund = self._refund_batch_overreserve(
                batch,
                actual_credit_units=actual,
                reason=reason,
            )
            return breakdown, SettlementAdjustment(actual, refunded_credit_units=refund)
        if actual > batch.reserved_credit_units:
            extra, debt, extra_payer_scope = self._charge_batch_overage(
                batch,
                billing_segments,
                actual_credit_units=actual,
                reason=reason,
            )
            return breakdown, SettlementAdjustment(
                actual,
                extra_charged_credit_units=extra,
                debt_applied_credit_units=debt,
                extra_payer_scope=extra_payer_scope,
            )
        return breakdown, SettlementAdjustment(actual)

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

        if self._batch_has_mixed_accounts(reservations):
            self._settle_reservations_individually(
                reservations,
                billing_segments,
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

        batch = self._build_batch_reservation(reservations)
        if billing_segments is None:
            self._retain_batch_reserve(batch, reason=reason)
            return

        breakdown, adjustment = self._calculate_batch_adjustment(
            batch,
            billing_segments,
            reason=reason,
        )
        self._record_batch_settlement(
            batch,
            adjustment,
            BatchSettlementRecord(
                reason=reason,
                breakdown=breakdown,
                billing_segments=billing_segments,
            ),
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

        reserved_credit_units = int(reservation_meta.get("reserved_credit_units", 0) or 0)
        source = "chat" if str(reservation_meta.get("source") or "user") == "chat" else "user"
        usage_tag = str(reservation_meta.get("usage_tag") or "ai_usage")
        refund_metadata = self._build_charge_metadata(
            usage_tag=usage_tag,
            extra={
                "reason": reason,
                "reserved_credit_units": reserved_credit_units,
                "settled_credit_units": 0,
                "refunded_credit_units": reserved_credit_units,
                **dict(metadata or {}),
            },
        )

        try:
            self.credits_db_service.refund_ai_charge(
                user_id=self.user_id,
                chat_id=reservation_meta.get("chat_scope_id"),
                amount=reserved_credit_units,
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
            return

        self._rollback_creditless_cap(reservation_meta)
        self.clear_persisted_reservation_fn(usage_tag)

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
                "reserved_credit_units": int(
                    charge_meta.get(
                        "credit_units_cost",
                        charge_meta.get("reserved_credit_units", 10),
                    )
                ),
            },
            reason=reason,
        )

    @staticmethod
    def is_transcribe_success_response(text: Optional[str]) -> bool:
        """Return True when a media transcription/description completed successfully."""

        if not text:
            return False
        success_prefixes = (
            tr("media.audio_result", text=""),
            tr("media.image_result"),
            tr("media.sticker_result"),
        )
        return text.startswith(success_prefixes)


__all__ = [
    "AIMessageBilling",
    "build_insufficient_credits_message",
    "build_topup_keyboard",
    "format_balance_command",
    "get_ai_billing_pack",
    "get_ai_billing_packs",
    "get_ai_onboarding_credits",
    "maybe_grant_onboarding_credits",
    "parse_topup_payload",
]
