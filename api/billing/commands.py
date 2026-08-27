"""Handle user-facing credit balance and transfer commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from api.core.constants import BILLING_UNAVAILABLE_MESSAGE
from api.billing.credit_units import format_credit_units, parse_credit_units

CommandResponse = Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]
_CHARGE_COMMANDS = {"/charges", "/history", "/gastos"}


class BillingCommandDeps(Protocol):
    @property
    def credits_db_service(self) -> Any: ...

    @property
    def balance_formatter(self) -> Any: ...

    @property
    def admin_report(self) -> Any: ...

    @property
    def maybe_grant_onboarding_credits(self) -> Any: ...

    def is_group_chat_type(self, chat_type: Optional[str]) -> bool: ...


def _require_billing(deps: BillingCommandDeps, command: str) -> Optional[CommandResponse]:
    if bool(deps.credits_db_service.is_configured()):
        return None
    return BILLING_UNAVAILABLE_MESSAGE, None, False, command


def handle_balance_command(
    deps: BillingCommandDeps,
    *,
    command: str,
    chat_type: str,
    chat_id: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
) -> CommandResponse:
    if command != "/balance":
        return None, None, False, None

    billing_required_response = _require_billing(deps, command)
    if billing_required_response is not None:
        return billing_required_response

    if user_id is None or numeric_chat_id is None:
        return (
            "no te pude leer bien el usuario para ver los saldos",
            None,
            False,
            command,
        )

    try:
        deps.maybe_grant_onboarding_credits(
            deps.credits_db_service, deps.admin_report, user_id
        )
        response_msg = deps.balance_formatter.format(
            chat_type=chat_type,
            user_id=user_id,
            chat_id=numeric_chat_id,
        )
    except Exception as error:
        deps.admin_report(
            "Error loading balance",
            error,
            {"chat_id": chat_id, "user_id": user_id},
        )
        response_msg = "se trabó leyendo tu saldo, probá de nuevo"
    return response_msg, None, False, command


def handle_transfer_command(
    deps: BillingCommandDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    chat_type: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
) -> CommandResponse:
    if command != "/transfer":
        return None, None, False, None

    billing_required_response = _require_billing(deps, command)
    if billing_required_response is not None:
        return billing_required_response

    if not deps.is_group_chat_type(chat_type):
        return "esto es para grupos, capo: /transfer <monto>", None, False, command

    if user_id is None or numeric_chat_id is None:
        return (
            "no te pude sacar bien el usuario o el grupo para transferir",
            None,
            False,
            command,
        )

    amount_token = sanitized_message_text.split(" ", 1)[0].strip()
    amount = parse_credit_units(amount_token)
    if amount is None:
        return "mandalo bien: /transfer <monto>", None, False, command

    if amount <= 0:
        return (
            "el monto tiene que ser mayor a 0, no me rompas las bolas",
            None,
            False,
            command,
        )

    try:
        transfer_result = deps.credits_db_service.transfer_user_to_chat(
            user_id=user_id,
            chat_id=numeric_chat_id,
            amount=amount,
        )
    except Exception as error:
        deps.admin_report(
            "Error transferring credits",
            error,
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "amount": amount,
            },
        )
        return "se trabó la transferencia, probá de nuevo", None, False, command

    if transfer_result.get("ok"):
        response_msg = (
            f"listo, le pasé {format_credit_units(amount)} créditos al grupo\n"
            f"- lo tuyo: {format_credit_units(transfer_result.get('user_balance', 0))}\n"
            f"- lo del grupo: {format_credit_units(transfer_result.get('chat_balance', 0))}"
        )
        return response_msg, None, False, command

    response_msg = (
        "no te alcanza lo tuyo para pasar esa guita al grupo\n"
        f"te quedan: {format_credit_units(transfer_result.get('user_balance', 0))}"
    )
    return response_msg, None, False, command


def _metadata_credit(metadata: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in metadata:
            return int(metadata.get(key) or 0)
    return 0


def _charged_credit_units(metadata: Mapping[str, Any], event_type: str = "") -> int:
    if "charged_credit_units_total" in metadata:
        return max(0, int(metadata.get("charged_credit_units_total") or 0))
    if event_type == "memory_compaction_settlement":
        return max(0, int(metadata.get("actual_credit_units") or 0))
    if event_type == "ai_reserve":
        return max(
            0,
            _metadata_credit(
                metadata,
                "reserved_credit_units",
                "reserved_credits",
            ),
        )
    reserved = _metadata_credit(
        metadata,
        "reserved_credit_units_total",
        "reserved_credit_units",
        "reserved_credits_total",
        "reserved_credits",
    )
    if reserved:
        return max(
            0,
            reserved
            - _metadata_credit(metadata, "refunded_credit_units", "refunded_credits")
            + _metadata_credit(
                metadata,
                "extra_charged_credit_units",
                "extra_charged_credits",
            )
            + _metadata_credit(
                metadata,
                "debt_applied_credit_units",
                "debt_applied_credits",
            ),
        )
    return max(
        0,
        _metadata_credit(metadata, "settled_credit_units", "settled_credits"),
    )


def _model_component_label(item: Mapping[str, Any]) -> str:
    kind = str(item.get("kind") or "").lower()
    if kind == "transcribe" or float(item.get("audio_seconds") or 0) > 0:
        return "transcripción"
    if kind == "vision":
        return "análisis de imagen"
    return "respuesta IA"


def _tool_component_label(item: Mapping[str, Any]) -> str:
    tool = str(item.get("tool") or "").lower()
    count = max(0, int(item.get("count") or 0))
    if tool == "web_search":
        return f"búsqueda web ({count}x)" if count > 1 else "búsqueda web"
    return "herramienta IA"


def _raw_charge_components(metadata: Mapping[str, Any]) -> List[Tuple[str, int]]:
    totals: Dict[str, int] = {}
    for raw_item in metadata.get("model_breakdown") or []:
        if not isinstance(raw_item, Mapping):
            continue
        label = _model_component_label(raw_item)
        totals[label] = totals.get(label, 0) + max(
            0, int(raw_item.get("usd_micros") or 0)
        )
    for raw_item in metadata.get("tool_breakdown") or []:
        if not isinstance(raw_item, Mapping):
            continue
        label = _tool_component_label(raw_item)
        totals[label] = totals.get(label, 0) + max(
            0, int(raw_item.get("usd_micros") or 0)
        )
    return [(label, amount) for label, amount in totals.items() if amount > 0]


def allocate_charge_components(
    charged_credit_units: int,
    components: Sequence[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Allocate rounded charge units while preserving the exact total."""

    total_units = max(0, int(charged_credit_units or 0))
    normalized = [(label, max(0, int(raw))) for label, raw in components if raw > 0]
    raw_total = sum(raw for _label, raw in normalized)
    if total_units <= 0 or raw_total <= 0:
        return []

    allocated: List[List[Any]] = []
    for index, (label, raw) in enumerate(normalized):
        numerator = total_units * raw
        allocated.append([label, numerator // raw_total, numerator % raw_total, index])

    leftover = total_units - sum(int(item[1]) for item in allocated)
    for item in sorted(allocated, key=lambda value: (-int(value[2]), int(value[3])))[
        :leftover
    ]:
        item[1] = int(item[1]) + 1
    return [(str(item[0]), int(item[1])) for item in allocated]


def _charge_activity(metadata: Mapping[str, Any], event_type: str = "") -> str:
    command = str(metadata.get("command") or "").strip()
    if command:
        return command
    usage_tag = str(metadata.get("usage_tag") or "")
    if "transcribe" in usage_tag or "audio" in usage_tag:
        return "audio"
    if "image" in usage_tag or "vision" in usage_tag:
        return "imagen"
    if event_type == "memory_compaction_settlement" or "memory_compaction" in usage_tag:
        return "memoria"
    return "respuesta IA"


def _payer_label(metadata: Mapping[str, Any]) -> str:
    payer_totals: Dict[str, int] = {}
    for raw_payer in metadata.get("payer_breakdown") or []:
        if not isinstance(raw_payer, Mapping):
            continue
        scope = "chat" if raw_payer.get("scope") == "chat" else "user"
        payer_totals[scope] = payer_totals.get(scope, 0) + max(
            0, int(raw_payer.get("credit_units") or 0)
        )
    visible_payers = [
        (scope, units) for scope, units in payer_totals.items() if units > 0
    ]
    if len(visible_payers) > 1:
        labels = {"user": "saldo personal", "chat": "saldo del grupo"}
        return " + ".join(
            f"{labels[scope]} {format_credit_units(units)}"
            for scope, units in visible_payers
        )

    payer_scope = str(metadata.get("payer_scope") or metadata.get("source") or "")
    return {
        "user": "saldo personal",
        "chat": "saldo del grupo",
    }.get(payer_scope, "saldo no especificado")


def _charge_time_label(value: Any, timezone_offset: float) -> str:
    parsed: Optional[datetime] = value if isinstance(value, datetime) else None
    if parsed is None and value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None:
        return "sin fecha"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local_time = parsed.astimezone(timezone(timedelta(hours=float(timezone_offset))))
    return local_time.strftime("%d/%m %H:%M")


def format_user_charge_history(
    entries: Sequence[Mapping[str, Any]],
    *,
    timezone_offset: float = 0,
    has_more: bool = False,
) -> str:
    lines = ["últimos gastos IA:"]
    shown_total = 0
    for entry in entries:
        raw_metadata = entry.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        event_type = str(entry.get("event_type") or "")
        charged_units = _charged_credit_units(metadata, event_type)
        shown_total += charged_units
        lines.extend(
            [
                "",
                (
                    f"{_charge_time_label(entry.get('created_at'), timezone_offset)}"
                    f" · {_charge_activity(metadata, event_type)}"
                    f" · {format_credit_units(charged_units)} créditos"
                ),
            ]
        )
        allocation = allocate_charge_components(
            charged_units,
            _raw_charge_components(metadata),
        )
        if allocation:
            lines.extend(
                f"- {label}: {format_credit_units(units)}"
                for label, units in allocation
            )
        elif metadata.get("missing_usage_billing") or metadata.get(
            "billing_zero_usage_fallback"
        ):
            lines.append("- cargo estimado por falta de detalle del proveedor")
        if event_type == "ai_reserve":
            lines.append("- liquidación pendiente; se muestra la reserva cobrada")
        lines.append(f"- pagó: {_payer_label(metadata)}")

    lines.extend(["", f"total mostrado: {format_credit_units(shown_total)} créditos"])
    if has_more and entries:
        lines.append(f"más: /charges 10 {int(entries[-1]['id'])}")
    return "\n".join(lines)


def handle_charges_command(
    deps: BillingCommandDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
    timezone_offset: float,
) -> CommandResponse:
    if command not in _CHARGE_COMMANDS:
        return None, None, False, None

    billing_required_response = _require_billing(deps, command)
    if billing_required_response is not None:
        return billing_required_response
    if user_id is None:
        return "no te pude leer el usuario para ver tus gastos", None, False, command

    tokens = str(sanitized_message_text or "").split()
    if len(tokens) > 2:
        return "mandalo bien: /charges [limite]", None, False, command
    try:
        limit = int(tokens[0]) if tokens else 10
        before_id = int(tokens[1]) if len(tokens) == 2 else None
    except (TypeError, ValueError):
        return "mandalo bien: /charges [limite]", None, False, command
    if limit <= 0 or (before_id is not None and before_id <= 0):
        return "mandalo bien: /charges [limite]", None, False, command
    limit = min(limit, 20)

    try:
        results = deps.credits_db_service.list_user_ai_charges(
            user_id,
            limit=limit + 1,
            before_id=before_id,
        )
    except Exception as error:
        deps.admin_report(
            "Error loading /charges",
            error,
            {"chat_id": chat_id, "user_id": user_id, "limit": limit},
        )
        return "se trabó leyendo tus gastos, probá de nuevo", None, False, command

    entries = list(results[:limit])
    if not entries:
        return "no tenés gastos IA recientes", None, False, command
    return (
        format_user_charge_history(
            entries,
            timezone_offset=timezone_offset,
            has_more=len(results) > limit,
        ),
        None,
        False,
        command,
    )


__all__ = [
    "allocate_charge_components",
    "format_user_charge_history",
    "handle_balance_command",
    "handle_charges_command",
    "handle_transfer_command",
]
