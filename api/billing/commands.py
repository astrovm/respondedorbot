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
        return "audio"
    if kind == "vision":
        return "imagen"
    return "respuesta"


def _tool_component_label(item: Mapping[str, Any]) -> str:
    tool = str(item.get("tool") or "").lower()
    count = max(0, int(item.get("count") or 0))
    if tool == "web_search":
        return f"web ({count}x)" if count > 1 else "web"
    return "herramienta"


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
    usage_tag = str(metadata.get("usage_tag") or "")
    if event_type == "memory_compaction_settlement" or "memory_compaction" in usage_tag:
        return "memoria"
    if "transcribe" in usage_tag or "audio" in usage_tag:
        return "audio"
    if "image" in usage_tag or "vision" in usage_tag:
        return "imagen"
    return "respuesta"


def _payer_totals(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {"user": 0, "chat": 0}
    for entry in entries:
        raw_metadata = entry.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        breakdown = metadata.get("payer_breakdown") or []
        found_breakdown = False
        for raw_payer in breakdown:
            if not isinstance(raw_payer, Mapping):
                continue
            found_breakdown = True
            scope = "chat" if raw_payer.get("scope") == "chat" else "user"
            totals[scope] += max(0, int(raw_payer.get("credit_units") or 0))
        if found_breakdown:
            continue
        scope = str(metadata.get("payer_scope") or metadata.get("source") or "")
        if scope in totals:
            totals[scope] += _charged_credit_units(
                metadata,
                str(entry.get("event_type") or ""),
            )
    return totals


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


def _entry_components(entry: Mapping[str, Any]) -> List[Tuple[str, int, bool]]:
    raw_metadata = entry.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    event_type = str(entry.get("event_type") or "")
    charged_units = _charged_credit_units(metadata, event_type)
    pending = bool(metadata.get("billing_pending") or event_type == "ai_reserve")
    if event_type == "memory_compaction_settlement" or "memory_compaction" in str(
        metadata.get("usage_tag") or ""
    ):
        return [("memoria", charged_units, pending)]
    allocation = allocate_charge_components(
        charged_units,
        _raw_charge_components(metadata),
    )
    if allocation:
        return [
            (label, units, pending)
            for label, units in allocation
            if units > 0
        ]
    return [(_charge_activity(metadata, event_type), charged_units, pending)]


def _component_sort_key(component: Tuple[str, int, bool]) -> Tuple[int, bool]:
    label, _units, pending = component
    if label == "respuesta":
        rank = 0
    elif label in {"audio", "imagen"}:
        rank = 1
    elif label.startswith("web"):
        rank = 2
    elif label == "memoria":
        rank = 4
    else:
        rank = 3
    return rank, pending


def _group_payer_suffix(entries: Sequence[Mapping[str, Any]]) -> str:
    payer_totals = _payer_totals(entries)
    user_paid = payer_totals["user"]
    chat_paid = payer_totals["chat"]
    if chat_paid <= 0:
        return ""
    if user_paid <= 0:
        return " · grupo"
    return (
        f" · grupo {format_credit_units(chat_paid)}"
        f" · personal {format_credit_units(user_paid)}"
    )


def format_user_charge_history(
    groups: Sequence[Mapping[str, Any]],
    *,
    timezone_offset: float = 0,
) -> str:
    lines = ["Gastos IA"]
    for group in groups:
        raw_entries = group.get("entries") or []
        entries = [entry for entry in raw_entries if isinstance(entry, Mapping)]
        components = [
            component
            for entry in entries
            for component in _entry_components(entry)
        ]
        components.sort(key=_component_sort_key)
        charged_units = sum(units for _label, units, _pending in components)
        time_label = _charge_time_label(group.get("created_at"), timezone_offset)
        payer_suffix = _group_payer_suffix(entries)
        lines.append("")
        if len(components) == 1:
            label, units, pending = components[0]
            pending_suffix = " · pendiente" if pending else ""
            lines.append(
                f"{time_label} · {label} · {format_credit_units(units)} cr"
                f"{pending_suffix}{payer_suffix}"
            )
            continue
        lines.append(
            f"{time_label} · {format_credit_units(charged_units)} cr{payer_suffix}"
        )
        for label, units, pending in components:
            pending_suffix = " · pendiente" if pending else ""
            lines.append(
                f"  {label} {format_credit_units(units)} cr{pending_suffix}"
            )
    return "\n".join(lines)


def build_charge_history_keyboard(
    page: Mapping[str, Any],
    *,
    user_id: int,
    limit: int,
    timezone_offset: float,
) -> Optional[Dict[str, Any]]:
    timezone_minutes = round(float(timezone_offset) * 60)
    buttons: List[Dict[str, str]] = []
    if page.get("has_newer") and page.get("newer_cursor") is not None:
        buttons.append(
            {
                "text": "‹ Anterior",
                "callback_data": (
                    f"chg:{int(user_id)}:{int(limit)}:n:"
                    f"{int(page['newer_cursor'])}:{timezone_minutes}"
                ),
            }
        )
    if page.get("has_older") and page.get("older_cursor") is not None:
        buttons.append(
            {
                "text": "Siguiente ›",
                "callback_data": (
                    f"chg:{int(user_id)}:{int(limit)}:o:"
                    f"{int(page['older_cursor'])}:{timezone_minutes}"
                ),
            }
        )
    return {"inline_keyboard": [buttons]} if buttons else None


def build_user_charge_history_page(
    credits_db_service: Any,
    *,
    user_id: int,
    limit: int,
    timezone_offset: float,
    cursor_id: Optional[int] = None,
    direction: str = "older",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    page = credits_db_service.list_user_ai_charge_page(
        user_id,
        limit=limit,
        cursor_id=cursor_id,
        direction="newer" if direction == "newer" else "older",
    )
    groups = list(page.get("groups") or [])
    if not groups:
        return "no tenés gastos IA recientes", None
    return (
        format_user_charge_history(groups, timezone_offset=timezone_offset),
        build_charge_history_keyboard(
            page,
            user_id=user_id,
            limit=limit,
            timezone_offset=timezone_offset,
        ),
    )


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
    if len(tokens) > 1:
        return "mandalo bien: /charges [cantidad]", None, False, command
    try:
        limit = int(tokens[0]) if tokens else 10
    except (TypeError, ValueError):
        return "mandalo bien: /charges [cantidad]", None, False, command
    if limit <= 0:
        return "mandalo bien: /charges [cantidad]", None, False, command
    limit = min(limit, 20)

    try:
        response_text, reply_markup = build_user_charge_history_page(
            deps.credits_db_service,
            user_id=user_id,
            limit=limit,
            timezone_offset=timezone_offset,
        )
    except Exception as error:
        deps.admin_report(
            "Error loading /charges",
            error,
            {"chat_id": chat_id, "user_id": user_id, "limit": limit},
        )
        return "se trabó leyendo tus gastos, probá de nuevo", None, False, command

    return response_text, reply_markup, False, command


__all__ = [
    "allocate_charge_components",
    "build_charge_history_keyboard",
    "build_user_charge_history_page",
    "format_user_charge_history",
    "handle_balance_command",
    "handle_charges_command",
    "handle_transfer_command",
]
