from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

from api.constants import BILLING_UNAVAILABLE_MESSAGE
from api.credit_units import format_credit_units, parse_credit_units

CommandResponse = Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]


class BillingCommandDeps(Protocol):
    credits_db_service: Any
    balance_formatter: Any
    admin_report: Any
    maybe_grant_onboarding_credits: Any

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


__all__ = ["handle_balance_command", "handle_transfer_command"]
