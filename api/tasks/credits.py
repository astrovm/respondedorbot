"""Personal-credit checks shared by task creation paths."""

from __future__ import annotations

from typing import Any

from api.billing.credit_units import format_credit_units
from api.core.constants import BILLING_UNAVAILABLE_MESSAGE


def task_credit_precondition_error(
    *,
    credits_db_service: Any,
    user_id: Any,
    required_credit_units: int,
) -> str | None:
    """Return an error when the task creator cannot fund execution."""

    if user_id is None:
        return "no pude identificar tu usuario para cobrar la tarea"
    if not credits_db_service.is_configured():
        return BILLING_UNAVAILABLE_MESSAGE

    try:
        balance = int(credits_db_service.get_balance("user", int(user_id)))
    except Exception:
        return "no pude verificar tus créditos personales, probá de nuevo"

    required = max(1, int(required_credit_units or 0))
    if balance >= required:
        return None

    return (
        "no tenés créditos personales suficientes para ejecutar esa tarea\n"
        f"- tenés: {format_credit_units(balance)}\n"
        f"- necesitás: {format_credit_units(required)}\n"
        "cargá con /topup antes de crearla"
    )


__all__ = ["task_credit_precondition_error"]
