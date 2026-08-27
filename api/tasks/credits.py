"""Personal-credit checks shared by task creation paths."""

from __future__ import annotations

from typing import Any

from api.billing.credit_units import format_credit_units
from api.core.i18n import tr


def task_credit_precondition_error(
    *,
    credits_db_service: Any,
    user_id: Any,
    required_credit_units: int,
) -> str | None:
    """Return an error when the task creator cannot fund execution."""

    if user_id is None:
        return tr("task.credit_user")
    if not credits_db_service.is_configured():
        return tr("billing.unavailable")

    try:
        balance = int(credits_db_service.get_balance("user", int(user_id)))
    except Exception:
        return tr("task.credit_check")

    required = max(1, int(required_credit_units or 0))
    if balance >= required:
        return None

    return tr(
        "task.credit_insufficient",
        balance=format_credit_units(balance),
        required=format_credit_units(required),
    )


__all__ = ["task_credit_precondition_error"]
