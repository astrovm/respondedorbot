"""Format and deliver operational error reports."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from os import environ
from typing import Any

from api.billing.credit_units import format_credit_units

SendMessage = Callable[[str, str], Any]
Redactor = Callable[[str], str]


def admin_report(
    message: str,
    error: Exception | None,
    extra_context: Mapping[str, Any] | None,
    *,
    send_message: SendMessage,
    redact: Redactor,
) -> None:
    admin_chat_id = environ.get("ADMIN_CHAT_ID")
    instance_name = environ.get("FRIENDLY_INSTANCE_NAME")
    formatted_message = f"reporte admin desde {instance_name}: {message}"

    if extra_context:
        context_details = "\n\ncontexto adicional:"
        for key, value in extra_context.items():
            context_details += f"\n{key}: {_format_context_value(key, value)}"
        formatted_message += context_details

    if error:
        formatted_message += _format_error(error, redact=redact)

    if admin_chat_id:
        send_message(admin_chat_id, formatted_message)


def _format_context_value(key: str, value: Any) -> Any:
    if "credit_units" not in str(key or "").lower() or isinstance(value, bool):
        return value
    try:
        units = int(value)
    except (TypeError, ValueError):
        return value
    return f"{format_credit_units(units)} créditos ({units} unidades)"


def _format_error(error: Exception, *, redact: Redactor) -> str:
    details = f"\n\ntipo de error: {type(error).__name__}"
    details += f"\nmensaje de error: {redact(str(error))}"
    if error.__traceback__ is not None:
        formatted_traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    else:
        formatted_traceback = "traceback unavailable"
    return details + f"\n\ntraceback:\n{redact(formatted_traceback)}"
