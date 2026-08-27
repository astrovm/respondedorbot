from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from api.billing.ai import AIBillingPack
from api.billing.commands import build_user_charge_history_page
from api.core.i18n import tr

ChargeCallbackParams = tuple[int, int, str, int, int, int, int]


def _answer_charge_callback(
    answer_callback: Callable[..., None],
    callback_id: str,
    *,
    text: str | None = None,
    show_alert: bool = False,
) -> None:
    if callback_id:
        if text is None and not show_alert:
            answer_callback(callback_id)
        else:
            answer_callback(callback_id, text=text, show_alert=show_alert)


def send_stars_invoice(
    *,
    chat_id: str,
    user_id: int,
    pack: AIBillingPack,
    format_credits: Callable[[int], str],
    telegram_request: Callable[..., tuple[Any, str | None]],
) -> bool:
    pack_credits = format_credits(pack["credits"])
    response, error = telegram_request(
        "sendInvoice",
        method="POST",
        json_payload={
            "chat_id": chat_id,
            "title": tr("topup.invoice_title", credits=pack_credits),
            "description": tr("topup.invoice_description", credits=pack_credits),
            "payload": f"topup:{pack['id']}:{user_id}",
            "provider_token": "",
            "currency": "XTR",
            "prices": [
                {
                    "label": tr("topup.invoice_label", credits=pack_credits),
                    "amount": pack["xtr"],
                }
            ],
        },
    )
    return error is None and bool(response)


def billing_unavailable_message() -> str:
    return tr("billing.unavailable")


def handle_topup_callback(
    callback_query: dict[str, Any],
    *,
    guard_callback: Callable[..., bool],
    billing_available: Callable[[], bool],
    get_pack: Callable[[str], AIBillingPack | None],
    send_invoice: Callable[..., bool],
    answer_callback: Callable[..., None],
    unavailable_alert: Callable[[], str],
) -> None:
    callback_data = str(callback_query.get("data") or "")
    callback_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    user = callback_query.get("from") or {}
    chat_id = chat.get("id")

    if guard_callback(callback_id, chat_id is None):
        return
    if guard_callback(
        callback_id,
        not billing_available(),
        text=unavailable_alert(),
        show_alert=True,
    ):
        return
    if guard_callback(
        callback_id,
        str(chat.get("type", "")) != "private",
        text=tr("topup.callback_private"),
        show_alert=True,
    ):
        return

    parts = callback_data.split(":", 1)
    pack = get_pack(parts[1] if len(parts) == 2 else "")
    if guard_callback(
        callback_id,
        not pack,
        text=tr("topup.invalid_pack"),
        show_alert=True,
    ):
        return
    try:
        user_id = int(str(user.get("id")))
    except TypeError, ValueError:
        guard_callback(callback_id, True)
        return
    assert pack is not None
    sent = send_invoice(chat_id=str(chat_id), user_id=user_id, pack=pack)
    if callback_id:
        if sent:
            answer_callback(callback_id, text=tr("topup.invoice_ready"))
        else:
            answer_callback(
                callback_id,
                text=tr("topup.invoice_error"),
                show_alert=True,
            )


def _parse_charge_history_callback(
    callback_query: Mapping[str, Any],
) -> ChargeCallbackParams | None:
    callback_data = str(callback_query.get("data") or "")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    try:
        prefix, owner_raw, limit_raw, direction_raw, cursor_raw, timezone_raw = callback_data.split(
            ":"
        )
        owner_id = int(owner_raw)
        limit = int(limit_raw)
        cursor_id = int(cursor_raw)
        timezone_minutes = int(timezone_raw)
        requester_id = int(str((callback_query.get("from") or {}).get("id")))
    except TypeError, ValueError:
        return None
    if (
        prefix != "chg"
        or direction_raw not in {"n", "o"}
        or not 1 <= limit <= 20
        or cursor_id <= 0
        or not -840 <= timezone_minutes <= 840
        or chat_id is None
        or message_id is None
    ):
        return None
    return (
        owner_id,
        limit,
        direction_raw,
        cursor_id,
        timezone_minutes,
        requester_id,
        int(message_id),
    )


def handle_charge_history_callback(
    callback_query: dict[str, Any],
    *,
    credits_db_service: Any,
    edit_message: Callable[..., bool],
    answer_callback: Callable[..., None],
    admin_report: Callable[..., None],
) -> None:
    callback_id = str(callback_query.get("id") or "")
    parsed = _parse_charge_history_callback(callback_query)
    if parsed is None:
        _answer_charge_callback(
            answer_callback,
            callback_id,
            text=tr("callback.expired"),
            show_alert=True,
        )
        return
    (
        owner_id,
        limit,
        direction_raw,
        cursor_id,
        timezone_minutes,
        requester_id,
        message_id,
    ) = parsed
    if requester_id != owner_id:
        _answer_charge_callback(
            answer_callback,
            callback_id,
            text=tr("callback.not_yours"),
            show_alert=True,
        )
        return

    try:
        text, keyboard = build_user_charge_history_page(
            credits_db_service,
            user_id=owner_id,
            limit=limit,
            timezone_offset=timezone_minutes / 60,
            cursor_id=cursor_id,
            direction="newer" if direction_raw == "n" else "older",
        )
        if keyboard is None and text == tr("charges.empty"):
            _answer_charge_callback(
                answer_callback,
                callback_id,
                text=tr("callback.no_more_charges"),
            )
            return
        chat_id = str(((callback_query.get("message") or {}).get("chat") or {})["id"])
        edited = edit_message(
            chat_id,
            message_id,
            text,
            keyboard or {"inline_keyboard": []},
        )
    except Exception as error:
        admin_report(
            "Error paginating /charges",
            error,
            {
                "chat_id": str(((callback_query.get("message") or {}).get("chat") or {}).get("id")),
                "user_id": owner_id,
                "cursor_id": cursor_id,
                "direction": direction_raw,
            },
        )
        _answer_charge_callback(
            answer_callback,
            callback_id,
            text=tr("charges.callback_error"),
            show_alert=True,
        )
        return
    _answer_charge_callback(
        answer_callback,
        callback_id,
        text=None if edited else tr("callback.update_failed"),
        show_alert=not edited,
    )


def handle_pre_checkout_query(
    query: dict[str, Any],
    *,
    billing_available: Callable[[], bool],
    answer_query: Callable[..., None],
    unavailable_alert: Callable[[], str],
    parse_payload: Callable[[str], tuple[str | None, int | None]],
    get_pack: Callable[[str], AIBillingPack | None],
) -> None:
    query_id = query.get("id")
    if not query_id:
        return
    if not billing_available():
        answer_query(
            str(query_id),
            ok=False,
            error_message=unavailable_alert(),
        )
        return

    payload = str(query.get("invoice_payload") or "")
    pack_id, payload_user_id = parse_payload(payload)
    pack = get_pack(pack_id or "")
    try:
        user_id = int(str((query.get("from") or {}).get("id")))
    except TypeError, ValueError:
        answer_query(
            str(query_id),
            ok=False,
            error_message=tr("payment.invalid_user"),
        )
        return
    try:
        total_amount = int(str(query.get("total_amount")))
    except TypeError, ValueError:
        total_amount = -1

    if (
        not pack
        or str(query.get("currency") or "") != "XTR"
        or int(pack["xtr"]) != total_amount
        or (payload_user_id is not None and payload_user_id != user_id)
    ):
        answer_query(
            str(query_id),
            ok=False,
            error_message=tr("payment.invalid"),
        )
        return
    answer_query(str(query_id), ok=True)


def handle_successful_payment(
    message: dict[str, Any],
    *,
    billing_available: Callable[[], bool],
    unavailable_message: Callable[[], str],
    send_message: Callable[[str, str], Any],
    extract_user_id: Callable[[Mapping[str, Any]], int | None],
    parse_payload: Callable[[str], tuple[str | None, int | None]],
    get_pack: Callable[[str], AIBillingPack | None],
    record_payment: Callable[..., dict[str, Any]],
    admin_report: Callable[..., None],
    format_credits: Callable[[int], str],
) -> str:
    chat_id_raw = (message.get("chat") or {}).get("id")
    if chat_id_raw is None:
        return "ok"
    chat_id = str(chat_id_raw)
    if not billing_available():
        send_message(chat_id, unavailable_message())
        return "ok"

    user_id = extract_user_id(message)
    if user_id is None:
        return "ok"
    payment = message.get("successful_payment") or {}
    currency = str(payment.get("currency") or "")
    payload = str(payment.get("invoice_payload") or "")
    charge_id = str(payment.get("telegram_payment_charge_id") or "")
    pack_id, payload_user_id = parse_payload(payload)
    pack = get_pack(pack_id or "")
    try:
        total_amount = int(str(payment.get("total_amount")))
    except TypeError, ValueError:
        total_amount = -1

    if (
        not charge_id
        or not pack
        or currency != "XTR"
        or total_amount != int(pack["xtr"])
        or (payload_user_id is not None and payload_user_id != user_id)
    ):
        send_message(
            chat_id,
            tr("payment.invalid_received"),
        )
        admin_report(
            "Invalid successful payment payload",
            None,
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "currency": currency,
                "payload": payload,
                "total_amount": total_amount,
                "charge_id": charge_id,
            },
        )
        return "ok"

    try:
        result = record_payment(
            telegram_payment_charge_id=charge_id,
            user_id=user_id,
            pack_id=str(pack["id"]),
            xtr_amount=int(pack["xtr"]),
            credits_awarded=int(pack["credits"]),
            payload=payload,
        )
    except Exception as error:
        admin_report(
            "falló persistencia de pago exitoso",
            error,
            {"chat_id": chat_id, "user_id": user_id, "charge_id": charge_id},
        )
        send_message(
            chat_id,
            tr("payment.credit_error"),
        )
        return "ok"

    balance = int(result.get("user_balance", 0))
    if result.get("inserted"):
        send_message(
            chat_id,
            tr(
                "payment.success",
                credits=format_credits(pack["credits"]),
                balance=format_credits(balance),
            ),
        )
    else:
        send_message(
            chat_id,
            tr("payment.duplicate", balance=format_credits(balance)),
        )
    return "ok"
