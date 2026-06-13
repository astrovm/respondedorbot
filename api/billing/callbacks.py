from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from api.billing.ai import AIBillingPack
from api.core.constants import BILLING_UNAVAILABLE_MESSAGE


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
            "title": f"Pack IA {pack_credits} créditos",
            "description": (
                f"Recarga de {pack_credits} créditos para mensajes IA"
            ),
            "payload": f"topup:{pack['id']}:{user_id}",
            "provider_token": "",
            "currency": "XTR",
            "prices": [
                {
                    "label": f"{pack_credits} créditos IA",
                    "amount": pack["xtr"],
                }
            ],
        },
    )
    return error is None and bool(response)


def billing_unavailable_message() -> str:
    return BILLING_UNAVAILABLE_MESSAGE


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
        text="cargá por privado, maestro",
        show_alert=True,
    ):
        return

    parts = callback_data.split(":", 1)
    pack = get_pack(parts[1] if len(parts) == 2 else "")
    if guard_callback(
        callback_id,
        not pack,
        text="ese pack es fruta, elegí otro",
        show_alert=True,
    ):
        return
    try:
        user_id = int(str(user.get("id")))
    except (TypeError, ValueError):
        guard_callback(callback_id, True)
        return
    assert pack is not None
    sent = send_invoice(chat_id=str(chat_id), user_id=user_id, pack=pack)
    if callback_id:
        if sent:
            answer_callback(callback_id, text="listo, te dejé la factura")
        else:
            answer_callback(
                callback_id,
                text="no pude armar la factura, probá de nuevo",
                show_alert=True,
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
    except (TypeError, ValueError):
        answer_query(
            str(query_id),
            ok=False,
            error_message="tu usuario vino medio roto para cobrar",
        )
        return
    try:
        total_amount = int(str(query.get("total_amount")))
    except (TypeError, ValueError):
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
            error_message="ese pago vino raro y no te lo pude validar",
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
    except (TypeError, ValueError):
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
            "me cayó un pago raro y no lo pude validar, avisale al admin",
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
            "me entró la guita pero se trabó la acreditación, avisale al admin",
        )
        return "ok"

    balance = int(result.get("user_balance", 0))
    if result.get("inserted"):
        send_message(
            chat_id,
            (
                f"listo, te cargué {format_credits(pack['credits'])} créditos\n"
                f"ahora te quedaron {format_credits(balance)}\n"
                "si querés mandarle al grupo: /transfer <monto>"
            ),
        )
    else:
        send_message(
            chat_id,
            (
                "ese pago ya estaba cargado, no rompas las bolas\n"
                f"te quedaron {format_credits(balance)}"
            ),
        )
    return "ok"
