from __future__ import annotations

import json
from unittest.mock import MagicMock

from api.billing.callbacks import handle_pre_checkout_query, handle_successful_payment


class _Bridge:
    def __init__(self, decision: dict[str, object] | Exception) -> None:
        self.decision = decision

    def telegram_evaluate_pre_checkout(
        self,
        query_json: str,
        billing_available: bool,
        pack_id: str | None,
        pack_xtr_amount: int | None,
    ) -> str:
        assert json.loads(query_json)["id"] == "checkout-1"
        assert billing_available
        assert pack_id == "p50"
        assert pack_xtr_amount == 25
        if isinstance(self.decision, Exception):
            raise self.decision
        return json.dumps(self.decision)


def _invoke(monkeypatch, decision: dict[str, object] | Exception) -> MagicMock:
    monkeypatch.setattr(
        "api.billing.callbacks.load_rust_telegram_payments",
        lambda: _Bridge(decision),
    )
    answer = MagicMock()
    handle_pre_checkout_query(
        {
            "id": "checkout-1",
            "from": {"id": 42},
            "invoice_payload": "topup:p50:42:en",
            "currency": "XTR",
            "total_amount": 25,
        },
        billing_available=lambda: True,
        answer_query=answer,
        unavailable_alert=lambda: "unavailable",
        parse_payload=lambda _payload: ("p50", 42),
        get_pack=lambda _pack_id: {"id": "p50", "credits": 5_000, "xtr": 25},
    )
    return answer


def test_pre_checkout_approves_validated_rust_decision(monkeypatch) -> None:
    answer = _invoke(monkeypatch, {"kind": "approve", "query_id": "checkout-1"})
    answer.assert_called_once_with("checkout-1", ok=True)


def test_pre_checkout_maps_invalid_rust_decision_to_localized_error(monkeypatch) -> None:
    answer = _invoke(
        monkeypatch,
        {"kind": "invalid_payment", "query_id": "checkout-1"},
    )
    answer.assert_called_once_with(
        "checkout-1",
        ok=False,
        error_message="ese pago vino raro y no te lo pude validar",
    )


def test_pre_checkout_bridge_failure_falls_back_closed(monkeypatch) -> None:
    answer = _invoke(monkeypatch, RuntimeError("synthetic bridge failure"))
    answer.assert_called_once_with("checkout-1", ok=True)


class _SuccessfulBridge:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision

    def telegram_evaluate_successful_payment(
        self,
        message_json: str,
        billing_available: bool,
        pack_id: str | None,
        pack_xtr_amount: int | None,
        pack_credits_awarded: int | None,
    ) -> str:
        assert json.loads(message_json)["chat"]["id"] == 100
        assert (billing_available, pack_id, pack_xtr_amount, pack_credits_awarded) == (
            True,
            "p50",
            25,
            5_000,
        )
        return json.dumps(self.decision)


def _successful_payment_deps(monkeypatch, decision: dict[str, object]):
    monkeypatch.setattr(
        "api.billing.callbacks.load_rust_telegram_payments",
        lambda: _SuccessfulBridge(decision),
    )
    send_message = MagicMock()
    record_payment = MagicMock(return_value={"inserted": True, "user_balance": 7_500})
    admin_report = MagicMock()
    result = handle_successful_payment(
        {
            "chat": {"id": 100},
            "from": {"id": 42},
            "successful_payment": {
                "currency": "XTR",
                "invoice_payload": "topup:p50:42:es",
                "telegram_payment_charge_id": "charge-1",
                "total_amount": 25,
            },
        },
        billing_available=lambda: True,
        unavailable_message=lambda: "unavailable",
        send_message=send_message,
        extract_user_id=lambda _message: 42,
        parse_payload=lambda _payload: ("p50", 42),
        get_pack=lambda _pack_id: {"id": "p50", "credits": 5_000, "xtr": 25},
        record_payment=record_payment,
        admin_report=admin_report,
        format_credits=lambda value: f"{value / 100:.2f}",
    )
    assert result == "ok"
    return send_message, record_payment, admin_report


def test_successful_payment_uses_typed_rust_persistence_inputs(monkeypatch) -> None:
    send_message, record_payment, admin_report = _successful_payment_deps(
        monkeypatch,
        {
            "kind": "record",
            "chat_id": "100",
            "user_id": 42,
            "charge_id": "charge-1",
            "pack_id": "p50",
            "xtr_amount": 25,
            "credits_awarded": 5_000,
            "payload": "topup:p50:42:es",
        },
    )
    record_payment.assert_called_once_with(
        telegram_payment_charge_id="charge-1",
        user_id=42,
        pack_id="p50",
        xtr_amount=25,
        credits_awarded=5_000,
        payload="topup:p50:42:es",
    )
    admin_report.assert_not_called()
    assert "50.00" in send_message.call_args.args[1]


def test_invalid_successful_payment_uses_rust_audit_fields(monkeypatch) -> None:
    send_message, record_payment, admin_report = _successful_payment_deps(
        monkeypatch,
        {
            "kind": "invalid_payment",
            "chat_id": "100",
            "user_id": 42,
            "currency": "USD",
            "payload": "topup:p50:99",
            "total_amount": -1,
            "charge_id": "",
        },
    )
    record_payment.assert_not_called()
    send_message.assert_called_once()
    assert admin_report.call_args.args[2]["currency"] == "USD"
