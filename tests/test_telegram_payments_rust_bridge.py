from __future__ import annotations

import json
from unittest.mock import MagicMock

from api.billing.callbacks import handle_pre_checkout_query


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
