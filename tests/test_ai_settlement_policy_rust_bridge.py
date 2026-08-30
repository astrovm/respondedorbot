from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.ai import service


class _FakeRustAISettlementPolicy:
    def __init__(self, action: str = "noop", *, fail: bool = False) -> None:
        self.action = action
        self.fail = fail
        self.calls: list[tuple[str, tuple[bool, ...]]] = []

    def _result(self, operation: str, facts: tuple[bool, ...]) -> str:
        self.calls.append((operation, facts))
        if self.fail:
            raise ValueError("synthetic Rust AI settlement failure")
        return self.action

    def ai_media_settlement_action(self, *facts: bool) -> str:
        return self._result("media", facts)

    def ai_conversation_settlement_action(self, *facts: bool) -> str:
        return self._result("conversation", facts)

    def ai_summary_settlement_action(self, *facts: bool) -> str:
        return self._result("summary", facts)

    def ai_delivery_failure_settlement_action(self, *facts: bool) -> str:
        return self._result("delivery_failure", facts)


def test_rust_media_settlement_policy_controls_financial_side_effect(monkeypatch) -> None:
    rust = _FakeRustAISettlementPolicy("noop")
    monkeypatch.setattr(service, "_load_rust_ai_settlement_policy", lambda: rust)
    billing = MagicMock()
    request = SimpleNamespace(billing_helper=billing)

    service.AIService._settle_incurred_media(
        request,
        {"reservation_id": "synthetic-reservation"},
        [{"kind": "vision"}],
        reason="synthetic_reason",
    )

    billing.settle_reserved_ai_credits_batch.assert_not_called()
    billing.refund_reserved_ai_credits.assert_not_called()
    assert rust.calls == [("media", (True, True))]


def test_rust_settlement_policy_failure_preserves_python_decisions(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAISettlementPolicy(fail=True)
    monkeypatch.setattr(service, "_load_rust_ai_settlement_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=service.__name__):
        assert service._media_settlement_action(True, True) == "settle_success"
        assert (
            service._conversation_settlement_action(True, True)
            == "settle_usage_before_fallback"
        )
        assert (
            service._summary_settlement_action(True, True, True)
            == "refund_provider_unavailable"
        )
        assert (
            service._delivery_failure_settlement_action(False)
            == "refund_delivery_failure"
        )

    assert caplog.text.count("using Python fallback") == 4


def test_invalid_or_cross_operation_rust_action_uses_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAISettlementPolicy("continue")
    monkeypatch.setattr(service, "_load_rust_ai_settlement_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=service.__name__):
        assert service._conversation_settlement_action(False, False) == "settle_success"

    assert "invalid Rust AI settlement action" in caplog.text


def test_python_ai_settlement_policy_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(service, "_load_rust_ai_settlement_policy", lambda: None)
    path = Path(__file__).parents[1] / "contracts" / "ai_settlement_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    functions = {
        "media_cases": service._media_settlement_action,
        "conversation_cases": service._conversation_settlement_action,
        "summary_cases": service._summary_settlement_action,
        "delivery_failure_cases": service._delivery_failure_settlement_action,
    }

    for group, function in functions.items():
        for case in contract[group]:
            assert function(*case["facts"]) == case["expected"], case["name"]
