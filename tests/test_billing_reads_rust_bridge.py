from __future__ import annotations

from contextlib import nullcontext
import json
import logging

import pytest

from api.services import credits_db


class _Cursor:
    def __init__(self, balance: int) -> None:
        self.balance = balance

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def execute(self, *_arguments: object) -> None:
        return None

    def fetchone(self) -> tuple[int]:
        return (self.balance,)


class _Connection:
    def __init__(self, balance: int) -> None:
        self.cursor_value = _Cursor(balance)

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def cursor(self) -> _Cursor:
        return self.cursor_value

    def commit(self) -> None:
        return None


class _FakeRustBillingReads:
    def __init__(self, balance: int, *, fail: bool = False) -> None:
        self.balance = balance
        self.fail = fail
        self.calls: list[tuple[str, str, int]] = []

    def billing_read_balance(
        self,
        database_url: str,
        scope_type: str,
        scope_id: int,
    ) -> int:
        if self.fail:
            raise ValueError("synthetic Rust billing read failure")
        self.calls.append((database_url, scope_type, scope_id))
        return self.balance

    def billing_get_or_create_balance(
        self,
        database_url: str,
        scope_type: str,
        scope_id: int,
    ) -> int:
        return self.billing_read_balance(database_url, scope_type, scope_id)


class _FakeRustOnboarding:
    def __init__(
        self,
        result: tuple[bool, int],
        *,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[str, int, int]] = []

    def billing_grant_onboarding(
        self,
        database_url: str,
        user_id: int,
        credits: int,
    ) -> tuple[bool, int]:
        self.calls.append((database_url, user_id, credits))
        if self.fail:
            raise ValueError("synthetic uncertain onboarding failure")
        return self.result


class _FakeRustStarPayments:
    def __init__(
        self,
        result: tuple[bool, int],
        *,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[str, dict[str, object]]] = []

    def billing_record_star_payment(
        self,
        database_url: str,
        payment_json: str,
    ) -> tuple[bool, int]:
        self.calls.append((database_url, json.loads(payment_json)))
        if self.fail:
            raise ValueError("synthetic uncertain Stars payment failure")
        return self.result


class _FakeRustManualCredits:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.mint_calls: list[tuple[str, int, int, int | None]] = []
        self.transfer_calls: list[tuple[str, int, int, int]] = []

    def billing_mint_user_credits(
        self,
        database_url: str,
        user_id: int,
        amount: int,
        actor_user_id: int | None,
    ) -> int:
        self.mint_calls.append((database_url, user_id, amount, actor_user_id))
        if self.fail:
            raise ValueError("synthetic uncertain mint failure")
        return 7000

    def billing_transfer_user_to_chat(
        self,
        database_url: str,
        user_id: int,
        chat_id: int,
        amount: int,
    ) -> tuple[bool, int, int]:
        self.transfer_calls.append((database_url, user_id, chat_id, amount))
        if self.fail:
            raise ValueError("synthetic uncertain transfer failure")
        return True, 200, 520


class _FakeRustChatAiCredits:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int, int, str, dict[str, object]]] = []

    def _record(
        self,
        database_url: str,
        chat_id: int,
        amount: int,
        event_type: str,
        metadata_json: str,
    ) -> None:
        self.calls.append(
            (database_url, chat_id, amount, event_type, json.loads(metadata_json))
        )
        if self.fail:
            raise ValueError("synthetic uncertain chat AI billing failure")

    def billing_charge_chat_ai_credits(
        self,
        database_url: str,
        chat_id: int,
        amount: int,
        event_type: str,
        metadata_json: str,
    ) -> tuple[bool, int]:
        self._record(database_url, chat_id, amount, event_type, metadata_json)
        return True, 700

    def billing_refund_chat_ai_credits(
        self,
        database_url: str,
        chat_id: int,
        amount: int,
        event_type: str,
        metadata_json: str,
    ) -> int:
        self._record(database_url, chat_id, amount, event_type, metadata_json)
        return 900

    def billing_apply_chat_ai_debt(
        self,
        database_url: str,
        chat_id: int,
        amount: int,
        event_type: str,
        metadata_json: str,
    ) -> int:
        self._record(database_url, chat_id, amount, event_type, metadata_json)
        return -100


class _FakeRustAiDebt:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[
            tuple[str, int, int | None, int, str, str, dict[str, object]]
        ] = []

    def billing_apply_ai_debt(
        self,
        database_url: str,
        user_id: int,
        chat_id: int | None,
        amount: int,
        source: str,
        event_type: str,
        metadata_json: str,
    ) -> tuple[int, int]:
        self.calls.append(
            (
                database_url,
                user_id,
                chat_id,
                amount,
                source,
                event_type,
                json.loads(metadata_json),
            )
        )
        if self.fail:
            raise ValueError("synthetic uncertain AI debt failure")
        return 500, -200


class _FakeRustAiRefunds:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[
            tuple[
                str,
                int,
                int | None,
                int,
                str,
                str,
                dict[str, object],
                str | None,
                str,
            ]
        ] = []

    def billing_refund_ai_charge(
        self,
        database_url: str,
        user_id: int,
        chat_id: int | None,
        amount: int,
        source: str,
        event_type: str,
        metadata_json: str,
        idempotency_key: str | None,
        operation_id: str,
    ) -> tuple[bool, str | None, int, int]:
        self.calls.append(
            (
                database_url,
                user_id,
                chat_id,
                amount,
                source,
                event_type,
                json.loads(metadata_json),
                idempotency_key,
                operation_id,
            )
        )
        if self.fail:
            raise ValueError("synthetic uncertain AI refund failure")
        return True, None, 500, 900


class _FakeRustAiCharges:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def billing_charge_ai_credits(
        self,
        database_url: str,
        user_id: int,
        chat_id: int | None,
        amount: int,
        event_type: str,
        metadata_json: str,
        source: str | None,
        idempotency_key: str | None,
        operation_id: str,
    ) -> tuple[bool, bool, str | None, str | None, int, int, int]:
        self.calls.append(
            (
                database_url,
                user_id,
                chat_id,
                amount,
                event_type,
                json.loads(metadata_json),
                source,
                idempotency_key,
                operation_id,
            )
        )
        if self.fail:
            raise ValueError("synthetic uncertain AI charge failure")
        return True, True, None, "chat", amount, 100, 500


class _FakeRustProviderUsage:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int, int | None, dict[str, object]]] = []

    def billing_record_ai_provider_usage(
        self,
        database_url: str,
        user_id: int,
        chat_id: int | None,
        metadata_json: str,
    ) -> bool:
        self.calls.append((database_url, user_id, chat_id, json.loads(metadata_json)))
        if self.fail:
            raise ValueError("synthetic uncertain provider usage failure")
        return True

    def billing_list_ai_provider_segments(
        self,
        database_url: str,
        user_id: int,
        operation_id: str,
    ) -> str:
        if self.fail:
            raise ValueError("synthetic provider usage read failure")
        return json.dumps([{"input_tokens": 12}])

    def billing_update_ai_provider_usage(
        self,
        database_url: str,
        operation_id: str,
        segment_id: str,
        segment_json: str,
    ) -> bool:
        if self.fail:
            raise ValueError("synthetic uncertain provider usage update failure")
        self.calls.append(
            (
                database_url,
                0,
                None,
                {
                    "operation_id": operation_id,
                    "segment_id": segment_id,
                    "segment": json.loads(segment_json),
                },
            )
        )
        return True


def _patch_python_balance(
    monkeypatch: pytest.MonkeyPatch,
    balance: int,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "_ensure_account", lambda *_arguments: None)
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: nullcontext(_Connection(balance)),
    )
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )


def test_billing_balance_shadow_matches_without_changing_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_python_balance(monkeypatch, 1234)
    rust = _FakeRustBillingReads(1234)
    monkeypatch.setattr(credits_db, "_load_rust_billing_reads", lambda: rust)

    assert credits_db.get_balance("user", 7) == 1234
    assert rust.calls == [
        ("postgresql://synthetic.invalid/db?sslmode=require", "user", 7)
    ]


def test_billing_balance_shadow_reports_mismatch_and_keeps_python_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_python_balance(monkeypatch, 1234)
    rust = _FakeRustBillingReads(1200)
    monkeypatch.setattr(credits_db, "_load_rust_billing_reads", lambda: rust)

    with caplog.at_level(logging.WARNING, logger=credits_db.__name__):
        assert credits_db.get_balance("chat", 9) == 1234

    assert "billing balance shadow mismatch" in caplog.text


def test_billing_balance_shadow_failure_keeps_python_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_python_balance(monkeypatch, 1234)
    rust = _FakeRustBillingReads(0, fail=True)
    monkeypatch.setattr(credits_db, "_load_rust_billing_reads", lambda: rust)

    assert credits_db.get_balance("user", 7) == 1234


def test_billing_balance_io_is_authoritative_without_python_account_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: pytest.fail("Python account I/O must not run"),
    )
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    rust = _FakeRustBillingReads(2468)
    monkeypatch.setattr(credits_db, "_load_rust_billing_balance_io", lambda: rust)

    assert credits_db.get_balance("chat", 9) == 2468
    assert rust.calls == [
        ("postgresql://synthetic.invalid/db?sslmode=require", "chat", 9)
    ]


def test_billing_balance_io_failure_uses_idempotent_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_python_balance(monkeypatch, 1234)
    rust = _FakeRustBillingReads(0, fail=True)
    monkeypatch.setattr(credits_db, "_load_rust_billing_balance_io", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=credits_db.__name__):
        assert credits_db.get_balance("user", 7) == 1234

    assert "Rust billing balance I/O failed; using Python fallback" in caplog.text


def test_billing_onboarding_is_authoritative_without_python_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python onboarding writer must not run"),
    )
    rust = _FakeRustOnboarding((True, 300))
    monkeypatch.setattr(credits_db, "_load_rust_billing_onboarding", lambda: rust)

    assert credits_db.grant_onboarding_if_needed(42, 300) == (True, 300)
    assert rust.calls == [
        ("postgresql://synthetic.invalid/db?sslmode=require", 42, 300)
    ]


def test_billing_onboarding_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    rust = _FakeRustOnboarding((False, 0), fail=True)
    monkeypatch.setattr(credits_db, "_load_rust_billing_onboarding", lambda: rust)

    with pytest.raises(credits_db.CreditsDBError, match="onboarding grant"):
        credits_db.grant_onboarding_if_needed(42, 300)


def test_billing_star_payment_is_authoritative_and_preserves_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python Stars writer must not run"),
    )
    rust = _FakeRustStarPayments((True, 500))
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_star_payments",
        lambda: rust,
    )

    assert credits_db.record_star_payment(
        "synthetic-charge",
        42,
        "small",
        100,
        500,
        "synthetic-payload",
    ) == {"inserted": True, "user_balance": 500}
    assert rust.calls == [
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            {
                "charge_id": "synthetic-charge",
                "user_id": 42,
                "pack_id": "small",
                "xtr_amount": 100,
                "credits_awarded": 500,
                "payload": "synthetic-payload",
            },
        )
    ]


def test_billing_star_payment_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    rust = _FakeRustStarPayments((False, 0), fail=True)
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_star_payments",
        lambda: rust,
    )

    with pytest.raises(credits_db.CreditsDBError, match="Stars payment"):
        credits_db.record_star_payment("synthetic-charge", 42, "small", 100, 500)


def test_billing_manual_credit_operations_are_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python manual-credit writer must not run"),
    )
    rust = _FakeRustManualCredits()
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_manual_credits",
        lambda: rust,
    )

    assert credits_db.mint_user_credits(42, 5000, 99) == {"user_balance": 7000}
    assert credits_db.transfer_user_to_chat(42, 202, 300) == {
        "ok": True,
        "error": None,
        "user_balance": 200,
        "chat_balance": 520,
    }
    assert rust.mint_calls == [
        ("postgresql://synthetic.invalid/db?sslmode=require", 42, 5000, 99)
    ]
    assert rust.transfer_calls == [
        ("postgresql://synthetic.invalid/db?sslmode=require", 42, 202, 300)
    ]


@pytest.mark.parametrize("operation", ["mint", "transfer"])
def test_billing_manual_credit_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_manual_credits",
        lambda: _FakeRustManualCredits(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match=f"credit {operation}"):
        if operation == "mint":
            credits_db.mint_user_credits(42, 5000, 99)
        else:
            credits_db.transfer_user_to_chat(42, 202, 300)


def test_billing_chat_ai_operations_are_authoritative_and_preserve_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python chat AI writer must not run"),
    )
    rust = _FakeRustChatAiCredits()
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_chat_ai_credits",
        lambda: rust,
    )

    assert credits_db.charge_chat_ai_credits(
        202,
        300,
        event_type="custom_reserve",
        metadata={"operation_id": "synthetic-operation"},
    ) == {
        "ok": True,
        "source": "chat",
        "chat_balance": 700,
        "chat_balance_credit_units": 700,
    }
    assert credits_db.refund_chat_ai_credits(
        202,
        200,
        metadata={"reason": "synthetic"},
    ) == {"chat_balance": 900}
    assert credits_db.apply_chat_ai_debt(202, 1000) == {"chat_balance": -100}
    assert rust.calls == [
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            202,
            300,
            "custom_reserve",
            {"operation_id": "synthetic-operation"},
        ),
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            202,
            200,
            "ai_refund",
            {"reason": "synthetic"},
        ),
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            202,
            1000,
            "ai_settlement_debt",
            {},
        ),
    ]


@pytest.mark.parametrize("operation", ["charge", "refund", "debt"])
def test_billing_chat_ai_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_chat_ai_credits",
        lambda: _FakeRustChatAiCredits(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match=f"chat AI {operation}"):
        if operation == "charge":
            credits_db.charge_chat_ai_credits(202, 300)
        elif operation == "refund":
            credits_db.refund_chat_ai_credits(202, 300)
        else:
            credits_db.apply_chat_ai_debt(202, 300)


def test_billing_ai_debt_is_authoritative_and_preserves_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python AI debt writer must not run"),
    )
    rust = _FakeRustAiDebt()
    monkeypatch.setattr(credits_db, "_load_rust_billing_ai_debt", lambda: rust)

    assert credits_db.apply_ai_debt(
        42,
        202,
        900,
        "chat",
        event_type="custom_debt",
        metadata={"operation_id": "synthetic-operation"},
    ) == {"user_balance": 500, "chat_balance": -200}
    assert rust.calls == [
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            42,
            202,
            900,
            "chat",
            "custom_debt",
            {"operation_id": "synthetic-operation"},
        )
    ]


def test_billing_ai_debt_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_ai_debt",
        lambda: _FakeRustAiDebt(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match="Rust AI debt"):
        credits_db.apply_ai_debt(42, 202, 900, "chat")


def test_billing_ai_refund_is_authoritative_and_preserves_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        credits_db,
        "get_database_url",
        lambda: "postgresql://synthetic.invalid/db?sslmode=require",
    )
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python AI refund writer must not run"),
    )
    rust = _FakeRustAiRefunds()
    monkeypatch.setattr(credits_db, "_load_rust_billing_ai_refunds", lambda: rust)

    assert credits_db.refund_ai_charge(
        42,
        202,
        300,
        "chat",
        event_type="custom_refund",
        metadata={"operation_id": "synthetic-operation"},
        idempotency_key=" synthetic-refund ",
    ) == {"applied": True, "user_balance": 500, "chat_balance": 900}
    assert rust.calls == [
        (
            "postgresql://synthetic.invalid/db?sslmode=require",
            42,
            202,
            300,
            "chat",
            "custom_refund",
            {
                "operation_id": "synthetic-operation",
                "idempotency_key": "synthetic-refund",
            },
            "synthetic-refund",
            "synthetic-operation",
        )
    ]


def test_billing_ai_refund_preserves_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustAiRefunds()
    monkeypatch.setattr(
        rust,
        "billing_refund_ai_charge",
        lambda *_arguments: (False, "operation_settled", 100, 200),
    )
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(credits_db, "_load_rust_billing_ai_refunds", lambda: rust)

    assert credits_db.refund_ai_charge(42, 202, 300, "chat") == {
        "applied": False,
        "reason": "operation_settled",
        "user_balance": 100,
        "chat_balance": 200,
    }


def test_billing_ai_refund_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_ai_refunds",
        lambda: _FakeRustAiRefunds(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match="Rust AI refund"):
        credits_db.refund_ai_charge(42, 202, 300, "chat")


def test_billing_ai_charge_is_authoritative_and_preserves_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("Python AI charge writer must not run"),
    )
    rust = _FakeRustAiCharges()
    monkeypatch.setattr(credits_db, "_load_rust_billing_ai_charges", lambda: rust)

    assert credits_db.charge_ai_credits(
        42,
        202,
        300,
        event_type="ai_reserve",
        metadata={"operation_id": "synthetic-operation"},
        source="chat",
        idempotency_key=" synthetic-reserve ",
    ) == {
        "ok": True,
        "applied": True,
        "source": "chat",
        "amount": 300,
        "user_balance": 100,
        "chat_balance": 500,
        "user_balance_credit_units": 100,
        "chat_balance_credit_units": 500,
    }
    assert rust.calls == [
        (
            "postgresql://db",
            42,
            202,
            300,
            "ai_reserve",
            {
                "operation_id": "synthetic-operation",
                "idempotency_key": "synthetic-reserve",
            },
            "chat",
            "synthetic-reserve",
            "synthetic-operation",
        )
    ]


def test_billing_ai_charge_preserves_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustAiCharges()
    monkeypatch.setattr(
        rust,
        "billing_charge_ai_credits",
        lambda *_arguments: (False, False, "operation_settled", None, 0, 100, 200),
    )
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(credits_db, "_load_rust_billing_ai_charges", lambda: rust)

    result = credits_db.charge_ai_credits(42, 202, 300)
    assert result["ok"] is False
    assert result["reason"] == "operation_settled"
    assert result["source"] is None


def test_billing_ai_charge_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "_run_credit_transaction",
        lambda *_arguments: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_ai_charges",
        lambda: _FakeRustAiCharges(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match="Rust AI charge"):
        credits_db.charge_ai_credits(42, 202, 300)


def test_billing_provider_usage_is_authoritative_and_preserves_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: pytest.fail("Python provider usage writer must not run"),
    )
    rust = _FakeRustProviderUsage()
    monkeypatch.setattr(credits_db, "_load_rust_billing_provider_usage", lambda: rust)

    assert credits_db.record_ai_provider_usage(
        42,
        202,
        "synthetic-operation",
        "segment-1",
        {"input_tokens": 12},
    )
    assert rust.calls == [
        (
            "postgresql://db",
            42,
            202,
            {
                "operation_id": "synthetic-operation",
                "segment_id": "segment-1",
                "segment": {"input_tokens": 12},
            },
        )
    ]


def test_billing_provider_usage_uncertain_failure_does_not_start_python_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: pytest.fail("uncertain writes must fail closed"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_provider_usage",
        lambda: _FakeRustProviderUsage(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match="provider usage"):
        credits_db.record_ai_provider_usage(42, None, "operation", "segment", {})


def test_billing_provider_usage_reads_and_updates_are_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: pytest.fail("Python provider usage I/O must not run"),
    )
    rust = _FakeRustProviderUsage()
    monkeypatch.setattr(credits_db, "_load_rust_billing_provider_usage", lambda: rust)

    assert credits_db.list_ai_provider_segments(42, "operation") == [
        {"input_tokens": 12}
    ]
    assert credits_db.update_ai_provider_usage(
        "operation",
        "segment",
        {"input_tokens": 99},
    )
    assert rust.calls == [
        (
            "postgresql://db",
            0,
            None,
            {
                "operation_id": "operation",
                "segment_id": "segment",
                "segment": {"input_tokens": 99},
            },
        )
    ]


@pytest.mark.parametrize("operation", ["list", "update"])
def test_billing_provider_usage_followup_failure_is_not_retried_in_python(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(credits_db, "ensure_schema", lambda: None)
    monkeypatch.setattr(credits_db, "get_database_url", lambda: "postgresql://db")
    monkeypatch.setattr(
        credits_db,
        "connect",
        lambda: pytest.fail("Rust-authoritative I/O must not run in Python"),
    )
    monkeypatch.setattr(
        credits_db,
        "_load_rust_billing_provider_usage",
        lambda: _FakeRustProviderUsage(fail=True),
    )

    with pytest.raises(credits_db.CreditsDBError, match="provider"):
        if operation == "list":
            credits_db.list_ai_provider_segments(42, "operation")
        else:
            credits_db.update_ai_provider_usage("operation", "segment", {})
