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
