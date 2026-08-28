import json
from unittest.mock import patch

from psycopg._queries import PostgresQuery
from psycopg.adapt import Transformer

from api.billing.credit_units import CREDIT_SCALE, whole_credits_to_units
from api.services import credits_db


class _FakeCursor:
    def __init__(
        self,
        *,
        hourly_count: int,
        daily_count: int,
        insert_granted: bool,
        has_existing_grant: bool = False,
    ):
        self.hourly_count = hourly_count
        self.daily_count = daily_count
        self.insert_granted = insert_granted
        self.has_existing_grant = has_existing_grant
        self.balance = 0
        self.chat_balance = 0
        self.fetchone_result = None
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())
        self.executed.append((normalized, params))

        if "COUNT(*) FILTER" in normalized:
            self.fetchone_result = (self.hourly_count, self.daily_count)
            return

        if "INSERT INTO onboarding_grants" in normalized:
            self.fetchone_result = (123,) if self.insert_granted else None
            return

        if "SELECT balance" in normalized and "FOR UPDATE" in normalized:
            if params and params[0] == "chat":
                self.fetchone_result = (self.chat_balance,)
            else:
                self.fetchone_result = (self.balance,)
            return

        if (
            "FROM onboarding_grants" in normalized
            and "WHERE user_id = %s" in normalized
        ):
            self.fetchone_result = (1,) if self.has_existing_grant else None
            return

        if "UPDATE credit_accounts" in normalized and params is not None:
            if params[1] == "chat":
                self.chat_balance = int(params[0])
            else:
                self.balance = int(params[0])

        self.fetchone_result = None

    def fetchone(self):
        return self.fetchone_result


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commit_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commit_count += 1


def _locked_accounts(cursor):
    return [
        params
        for query, params in cursor.executed
        if "SELECT balance" in query and "FOR UPDATE" in query
    ]


class _MigrationCursor:
    def __init__(self):
        self.balance = 3
        self.chat_balance = 2
        self.onboarding_credits = 3
        self.star_credits_awarded = 100
        self.ledger_amounts = [-3, 3]
        self.ledger_metadata = [
            {
                "reserved_credit_units": 3,
                "settled_credits": 2,
                "note": "keep",
            },
            {"actual_credit_units": 2},
        ]
        self.applied_migrations = set()
        self.fetchone_result = None

    def _scale_ledger_metadata(self, params):
        factor = int(params[0])
        legacy_keys = set(params[1])
        legacy_factor = int(params[2])
        credit_scale = int(params[3])
        for metadata in self.ledger_metadata:
            for key, value in list(metadata.items()):
                if "credit_units" in key and isinstance(value, (int, float)):
                    metadata[key] = int(value) * factor
                elif key in legacy_keys and isinstance(value, (int, float)):
                    metadata[key] = int(value) * legacy_factor
            if any(
                "credit_units" in key or key in legacy_keys for key in metadata
            ):
                metadata["credit_scale"] = credit_scale

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())

        if "INSERT INTO credit_schema_migrations" in normalized:
            migration_name = params[0]
            if migration_name not in self.applied_migrations:
                self.applied_migrations.add(migration_name)
                self.fetchone_result = (migration_name,)
            else:
                self.fetchone_result = None
            return

        if normalized == "UPDATE credit_accounts SET balance = balance * %s":
            scale = int(params[0])
            self.balance *= scale
            self.chat_balance *= scale
            self.fetchone_result = None
            return

        if normalized == "UPDATE onboarding_grants SET credits = credits * %s":
            self.onboarding_credits *= int(params[0])
            self.fetchone_result = None
            return

        if (
            normalized
            == "UPDATE star_payments SET credits_awarded = credits_awarded * %s"
        ):
            self.star_credits_awarded *= int(params[0])
            self.fetchone_result = None
            return

        if normalized == "UPDATE credit_ledger SET amount = amount * %s":
            scale = int(params[0])
            self.ledger_amounts = [amount * scale for amount in self.ledger_amounts]
            self.fetchone_result = None
            return

        if normalized.startswith("UPDATE credit_ledger SET metadata ="):
            self._scale_ledger_metadata(params)
            self.fetchone_result = None
            return

        self.fetchone_result = None

    def fetchone(self):
        return self.fetchone_result


class _PsycopgValidatingMigrationCursor(_MigrationCursor):
    def execute(self, query, params=None):
        PostgresQuery(Transformer(None)).convert(query, params)
        super().execute(query, params)


class _CompactionRepairCursor:
    def __init__(self):
        self.balance = 100
        self.migration_applied = False
        self.fetchone_result = None
        self.fetchall_result = []
        self.corrections = []

    def execute(self, query, params=None):
        PostgresQuery(Transformer(None)).convert(query, params)
        normalized = " ".join(str(query).split())
        if "INSERT INTO credit_schema_migrations" in normalized:
            if self.migration_applied:
                self.fetchone_result = None
            else:
                self.migration_applied = True
                self.fetchone_result = (params[0],)
            return
        if normalized.startswith("SELECT DISTINCT ON (refund.id)"):
            self.fetchall_result = [
                (
                    15828,
                    42,
                    None,
                    5,
                    "user",
                    "operation-1",
                    "memory_compaction:123:m1",
                )
            ]
            return
        if "SELECT balance" in normalized and "FOR UPDATE" in normalized:
            self.fetchone_result = (self.balance,)
            return
        if normalized.startswith("UPDATE credit_accounts SET balance"):
            self.balance = int(params[0])
            return
        if (
            "INSERT INTO credit_ledger" in normalized
            and params[0] == "ai_reconciliation_correction"
        ):
            self.corrections.append(params)

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result


def test_should_deny_onboarding_grant_when_hourly_limit_reached():
    assert credits_db._should_deny_onboarding_grant(
        credits_db.ONBOARDING_MAX_GRANTS_PER_HOUR,
        0,
    )


def test_should_deny_onboarding_grant_when_daily_limit_reached():
    assert credits_db._should_deny_onboarding_grant(
        0,
        credits_db.ONBOARDING_MAX_GRANTS_PER_DAY,
    )


def test_grant_onboarding_if_needed_denies_when_overflow_detected():
    fake_cursor = _FakeCursor(
        hourly_count=credits_db.ONBOARDING_MAX_GRANTS_PER_HOUR,
        daily_count=0,
        insert_granted=True,
    )
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(
            42, whole_credits_to_units(3)
        )

    assert granted is False
    assert balance == 0
    assert fake_connection.commit_count == 1
    assert any(
        "INSERT INTO credit_ledger" in query
        and params is not None
        and params[0] == "onboarding_denied_overflow"
        for query, params in fake_cursor.executed
    )
    assert not any(
        "INSERT INTO onboarding_grants" in query
        for query, _params in fake_cursor.executed
    )


def test_grant_onboarding_if_needed_grants_credits_when_under_limit():
    fake_cursor = _FakeCursor(hourly_count=1, daily_count=2, insert_granted=True)
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(
            42, whole_credits_to_units(3)
        )

    assert granted is True
    assert balance == 300
    assert fake_connection.commit_count == 1
    assert any(
        "INSERT INTO onboarding_grants" in query
        for query, _params in fake_cursor.executed
    )
    assert any(
        "UPDATE credit_accounts" in query for query, _params in fake_cursor.executed
    )


def test_grant_onboarding_if_needed_skips_overflow_logic_for_existing_users():
    fake_cursor = _FakeCursor(
        hourly_count=credits_db.ONBOARDING_MAX_GRANTS_PER_HOUR,
        daily_count=credits_db.ONBOARDING_MAX_GRANTS_PER_DAY,
        insert_granted=False,
        has_existing_grant=True,
    )
    fake_cursor.balance = 7
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(
            42, whole_credits_to_units(3)
        )

    assert granted is False
    assert balance == 7
    assert fake_connection.commit_count == 1
    assert not any(
        "COUNT(*) FILTER" in query for query, _params in fake_cursor.executed
    )
    assert not any(
        "onboarding_denied_overflow" in str(params)
        for _query, params in fake_cursor.executed
    )


def test_apply_ai_debt_allows_negative_user_balance():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = 1
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        balances = credits_db.apply_ai_debt(42, None, whole_credits_to_units(3), "user")

    assert balances == {"user_balance": -299, "chat_balance": 0}
    assert any(
        "INSERT INTO credit_ledger" in query
        and params is not None
        and params[0] == "ai_settlement_debt"
        and params[4] == -300
        for query, params in fake_cursor.executed
    )


def test_charge_ai_credits_locks_user_before_chat():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = 10
    fake_cursor.chat_balance = 400
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        result = credits_db.charge_ai_credits(
            user_id=42,
            chat_id=202,
            amount=whole_credits_to_units(3),
        )

    assert _locked_accounts(fake_cursor) == [("user", 42), ("chat", 202)]
    assert result["source"] == "chat"
    assert result["chat_balance"] == 100


def test_charge_ai_credits_rejects_a_new_hold_after_operation_settlement():
    class SettledOperationCursor(_FakeCursor):
        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if "event_type = 'ai_settlement_result'" in normalized:
                self.executed.append((normalized, params))
                self.fetchone_result = (1,)
                return
            super().execute(query, params)

    cursor = SettledOperationCursor(
        hourly_count=0,
        daily_count=0,
        insert_granted=False,
    )
    cursor.balance = 100
    connection = _FakeConnection(cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        result = credits_db.charge_ai_credits(
            user_id=42,
            chat_id=None,
            amount=10,
            event_type="ai_reserve",
            metadata={"operation_id": "operation-1"},
        )

    assert result["ok"] is False
    assert result["reason"] == "operation_settled"
    assert cursor.balance == 100
    assert not any(
        "INSERT INTO credit_ledger" in query for query, _params in cursor.executed
    )


def test_charge_ai_credits_does_not_reuse_a_refunded_reservation():
    class RefundedReservationCursor(_FakeCursor):
        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if "metadata->>'idempotency_key' = %s" in normalized:
                self.executed.append((normalized, params))
                self.fetchone_result = (-10, "user")
                return
            if "event_type = 'ai_refund'" in normalized:
                self.executed.append((normalized, params))
                self.fetchone_result = (1,)
                return
            super().execute(query, params)

    cursor = RefundedReservationCursor(
        hourly_count=0,
        daily_count=0,
        insert_granted=False,
    )
    cursor.balance = 100
    connection = _FakeConnection(cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        result = credits_db.charge_ai_credits(
            user_id=42,
            chat_id=None,
            amount=10,
            event_type="ai_reserve",
            metadata={
                "operation_id": "operation-1",
                "settlement_id": "settlement-1",
            },
            idempotency_key="settlement-1",
        )

    assert result["ok"] is False
    assert result["reason"] == "reservation_refunded"
    assert cursor.balance == 100
    assert not any(
        "INSERT INTO credit_ledger" in query for query, _params in cursor.executed
    )


def test_refund_ai_charge_chat_source_locks_user_before_chat():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = 110
    fake_cursor.chat_balance = 220
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        result = credits_db.refund_ai_charge(
            user_id=42,
            chat_id=202,
            amount=whole_credits_to_units(3),
            source="chat",
        )

    assert _locked_accounts(fake_cursor) == [("user", 42), ("chat", 202)]
    assert result == {"applied": True, "user_balance": 110, "chat_balance": 520}


def test_refund_ai_charge_reports_an_idempotent_replay_as_not_applied():
    class ReplayedRefundCursor(_FakeCursor):
        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if "metadata->>'idempotency_key' = %s" in normalized:
                self.executed.append((normalized, params))
                self.fetchone_result = (1,)
                return
            super().execute(query, params)

    cursor = ReplayedRefundCursor(
        hourly_count=0,
        daily_count=0,
        insert_granted=False,
    )
    cursor.balance = 100
    connection = _FakeConnection(cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        result = credits_db.refund_ai_charge(
            user_id=42,
            chat_id=None,
            amount=10,
            source="user",
            idempotency_key="settlement-1:refund",
        )

    assert result == {"applied": False, "user_balance": 100, "chat_balance": 0}
    assert cursor.balance == 100
    assert not any(
        "INSERT INTO credit_ledger" in query for query, _params in cursor.executed
    )


def test_refund_ai_charge_does_not_mutate_a_settled_operation():
    class SettledOperationCursor(_FakeCursor):
        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if "event_type = 'ai_settlement_result'" in normalized:
                self.executed.append((normalized, params))
                self.fetchone_result = (1,)
                return
            super().execute(query, params)

    cursor = SettledOperationCursor(
        hourly_count=0,
        daily_count=0,
        insert_granted=False,
    )
    cursor.balance = 100
    connection = _FakeConnection(cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        result = credits_db.refund_ai_charge(
            user_id=42,
            chat_id=None,
            amount=10,
            source="user",
            metadata={"operation_id": "operation-1"},
        )

    assert result["applied"] is False
    assert result["reason"] == "operation_settled"
    assert cursor.balance == 100
    assert not any(
        "INSERT INTO credit_ledger" in query for query, _params in cursor.executed
    )


def test_settle_ai_reservation_once_is_idempotent():
    class SettlementCursor(_FakeCursor):
        def __init__(self):
            super().__init__(hourly_count=0, daily_count=0, insert_granted=False)
            self.balance = 10
            self.settled_usage_tags = set()

        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if (
                "FROM credit_ledger" in normalized
                and "memory_compaction_settlement" in normalized
            ):
                self.fetchone_result = (
                    (1,) if str(params[0]) in self.settled_usage_tags else None
                )
                return
            if (
                "INSERT INTO credit_ledger" in normalized
                and params
                and params[0] == "memory_compaction_settlement"
            ):
                metadata = json.loads(params[5])
                self.settled_usage_tags.add(metadata["usage_tag"])
                self.executed.append((normalized, params))
                self.fetchone_result = None
                return
            super().execute(query, params)

    cursor = SettlementCursor()
    connection = _FakeConnection(cursor)
    kwargs = {
        "user_id": 42,
        "chat_id": None,
        "source": "user",
        "reserved_credit_units": 5,
        "actual_credit_units": 2,
        "usage_tag": "memory_compaction:123:m1",
    }

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        first = credits_db.settle_ai_reservation_once(**kwargs)
        second = credits_db.settle_ai_reservation_once(**kwargs)

    assert first["applied"] is True
    assert second["applied"] is False
    assert cursor.balance == 13
    settlement_rows = [
        params
        for query, params in cursor.executed
        if "INSERT INTO credit_ledger" in query
        and params[0] == "memory_compaction_settlement"
    ]
    assert len(settlement_rows) == 1


def test_settle_ai_reservation_once_uses_atomic_operation_when_available():
    with patch(
        "api.services.credits_db.settle_ai_operation_once",
        return_value={"applied": True},
    ) as settle_operation:
        result = credits_db.settle_ai_reservation_once(
            user_id=42,
            chat_id=None,
            source="user",
            reserved_credit_units=5,
            actual_credit_units=2,
            usage_tag="memory_compaction:123:m1",
            metadata={
                "operation_id": "operation-1",
                "settlement_id": "settlement-1",
                "reason": "memory_compaction_success",
            },
        )

    assert result == {"applied": True}
    kwargs = settle_operation.call_args.kwargs
    assert kwargs["operation_id"] == "operation-1"
    assert kwargs["actual_credit_units"] == 2
    assert kwargs["metadata"]["usage_tag"] == "memory_compaction:123:m1"
    assert kwargs["metadata"]["payer_breakdown"] == [
        {"scope": "user", "credit_units": 2}
    ]


def test_duplicate_compaction_refund_repair_restores_the_charged_balance_once():
    cursor = _CompactionRepairCursor()

    assert credits_db._repair_duplicate_compaction_refunds(cursor) == 1
    assert cursor.balance == 95
    assert len(cursor.corrections) == 1
    assert cursor.corrections[0][4] == -5
    assert json.loads(cursor.corrections[0][5]) == {
        "source": "user",
        "operation_id": "operation-1",
        "usage_tag": "memory_compaction:123:m1",
        "reversed_refund_ledger_id": 15828,
        "reason": "duplicate_compaction_refund",
    }

    assert credits_db._repair_duplicate_compaction_refunds(cursor) == 0
    assert cursor.balance == 95
    assert len(cursor.corrections) == 1


def test_apply_ai_debt_chat_source_locks_user_before_chat():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = 110
    fake_cursor.chat_balance = 220
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        result = credits_db.apply_ai_debt(
            user_id=42,
            chat_id=202,
            amount=whole_credits_to_units(3),
            source="chat",
        )

    assert _locked_accounts(fake_cursor) == [("user", 42), ("chat", 202)]
    assert result == {"user_balance": 110, "chat_balance": -80}


def test_transfer_user_to_chat_locks_user_before_chat():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = 500
    fake_cursor.chat_balance = 220
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        result = credits_db.transfer_user_to_chat(
            user_id=42,
            chat_id=202,
            amount=whole_credits_to_units(3),
        )

    assert _locked_accounts(fake_cursor) == [("user", 42), ("chat", 202)]
    assert result == {
        "ok": True,
        "error": None,
        "user_balance": 200,
        "chat_balance": 520,
    }


def test_credit_transaction_retries_deadlocks():
    class DeadlockDetected(Exception):
        pass

    attempts = 0
    first_connection = _FakeConnection(
        _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    )
    second_connection = _FakeConnection(
        _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    )

    def operation(_cur):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DeadlockDetected("deadlock detected")
        return "ok"

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch(
            "api.services.credits_db.connect",
            side_effect=[first_connection, second_connection],
        ),
    ):
        result = credits_db._run_credit_transaction(operation)

    assert result == "ok"
    assert attempts == 2
    assert first_connection.commit_count == 0
    assert second_connection.commit_count == 1


def test_mint_user_credits_increases_balance_and_writes_ledger():
    fake_cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    fake_cursor.balance = whole_credits_to_units(20)
    fake_connection = _FakeConnection(fake_cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=fake_connection),
    ):
        result = credits_db.mint_user_credits(
            user_id=42,
            amount=whole_credits_to_units(50),
            actor_user_id=99,
        )

    assert result == {"user_balance": 7000}
    assert any(
        "INSERT INTO credit_ledger" in query
        and params is not None
        and params[0] == "printcredits"
        and params[1] == 99
        and params[2] == 42
        and params[3] == 5000
        for query, params in fake_cursor.executed
    )


def test_migrate_credit_amounts_to_units_scales_existing_rows_once():
    cursor = _MigrationCursor()

    migrated = credits_db._migrate_credit_amounts_to_units(cursor)

    assert migrated is True
    assert cursor.balance == 30
    assert cursor.chat_balance == 20
    assert cursor.onboarding_credits == 30
    assert cursor.star_credits_awarded == 1000
    assert cursor.ledger_amounts == [-30, 30]

    migrated_again = credits_db._migrate_credit_amounts_to_units(cursor)

    assert migrated_again is False
    assert cursor.balance == 30
    assert cursor.star_credits_awarded == 1000


def test_migrate_credit_amounts_to_hundredths_scales_tenths_once():
    cursor = _PsycopgValidatingMigrationCursor()
    credits_db._migrate_credit_amounts_to_units(cursor)

    migrated = credits_db._migrate_credit_amounts_to_hundredths(cursor)

    assert migrated is True
    assert cursor.balance == 3 * CREDIT_SCALE
    assert cursor.chat_balance == 2 * CREDIT_SCALE
    assert cursor.onboarding_credits == 3 * CREDIT_SCALE
    assert cursor.star_credits_awarded == 100 * CREDIT_SCALE
    assert cursor.ledger_amounts == [-3 * CREDIT_SCALE, 3 * CREDIT_SCALE]
    assert cursor.ledger_metadata == [
        {
            "reserved_credit_units": 30,
            "settled_credits": 200,
            "note": "keep",
            "credit_scale": CREDIT_SCALE,
        },
        {"actual_credit_units": 20, "credit_scale": CREDIT_SCALE},
    ]

    migrated_again = credits_db._migrate_credit_amounts_to_hundredths(cursor)

    assert migrated_again is False
    assert cursor.balance == 3 * CREDIT_SCALE
    assert cursor.ledger_metadata[0]["reserved_credit_units"] == 30


def test_list_user_ai_charge_page_groups_rows_and_applies_internal_cursor():
    class Cursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, query, params=None):
            PostgresQuery(Transformer(None)).convert(query, params)
            self.executed.append((" ".join(str(query).split()), params))

        def fetchall(self):
            return [
                (
                    98,
                    "ai_settlement_result",
                    42,
                    42,
                    202,
                    0,
                    {"charged_credit_units_total": 7},
                    "2026-08-26T17:00:00+00:00",
                    "202:9",
                    98,
                    "2026-08-26T17:00:00+00:00",
                )
            ]

    cursor = Cursor()
    connection = _FakeConnection(cursor)
    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        results = credits_db.list_user_ai_charge_page(
            42,
            limit=11,
            cursor_id=99,
            direction="older",
        )

    query, params = cursor.executed[0]
    assert "WHERE user_id = %s" in query
    assert "group_cursor < %s" in query
    assert "memory_compaction_settlement" in query
    assert "legacy.metadata->>'usage_tag'" in query
    assert "event_type = 'ai_reserve'" in query
    assert "NOT EXISTS" in query
    assert "SUM(mutation.amount)" in query
    assert "payer_breakdown" in query
    assert params == (
        42,
        99,
        "older",
        99,
        "older",
        99,
        "older",
        "older",
        12,
        "older",
        "older",
    )
    assert results["groups"][0]["entries"][0]["user_id"] == 42
    assert (
        results["groups"][0]["entries"][0]["metadata"][
            "charged_credit_units_total"
        ]
        == 7
    )
    assert results["has_newer"] is True
    assert results["has_older"] is False


def test_unsettled_operations_ignore_legacy_compaction_settlements():
    class Cursor:
        def __init__(self):
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, query, params=None):
            PostgresQuery(Transformer(None)).convert(query, params)
            self.query = " ".join(str(query).split())

        def fetchall(self):
            return []

    cursor = Cursor()
    with (
        patch("api.services.credits_db.ensure_schema"),
        patch(
            "api.services.credits_db.connect",
            return_value=_FakeConnection(cursor),
        ),
    ):
        assert credits_db.list_unsettled_ai_operations() == []

    assert "settled.event_type = 'memory_compaction_settlement'" in cursor.query
    assert "settled.metadata->>'usage_tag'" in cursor.query


def test_list_user_ai_charge_page_reverses_newer_results_and_keeps_groups():
    def row(entry_id, group_key, group_cursor):
        return (
            entry_id,
            "ai_settlement_result",
            42,
            42,
            202,
            0,
            {"charged_credit_units_total": 1},
            "2026-08-26T17:00:00+00:00",
            group_key,
            group_cursor,
            "2026-08-26T17:00:00+00:00",
        )

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, query, params=None):
            PostgresQuery(Transformer(None)).convert(query, params)

        def fetchall(self):
            return [
                row(71, "group-70", 70),
                row(70, "group-70", 70),
                row(80, "group-80", 80),
                row(90, "group-90", 90),
            ]

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch(
            "api.services.credits_db.connect",
            return_value=_FakeConnection(Cursor()),
        ),
    ):
        page = credits_db.list_user_ai_charge_page(
            42,
            limit=2,
            cursor_id=60,
            direction="newer",
        )

    assert [group["cursor_id"] for group in page["groups"]] == [80, 70]
    assert len(page["groups"][1]["entries"]) == 2
    assert page["has_newer"] is True
    assert page["has_older"] is True
    assert page["newer_cursor"] == 80
    assert page["older_cursor"] == 70


def test_record_ai_settlement_result_is_idempotent_by_settlement_id():
    cursor = _FakeCursor(hourly_count=0, daily_count=0, insert_granted=False)
    connection = _FakeConnection(cursor)
    metadata = {"settlement_id": "42:202:9:ai_response_base"}

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        credits_db.record_ai_settlement_result(
            user_id=42,
            chat_id=202,
            metadata=metadata,
        )

    query, params = next(
        (query, params)
        for query, params in cursor.executed
        if "INSERT INTO credit_ledger" in query
    )
    assert "ON CONFLICT DO NOTHING" in query
    assert json.loads(params[5])["settlement_id"] == metadata["settlement_id"]


def test_purge_expired_ai_ledger_events_defaults_to_30_days():
    class Cursor:
        def __init__(self):
            self.rowcount = 4
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, query, params=None):
            self.executed.append((" ".join(str(query).split()), params))

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor
            self.commit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def cursor(self):
            return self._cursor

        def commit(self):
            self.commit_count += 1

    cursor = Cursor()
    connection = Connection(cursor)

    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        result = credits_db.purge_expired_ai_ledger_events()

    assert result == {"deleted_rows": 4, "retention_days": 30}
    assert cursor.executed[0][1][1] == 30
