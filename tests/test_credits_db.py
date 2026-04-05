from unittest.mock import patch

from api.credit_units import CREDIT_SCALE, whole_credits_to_units
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


class _MigrationCursor:
    def __init__(self):
        self.balance = 3
        self.chat_balance = 2
        self.onboarding_credits = 3
        self.star_credits_awarded = 100
        self.ledger_amounts = [-3, 3]
        self.migration_inserted = False
        self.fetchone_result = None

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())

        if "INSERT INTO credit_schema_migrations" in normalized:
            if not self.migration_inserted:
                self.migration_inserted = True
                self.fetchone_result = (credits_db.CREDIT_UNITS_MIGRATION_NAME,)
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

        self.fetchone_result = None

    def fetchone(self):
        return self.fetchone_result


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
    assert balance == 30
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

    assert balances == {"user_balance": -29, "chat_balance": 0}
    assert any(
        "INSERT INTO credit_ledger" in query
        and params is not None
        and params[0] == "ai_settlement_debt"
        and params[4] == -30
        for query, params in fake_cursor.executed
    )


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

    assert result == {"user_balance": 700}
    assert any(
        "INSERT INTO credit_ledger" in query
        and params is not None
        and params[0] == "printcredits"
        and params[1] == 99
        and params[2] == 42
        and params[3] == 500
        for query, params in fake_cursor.executed
    )


def test_migrate_credit_amounts_to_units_scales_existing_rows_once():
    cursor = _MigrationCursor()

    migrated = credits_db._migrate_credit_amounts_to_units(cursor)

    assert migrated is True
    assert cursor.balance == 3 * CREDIT_SCALE
    assert cursor.chat_balance == 2 * CREDIT_SCALE
    assert cursor.onboarding_credits == 3 * CREDIT_SCALE
    assert cursor.star_credits_awarded == 100 * CREDIT_SCALE
    assert cursor.ledger_amounts == [-3 * CREDIT_SCALE, 3 * CREDIT_SCALE]

    migrated_again = credits_db._migrate_credit_amounts_to_units(cursor)

    assert migrated_again is False
    assert cursor.balance == 3 * CREDIT_SCALE
    assert cursor.star_credits_awarded == 100 * CREDIT_SCALE


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
