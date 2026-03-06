from unittest.mock import patch

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
            self.fetchone_result = (self.balance,)
            return

        if "FROM onboarding_grants" in normalized and "WHERE user_id = %s" in normalized:
            self.fetchone_result = (1,) if self.has_existing_grant else None
            return

        if "UPDATE credit_accounts" in normalized and params is not None:
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

    with patch("api.services.credits_db.ensure_schema"), patch(
        "api.services.credits_db.connect", return_value=fake_connection
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(42, 3)

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
        "INSERT INTO onboarding_grants" in query for query, _params in fake_cursor.executed
    )


def test_grant_onboarding_if_needed_grants_credits_when_under_limit():
    fake_cursor = _FakeCursor(hourly_count=1, daily_count=2, insert_granted=True)
    fake_connection = _FakeConnection(fake_cursor)

    with patch("api.services.credits_db.ensure_schema"), patch(
        "api.services.credits_db.connect", return_value=fake_connection
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(42, 3)

    assert granted is True
    assert balance == 3
    assert fake_connection.commit_count == 1
    assert any(
        "INSERT INTO onboarding_grants" in query for query, _params in fake_cursor.executed
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

    with patch("api.services.credits_db.ensure_schema"), patch(
        "api.services.credits_db.connect", return_value=fake_connection
    ):
        granted, balance = credits_db.grant_onboarding_if_needed(42, 3)

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
