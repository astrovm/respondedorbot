"""Postgres-backed credits and payments storage for AI billing."""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import json

from api.billing.credit_units import CREDIT_SCALE

ScopeType = Literal["user", "chat"]
T = TypeVar("T")

_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False

ONBOARDING_MAX_GRANTS_PER_HOUR = 4
ONBOARDING_MAX_GRANTS_PER_DAY = 16
ONBOARDING_GRANTS_ADVISORY_LOCK_KEY = 48_610_001
AI_LEDGER_EVENT_TYPES = (
    "ai_reserve",
    "ai_provider_usage",
    "ai_refund",
    "ai_settlement_charge",
    "ai_settlement_debt",
    "ai_settlement_result",
    "memory_compaction_settlement",
)
CREDIT_UNITS_MIGRATION_ADVISORY_LOCK_KEY = 48_610_002
CREDIT_UNITS_MIGRATION_NAME = "credit_amounts_scaled_to_tenths_v1"
CREDIT_HUNDREDTHS_MIGRATION_ADVISORY_LOCK_KEY = 48_610_003
CREDIT_HUNDREDTHS_MIGRATION_NAME = "credit_amounts_scaled_to_hundredths_v2"
_LEGACY_WHOLE_TO_TENTHS_FACTOR = 10
_TENTHS_TO_HUNDREDTHS_FACTOR = CREDIT_SCALE // 10
_LEGACY_METADATA_CREDIT_KEYS = (
    "reserved_credits",
    "reserved_credits_total",
    "settled_credits",
    "refunded_credits",
    "extra_charged_credits",
    "debt_applied_credits",
)
AI_LEDGER_RETENTION_DAYS = 30
CREDIT_TRANSACTION_MAX_ATTEMPTS = 3
_RETRYABLE_CREDIT_TRANSACTION_ERRORS = {
    "DeadlockDetected",
    "SerializationFailure",
}


class CreditsDBError(RuntimeError):
    """Raised when credits persistence cannot be completed."""


def _schema_is_ready() -> bool:
    """Read schema state without assuming it is stable across threads."""

    return _SCHEMA_READY


def _append_sslmode_if_missing(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    # Supabase URLs may include pooler options that psycopg does not accept.
    allowed_params = {"sslmode", "connect_timeout", "options", "application_name"}
    query = {k: v for k, v in query.items() if k in allowed_params}

    if "sslmode" not in query:
        query["sslmode"] = "require"

    return urlunparse(parsed._replace(query=urlencode(query)))


def get_database_url() -> Optional[str]:
    """Return the Postgres connection URL from env vars."""

    value = str(environ.get("SUPABASE_POSTGRES_URL") or "").strip()
    if value and value.lower().startswith(("postgres://", "postgresql://")):
        return _append_sslmode_if_missing(value)

    return None


def is_configured() -> bool:
    """Return whether Postgres credentials are available."""

    return bool(get_database_url())


def _load_psycopg() -> Any:
    try:
        import psycopg
    except Exception as exc:  # pragma: no cover - import path depends on env
        raise CreditsDBError(
            "psycopg is required for AI billing, install psycopg[binary]"
        ) from exc

    return psycopg


@contextmanager
def connect() -> Iterator[Any]:
    """Yield a psycopg connection using configured env vars."""

    database_url = get_database_url()
    if not database_url:
        raise CreditsDBError("Postgres is not configured")

    psycopg = _load_psycopg()
    conn = psycopg.connect(database_url)
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema() -> None:
    """Create billing tables if they don't exist."""

    global _SCHEMA_READY

    if _schema_is_ready():
        return

    with _SCHEMA_LOCK:
        if _schema_is_ready():
            return

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS credit_accounts (
                        scope_type TEXT NOT NULL CHECK (scope_type IN ('user', 'chat')),
                        scope_id BIGINT NOT NULL,
                        balance INTEGER NOT NULL DEFAULT 0,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (scope_type, scope_id)
                    )
                    """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS onboarding_grants (
                        user_id BIGINT PRIMARY KEY,
                        credits INTEGER NOT NULL,
                        granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS star_payments (
                        telegram_payment_charge_id TEXT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        pack_id TEXT NOT NULL,
                        xtr_amount INTEGER NOT NULL,
                        credits_awarded INTEGER NOT NULL,
                        payload TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS credit_ledger (
                        id BIGSERIAL PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        actor_user_id BIGINT,
                        user_id BIGINT,
                        chat_id BIGINT,
                        amount INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_compaction_usage_tag
                    ON credit_ledger ((metadata->>'usage_tag'))
                    WHERE event_type = 'memory_compaction_settlement'
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_user_ai_settlements
                    ON credit_ledger (user_id, created_at DESC, id DESC)
                    WHERE event_type = 'ai_settlement_result'
                    """)
                cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_credit_ledger_unique_ai_settlement
                    ON credit_ledger (
                        user_id,
                        (metadata->>'settlement_id')
                    )
                    WHERE event_type = 'ai_settlement_result'
                      AND metadata ? 'settlement_id'
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_settlement_id
                    ON credit_ledger ((metadata->>'settlement_id'))
                    WHERE metadata ? 'settlement_id'
                    """)
                cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_credit_ledger_unique_ai_provider_segment
                    ON credit_ledger (
                        (metadata->>'operation_id'),
                        (metadata->>'segment_id')
                    )
                    WHERE event_type = 'ai_provider_usage'
                      AND metadata ? 'operation_id'
                      AND metadata ? 'segment_id'
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_user_charge_history
                    ON credit_ledger (user_id, id DESC)
                    WHERE event_type IN (
                        'ai_settlement_result',
                        'memory_compaction_settlement',
                        'ai_reserve'
                    )
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_user_charge_operations
                    ON credit_ledger (user_id, id DESC)
                    WHERE event_type IN (
                        'ai_settlement_result',
                        'memory_compaction_settlement',
                        'ai_reserve',
                        'ai_refund',
                        'ai_settlement_charge',
                        'ai_settlement_debt'
                    )
                    """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS
                        idx_credit_ledger_user_settlement_lookup
                    ON credit_ledger (
                        user_id,
                        (metadata->>'settlement_id'),
                        id DESC
                    )
                    WHERE metadata ? 'settlement_id'
                    """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS credit_schema_migrations (
                        name TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)
                _migrate_credit_amounts_to_units(cur)
                _migrate_credit_amounts_to_hundredths(cur)
            conn.commit()

        _SCHEMA_READY = True


def _migrate_credit_amounts_to_units(cur: Any) -> bool:
    """Scale legacy whole-credit rows into tenth-credit units once."""

    # Multiple bot instances may start together; the transaction lock elects one.
    cur.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (CREDIT_UNITS_MIGRATION_ADVISORY_LOCK_KEY,),
    )
    cur.execute(
        """
        INSERT INTO credit_schema_migrations (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        RETURNING name
        """,
        (CREDIT_UNITS_MIGRATION_NAME,),
    )
    inserted = cur.fetchone() is not None
    if not inserted:
        return False

    cur.execute(
        "UPDATE credit_accounts SET balance = balance * %s",
        (_LEGACY_WHOLE_TO_TENTHS_FACTOR,),
    )
    cur.execute(
        "UPDATE onboarding_grants SET credits = credits * %s",
        (_LEGACY_WHOLE_TO_TENTHS_FACTOR,),
    )
    cur.execute(
        "UPDATE star_payments SET credits_awarded = credits_awarded * %s",
        (_LEGACY_WHOLE_TO_TENTHS_FACTOR,),
    )
    cur.execute(
        "UPDATE credit_ledger SET amount = amount * %s",
        (_LEGACY_WHOLE_TO_TENTHS_FACTOR,),
    )
    return True


def _migrate_credit_amounts_to_hundredths(cur: Any) -> bool:
    """Scale stored credit amounts and legacy ledger metadata once."""

    cur.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (CREDIT_HUNDREDTHS_MIGRATION_ADVISORY_LOCK_KEY,),
    )
    cur.execute(
        """
        INSERT INTO credit_schema_migrations (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        RETURNING name
        """,
        (CREDIT_HUNDREDTHS_MIGRATION_NAME,),
    )
    inserted = cur.fetchone() is not None
    if not inserted:
        return False

    factor = _TENTHS_TO_HUNDREDTHS_FACTOR
    cur.execute("UPDATE credit_accounts SET balance = balance * %s", (factor,))
    cur.execute("UPDATE onboarding_grants SET credits = credits * %s", (factor,))
    cur.execute(
        "UPDATE star_payments SET credits_awarded = credits_awarded * %s",
        (factor,),
    )
    cur.execute("UPDATE credit_ledger SET amount = amount * %s", (factor,))
    cur.execute(
        """
        UPDATE credit_ledger
        SET metadata = (
            SELECT jsonb_object_agg(
                item.key,
                CASE
                    WHEN item.key LIKE '%%credit_units%%'
                         AND jsonb_typeof(item.value) = 'number'
                    THEN to_jsonb((item.value #>> '{}')::bigint * %s)
                    WHEN item.key = ANY(%s)
                         AND jsonb_typeof(item.value) = 'number'
                    THEN to_jsonb((item.value #>> '{}')::bigint * %s)
                    ELSE item.value
                END
            ) AS metadata
            FROM jsonb_each(credit_ledger.metadata) AS item
        ) || jsonb_build_object('credit_scale', %s)
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_each(credit_ledger.metadata) AS item
            WHERE jsonb_typeof(item.value) = 'number'
              AND (
                  item.key LIKE '%%credit_units%%'
                  OR item.key = ANY(%s)
              )
        )
        """,
        (
            factor,
            list(_LEGACY_METADATA_CREDIT_KEYS),
            CREDIT_SCALE,
            CREDIT_SCALE,
            list(_LEGACY_METADATA_CREDIT_KEYS),
        ),
    )
    return True


def _ensure_account(cur: Any, scope_type: ScopeType, scope_id: int) -> None:
    cur.execute(
        """
        INSERT INTO credit_accounts (scope_type, scope_id, balance)
        VALUES (%s, %s, 0)
        ON CONFLICT (scope_type, scope_id) DO NOTHING
        """,
        (scope_type, int(scope_id)),
    )


def _get_balance_for_update(cur: Any, scope_type: ScopeType, scope_id: int) -> int:
    _ensure_account(cur, scope_type, scope_id)
    # Serialize concurrent charges so two requests cannot spend one balance.
    cur.execute(
        """
        SELECT balance
        FROM credit_accounts
        WHERE scope_type = %s AND scope_id = %s
        FOR UPDATE
        """,
        (scope_type, int(scope_id)),
    )
    row = cur.fetchone()
    if not row:
        return 0
    return int(row[0])


def _get_user_and_chat_balances_for_update(
    cur: Any, user_id: int, chat_id: Optional[int]
) -> Tuple[int, int]:
    """Lock user first, then chat, for every two-account credit transaction."""

    user_balance = _get_balance_for_update(cur, "user", user_id)
    chat_balance = 0
    if chat_id is not None:
        chat_balance = _get_balance_for_update(cur, "chat", chat_id)
    return user_balance, chat_balance


def _is_retryable_credit_transaction_error(error: BaseException) -> bool:
    current: Optional[BaseException] = error
    while current is not None:
        if current.__class__.__name__ in _RETRYABLE_CREDIT_TRANSACTION_ERRORS:
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_credit_transaction(operation: Callable[[Any], T]) -> T:
    """Run one credits DB transaction, retrying Postgres concurrency aborts."""

    ensure_schema()
    for attempt in range(CREDIT_TRANSACTION_MAX_ATTEMPTS):
        try:
            with connect() as conn, conn.cursor() as cur:
                result = operation(cur)
                conn.commit()
                return result
        except Exception as error:
            if (
                attempt == CREDIT_TRANSACTION_MAX_ATTEMPTS - 1
                or not _is_retryable_credit_transaction_error(error)
            ):
                raise

    raise CreditsDBError("Credit transaction retry loop exited unexpectedly")


def _set_balance(cur: Any, scope_type: ScopeType, scope_id: int, balance: int) -> None:
    cur.execute(
        """
        UPDATE credit_accounts
        SET balance = %s, updated_at = NOW()
        WHERE scope_type = %s AND scope_id = %s
        """,
        (int(balance), scope_type, int(scope_id)),
    )


def _get_recent_onboarding_grant_counts(cur: Any) -> Tuple[int, int]:
    """Return onboarding grant counts for the last hour and day."""

    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 hour') AS hourly_count,
            COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 day') AS daily_count
        FROM onboarding_grants
        """)
    row = cur.fetchone()
    if not row:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def _should_deny_onboarding_grant(hourly_count: int, daily_count: int) -> bool:
    """Return whether onboarding should be denied due to recent overflow."""

    return (
        hourly_count >= ONBOARDING_MAX_GRANTS_PER_HOUR
        or daily_count >= ONBOARDING_MAX_GRANTS_PER_DAY
    )


def _has_existing_onboarding_grant(cur: Any, user_id: int) -> bool:
    """Return whether the user already received onboarding credits."""

    cur.execute(
        """
        SELECT 1
        FROM onboarding_grants
        WHERE user_id = %s
        """,
        (int(user_id),),
    )
    return cur.fetchone() is not None


def get_balance(scope_type: ScopeType, scope_id: int) -> int:
    """Return the current account balance."""

    ensure_schema()
    with connect() as conn:
        with conn.cursor() as cur:
            _ensure_account(cur, scope_type, scope_id)
            cur.execute(
                """
                SELECT balance
                FROM credit_accounts
                WHERE scope_type = %s AND scope_id = %s
                """,
                (scope_type, int(scope_id)),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        return 0
    return int(row[0])


def grant_onboarding_if_needed(user_id: int, credits: int) -> Tuple[bool, int]:
    """Grant onboarding credits once and return (granted, user_balance)."""

    def operation(cur: Any) -> Tuple[bool, int]:
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (ONBOARDING_GRANTS_ADVISORY_LOCK_KEY,),
        )
        user_balance = _get_balance_for_update(cur, "user", user_id)
        if _has_existing_onboarding_grant(cur, user_id):
            return False, int(user_balance)

        hourly_count, daily_count = _get_recent_onboarding_grant_counts(cur)

        if _should_deny_onboarding_grant(hourly_count, daily_count):
            cur.execute(
                """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    "onboarding_denied_overflow",
                    int(user_id),
                    int(user_id),
                    0,
                    json.dumps(
                        {
                            "credits": int(credits),
                            "hourly_count": int(hourly_count),
                            "daily_count": int(daily_count),
                            "hourly_limit": ONBOARDING_MAX_GRANTS_PER_HOUR,
                            "daily_limit": ONBOARDING_MAX_GRANTS_PER_DAY,
                        }
                    ),
                ),
            )
            return False, int(user_balance)

        cur.execute(
            """
            INSERT INTO onboarding_grants (user_id, credits)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO NOTHING
            RETURNING user_id
            """,
            (int(user_id), int(credits)),
        )
        granted = cur.fetchone() is not None

        if granted:
            user_balance += int(credits)
            _set_balance(cur, "user", user_id, user_balance)
            cur.execute(
                """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    "onboarding_grant",
                    int(user_id),
                    int(user_id),
                    int(credits),
                    json.dumps({"credits": int(credits)}),
                ),
            )
        return granted, int(user_balance)

    return _run_credit_transaction(operation)


def _ai_operation_is_settled(cur: Any, user_id: int, operation_id: str) -> bool:
    if not operation_id:
        return False
    cur.execute(
        """
        SELECT 1
        FROM credit_ledger
        WHERE user_id = %s
          AND event_type = 'ai_settlement_result'
          AND metadata->>'operation_id' = %s
        LIMIT 1
        """,
        (int(user_id), operation_id),
    )
    return cur.fetchone() is not None


def _existing_ai_charge_result(
    cur: Any,
    *,
    user_id: int,
    event_type: str,
    idempotency_key: str,
    operation_id: str,
    user_balance: int,
    chat_balance: int,
) -> Optional[Dict[str, Any]]:
    cur.execute(
        """
        SELECT amount, metadata->>'source'
        FROM credit_ledger
        WHERE user_id = %s
          AND event_type = %s
          AND metadata->>'idempotency_key' = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(user_id), event_type, idempotency_key),
    )
    existing = cur.fetchone()
    if existing is None:
        return None

    rejection_reason: Optional[str] = None
    if event_type == "ai_reserve":
        cur.execute(
            """
            SELECT 1
            FROM credit_ledger
            WHERE user_id = %s
              AND event_type = 'ai_refund'
              AND metadata->>'settlement_id' = %s
            LIMIT 1
            """,
            (int(user_id), idempotency_key),
        )
        if cur.fetchone() is not None:
            rejection_reason = "reservation_refunded"
        elif _ai_operation_is_settled(cur, user_id, operation_id):
            rejection_reason = "operation_settled"

    if rejection_reason:
        return {
            "ok": False,
            "applied": False,
            "reason": rejection_reason,
            "source": None,
            "amount": 0,
            "user_balance": user_balance,
            "chat_balance": chat_balance,
            "user_balance_credit_units": user_balance,
            "chat_balance_credit_units": chat_balance,
        }
    return {
        "ok": True,
        "applied": False,
        "source": str(existing[1] or "user"),
        "amount": max(0, -int(existing[0] or 0)),
        "user_balance": user_balance,
        "chat_balance": chat_balance,
        "user_balance_credit_units": user_balance,
        "chat_balance_credit_units": chat_balance,
    }


def charge_ai_credits(
    user_id: int,
    chat_id: Optional[int],
    amount: int,
    *,
    event_type: str = "ai_charge",
    metadata: Optional[Mapping[str, Any]] = None,
    source: Optional[ScopeType] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Charge credits for an AI interaction.

    User balance is consumed first unless an existing interaction pinned its payer.
    """

    charge_amount = int(amount)
    normalized_event_type = str(event_type or "ai_charge")
    metadata_dict = dict(metadata or {})
    operation_id = str(metadata_dict.get("operation_id") or "").strip()
    normalized_source: Optional[ScopeType] = (
        "chat" if source == "chat" else "user" if source == "user" else None
    )
    normalized_key = str(idempotency_key or "").strip() or None
    if normalized_key:
        metadata_dict["idempotency_key"] = normalized_key

    def operation(cur: Any) -> Dict[str, Any]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )

        if normalized_key:
            existing_result = _existing_ai_charge_result(
                cur,
                user_id=user_id,
                event_type=normalized_event_type,
                idempotency_key=normalized_key,
                operation_id=operation_id,
                user_balance=user_balance,
                chat_balance=chat_balance,
            )
            if existing_result is not None:
                return existing_result

        if normalized_event_type == "ai_reserve":
            if _ai_operation_is_settled(cur, user_id, operation_id):
                return {
                    "ok": False,
                    "applied": False,
                    "reason": "operation_settled",
                    "source": None,
                    "amount": 0,
                    "user_balance": user_balance,
                    "chat_balance": chat_balance,
                    "user_balance_credit_units": user_balance,
                    "chat_balance_credit_units": chat_balance,
                }

        if normalized_source in {None, "user"} and user_balance >= charge_amount:
            updated_user_balance = user_balance - charge_amount
            _set_balance(cur, "user", user_id, updated_user_balance)
            cur.execute(
                """
                    INSERT INTO credit_ledger (
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                (
                    normalized_event_type,
                    int(user_id),
                    int(user_id),
                    int(chat_id) if chat_id is not None else None,
                    -charge_amount,
                    json.dumps({"source": "user", **metadata_dict}),
                ),
            )
            return {
                "ok": True,
                "applied": True,
                "source": "user",
                "amount": charge_amount,
                "user_balance": updated_user_balance,
                "chat_balance": chat_balance,
                "user_balance_credit_units": updated_user_balance,
                "chat_balance_credit_units": chat_balance,
            }

        if (
            normalized_source in {None, "chat"}
            and chat_id is not None
            and chat_balance >= charge_amount
        ):
            updated_chat_balance = chat_balance - charge_amount
            _set_balance(cur, "chat", chat_id, updated_chat_balance)
            cur.execute(
                """
                    INSERT INTO credit_ledger (
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                (
                    normalized_event_type,
                    int(user_id),
                    int(user_id),
                    int(chat_id),
                    -charge_amount,
                    json.dumps({"source": "chat", **metadata_dict}),
                ),
            )
            return {
                "ok": True,
                "applied": True,
                "source": "chat",
                "amount": charge_amount,
                "user_balance": user_balance,
                "chat_balance": updated_chat_balance,
                "user_balance_credit_units": user_balance,
                "chat_balance_credit_units": updated_chat_balance,
            }

        return {
            "ok": False,
            "applied": False,
            "source": None,
            "amount": 0,
            "user_balance": user_balance,
            "chat_balance": chat_balance,
            "user_balance_credit_units": user_balance,
            "chat_balance_credit_units": chat_balance,
        }

    return _run_credit_transaction(operation)


def record_ai_provider_usage(
    user_id: int,
    chat_id: Optional[int],
    operation_id: str,
    segment_id: str,
    segment: Mapping[str, Any],
) -> bool:
    """Persist one provider result before the next external call starts."""

    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO credit_ledger (
                event_type,
                actor_user_id,
                user_id,
                chat_id,
                amount,
                metadata
            )
            VALUES ('ai_provider_usage', %s, %s, %s, 0, %s::jsonb)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                int(user_id),
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                json.dumps(
                    {
                        "operation_id": str(operation_id),
                        "segment_id": str(segment_id),
                        "segment": dict(segment),
                    },
                    default=str,
                ),
            ),
        )
        inserted = cur.fetchone() is not None
        conn.commit()
    return inserted


def update_ai_provider_usage(
    operation_id: str,
    segment_id: str,
    segment: Mapping[str, Any],
) -> bool:
    """Replace one pending provider segment with reconciled usage."""

    ensure_schema()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE credit_ledger
            SET metadata = jsonb_set(metadata, '{segment}', %s::jsonb)
            WHERE event_type = 'ai_provider_usage'
              AND metadata->>'operation_id' = %s
              AND metadata->>'segment_id' = %s
            RETURNING id
            """,
            (
                json.dumps(dict(segment), default=str),
                str(operation_id),
                str(segment_id),
            ),
        )
        updated = cur.fetchone() is not None
        conn.commit()
    return updated


def list_unsettled_ai_operations(limit: int = 100) -> List[Dict[str, Any]]:
    """Return durable provider usage that still needs final settlement."""

    ensure_schema()
    normalized_limit = max(1, min(int(limit or 100), 500))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH pending AS (
                SELECT
                    ledger.metadata->>'operation_id' AS operation_id,
                    MIN(ledger.user_id) AS user_id,
                    MIN(ledger.chat_id) AS chat_id,
                    GREATEST(0, COALESCE(SUM(-ledger.amount), 0)) AS authorized,
                    MIN(ledger.metadata->>'source') AS source,
                    MIN(ledger.created_at) AS created_at,
                    MAX(ledger.created_at) AS hold_activity_at,
                    (ARRAY_AGG(ledger.metadata ORDER BY ledger.id)
                        FILTER (WHERE ledger.event_type = 'ai_reserve'))[1]
                        AS reserve_metadata
                FROM credit_ledger AS ledger
                WHERE ledger.event_type IN ('ai_reserve', 'ai_refund')
                  AND ledger.metadata ? 'operation_id'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM credit_ledger AS settled
                      WHERE settled.event_type = 'ai_settlement_result'
                        AND settled.metadata->>'operation_id'
                            = ledger.metadata->>'operation_id'
                  )
                GROUP BY ledger.metadata->>'operation_id'
            )
            SELECT
                pending.operation_id,
                pending.user_id,
                pending.chat_id,
                pending.authorized,
                pending.source,
                pending.created_at,
                GREATEST(
                    pending.hold_activity_at,
                    COALESCE(
                        (
                            SELECT MAX(usage.created_at)
                            FROM credit_ledger AS usage
                            WHERE usage.event_type = 'ai_provider_usage'
                              AND usage.metadata->>'operation_id' = pending.operation_id
                        ),
                        pending.hold_activity_at
                    )
                ) AS last_activity_at,
                pending.reserve_metadata,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'segment_id', usage.metadata->>'segment_id',
                                'segment', usage.metadata->'segment'
                            )
                            ORDER BY usage.id
                        )
                        FROM credit_ledger AS usage
                        WHERE usage.event_type = 'ai_provider_usage'
                          AND usage.metadata->>'operation_id' = pending.operation_id
                    ),
                    '[]'::jsonb
                ) AS segments
            FROM pending
            WHERE pending.authorized > 0
               OR EXISTS (
                    SELECT 1
                    FROM credit_ledger AS usage
                    WHERE usage.event_type = 'ai_provider_usage'
                      AND usage.metadata->>'operation_id' = pending.operation_id
                )
            ORDER BY pending.created_at
            LIMIT %s
            """,
            (normalized_limit,),
        )
        rows = cur.fetchall() or []

    return [
        {
            "operation_id": str(row[0]),
            "user_id": int(row[1]),
            "chat_id": int(row[2]) if row[2] is not None else None,
            "authorized_credit_units": int(row[3] or 0),
            "source": "chat" if row[4] == "chat" else "user",
            "created_at": row[5],
            "last_activity_at": row[6],
            "reserve_metadata": dict(row[7] or {}),
            "segments": list(row[8] or []),
        }
        for row in rows
    ]


def settle_ai_operation_once(
    user_id: int,
    chat_id: Optional[int],
    operation_id: str,
    actual_credit_units: int,
    *,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Settle all holds for one interaction atomically and exactly once."""

    actual = max(0, int(actual_credit_units or 0))
    normalized_operation_id = str(operation_id)

    def operation(cur: Any) -> Dict[str, Any]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )
        cur.execute(
            """
            SELECT 1
            FROM credit_ledger
            WHERE user_id = %s
              AND event_type = 'ai_settlement_result'
              AND metadata->>'operation_id' = %s
            LIMIT 1
            """,
            (int(user_id), normalized_operation_id),
        )
        if cur.fetchone() is not None:
            return {
                "applied": False,
                "user_balance": user_balance,
                "chat_balance": chat_balance,
            }

        cur.execute(
            """
            SELECT
                COALESCE(SUM(-amount), 0),
                COUNT(DISTINCT metadata->>'source'),
                MIN(metadata->>'source')
            FROM credit_ledger
            WHERE user_id = %s
              AND event_type IN ('ai_reserve', 'ai_refund')
              AND metadata->>'operation_id' = %s
            """,
            (int(user_id), normalized_operation_id),
        )
        hold_row = cur.fetchone() or (0, 0, None)
        authorized = max(0, int(hold_row[0] or 0))
        payer_count = int(hold_row[1] or 0)
        payer: ScopeType = "chat" if hold_row[2] == "chat" else "user"
        if payer_count > 1:
            raise CreditsDBError("AI operation has more than one payer")
        if payer == "chat" and chat_id is None:
            raise CreditsDBError("chat-funded AI operation requires chat_id")

        adjustment = authorized - actual
        if payer == "chat":
            assert chat_id is not None
            chat_balance += adjustment
            _set_balance(cur, "chat", chat_id, chat_balance)
        else:
            user_balance += adjustment
            _set_balance(cur, "user", user_id, user_balance)

        settlement_metadata = {
            **dict(metadata),
            "operation_id": normalized_operation_id,
            "source": payer,
            "payer_scope": payer,
            "reserved_credit_units_total": authorized,
            "settled_credit_units": actual,
            "refunded_credit_units": max(0, adjustment),
            "debt_applied_credit_units": max(0, -adjustment),
            "charged_credit_units_total": actual,
        }
        if adjustment:
            cur.execute(
                """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    "ai_refund" if adjustment > 0 else "ai_settlement_debt",
                    int(user_id),
                    int(user_id),
                    int(chat_id) if chat_id is not None else None,
                    int(adjustment),
                    json.dumps(settlement_metadata, default=str),
                ),
            )
        cur.execute(
            """
            INSERT INTO credit_ledger (
                event_type,
                actor_user_id,
                user_id,
                chat_id,
                amount,
                metadata
            )
            VALUES ('ai_settlement_result', %s, %s, %s, 0, %s::jsonb)
            """,
            (
                int(user_id),
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                json.dumps(settlement_metadata, default=str),
            ),
        )
        return {
            "applied": True,
            "source": payer,
            "authorized_credit_units": authorized,
            "actual_credit_units": actual,
            "refunded_credit_units": max(0, adjustment),
            "debt_applied_credit_units": max(0, -adjustment),
            "user_balance": user_balance,
            "chat_balance": chat_balance,
        }

    return _run_credit_transaction(operation)


def charge_chat_ai_credits(
    chat_id: int,
    amount: int,
    *,
    event_type: str = "ai_reserve",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Charge credits directly to a chat balance for chat-owned automation."""

    charge_amount = int(amount)
    metadata_dict = dict(metadata or {})

    def operation(cur: Any) -> Dict[str, Any]:
        chat_balance = _get_balance_for_update(cur, "chat", chat_id)
        if chat_balance >= charge_amount:
            chat_balance -= charge_amount
            _set_balance(cur, "chat", chat_id, chat_balance)
            cur.execute(
                """
                    INSERT INTO credit_ledger (
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                (
                    str(event_type or "ai_reserve"),
                    None,
                    None,
                    int(chat_id),
                    -charge_amount,
                    json.dumps({"source": "chat", **metadata_dict}),
                ),
            )
            return {
                "ok": True,
                "source": "chat",
                "chat_balance": chat_balance,
                "chat_balance_credit_units": chat_balance,
            }

        return {
            "ok": False,
            "source": None,
            "chat_balance": chat_balance,
            "chat_balance_credit_units": chat_balance,
        }

    return _run_credit_transaction(operation)


def refund_chat_ai_credits(
    chat_id: int,
    amount: int,
    *,
    event_type: str = "ai_refund",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """Refund a chat-owned AI reservation."""

    refund_amount = int(amount)
    metadata_dict = dict(metadata or {})

    def operation(cur: Any) -> Dict[str, int]:
        chat_balance = _get_balance_for_update(cur, "chat", chat_id) + refund_amount
        _set_balance(cur, "chat", chat_id, chat_balance)
        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                str(event_type or "ai_refund"),
                None,
                None,
                int(chat_id),
                refund_amount,
                json.dumps({"source": "chat", **metadata_dict}),
            ),
        )
        return {"chat_balance": int(chat_balance)}

    return _run_credit_transaction(operation)


def apply_chat_ai_debt(
    chat_id: int,
    amount: int,
    *,
    event_type: str = "ai_settlement_debt",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """Apply settlement debt to a chat-owned automation charge."""

    debt_amount = int(amount)
    metadata_dict = dict(metadata or {})

    def operation(cur: Any) -> Dict[str, int]:
        chat_balance = _get_balance_for_update(cur, "chat", chat_id) - debt_amount
        _set_balance(cur, "chat", chat_id, chat_balance)
        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                str(event_type or "ai_settlement_debt"),
                None,
                None,
                int(chat_id),
                -debt_amount,
                json.dumps({"source": "chat", **metadata_dict}),
            ),
        )
        return {"chat_balance": int(chat_balance)}

    return _run_credit_transaction(operation)


def refund_ai_charge(
    user_id: int,
    chat_id: Optional[int],
    amount: int,
    source: ScopeType,
    *,
    event_type: str = "ai_refund",
    metadata: Optional[Mapping[str, Any]] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Refund a previously charged AI credit."""

    refund_amount = int(amount)
    metadata_dict = dict(metadata or {})
    normalized_key = str(idempotency_key or "").strip() or None
    if normalized_key:
        metadata_dict["idempotency_key"] = normalized_key

    def operation(cur: Any) -> Dict[str, Any]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )
        if normalized_key:
            cur.execute(
                """
                SELECT 1
                FROM credit_ledger
                WHERE user_id = %s
                  AND event_type = %s
                  AND metadata->>'idempotency_key' = %s
                LIMIT 1
                """,
                (int(user_id), str(event_type or "ai_refund"), normalized_key),
            )
            if cur.fetchone() is not None:
                return {
                    "user_balance": int(user_balance),
                    "chat_balance": int(chat_balance),
                }

        operation_id = str(metadata_dict.get("operation_id") or "").strip()
        if _ai_operation_is_settled(cur, user_id, operation_id):
            return {
                "applied": False,
                "reason": "operation_settled",
                "user_balance": int(user_balance),
                "chat_balance": int(chat_balance),
            }

        if source == "chat" and chat_id is not None:
            updated_chat_balance = chat_balance + refund_amount
            _set_balance(cur, "chat", chat_id, updated_chat_balance)
            cur.execute(
                """
                    INSERT INTO credit_ledger (
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                (
                    str(event_type or "ai_refund"),
                    int(user_id),
                    int(user_id),
                    int(chat_id),
                    refund_amount,
                    json.dumps({"source": "chat", **metadata_dict}),
                ),
            )
            return {
                "user_balance": int(user_balance),
                "chat_balance": int(updated_chat_balance),
            }

        updated_user_balance = user_balance + refund_amount
        _set_balance(cur, "user", user_id, updated_user_balance)
        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                str(event_type or "ai_refund"),
                int(user_id),
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                refund_amount,
                json.dumps({"source": "user", **metadata_dict}),
            ),
        )
        return {
            "user_balance": int(updated_user_balance),
            "chat_balance": int(chat_balance),
        }

    return _run_credit_transaction(operation)


def settle_ai_reservation_once(
    user_id: int,
    chat_id: Optional[int],
    source: ScopeType,
    reserved_credit_units: int,
    actual_credit_units: int,
    usage_tag: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Settle one reservation exactly once in the credits transaction.

    Locking the payer account serializes concurrent attempts. The settlement
    ledger row is the durable idempotency record, so a worker restart cannot
    apply the same refund or extra charge twice.
    """

    reserved = max(0, int(reserved_credit_units or 0))
    actual = max(0, int(actual_credit_units or 0))
    normalized_source: ScopeType = "chat" if source == "chat" else "user"
    metadata_dict = dict(metadata or {})

    if normalized_source == "chat" and chat_id is None:
        raise CreditsDBError("chat-funded settlement requires chat_id")
    normalized_chat_id = int(chat_id) if chat_id is not None else None

    def operation(cur: Any) -> Dict[str, Any]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )
        cur.execute(
            """
            SELECT 1
            FROM credit_ledger
            WHERE event_type = 'memory_compaction_settlement'
              AND metadata->>'usage_tag' = %s
            LIMIT 1
            """,
            (str(usage_tag),),
        )
        if cur.fetchone() is not None:
            return {
                "applied": False,
                "user_balance": int(user_balance),
                "chat_balance": int(chat_balance),
            }

        adjustment = reserved - actual
        if normalized_source == "chat":
            assert normalized_chat_id is not None
            chat_balance += adjustment
            _set_balance(cur, "chat", normalized_chat_id, chat_balance)
        else:
            user_balance += adjustment
            _set_balance(cur, "user", user_id, user_balance)

        settlement_metadata = {
            "source": normalized_source,
            "usage_tag": str(usage_tag),
            "reserved_credit_units": reserved,
            "actual_credit_units": actual,
            "adjustment_credit_units": adjustment,
            **metadata_dict,
        }
        cur.execute(
            """
            INSERT INTO credit_ledger (
                event_type,
                actor_user_id,
                user_id,
                chat_id,
                amount,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                "memory_compaction_settlement",
                int(user_id),
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                int(adjustment),
                json.dumps(settlement_metadata),
            ),
        )
        return {
            "applied": True,
            "adjustment_credit_units": int(adjustment),
            "user_balance": int(user_balance),
            "chat_balance": int(chat_balance),
        }

    return _run_credit_transaction(operation)


def apply_ai_debt(
    user_id: int,
    chat_id: Optional[int],
    amount: int,
    source: ScopeType,
    *,
    event_type: str = "ai_settlement_debt",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """Apply an AI debt, allowing the selected balance to go negative."""

    debt_amount = int(amount)
    metadata_dict = dict(metadata or {})

    def operation(cur: Any) -> Dict[str, int]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )

        if source == "chat" and chat_id is not None:
            updated_chat_balance = chat_balance - debt_amount
            _set_balance(cur, "chat", chat_id, updated_chat_balance)
            cur.execute(
                """
                    INSERT INTO credit_ledger (
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                (
                    str(event_type or "ai_settlement_debt"),
                    int(user_id),
                    int(user_id),
                    int(chat_id),
                    -debt_amount,
                    json.dumps({"source": "chat", **metadata_dict}),
                ),
            )
            return {
                "user_balance": int(user_balance),
                "chat_balance": int(updated_chat_balance),
            }

        updated_user_balance = user_balance - debt_amount
        _set_balance(cur, "user", user_id, updated_user_balance)
        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                str(event_type or "ai_settlement_debt"),
                int(user_id),
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                -debt_amount,
                json.dumps({"source": "user", **metadata_dict}),
            ),
        )

        return {
            "user_balance": int(updated_user_balance),
            "chat_balance": int(chat_balance),
        }

    return _run_credit_transaction(operation)


def transfer_user_to_chat(user_id: int, chat_id: int, amount: int) -> Dict[str, Any]:
    """Transfer credits from personal balance to group balance."""

    transfer_amount = int(amount)

    def operation(cur: Any) -> Dict[str, Any]:
        user_balance, chat_balance = _get_user_and_chat_balances_for_update(
            cur, user_id, chat_id
        )

        if user_balance < transfer_amount:
            return {
                "ok": False,
                "error": "insufficient",
                "user_balance": user_balance,
                "chat_balance": chat_balance,
            }

        user_balance -= transfer_amount
        chat_balance += transfer_amount
        _set_balance(cur, "user", user_id, user_balance)
        _set_balance(cur, "chat", chat_id, chat_balance)

        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                "transfer_user_to_chat",
                int(user_id),
                int(user_id),
                int(chat_id),
                -transfer_amount,
                json.dumps({"direction": "user_to_chat"}),
            ),
        )
        cur.execute(
            """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
            (
                "transfer_user_to_chat",
                int(user_id),
                int(user_id),
                int(chat_id),
                transfer_amount,
                json.dumps({"direction": "chat_from_user"}),
            ),
        )

        return {
            "ok": True,
            "error": None,
            "user_balance": user_balance,
            "chat_balance": chat_balance,
        }

    return _run_credit_transaction(operation)


def mint_user_credits(
    user_id: int, amount: int, actor_user_id: Optional[int] = None
) -> Dict[str, int]:
    """Mint credits to a user account and return the updated balance."""

    mint_amount = int(amount)
    actor_id = int(actor_user_id) if actor_user_id is not None else int(user_id)

    def operation(cur: Any) -> Dict[str, int]:
        user_balance = _get_balance_for_update(cur, "user", user_id)
        user_balance += mint_amount
        _set_balance(cur, "user", user_id, user_balance)

        cur.execute(
            """
            INSERT INTO credit_ledger (
                event_type,
                actor_user_id,
                user_id,
                amount,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            (
                "printcredits",
                actor_id,
                int(user_id),
                mint_amount,
                json.dumps({"source": "admin_command"}),
            ),
        )
        return {"user_balance": int(user_balance)}

    return _run_credit_transaction(operation)


def record_ai_settlement_result(
    user_id: int,
    chat_id: Optional[int],
    *,
    actor_user_id: Optional[int] = None,
    event_type: str = "ai_settlement_result",
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist a non-monetary AI settlement audit event."""

    actor_id = int(actor_user_id) if actor_user_id is not None else int(user_id)
    metadata_dict = dict(metadata or {})

    def operation(cur: Any) -> None:
        cur.execute(
            """
            INSERT INTO credit_ledger (
                event_type,
                actor_user_id,
                user_id,
                chat_id,
                amount,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            (
                str(event_type or "ai_settlement_result"),
                actor_id,
                int(user_id),
                int(chat_id) if chat_id is not None else None,
                0,
                json.dumps(metadata_dict),
            ),
        )

    _run_credit_transaction(operation)


def _credit_ledger_rows_to_dicts(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row[6]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        elif not isinstance(metadata, Mapping):
            metadata = {}
        results.append(
            {
                "id": int(row[0]),
                "event_type": str(row[1]),
                "actor_user_id": int(row[2]) if row[2] is not None else None,
                "user_id": int(row[3]) if row[3] is not None else None,
                "chat_id": int(row[4]) if row[4] is not None else None,
                "amount": int(row[5]),
                "metadata": dict(metadata),
                "created_at": row[7],
            }
        )
    return results


def list_recent_ai_settlement_results(limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent AI settlement audit events ordered newest first."""

    ensure_schema()
    normalized_limit = max(1, min(int(limit or 10), 50))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
                SELECT
                    id,
                    event_type,
                    actor_user_id,
                    user_id,
                    chat_id,
                    amount,
                    metadata,
                    created_at
                FROM credit_ledger
                WHERE event_type = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
            ("ai_settlement_result", normalized_limit),
        )
        rows = cur.fetchall() or []

    return _credit_ledger_rows_to_dicts(rows)


def list_user_ai_charge_page(
    user_id: int,
    *,
    limit: int = 10,
    cursor_id: Optional[int] = None,
    direction: Literal["older", "newer"] = "older",
) -> Dict[str, Any]:
    """Return one user's AI charges grouped by originating Telegram message."""

    ensure_schema()
    normalized_limit = max(1, min(int(limit or 10), 20))
    normalized_cursor_id = int(cursor_id) if cursor_id is not None else None
    normalized_direction: Literal["older", "newer"] = (
        "newer" if direction == "newer" else "older"
    )

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
                WITH user_ledger AS (
                    SELECT *
                    FROM credit_ledger
                    WHERE user_id = %s
                      AND event_type IN (
                          'ai_settlement_result',
                          'memory_compaction_settlement',
                          'ai_reserve',
                          'ai_refund',
                          'ai_settlement_charge',
                          'ai_settlement_debt'
                      )
                ),
                finalized_ids AS (
                    SELECT metadata->>'settlement_id' AS settlement_id
                    FROM user_ledger
                    WHERE event_type IN (
                        'ai_settlement_result',
                        'memory_compaction_settlement'
                    )
                      AND metadata ? 'settlement_id'
                    UNION
                    SELECT settlement_id.value
                    FROM user_ledger
                    CROSS JOIN LATERAL jsonb_array_elements_text(
                        CASE
                            WHEN jsonb_typeof(metadata->'settlement_ids') = 'array'
                            THEN metadata->'settlement_ids'
                            ELSE '[]'::jsonb
                        END
                    ) AS settlement_id(value)
                    WHERE event_type = 'ai_settlement_result'
                ),
                finalized_operations AS (
                    SELECT
                        id,
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata,
                        created_at
                    FROM user_ledger
                    WHERE event_type IN (
                        'ai_settlement_result',
                        'memory_compaction_settlement'
                    )
                ),
                pending_reservations AS (
                    SELECT DISTINCT ON (metadata->>'settlement_id')
                        id,
                        event_type,
                        actor_user_id,
                        user_id,
                        chat_id,
                        amount,
                        metadata,
                        created_at
                    FROM user_ledger AS reserve
                    WHERE event_type = 'ai_reserve'
                      AND metadata ? 'settlement_id'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM finalized_ids
                          WHERE finalized_ids.settlement_id
                              = reserve.metadata->>'settlement_id'
                      )
                    ORDER BY metadata->>'settlement_id', id DESC
                ),
                pending_operations AS (
                    SELECT
                        reserve.id,
                        reserve.event_type,
                        reserve.actor_user_id,
                        reserve.user_id,
                        reserve.chat_id,
                        -GREATEST(0, -totals.net_amount) AS amount,
                        reserve.metadata || jsonb_build_object(
                            'charged_credit_units_total',
                                GREATEST(0, -totals.net_amount),
                            'billing_pending', TRUE,
                            'payer_scope',
                                CASE
                                    WHEN totals.user_paid > 0
                                         AND totals.chat_paid > 0
                                    THEN 'mixed'
                                    WHEN totals.chat_paid > 0
                                    THEN 'chat'
                                    ELSE 'user'
                                END,
                            'payer_breakdown', jsonb_build_array(
                                jsonb_build_object(
                                    'scope', 'user',
                                    'credit_units', totals.user_paid
                                ),
                                jsonb_build_object(
                                    'scope', 'chat',
                                    'credit_units', totals.chat_paid
                                )
                            )
                        ) AS metadata,
                        reserve.created_at
                    FROM pending_reservations AS reserve
                    CROSS JOIN LATERAL (
                        SELECT
                            COALESCE(SUM(mutation.amount), 0) AS net_amount,
                            GREATEST(
                                0,
                                COALESCE(
                                    SUM(-mutation.amount) FILTER (
                                        WHERE mutation.metadata->>'source' = 'user'
                                    ),
                                    0
                                )
                            ) AS user_paid,
                            GREATEST(
                                0,
                                COALESCE(
                                    SUM(-mutation.amount) FILTER (
                                        WHERE mutation.metadata->>'source' = 'chat'
                                    ),
                                    0
                                )
                            ) AS chat_paid
                        FROM user_ledger AS mutation
                        WHERE mutation.event_type IN (
                            'ai_reserve',
                            'ai_refund',
                            'ai_settlement_charge',
                            'ai_settlement_debt'
                        )
                          AND mutation.metadata->>'settlement_id'
                              = reserve.metadata->>'settlement_id'
                    ) AS totals
                ),
                operations AS (
                    SELECT * FROM finalized_operations
                    UNION ALL
                    SELECT * FROM pending_operations
                ),
                grouped_operations AS (
                    SELECT
                        operations.*,
                        CONCAT(
                            COALESCE(
                                NULLIF(metadata->>'origin_chat_id', ''),
                                NULLIF(
                                    SPLIT_PART(
                                        metadata->>'settlement_id',
                                        ':',
                                        2
                                    ),
                                    ''
                                ),
                                chat_id::text,
                                ''
                            ),
                            ':',
                            COALESCE(
                                NULLIF(metadata->>'message_id', ''),
                                'ledger:' || id::text
                            )
                        ) AS group_key
                    FROM operations
                ),
                charge_groups AS (
                    SELECT
                        group_key,
                        MIN(id) AS group_cursor,
                        MIN(created_at) AS group_created_at
                    FROM grouped_operations
                    GROUP BY group_key
                ),
                page_groups AS (
                    SELECT group_key, group_cursor, group_created_at
                    FROM charge_groups
                    WHERE %s::bigint IS NULL
                       OR (%s = 'older' AND group_cursor < %s)
                       OR (%s = 'newer' AND group_cursor > %s)
                    ORDER BY
                        CASE WHEN %s = 'newer' THEN group_cursor END ASC,
                        CASE WHEN %s = 'older' THEN group_cursor END DESC
                    LIMIT %s
                )
                SELECT
                    operation.id,
                    operation.event_type,
                    operation.actor_user_id,
                    operation.user_id,
                    operation.chat_id,
                    operation.amount,
                    operation.metadata,
                    operation.created_at,
                    page.group_key,
                    page.group_cursor,
                    page.group_created_at
                FROM page_groups AS page
                JOIN grouped_operations AS operation USING (group_key)
                ORDER BY
                    CASE WHEN %s = 'newer' THEN page.group_cursor END ASC,
                    CASE WHEN %s = 'older' THEN page.group_cursor END DESC,
                    operation.id DESC
                """,
            (
                int(user_id),
                normalized_cursor_id,
                normalized_direction,
                normalized_cursor_id,
                normalized_direction,
                normalized_cursor_id,
                normalized_direction,
                normalized_direction,
                normalized_limit + 1,
                normalized_direction,
                normalized_direction,
            ),
        )
        rows = cur.fetchall() or []

    grouped: Dict[str, Dict[str, Any]] = {}
    group_order: List[str] = []
    for row in rows:
        group_key = str(row[8])
        if group_key not in grouped:
            group_order.append(group_key)
            grouped[group_key] = {
                "cursor_id": int(row[9]),
                "created_at": row[10],
                "entries": [],
            }
        grouped[group_key]["entries"].extend(_credit_ledger_rows_to_dicts([row]))

    has_extra = len(group_order) > normalized_limit
    selected_order = group_order[:normalized_limit]
    if normalized_direction == "newer":
        selected_order.reverse()
    groups = [grouped[key] for key in selected_order]
    return {
        "groups": groups,
        "has_newer": (
            has_extra if normalized_direction == "newer" else cursor_id is not None
        ),
        "has_older": (
            cursor_id is not None if normalized_direction == "newer" else has_extra
        ),
        "newer_cursor": int(groups[0]["cursor_id"]) if groups else None,
        "older_cursor": int(groups[-1]["cursor_id"]) if groups else None,
    }


def purge_expired_ai_ledger_events(
    retention_days: int = AI_LEDGER_RETENTION_DAYS,
) -> Dict[str, Any]:
    """Delete AI ledger events older than the retention window."""

    normalized_retention_days = max(1, int(retention_days or 7))

    def operation(cur: Any) -> Dict[str, Any]:
        cur.execute(
            """
            DELETE FROM credit_ledger
            WHERE event_type = ANY(%s)
              AND created_at < NOW() - (%s * INTERVAL '1 day')
            """,
            (list(AI_LEDGER_EVENT_TYPES), normalized_retention_days),
        )
        return {
            "deleted_rows": int(cur.rowcount or 0),
            "retention_days": normalized_retention_days,
        }

    return _run_credit_transaction(operation)


def record_star_payment(
    telegram_payment_charge_id: str,
    user_id: int,
    pack_id: str,
    xtr_amount: int,
    credits_awarded: int,
    payload: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist successful payment and credit the user idempotently."""

    def operation(cur: Any) -> Dict[str, Any]:
        cur.execute(
            """
            INSERT INTO star_payments (
                telegram_payment_charge_id,
                user_id,
                pack_id,
                xtr_amount,
                credits_awarded,
                payload
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (telegram_payment_charge_id) DO NOTHING
            RETURNING telegram_payment_charge_id
            """,
            (
                str(telegram_payment_charge_id),
                int(user_id),
                str(pack_id),
                int(xtr_amount),
                int(credits_awarded),
                str(payload) if payload else None,
            ),
        )
        inserted = cur.fetchone() is not None

        # A repeated Telegram delivery returns the balance without crediting twice.
        user_balance = _get_balance_for_update(cur, "user", user_id)
        if inserted:
            user_balance += int(credits_awarded)
            _set_balance(cur, "user", user_id, user_balance)
            cur.execute(
                """
                INSERT INTO credit_ledger (
                    event_type,
                    actor_user_id,
                    user_id,
                    amount,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    "topup",
                    int(user_id),
                    int(user_id),
                    int(credits_awarded),
                    json.dumps(
                        {
                            "pack_id": str(pack_id),
                            "xtr_amount": int(xtr_amount),
                            "charge_id": str(telegram_payment_charge_id),
                        }
                    ),
                ),
            )
        return {"inserted": inserted, "user_balance": int(user_balance)}

    return _run_credit_transaction(operation)
