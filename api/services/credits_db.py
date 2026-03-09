"""Postgres-backed credits and payments storage for AI billing."""

from __future__ import annotations

from contextlib import contextmanager
from os import environ
from threading import Lock
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import json

ScopeType = Literal["user", "chat"]

_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False

ONBOARDING_MAX_GRANTS_PER_HOUR = 4
ONBOARDING_MAX_GRANTS_PER_DAY = 16
ONBOARDING_GRANTS_ADVISORY_LOCK_KEY = 48_610_001


class CreditsDBError(RuntimeError):
    """Raised when credits persistence cannot be completed."""


def _append_sslmode_if_missing(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "sslmode" not in query:
        query["sslmode"] = "require"
    return urlunparse(parsed._replace(query=urlencode(query)))


def get_database_url() -> Optional[str]:
    """Return the Postgres connection URL from env vars."""

    direct_keys = (
        "DATABASE_URL",
        "POSTGRES_URL",
        "POSTGRES_PRISMA_URL",
        "POSTGRES_URL_NON_POOLING",
        "NEON_DATABASE_URL",
    )

    for key in direct_keys:
        value = str(environ.get(key) or "").strip()
        if value and value.lower().startswith(("postgres://", "postgresql://")):
            return _append_sslmode_if_missing(value)

    host = str(environ.get("PGHOST") or "").strip()
    database = str(environ.get("PGDATABASE") or "").strip()
    user = str(environ.get("PGUSER") or "").strip()
    password = str(environ.get("PGPASSWORD") or "").strip()
    port = str(environ.get("PGPORT") or "5432").strip() or "5432"

    if not host or not database or not user:
        return None

    return _append_sslmode_if_missing(
        f"postgresql://{user}:{password}@{host}:{port}/{database}"
    )


def is_configured() -> bool:
    """Return whether Postgres credentials are available."""

    return bool(get_database_url())


def _load_psycopg() -> Any:
    try:
        import psycopg  # type: ignore
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

    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS credit_accounts (
                        scope_type TEXT NOT NULL CHECK (scope_type IN ('user', 'chat')),
                        scope_id BIGINT NOT NULL,
                        balance INTEGER NOT NULL DEFAULT 0,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (scope_type, scope_id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS onboarding_grants (
                        user_id BIGINT PRIMARY KEY,
                        credits INTEGER NOT NULL,
                        granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS star_payments (
                        telegram_payment_charge_id TEXT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        pack_id TEXT NOT NULL,
                        xtr_amount INTEGER NOT NULL,
                        credits_awarded INTEGER NOT NULL,
                        payload TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
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
                    """
                )
            conn.commit()

        _SCHEMA_READY = True


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

    cur.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 hour') AS hourly_count,
            COUNT(*) FILTER (WHERE granted_at >= NOW() - INTERVAL '1 day') AS daily_count
        FROM onboarding_grants
        """
    )
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

    ensure_schema()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (ONBOARDING_GRANTS_ADVISORY_LOCK_KEY,),
            )
            user_balance = _get_balance_for_update(cur, "user", user_id)
            if _has_existing_onboarding_grant(cur, user_id):
                conn.commit()
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
                conn.commit()
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
        conn.commit()

    return granted, int(user_balance)


def charge_ai_credits(
    user_id: int,
    chat_id: Optional[int],
    amount: int,
    *,
    event_type: str = "ai_charge",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Charge credits for an AI interaction.

    User balance is consumed first; in groups, chat balance is used as fallback.
    """

    ensure_schema()
    charge_amount = int(amount)
    metadata_dict = dict(metadata or {})

    with connect() as conn:
        with conn.cursor() as cur:
            user_balance = _get_balance_for_update(cur, "user", user_id)
            chat_balance = 0
            if chat_id is not None:
                chat_balance = _get_balance_for_update(cur, "chat", chat_id)

            if user_balance >= charge_amount:
                user_balance -= charge_amount
                _set_balance(cur, "user", user_id, user_balance)
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
                        str(event_type or "ai_charge"),
                        int(user_id),
                        int(user_id),
                        int(chat_id) if chat_id is not None else None,
                        -charge_amount,
                        json.dumps({"source": "user", **metadata_dict}),
                    ),
                )
                conn.commit()
                return {
                    "ok": True,
                    "source": "user",
                    "user_balance": user_balance,
                    "chat_balance": chat_balance,
                }

            if chat_id is not None and chat_balance >= charge_amount:
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
                        str(event_type or "ai_charge"),
                        int(user_id),
                        int(user_id),
                        int(chat_id),
                        -charge_amount,
                        json.dumps({"source": "chat", **metadata_dict}),
                    ),
                )
                conn.commit()
                return {
                    "ok": True,
                    "source": "chat",
                    "user_balance": user_balance,
                    "chat_balance": chat_balance,
                }

            conn.commit()
            return {
                "ok": False,
                "source": None,
                "user_balance": user_balance,
                "chat_balance": chat_balance,
            }


def refund_ai_charge(
    user_id: int,
    chat_id: Optional[int],
    amount: int,
    source: ScopeType,
    *,
    event_type: str = "ai_refund",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, int]:
    """Refund a previously charged AI credit."""

    ensure_schema()
    refund_amount = int(amount)
    metadata_dict = dict(metadata or {})

    with connect() as conn:
        with conn.cursor() as cur:
            if source == "chat" and chat_id is not None:
                chat_balance = _get_balance_for_update(cur, "chat", chat_id) + refund_amount
                _set_balance(cur, "chat", chat_id, chat_balance)
                user_balance = _get_balance_for_update(cur, "user", user_id)
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
                conn.commit()
                return {"user_balance": int(user_balance), "chat_balance": int(chat_balance)}

            user_balance = _get_balance_for_update(cur, "user", user_id) + refund_amount
            _set_balance(cur, "user", user_id, user_balance)
            chat_balance = 0
            if chat_id is not None:
                chat_balance = _get_balance_for_update(cur, "chat", chat_id)

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
            conn.commit()

            return {"user_balance": int(user_balance), "chat_balance": int(chat_balance)}


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

    ensure_schema()
    debt_amount = int(amount)
    metadata_dict = dict(metadata or {})

    with connect() as conn:
        with conn.cursor() as cur:
            if source == "chat" and chat_id is not None:
                chat_balance = _get_balance_for_update(cur, "chat", chat_id) - debt_amount
                _set_balance(cur, "chat", chat_id, chat_balance)
                user_balance = _get_balance_for_update(cur, "user", user_id)
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
                conn.commit()
                return {"user_balance": int(user_balance), "chat_balance": int(chat_balance)}

            user_balance = _get_balance_for_update(cur, "user", user_id) - debt_amount
            _set_balance(cur, "user", user_id, user_balance)
            chat_balance = 0
            if chat_id is not None:
                chat_balance = _get_balance_for_update(cur, "chat", chat_id)

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
            conn.commit()

            return {"user_balance": int(user_balance), "chat_balance": int(chat_balance)}


def transfer_user_to_chat(user_id: int, chat_id: int, amount: int) -> Dict[str, Any]:
    """Transfer credits from personal balance to group balance."""

    ensure_schema()
    transfer_amount = int(amount)

    with connect() as conn:
        with conn.cursor() as cur:
            user_balance = _get_balance_for_update(cur, "user", user_id)
            chat_balance = _get_balance_for_update(cur, "chat", chat_id)

            if user_balance < transfer_amount:
                conn.commit()
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
            conn.commit()

            return {
                "ok": True,
                "error": None,
                "user_balance": user_balance,
                "chat_balance": chat_balance,
            }



def mint_user_credits(user_id: int, amount: int, actor_user_id: Optional[int] = None) -> Dict[str, int]:
    """Mint credits to a user account and return the updated balance."""

    ensure_schema()
    mint_amount = int(amount)
    actor_id = int(actor_user_id) if actor_user_id is not None else int(user_id)

    with connect() as conn:
        with conn.cursor() as cur:
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
        conn.commit()

    return {"user_balance": int(user_balance)}

def record_star_payment(
    telegram_payment_charge_id: str,
    user_id: int,
    pack_id: str,
    xtr_amount: int,
    credits_awarded: int,
    payload: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist successful payment and credit the user idempotently."""

    ensure_schema()

    with connect() as conn:
        with conn.cursor() as cur:
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
        conn.commit()

    return {"inserted": inserted, "user_balance": int(user_balance)}
