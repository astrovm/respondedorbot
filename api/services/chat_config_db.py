"""Postgres-backed chat configuration storage."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Mapping, Optional
import json

from api.services import credits_db as credits_db_service


_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False


class ChatConfigDBError(RuntimeError):
    """Raised when chat configuration persistence cannot be completed."""


def is_configured() -> bool:
    """Return whether Postgres credentials are available."""

    return credits_db_service.is_configured()


def ensure_schema() -> None:
    """Create chat configuration table if it doesn't exist."""

    global _SCHEMA_READY

    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return

        with credits_db_service.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_configs (
                        chat_id TEXT PRIMARY KEY,
                        config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            conn.commit()

        _SCHEMA_READY = True


def _normalize(
    raw_config: Optional[Mapping[str, Any]],
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = dict(defaults)
    if not isinstance(raw_config, Mapping):
        return normalized

    for key in defaults:
        if key in raw_config:
            normalized[key] = raw_config[key]

    return normalized


def get_chat_config(chat_id: str, defaults: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Load chat config from Postgres, or return ``None`` when absent."""

    ensure_schema()

    with credits_db_service.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM chat_configs
                WHERE chat_id = %s
                """,
                (str(chat_id),),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        return None

    payload = row[0]
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
        return _normalize(parsed, defaults)

    return _normalize(payload, defaults)


def set_chat_config(chat_id: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist chat config to Postgres and return normalized stored config."""

    ensure_schema()

    stored = dict(config)

    with credits_db_service.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_configs (chat_id, config)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (chat_id)
                DO UPDATE SET
                    config = EXCLUDED.config,
                    updated_at = NOW()
                """,
                (str(chat_id), json.dumps(stored)),
            )
        conn.commit()

    return stored
