"""Postgres-backed chat configuration storage."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Mapping, Optional, Protocol, cast
import json

from api.core.rust_bridge import load_rust_bridge
from api.services import credits_db as credits_db_service

_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False


class ChatConfigDBError(RuntimeError):
    """Raised when chat configuration persistence cannot be completed."""


class _RustChatConfig(Protocol):
    def chat_config_ensure_schema(self, database_url: str) -> None: ...

    def chat_config_get(self, database_url: str, chat_id: str) -> str: ...

    def chat_config_set(
        self,
        database_url: str,
        chat_id: str,
        config_json: str,
    ) -> str: ...


def _load_rust_chat_config() -> _RustChatConfig | None:
    module = load_rust_bridge("RUST_CHAT_CONFIG_IO_ENABLED")
    if module is None:
        return None
    return cast(_RustChatConfig, module)


def _rust_context() -> tuple[_RustChatConfig, str] | None:
    rust = _load_rust_chat_config()
    if rust is None:
        return None
    database_url = credits_db_service.get_database_url()
    if not database_url:
        raise ChatConfigDBError("Postgres is not configured")
    return rust, database_url


def _schema_is_ready() -> bool:
    """Read schema state without assuming it is stable across threads."""

    return _SCHEMA_READY


def is_configured() -> bool:
    """Return whether Postgres credentials are available."""

    return credits_db_service.is_configured()


def ensure_schema() -> None:
    """Create chat configuration table if it doesn't exist."""

    global _SCHEMA_READY

    if _schema_is_ready():
        return

    with _SCHEMA_LOCK:
        if _schema_is_ready():
            return

        rust_context = _rust_context()
        if rust_context is not None:
            rust, database_url = rust_context
            try:
                rust.chat_config_ensure_schema(database_url)
            except Exception as error:
                raise ChatConfigDBError(
                    "Rust chat configuration schema initialization failed"
                ) from error
            _SCHEMA_READY = True
            return

        with credits_db_service.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_configs (
                        chat_id TEXT PRIMARY KEY,
                        config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """)
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

    rust_context = _rust_context()
    if rust_context is not None:
        rust, database_url = rust_context
        try:
            payload = json.loads(rust.chat_config_get(database_url, str(chat_id)))
        except Exception as error:
            raise ChatConfigDBError("Rust chat configuration read failed") from error
        if payload is None:
            return None
        return _normalize(payload, defaults)

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

    rust_context = _rust_context()
    if rust_context is not None:
        rust, database_url = rust_context
        try:
            payload = json.loads(
                rust.chat_config_set(
                    database_url,
                    str(chat_id),
                    json.dumps(stored),
                )
            )
        except Exception as error:
            raise ChatConfigDBError("Rust chat configuration write failed") from error
        if not isinstance(payload, Mapping):
            raise ChatConfigDBError("Rust chat configuration write returned invalid data")
        return dict(payload)

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
