"""Service layer for chat configuration business logic.

This module centralizes config loading, migration from Redis to Postgres, and
update semantics. It exposes a ChatConfigService that can be tested in
isolation from the persistence implementation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional

import redis

from api.chat_config_defaults import CHAT_CONFIG_DEFAULTS


ConfigLogger = Callable[[str, Optional[Mapping[str, Any]]], None]
AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


def decode_redis_value(value: Any) -> Optional[str]:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if value is not None:
        return str(value)
    return None


def load_chat_config_from_redis(
    redis_client: redis.Redis,
    chat_id: str,
    *,
    log_event: ConfigLogger,
) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(CHAT_CONFIG_DEFAULTS)
    raw = redis_client.get(f"chat_config:{chat_id}")
    raw_text = decode_redis_value(raw)
    log_event(
        "Chat config raw value fetched", {"chat_id": chat_id, "raw_value": raw_text}
    )

    if raw_text:
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key in config:
                    config[key] = value
            return config

    log_event("No stored chat config found; using defaults", {"chat_id": chat_id})
    return config


class ChatConfigService:
    def __init__(
        self,
        repository,
        *,
        admin_reporter: AdminReporter,
        log_event: ConfigLogger,
    ) -> None:
        self._repo = repository
        self._admin_reporter = admin_reporter
        self._log_event = log_event

    def get_chat_config(
        self, redis_client: redis.Redis, chat_id: str
    ) -> Dict[str, Any]:
        config = dict(CHAT_CONFIG_DEFAULTS)
        try:
            self._log_event("Loading chat config", {"chat_id": chat_id})
            if not self._repo.is_configured():
                self._log_event(
                    "Chat config storage is not configured; using defaults",
                    {"chat_id": chat_id},
                )
                return config

            pg_config = self._repo.get_chat_config(chat_id, CHAT_CONFIG_DEFAULTS)
            if isinstance(pg_config, dict):
                return pg_config

            redis_config = load_chat_config_from_redis(
                redis_client,
                chat_id,
                log_event=self._log_event,
            )
            if redis_config != CHAT_CONFIG_DEFAULTS:
                try:
                    self._repo.set_chat_config(chat_id, redis_config)
                except Exception as persist_error:
                    self._admin_reporter(
                        "Error migrating chat config from Redis to Postgres",
                        persist_error,
                        {"chat_id": chat_id},
                    )
                else:
                    self._log_event(
                        "Migrated chat config from Redis to Postgres",
                        {"chat_id": chat_id, "config": redis_config},
                    )
            return redis_config
        except Exception as error:
            self._admin_reporter(
                "Error loading chat config",
                error,
                {"chat_id": chat_id, "postgres_configured": self._repo.is_configured()},
            )
        return config

    def set_chat_config(
        self, redis_client: redis.Redis, chat_id: str, **updates: Any
    ) -> Dict[str, Any]:
        config = self.get_chat_config(redis_client, chat_id)
        for key, value in updates.items():
            if key in config:
                config[key] = value

        try:
            self._log_event(
                "Saving chat config",
                {"chat_id": chat_id, "updates": updates, "config": config},
            )
            if not self._repo.is_configured():
                self._log_event(
                    "Chat config storage is not configured; skipping persistence",
                    {"chat_id": chat_id, "config": config},
                )
                return config

            self._repo.set_chat_config(chat_id, config)
        except Exception as error:
            self._admin_reporter(
                "Error saving chat config",
                error,
                {"chat_id": chat_id, "updates": updates},
            )

        return config


def build_chat_config_service(
    repository=None, *, admin_reporter=None, log_event=None
) -> ChatConfigService:
    if repository is None:
        from api.storage.chat_config_repository import build_chat_config_repository

        repository = build_chat_config_repository()
    if admin_reporter is None:

        def _noop_admin_report(*args, **kwargs):
            return None

        admin_reporter = _noop_admin_report
    if log_event is None:

        def _noop_log_event(*args, **kwargs):
            return None

        log_event = _noop_log_event

    return ChatConfigService(
        repository, admin_reporter=admin_reporter, log_event=log_event
    )
