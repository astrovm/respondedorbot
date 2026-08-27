"""Service layer for chat configuration business logic.

This module centralizes PostgreSQL-backed config loading and update semantics.
It exposes a ChatConfigService that can be tested in isolation from the
persistence implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from api.bot.chat_config_defaults import CHAT_CONFIG_DEFAULTS
from api.storage.chat_config_repository import (
    ChatConfigRepository,
    build_chat_config_repository,
)

ConfigLogger = Callable[[str, Optional[Mapping[str, Any]]], None]
AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


class ChatConfigService:
    def __init__(
        self,
        repository: ChatConfigRepository,
        *,
        admin_reporter: AdminReporter,
        log_event: ConfigLogger,
    ) -> None:
        self._repo = repository
        self._admin_reporter = admin_reporter
        self._log_event = log_event
        self._cache: Dict[str, Dict[str, Any]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_chat_config(self, chat_id: str) -> Dict[str, Any]:
        if chat_id in self._cache:
            return self._cache[chat_id]

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
                self._cache[chat_id] = pg_config
                return pg_config
            self._cache[chat_id] = config
            return config
        except Exception as error:
            self._admin_reporter(
                "Error loading chat config",
                error,
                {"chat_id": chat_id, "postgres_configured": self._repo.is_configured()},
            )
        return config

    def set_chat_config(self, chat_id: str, **updates: Any) -> Dict[str, Any]:
        config = self.get_chat_config(chat_id)
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

        self._cache[chat_id] = config
        return config


def build_chat_config_service(
    repository: Optional[ChatConfigRepository] = None,
    *,
    admin_reporter: Optional[AdminReporter] = None,
    log_event: Optional[ConfigLogger] = None,
) -> ChatConfigService:
    if repository is None:
        repository = build_chat_config_repository()
    if admin_reporter is None:

        def _noop_admin_report(
            _message: str,
            _error: Optional[Exception],
            _extra: Optional[Dict[str, Any]],
        ) -> None:
            pass

        admin_reporter = _noop_admin_report
    if log_event is None:

        def _noop_log_event(_message: str, _extra: Optional[Mapping[str, Any]]) -> None:
            pass

        log_event = _noop_log_event

    return ChatConfigService(repository, admin_reporter=admin_reporter, log_event=log_event)
