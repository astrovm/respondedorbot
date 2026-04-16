"""Repository adapter for chat configuration persistence.

This file provides a thin, testable adapter around the existing
`api.services.chat_config_db` implementation. The goal is to provide an
interface that the ChatConfigService can depend on so implementations can
be swapped or mocked in tests.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from api.services import chat_config_db as _pg


class ChatConfigRepository:
    """Adapter around `api.services.chat_config_db`.

    Methods mirror the lower-level implementation but live behind a class
    so the service can receive a concrete dependency in tests.
    """

    def is_configured(self) -> bool:
        return _pg.is_configured()

    def get_chat_config(
        self, chat_id: str, defaults: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return _pg.get_chat_config(str(chat_id), defaults)

    def set_chat_config(
        self, chat_id: str, config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return _pg.set_chat_config(str(chat_id), config)


def build_chat_config_repository() -> ChatConfigRepository:
    return ChatConfigRepository()
