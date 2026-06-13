"""Admin alerts and Telegram chat-admin checks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

import redis

from api.admin.reporting import admin_report
from api.bot.chat_settings import (
    is_chat_admin,
    report_unauthorized_config_attempt,
)
from api.core.config_runtime import ConfigRuntime
from api.bot.telegram import TelegramGateway


class AdminService:
    """Central place for operations that require administrator awareness.

    Error reports are sent through Telegram with secrets redacted. Chat-admin
    checks use a short Redis cache before asking Telegram again.
    """

    def __init__(
        self,
        *,
        telegram: TelegramGateway,
        config: ConfigRuntime,
        log_event: Callable[[str, Optional[Mapping[str, Any]]], None],
        redis_get_json: Callable[[Any, str], Any],
        redis_setex_json: Callable[[Any, str, int, Mapping[str, Any]], Any],
    ) -> None:
        self._telegram = telegram
        self._config = config
        self._log_event = log_event
        self._redis_get_json = redis_get_json
        self._redis_setex_json = redis_setex_json

    def report(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        admin_report(
            message,
            error,
            extra_context,
            send_message=self._telegram.send_message,
            redact=self._telegram.redact_tokens,
        )

    def is_chat_admin(
        self,
        chat_id: str,
        user_id: Optional[Union[str, int]],
        *,
        redis_client: Optional[redis.Redis] = None,
    ) -> bool:
        return is_chat_admin(
            chat_id,
            user_id,
            redis_client=redis_client,
            optional_redis_client=self._config.optional_redis,
            telegram_request=self._telegram.request,
            log_event=self._log_event,
            redis_get_json_fn=self._redis_get_json,
            redis_setex_json_fn=self._redis_setex_json,
        )

    def report_unauthorized_config_attempt(
        self,
        chat_id: str,
        user: Mapping[str, Any],
        *,
        chat_type: Optional[str],
        action: str,
        callback_data: Optional[str] = None,
    ) -> None:
        report_unauthorized_config_attempt(
            chat_id,
            user,
            chat_type=chat_type,
            action=action,
            log_event=self._log_event,
            callback_data=callback_data,
        )


__all__ = ["AdminService"]
