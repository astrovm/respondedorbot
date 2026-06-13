"""Small runtime-facing wrapper around environment and Redis configuration."""

from __future__ import annotations

from logging import Logger
from typing import Any, Callable, Dict, Optional, Union

import redis as redis_module

from api import config

AdminReporter = Callable[
    [str, Optional[Exception], Optional[Dict[str, Any]]],
    None,
]


class ConfigRuntime:
    """Give services one consistent way to reach config and Redis.

    The lower-level ``api.config`` module keeps some process-wide configuration.
    This object makes sure the current admin reporter is installed before using
    it, so callers do not need to remember that setup step.
    """

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._admin_reporter: Optional[AdminReporter] = None

    def set_admin_reporter(self, reporter: AdminReporter) -> None:
        self._admin_reporter = reporter
        config.configure(admin_reporter=reporter)

    def redis(
        self,
        host: Optional[str] = None,
        port: Optional[Union[int, str]] = None,
        password: Optional[str] = None,
    ) -> redis_module.Redis:
        config.configure(admin_reporter=self._admin_reporter)
        return config.config_redis(host=host, port=port, password=password)

    def optional_redis(self, **kwargs: Any) -> Optional[redis_module.Redis]:
        """Return Redis when available, or ``None`` for optional features."""

        try:
            return self.redis(**kwargs)
        except Exception as error:
            self._logger.warning("optional Redis client unavailable: %s", error)
            return None

    def load_bot_config(self) -> Dict[str, Any]:
        config.configure(admin_reporter=self._admin_reporter)
        return config.load_bot_config()


__all__ = ["ConfigRuntime"]
