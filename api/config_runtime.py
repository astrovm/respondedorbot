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
        try:
            return self.redis(**kwargs)
        except Exception as error:
            self._logger.warning("optional Redis client unavailable: %s", error)
            return None

    def load_bot_config(self) -> Dict[str, Any]:
        config.configure(admin_reporter=self._admin_reporter)
        return config.load_bot_config()


__all__ = ["ConfigRuntime"]
