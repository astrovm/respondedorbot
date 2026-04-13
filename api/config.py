"""Application-wide configuration helpers."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import redis


AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


_bot_config: Optional[Dict[str, Any]] = None
_admin_reporter: Optional[AdminReporter] = None


def configure(*, admin_reporter: Optional[AdminReporter] = None) -> None:
    """Register optional admin reporter callbacks."""

    global _admin_reporter
    _admin_reporter = admin_reporter


_SYSTEM_PROMPT = """[historical bot personality removed]"""


def load_bot_config() -> Dict[str, Any]:
    """Load bot configuration."""

    global _bot_config

    if _bot_config is not None:
        return _bot_config

    _bot_config = {
        "trigger_words": [
            "gordo",
            "respondedor",
            "atendedor",
            "gordito",
            "dogor",
            "bot",
        ],
        "system_prompt": _SYSTEM_PROMPT,
    }

    return _bot_config


def _admin_report(
    message: str, error: Optional[Exception], extra: Optional[Dict[str, Any]]
) -> None:
    if _admin_reporter:
        _admin_reporter(message, error, extra)


def config_redis(host=None, port=None, password=None):
    try:
        host = host or os.environ.get("REDIS_HOST", "localhost")
        port = int(port or os.environ.get("REDIS_PORT", 6379))
        password = password or os.environ.get("REDIS_PASSWORD", None)
        redis_client = redis.Redis(
            host=host, port=port, password=password, decode_responses=True
        )
        redis_client.ping()
        return redis_client
    except Exception as exc:  # pragma: no cover - passthrough for callers
        error_context = {
            "host": host,
            "port": port,
            "password": "***" if password else None,
        }
        error_msg = f"Redis connection error: {exc}"
        print(error_msg)
        _admin_report(error_msg, exc, error_context)
        raise


def reset_cache() -> None:
    """Clear cached configuration (used primarily in tests)."""

    global _bot_config
    _bot_config = None


def set_cache(config: Optional[Dict[str, Any]]) -> None:
    """Override cached configuration (test helper)."""

    global _bot_config
    _bot_config = config


__all__ = [
    "configure",
    "config_redis",
    "load_bot_config",
    "reset_cache",
    "set_cache",
]
