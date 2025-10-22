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


def load_bot_config() -> Dict[str, Any]:
    """Load bot configuration from environment variables."""

    global _bot_config

    if _bot_config is not None:
        return _bot_config

    system_prompt = os.environ.get("BOT_SYSTEM_PROMPT")
    trigger_words_str = os.environ.get("BOT_TRIGGER_WORDS")

    if not system_prompt:
        raise ValueError("BOT_SYSTEM_PROMPT environment variable is required")

    if not trigger_words_str:
        raise ValueError("BOT_TRIGGER_WORDS environment variable is required")

    trigger_words = [word.strip() for word in trigger_words_str.split(",")]

    _bot_config = {"trigger_words": trigger_words, "system_prompt": system_prompt}

    return _bot_config


def _admin_report(message: str, error: Optional[Exception], extra: Optional[Dict[str, Any]]) -> None:
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
