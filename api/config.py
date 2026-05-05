"""Application-wide configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import redis

AdminReporter = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]

_bot_config: Optional[Dict[str, Any]] = None
_admin_reporter: Optional[AdminReporter] = None
_WORKSPACE_DIR: Optional[Path] = None
_RedisPoolKey = Tuple[str, int, Optional[str], bool]
_REDIS_POOLS: Dict[_RedisPoolKey, redis.ConnectionPool] = {}


def configure(*, admin_reporter: Optional[AdminReporter] = None) -> None:
    """Register optional admin reporter callbacks."""

    global _admin_reporter
    _admin_reporter = admin_reporter


def _resolve_workspace_dir() -> Path:
    global _WORKSPACE_DIR
    if _WORKSPACE_DIR is not None:
        return _WORKSPACE_DIR

    candidate = Path(__file__).resolve().parent.parent / "workspace"
    if candidate.is_dir():
        _WORKSPACE_DIR = candidate
        return candidate

    candidate = Path.cwd() / "workspace"
    if candidate.is_dir():
        _WORKSPACE_DIR = candidate
        return candidate

    _WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
    return _WORKSPACE_DIR


def _read_bootstrap_file(name: str) -> Optional[str]:
    path = _resolve_workspace_dir() / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _build_system_prompt_from_workspace() -> Optional[str]:
    soul = _read_bootstrap_file("SOUL.md")
    rules = _read_bootstrap_file("RULES.md")

    if not soul and not rules:
        return None

    parts = []
    if soul:
        parts.append(soul)
    if rules:
        parts.append(rules)
    return "\n\n".join(parts)


def load_bot_config() -> Dict[str, Any]:
    global _bot_config

    if _bot_config is not None:
        return _bot_config

    env_prompt = os.environ.get("BOT_SYSTEM_PROMPT")
    if env_prompt:
        system_prompt = env_prompt
    else:
        workspace_prompt = _build_system_prompt_from_workspace()
        if not workspace_prompt:
            raise RuntimeError(
                "workspace/ directory missing. "
                "Create workspace/SOUL.md and workspace/RULES.md, or set BOT_SYSTEM_PROMPT env var."
            )
        system_prompt = workspace_prompt

    _bot_config = {
        "trigger_words": [
            "gordo",
            "respondedor",
            "atendedor",
            "gordito",
            "dogor",
            "bot",
        ],
        "system_prompt": system_prompt,
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
        port = int(port or os.environ.get("REDIS_PORT", "6379"))
        password = password or os.environ.get("REDIS_PASSWORD", None)
        decode_responses = True
        pool_key = (host, port, password, decode_responses)
        pool = _REDIS_POOLS.get(pool_key)
        if pool is None:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                decode_responses=decode_responses,
            )
            _REDIS_POOLS[pool_key] = pool
        redis_client = redis.Redis(connection_pool=pool)
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

    global _bot_config, _WORKSPACE_DIR
    _bot_config = None
    _WORKSPACE_DIR = None


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
