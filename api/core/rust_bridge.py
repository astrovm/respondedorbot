"""Load the temporary Rust extension for feature-flagged migration slices."""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_ENABLED_VALUES = frozenset({"1", "on", "true", "yes"})


def feature_enabled(environment_key: str) -> bool:
    value = str(os.environ.get(environment_key) or "").strip().lower()
    return value in _ENABLED_VALUES


@lru_cache(maxsize=1)
def _import_bridge() -> Optional[ModuleType]:
    try:
        return importlib.import_module("respondedorbot_rs")
    except ImportError as error:
        logger.warning(
            "Rust bridge unavailable; using Python fallback: error_type=%s",
            type(error).__name__,
        )
        return None


def load_rust_bridge(environment_key: str) -> Optional[ModuleType]:
    """Return the extension when one migration feature is enabled."""

    if not feature_enabled(environment_key):
        return None
    return _import_bridge()


def reset_rust_bridge_cache() -> None:
    """Clear the import cache for tests and controlled runtime verification."""

    _import_bridge.cache_clear()


__all__ = [
    "feature_enabled",
    "load_rust_bridge",
    "reset_rust_bridge_cache",
]
