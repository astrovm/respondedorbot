"""Simple RAM-based provider backoff.

When a provider returns HTTP 429, we mark it as cooled down for a period
extracted from retry-after headers (or a default). Calls always fall through
to the next provider in the chain — backoff is advisory, not a hard block.
"""

from __future__ import annotations

import threading
import time
from typing import Dict

_cooldowns: Dict[str, float] = {}
_lock = threading.Lock()


def mark_provider_cooldown(provider_key: str, seconds: float) -> None:
    if not provider_key or seconds <= 0:
        return
    new_until = time.time() + seconds
    with _lock:
        current = _cooldowns.get(provider_key, 0.0)
        if new_until > current:
            _cooldowns[provider_key] = new_until


def is_provider_cooled_down(provider_key: str) -> bool:
    with _lock:
        remaining = _cooldowns.get(provider_key, 0.0) - time.time()
    if remaining <= 0:
        with _lock:
            _cooldowns.pop(provider_key, None)
        return False
    return True


def get_provider_cooldown_remaining(provider_key: str) -> float:
    with _lock:
        remaining = _cooldowns.get(provider_key, 0.0) - time.time()
    return max(0.0, remaining)


def clear_provider_cooldown(provider_key: str) -> None:
    with _lock:
        _cooldowns.pop(provider_key, None)


def clear_all_cooldowns() -> None:
    with _lock:
        _cooldowns.clear()
