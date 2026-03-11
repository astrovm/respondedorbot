"""Shared helpers for randomized degraded replies."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def resolve_random_reply_name(sender: Mapping[str, Any]) -> str:
    """Prefer first_name, then username, then the empty string."""

    first_name = str(sender.get("first_name") or "").strip()
    if first_name:
        return first_name
    username = str(sender.get("username") or "").strip()
    if username:
        return username
    return ""


def build_random_reply(
    gen_random_fn: Callable[[str], str],
    sender: Mapping[str, Any],
) -> str:
    """Build a random reply using the shared sender-name resolution."""

    return gen_random_fn(resolve_random_reply_name(sender))
