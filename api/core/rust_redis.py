"""Temporary shared Redis endpoint boundary for Rust migration flags."""

from __future__ import annotations

import os


def redis_endpoint_from_env() -> tuple[str, int, str | None]:
    return (
        str(os.environ.get("REDIS_HOST") or "localhost"),
        int(os.environ.get("REDIS_PORT") or "6379"),
        os.environ.get("REDIS_PASSWORD") or None,
    )


__all__ = ["redis_endpoint_from_env"]
