"""Shared logging configuration for the bot."""

from __future__ import annotations

import logging
import sys
from typing import Any, Mapping


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured with the bot's root handler."""
    return logging.getLogger(f"respondedorbot.{name}")


def format_log_context(context: Mapping[str, Any] | None) -> str:
    """Return stable key=value context for human-readable podman logs."""

    if not context:
        return ""
    parts = []
    for key in (
        "source",
        "chat_id",
        "chat_title",
        "chat_type",
        "message_id",
        "user_id",
        "user_name",
        "tool_round",
        "model",
    ):
        value = context.get(key)
        if value is None or value == "":
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 120:
            text = text[:117].rstrip() + "..."
        parts.append(f"{key}={text}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for the bot."""
    root = logging.getLogger("respondedorbot")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(level)
