"""Shared logging configuration for the bot."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured with the bot's root handler."""
    return logging.getLogger(f"respondedorbot.{name}")


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
