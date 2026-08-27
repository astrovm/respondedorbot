"""Localized model-facing prompts."""

from __future__ import annotations

from typing import Any

from api.i18n import Locale, tr


def prompt(name: str, *, locale: Locale | None = None, **values: Any) -> str:
    return tr(f"prompt.{name}", locale=locale, **values)


__all__ = ["prompt"]
