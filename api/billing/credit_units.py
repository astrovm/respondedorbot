"""Helpers for fixed-precision AI credit amounts."""

from __future__ import annotations

import importlib
import logging
import os
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Optional, Protocol, cast

CREDIT_SCALE = 100
LEGACY_CREDIT_SCALE = 10
_CREDIT_SCALE_DECIMAL = Decimal(CREDIT_SCALE)
_RUST_ENABLED_VALUES = frozenset({"1", "on", "true", "yes"})

logger = logging.getLogger(__name__)


class _RustCreditUnits(Protocol):
    def whole_credits_to_units(self, credits: int) -> int: ...

    def rescale_credit_units(self, units: int, source_scale: int) -> int: ...

    def parse_credit_units(self, value: str) -> Optional[int]: ...

    def format_credit_units(self, units: int) -> str: ...


@lru_cache(maxsize=1)
def _import_rust_credit_units() -> Optional[_RustCreditUnits]:
    try:
        module = importlib.import_module("respondedorbot_rs")
    except ImportError as error:
        logger.warning(
            "Rust credit-unit bridge unavailable; using Python fallback: error_type=%s",
            type(error).__name__,
        )
        return None
    return cast(_RustCreditUnits, module)


def _load_rust_credit_units() -> Optional[_RustCreditUnits]:
    enabled = str(os.environ.get("RUST_CREDIT_UNITS_ENABLED") or "").strip().lower()
    if enabled not in _RUST_ENABLED_VALUES:
        return None
    return _import_rust_credit_units()


def _log_rust_fallback(operation: str, error: Exception) -> None:
    logger.warning(
        "Rust credit-unit operation failed; using Python fallback: operation=%s error_type=%s",
        operation,
        type(error).__name__,
    )


def whole_credits_to_units(credits: int) -> int:
    """Convert whole credits into internal hundredths-of-credit units."""

    normalized = int(credits)
    rust = _load_rust_credit_units()
    if rust is not None:
        try:
            return int(rust.whole_credits_to_units(normalized))
        except Exception as error:
            _log_rust_fallback("whole_credits_to_units", error)
    return normalized * CREDIT_SCALE


def rescale_credit_units(units: Any, source_scale: Any) -> int:
    """Convert stored units from a known scale into the current scale."""

    normalized_scale = int(source_scale or LEGACY_CREDIT_SCALE)
    if normalized_scale <= 0 or CREDIT_SCALE % normalized_scale != 0:
        raise ValueError("unsupported credit scale")
    normalized_units = int(units or 0)
    rust = _load_rust_credit_units()
    if rust is not None:
        try:
            return int(rust.rescale_credit_units(normalized_units, normalized_scale))
        except Exception as error:
            _log_rust_fallback("rescale_credit_units", error)
    return normalized_units * (CREDIT_SCALE // normalized_scale)


def _parse_credit_units_python(text: str) -> Optional[int]:
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    if not parsed.is_finite():
        return None

    scaled = parsed * _CREDIT_SCALE_DECIMAL
    if scaled != scaled.to_integral_value():
        return None

    return int(scaled)


def parse_credit_units(value: Any) -> Optional[int]:
    """Parse a human credit amount with up to two decimals into credit units."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    rust = _load_rust_credit_units()
    if rust is not None:
        try:
            result = rust.parse_credit_units(text)
            if result is not None:
                return int(result)
        except Exception as error:
            _log_rust_fallback("parse_credit_units", error)
    return _parse_credit_units_python(text)


def format_credit_units(units: Any) -> str:
    """Render internal credit units as a decimal string with two decimals."""

    normalized = int(units or 0)
    rust = _load_rust_credit_units()
    if rust is not None:
        try:
            return str(rust.format_credit_units(normalized))
        except Exception as error:
            _log_rust_fallback("format_credit_units", error)
    sign = "-" if normalized < 0 else ""
    absolute = abs(normalized)
    whole, decimal = divmod(absolute, CREDIT_SCALE)
    return f"{sign}{whole}.{decimal:02d}"


__all__ = [
    "CREDIT_SCALE",
    "LEGACY_CREDIT_SCALE",
    "format_credit_units",
    "parse_credit_units",
    "rescale_credit_units",
    "whole_credits_to_units",
]
