"""Helpers for fixed-precision AI credit amounts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional

CREDIT_SCALE = 100
LEGACY_CREDIT_SCALE = 10
_CREDIT_SCALE_DECIMAL = Decimal(CREDIT_SCALE)


def whole_credits_to_units(credits: int) -> int:
    """Convert whole credits into internal hundredths-of-credit units."""

    return int(credits) * CREDIT_SCALE


def rescale_credit_units(units: Any, source_scale: Any) -> int:
    """Convert stored units from a known scale into the current scale."""

    normalized_scale = int(source_scale or LEGACY_CREDIT_SCALE)
    if normalized_scale <= 0 or CREDIT_SCALE % normalized_scale != 0:
        raise ValueError("unsupported credit scale")
    return int(units or 0) * (CREDIT_SCALE // normalized_scale)


def parse_credit_units(value: Any) -> Optional[int]:
    """Parse a human credit amount with up to two decimals into credit units."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

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


def format_credit_units(units: Any) -> str:
    """Render internal credit units as a decimal string with two decimals."""

    normalized = int(units or 0)
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
