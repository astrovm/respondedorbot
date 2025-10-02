"""Formatting and parsing helpers shared across the bot."""

from datetime import datetime
from typing import Optional, Union
from decimal import Decimal

__all__ = [
    "fmt_num",
    "fmt_signed_pct",
    "parse_date_string",
    "parse_monetary_number",
    "to_es_number",
    "to_ddmmyy",
]


def fmt_num(value: float, decimals: int = 2) -> str:
    """Format a number with up to ``decimals`` decimal places."""
    try:
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def fmt_signed_pct(value: float, decimals: int = 2) -> str:
    """Format a signed percentage with trimmed trailing zeros."""
    try:
        return f"{value:+.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def parse_date_string(value: str) -> Optional[datetime]:
    """Parse commonly used date formats returning a ``datetime`` at midnight."""
    try:
        s = (value or "").strip().split()[0]
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
    except Exception:
        pass
    return None


def parse_monetary_number(value: Union[str, float, int, Decimal]) -> Optional[float]:
    """Parse localized monetary strings like ``1.234,56`` into a float."""
    try:
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        s = str(value)
        return float(s.replace(".", "").replace(",", "."))
    except Exception:
        return None


def to_es_number(n: Union[float, int]) -> str:
    """Return a number formatted using Spanish separators."""
    try:
        s = f"{float(n):,.2f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(n)


def to_ddmmyy(date_iso: str) -> str:
    """Format ISO date strings as ``DD/MM/YY``."""
    dt = parse_date_string(date_iso)
    if not dt:
        return str(date_iso)
    return f"{dt.day:02d}/{dt.month:02d}/{dt.year % 100:02d}"
