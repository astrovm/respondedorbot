from __future__ import annotations

from typing import Any, Dict, List

from api.markets.models import (
    CryptoQuote,
    DollarQuote,
    normalize_market_snapshot,
)
from api.utils import fmt_num, fmt_signed_pct


def _format_crypto(quote: CryptoQuote) -> str:
    line = f"- {quote.symbol}: {fmt_num(quote.price, 2)} usd"
    if quote.change_24h is not None:
        line += f" ({fmt_signed_pct(quote.change_24h, 2)} 24h)"
    if quote.dominance is not None:
        line += f", dom {fmt_num(quote.dominance, 1)}%"
    return line


def _format_dollar(quote: DollarQuote) -> str:
    line = f"- {quote.label}: {fmt_num(quote.price, 2)}"
    return (
        f"{line} (bid {fmt_num(quote.bid, 2)})"
        if quote.bid is not None
        else line
    )


def _section(title: str, lines: List[str]) -> List[str]:
    return [title, *lines] if lines else []


def format_market_info(market: Dict[str, Any]) -> str:
    snapshot = normalize_market_snapshot(market)
    lines = _section(
        "PRECIOS DE CRIPTOS:",
        [_format_crypto(quote) for quote in snapshot.crypto],
    )
    lines.extend(
        _section(
            "DOLARES:",
            [_format_dollar(quote) for quote in snapshot.dollars],
        )
    )
    return "\n".join(lines)
