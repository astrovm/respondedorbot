"""Format normalized market snapshots for AI prompt context."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, cast

from api.core.rust_bridge import load_rust_bridge
from api.markets.models import (
    CryptoQuote,
    DollarQuote,
    normalize_market_snapshot,
)
from api.utils import fmt_num, fmt_signed_pct

logger = logging.getLogger(__name__)


class _RustMarketContextFormatter(Protocol):
    def format_market_info(self, market_json: str) -> str: ...


def _load_rust_market_context_formatter() -> Optional[_RustMarketContextFormatter]:
    module = load_rust_bridge("RUST_MARKET_CONTEXT_ENABLED")
    if module is None:
        return None
    return cast(_RustMarketContextFormatter, module)


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


def _format_market_info_python(market: Dict[str, Any]) -> str:
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


def format_market_info(market: Dict[str, Any]) -> str:
    rust = _load_rust_market_context_formatter()
    if rust is not None:
        try:
            return str(
                rust.format_market_info(
                    json.dumps(market, separators=(",", ":")),
                )
            )
        except Exception as error:
            logger.warning(
                "Rust market context formatting failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _format_market_info_python(market)
