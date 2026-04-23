from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from api.ai_pricing import ensure_mapping
from api.utils import fmt_num, fmt_signed_pct


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def format_market_info(market: Dict) -> str:
    info: List[str] = []

    crypto_rows = market.get("crypto")
    if isinstance(crypto_rows, Sequence) and not isinstance(
        crypto_rows, (str, bytes, bytearray)
    ):
        crypto_lines: List[str] = []
        for crypto in list(crypto_rows)[:3]:
            if not isinstance(crypto, Mapping):
                continue
            symbol = (
                str(crypto.get("symbol") or crypto.get("name") or "").strip().upper()
            )
            usd_quote = (
                ensure_mapping((ensure_mapping(crypto.get("quote")) or {}).get("USD"))
                or {}
            )
            price = _safe_float(
                usd_quote.get("price") if usd_quote else crypto.get("price")
            )
            change_24h = _safe_float(
                (ensure_mapping(usd_quote.get("changes")) or {}).get("24h")
                if usd_quote
                else crypto.get("change_24h")
            )
            dominance = _safe_float(usd_quote.get("dominance")) if usd_quote else None
            if not symbol or price is None:
                continue
            line = f"- {symbol}: {fmt_num(price, 2)} usd"
            if change_24h is not None:
                line += f" ({fmt_signed_pct(change_24h, 2)} 24h)"
            if dominance is not None:
                line += f", dom {fmt_num(dominance, 1)}%"
            crypto_lines.append(line)
        if crypto_lines:
            info.append("PRECIOS DE CRIPTOS:")
            info.extend(crypto_lines)

    dollar_value = market.get("dollar")
    dollar_data = ensure_mapping(dollar_value)
    if (
        not dollar_data
        and isinstance(dollar_value, Sequence)
        and not isinstance(dollar_value, (str, bytes, bytearray))
    ):
        sequence_lines: List[str] = []
        for item in dollar_value:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("name") or item.get("label") or "").strip().lower()
            price = _safe_float(item.get("price"))
            if label and price is not None:
                sequence_lines.append(f"- {label}: {fmt_num(price, 2)}")
        if sequence_lines:
            info.append("DOLARES:")
            info.extend(sequence_lines)
    elif dollar_data:
        dollar_lines: List[str] = []

        oficial_price = _safe_float(
            (ensure_mapping(dollar_data.get("oficial")) or {}).get("price")
        )
        if oficial_price is not None:
            dollar_lines.append(f"- oficial: {fmt_num(oficial_price, 2)}")

        blue_data = ensure_mapping(dollar_data.get("blue")) or {}
        blue_ask = _safe_float(blue_data.get("ask") or blue_data.get("price"))
        blue_bid = _safe_float(blue_data.get("bid"))
        if blue_ask is not None:
            blue_line = f"- blue: {fmt_num(blue_ask, 2)}"
            if blue_bid is not None:
                blue_line += f" (bid {fmt_num(blue_bid, 2)})"
            dollar_lines.append(blue_line)

        mep_ci = ensure_mapping(
            (ensure_mapping(dollar_data.get("mep")) or {}).get("al30")
        )
        mep_ci = ensure_mapping((mep_ci or {}).get("ci")) or {}
        mep_price = _safe_float(mep_ci.get("price"))
        if mep_price is not None:
            dollar_lines.append(f"- mep al30 ci: {fmt_num(mep_price, 2)}")

        tarjeta_price = _safe_float(
            (ensure_mapping(dollar_data.get("tarjeta")) or {}).get("price")
        )
        if tarjeta_price is not None:
            dollar_lines.append(f"- tarjeta: {fmt_num(tarjeta_price, 2)}")

        usdt_data = (
            ensure_mapping(
                (ensure_mapping(dollar_data.get("cripto")) or {}).get("usdt")
            )
            or {}
        )
        usdt_ask = _safe_float(usdt_data.get("ask"))
        usdt_bid = _safe_float(usdt_data.get("bid"))
        if usdt_ask is not None:
            usdt_line = f"- usdt: {fmt_num(usdt_ask, 2)}"
            if usdt_bid is not None:
                usdt_line += f" (bid {fmt_num(usdt_bid, 2)})"
            dollar_lines.append(usdt_line)

        if dollar_lines:
            info.append("DOLARES:")
            info.extend(dollar_lines)

    return "\n".join(info)
