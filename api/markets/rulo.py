from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from api.utils import fmt_signed_pct


USD_AMOUNT = 1000.0
EXCLUDED_USD_TO_USDT_EXCHANGES = {"banexcoin", "xapo", "x4t"}
EXCLUDED_USDT_TO_ARS_EXCHANGES = {"okexp2p"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _format_local_currency(value: float, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    if decimals:
        formatted = formatted.rstrip("0").rstrip(",")
    return formatted


def _format_local_signed(value: float, decimals: int = 2) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_local_currency(abs(value), decimals)}"


def _format_spread_line(
    label: str, sell_price: float, oficial_price: float, details: Sequence[str]
) -> str:
    diff = sell_price - oficial_price
    pct = (diff / oficial_price) * 100 if oficial_price else 0.0
    lines = [
        f"- {label}",
        f"  • Precio venta: {_format_local_currency(sell_price)} ARS/USD",
        f"  • Diferencia vs oficial: {_format_local_signed(diff)} ARS ({fmt_signed_pct(pct, 2)}%)",
    ]
    lines.extend(f"  • {detail}" for detail in details)
    return "\n".join(lines)


def _best_ask(
    quotes: Mapping[str, Any], excluded_exchanges: set[str]
) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for exchange, quote in quotes.items():
        if not isinstance(quote, Mapping):
            continue
        if exchange.lower() in excluded_exchanges:
            continue
        ask = _safe_float(quote.get("totalAsk")) or _safe_float(quote.get("ask"))
        if not ask or ask <= 0:
            continue
        if best is None or ask < best[1]:
            best = (exchange, ask)
    return best


def _best_bid(
    quotes: Mapping[str, Any], excluded_exchanges: set[str]
) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for exchange, quote in quotes.items():
        if not isinstance(quote, Mapping):
            continue
        if exchange.lower() in excluded_exchanges:
            continue
        bid = _safe_float(quote.get("totalBid")) or _safe_float(quote.get("bid"))
        if not bid or bid <= 0:
            continue
        if best is None or bid > best[1]:
            best = (exchange, bid)
    return best


def build_rulo_message(
    data: Mapping[str, Any],
    usd_usdt_data: Optional[Mapping[str, Any]],
    usdt_ars_data: Optional[Mapping[str, Any]],
    usd_amount: float = USD_AMOUNT,
) -> str:
    oficial_price = _safe_float((data.get("oficial") or {}).get("price"))

    if not oficial_price or oficial_price <= 0:
        return "No pude conseguir el oficial para armar el rulo"

    oficial_cost_ars = oficial_price * usd_amount
    base_usd = _format_local_currency(usd_amount, 0)
    base_ars = _format_local_currency(oficial_cost_ars)

    lines = [
        f"Rulos desde Oficial (precio oficial: {_format_local_currency(oficial_price)} ARS/USD)",
        f"Inversión base: {base_usd} USD → {base_ars} ARS",
        "",
    ]

    mep_best_price = _safe_float(
        ((data.get("mep") or {}).get("al30") or {}).get("ci", {}).get("price")
    )
    if mep_best_price:
        mep_final_ars = mep_best_price * usd_amount
        mep_profit_ars = mep_final_ars - oficial_cost_ars
        lines.append(
            _format_spread_line(
                "MEP (AL30 CI)",
                mep_best_price,
                oficial_price,
                [
                    f"Resultado: {base_usd} USD → {_format_local_currency(mep_final_ars)} ARS",
                    f"Ganancia: {_format_local_signed(mep_profit_ars)} ARS",
                ],
            )
        )

    blue_data = data.get("blue") or {}
    blue_price = _safe_float(blue_data.get("bid")) or _safe_float(
        blue_data.get("price")
    )
    if blue_price:
        blue_final_ars = blue_price * usd_amount
        blue_profit_ars = blue_final_ars - oficial_cost_ars
        lines.append(
            _format_spread_line(
                "Blue",
                blue_price,
                oficial_price,
                [
                    f"Resultado: {base_usd} USD → {_format_local_currency(blue_final_ars)} ARS",
                    f"Ganancia: {_format_local_signed(blue_profit_ars)} ARS",
                ],
            )
        )

    best_usd_to_usdt = _best_ask(
        usd_usdt_data or {}, EXCLUDED_USD_TO_USDT_EXCHANGES
    )
    best_usdt_to_ars = _best_bid(
        usdt_ars_data or {}, EXCLUDED_USDT_TO_ARS_EXCHANGES
    )
    if best_usd_to_usdt and best_usdt_to_ars:
        usd_to_usdt_rate = best_usd_to_usdt[1]
        usdt_to_ars_rate = best_usdt_to_ars[1]
        usdt_obtained = usd_amount / usd_to_usdt_rate
        ars_obtained = usdt_obtained * usdt_to_ars_rate
        final_price = ars_obtained / usd_amount
        usdt_profit_ars = ars_obtained - oficial_cost_ars
        lines.append(
            _format_spread_line(
                "USDT",
                final_price,
                oficial_price,
                [
                    (
                        f"Tramos: USD→USDT {best_usd_to_usdt[0].upper()}, "
                        f"USDT→ARS {best_usdt_to_ars[0].upper()}"
                    ),
                    (
                        f"Resultado: {base_usd} USD → {_format_local_currency(usdt_obtained, 2)} USDT → "
                        f"{_format_local_currency(ars_obtained)} ARS"
                    ),
                    f"Ganancia: {_format_local_signed(usdt_profit_ars)} ARS",
                ],
            )
        )

    if len(lines) <= 2:
        return "No encontré ningún rulo potable"

    return "\n".join(lines)
