from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, cast

from api.core.rust_bridge import load_rust_bridge
from api.i18n import tr
from api.utils import fmt_signed_pct


USD_AMOUNT = 1000.0
EXCLUDED_USD_TO_USDT_EXCHANGES = {"banexcoin", "xapo", "x4t"}
EXCLUDED_USDT_TO_ARS_EXCHANGES = {"okexp2p"}
logger = logging.getLogger(__name__)


class _RustRuloEvaluator(Protocol):
    def evaluate_rulo(self, input_json: str) -> str: ...


def _load_rust_rulo_evaluator() -> _RustRuloEvaluator | None:
    module = load_rust_bridge("RUST_RULO_ENABLED")
    if module is None:
        return None
    return cast(_RustRuloEvaluator, module)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except TypeError, ValueError:
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
        "  • "
        + tr(
            "market.rulo.sell_price",
            price=_format_local_currency(sell_price),
        ),
        "  • "
        + tr(
            "market.rulo.official_diff",
            difference=_format_local_signed(diff),
            percentage=fmt_signed_pct(pct, 2),
        ),
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


def _build_rulo_message_python(
    data: Mapping[str, Any],
    usd_usdt_data: Optional[Mapping[str, Any]],
    usdt_ars_data: Optional[Mapping[str, Any]],
    usd_amount: float = USD_AMOUNT,
) -> str:
    oficial_price = _safe_float((data.get("oficial") or {}).get("price"))

    if not oficial_price or oficial_price <= 0:
        return tr("market.rulo.official_error")

    oficial_cost_ars = oficial_price * usd_amount
    base_usd = _format_local_currency(usd_amount, 0)
    base_ars = _format_local_currency(oficial_cost_ars)

    lines = [
        tr(
            "market.rulo.title",
            price=_format_local_currency(oficial_price),
        ),
        tr("market.rulo.base", usd=base_usd, ars=base_ars),
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
                    tr(
                        "market.rulo.result",
                        result=f"{base_usd} USD → {_format_local_currency(mep_final_ars)} ARS",
                    ),
                    tr(
                        "market.rulo.profit",
                        profit=_format_local_signed(mep_profit_ars),
                    ),
                ],
            )
        )

    blue_data = data.get("blue") or {}
    blue_price = _safe_float(blue_data.get("bid")) or _safe_float(blue_data.get("price"))
    if blue_price:
        blue_final_ars = blue_price * usd_amount
        blue_profit_ars = blue_final_ars - oficial_cost_ars
        lines.append(
            _format_spread_line(
                "Blue",
                blue_price,
                oficial_price,
                [
                    tr(
                        "market.rulo.result",
                        result=f"{base_usd} USD → {_format_local_currency(blue_final_ars)} ARS",
                    ),
                    tr(
                        "market.rulo.profit",
                        profit=_format_local_signed(blue_profit_ars),
                    ),
                ],
            )
        )

    best_usd_to_usdt = _best_ask(usd_usdt_data or {}, EXCLUDED_USD_TO_USDT_EXCHANGES)
    best_usdt_to_ars = _best_bid(usdt_ars_data or {}, EXCLUDED_USDT_TO_ARS_EXCHANGES)
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
                        tr(
                            "market.rulo.steps",
                            steps=(
                                f"USD→USDT {best_usd_to_usdt[0].upper()}, "
                                f"USDT→ARS {best_usdt_to_ars[0].upper()}"
                            ),
                        )
                    ),
                    (
                        tr(
                            "market.rulo.result",
                            result=(
                                f"{base_usd} USD → {_format_local_currency(usdt_obtained, 2)} USDT → "
                                f"{_format_local_currency(ars_obtained)} ARS"
                            ),
                        )
                    ),
                    tr(
                        "market.rulo.profit",
                        profit=_format_local_signed(usdt_profit_ars),
                    ),
                ],
            )
        )

    if len(lines) <= 2:
        return tr("market.rulo.none")

    return "\n".join(lines)


def _normalized_exchange_quotes(
    quotes: Mapping[str, Any],
    excluded: set[str],
    primary_key: str,
    fallback_key: str,
) -> list[dict[str, Any]]:
    result = []
    for exchange, quote in quotes.items():
        if not isinstance(quote, Mapping) or exchange.lower() in excluded:
            continue
        price = _safe_float(quote.get(primary_key)) or _safe_float(quote.get(fallback_key))
        result.append({"exchange": exchange, "price": price})
    return result


def _normalize_rulo_input(
    data: Mapping[str, Any],
    usd_usdt_data: Optional[Mapping[str, Any]],
    usdt_ars_data: Optional[Mapping[str, Any]],
    usd_amount: float,
) -> dict[str, Any]:
    blue_data = data.get("blue") or {}
    return {
        "official": _safe_float((data.get("oficial") or {}).get("price")),
        "mep": _safe_float(
            ((data.get("mep") or {}).get("al30") or {}).get("ci", {}).get("price")
        ),
        "blue": _safe_float(blue_data.get("bid")) or _safe_float(blue_data.get("price")),
        "usd_to_usdt": _normalized_exchange_quotes(
            usd_usdt_data or {},
            EXCLUDED_USD_TO_USDT_EXCHANGES,
            "totalAsk",
            "ask",
        ),
        "usdt_to_ars": _normalized_exchange_quotes(
            usdt_ars_data or {},
            EXCLUDED_USDT_TO_ARS_EXCHANGES,
            "totalBid",
            "bid",
        ),
        "usd_amount": usd_amount,
    }


def _render_rulo_result(result: Mapping[str, Any]) -> str:
    kind = result.get("kind")
    if kind == "official_error":
        return tr("market.rulo.official_error")
    if kind == "no_routes":
        return tr("market.rulo.none")
    if kind != "routes":
        raise ValueError("Rust rulo result has an unknown kind")
    routes = result.get("routes")
    if not isinstance(routes, list):
        raise ValueError("Rust rulo routes are invalid")
    lines = [
        tr("market.rulo.title", price=result["official"]),
        tr("market.rulo.base", usd=result["base_usd"], ars=result["base_ars"]),
        "",
    ]
    for route in routes:
        if not isinstance(route, Mapping):
            raise ValueError("Rust rulo route is invalid")
        route_lines = [
            f"- {route['label']}",
            "  • " + tr("market.rulo.sell_price", price=route["sell_price"]),
            "  • "
            + tr(
                "market.rulo.official_diff",
                difference=route["difference"],
                percentage=route["percentage"],
            ),
        ]
        details = route.get("details")
        if not isinstance(details, list):
            raise ValueError("Rust rulo details are invalid")
        for detail in details:
            if not isinstance(detail, Mapping) or detail.get("kind") not in {
                "steps",
                "result",
                "profit",
            }:
                raise ValueError("Rust rulo detail is invalid")
            detail_kind = str(detail["kind"])
            route_lines.append(
                "  • " + tr(f"market.rulo.{detail_kind}", **{detail_kind: detail["text"]})
            )
        lines.append("\n".join(route_lines))
    return "\n".join(lines)


def build_rulo_message(
    data: Mapping[str, Any],
    usd_usdt_data: Optional[Mapping[str, Any]],
    usdt_ars_data: Optional[Mapping[str, Any]],
    usd_amount: float = USD_AMOUNT,
) -> str:
    rust = _load_rust_rulo_evaluator()
    if rust is not None:
        try:
            normalized = _normalize_rulo_input(
                data,
                usd_usdt_data,
                usdt_ars_data,
                usd_amount,
            )
            return _render_rulo_result(
                json.loads(rust.evaluate_rulo(json.dumps(normalized, separators=(",", ":"))))
            )
        except Exception as error:
            logger.warning(
                "Rust rulo evaluation failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _build_rulo_message_python(data, usd_usdt_data, usdt_ars_data, usd_amount)
