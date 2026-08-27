from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from api.i18n import tr
from api.utils import fmt_num, parse_date_string


BA_TZ = timezone(timedelta(hours=-3))


def normalize_text(value: Any) -> str:
    """Return lowercase ASCII-normalized text for fuzzy comparisons."""

    try:
        text = str(value or "")
    except Exception:
        text = ""
    normalized = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    return normalized.lower()


def format_bcra_value(value_str: str, is_percentage: bool = False) -> str:
    try:
        clean_value = (
            value_str.replace(".", "").replace(",", ".")
            if not is_percentage
            else value_str.replace(",", ".")
        )
        num = float(clean_value)
        if is_percentage:
            return f"{num:.1f}%" if num >= 10 else f"{num:.2f}%"
        if num >= 1_000_000:
            return f"{num / 1000:,.0f}".replace(",", ".")
        if num >= 1000:
            return f"{num:,.0f}".replace(",", ".")
        return f"{num:.2f}".replace(".", ",")
    except Exception:
        return f"{value_str}%" if is_percentage else value_str


def _format_country_risk(country_risk: Optional[Dict[str, Any]]) -> Optional[str]:
    if not country_risk:
        return None

    value_bps = country_risk.get("value_bps")
    if not isinstance(value_bps, (int, float)):
        return None

    value_decimals = 1 if abs(value_bps) < 100 else 0
    value_text = fmt_num(float(value_bps), value_decimals).replace(".", ",")
    risk_line = tr("bcra.country_risk", value=value_text)

    details: List[str] = []
    label = country_risk.get("valuation_label")
    if isinstance(label, str) and label:
        details.append(label)

    delta_value = country_risk.get("delta_one_day")
    if isinstance(delta_value, (int, float)):
        abs_delta = abs(delta_value)
        if abs_delta >= 0.05:
            delta_decimals = 1 if abs_delta < 100 else 0
            delta_text = fmt_num(abs_delta, delta_decimals).replace(".", ",")
            sign = "+" if delta_value > 0 else "-"
            details.append(tr("bcra.yesterday", change=f"{sign}{delta_text}"))

    if details:
        risk_line += " (" + " | ".join(details) + ")"
    return risk_line


def _format_currency_bands(band_limits: Optional[Dict[str, Any]]) -> Optional[str]:
    if not band_limits:
        return None

    lower = band_limits.get("lower")
    upper = band_limits.get("upper")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return None

    date_label = band_limits.get("date")
    lower_text = fmt_num(float(lower), 2)
    upper_text = fmt_num(float(upper), 2)
    line = tr("bcra.bands", lower=lower_text, upper=upper_text)
    if isinstance(date_label, str) and date_label:
        line += f" ({date_label})"
    return line


def _format_itcrm(details: Optional[Tuple[float, str]]) -> Optional[str]:
    if not details:
        return None

    itcrm_value, date_str = details
    return f"tcrm: {fmt_num(float(itcrm_value), 2)}" + (
        f" ({date_str})" if date_str else ""
    )


def format_bcra_variables(
    variables: Dict[str, Any],
    *,
    band_limits: Optional[Dict[str, Any]] = None,
    itcrm_details: Optional[Tuple[float, str]] = None,
    country_risk: Optional[Dict[str, Any]] = None,
    today: Optional[date] = None,
) -> str:
    """Format BCRA variables for display (robust to naming changes)."""

    if not variables:
        return tr("bcra.none")

    specs: List[Tuple[str, Callable[[Any], str]]] = [
        (
            r"base\s*monetaria",
            lambda v: tr("bcra.monetary_base", value=format_bcra_value(v)),
        ),
        (
            r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual",
            lambda v: tr("bcra.monthly_inflation", value=format_bcra_value(v, True)),
        ),
        (
            r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual",
            lambda v: tr("bcra.yearly_inflation", value=format_bcra_value(v, True)),
        ),
        (
            r"(mediana.*variacion.*interanual.*(12|doce).*meses.*(relevamiento.*expectativas.*mercado|rem)|inflacion.*esperada)",
            lambda v: tr("bcra.expected_inflation", value=format_bcra_value(v, True)),
        ),
        (r"tamar", lambda v: f"TAMAR: {format_bcra_value(v, True)}"),
        (r"badlar", lambda v: f"BADLAR: {format_bcra_value(v, True)}"),
        (
            r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor",
            lambda v: tr("bcra.retail_dollar", value=v),
        ),
        (r"tipo.*cambio.*mayorista", lambda v: tr("bcra.wholesale_dollar", value=v)),
        (r"unidad.*valor.*adquisitivo|\buva\b", lambda v: f"UVA: ${v}"),
        (r"coeficiente.*estabilizacion.*referencia|\bcer\b", lambda v: f"CER: {v}"),
        (
            r"reservas.*internacionales",
            lambda v: tr("bcra.reserves", value=format_bcra_value(v)),
        ),
    ]

    meta_info: Dict[str, Any] = {}
    if isinstance(variables, dict):
        candidate_meta = variables.get("_meta")
        if isinstance(candidate_meta, dict):
            meta_info = candidate_meta

    lines = [f"{tr('bcra.title')}\n"]
    latest_dt: Optional[datetime] = None
    for pattern, formatter in specs:
        compiled = re.compile(pattern)
        for key, data in variables.items():
            if str(key).startswith("_"):
                continue
            if not isinstance(data, Mapping):
                continue
            if compiled.search(normalize_text(key)):
                value = data.get("value", "")
                date_label = data.get("date", "")
                line = formatter(value)
                if date_label and date_label != value:
                    line += f" ({str(date_label).replace('/2025', '/25')})"
                lines.append(line)
                parsed_dt = parse_date_string(str(date_label))
                if parsed_dt and (latest_dt is None or parsed_dt > latest_dt):
                    latest_dt = parsed_dt
                break

    risk_line = _format_country_risk(country_risk)
    if risk_line:
        lines.append(risk_line)

    band_line = _format_currency_bands(band_limits)
    if band_line:
        lines.append(band_line)

    itcrm_line = _format_itcrm(itcrm_details)
    if itcrm_line:
        lines.append(itcrm_line)

    if meta_info.get("stale"):
        stale_msg = tr("bcra.stale")
        if stale_msg not in lines:
            lines.append(stale_msg)

    if latest_dt:
        effective_today = today or datetime.now(BA_TZ).date()
        age_days = (effective_today - latest_dt.date()).days
        if age_days >= 3:
            lines.append(tr("bcra.delayed", days=age_days))

    return "\n".join(lines)


__all__ = ["format_bcra_value", "format_bcra_variables", "normalize_text"]
