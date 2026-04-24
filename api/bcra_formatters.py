from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    risk_line = f"riesgo país: {value_text} bps"

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
            details.append(f"{sign}{delta_text} bps vs ayer")

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
    line = f"bandas cambiarias: piso ${lower_text} / techo ${upper_text}"
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
        return "No se pudieron obtener las variables del BCRA"

    specs = [
        (
            r"base\s*monetaria",
            lambda v: f"base monetaria: ${format_bcra_value(v)} mill. pesos",
        ),
        (
            r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual",
            lambda v: f"inflación mensual: {format_bcra_value(v, True)}",
        ),
        (
            r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual",
            lambda v: f"inflación interanual: {format_bcra_value(v, True)}",
        ),
        (
            r"(mediana.*variacion.*interanual.*(12|doce).*meses.*(relevamiento.*expectativas.*mercado|rem)|inflacion.*esperada)",
            lambda v: f"inflación esperada: {format_bcra_value(v, True)}",
        ),
        (r"tamar", lambda v: f"TAMAR: {format_bcra_value(v, True)}"),
        (r"badlar", lambda v: f"BADLAR: {format_bcra_value(v, True)}"),
        (
            r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor",
            lambda v: f"dólar minorista: ${v}",
        ),
        (r"tipo.*cambio.*mayorista", lambda v: f"dólar mayorista: ${v}"),
        (r"unidad.*valor.*adquisitivo|\buva\b", lambda v: f"UVA: ${v}"),
        (r"coeficiente.*estabilizacion.*referencia|\bcer\b", lambda v: f"CER: {v}"),
        (
            r"reservas.*internacionales",
            lambda v: f"reservas: USD {format_bcra_value(v)} millones",
        ),
    ]

    meta_info: Dict[str, Any] = {}
    if isinstance(variables, dict):
        candidate_meta = variables.get("_meta")
        if isinstance(candidate_meta, dict):
            meta_info = candidate_meta

    lines = ["variables principales bcra\n"]
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
        stale_msg = (
            "no hay actualización nueva del bcra, te muestro lo último que tengo"
        )
        if stale_msg not in lines:
            lines.append(stale_msg)

    if latest_dt:
        effective_today = today or datetime.now(BA_TZ).date()
        age_days = (effective_today - latest_dt.date()).days
        if age_days >= 3:
            lines.append(
                f"datos del bcra con {age_days} días de atraso, chequeá más tarde"
            )

    return "\n".join(lines)


__all__ = ["format_bcra_value", "format_bcra_variables", "normalize_text"]
