from __future__ import annotations

from os import environ
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from api.ai_pricing import MODEL_PRICING_USD_MICROS
from api.constants import BILLING_UNAVAILABLE_MESSAGE
from api.credit_units import format_credit_units, parse_credit_units

CommandResponse = Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]


class AdminCommandDeps(Protocol):
    credits_db_service: Any
    admin_report: Any


def _get_admin_chat_id() -> str:
    return str(environ.get("ADMIN_CHAT_ID") or "").strip()


def _require_billing(deps: AdminCommandDeps, command: str) -> Optional[CommandResponse]:
    if bool(deps.credits_db_service.is_configured()):
        return None
    return BILLING_UNAVAILABLE_MESSAGE, None, False, command


def handle_admin_printcredits_command(
    deps: AdminCommandDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> CommandResponse:
    if command != "/printcredits":
        return None, None, False, None

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id or str(user_id or "") != admin_chat_id:
        return "este comando es solo para el admin", None, False, command

    billing_required_response = _require_billing(deps, command)
    if billing_required_response is not None:
        return billing_required_response

    amount_token = sanitized_message_text.split(" ", 1)[0].strip()
    amount = parse_credit_units(amount_token)
    if amount is None:
        return "mandalo bien: /printcredits <monto>", None, False, command

    if amount <= 0:
        return (
            "el monto tiene que ser mayor a 0, no me rompas las bolas",
            None,
            False,
            command,
        )

    if user_id is None:
        return "se trabó imprimiendo créditos, probá de nuevo", None, False, command

    try:
        mint_result = deps.credits_db_service.mint_user_credits(
            user_id=user_id,
            amount=amount,
            actor_user_id=user_id,
        )
    except Exception as error:
        deps.admin_report(
            "Error minting credits with /printcredits",
            error,
            {"chat_id": chat_id, "user_id": user_id, "amount": amount},
        )
        return "se trabó imprimiendo créditos, probá de nuevo", None, False, command

    return (
        (
            f"listo, te imprimí {format_credit_units(amount)} créditos\n"
            f"te quedaron {format_credit_units(mint_result.get('user_balance', 0))}"
        ),
        None,
        False,
        command,
    )


def build_creditlog_lines(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    def _summarize_models(items: Sequence[Mapping[str, Any]]) -> str:
        totals: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("model") or "?")
            totals[name] = totals.get(name, 0) + int(item.get("usd_micros") or 0)
        if not totals:
            return "sin modelos"
        ordered = sorted(totals.items(), key=lambda entry: (-entry[1], entry[0]))
        visible = ordered[:5]
        summary = ", ".join(f"{name}={usd}" for name, usd in visible)
        hidden_count = len(ordered) - len(visible)
        if hidden_count > 0:
            summary += f", +{hidden_count} más"
        return summary

    def _summarize_tools(items: Sequence[Mapping[str, Any]]) -> str:
        totals: Dict[str, Dict[str, int]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("tool") or "?")
            current = totals.setdefault(name, {"usd_micros": 0, "count": 0})
            current["usd_micros"] += int(item.get("usd_micros") or 0)
            current["count"] += int(item.get("count") or 0)
        if not totals:
            return "sin tools"
        ordered = sorted(
            totals.items(),
            key=lambda entry: (-entry[1]["usd_micros"], -entry[1]["count"], entry[0]),
        )
        visible = ordered[:5]
        summary = ", ".join(
            f"{name}={values['usd_micros']} ({values['count']}x)"
            for name, values in visible
        )
        hidden_count = len(ordered) - len(visible)
        if hidden_count > 0:
            summary += f", +{hidden_count} más"
        return summary

    def _summarize_segments(items: Sequence[Mapping[str, Any]]) -> str:
        totals: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("source") or "").strip().lower() == "cache":
                continue
            kind = str(item.get("kind") or "unknown")
            totals[kind] = totals.get(kind, 0) + 1
        if not totals:
            return "sin segmentos"
        ordered = sorted(totals.items(), key=lambda entry: entry[0])
        return ", ".join(f"{kind}={count}" for kind, count in ordered)

    def _summarize_cache_hits(items: Sequence[Mapping[str, Any]]) -> Optional[str]:
        totals: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("source") or "").strip().lower() != "cache":
                continue
            kind = str(item.get("kind") or "unknown")
            totals[kind] = totals.get(kind, 0) + 1
        if not totals:
            return None
        ordered = sorted(totals.items(), key=lambda entry: entry[0])
        return ", ".join(f"{kind}={count}" for kind, count in ordered)

    def _summarize_cache(items: Sequence[Mapping[str, Any]]) -> Optional[str]:
        total_cached_tokens = 0
        total_cached_savings_usd_micros = 0
        for item in items:
            if not isinstance(item, Mapping):
                continue
            cached_tokens = int(item.get("input_cached_tokens") or 0)
            non_cached_tokens = int(item.get("input_non_cached_tokens") or 0)
            input_tokens = int(item.get("input_tokens") or 0)
            if cached_tokens <= 0:
                continue
            model_name = str(item.get("model") or "")
            pricing = MODEL_PRICING_USD_MICROS.get(model_name) or {}
            input_per_million = int(pricing.get("input_per_million") or 0)
            cached_input_per_million = int(
                pricing.get("cached_input_per_million") or input_per_million
            )
            total_cached_tokens += cached_tokens
            if input_per_million > cached_input_per_million:
                total_cached_savings_usd_micros += (
                    cached_tokens * (input_per_million - cached_input_per_million)
                ) // 1_000_000
            elif input_tokens > 0 and non_cached_tokens == 0:
                continue
        if total_cached_tokens <= 0:
            return None
        return f"cacheados={total_cached_tokens} ahorro_cache={total_cached_savings_usd_micros}"

    lines: List[str] = ["últimas liquidaciones IA:"]
    for entry in entries:
        raw_metadata = entry.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        model_breakdown = metadata.get("model_breakdown") or []
        tool_breakdown = metadata.get("tool_breakdown") or []
        billing_segments = metadata.get("billing_segments") or []
        command = str(
            metadata.get("command") or metadata.get("usage_tag") or "sin comando"
        )
        created_at = str(entry.get("created_at") or "")
        created_label = created_at.replace("T", " ")[:19] if created_at else "sin fecha"
        reserved_total = int(
            metadata.get("reserved_credit_units_total")
            or metadata.get("reserved_credit_units")
            or metadata.get("reserved_credits_total")
            or metadata.get("reserved_credits")
            or 0
        )
        settled_credits = int(
            metadata.get("settled_credit_units") or metadata.get("settled_credits") or 0
        )
        refunded_credits = int(
            metadata.get("refunded_credit_units")
            or metadata.get("refunded_credits")
            or 0
        )
        extra_charged_credits = int(
            metadata.get("extra_charged_credit_units")
            or metadata.get("extra_charged_credits")
            or 0
        )
        debt_applied_credits = int(
            metadata.get("debt_applied_credit_units")
            or metadata.get("debt_applied_credits")
            or 0
        )
        raw_usd_micros = int(metadata.get("raw_usd_micros") or 0)
        chat_value = metadata.get("chat_id", entry.get("chat_id"))
        user_value = metadata.get("user_id", entry.get("user_id"))
        if bool(metadata.get("billing_zero_usage_fallback")):
            status_label = "estado=groq_zero_usage"
        elif bool(metadata.get("missing_usage_billing")):
            status_label = "estado=missing_usage"
        else:
            status_label = "estado=ok"
        model_summary = _summarize_models(model_breakdown)
        tool_summary = _summarize_tools(tool_breakdown)
        segment_summary = _summarize_segments(billing_segments)
        cache_hit_summary = _summarize_cache_hits(billing_segments)
        cache_summary = _summarize_cache(model_breakdown)
        detail_lines = [
            f"{created_label} | cmd={command} | {status_label}",
            (
                f"chat={chat_value} user={user_value} "
                f"reservado={format_credit_units(reserved_total)} "
                f"cobrado={format_credit_units(settled_credits)} "
                f"refund={format_credit_units(refunded_credits)} "
                f"extra={format_credit_units(extra_charged_credits)} "
                f"deuda={format_credit_units(debt_applied_credits)}"
            ),
            f"usd_micros={raw_usd_micros}",
            f"requests: {segment_summary}",
        ]
        if cache_hit_summary:
            detail_lines.append(f"cache_hits: {cache_hit_summary}")
        if cache_summary:
            detail_lines.append(cache_summary)
        detail_lines.extend(
            [
                f"modelos: {model_summary}",
                f"tools: {tool_summary}",
            ]
        )
        lines.append("\n".join(detail_lines))
    return lines


def truncate_creditlog_message(text: str, max_length: int = 3500) -> str:
    if len(text) <= max_length:
        return text
    suffix = "\n\n[truncado]"
    return text[: max_length - len(suffix)].rstrip() + suffix


def handle_admin_creditlog_command(
    deps: AdminCommandDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> CommandResponse:
    if command != "/creditlog":
        return None, None, False, None

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id or str(user_id or "") != admin_chat_id:
        return "este comando es solo para el admin", None, False, command

    billing_required_response = _require_billing(deps, command)
    if billing_required_response is not None:
        return billing_required_response

    raw_limit = str(sanitized_message_text or "").strip()
    limit = 10
    if raw_limit:
        try:
            limit = int(raw_limit.split(" ", 1)[0].strip())
        except (TypeError, ValueError):
            return "mandalo bien: /creditlog [limite]", None, False, command
    limit = max(1, min(limit, 25))

    try:
        entries = deps.credits_db_service.list_recent_ai_settlement_results(limit=limit)
    except Exception as error:
        deps.admin_report(
            "Error loading /creditlog",
            error,
            {"chat_id": chat_id, "user_id": user_id, "limit": limit},
        )
        return "se trabó leyendo el creditlog, probá de nuevo", None, False, command

    if not entries:
        return "no hay liquidaciones ia recientes", None, False, command

    return (
        truncate_creditlog_message("\n\n".join(build_creditlog_lines(entries))),
        None,
        False,
        command,
    )


__all__ = [
    "build_creditlog_lines",
    "handle_admin_creditlog_command",
    "handle_admin_printcredits_command",
    "truncate_creditlog_message",
]
