from __future__ import annotations

from os import environ
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from api.ai.pricing import MODEL_PRICING_USD_MICROS
from api.core.constants import BILLING_UNAVAILABLE_MESSAGE
from api.billing.credit_units import format_credit_units, parse_credit_units

CommandResponse = Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]


class AdminCommandDeps(Protocol):
    @property
    def credits_db_service(self) -> Any: ...

    @property
    def admin_report(self) -> Any: ...


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


def _with_hidden_count(summary: str, total: int, visible: int) -> str:
    hidden = total - visible
    return f"{summary}, +{hidden} más" if hidden > 0 else summary


def _summarize_models(items: Sequence[object]) -> str:
    totals: Dict[str, int] = {}
    for item in items:
        if isinstance(item, Mapping):
            name = str(item.get("model") or "?")
            totals[name] = totals.get(name, 0) + int(item.get("usd_micros") or 0)
    if not totals:
        return "sin modelos"
    ordered = sorted(totals.items(), key=lambda entry: (-entry[1], entry[0]))
    visible = ordered[:5]
    summary = ", ".join(f"{name}={usd}" for name, usd in visible)
    return _with_hidden_count(summary, len(ordered), len(visible))


def _summarize_tools(items: Sequence[object]) -> str:
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
    return _with_hidden_count(summary, len(ordered), len(visible))


def _summarize_segments(
    items: Sequence[object],
    *,
    cache_only: bool,
) -> Optional[str]:
    totals: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        is_cache = str(item.get("source") or "").strip().lower() == "cache"
        if is_cache != cache_only:
            continue
        kind = str(item.get("kind") or "unknown")
        totals[kind] = totals.get(kind, 0) + 1
    if not totals:
        return None
    return ", ".join(f"{kind}={totals[kind]}" for kind in sorted(totals))


def _summarize_model_cache(items: Sequence[object]) -> Optional[str]:
    cached_tokens_total = 0
    savings_total = 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        cached_tokens = int(item.get("input_cached_tokens") or 0)
        if cached_tokens <= 0:
            continue
        model_name = str(item.get("model") or "")
        pricing = MODEL_PRICING_USD_MICROS.get(model_name) or {}
        input_price = int(pricing.get("input_per_million") or 0)
        cached_price = int(pricing.get("cached_input_per_million") or input_price)
        cached_tokens_total += cached_tokens
        if input_price > cached_price:
            savings_total += (
                cached_tokens * (input_price - cached_price)
            ) // 1_000_000
    if cached_tokens_total <= 0:
        return None
    return f"cacheados={cached_tokens_total} ahorro_cache={savings_total}"


def _metadata_credit(metadata: Mapping[str, Any], *keys: str) -> int:
    return int(next((metadata[key] for key in keys if metadata.get(key)), 0))


def _creditlog_status(metadata: Mapping[str, Any]) -> str:
    if metadata.get("billing_zero_usage_fallback"):
        return "estado=groq_zero_usage"
    if metadata.get("missing_usage_billing"):
        return "estado=missing_usage"
    return "estado=ok"


def _format_creditlog_entry(entry: Mapping[str, Any]) -> str:
    raw_metadata = entry.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    models = metadata.get("model_breakdown") or []
    tools = metadata.get("tool_breakdown") or []
    segments = metadata.get("billing_segments") or []
    created_at = str(entry.get("created_at") or "")
    created_label = created_at.replace("T", " ")[:19] if created_at else "sin fecha"
    command = str(metadata.get("command") or metadata.get("usage_tag") or "sin comando")
    detail_lines = [
        f"{created_label} | cmd={command} | {_creditlog_status(metadata)}",
        (
            f"chat={metadata.get('chat_id', entry.get('chat_id'))} "
            f"user={metadata.get('user_id', entry.get('user_id'))} "
            f"reservado={format_credit_units(_metadata_credit(metadata, 'reserved_credit_units_total', 'reserved_credit_units', 'reserved_credits_total', 'reserved_credits'))} "
            f"cobrado={format_credit_units(_metadata_credit(metadata, 'settled_credit_units', 'settled_credits'))} "
            f"refund={format_credit_units(_metadata_credit(metadata, 'refunded_credit_units', 'refunded_credits'))} "
            f"extra={format_credit_units(_metadata_credit(metadata, 'extra_charged_credit_units', 'extra_charged_credits'))} "
            f"deuda={format_credit_units(_metadata_credit(metadata, 'debt_applied_credit_units', 'debt_applied_credits'))}"
        ),
        f"usd_micros={int(metadata.get('raw_usd_micros') or 0)}",
        f"requests: {_summarize_segments(segments, cache_only=False) or 'sin segmentos'}",
    ]
    cache_hits = _summarize_segments(segments, cache_only=True)
    if cache_hits:
        detail_lines.append(f"cache_hits: {cache_hits}")
    cache_summary = _summarize_model_cache(models)
    if cache_summary:
        detail_lines.append(cache_summary)
    detail_lines.extend([
        f"modelos: {_summarize_models(models)}",
        f"tools: {_summarize_tools(tools)}",
    ])
    return "\n".join(detail_lines)


def build_creditlog_lines(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    return ["últimas liquidaciones IA:", *map(_format_creditlog_entry, entries)]


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
