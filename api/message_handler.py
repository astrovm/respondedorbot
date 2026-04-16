from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from api.ai_billing import AIMessageBilling
from api.ai_pricing import (
    MODEL_PRICING_USD_MICROS,
    estimate_transcribe_reserve_credits,
)
from api.ai_service import AIService, run_ai_flow as _run_ai_flow
from api.chat_context import format_user_identity
from api.credit_units import format_credit_units, parse_credit_units


CommandTuple = Tuple[Callable[..., str], bool, bool]
_BILLING_UNAVAILABLE_MESSAGE = "el cobro de ia no está andando, avisale al admin"


@dataclass(frozen=True)
class MessageHandlerDeps:
    config_redis: Callable[[], Any]
    get_chat_config: Callable[[Any, str], Dict[str, Any]]
    initialize_commands: Callable[[], Dict[str, CommandTuple]]
    parse_command: Callable[[str, str], Tuple[str, str]]
    should_auto_process_media: Callable[
        [Mapping[str, CommandTuple], str, str, Mapping[str, Any]], bool
    ]
    extract_message_content: Callable[
        [Dict[str, Any]], Tuple[str, Optional[str], Optional[str]]
    ]
    replace_links: Callable[[str], Tuple[str, bool, List[str]]]
    send_msg: Callable[..., Optional[int]]
    send_animation: Callable[..., Optional[int]]
    delete_msg: Callable[[str, str], None]
    admin_report: Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
    get_bot_message_metadata: Callable[[Any, str, Any], Optional[Dict[str, Any]]]
    save_bot_message_metadata: Callable[[Any, str, Any, Mapping[str, Any]], None]
    build_reply_context_text: Callable[[Mapping[str, Any]], Optional[str]]
    build_message_links_context: Callable[[Mapping[str, Any]], str]
    should_gordo_respond: Callable[
        [
            Mapping[str, CommandTuple],
            str,
            str,
            Mapping[str, Any],
            Mapping[str, Any],
            Optional[Mapping[str, Any]],
        ],
        bool,
    ]
    format_user_message: Callable[[Dict[str, Any], str, Optional[str]], str]
    save_message_to_redis: Callable[[str, str, str, Any], None]
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]]
    build_ai_messages: Callable[
        [Dict[str, Any], List[Dict[str, Any]], str, Optional[str]], List[Dict[str, Any]]
    ]
    handle_ai_response: Callable[..., str]
    ask_ai: Callable[..., str]
    gen_random: Callable[[str], str]
    build_insufficient_credits_message: Callable[..., str]
    build_topup_keyboard: Callable[[], Dict[str, Any]]
    credits_db_service: Any
    is_group_chat_type: Callable[[Optional[str]], bool]
    extract_user_id: Callable[[Mapping[str, Any]], Optional[int]]
    extract_numeric_chat_id: Callable[[str], Optional[int]]
    maybe_grant_onboarding_credits: Callable[
        [Any, Callable[..., None], Optional[int]], None
    ]
    format_balance_command: Callable[..., str]
    handle_transcribe_with_message: Callable[[Dict[str, Any]], str]
    handle_transcribe_with_message_result: Callable[
        [Dict[str, Any]], Tuple[str, List[Dict[str, Any]]]
    ]
    check_provider_available: Callable[..., bool]
    has_openrouter_fallback: Callable[[], bool]
    handle_rate_limit: Callable[[str, Dict[str, Any]], str]
    handle_successful_payment_message: Callable[[Dict[str, Any]], str]
    handle_config_command: Callable[[str, str], Tuple[str, Dict[str, Any]]]
    is_chat_admin: Callable[..., bool]
    report_unauthorized_config_attempt: Callable[..., None]
    handle_transcribe: Callable[[], str]
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]]
    estimate_image_context_reserve_credits: Callable[[bytes, str], int]
    _transcribe_audio_file: Callable[
        ..., Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
    ]
    _transcription_error_message: Callable[..., Optional[str]]
    download_telegram_file: Callable[[str], Optional[bytes]]
    measure_audio_duration_seconds: Callable[[bytes], Optional[float]]
    resize_image_if_needed: Callable[[bytes], bytes]
    encode_image_to_base64: Callable[[bytes], str]
    load_persisted_reservation: Callable[[str], Optional[Mapping[str, Any]]] = (
        lambda _usage_tag: None
    )
    persist_reservation: Callable[[str, Mapping[str, Any]], None] = (
        lambda _usage_tag, _reservation: None
    )
    clear_persisted_reservation: Callable[[str], None] = lambda _usage_tag: None
    ai_service: Optional[AIService] = None


@dataclass
class PreparedMessage:
    message_text: str
    photo_file_id: Optional[str]
    audio_file_id: Optional[str]
    resized_image_data: Optional[bytes] = None
    early_response: Optional[str] = None
    audio_duration_seconds: float = 0.0


def _billing_is_available(deps: MessageHandlerDeps) -> bool:
    return bool(deps.credits_db_service.is_configured())


def _billing_unavailable_command_response(
    command: str,
) -> Tuple[str, None, bool, str]:
    return _BILLING_UNAVAILABLE_MESSAGE, None, False, command


def _get_admin_chat_id() -> str:
    return str(environ.get("ADMIN_CHAT_ID") or "").strip()


def _require_billing_for_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
) -> Optional[Tuple[str, None, bool, str]]:
    if _billing_is_available(deps):
        return None
    return _billing_unavailable_command_response(command)


def _store_user_message_if_present(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    message_id: str,
    message: Dict[str, Any],
    message_text: str,
    reply_context_text: Optional[str],
    redis_client: Any,
) -> None:
    if not message_text:
        return
    formatted_message = deps.format_user_message(
        message,
        message_text,
        reply_context_text,
    )
    deps.save_message_to_redis(chat_id, message_id, formatted_message, redis_client)


def _send_response_and_store_metadata(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    message_id: str,
    response_msg: str,
    response_markup: Optional[Dict[str, Any]],
    response_command: Optional[str],
    response_uses_ai: bool,
    redis_client: Any,
) -> None:
    sent_message_id = deps.send_msg(
        chat_id,
        response_msg,
        message_id,
        reply_markup=response_markup,
    )
    key = (
        f"bot_{sent_message_id}" if sent_message_id is not None else f"bot_{message_id}"
    )
    deps.save_message_to_redis(
        chat_id,
        key,
        response_msg,
        redis_client,
    )
    if sent_message_id is None:
        return
    metadata = (
        {
            "type": "command",
            "command": response_command,
            "uses_ai": response_uses_ai,
        }
        if response_command
        else {"type": "ai"}
    )
    deps.save_bot_message_metadata(
        redis_client,
        chat_id,
        sent_message_id,
        metadata,
    )


def _build_billing_helper(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    chat_type: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
    command: str,
    message: Mapping[str, Any],
    redis_client: Any = None,
    creditless_user_hourly_limit: int = 0,
) -> AIMessageBilling:
    return AIMessageBilling(
        credits_db_service=deps.credits_db_service,
        admin_reporter=deps.admin_report,
        gen_random_fn=deps.gen_random,
        build_insufficient_credits_message_fn=deps.build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda target_user_id: (
            deps.maybe_grant_onboarding_credits(
                deps.credits_db_service,
                deps.admin_report,
                target_user_id,
            )
        ),
        command=command,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
        message=message,
        redis_client=redis_client,
        creditless_user_hourly_limit=creditless_user_hourly_limit,
        load_persisted_reservation_fn=deps.load_persisted_reservation,
        persist_reservation_fn=deps.persist_reservation,
        clear_persisted_reservation_fn=deps.clear_persisted_reservation,
    )


def _prepare_message_content(
    message: Dict[str, Any],
    *,
    auto_process_media: bool,
    deps: MessageHandlerDeps,
    billing_helper: AIMessageBilling,
    redis_client: Any,
) -> PreparedMessage:
    message_text, photo_file_id, audio_file_id = deps.extract_message_content(message)
    audio_duration_seconds = 0.0

    if (
        auto_process_media
        and audio_file_id
        and not (
            message_text and message_text.strip().lower().startswith("/transcribe")
        )
    ):
        resolved_audio_duration = _resolve_audio_duration_seconds(
            message,
            audio_file_id=audio_file_id,
            deps=deps,
        )
        if resolved_audio_duration is None:
            if message_text:
                return PreparedMessage(
                    message_text=message_text,
                    photo_file_id=photo_file_id,
                    audio_file_id=audio_file_id,
                    audio_duration_seconds=0.0,
                )
            return PreparedMessage(
                message_text=message_text,
                photo_file_id=photo_file_id,
                audio_file_id=audio_file_id,
                audio_duration_seconds=0.0,
                early_response="ok",
            )
        audio_duration_seconds = resolved_audio_duration
        if (
            not deps.check_provider_available(
                scope="transcribe",
            )
            and not deps.has_openrouter_fallback()
        ):
            rate_limited_chat_id = str(
                cast(Mapping[str, Any], message.get("chat") or {}).get("id") or ""
            )
            return PreparedMessage(
                message_text=message_text,
                photo_file_id=photo_file_id,
                audio_file_id=audio_file_id,
                audio_duration_seconds=audio_duration_seconds,
                early_response=deps.handle_rate_limit(rate_limited_chat_id, message),
            )
        reserved_credits = estimate_transcribe_reserve_credits(audio_duration_seconds)
        media_charge_meta, media_charge_error = billing_helper.reserve_ai_credits(
            "auto_audio_media",
            reserved_credits,
            metadata={"audio_seconds": audio_duration_seconds},
        )
        if media_charge_error:
            return PreparedMessage(
                message_text=message_text,
                photo_file_id=photo_file_id,
                audio_file_id=audio_file_id,
                audio_duration_seconds=audio_duration_seconds,
                early_response=media_charge_error,
            )

        transcription, err, billing_segment = deps._transcribe_audio_file(
            audio_file_id, use_cache=False
        )
        if transcription:
            billing_helper.settle_reserved_ai_credits_batch(
                [media_charge_meta] if media_charge_meta else [],
                [billing_segment] if billing_segment else [],
                reason="auto_audio_media_success",
            )
            message_text = transcription
        else:
            billing_helper.refund_reserved_ai_credits(
                media_charge_meta, reason="auto_audio_transcribe_failed"
            )
            message_text = (
                deps._transcription_error_message(
                    err,
                    download_message="no pude bajar tu audio, mandalo de vuelta",
                    transcribe_message="mandame texto que no soy alexa, boludo",
                )
                or "mandame texto que no soy alexa, boludo"
            )

    resized_image_data: Optional[bytes] = None
    if (
        auto_process_media
        and photo_file_id
        and not (
            message_text and message_text.strip().lower().startswith("/transcribe")
        )
    ):
        image_data = deps.download_telegram_file(photo_file_id)
        if image_data:
            resized_image_data = deps.resize_image_if_needed(image_data)
            if not message_text:
                message_text = "que onda con esta foto"
        elif not message_text:
            message_text = "no pude ver tu foto, boludo"

    return PreparedMessage(
        message_text=message_text,
        photo_file_id=photo_file_id,
        audio_file_id=audio_file_id,
        resized_image_data=resized_image_data,
        audio_duration_seconds=audio_duration_seconds,
    )


def _extract_audio_duration_seconds(message: Mapping[str, Any]) -> float:
    for container in (
        message,
        cast(Mapping[str, Any], message.get("reply_to_message") or {}),
    ):
        for key in ("voice", "audio", "video", "video_note"):
            media = container.get(key)
            if isinstance(media, Mapping):
                try:
                    return max(0.0, float(media.get("duration") or 0.0))
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _resolve_audio_duration_seconds(
    message: Mapping[str, Any],
    *,
    audio_file_id: Optional[str],
    deps: MessageHandlerDeps,
) -> Optional[float]:
    metadata_duration = _extract_audio_duration_seconds(message)
    if metadata_duration > 0:
        return metadata_duration
    if not audio_file_id:
        return None
    media_bytes = deps.download_telegram_file(audio_file_id)
    if not media_bytes:
        return None
    measured_duration = deps.measure_audio_duration_seconds(media_bytes)
    if measured_duration is None or measured_duration <= 0:
        return None
    return measured_duration


def _handle_link_replacement(
    deps: MessageHandlerDeps,
    *,
    chat_config: Mapping[str, Any],
    message: Dict[str, Any],
    message_text: str,
    chat_id: str,
    message_id: str,
    redis_client: Any,
) -> bool:
    link_mode = str(chat_config.get("link_mode", "reply"))
    if link_mode == "off" or not message_text or message_text.startswith("/"):
        return False

    fixed_text, changed, original_links = deps.replace_links(message_text)
    if not changed:
        return False

    user_info = message.get("from", {})
    username = user_info.get("username")
    if username:
        shared_by = f"@{username}"
    else:
        name_parts = [
            part
            for part in (user_info.get("first_name"), user_info.get("last_name"))
            if part
        ]
        shared_by = " ".join(name_parts)

    if shared_by:
        fixed_text += f"\n\ncompartido por {shared_by}"

    link_context = deps.build_message_links_context({"text": fixed_text})
    stored_bot_message = fixed_text
    if link_context:
        stored_bot_message = f"{stored_bot_message}\n\n{link_context}"

    reply_id = message.get("reply_to_message", {}).get("message_id")
    reply_id = str(reply_id) if reply_id is not None else None

    if link_mode == "delete":
        deps.delete_msg(chat_id, message_id)
        if reply_id:
            sent_message_id = deps.send_msg(
                chat_id, fixed_text, reply_id, original_links
            )
        else:
            sent_message_id = deps.send_msg(chat_id, fixed_text, buttons=original_links)
        if sent_message_id is not None:
            deps.save_message_to_redis(
                chat_id,
                f"bot_{sent_message_id}",
                stored_bot_message,
                redis_client,
            )
        return True

    sent_message_id = deps.send_msg(
        chat_id, fixed_text, reply_id or message_id, original_links
    )
    if sent_message_id is not None:
        deps.save_message_to_redis(
            chat_id,
            f"bot_{sent_message_id}",
            stored_bot_message,
            redis_client,
        )
    return True


def _load_reply_metadata(
    deps: MessageHandlerDeps,
    *,
    redis_client: Any,
    chat_id: str,
    message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    reply_msg = message.get("reply_to_message")
    if not isinstance(reply_msg, Mapping):
        return None

    if reply_msg.get("from", {}).get("username") != environ.get("TELEGRAM_USERNAME"):
        return None

    reply_id = reply_msg.get("message_id")
    if reply_id is None:
        return None

    return deps.get_bot_message_metadata(redis_client, chat_id, reply_id)


def _save_replied_message_context(
    deps: MessageHandlerDeps,
    *,
    message: Dict[str, Any],
    chat_id: str,
    redis_client: Any,
) -> None:
    reply_msg = message.get("reply_to_message")
    if not isinstance(reply_msg, Mapping):
        return

    reply_text = deps.extract_message_content(cast(Dict[str, Any], reply_msg))[0]
    reply_id = str(reply_msg["message_id"])
    is_bot = reply_msg.get("from", {}).get("username", "") == environ.get(
        "TELEGRAM_USERNAME"
    )

    if not reply_text:
        return

    if is_bot:
        deps.save_message_to_redis(chat_id, f"bot_{reply_id}", reply_text, redis_client)
        return

    formatted_reply = deps.format_user_message(
        cast(Dict[str, Any], reply_msg), reply_text
    )
    deps.save_message_to_redis(chat_id, reply_id, formatted_reply, redis_client)


def _handle_config_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    chat_id: str,
    chat_type: str,
    message: Dict[str, Any],
    redis_client: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/config":
        return None, None, False, None

    if deps.is_group_chat_type(chat_type) and not deps.is_chat_admin(
        chat_id,
        message.get("from", {}).get("id"),
        redis_client=redis_client,
    ):
        denial_message = "solo los admins pueden tocar esta config, maestro"
        deps.send_msg(chat_id, denial_message, str(message.get("message_id")))
        deps.report_unauthorized_config_attempt(
            chat_id,
            message.get("from", {}),
            chat_type=chat_type,
            action="command:/config",
        )
        return "ok", None, False, None

    response_msg, response_markup = deps.handle_config_command(chat_id, chat_type)
    return response_msg, response_markup, False, command


def _handle_topup_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    chat_type: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/topup":
        return None, None, False, None

    billing_required_response = _require_billing_for_command(deps, command=command)
    if billing_required_response is not None:
        return billing_required_response

    if chat_type != "private":
        bot_username = str(environ.get("TELEGRAM_USERNAME") or "").strip("@")
        response_msg = (
            f"la recarga va por privado, boludo.\nabrime en @{bot_username}"
            if bot_username
            else "la recarga va por privado, abrime en dm"
        )
        return response_msg, None, False, command

    return "elegí cuánto querés cargar:", deps.build_topup_keyboard(), False, command


def _handle_balance_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    chat_type: str,
    chat_id: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/balance":
        return None, None, False, None

    billing_required_response = _require_billing_for_command(deps, command=command)
    if billing_required_response is not None:
        return billing_required_response

    if user_id is None or numeric_chat_id is None:
        return (
            "no te pude leer bien el usuario para ver los saldos",
            None,
            False,
            command,
        )

    try:
        deps.maybe_grant_onboarding_credits(
            deps.credits_db_service, deps.admin_report, user_id
        )
        response_msg = deps.format_balance_command(
            deps.credits_db_service,
            chat_type=chat_type,
            user_id=user_id,
            chat_id=numeric_chat_id,
        )
    except Exception as error:
        deps.admin_report(
            "Error loading balance",
            error,
            {"chat_id": chat_id, "user_id": user_id},
        )
        response_msg = "se trabó leyendo tu saldo, probá de nuevo"
    return response_msg, None, False, command


def _handle_admin_printcredits_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/printcredits":
        return None, None, False, None

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id or str(user_id or "") != admin_chat_id:
        return "este comando es solo para el admin", None, False, command

    billing_required_response = _require_billing_for_command(deps, command=command)
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

    try:
        mint_result = deps.credits_db_service.mint_user_credits(
            user_id=int(user_id),
            amount=amount,
            actor_user_id=int(user_id),
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


def _build_creditlog_lines(entries: Sequence[Mapping[str, Any]]) -> List[str]:
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


def _truncate_creditlog_message(text: str, max_length: int = 3500) -> str:
    if len(text) <= max_length:
        return text
    suffix = "\n\n[truncado]"
    return text[: max_length - len(suffix)].rstrip() + suffix


def _handle_admin_creditlog_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/creditlog":
        return None, None, False, None

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id or str(user_id or "") != admin_chat_id:
        return "este comando es solo para el admin", None, False, command

    billing_required_response = _require_billing_for_command(deps, command=command)
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
        _truncate_creditlog_message("\n\n".join(_build_creditlog_lines(entries))),
        None,
        False,
        command,
    )


def _handle_transfer_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    chat_type: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/transfer":
        return None, None, False, None

    billing_required_response = _require_billing_for_command(deps, command=command)
    if billing_required_response is not None:
        return billing_required_response

    if not deps.is_group_chat_type(chat_type):
        return "esto es para grupos, capo: /transfer <monto>", None, False, command

    if user_id is None or numeric_chat_id is None:
        return (
            "no te pude sacar bien el usuario o el grupo para transferir",
            None,
            False,
            command,
        )

    amount_token = sanitized_message_text.split(" ", 1)[0].strip()
    amount = parse_credit_units(amount_token)
    if amount is None:
        return "mandalo bien: /transfer <monto>", None, False, command

    if amount <= 0:
        return (
            "el monto tiene que ser mayor a 0, no me rompas las bolas",
            None,
            False,
            command,
        )

    try:
        transfer_result = deps.credits_db_service.transfer_user_to_chat(
            user_id=user_id,
            chat_id=numeric_chat_id,
            amount=amount,
        )
    except Exception as error:
        deps.admin_report(
            "Error transferring credits",
            error,
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "amount": amount,
            },
        )
        return "se trabó la transferencia, probá de nuevo", None, False, command

    if transfer_result.get("ok"):
        response_msg = (
            f"listo, le pasé {format_credit_units(amount)} créditos al grupo\n"
            f"- lo tuyo: {format_credit_units(transfer_result.get('user_balance', 0))}\n"
            f"- lo del grupo: {format_credit_units(transfer_result.get('chat_balance', 0))}"
        )
        return response_msg, None, False, command

    response_msg = (
        "no te alcanza lo tuyo para pasar esa guita al grupo\n"
        f"te quedan: {format_credit_units(transfer_result.get('user_balance', 0))}"
    )
    return response_msg, None, False, command


def _handle_non_ai_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    commands: Mapping[str, CommandTuple],
    sanitized_message_text: str,
    message: Dict[str, Any],
    chat_id: str,
    redis_client: Any,
    billing_helper: AIMessageBilling,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command not in commands:
        return None, None, False, None

    handler_func, uses_ai, takes_params = commands[command]
    if uses_ai:
        return None, None, False, None

    if command == "/transcribe":
        reserve_credits = 0
        replied_message = cast(Mapping[str, Any], message.get("reply_to_message") or {})
        if replied_message.get("voice") or replied_message.get("audio"):
            replied_audio_file_id = None
            voice = replied_message.get("voice")
            if isinstance(voice, Mapping):
                replied_audio_file_id = str(voice.get("file_id") or "") or None
            audio = replied_message.get("audio")
            if not replied_audio_file_id and isinstance(audio, Mapping):
                replied_audio_file_id = str(audio.get("file_id") or "") or None
            audio_duration_seconds = _resolve_audio_duration_seconds(
                message,
                audio_file_id=replied_audio_file_id,
                deps=deps,
            )
            if audio_duration_seconds is None:
                return "ok", None, False, None
            if (
                not deps.check_provider_available(
                    scope="transcribe",
                )
                and not deps.has_openrouter_fallback()
            ):
                return deps.handle_rate_limit(chat_id, message), None, False, command
            reserve_credits = estimate_transcribe_reserve_credits(
                audio_duration_seconds
            )
        else:
            replied_photo_file_id = deps.extract_message_content(message)[1]
            replied_image_data = (
                deps.download_telegram_file(replied_photo_file_id)
                if replied_photo_file_id
                else None
            )
            if (
                isinstance(replied_image_data, (bytes, bytearray))
                and replied_image_data
            ):
                resized_image = deps.resize_image_if_needed(bytes(replied_image_data))
                if (
                    not deps.check_provider_available(
                        scope="vision",
                    )
                    and not deps.has_openrouter_fallback()
                ):
                    return (
                        deps.handle_rate_limit(chat_id, message),
                        None,
                        False,
                        command,
                    )
                reserve_credits = deps.estimate_image_context_reserve_credits(
                    resized_image,
                    "Describe what you see in this image in detail.",
                )
            else:
                reserve_credits = 1

        media_charge_meta, media_charge_error = billing_helper.reserve_ai_credits(
            "transcribe_command_media",
            reserve_credits,
        )
        if media_charge_error:
            return media_charge_error, None, False, command

        response_msg, billing_segments = deps.handle_transcribe_with_message_result(
            message
        )
        transcribe_succeeded = bool(
            billing_segments
        ) or billing_helper.is_transcribe_success_response(response_msg)
        if media_charge_meta and not transcribe_succeeded:
            billing_helper.refund_reserved_ai_credits(
                media_charge_meta, reason="transcribe_command_unsuccessful"
            )
        else:
            billing_helper.settle_reserved_ai_credits_batch(
                [media_charge_meta] if media_charge_meta else [],
                billing_segments,
                reason="transcribe_command_success",
            )
        return response_msg, None, False, command

    if command in ("/gm", "/gn"):
        gif_url = handler_func()
        msg_id = str(message.get("message_id", ""))
        if gif_url.startswith("http"):
            deps.send_animation(chat_id, gif_url, msg_id=msg_id)
            return None, None, False, command
        return gif_url, None, False, command

    if command in ("/tareas", "/tasks"):
        result = handler_func(chat_id)
        response_markup = None
        if isinstance(result, tuple):
            response_msg, response_markup = result
        else:
            response_msg = result
        return response_msg, response_markup, False, command

    response_msg = (
        handler_func(sanitized_message_text) if takes_params else handler_func()
    )
    return response_msg, None, False, command


def _handle_known_command(
    deps: MessageHandlerDeps,
    *,
    commands: Mapping[str, CommandTuple],
    command: str,
    sanitized_message_text: str,
    message: Dict[str, Any],
    chat_id: str,
    chat_type: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
    prepared_message: PreparedMessage,
    billing_helper: AIMessageBilling,
    reply_context_text: Optional[str],
    user_identity: str,
    redis_client: Any,
    timezone_offset: int = -3,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    response_markup: Optional[Dict[str, Any]] = None
    response_command: Optional[str] = None

    response = _handle_config_command(
        deps,
        command=command,
        chat_id=chat_id,
        chat_type=chat_type,
        message=message,
        redis_client=redis_client,
    )
    if response[0] is not None or response[3] is not None:
        return response

    response = _handle_topup_command(
        deps,
        command=command,
        chat_type=chat_type,
    )
    if response[0] is not None or response[3] is not None:
        return response

    response = _handle_balance_command(
        deps,
        command=command,
        chat_type=chat_type,
        chat_id=chat_id,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
    )
    if response[0] is not None or response[3] is not None:
        return response

    response = _handle_transfer_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
    )
    if response[0] is not None or response[3] is not None:
        return response

    response = _handle_admin_printcredits_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        user_id=user_id,
    )
    if response[0] is not None or response[3] is not None:
        return response

    response = _handle_admin_creditlog_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        user_id=user_id,
    )
    if response[0] is not None or response[3] is not None:
        return response

    if command in commands:
        handler_func, uses_ai, takes_params = commands[command]
        response_command = command

        if uses_ai:
            response_msg, response_uses_ai = _run_ai_flow(
                deps,
                chat_id=chat_id,
                message=message,
                user_id=user_id,
                prepared_message=prepared_message,
                billing_helper=billing_helper,
                prompt_text=sanitized_message_text,
                reply_context_text=reply_context_text,
                user_identity=user_identity,
                handler_func=handler_func,
                redis_client=redis_client,
                timezone_offset=timezone_offset,
            )
            return response_msg, response_markup, response_uses_ai, response_command

        response_msg, response_markup, _, response_command = _handle_non_ai_command(
            deps,
            command=command,
            commands=commands,
            sanitized_message_text=sanitized_message_text,
            message=message,
            chat_id=chat_id,
            redis_client=redis_client,
            billing_helper=billing_helper,
        )
        return response_msg, response_markup, False, response_command

    response_msg, response_uses_ai = _run_ai_flow(
        deps,
        chat_id=chat_id,
        message=message,
        user_id=user_id,
        prepared_message=prepared_message,
        billing_helper=billing_helper,
        prompt_text=prepared_message.message_text,
        reply_context_text=reply_context_text,
        user_identity=user_identity,
        handler_func=deps.ask_ai,
        redis_client=redis_client,
        timezone_offset=timezone_offset,
        is_spontaneous=True,
    )
    return response_msg, response_markup, response_uses_ai, response_command


def handle_msg(message: Dict[str, Any], deps: MessageHandlerDeps) -> str:
    """Handle an incoming Telegram message."""

    try:
        chat = cast(Dict[str, Any], message.get("chat", {}))
        sender = cast(Mapping[str, Any], message.get("from", {}))
        if not chat or chat.get("id") is None or not sender:
            return "ok"
        message_id = str(message.get("message_id"))
        chat_id = str(chat.get("id"))
        chat_type = str(chat.get("type", ""))
        user_identity = format_user_identity(sender)

        user_id = deps.extract_user_id(message)
        numeric_chat_id = deps.extract_numeric_chat_id(chat_id)

        if isinstance(message.get("successful_payment"), Mapping):
            return deps.handle_successful_payment_message(message)

        redis_client = deps.config_redis()
        chat_config = deps.get_chat_config(redis_client, chat_id)

        commands = deps.initialize_commands()
        bot_name = f"@{environ.get('TELEGRAM_USERNAME')}"
        raw_message_text, _, _ = deps.extract_message_content(message)
        command, _ = deps.parse_command(raw_message_text, bot_name)
        auto_process_media = deps.should_auto_process_media(
            commands,
            command,
            raw_message_text,
            message,
        )

        billing_helper = _build_billing_helper(
            deps,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            numeric_chat_id=numeric_chat_id,
            command=command,
            message=message,
            redis_client=redis_client,
            creditless_user_hourly_limit=int(
                chat_config.get(
                    "creditless_user_hourly_limit",
                    chat_config.get("creditless_user_daily_limit", 0),
                )
            ),
        )
        prepared_message = _prepare_message_content(
            message,
            auto_process_media=auto_process_media,
            deps=deps,
            billing_helper=billing_helper,
            redis_client=redis_client,
        )
        if prepared_message.early_response:
            if prepared_message.early_response != "ok":
                deps.send_msg(chat_id, prepared_message.early_response, message_id)
            return "ok"

        if _handle_link_replacement(
            deps,
            chat_config=chat_config,
            message=message,
            message_text=prepared_message.message_text,
            chat_id=chat_id,
            message_id=message_id,
            redis_client=redis_client,
        ):
            return "ok"

        command, sanitized_message_text = deps.parse_command(
            prepared_message.message_text, bot_name
        )
        reply_metadata = _load_reply_metadata(
            deps,
            redis_client=redis_client,
            chat_id=chat_id,
            message=message,
        )
        reply_context_text = deps.build_reply_context_text(message)

        should_respond = deps.should_gordo_respond(
            commands,
            command,
            sanitized_message_text,
            message,
            chat_config,
            reply_metadata,
        )
        if not should_respond:
            _store_user_message_if_present(
                deps,
                chat_id=chat_id,
                message_id=message_id,
                message=message,
                message_text=prepared_message.message_text,
                reply_context_text=reply_context_text,
                redis_client=redis_client,
            )
            return "ok"

        if (
            command in ["/comando", "/command"]
            and not sanitized_message_text
            and "reply_to_message" in message
        ):
            sanitized_message_text = deps.extract_message_content(
                cast(Dict[str, Any], message["reply_to_message"])
            )[0]

        _save_replied_message_context(
            deps,
            message=message,
            chat_id=chat_id,
            redis_client=redis_client,
        )

        response_msg, response_markup, response_uses_ai, response_command = (
            _handle_known_command(
                deps,
                commands=commands,
                command=command,
                sanitized_message_text=sanitized_message_text,
                message=message,
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                numeric_chat_id=numeric_chat_id,
                prepared_message=prepared_message,
                billing_helper=billing_helper,
                reply_context_text=reply_context_text,
                user_identity=user_identity,
                redis_client=redis_client,
                timezone_offset=int(chat_config.get("timezone_offset", -3)),
            )
        )

        if (
            response_msg == "ok"
            and response_command is None
            and response_markup is None
        ):
            return "ok"

        _store_user_message_if_present(
            deps,
            chat_id=chat_id,
            message_id=message_id,
            message=message,
            message_text=prepared_message.message_text,
            reply_context_text=reply_context_text,
            redis_client=redis_client,
        )

        if response_msg:
            _send_response_and_store_metadata(
                deps,
                chat_id=chat_id,
                message_id=message_id,
                response_msg=response_msg,
                response_markup=response_markup,
                response_command=response_command,
                response_uses_ai=response_uses_ai,
                redis_client=redis_client,
            )

        return "ok"

    except Exception as error:
        deps.admin_report(
            f"Message handling error: {error}",
            error,
            {
                "message_id": message.get("message_id"),
                "chat_id": message.get("chat", {}).get("id"),
                "user": message.get("from", {}).get("username", "Unknown"),
            },
        )
        return "error procesando mensaje"


__all__ = ["MessageHandlerDeps", "handle_msg"]
