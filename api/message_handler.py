"""Main Telegram message handling flow extracted from the Flask entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from api.ai_billing import AIMessageBilling
from api.chat_context import format_user_identity
from api.groq_billing import (
    IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE,
    estimate_transcribe_reserve_credits,
)


CommandTuple = Tuple[Callable[..., str], bool, bool]


@dataclass(frozen=True)
class MessageHandlerDeps:
    config_redis: Callable[[], Any]
    get_chat_config: Callable[[Any, str], Dict[str, Any]]
    initialize_commands: Callable[[], Dict[str, CommandTuple]]
    parse_command: Callable[[str, str], Tuple[str, str]]
    should_auto_process_media: Callable[[Mapping[str, CommandTuple], str, str, Mapping[str, Any]], bool]
    extract_message_content: Callable[[Dict[str, Any]], Tuple[str, Optional[str], Optional[str]]]
    replace_links: Callable[[str], Tuple[str, bool, List[str]]]
    send_msg: Callable[..., Optional[int]]
    delete_msg: Callable[[str, str], None]
    admin_report: Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
    get_bot_message_metadata: Callable[[Any, str, Any], Optional[Dict[str, Any]]]
    save_bot_message_metadata: Callable[[Any, str, Any, Mapping[str, Any]], None]
    build_reply_context_text: Callable[[Mapping[str, Any]], Optional[str]]
    should_gordo_respond: Callable[
        [Mapping[str, CommandTuple], str, str, Mapping[str, Any], Mapping[str, Any], Optional[Mapping[str, Any]]],
        bool,
    ]
    format_user_message: Callable[[Dict[str, Any], str, Optional[str]], str]
    save_message_to_redis: Callable[[str, str, str, Any], None]
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]]
    build_ai_messages: Callable[[Dict[str, Any], List[Dict[str, Any]], str, Optional[str]], List[Dict[str, Any]]]
    handle_ai_response: Callable[..., str]
    ask_ai: Callable[..., str]
    gen_random: Callable[[str], str]
    build_insufficient_credits_message: Callable[..., str]
    get_ai_credits_per_response: Callable[[], int]
    build_topup_keyboard: Callable[[], Dict[str, Any]]
    credits_db_service: Any
    is_group_chat_type: Callable[[Optional[str]], bool]
    extract_user_id: Callable[[Mapping[str, Any]], Optional[int]]
    extract_numeric_chat_id: Callable[[str], Optional[int]]
    maybe_grant_onboarding_credits: Callable[[Any, Callable[..., None], Optional[int]], None]
    format_balance_command: Callable[..., str]
    handle_transcribe_with_message: Callable[[Dict[str, Any]], str]
    handle_transcribe_with_message_result: Callable[[Dict[str, Any]], Tuple[str, List[Dict[str, Any]]]]
    check_global_rate_limit: Callable[[Any], bool]
    handle_rate_limit: Callable[[str, Dict[str, Any]], str]
    handle_successful_payment_message: Callable[[Dict[str, Any]], str]
    handle_config_command: Callable[[str], Tuple[str, Dict[str, Any]]]
    ensure_callback_updates_enabled: Callable[[], None]
    is_chat_admin: Callable[..., bool]
    report_unauthorized_config_attempt: Callable[..., None]
    handle_transcribe: Callable[[], str]
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]]
    estimate_image_context_reserve_credits: Callable[[bytes, str], int]
    _transcribe_audio_file: Callable[..., Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]
    _transcription_error_message: Callable[..., Optional[str]]
    download_telegram_file: Callable[[str], Optional[bytes]]
    measure_audio_duration_seconds: Callable[[bytes], Optional[float]]
    resize_image_if_needed: Callable[[bytes], bytes]
    encode_image_to_base64: Callable[[bytes], str]


@dataclass
class PreparedMessage:
    message_text: str
    photo_file_id: Optional[str]
    audio_file_id: Optional[str]
    resized_image_data: Optional[bytes] = None
    early_response: Optional[str] = None
    audio_duration_seconds: float = 0.0


def _build_billing_helper(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    chat_type: str,
    user_id: Optional[int],
    numeric_chat_id: Optional[int],
    command: str,
    message: Mapping[str, Any],
) -> AIMessageBilling:
    return AIMessageBilling(
        credits_db_service=deps.credits_db_service,
        admin_reporter=deps.admin_report,
        gen_random_fn=deps.gen_random,
        build_insufficient_credits_message_fn=deps.build_insufficient_credits_message,
        maybe_grant_onboarding_credits_fn=lambda target_user_id: deps.maybe_grant_onboarding_credits(
            deps.credits_db_service,
            deps.admin_report,
            target_user_id,
        ),
        get_ai_credits_per_response_fn=deps.get_ai_credits_per_response,
        command=command,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
        message=message,
    )


def _prepare_message_content(
    message: Dict[str, Any],
    *,
    auto_process_media: bool,
    deps: MessageHandlerDeps,
    billing_helper: AIMessageBilling,
) -> PreparedMessage:
    message_text, photo_file_id, audio_file_id = deps.extract_message_content(message)
    audio_duration_seconds = 0.0

    if auto_process_media and audio_file_id and not (
        message_text and message_text.strip().lower().startswith("/transcribe")
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
            billing_helper.settle_reserved_ai_credits(
                media_charge_meta,
                [billing_segment] if billing_segment else [],
                reason="auto_audio_media_success",
            )
            message_text = transcription
        else:
            billing_helper.refund_reserved_ai_credits(
                media_charge_meta, reason="auto_audio_transcribe_failed"
            )
            message_text = deps._transcription_error_message(
                err,
                download_message="no pude bajar tu audio, mandalo de vuelta",
                transcribe_message="mandame texto que no soy alexa, boludo",
            ) or "mandame texto que no soy alexa, boludo"

    resized_image_data: Optional[bytes] = None
    if auto_process_media and photo_file_id and not (
        message_text and message_text.strip().lower().startswith("/transcribe")
    ):
        image_data = deps.download_telegram_file(photo_file_id)
        if image_data:
            resized_image_data = deps.resize_image_if_needed(image_data)
            deps.encode_image_to_base64(resized_image_data)
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
        voice = container.get("voice")
        if isinstance(voice, Mapping):
            try:
                return max(0.0, float(voice.get("duration") or 0.0))
            except (TypeError, ValueError):
                return 0.0
        audio = container.get("audio")
        if isinstance(audio, Mapping):
            try:
                return max(0.0, float(audio.get("duration") or 0.0))
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


def _select_main_billing_segments(
    billing_segments: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        dict(segment)
        for segment in billing_segments
        if str(segment.get("kind") or "") != "vision"
    ]


def _handle_link_replacement(
    deps: MessageHandlerDeps,
    *,
    chat_config: Mapping[str, Any],
    message: Dict[str, Any],
    message_text: str,
    chat_id: str,
    message_id: str,
) -> bool:
    link_mode = str(chat_config.get("link_mode", "off"))
    if link_mode == "off" or not message_text or message_text.startswith("/"):
        return False

    fixed_text, changed, original_links = deps.replace_links(message_text)
    if not changed:
        return "http://" in message_text or "https://" in message_text

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

    reply_id = message.get("reply_to_message", {}).get("message_id")
    reply_id = str(reply_id) if reply_id is not None else None

    if link_mode == "delete":
        deps.delete_msg(chat_id, message_id)
        if reply_id:
            deps.send_msg(chat_id, fixed_text, reply_id, original_links)
        else:
            deps.send_msg(chat_id, fixed_text, buttons=original_links)
        return True

    deps.send_msg(chat_id, fixed_text, reply_id or message_id, original_links)
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

    formatted_reply = deps.format_user_message(cast(Dict[str, Any], reply_msg), reply_text)
    deps.save_message_to_redis(chat_id, reply_id, formatted_reply, redis_client)


def _run_ai_flow(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    message: Dict[str, Any],
    prepared_message: PreparedMessage,
    billing_helper: AIMessageBilling,
    prompt_text: str,
    reply_context_text: Optional[str],
    user_identity: str,
    handler_func: Callable[..., str],
    redis_client: Any,
) -> Tuple[str, bool]:
    if not deps.check_global_rate_limit(redis_client):
        return deps.handle_rate_limit(chat_id, message), False

    chat_history = deps.get_chat_history(chat_id, redis_client)
    ai_messages = deps.build_ai_messages(
        message,
        chat_history,
        prompt_text,
        reply_context_text,
    )

    main_reserve_credits, reserve_meta = deps.estimate_ai_base_reserve_credits(
        ai_messages,
        extra_input_tokens=(
            IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE
            if prepared_message.resized_image_data and prepared_message.photo_file_id
            else 0
        ),
    )
    base_charge_meta, base_charge_error = billing_helper.reserve_ai_credits(
        "ai_response_base",
        main_reserve_credits,
        metadata={
            "estimated_prompt_messages": len(ai_messages),
            **reserve_meta,
        },
    )
    if base_charge_error:
        return base_charge_error, False

    media_charge_meta: Optional[Dict[str, Any]] = None
    if prepared_message.resized_image_data and prepared_message.photo_file_id:
        media_charge_meta, media_charge_error = billing_helper.reserve_ai_credits(
            "image_context_media",
            deps.estimate_image_context_reserve_credits(
                prepared_message.resized_image_data,
                "Describe what you see in this image in detail.",
            ),
            metadata={"photo_file_id": prepared_message.photo_file_id},
        )
        if media_charge_error:
            billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="image_context_reserve_failed"
            )
            return media_charge_error, False

    ai_response_meta: Dict[str, Any] = {}
    response_msg = deps.handle_ai_response(
        chat_id,
        handler_func,
        ai_messages,
        image_data=prepared_message.resized_image_data if prepared_message.photo_file_id else None,
        image_file_id=prepared_message.photo_file_id,
        context_texts=[reply_context_text],
        user_identity=user_identity,
        response_meta=ai_response_meta,
    )

    billing_segments = list(ai_response_meta.get("billing_segments") or [])
    if response_msg == "me quedé reculando y no te pude responder, probá de nuevo" or bool(
        ai_response_meta.get("ai_fallback")
    ):
        if media_charge_meta:
            billing_helper.refund_reserved_ai_credits(
                media_charge_meta, reason="ai_response_fallback"
            )
        billing_helper.refund_reserved_ai_credits(
            base_charge_meta, reason="ai_response_fallback"
        )
        return response_msg, True

    if media_charge_meta:
        image_segments = [
            dict(segment)
            for segment in billing_segments
            if str(segment.get("kind") or "") == "vision"
        ]
        billing_helper.settle_reserved_ai_credits(
            media_charge_meta,
            image_segments,
            reason="image_context_media_success",
        )

    billing_helper.settle_reserved_ai_credits(
        base_charge_meta,
        _select_main_billing_segments(billing_segments),
        reason="ai_response_success",
    )

    return response_msg, True


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

    response_msg, response_markup = deps.handle_config_command(chat_id)
    return response_msg, response_markup, False, command


def _handle_topup_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    chat_type: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command != "/topup":
        return None, None, False, None

    if not deps.credits_db_service.is_configured():
        return "el cobro de ia no está andando, avisale al admin", None, False, command

    if chat_type != "private":
        bot_username = str(environ.get("TELEGRAM_USERNAME") or "").strip("@")
        response_msg = (
            "la recarga va por privado, boludo.\n"
            f"abrime en @{bot_username}"
            if bot_username
            else "la recarga va por privado, abrime en dm"
        )
        return response_msg, None, False, command

    deps.ensure_callback_updates_enabled()
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

    if not deps.credits_db_service.is_configured():
        return "el cobro de ia no está andando, avisale al admin", None, False, command

    if user_id is None or numeric_chat_id is None:
        return "no te pude leer bien el usuario para ver los saldos", None, False, command

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

    if not deps.credits_db_service.is_configured():
        return "el cobro de ia no está andando, avisale al admin", None, False, command

    if not deps.is_group_chat_type(chat_type):
        return "esto es para grupos, capo: /transfer <monto>", None, False, command

    if user_id is None or numeric_chat_id is None:
        return "no te pude sacar bien el usuario o el grupo para transferir", None, False, command

    amount_token = sanitized_message_text.split(" ", 1)[0].strip()
    try:
        amount = int(amount_token)
    except (TypeError, ValueError):
        return "mandalo bien: /transfer <monto>", None, False, command

    if amount <= 0:
        return "el monto tiene que ser mayor a 0, no me rompas las bolas", None, False, command

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
            f"listo, le pasé {amount} créditos al grupo\n"
            f"- lo tuyo: {int(transfer_result.get('user_balance', 0))}\n"
            f"- lo del grupo: {int(transfer_result.get('chat_balance', 0))}"
        )
        return response_msg, None, False, command

    response_msg = (
        "no te alcanza lo tuyo para pasar esa guita al grupo\n"
        f"te quedan: {int(transfer_result.get('user_balance', 0))}"
    )
    return response_msg, None, False, command


def _handle_non_ai_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    commands: Mapping[str, CommandTuple],
    sanitized_message_text: str,
    message: Dict[str, Any],
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
            reserve_credits = estimate_transcribe_reserve_credits(audio_duration_seconds)
        else:
            replied_photo_file_id = deps.extract_message_content(message)[1]
            replied_image_data = (
                deps.download_telegram_file(replied_photo_file_id)
                if replied_photo_file_id
                else None
            )
            if isinstance(replied_image_data, (bytes, bytearray)) and replied_image_data:
                reserve_credits = deps.estimate_image_context_reserve_credits(
                    deps.resize_image_if_needed(bytes(replied_image_data)),
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

        response_msg, billing_segments = deps.handle_transcribe_with_message_result(message)
        transcribe_succeeded = bool(billing_segments) or billing_helper.is_transcribe_success_response(
            response_msg
        )
        if media_charge_meta and not transcribe_succeeded:
            billing_helper.refund_reserved_ai_credits(
                media_charge_meta, reason="transcribe_command_unsuccessful"
            )
        else:
            billing_helper.settle_reserved_ai_credits(
                media_charge_meta,
                billing_segments,
                reason="transcribe_command_success",
            )
        return response_msg, None, False, command

    response_msg = handler_func(sanitized_message_text) if takes_params else handler_func()
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

    if command in {"/buscar", "/search"}:
        response_command = command
        handler_func, uses_ai, takes_params = commands[command]
        response_msg = handler_func(sanitized_message_text) if takes_params else handler_func()
        return response_msg, response_markup, uses_ai, response_command

    if command in commands:
        handler_func, uses_ai, takes_params = commands[command]
        response_command = command

        if uses_ai:
            response_msg, response_uses_ai = _run_ai_flow(
                deps,
                chat_id=chat_id,
                message=message,
                prepared_message=prepared_message,
                billing_helper=billing_helper,
                prompt_text=sanitized_message_text,
                reply_context_text=reply_context_text,
                user_identity=user_identity,
                handler_func=handler_func,
                redis_client=redis_client,
            )
            return response_msg, response_markup, response_uses_ai, response_command

        response_msg, response_markup, _, response_command = _handle_non_ai_command(
            deps,
            command=command,
            commands=commands,
            sanitized_message_text=sanitized_message_text,
            message=message,
            billing_helper=billing_helper,
        )
        return response_msg, response_markup, False, response_command

    response_msg, response_uses_ai = _run_ai_flow(
        deps,
        chat_id=chat_id,
        message=message,
        prepared_message=prepared_message,
        billing_helper=billing_helper,
        prompt_text=prepared_message.message_text,
        reply_context_text=reply_context_text,
        user_identity=user_identity,
        handler_func=deps.ask_ai,
        redis_client=redis_client,
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
        )
        prepared_message = _prepare_message_content(
            message,
            auto_process_media=auto_process_media,
            deps=deps,
            billing_helper=billing_helper,
        )
        if prepared_message.early_response:
            return prepared_message.early_response

        if _handle_link_replacement(
            deps,
            chat_config=chat_config,
            message=message,
            message_text=prepared_message.message_text,
            chat_id=chat_id,
            message_id=message_id,
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

        if not deps.should_gordo_respond(
            commands,
            command,
            sanitized_message_text,
            message,
            chat_config,
            reply_metadata,
        ):
            if prepared_message.message_text:
                formatted_message = deps.format_user_message(
                    message,
                    prepared_message.message_text,
                    reply_context_text,
                )
                deps.save_message_to_redis(
                    chat_id, message_id, formatted_message, redis_client
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

        response_msg, response_markup, response_uses_ai, response_command = _handle_known_command(
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
        )

        if response_msg == "ok" and response_command is None and response_markup is None:
            return "ok"

        if prepared_message.message_text:
            formatted_message = deps.format_user_message(
                message,
                prepared_message.message_text,
                reply_context_text,
            )
            deps.save_message_to_redis(chat_id, message_id, formatted_message, redis_client)

        if response_msg:
            deps.save_message_to_redis(
                chat_id,
                f"bot_{message_id}",
                response_msg,
                redis_client,
            )
            sent_message_id = deps.send_msg(
                chat_id,
                response_msg,
                message_id,
                reply_markup=response_markup,
            )
            if sent_message_id is not None:
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
