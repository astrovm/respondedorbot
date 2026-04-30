from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

from .admin_commands import (
    handle_admin_creditlog_command,
    handle_admin_printcredits_command,
)
from api.ai_billing import AIMessageBilling
from api.ai_pricing import estimate_transcribe_reserve_credits
from api.constants import BILLING_UNAVAILABLE_MESSAGE, PROMPT_NO_MARKDOWN
from api.ai_service import AIConversationRequest, AIService, SummaryCommandRequest
from api.chat_context import format_user_identity
from .billing_commands import (
    handle_balance_command,
    handle_transfer_command,
)
from api.message_state import save_user_chat_compacted_until, save_user_chat_summary
from .message_links import handle_link_replacement
from api.utils.text import sanitize_summary_text
from api.streaming import (
    consume_stream_to_telegram,
    extract_stream_metadata,
)

CommandTuple = Tuple[Callable[..., str], bool, bool]


def _reserve_media_credits(
    deps: Any,
    billing_helper: AIMessageBilling,
    scope: str,
    reserve_credits: int,
    *,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[AIMessageBilling], Optional[str]]:
    if (
        not deps.check_provider_available(scope=scope)
        and not deps.has_openrouter_fallback()
    ):
        return None, "rate_limited"
    media_charge_meta, media_charge_error = billing_helper.reserve_ai_credits(
        reason, reserve_credits, metadata=metadata
    )
    if media_charge_error:
        return None, media_charge_error
    return media_charge_meta, None


def _settle_media_result(
    billing_helper: AIMessageBilling,
    media_charge_meta: Optional[AIMessageBilling],
    billing_segments: List[Dict[str, Any]],
    success: bool,
    *,
    settle_reason: str,
    refund_reason: str,
) -> None:
    if media_charge_meta and not success:
        billing_helper.refund_reserved_ai_credits(
            media_charge_meta, reason=refund_reason
        )
    else:
        billing_helper.settle_reserved_ai_credits_batch(
            [media_charge_meta] if media_charge_meta else [],
            billing_segments,
            reason=settle_reason,
        )


@dataclass(frozen=True)
class MessageChatDeps:
    config_redis: Callable[[], Any]
    get_chat_config: Callable[[Any, str], Dict[str, Any]]
    extract_user_id: Callable[[Mapping[str, Any]], Optional[int]]
    extract_numeric_chat_id: Callable[[str], Optional[int]]


@dataclass(frozen=True)
class MessageRoutingDeps:
    initialize_commands: Callable[[], Dict[str, CommandTuple]]
    parse_command: Callable[[str, str], Tuple[str, str]]
    should_auto_process_media: Callable[
        [Mapping[str, CommandTuple], str, str, Mapping[str, Any]], bool
    ]
    replace_links: Callable[[str], Tuple[str, bool, List[str]]]
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
    is_group_chat_type: Callable[[Optional[str]], bool]


@dataclass(frozen=True)
class MessageIODeps:
    send_msg: Callable[..., Optional[int]]
    send_animation: Callable[..., Optional[int]]
    delete_msg: Callable[[str, str], None]
    edit_message: Callable[[str, str, str], None]
    admin_report: Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]


@dataclass(frozen=True)
class MessageStateDeps:
    get_bot_message_metadata: Callable[[Any, str, Any], Optional[Dict[str, Any]]]
    save_bot_message_metadata: Callable[[Any, str, Any, Mapping[str, Any]], None]
    build_reply_context_text: Callable[[Mapping[str, Any]], Optional[str]]
    build_message_links_context: Callable[[Mapping[str, Any]], str]
    format_user_message: Callable[[Dict[str, Any], str, Optional[str]], str]
    save_message_to_redis: Callable[..., None]


@dataclass(frozen=True)
class MessageAIDeps:
    ai_service: AIService
    balance_formatter: Any
    handle_ai_stream: Callable[..., str]
    gen_random: Callable[[str], str]
    build_insufficient_credits_message: Callable[..., str]
    build_topup_keyboard: Callable[[], Dict[str, Any]]
    credits_db_service: Any
    maybe_grant_onboarding_credits: Callable[
        [Any, Callable[..., None], Optional[int]], None
    ]
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
    load_persisted_reservation: Callable[[str], Optional[Mapping[str, Any]]] = (
        lambda _usage_tag: None
    )
    persist_reservation: Callable[[str, Mapping[str, Any]], None] = (
        lambda _usage_tag, _reservation: None
    )
    clear_persisted_reservation: Callable[[str], None] = lambda _usage_tag: None


@dataclass(frozen=True)
class MessageMediaDeps:
    extract_message_content: Callable[
        [Dict[str, Any]], Tuple[str, Optional[str], Optional[str]]
    ]
    _transcribe_audio_file: Callable[
        ..., Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]
    ]
    _transcription_error_message: Callable[..., Optional[str]]
    download_telegram_file: Callable[[str], Optional[bytes]]
    measure_audio_duration_seconds: Callable[[bytes], Optional[float]]
    resize_image_if_needed: Callable[[bytes], bytes]
    encode_image_to_base64: Callable[[bytes], str]

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
    edit_message: Callable[[str, str, str], None]
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
    save_message_to_redis: Callable[..., None]
    handle_ai_stream: Callable[..., str]
    gen_random: Callable[[str], str]
    build_insufficient_credits_message: Callable[..., str]
    build_topup_keyboard: Callable[[], Dict[str, Any]]
    credits_db_service: Any
    balance_formatter: Any
    is_group_chat_type: Callable[[Optional[str]], bool]
    extract_user_id: Callable[[Mapping[str, Any]], Optional[int]]
    extract_numeric_chat_id: Callable[[str], Optional[int]]
    maybe_grant_onboarding_credits: Callable[
        [Any, Callable[..., None], Optional[int]], None
    ]
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
    ai_service: AIService
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


def build_message_handler_deps(
    *,
    chat: MessageChatDeps,
    routing: MessageRoutingDeps,
    io: MessageIODeps,
    state: MessageStateDeps,
    ai: MessageAIDeps,
    media: MessageMediaDeps,
) -> MessageHandlerDeps:
    return MessageHandlerDeps(
        config_redis=chat.config_redis,
        get_chat_config=chat.get_chat_config,
        initialize_commands=routing.initialize_commands,
        parse_command=routing.parse_command,
        should_auto_process_media=routing.should_auto_process_media,
        extract_message_content=media.extract_message_content,
        replace_links=routing.replace_links,
        send_msg=io.send_msg,
        send_animation=io.send_animation,
        delete_msg=io.delete_msg,
        edit_message=io.edit_message,
        admin_report=io.admin_report,
        get_bot_message_metadata=state.get_bot_message_metadata,
        save_bot_message_metadata=state.save_bot_message_metadata,
        build_reply_context_text=state.build_reply_context_text,
        build_message_links_context=state.build_message_links_context,
        should_gordo_respond=routing.should_gordo_respond,
        format_user_message=state.format_user_message,
        save_message_to_redis=state.save_message_to_redis,
        handle_ai_stream=ai.handle_ai_stream,
        gen_random=ai.gen_random,
        build_insufficient_credits_message=ai.build_insufficient_credits_message,
        build_topup_keyboard=ai.build_topup_keyboard,
        credits_db_service=ai.credits_db_service,
        balance_formatter=ai.balance_formatter,
        is_group_chat_type=routing.is_group_chat_type,
        extract_user_id=chat.extract_user_id,
        extract_numeric_chat_id=chat.extract_numeric_chat_id,
        maybe_grant_onboarding_credits=ai.maybe_grant_onboarding_credits,
        handle_transcribe_with_message=ai.handle_transcribe_with_message,
        handle_transcribe_with_message_result=ai.handle_transcribe_with_message_result,
        check_provider_available=ai.check_provider_available,
        has_openrouter_fallback=ai.has_openrouter_fallback,
        handle_rate_limit=ai.handle_rate_limit,
        handle_successful_payment_message=ai.handle_successful_payment_message,
        handle_config_command=ai.handle_config_command,
        is_chat_admin=ai.is_chat_admin,
        report_unauthorized_config_attempt=ai.report_unauthorized_config_attempt,
        handle_transcribe=ai.handle_transcribe,
        estimate_ai_base_reserve_credits=ai.estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=ai.estimate_image_context_reserve_credits,
        _transcribe_audio_file=media._transcribe_audio_file,
        _transcription_error_message=media._transcription_error_message,
        download_telegram_file=media.download_telegram_file,
        measure_audio_duration_seconds=media.measure_audio_duration_seconds,
        resize_image_if_needed=media.resize_image_if_needed,
        encode_image_to_base64=media.encode_image_to_base64,
        load_persisted_reservation=ai.load_persisted_reservation,
        persist_reservation=ai.persist_reservation,
        clear_persisted_reservation=ai.clear_persisted_reservation,
        ai_service=ai.ai_service,
    )


@dataclass
class PreparedMessage:
    message_text: str
    photo_file_id: Optional[str]
    audio_file_id: Optional[str]
    resized_image_data: Optional[bytes] = None
    early_response: Optional[str] = None
    audio_duration_seconds: float = 0.0


@dataclass(frozen=True)
class MessageContext:
    message_id: str
    chat_id: str
    chat_type: str
    user_identity: str
    user_id: Optional[int]
    numeric_chat_id: Optional[int]


@dataclass(frozen=True)
class MessageRuntime:
    redis_client: Any
    chat_config: Mapping[str, Any]
    commands: Mapping[str, CommandTuple]
    bot_name: str
    billing_helper: AIMessageBilling
    prepared_message: PreparedMessage
    auto_process_media: bool = False


@dataclass(frozen=True)
class MessageIntent:
    command: str
    sanitized_message_text: str
    reply_context_text: Optional[str]
    should_respond: bool


def _billing_is_available(deps: MessageHandlerDeps) -> bool:
    return bool(deps.credits_db_service.is_configured())


def _build_message_context(
    message: Dict[str, Any], deps: MessageHandlerDeps
) -> Optional[MessageContext]:
    chat = cast(Dict[str, Any], message.get("chat", {}))
    sender = cast(Mapping[str, Any], message.get("from", {}))
    if not chat or chat.get("id") is None or not sender:
        return None
    chat_id = str(chat.get("id"))
    return MessageContext(
        message_id=str(message.get("message_id")),
        chat_id=chat_id,
        chat_type=str(chat.get("type", "")),
        user_identity=format_user_identity(sender),
        user_id=deps.extract_user_id(message),
        numeric_chat_id=deps.extract_numeric_chat_id(chat_id),
    )


def _handle_prepared_message_early_response(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    message_id: str,
    prepared_message: PreparedMessage,
) -> bool:
    if not prepared_message.early_response:
        return False
    if prepared_message.early_response != "ok":
        deps.send_msg(chat_id, prepared_message.early_response, message_id)
    return True


def _initialize_message_runtime(
    deps: MessageHandlerDeps,
    *,
    context: MessageContext,
    message: Dict[str, Any],
) -> MessageRuntime:
    redis_client = deps.config_redis()
    chat_config = deps.get_chat_config(redis_client, context.chat_id)
    commands = deps.initialize_commands()
    bot_name = f"@{environ.get('TELEGRAM_USERNAME')}"
    raw_message_text, photo_file_id, audio_file_id = _probe_message_content(
        message, deps=deps
    )
    command, _ = deps.parse_command(raw_message_text, bot_name)
    auto_process_media = deps.should_auto_process_media(
        commands,
        command,
        raw_message_text,
        message,
    )
    billing_helper = _build_billing_helper(
        deps,
        chat_id=context.chat_id,
        chat_type=context.chat_type,
        user_id=context.user_id,
        numeric_chat_id=context.numeric_chat_id,
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
    prepared_message = PreparedMessage(
        message_text=raw_message_text,
        photo_file_id=photo_file_id,
        audio_file_id=audio_file_id,
    )
    return MessageRuntime(
        redis_client=redis_client,
        chat_config=chat_config,
        commands=commands,
        bot_name=bot_name,
        billing_helper=billing_helper,
        prepared_message=prepared_message,
        auto_process_media=auto_process_media,
    )


def _finalize_message_response(
    deps: MessageHandlerDeps,
    *,
    context: MessageContext,
    message: Dict[str, Any],
    prepared_message: PreparedMessage,
    reply_context_text: Optional[str],
    redis_client: Any,
    response_msg: str,
    response_markup: Optional[Dict[str, Any]],
    response_uses_ai: bool,
    response_command: Optional[str],
) -> str:
    if response_msg == "ok" and response_command is None and response_markup is None:
        return "ok"

    _store_user_message_if_present(
        deps,
        chat_id=context.chat_id,
        message_id=context.message_id,
        message=message,
        message_text=prepared_message.message_text,
        reply_context_text=reply_context_text,
        redis_client=redis_client,
    )

    if response_uses_ai:
        actual_streamed_message_id, actual_streamed_response_text = (
            extract_stream_metadata()
        )
        if actual_streamed_message_id:
            deps.save_message_to_redis(
                context.chat_id,
                f"bot_{actual_streamed_message_id}",
                actual_streamed_response_text,
                redis_client,
                role="assistant",
            )
            metadata = (
                {
                    "type": "command",
                    "command": response_command,
                    "uses_ai": True,
                }
                if response_command
                else {"type": "ai"}
            )
            deps.save_bot_message_metadata(
                redis_client,
                context.chat_id,
                actual_streamed_message_id,
                metadata,
            )
        return "ok"

    if response_msg:
        _send_response_and_store_metadata(
            deps,
            chat_id=context.chat_id,
            message_id=context.message_id,
            response_msg=response_msg,
            response_markup=response_markup,
            response_command=response_command,
            response_uses_ai=response_uses_ai,
            redis_client=redis_client,
        )

    return "ok"


def _resolve_message_intent(
    deps: MessageHandlerDeps,
    *,
    context: MessageContext,
    runtime: MessageRuntime,
    message: Dict[str, Any],
) -> MessageIntent:
    command, sanitized_message_text = deps.parse_command(
        runtime.prepared_message.message_text,
        runtime.bot_name,
    )
    reply_metadata = _load_reply_metadata(
        deps,
        redis_client=runtime.redis_client,
        chat_id=context.chat_id,
        message=message,
    )
    reply_context_text = deps.build_reply_context_text(message)
    should_respond = deps.should_gordo_respond(
        runtime.commands,
        command,
        sanitized_message_text,
        message,
        runtime.chat_config,
        reply_metadata,
    )
    if (
        command in ["/comando", "/command"]
        and not sanitized_message_text
        and "reply_to_message" in message
    ):
        sanitized_message_text = deps.extract_message_content(
            cast(Dict[str, Any], message["reply_to_message"])
        )[0]

    return MessageIntent(
        command=command,
        sanitized_message_text=sanitized_message_text,
        reply_context_text=reply_context_text,
        should_respond=should_respond,
    )


def _billing_unavailable_command_response(
    command: str,
) -> Tuple[str, None, bool, str]:
    return BILLING_UNAVAILABLE_MESSAGE, None, False, command


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
    sender = cast(Mapping[str, Any], message.get("from", {}))
    reply_to_message = cast(Optional[Mapping[str, Any]], message.get("reply_to_message"))
    formatted_message = deps.format_user_message(
        message,
        message_text,
        reply_context_text,
    )
    deps.save_message_to_redis(
        chat_id,
        message_id,
        formatted_message,
        redis_client,
        role="user",
        user_id=str(sender.get("id") or ""),
        username=str(sender.get("username") or ""),
        reply_to_message_id=(
            str(reply_to_message.get("message_id"))
            if isinstance(reply_to_message, Mapping) and reply_to_message.get("message_id") is not None
            else None
        ),
        mentions_bot=("@" in message_text or message_text.startswith("/")),
    )


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
        role="assistant",
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


def _get_ai_service(deps: MessageHandlerDeps) -> AIService:
    return deps.ai_service


def _run_ai_flow(
    deps: MessageHandlerDeps,
    *,
    chat_id: str,
    message: Dict[str, Any],
    user_id: Optional[int],
    prepared_message: Any,
    billing_helper: Any,
    prompt_text: str,
    reply_context_text: Optional[str],
    user_identity: str,
    handler_func: Callable[..., str],
    redis_client: Any,
    timezone_offset: int = -3,
    is_spontaneous: bool = False,
    compaction_threshold: Optional[int] = None,
    compaction_keep: Optional[int] = None,
) -> Tuple[str, bool]:
    raw_message_id = message.get("message_id")
    reply_to_message_id = str(raw_message_id) if raw_message_id is not None else None
    return _get_ai_service(deps).run_conversation(
        AIConversationRequest(
            chat_id=chat_id,
            message=message,
            user_id=user_id,
            prepared_message=prepared_message,
            billing_helper=billing_helper,
            prompt_text=prompt_text,
            reply_context_text=reply_context_text,
            user_identity=user_identity,
            handler_func=handler_func,
            redis_client=redis_client,
            timezone_offset=timezone_offset,
            is_spontaneous=is_spontaneous,
            compaction_threshold=compaction_threshold,
            compaction_keep=compaction_keep,
            reply_to_message_id=reply_to_message_id,
        )
    )


def _probe_message_content(
    message: Dict[str, Any],
    *,
    deps: MessageHandlerDeps,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Extract raw text and media file_ids without any network calls."""
    return deps.extract_message_content(message)


def _process_message_media(
    prepared: PreparedMessage,
    *,
    message: Dict[str, Any],
    auto_process_media: bool,
    deps: MessageHandlerDeps,
    billing_helper: AIMessageBilling,
) -> PreparedMessage:
    """Transcribe audio and resize images. Expensive — only call after should_respond."""
    message_text = prepared.message_text
    photo_file_id = prepared.photo_file_id
    audio_file_id = prepared.audio_file_id
    audio_duration_seconds = prepared.audio_duration_seconds
    resized_image_data: Optional[bytes] = None

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
            if not message_text:
                return PreparedMessage(
                    message_text=message_text,
                    photo_file_id=photo_file_id,
                    audio_file_id=audio_file_id,
                    audio_duration_seconds=0.0,
                    early_response="ok",
                )
            return PreparedMessage(
                message_text=message_text,
                photo_file_id=photo_file_id,
                audio_file_id=audio_file_id,
                audio_duration_seconds=0.0,
            )
        audio_duration_seconds = resolved_audio_duration
        reserved_credits = estimate_transcribe_reserve_credits(audio_duration_seconds)
        media_charge_meta, reserve_error = _reserve_media_credits(
            deps, billing_helper, "transcribe", reserved_credits,
            reason="auto_audio_media",
            metadata={"audio_seconds": audio_duration_seconds},
        )
        if reserve_error == "rate_limited":
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
        if reserve_error:
            return PreparedMessage(
                message_text=message_text,
                photo_file_id=photo_file_id,
                audio_file_id=audio_file_id,
                audio_duration_seconds=audio_duration_seconds,
                early_response=reserve_error,
            )

        transcription, err, billing_segment = deps._transcribe_audio_file(
            audio_file_id, use_cache=False
        )
        _settle_media_result(
            billing_helper, media_charge_meta,
            [billing_segment] if billing_segment else [],
            bool(transcription),
            settle_reason="auto_audio_media_success",
            refund_reason="auto_audio_transcribe_failed",
        )
        if transcription:
            message_text = transcription
        else:
            message_text = (
                deps._transcription_error_message(
                    err,
                    download_message="no pude bajar tu audio, mandalo de vuelta",
                    transcribe_message="mandame texto que no soy alexa, boludo",
                )
                or "mandame texto que no soy alexa, boludo"
            )

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
    return handle_link_replacement(
        deps,
        chat_config=chat_config,
        message=message,
        message_text=message_text,
        chat_id=chat_id,
        message_id=message_id,
        redis_client=redis_client,
    )


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
        cast(Dict[str, Any], reply_msg), reply_text, None
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
        from api.constants import ADMIN_CONFIG_DENIAL_MESSAGE
        deps.send_msg(chat_id, ADMIN_CONFIG_DENIAL_MESSAGE, str(message.get("message_id")))
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
    return handle_balance_command(
        deps,
        command=command,
        chat_type=chat_type,
        chat_id=chat_id,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
    )


def _handle_admin_printcredits_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    return handle_admin_printcredits_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        user_id=user_id,
    )



def _handle_admin_creditlog_command(
    deps: MessageHandlerDeps,
    *,
    command: str,
    sanitized_message_text: str,
    chat_id: str,
    user_id: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    return handle_admin_creditlog_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        user_id=user_id,
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
    return handle_transfer_command(
        deps,
        command=command,
        sanitized_message_text=sanitized_message_text,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        numeric_chat_id=numeric_chat_id,
    )


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
    user_id: Optional[int] = None,
    user_identity: str = "",
    timezone_offset: int = -3,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, Optional[str]]:
    if command not in commands:
        return None, None, False, None

    handler_func, uses_ai, takes_params = commands[command]
    if uses_ai:
        return None, None, False, None

    if command in ("/resumen", "/summary", "/tldr"):
        custom_instruction = None
        parts = sanitized_message_text.strip().split(None, 1)
        if parts and parts[0].isdigit() and len(parts) > 1:
            custom_instruction = parts[1]
        elif parts and not parts[0].isdigit():
            custom_instruction = sanitized_message_text.strip()

        base_prompt = (
            "actualizá el resumen anterior con los mensajes nuevos. "
            "máximo 10 items cortos y concretos, uno por línea. "
            "incluí solo hechos relevantes: tema, decisiones, pendientes y datos clave. "
            "evitá relleno, repetición, contexto innecesario y frases largas. "
            f"{PROMPT_NO_MARKDOWN} "
            "usá solo guiones (-) al inicio de cada item."
        )
        if custom_instruction:
            prompt_text = f"{custom_instruction}. {base_prompt}"
        else:
            prompt_text = base_prompt

        raw_message_id = message.get("message_id")
        reply_to_message_id = str(raw_message_id) if raw_message_id is not None else None

        def _consume_summary_stream(iterator):
            final_text, _message_id = consume_stream_to_telegram(
                chat_id,
                iterator,
                deps.send_msg,
                deps.edit_message,
                reply_to_message_id=reply_to_message_id,
            )
            return final_text

        final_text, pending_marker, is_fallback = deps.ai_service.run_summary_command_stream(
            SummaryCommandRequest(
                chat_id=chat_id,
                message=message,
                billing_helper=billing_helper,
                prompt_text=prompt_text,
                redis_client=redis_client,
            ),
            stream_consumer=_consume_summary_stream,
        )

        final_text = sanitize_summary_text(final_text)

        if not is_fallback and pending_marker is not None:
            save_user_chat_summary(redis_client, chat_id, final_text)
            save_user_chat_compacted_until(redis_client, chat_id, pending_marker)

        if is_fallback:
            return final_text, None, False, command
        return None, None, True, command

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
        _settle_media_result(
            billing_helper, media_charge_meta,
            cast(List[Mapping[str, Any]], billing_segments),
            transcribe_succeeded,
            settle_reason="transcribe_command_success",
            refund_reason="transcribe_command_unsuccessful",
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

    def _dispatch(cmd_handlers):
        for handler in cmd_handlers:
            resp = handler()
            if resp[0] is not None or resp[3] is not None:
                return resp
        return None

    dispatch_result = _dispatch([
        lambda: _handle_config_command(
            deps, command=command, chat_id=chat_id,
            chat_type=chat_type, message=message,
            redis_client=redis_client,
        ),
        lambda: _handle_topup_command(
            deps, command=command, chat_type=chat_type,
        ),
        lambda: _handle_balance_command(
            deps, command=command, chat_type=chat_type,
            chat_id=chat_id, user_id=user_id,
            numeric_chat_id=numeric_chat_id,
        ),
        lambda: _handle_transfer_command(
            deps, command=command,
            sanitized_message_text=sanitized_message_text,
            chat_id=chat_id, chat_type=chat_type,
            user_id=user_id, numeric_chat_id=numeric_chat_id,
        ),
        lambda: _handle_admin_printcredits_command(
            deps, command=command,
            sanitized_message_text=sanitized_message_text,
            chat_id=chat_id, user_id=user_id,
        ),
        lambda: _handle_admin_creditlog_command(
            deps, command=command,
            sanitized_message_text=sanitized_message_text,
            chat_id=chat_id, user_id=user_id,
        ),
    ])
    if dispatch_result is not None:
        return dispatch_result

    if command in commands:
        _handler_func, uses_ai, _ = commands[command]
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
                handler_func=deps.handle_ai_stream,
                redis_client=redis_client,
                timezone_offset=timezone_offset,
            )
            return response_msg, response_markup, response_uses_ai, response_command

        response_msg, response_markup, response_uses_ai, response_command = _handle_non_ai_command(
            deps,
            command=command,
            commands=commands,
            sanitized_message_text=sanitized_message_text,
            message=message,
            chat_id=chat_id,
            redis_client=redis_client,
            billing_helper=billing_helper,
            user_id=user_id,
            user_identity=user_identity,
            timezone_offset=timezone_offset,
        )
        return response_msg, response_markup, response_uses_ai, response_command

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
        handler_func=deps.handle_ai_stream,
        redis_client=redis_client,
        timezone_offset=timezone_offset,
        is_spontaneous=True,
    )
    return response_msg, response_markup, response_uses_ai, response_command


def handle_msg(message: Dict[str, Any], deps: MessageHandlerDeps) -> str:
    """Handle an incoming Telegram message."""

    try:
        context = _build_message_context(message, deps)
        if context is None:
            return "ok"

        if isinstance(message.get("successful_payment"), Mapping):
            return deps.handle_successful_payment_message(message)

        runtime = _initialize_message_runtime(
            deps,
            context=context,
            message=message,
        )
        if _handle_prepared_message_early_response(
            deps,
            chat_id=context.chat_id,
            message_id=context.message_id,
            prepared_message=runtime.prepared_message,
        ):
            return "ok"

        if _handle_link_replacement(
            deps,
            chat_config=runtime.chat_config,
            message=message,
            message_text=runtime.prepared_message.message_text,
            chat_id=context.chat_id,
            message_id=context.message_id,
            redis_client=runtime.redis_client,
        ):
            return "ok"

        intent = _resolve_message_intent(
            deps,
            context=context,
            runtime=runtime,
            message=message,
        )
        if not intent.should_respond:
            _store_user_message_if_present(
                deps,
                chat_id=context.chat_id,
                message_id=context.message_id,
                message=message,
                message_text=runtime.prepared_message.message_text,
                reply_context_text=intent.reply_context_text,
                redis_client=runtime.redis_client,
            )
            return "ok"

        prepared_message = _process_message_media(
            runtime.prepared_message,
            message=message,
            auto_process_media=runtime.auto_process_media,
            deps=deps,
            billing_helper=runtime.billing_helper,
        )
        if prepared_message.early_response:
            deps.send_msg(
                context.chat_id,
                prepared_message.early_response,
                context.message_id,
            )
            return "ok"

        _save_replied_message_context(
            deps,
            message=message,
            chat_id=context.chat_id,
            redis_client=runtime.redis_client,
        )

        response_msg, response_markup, response_uses_ai, response_command = (
            _handle_known_command(
                deps,
                commands=runtime.commands,
                command=intent.command,
                sanitized_message_text=intent.sanitized_message_text,
                message=message,
                chat_id=context.chat_id,
                chat_type=context.chat_type,
                user_id=context.user_id,
                numeric_chat_id=context.numeric_chat_id,
                prepared_message=prepared_message,
                billing_helper=runtime.billing_helper,
                reply_context_text=intent.reply_context_text,
                user_identity=context.user_identity,
                redis_client=runtime.redis_client,
                timezone_offset=int(runtime.chat_config.get("timezone_offset", -3)),
            )
        )

        return _finalize_message_response(
            deps,
            context=context,
            message=message,
            prepared_message=prepared_message,
            reply_context_text=intent.reply_context_text,
            redis_client=runtime.redis_client,
            response_msg=response_msg or "",
            response_markup=response_markup,
            response_uses_ai=response_uses_ai,
            response_command=response_command,
        )

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


__all__ = [
    "MessageAIDeps",
    "MessageChatDeps",
    "MessageHandlerDeps",
    "MessageIODeps",
    "MessageMediaDeps",
    "MessageRoutingDeps",
    "MessageStateDeps",
    "build_message_handler_deps",
    "handle_msg",
]
