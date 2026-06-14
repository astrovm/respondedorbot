from tests.support import *


def _telegram_message(
    *,
    message_id=1,
    chat_id=123,
    chat_type="private",
    user_id=88,
    first_name="John",
    username="john",
    text=None,
    voice=None,
    photo=None,
    reply_to_message=None,
):
    message = {
        "message_id": message_id,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"first_name": first_name, "username": username},
    }
    if user_id is not None:
        message["from"]["id"] = user_id
    if text is not None:
        message["text"] = text
    if voice is not None:
        message["voice"] = voice
    if photo is not None:
        message["photo"] = photo
    if reply_to_message is not None:
        message["reply_to_message"] = reply_to_message
    return message


def _private_voice_message(
    *, message_id=1, user_id=88, duration=None, file_id="voice_123"
):
    voice = {"file_id": file_id}
    if duration is not None:
        voice["duration"] = duration
    return _telegram_message(message_id=message_id, user_id=user_id, voice=voice)


def _private_photo_message(*, message_id=1, chat_id=123, user_id=88, file_id="img1"):
    return _telegram_message(
        message_id=message_id,
        chat_id=chat_id,
        user_id=user_id,
        first_name="Ana",
        username="ana",
        photo=[{"file_id": file_id}],
    )


def _build_message_handler_flat_defaults(redis_client, mock_credits):
    from api.bot.command_registry import (
        parse_command as _parse_command,
        should_auto_process_media as _should_auto_process_media,
    )
    from api import index as _api_index
    from api.bot.streaming import set_streamed_response_metadata

    link_service = MagicMock()
    link_service.replace.side_effect = lambda text: (text, False, [])
    link_service.download_oversized_instagram_video.return_value = None
    link_service.build_context.return_value = ""

    defaults = {
        "config_redis": lambda: redis_client,
        "get_chat_config": lambda _rc, _cid: dict(CHAT_CONFIG_DEFAULTS),
        "initialize_commands": _api_index.initialize_commands,
        "parse_command": _parse_command,
        "should_auto_process_media": _should_auto_process_media,
        "extract_message_content": _api_index.extract_message_content,
        "link_service": link_service,
        "send_msg": MagicMock(return_value=999),
        "send_animation": MagicMock(return_value=999),
        "send_photo": MagicMock(return_value=999),
        "send_video": MagicMock(return_value=999),
        "delete_msg": MagicMock(),
        "edit_message": MagicMock(),
        "admin_report": MagicMock(),
        "get_bot_message_metadata": MagicMock(return_value=None),
        "save_bot_message_metadata": MagicMock(),
        "build_reply_context_text": MagicMock(return_value=None),
        "should_gordo_respond": MagicMock(),
        "format_user_message": lambda message, text, _reply_context: (
            f"{message['from']['first_name']}: {text}"
        ),
        "save_message_to_redis": MagicMock(),
        "get_chat_history": MagicMock(return_value=[]),
        "prepare_chat_memory": MagicMock(return_value=([], None, [], 0)),
        "build_ai_messages": MagicMock(
            return_value=[{"role": "user", "content": "hola"}]
        ),
        "handle_ai_response": _api_index.app_runtime.responses.handle,
        "gen_random": MagicMock(return_value="random"),
        "build_insufficient_credits_message": MagicMock(
            return_value="insufficient credits"
        ),
        "build_topup_keyboard": MagicMock(return_value={}),
        "credits_db_service": mock_credits,
        "balance_formatter": MagicMock(format=MagicMock(return_value="balance info")),
        "is_group_chat_type": lambda chat_type: chat_type in {"group", "supergroup"},
        "extract_user_id": lambda message: message.get("from", {}).get("id"),
        "extract_numeric_chat_id": lambda chat_id: (
            int(chat_id) if chat_id.isdigit() else None
        ),
        "maybe_grant_onboarding_credits": MagicMock(),
        "handle_transcribe_with_message": MagicMock(return_value="transcribed"),
        "handle_transcribe_with_message_result": MagicMock(
            return_value=("transcribed", [])
        ),
        "check_provider_available": MagicMock(return_value=True),
        "has_openrouter_fallback": MagicMock(return_value=False),
        "handle_rate_limit": MagicMock(return_value="no boludo"),
        "handle_successful_payment_message": MagicMock(return_value="payment ok"),
        "handle_config_command": MagicMock(return_value=("config ok", {})),
        "is_chat_admin": MagicMock(return_value=False),
        "report_unauthorized_config_attempt": MagicMock(),
        "handle_transcribe": MagicMock(return_value="transcribed"),
        "estimate_ai_base_reserve_credits": MagicMock(return_value=(1, {})),
        "estimate_image_context_reserve_credits": MagicMock(return_value=1),
        "_transcribe_audio_file": MagicMock(return_value=("transcribed", None, None)),
        "_transcription_error_message": MagicMock(return_value=None),
        "download_telegram_file": MagicMock(return_value=None),
        "measure_audio_duration_seconds": MagicMock(return_value=None),
        "resize_image_if_needed": MagicMock(return_value=b""),
        "encode_image_to_base64": MagicMock(return_value="base64"),
        "load_persisted_reservation": MagicMock(return_value=None),
        "persist_reservation": MagicMock(),
        "clear_persisted_reservation": MagicMock(),
    }

    def _default_handle_ai_stream(messages, *, chat_id=None, **_kwargs):
        response_text = str(messages[0].get("content") or "respuesta ok")
        sent_message_id = defaults["send_msg"](str(chat_id or ""), response_text)
        set_streamed_response_metadata(
            str(sent_message_id) if sent_message_id is not None else None,
            response_text,
        )
        return response_text

    defaults["handle_ai_stream"] = MagicMock(side_effect=_default_handle_ai_stream)
    return defaults


def _build_test_ai_service(flat_defaults):
    from api.ai.service import build_ai_service

    return build_ai_service(
        credits_db_service=flat_defaults["credits_db_service"],
        get_chat_history=flat_defaults["get_chat_history"],
        prepare_chat_memory=flat_defaults["prepare_chat_memory"],
        build_ai_messages=flat_defaults["build_ai_messages"],
        check_provider_available=flat_defaults["check_provider_available"],
        has_openrouter_fallback=flat_defaults["has_openrouter_fallback"],
        handle_rate_limit=flat_defaults["handle_rate_limit"],
        handle_ai_response=flat_defaults["handle_ai_response"],
        estimate_ai_base_reserve_credits=flat_defaults[
            "estimate_ai_base_reserve_credits"
        ],
        estimate_image_context_reserve_credits=flat_defaults[
            "estimate_image_context_reserve_credits"
        ],
    )


def _build_grouped_message_handler_deps(flat_defaults):
    from api.bot.message_handler import (
        MessageAIDeps,
        MessageChatDeps,
        MessageHandlerDeps,
        MessageIODeps,
        MessageMediaDeps,
        MessageRoutingDeps,
        MessageStateDeps,
        build_message_handler_deps,
    )

    ai_service = flat_defaults.get("ai_service") or _build_test_ai_service(
        flat_defaults
    )
    deps = build_message_handler_deps(
        chat=MessageChatDeps(
            config_redis=flat_defaults["config_redis"],
            get_chat_config=flat_defaults["get_chat_config"],
            extract_user_id=flat_defaults["extract_user_id"],
            extract_numeric_chat_id=flat_defaults["extract_numeric_chat_id"],
        ),
        routing=MessageRoutingDeps(
            initialize_commands=flat_defaults["initialize_commands"],
            parse_command=flat_defaults["parse_command"],
            should_auto_process_media=flat_defaults["should_auto_process_media"],
            link_service=flat_defaults["link_service"],
            should_gordo_respond=flat_defaults["should_gordo_respond"],
            is_group_chat_type=flat_defaults["is_group_chat_type"],
        ),
        io=MessageIODeps(
            send_msg=flat_defaults["send_msg"],
            send_animation=flat_defaults["send_animation"],
            send_photo=flat_defaults["send_photo"],
            send_video=flat_defaults["send_video"],
            delete_msg=flat_defaults["delete_msg"],
            edit_message=flat_defaults["edit_message"],
            admin_report=flat_defaults["admin_report"],
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=flat_defaults["get_bot_message_metadata"],
            save_bot_message_metadata=flat_defaults["save_bot_message_metadata"],
            build_reply_context_text=flat_defaults["build_reply_context_text"],
            format_user_message=flat_defaults["format_user_message"],
            save_message_to_redis=flat_defaults["save_message_to_redis"],
            save_chat_member=flat_defaults.get("save_chat_member", MagicMock()),
        ),
        ai=MessageAIDeps(
            ai_service=ai_service,
            balance_formatter=flat_defaults["balance_formatter"],
            handle_ai_stream=flat_defaults["handle_ai_stream"],
            gen_random=flat_defaults["gen_random"],
            build_insufficient_credits_message=flat_defaults[
                "build_insufficient_credits_message"
            ],
            build_topup_keyboard=flat_defaults["build_topup_keyboard"],
            credits_db_service=flat_defaults["credits_db_service"],
            maybe_grant_onboarding_credits=flat_defaults[
                "maybe_grant_onboarding_credits"
            ],
            handle_transcribe_with_message=flat_defaults[
                "handle_transcribe_with_message"
            ],
            handle_transcribe_with_message_result=flat_defaults[
                "handle_transcribe_with_message_result"
            ],
            check_provider_available=flat_defaults["check_provider_available"],
            has_openrouter_fallback=flat_defaults["has_openrouter_fallback"],
            handle_rate_limit=flat_defaults["handle_rate_limit"],
            handle_successful_payment_message=flat_defaults[
                "handle_successful_payment_message"
            ],
            handle_config_command=flat_defaults["handle_config_command"],
            is_chat_admin=flat_defaults["is_chat_admin"],
            report_unauthorized_config_attempt=flat_defaults[
                "report_unauthorized_config_attempt"
            ],
            handle_transcribe=flat_defaults["handle_transcribe"],
            estimate_ai_base_reserve_credits=flat_defaults[
                "estimate_ai_base_reserve_credits"
            ],
            estimate_image_context_reserve_credits=flat_defaults[
                "estimate_image_context_reserve_credits"
            ],
            load_persisted_reservation=flat_defaults["load_persisted_reservation"],
            persist_reservation=flat_defaults["persist_reservation"],
            clear_persisted_reservation=flat_defaults["clear_persisted_reservation"],
        ),
        media=MessageMediaDeps(
            extract_message_content=flat_defaults["extract_message_content"],
            _transcribe_audio_file=flat_defaults["_transcribe_audio_file"],
            _transcription_error_message=flat_defaults["_transcription_error_message"],
            download_telegram_file=flat_defaults["download_telegram_file"],
            measure_audio_duration_seconds=flat_defaults[
                "measure_audio_duration_seconds"
            ],
            resize_image_if_needed=flat_defaults["resize_image_if_needed"],
            encode_image_to_base64=flat_defaults["encode_image_to_base64"],
        ),
    )
    assert isinstance(deps, MessageHandlerDeps)
    return deps


def _build_message_handler_deps():
    redis_client = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits.return_value = {"ok": True, "source": "user"}

    def make_deps(**overrides):
        flat_defaults = _build_message_handler_flat_defaults(redis_client, mock_credits)
        flat_defaults.update(overrides)
        return _build_grouped_message_handler_deps(flat_defaults)

    return make_deps, redis_client


def _simulate_streamed_ai_response(mock_send_msg, response_text, billing_segments=None):
    from api.bot.streaming import set_streamed_response_metadata

    def _fake_handle_ai_response(*args, **kwargs):
        if args and isinstance(args[0], list):
            chat_id = str(kwargs.get("chat_id") or "")
        else:
            chat_id = str(args[0])
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict) and billing_segments is not None:
            response_meta["billing_segments"] = billing_segments
        sent_message_id = mock_send_msg(chat_id, response_text)
        set_streamed_response_metadata(
            str(sent_message_id) if sent_message_id is not None else None,
            response_text,
        )
        return response_text

    return _fake_handle_ai_response
