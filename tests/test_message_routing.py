from tests.support import *
from tests.message_handler_support import (
    _build_message_handler_deps,
    _simulate_streamed_ai_response,
)


def test_build_message_handler_deps_from_groups_exposes_flat_runtime_contract():
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

    ai_service = MagicMock()
    credits = MagicMock()

    deps = build_message_handler_deps(
        chat=MessageChatDeps(
            config_redis=MagicMock(),
            get_chat_config=MagicMock(),
            extract_user_id=MagicMock(),
            extract_numeric_chat_id=MagicMock(),
        ),
        routing=MessageRoutingDeps(
            initialize_commands=MagicMock(),
            parse_command=MagicMock(),
            should_auto_process_media=MagicMock(),
            link_service=MagicMock(),
            should_gordo_respond=MagicMock(),
            is_group_chat_type=MagicMock(),
        ),
        io=MessageIODeps(
            send_msg=MagicMock(),
            send_animation=MagicMock(),
            send_photo=MagicMock(),
            send_video=MagicMock(),
            delete_msg=MagicMock(),
            edit_message=MagicMock(),
            admin_report=MagicMock(),
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=MagicMock(),
            save_bot_message_metadata=MagicMock(),
            build_reply_context_text=MagicMock(),
            format_user_message=MagicMock(),
            save_message_to_redis=MagicMock(),
            save_chat_member=MagicMock(),
        ),
        ai=MessageAIDeps(
            ai_service=ai_service,
            handle_ai_stream=MagicMock(return_value="streamed response"),
            gen_random=MagicMock(),
            build_insufficient_credits_message=MagicMock(),
            build_topup_keyboard=MagicMock(),
            credits_db_service=credits,
            balance_formatter=MagicMock(),
            maybe_grant_onboarding_credits=MagicMock(),
            handle_transcribe_with_message=MagicMock(),
            handle_transcribe_with_message_result=MagicMock(),
            check_provider_available=MagicMock(),
            has_openrouter_fallback=MagicMock(),
            handle_rate_limit=MagicMock(),
            handle_successful_payment_message=MagicMock(),
            handle_config_command=MagicMock(),
            is_chat_admin=MagicMock(),
            report_unauthorized_config_attempt=MagicMock(),
            handle_transcribe=MagicMock(),
            estimate_ai_base_reserve_credits=MagicMock(),
            estimate_image_context_reserve_credits=MagicMock(),
        ),
        media=MessageMediaDeps(
            extract_message_content=MagicMock(),
            _transcribe_audio_file=MagicMock(),
            _transcription_error_message=MagicMock(),
            download_telegram_file=MagicMock(),
            measure_audio_duration_seconds=MagicMock(),
            resize_image_if_needed=MagicMock(),
            encode_image_to_base64=MagicMock(),
        ),
    )

    assert isinstance(deps, MessageHandlerDeps)
    assert deps.ai_service is ai_service
    assert deps.credits_db_service is credits
    assert deps.config_redis is not None
    assert deps.parse_command is not None
    assert deps.send_msg is not None
    assert deps.save_message_to_redis is not None
    assert deps.extract_message_content is not None


def test_build_message_context_extracts_chat_sender_and_ids():
    from api.bot.message_handler import _build_message_context

    deps = MagicMock()
    deps.extract_user_id.return_value = 77
    deps.extract_numeric_chat_id.return_value = 555

    message = {
        "message_id": 42,
        "chat": {"id": 555, "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
    }

    context = _build_message_context(message, deps)

    assert context is not None
    assert context.message_id == "42"
    assert context.chat_id == "555"
    assert context.chat_type == "private"
    assert context.user_id == 77
    assert context.numeric_chat_id == 555
    assert context.user_identity == "Ana (ana)"


def test_handle_prepared_message_early_response_sends_and_stops():
    from api.bot.message_handler import (
        PreparedMessage,
        _handle_prepared_message_early_response,
    )

    deps = MagicMock()

    handled = _handle_prepared_message_early_response(
        deps,
        chat_id="555",
        message_id="42",
        prepared_message=PreparedMessage(
            message_text="hola",
            photo_file_id=None,
            audio_file_id=None,
            early_response="no boludo",
        ),
    )

    assert handled is True
    deps.send_msg.assert_called_once_with("555", "no boludo", "42")


def test_resolve_message_intent_uses_reply_text_for_command_without_params():
    from api.bot.message_handler import MessageContext, MessageRuntime, PreparedMessage
    from api.bot.message_handler import _resolve_message_intent

    deps = MagicMock()
    deps.parse_command.return_value = ("/command", "")
    deps.build_reply_context_text.return_value = None
    deps.should_gordo_respond.return_value = True
    deps.extract_message_content.return_value = ("texto citado", None, None)

    intent = _resolve_message_intent(
        deps,
        context=MessageContext(
            message_id="42",
            chat_id="555",
            chat_type="private",
            user_identity="Ana (ana)",
            user_id=77,
            numeric_chat_id=555,
        ),
        runtime=MessageRuntime(
            redis_client=MagicMock(),
            chat_config={},
            commands={},
            bot_name="@testbot",
            billing_helper=MagicMock(),
            prepared_message=PreparedMessage(
                message_text="/command",
                photo_file_id=None,
                audio_file_id=None,
            ),
        ),
        message={
            "reply_to_message": {"text": "quoted text"},
        },
    )

    assert intent.command == "/command"
    assert intent.sanitized_message_text == "texto citado"
    assert intent.should_respond is True


def test_handle_non_ai_command_summary_uses_streaming():
    from api.bot.message_handler import (
        CommandDispatchContext,
        PreparedMessage,
        _handle_non_ai_command,
    )

    deps = MagicMock()
    deps.ai_service.run_summary_command_stream.return_value = (
        "resumen listo",
        "m10",
        False,
    )
    commands = {"/resumen": (MagicMock(), False, True)}

    response = _handle_non_ai_command(
        deps,
        CommandDispatchContext(
            commands=commands,
            command="/resumen",
            sanitized_message_text="focus en crypto",
            message={"message_id": "10"},
            chat_id="123",
            chat_type="private",
            user_id=7,
            numeric_chat_id=123,
            prepared_message=PreparedMessage("/resumen", None, None),
            billing_helper=MagicMock(),
            reply_context_text=None,
            user_identity="Ana (ana)",
            redis_client=MagicMock(),
        ),
    )

    assert response == (None, None, True, "/resumen")
    deps.ai_service.run_summary_command_stream.assert_called_once()
    call_args = deps.ai_service.run_summary_command_stream.call_args
    assert call_args[0][0].chat_id == "123"
    assert "focus en crypto" in call_args[0][0].prompt_text


def test_handle_non_ai_command_summary_fallback_returns_text():
    from api.bot.message_handler import (
        CommandDispatchContext,
        PreparedMessage,
        _handle_non_ai_command,
    )

    deps = MagicMock()
    deps.ai_service.run_summary_command_stream.return_value = (
        "no pude generar el resumen",
        None,
        True,
    )
    commands = {"/resumen": (MagicMock(), False, True)}

    response = _handle_non_ai_command(
        deps,
        CommandDispatchContext(
            commands=commands,
            command="/resumen",
            sanitized_message_text="focus en crypto",
            message={"message_id": "10"},
            chat_id="123",
            chat_type="private",
            user_id=7,
            numeric_chat_id=123,
            prepared_message=PreparedMessage("/resumen", None, None),
            billing_helper=MagicMock(),
            reply_context_text=None,
            user_identity="Ana (ana)",
            redis_client=MagicMock(),
        ),
    )

    assert response == ("no pude generar el resumen", None, False, "/resumen")


def test_handle_known_command_preserves_ai_flag_from_summary_non_ai_branch():
    from api.bot.message_handler import (
        CommandDispatchContext,
        PreparedMessage,
        _handle_known_command,
    )

    deps = MagicMock()
    commands = {"/resumen": (MagicMock(), False, True)}

    with patch(
        "api.bot.message_handler._handle_non_ai_command",
        return_value=("resumen listo", None, True, "/resumen"),
    ):
        response = _handle_known_command(
            deps,
            CommandDispatchContext(
                commands=commands,
                command="/resumen",
                sanitized_message_text="focus en crypto",
                message={"message_id": "10"},
                chat_id="123",
                chat_type="private",
                user_id=7,
                numeric_chat_id=123,
                prepared_message=PreparedMessage(
                    message_text="/resumen focus en crypto",
                    photo_file_id=None,
                    audio_file_id=None,
                ),
                billing_helper=MagicMock(),
                reply_context_text=None,
                user_identity="Ana (ana)",
                redis_client=MagicMock(),
            ),
        )

    assert response == ("resumen listo", None, True, "/resumen")


def test_handle_non_ai_command_passes_world_cup_country_query():
    from api.bot.message_handler import (
        CommandDispatchContext,
        PreparedMessage,
        _handle_non_ai_command,
    )

    handler = MagicMock(return_value="fixture")
    commands = {"/mundial": (handler, False, True)}

    response = _handle_non_ai_command(
        MagicMock(),
        CommandDispatchContext(
            commands=commands,
            command="/mundial",
            sanitized_message_text="argentina",
            message={"message_id": "10"},
            chat_id="123",
            chat_type="private",
            user_id=7,
            numeric_chat_id=123,
            prepared_message=PreparedMessage(
                message_text="/mundial argentina",
                photo_file_id=None,
                audio_file_id=None,
            ),
            billing_helper=MagicMock(),
            reply_context_text=None,
            user_identity="Ana (ana)",
            redis_client=MagicMock(),
            timezone_offset=-3,
        ),
    )

    assert response == ("fixture", None, False, "/mundial")
    handler.assert_called_once_with(timezone_offset=-3, team_query="argentina")


def test_message_handler_routes_ai_command_through_known_command_path(monkeypatch):
    from api.bot.message_handler import handle_msg

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    make_deps, redis_client = _build_message_handler_deps()
    mock_send_msg = MagicMock(return_value=999)
    mock_handle_ai_stream = MagicMock(
        side_effect=_simulate_streamed_ai_response(mock_send_msg, "respuesta ok")
    )
    mock_save_message = MagicMock()
    mock_save_metadata = MagicMock()
    deps = make_deps(
        send_msg=mock_send_msg,
        handle_ai_stream=mock_handle_ai_stream,
        save_message_to_redis=mock_save_message,
        save_bot_message_metadata=mock_save_metadata,
    )

    message = {
        "message_id": 401,
        "chat": {"id": 555, "type": "private"},
        "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_handle_ai_stream.called
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args.args[1] == "respuesta ok"
    mock_save_message.assert_any_call(
        "555", "bot_999", "respuesta ok", redis_client, role="assistant"
    )
    mock_save_metadata.assert_called_once_with(
        redis_client,
        "555",
        "999",
        {"type": "command", "command": "/ask", "uses_ai": True},
    )


def test_message_handler_ai_command_passes_single_request_object(monkeypatch):
    from api.ai.service import AIConversationRequest
    from api.bot.message_handler import handle_msg
    from api.bot.streaming import set_streamed_response_metadata

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    ai_service = MagicMock()
    def run_conversation(_request):
        set_streamed_response_metadata("999", "respuesta ai")
        return ("respuesta ai", True)

    ai_service.run_conversation.side_effect = run_conversation

    make_deps, _ = _build_message_handler_deps()
    mock_send_msg = MagicMock(return_value=999)
    mock_save_message = MagicMock()
    deps = make_deps(
        ai_service=ai_service,
        send_msg=mock_send_msg,
        save_message_to_redis=mock_save_message,
    )

    message = {
        "message_id": 501,
        "chat": {"id": 557, "type": "private"},
        "from": {"id": 1003, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    ai_service.run_conversation.assert_called_once()
    request = ai_service.run_conversation.call_args.args[0]
    assert isinstance(request, AIConversationRequest)
    assert request.chat_id == "557"
    assert request.prompt_text == "hola"
    assert request.is_spontaneous is False
    assert request.handler_func is deps.handle_ai_stream
    mock_send_msg.assert_not_called()
    mock_save_message.assert_any_call(
        "557", "bot_999", "respuesta ai", ANY, role="assistant"
    )


def test_message_handler_spontaneous_reply_passes_single_request_object(monkeypatch):
    from api.ai.service import AIConversationRequest
    from api.bot.message_handler import handle_msg
    from api.bot.streaming import set_streamed_response_metadata

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    ai_service = MagicMock()
    def run_conversation(_request):
        set_streamed_response_metadata("999", "respuesta espontanea")
        return ("respuesta espontanea", True)

    ai_service.run_conversation.side_effect = run_conversation

    make_deps, _ = _build_message_handler_deps()
    mock_send_msg = MagicMock(return_value=999)
    mock_save_message = MagicMock()
    deps = make_deps(
        ai_service=ai_service,
        send_msg=mock_send_msg,
        save_message_to_redis=mock_save_message,
        should_gordo_respond=MagicMock(return_value=True),
    )

    message = {
        "message_id": 502,
        "chat": {"id": 558, "type": "private"},
        "from": {"id": 1004, "first_name": "Ana", "username": "ana"},
        "text": "hola gordo",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    ai_service.run_conversation.assert_called_once()
    request = ai_service.run_conversation.call_args.args[0]
    assert isinstance(request, AIConversationRequest)
    assert request.chat_id == "558"
    assert request.prompt_text == "hola gordo"
    assert request.handler_func is deps.handle_ai_stream
    assert request.is_spontaneous is True
    mock_send_msg.assert_not_called()
    mock_save_message.assert_any_call(
        "558", "bot_999", "respuesta espontanea", ANY, role="assistant"
    )


def test_message_handler_stores_user_message_when_bot_should_not_respond(monkeypatch):
    from api.bot.message_handler import handle_msg

    make_deps, redis_client = _build_message_handler_deps()
    mock_send_msg = MagicMock()
    mock_save_message = MagicMock()
    deps = make_deps(
        should_gordo_respond=MagicMock(return_value=False),
        send_msg=mock_send_msg,
        save_message_to_redis=mock_save_message,
    )
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    message = {
        "message_id": 402,
        "chat": {"id": 556, "type": "group"},
        "from": {"id": 1002, "first_name": "Ana", "username": "ana"},
        "text": "hola gordo",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_send_msg.assert_not_called()
    mock_save_message.assert_called_once()
    args, kwargs = mock_save_message.call_args
    assert args == ("556", "402", "Ana: hola gordo", redis_client)
    assert kwargs["role"] == "user"
    assert kwargs["user_id"] == "1002"


@pytest.mark.parametrize(
    ("chat_type", "text", "expected_saved"),
    [
        (
            "private",
            "https://www.instagram.com/reel/DYcbT4OBtyZ/",
            "Ana: https://www.instagram.com/reel/DYcbT4OBtyZ/",
        ),
        (
            "group",
            "mirá https://x.com/user/status/1",
            "Ana: mirá https://x.com/user/status/1",
        ),
    ],
)
def test_message_handler_suppresses_supported_link_when_not_replaced(
    monkeypatch,
    chat_type,
    text,
    expected_saved,
):
    from api.bot.message_handler import handle_msg

    make_deps, redis_client = _build_message_handler_deps()
    mock_send_msg = MagicMock()
    mock_save_message = MagicMock()
    mock_should_respond = MagicMock(return_value=True)
    link_service = MagicMock()
    link_service.replace.return_value = (text, False, [])
    deps = make_deps(
        should_gordo_respond=mock_should_respond,
        send_msg=mock_send_msg,
        save_message_to_redis=mock_save_message,
        link_service=link_service,
    )
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    message = {
        "message_id": 501,
        "chat": {"id": 557, "type": chat_type},
        "from": {"id": 1003, "first_name": "Ana", "username": "ana"},
        "text": text,
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_should_respond.assert_not_called()
    mock_send_msg.assert_not_called()
    mock_save_message.assert_called_once()
    args, kwargs = mock_save_message.call_args
    assert args == (
        "557",
        "501",
        expected_saved,
        redis_client,
    )
    assert kwargs["role"] == "user"
    assert kwargs["user_id"] == "1003"


def test_message_handler_private_text_without_supported_link_still_responds(monkeypatch):
    from api.bot.message_handler import handle_msg

    make_deps, _ = _build_message_handler_deps()
    mock_should_respond = MagicMock(return_value=True)
    deps = make_deps(should_gordo_respond=mock_should_respond)
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    message = {
        "message_id": 503,
        "chat": {"id": 559, "type": "private"},
        "from": {"id": 1005, "first_name": "Ana", "username": "ana"},
        "text": "hola gordo",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_should_respond.assert_called_once()
    deps.handle_ai_stream.assert_called_once()


@pytest.mark.parametrize(
    "message",
    [
        {
            "message_id": 504,
            "chat": {"id": 560, "type": "private"},
            "from": {"id": 1006, "first_name": "Ana", "username": "ana"},
            "text": "/ask qué onda https://x.com/user/status/1",
        },
        {
            "message_id": 505,
            "chat": {"id": 561, "type": "group"},
            "from": {"id": 1007, "first_name": "Ana", "username": "ana"},
            "reply_to_message": {
                "message_id": 404,
                "from": {"username": "testbot"},
                "text": "respuesta anterior",
            },
            "text": "qué onda https://x.com/user/status/1",
        },
    ],
)
def test_message_handler_explicit_intent_with_link_bypasses_replacement(
    monkeypatch,
    message,
):
    from api.bot.message_handler import handle_msg

    make_deps, _ = _build_message_handler_deps()
    mock_replace_links = MagicMock(return_value=(
        "qué onda https://fixupx.com/user/status/1",
        True,
        ["https://x.com/user/status/1"],
    ))
    link_service = MagicMock()
    link_service.replace = mock_replace_links
    deps = make_deps(link_service=link_service)
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_replace_links.assert_not_called()
    deps.handle_ai_stream.assert_called_once()


def test_message_handler_replaces_plain_link_reply_to_bot(monkeypatch):
    from api.bot.message_handler import handle_msg

    make_deps, redis_client = _build_message_handler_deps()
    link_service = MagicMock()
    link_service.replace.return_value = (
        "https://fixupx.com/status/2071194631480414436",
        True,
        ["https://x.com/i/status/2071194631480414436"],
    )
    link_service.build_context.return_value = ""
    link_service.download_oversized_instagram_video.return_value = None
    deps = make_deps(link_service=link_service)
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    message = {
        "message_id": 506,
        "chat": {"id": 562, "type": "group"},
        "from": {"id": 1008, "first_name": "Profe", "username": "profe"},
        "reply_to_message": {
            "message_id": 405,
            "from": {"username": "testbot"},
            "text": "https://kkinstagram.com/reel/DaHY1E7Ar9d/?tg=495182",
        },
        "text": "https://x.com/i/status/2071194631480414436",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    link_service.replace.assert_called_once_with(
        "https://x.com/i/status/2071194631480414436"
    )
    deps.send_msg.assert_called_once_with(
        "562",
        "https://fixupx.com/status/2071194631480414436\n\ncompartido por @profe",
        "405",
        ["https://x.com/i/status/2071194631480414436"],
    )
    deps.save_message_to_redis.assert_called_once_with(
        "562",
        "bot_999",
        "https://fixupx.com/status/2071194631480414436\n\ncompartido por @profe",
        redis_client,
    )
    deps.handle_ai_stream.assert_not_called()


def test_handle_msg_with_unknown_command():
    from api.bot.message_handler import handle_msg

    mock_config_redis = MagicMock()
    mock_redis = MagicMock()
    mock_config_redis.return_value = mock_redis
    mock_send_msg = MagicMock()

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=mock_config_redis,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=True),
        should_gordo_respond=MagicMock(return_value=False),
    )

    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "group"},
        "from": {"first_name": "John", "username": "john"},
        "text": "/unknown_command",
    }

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_send_msg.assert_not_called()


def test_handle_msg_with_exception():
    from api.index import handle_msg

    with (
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
        patch("api.index.app_runtime.admin.report") as mock_admin_report,
        patch("os.environ.get") as mock_env,
    ):
        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_config_redis.side_effect = Exception("Redis error")

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "hello",
        }

        result = handle_msg(message)
        assert result == "error procesando mensaje"
        mock_admin_report.assert_called_once()


def test_handle_msg_edge_cases(monkeypatch):
    from api.bot.message_handler import handle_msg

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    mock_redis = MagicMock()

    def redis_get(key):
        if key == "chat_config:456":
            return json.dumps(CHAT_CONFIG_DEFAULTS)
        if key.startswith("prices:"):
            return json.dumps({"timestamp": 123, "data": {}})
        return None

    mock_redis.get.side_effect = redis_get

    mock_send_msg = MagicMock()
    mock_send_typing = MagicMock()
    mock_handle_ai_stream = MagicMock(
        side_effect=_simulate_streamed_ai_response(mock_send_msg, "test response")
    )
    mock_should_respond = MagicMock(return_value=False)
    mock_admin_report = MagicMock()
    mock_cached_requests = MagicMock(return_value=None)
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: mock_redis,
        send_msg=mock_send_msg,
        send_typing=mock_send_typing,
        gen_random=MagicMock(return_value="no boludo"),
        cached_requests=mock_cached_requests,
        admin_report=mock_admin_report,
        handle_ai_stream=mock_handle_ai_stream,
        should_gordo_respond=mock_should_respond,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
    )

    message = {
        "message_id": "123",
        "chat": {"id": "456", "type": "private"},
        "from": {"id": 14, "first_name": "John", "username": "john123"},
    }
    assert handle_msg(message, deps) == "ok"
    mock_send_msg.assert_not_called()

    message["text"] = "   \n   \t   "
    mock_send_msg.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_send_msg.assert_not_called()

    mock_should_respond.return_value = True
    message["text"] = "test"
    mock_send_msg.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args.args[1] == "test response"

    mock_redis.get.side_effect = lambda key: "invalid json"
    mock_send_msg.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_redis.get.side_effect = redis_get

    mock_admin_report.reset_mock()
    message = {"message_id": "123"}
    mock_send_msg.reset_mock()
    result = handle_msg(message, deps)
    assert result == "ok"
    mock_admin_report.assert_not_called()

    mock_admin_report.reset_mock()
    message = {
        "message_id": "123",
        "chat": {"id": None},
        "from": {"username": None},
        "text": None,
    }
    mock_send_msg.reset_mock()
    result = handle_msg(message, deps)
    assert result == "ok"
    mock_admin_report.assert_not_called()

    mock_admin_report.reset_mock()
    message = {"chat": {"id": "456"}, "from": {"first_name": "John"}}
    mock_send_msg.reset_mock()
    result = handle_msg(message, deps)
    assert result == "ok"
    mock_admin_report.assert_not_called()
