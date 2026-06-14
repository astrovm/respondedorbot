from tests.support import *
from tests.message_handler_support import (
    _build_message_handler_deps,
    _private_photo_message,
    _simulate_streamed_ai_response,
)


def test_handle_msg_image_conversation_charges_media_and_response_credits(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"img-bytes")
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        download_telegram_file=mock_download,
        resize_image_if_needed=MagicMock(return_value=b"img-resized"),
        encode_image_to_base64=MagicMock(return_value="abc"),
        should_gordo_respond=MagicMock(return_value=True),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=_simulate_streamed_ai_response(mock_send_msg, "todo piola"),
    )
    message = _private_photo_message(message_id=22, chat_id=555, user_id=99)

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_download.called
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_token_signal_skips_ai_flow(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_should = MagicMock(return_value=True)
    mock_ai_response = MagicMock(return_value="ai")

    monkeypatch.setattr(
        "api.bot.message_handler.handle_token_signal_message",
        lambda *_args, **_kwargs: True,
    )

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        should_gordo_respond=mock_should,
        handle_ai_response=mock_ai_response,
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 10,
        "chat": {"id": 100, "type": "group"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump",
    }

    assert handle_msg(message, deps) == "ok"
    mock_should.assert_not_called()
    mock_ai_response.assert_not_called()


def test_handle_msg_image_conversation_with_two_provider_requests_reserves_base_and_media(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"img-bytes")
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    fake_handle_ai_response = _simulate_streamed_ai_response(
        mock_send_msg,
        "todo piola x2",
        billing_segments=[
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
        ],
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        download_telegram_file=mock_download,
        resize_image_if_needed=MagicMock(return_value=b"img-resized"),
        encode_image_to_base64=MagicMock(return_value="abc"),
        should_gordo_respond=MagicMock(return_value=True),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    message = _private_photo_message(message_id=33, chat_id=555, user_id=991)

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_download.called
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_settles_in_single_batch(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"img-bytes")
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = MagicMock(
        return_value={"ok": True, "source": "user"}
    )

    fake_handle_ai_response = _simulate_streamed_ai_response(
        mock_send_msg,
        "todo piola",
        billing_segments=[
            {
                "kind": "vision",
                "model": "google/gemini-3.1-flash-lite-preview",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ],
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    with (
        patch(
            "api.bot.message_handler.AIMessageBilling.settle_reserved_ai_credits_batch",
            autospec=True,
        ) as mock_settle_batch,
        patch(
            "api.bot.message_handler.AIMessageBilling.settle_reserved_ai_credits",
            autospec=True,
        ) as mock_settle_single,
    ):
        make_deps, _ = _build_message_handler_deps()
        deps = make_deps(
            config_redis=lambda: redis_client,
            send_msg=mock_send_msg,
            download_telegram_file=mock_download,
            resize_image_if_needed=MagicMock(return_value=b"img-resized"),
            encode_image_to_base64=MagicMock(return_value="abc"),
            should_gordo_respond=MagicMock(return_value=True),
            check_provider_available=MagicMock(return_value=True),
            credits_db_service=mock_credits,
            get_chat_history=MagicMock(return_value=[]),
            build_ai_messages=MagicMock(
                return_value=[{"role": "user", "content": "hola"}]
            ),
            handle_ai_response=fake_handle_ai_response,
        )
        message = {
            "message_id": 44,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 992, "first_name": "Ana", "username": "ana"},
            "photo": [{"file_id": "img1"}],
        }

        result = handle_msg(message, deps)

    assert result == "ok"
    mock_settle_batch.assert_called_once()
    reservations = mock_settle_batch.call_args.args[1]
    assert len(reservations) == 2
    mock_settle_single.assert_not_called()
    mock_send_msg.assert_called_once()


def test_handle_msg_command_reply_to_link_fix_message_is_not_blocked(monkeypatch):
    from api.core import config as config_module
    from api.bot.message_handler import handle_msg

    config_module.reset_cache()
    chat_config = {
        **CHAT_CONFIG_DEFAULTS,
        "ai_command_followups": False,
        "ignore_link_fix_followups": True,
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(chat_config)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "You are a test bot")
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "gordo,test,bot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        get_bot_message_metadata=MagicMock(return_value=None),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(
            return_value=[{"role": "user", "content": "investigá eso"}]
        ),
        handle_ai_response=_simulate_streamed_ai_response(mock_send_msg, "respuesta ok"),
    )
    message = {
        "message_id": 201,
        "chat": {"id": 555, "type": "group"},
        "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
        "text": "/ask investigá eso",
        "reply_to_message": {
            "message_id": 200,
            "from": {"username": "testbot"},
            "text": "https://fixupx.com/status/2032173338240467235",
        },
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_charge.assert_called_once()
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args[0][1] == "respuesta ok"


def test_message_links_handle_link_replacement_delete_mode_stores_fixed_context():
    message_links = __import__("api.bot.message_links", fromlist=["message_links"])
    handle_link_replacement = message_links.handle_link_replacement

    deps = MagicMock()
    deps.link_service.replace.return_value = (
        "mirá https://fixupx.com/user/status/1",
        True,
        ["https://x.com/user/status/1"],
    )
    deps.link_service.build_context.return_value = "links: x original"
    deps.link_service.download_oversized_instagram_video.return_value = None
    deps.send_msg.return_value = 777
    redis_client = MagicMock()
    message = {
        "from": {"first_name": "Ana", "last_name": "Pérez"},
        "reply_to_message": {"message_id": 42},
    }

    handled = handle_link_replacement(
        deps,
        chat_config={"link_mode": "delete"},
        message=message,
        message_text="mirá https://x.com/user/status/1",
        chat_id="555",
        message_id="100",
        redis_client=redis_client,
    )

    assert handled is True
    deps.delete_msg.assert_called_once_with("555", "100")
    deps.send_msg.assert_called_once_with(
        "555",
        "mirá https://fixupx.com/user/status/1\n\ncompartido por Ana Pérez",
        "42",
        ["https://x.com/user/status/1"],
    )
    deps.save_message_to_redis.assert_called_once_with(
        "555",
        "bot_777",
        "mirá https://fixupx.com/user/status/1\n\ncompartido por Ana Pérez\n\nlinks: x original",
        redis_client,
    )


def test_message_links_uploads_oversized_instagram_video():
    from api.bot.message_links import handle_link_replacement

    deps = MagicMock()
    deps.link_service.replace.return_value = (
        "https://eeinstagram.com/reel/example?tg=2",
        True,
        ["https://www.instagram.com/reel/example"],
    )
    deps.link_service.download_oversized_instagram_video.return_value = b"video"
    deps.send_video.return_value = 778
    deps.link_service.build_context.return_value = ""

    handled = handle_link_replacement(
        deps,
        chat_config={"link_mode": "reply"},
        message={"from": {"username": "ana"}},
        message_text="https://www.instagram.com/reel/example",
        chat_id="555",
        message_id="100",
        redis_client=MagicMock(),
    )

    assert handled is True
    deps.send_video.assert_called_once_with(
        "555",
        b"video",
        caption="https://eeinstagram.com/reel/example?tg=2\n\ncompartido por @ana",
        msg_id="100",
        buttons=["https://www.instagram.com/reel/example"],
    )
    deps.send_msg.assert_not_called()


def test_message_links_falls_back_to_link_when_video_upload_fails():
    from api.bot.message_links import handle_link_replacement

    deps = MagicMock()
    deps.link_service.replace.return_value = (
        "https://eeinstagram.com/reel/example?tg=2",
        True,
        ["https://www.instagram.com/reel/example"],
    )
    deps.link_service.download_oversized_instagram_video.return_value = b"video"
    deps.send_video.return_value = None
    deps.send_msg.return_value = 779
    deps.link_service.build_context.return_value = ""

    handled = handle_link_replacement(
        deps,
        chat_config={"link_mode": "reply"},
        message={"from": {}},
        message_text="https://www.instagram.com/reel/example",
        chat_id="555",
        message_id="100",
        redis_client=MagicMock(),
    )

    assert handled is True
    deps.send_msg.assert_called_once_with(
        "555",
        "https://eeinstagram.com/reel/example?tg=2",
        "100",
        ["https://www.instagram.com/reel/example"],
    )


def test_handle_msg_ai_flow_settles_with_single_base_reserve_when_usage_is_tiny(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    fake_handle_ai_response = _simulate_streamed_ai_response(
        mock_send_msg,
        "respuesta ok",
        billing_segments=[
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
        ],
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    message = {
        "message_id": 199,
        "chat": {"id": 555, "type": "private"},
        "from": {"id": 2001, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_charge.call_count == 1
    assert mock_send_msg.call_args[0][1] == "respuesta ok"


def test_handle_msg_ai_flow_allows_openrouter_fallback_when_groq_rate_limit_blocks(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_handle_ai_response = MagicMock(
        side_effect=_simulate_streamed_ai_response(mock_send_msg, "respuesta ok")
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = MagicMock(
        return_value={"ok": True, "source": "user"}
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=False),
        has_openrouter_fallback=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=mock_handle_ai_response,
    )
    message = {
        "message_id": 301,
        "chat": {"id": 777, "type": "group"},
        "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_handle_ai_response.assert_called_once()
    mock_send_msg.assert_called_once_with("777", "respuesta ok")


def test_handle_msg_ai_flow_keeps_single_reserve_for_three_tiny_segments(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    fake_handle_ai_response = _simulate_streamed_ai_response(
        mock_send_msg,
        "respuesta ok x3",
        billing_segments=[
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            },
        ],
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    message = {
        "message_id": 200,
        "chat": {"id": 555, "type": "private"},
        "from": {"id": 2002, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_charge.call_count == 1
    assert mock_send_msg.call_args[0][1] == "respuesta ok x3"


def test_handle_msg_transcribe_image_does_not_preprocess_image_or_double_charge(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        download_telegram_file=mock_download,
        handle_transcribe_with_message_result=MagicMock(
            return_value=(
                "🖼️ en la imagen veo: todo piola",
                [
                    {
                        "kind": "vision",
                        "model": "google/gemini-3.1-flash-lite-preview",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ],
            ),
        ),
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 23,
        "chat": {"id": 556, "type": "private"},
        "from": {"id": 100, "first_name": "Ana", "username": "ana"},
        "text": "/transcribe",
        "reply_to_message": {"message_id": 24, "photo": [{"file_id": "img_reply"}]},
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_download.assert_called_once_with("img_reply")
    assert mock_charge.call_count == 1
    assert mock_charge.call_args.kwargs["amount"] == 1
    mock_send_msg.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_vision():
    from api.billing.ai import AIMessageBilling
    from api.ai.service import build_ai_service
    from api.bot.message_handler import PreparedMessage, _run_ai_flow

    deps = MagicMock()
    handle_ai_response = MagicMock(return_value="respuesta ok")
    deps.ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=False),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(return_value="no boludo"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=lambda **_: "insufficient",
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/ask",
        chat_id="557",
        chat_type="private",
        user_id=101,
        numeric_chat_id=557,
        message={"from": {"first_name": "Ana"}},
    )

    response_msg, handled = _run_ai_flow(
        deps,
        chat_id="557",
        message={"chat": {"id": 557, "type": "private"}},
        user_id=None,
        prepared_message=PreparedMessage(
            message_text="/ask describe",
            photo_file_id="img_1",
            audio_file_id=None,
            resized_image_data=b"resized",
        ),
        billing_helper=billing_helper,
        prompt_text="Describe",
        reply_context_text=None,
        user_identity="101",
        handler_func=lambda: None,
        redis_client=MagicMock(),
    )

    assert handled is True
    assert response_msg == "respuesta ok"
    handle_ai_response.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_transcribe():
    from api.billing.ai import AIMessageBilling
    from api.ai.service import build_ai_service
    from api.bot.message_handler import PreparedMessage, _run_ai_flow

    deps = MagicMock()
    handle_ai_response = MagicMock(return_value="🖼️ en la imagen veo: todo piola")
    deps.ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=False),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(return_value="no boludo"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=lambda **_: "insufficient",
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/transcribe",
        chat_id="558",
        chat_type="private",
        user_id=102,
        numeric_chat_id=558,
        message={"from": {"first_name": "Ana"}},
    )

    response_msg, handled = _run_ai_flow(
        deps,
        chat_id="558",
        message={"chat": {"id": 558, "type": "private"}},
        user_id=None,
        prepared_message=PreparedMessage(
            message_text="/transcribe",
            photo_file_id="img_reply",
            audio_file_id=None,
            resized_image_data=b"resized",
        ),
        billing_helper=billing_helper,
        prompt_text="Describe what you see in this image in detail.",
        reply_context_text=None,
        user_identity="102",
        handler_func=lambda: None,
        redis_client=MagicMock(),
    )

    assert handled is True
    assert response_msg == "🖼️ en la imagen veo: todo piola"
    handle_ai_response.assert_called_once()
