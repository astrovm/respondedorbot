from tests.support import *
from tests.message_handler_support import (
    _build_message_handler_deps,
    _private_photo_message,
    _private_voice_message,
    _simulate_streamed_ai_response,
    _telegram_message,
)


def test_handle_msg():
    from api.bot.message_handler import handle_msg

    mock_config_redis = MagicMock()
    mock_redis = MagicMock()
    mock_config_redis.return_value = mock_redis

    def redis_get(key):
        if key in {"chat_config:456"}:
            return json.dumps(CHAT_CONFIG_DEFAULTS)
        return None

    mock_redis.get.side_effect = redis_get
    mock_redis.lrange.return_value = []

    mock_send_msg = MagicMock()
    mock_handle_ai_stream = MagicMock(
        side_effect=_simulate_streamed_ai_response(mock_send_msg, "test response")
    )
    mock_send_typing = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    def check_provider_side_effect(*args, **kwargs):
        return True

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=mock_config_redis,
        send_msg=mock_send_msg,
        handle_ai_stream=mock_handle_ai_stream,
        send_typing=mock_send_typing,
        check_provider_available=MagicMock(side_effect=[True, False]),
        credits_db_service=mock_credits,
    )

    message = {
        "message_id": "123",
        "chat": {"id": "456", "type": "private"},
        "from": {"id": 9, "first_name": "John", "username": "john123"},
        "text": "/help",
    }
    assert handle_msg(message, deps) == "ok"
    mock_send_msg.assert_called_once()

    message["text"] = "hello bot"
    mock_send_msg.reset_mock()
    mock_send_typing.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_send_msg.assert_called_once()
    mock_handle_ai_stream.assert_called_once()

    mock_send_msg.reset_mock()
    mock_send_typing.reset_mock()
    mock_handle_ai_stream.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_handle_ai_stream.assert_not_called()


def test_handle_msg_with_crypto_command():
    from api.bot.message_handler import handle_msg

    mock_config_redis = MagicMock()
    mock_redis = MagicMock()
    mock_config_redis.return_value = mock_redis

    def redis_get(key):
        if key == "chat_config:123":
            return json.dumps(CHAT_CONFIG_DEFAULTS)
        return None

    mock_redis.get.side_effect = redis_get

    mock_send_msg = MagicMock()
    mock_get_prices = MagicMock(return_value="BTC: 50000")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=mock_config_redis,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=True),
    )

    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "private"},
        "from": {"first_name": "John", "username": "john"},
        "text": "/prices btc",
    }

    with patch("api.index.app_runtime.prices.get_prices", mock_get_prices):
        result = handle_msg(message, deps)
    assert result == "ok"
    mock_get_prices.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_image(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"image data")
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = MagicMock(
        return_value={"ok": True, "source": "user"}
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setenv("TELEGRAM_TOKEN", "test_token")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        describe_image_groq=MagicMock(return_value="A beautiful landscape"),
        download_telegram_file=mock_download,
        resize_image_if_needed=MagicMock(return_value=b"resized image data"),
        encode_image_to_base64=MagicMock(return_value="base64_encoded_image"),
        cached_requests=MagicMock(
            return_value={"description": "A beautiful landscape"}
        ),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
    )
    message = _private_photo_message(message_id=1, user_id=11, file_id="photo_123")

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_download.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_audio(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    mock_send_msg = MagicMock()
    mock_transcribe = MagicMock(
        return_value=(
            "transcribed text",
            None,
            {
                "kind": "transcribe",
                "model": "whisper-large-v3",
                "audio_seconds": 1,
            },
        ),
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = MagicMock(
        return_value={"ok": True, "source": "user"}
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=mock_transcribe,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
    )
    message = _private_voice_message(message_id=1, user_id=12, duration=10)

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    mock_send_msg.assert_called_once()


def test_handle_msg_group_audio_without_invocation_skips_auto_transcription(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.lrange.return_value = []

    def redis_get(key):
        if key == "chat_config:123":
            return json.dumps(CHAT_CONFIG_DEFAULTS)
        return None

    redis_client.get.side_effect = redis_get
    mock_send_msg = MagicMock()
    mock_transcribe = MagicMock(return_value=("transcribed text", None, None))

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "You are a test bot")
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "gordo,test,bot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=mock_transcribe,
    )
    message = _telegram_message(
        message_id=1,
        chat_type="group",
        user_id=None,
        voice={"file_id": "voice_123"},
    )

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_transcribe.assert_not_called()
    mock_send_msg.assert_not_called()


def test_handle_msg_with_transcribe_command(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    mock_send_msg = MagicMock()
    mock_handle_transcribe = MagicMock(
        return_value=(
            "Transcription result",
            [
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                }
            ],
        ),
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
        handle_transcribe_with_message_result=mock_handle_transcribe,
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 13, "first_name": "John", "username": "john"},
        "text": "/transcribe",
        "reply_to_message": {
            "message_id": 2,
            "voice": {"file_id": "voice_123", "duration": 10},
        },
    }

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_handle_transcribe.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_charges_media_credits(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_handle_transcribe = MagicMock(
        return_value=(
            "🎵 te saqué esto del audio: todo piola",
            [
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                }
            ],
        ),
    )
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_refund = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge
    mock_credits.refund_ai_charge = mock_refund

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        check_provider_available=MagicMock(return_value=True),
        handle_transcribe_with_message_result=mock_handle_transcribe,
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 1,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 77, "first_name": "John", "username": "john"},
        "text": "/transcribe",
        "reply_to_message": {
            "message_id": 2,
            "voice": {"file_id": "voice_123", "duration": 10},
        },
    }

    result = handle_msg(message, deps)
    assert result == "ok"
    assert mock_charge.call_count == 1
    assert mock_charge.call_args.kwargs["amount"] == 1
    mock_refund.assert_not_called()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_refunds_on_unsuccessful_response(
    monkeypatch,
):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_handle_transcribe = MagicMock(
        return_value=("no pude sacar nada de ese audio, probá más tarde", []),
    )
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_refund = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge
    mock_credits.refund_ai_charge = mock_refund

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        handle_transcribe_with_message_result=mock_handle_transcribe,
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 2,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 177, "first_name": "John", "username": "john"},
        "text": "/transcribe",
        "reply_to_message": {
            "message_id": 3,
            "voice": {"file_id": "voice_123", "duration": 10},
        },
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_charge.call_count == 1
    assert mock_charge.call_args.kwargs["amount"] == 1
    mock_refund.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_rejects_audio_without_duration(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"audio-bytes")
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        handle_transcribe_with_message_result=MagicMock(),
        download_telegram_file=mock_download,
        measure_audio_duration_seconds=MagicMock(return_value=None),
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 2,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 177, "first_name": "John", "username": "john"},
        "text": "/transcribe",
        "reply_to_message": {"message_id": 3, "voice": {"file_id": "voice_123"}},
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_credits.charge_ai_credits.assert_not_called()


def test_handle_msg_auto_audio_charges_media_credits(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_transcribe = MagicMock(
        return_value=(
            "audio transcripto",
            None,
            {
                "kind": "transcribe",
                "model": "whisper-large-v3",
                "audio_seconds": 1,
            },
        ),
    )
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=mock_transcribe,
        should_gordo_respond=MagicMock(return_value=True),
        credits_db_service=mock_credits,
    )
    message = _private_voice_message(message_id=1, user_id=88, duration=10)

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    assert mock_charge.call_count == 2
    assert mock_charge.call_args_list[0].kwargs["amount"] == 1
    mock_send_msg.assert_called_once()


def test_handle_msg_auto_audio_rejects_missing_duration(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock(return_value=b"audio-bytes")
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=MagicMock(),
        download_telegram_file=mock_download,
        measure_audio_duration_seconds=MagicMock(return_value=None),
        credits_db_service=mock_credits,
    )
    message = _private_voice_message(message_id=1, user_id=88)

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_credits.charge_ai_credits.assert_not_called()


def test_handle_msg_auto_audio_measures_duration_when_missing_in_message(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_transcribe = MagicMock(
        return_value=(
            "audio transcripto",
            None,
            {
                "kind": "transcribe",
                "model": "whisper-large-v3",
                "audio_seconds": 12,
            },
        ),
    )
    mock_download = MagicMock(return_value=b"audio-bytes")
    mock_measure = MagicMock(return_value=12.0)
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=mock_transcribe,
        download_telegram_file=mock_download,
        measure_audio_duration_seconds=mock_measure,
        should_gordo_respond=MagicMock(return_value=True),
        credits_db_service=mock_credits,
    )
    message = _private_voice_message(message_id=1, user_id=88)

    result = handle_msg(message, deps)

    assert result == "ok"
    mock_download.assert_called_once_with("voice_123")
    mock_measure.assert_called_once_with(b"audio-bytes")
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_auto_audio_skips_transcription_when_should_not_respond(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_transcribe = MagicMock()
    mock_download = MagicMock()
    mock_measure = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=mock_transcribe,
        download_telegram_file=mock_download,
        measure_audio_duration_seconds=mock_measure,
        should_gordo_respond=MagicMock(return_value=False),
        credits_db_service=mock_credits,
    )
    message = _private_voice_message(message_id=1, user_id=88)

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_download.assert_not_called()
    mock_measure.assert_not_called()
    mock_transcribe.assert_not_called()
    mock_credits.charge_ai_credits.assert_not_called()
    mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_skips_image_download_when_should_not_respond(monkeypatch):
    from api.bot.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_download = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        download_telegram_file=mock_download,
        should_gordo_respond=MagicMock(return_value=False),
        credits_db_service=mock_credits,
    )
    message = {
        "message_id": 1,
        "chat": {"id": 555, "type": "private"},
        "from": {"id": 88, "first_name": "Ana", "username": "ana"},
        "photo": [{"file_id": "photo_123"}],
    }

    result = handle_msg(message, deps)
    assert result == "ok"
    mock_download.assert_not_called()
    mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_plus_ai_response_charges_three_requests(monkeypatch):
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
        "respuesta final",
        billing_segments=[
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {
                    "input_tokens": 1,
                    "input_non_cached_tokens": 1,
                    "output_tokens": 1,
                },
            }
        ],
    )

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        _transcribe_audio_file=MagicMock(
            return_value=(
                "audio transcripto",
                None,
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                },
            ),
        ),
        should_gordo_respond=MagicMock(return_value=True),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    message = {
        "message_id": 31,
        "chat": {"id": 123, "type": "private"},
        "from": {"id": 89, "first_name": "John", "username": "john"},
        "voice": {"file_id": "voice_123", "duration": 10},
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()
