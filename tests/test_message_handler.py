from tests.support import *  # noqa: F401,F403

def test_handle_msg_topup_private_returns_keyboard():
    message = {
        "message_id": "10",
        "chat": {"id": "100", "type": "private"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "/topup",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("api.index.ensure_callback_updates_enabled"), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.build_topup_keyboard",
        return_value={"inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]},
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_send_msg.assert_called_once_with(
        "100",
        "elegí cuánto querés cargar, papá:",
        "10",
        reply_markup={"inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]},
    )


def test_handle_msg_topup_group_redirects_private():
    message = {
        "message_id": "10",
        "chat": {"id": "100", "type": "group"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "/topup",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "os.environ.get",
        side_effect=lambda key, default=None: {"TELEGRAM_USERNAME": "testbot"}.get(
            key, default
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert "@testbot" in mock_send_msg.call_args[0][1]


def test_handle_msg_balance_private_uses_personal_balance():
    message = {
        "message_id": "11",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/balance",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index._fetch_balance", return_value=42
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert "42" in mock_send_msg.call_args[0][1]
    assert "/topup" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_moves_credits():
    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 20",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.credits_db_service.transfer_user_to_chat",
        return_value={"ok": True, "user_balance": 30, "chat_balance": 120},
    ) as mock_transfer:
        result = handle_msg(message)

    assert result == "ok"
    mock_transfer.assert_called_once_with(user_id=55, chat_id=202, amount=20)
    assert "le pasé 20 créditos al grupo" in mock_send_msg.call_args[0][1]


def test_handle_msg_successful_payment_credits_user():
    message = {
        "message_id": "13",
        "chat": {"id": "303", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "successful_payment": {
            "currency": "XTR",
            "total_amount": 50,
            "invoice_payload": "topup:p100:55",
            "telegram_payment_charge_id": "charge_1",
        },
    }

    with patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.credits_db_service.record_star_payment",
        return_value={"inserted": True, "user_balance": 777},
    ) as mock_record, patch("api.index.send_msg") as mock_send_msg:
        result = handle_msg(message)

    assert result == "ok"
    mock_record.assert_called_once()
    assert "777" in mock_send_msg.call_args[0][1]


def test_handle_msg_refunds_credits_on_internal_ai_fallback(monkeypatch):
    message = {
        "message_id": "14",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index.credits_db_service.refund_ai_charge"
    ) as mock_refund, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.ask_ai", return_value="[[AI_FALLBACK]]no boludo"
    ), patch(
        "api.index.send_msg", return_value=999
    ) as mock_send, patch(
        "api.index.save_message_to_redis"
    ), patch(
        "api.index.save_bot_message_metadata"
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_refund.assert_called_once()
    assert mock_send.call_args[0][1] == "no boludo"


def test_handle_msg_refunds_all_charges_when_fallback_after_multiple_provider_requests(
    monkeypatch,
):
    message = {
        "message_id": "14b",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 3
            response_meta["ai_fallback"] = True
        return "no boludo"

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index.credits_db_service.refund_ai_charge"
    ) as mock_refund, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", side_effect=fake_handle_ai_response
    ), patch(
        "api.index.send_msg", return_value=999
    ), patch(
        "api.index.save_message_to_redis"
    ), patch(
        "api.index.save_bot_message_metadata"
    ):
        result = handle_msg(message)

    assert result == "ok"
    # provider_request_count=3 => 3 cobros y 3 reintegros en fallback
    assert mock_refund.call_count == 3


def test_handle_msg_insufficient_credits_returns_random_plus_topup_hint(monkeypatch):
    message = {
        "message_id": "15",
        "chat": {"id": "405", "type": "private"},
        "from": {"id": 78, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    with patch("api.index.config_redis", return_value=redis_client), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": False, "user_balance": 0, "chat_balance": 0},
    ), patch(
        "api.index.gen_random", return_value="no boludo"
    ), patch(
        "api.index.send_msg", return_value=1001
    ) as mock_send, patch(
        "api.index.save_message_to_redis"
    ), patch(
        "api.index.save_bot_message_metadata"
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert (
        mock_send.call_args[0][1]
        == "no boludo\n\nte quedaste seco de créditos ia, boludo.\nsaldo: 0\nmetele /topup si querés que siga laburando"
    )


def test_handle_rate_limit():
    from api.index import handle_rate_limit

    with patch("api.index.send_typing") as mock_send_typing, patch(
        "time.sleep"
    ) as mock_sleep, patch("api.index.gen_random") as mock_gen_random, patch(
        "os.environ.get"
    ) as mock_env:

        chat_id = "123"
        message = {"from": {"first_name": "John"}}
        mock_gen_random.return_value = "no boludo"
        mock_env.return_value = "fake_token"  # Mock TELEGRAM_TOKEN

        response = handle_rate_limit(chat_id, message)

        mock_send_typing.assert_called_once()
        mock_sleep.assert_called_once()
        assert response == "no boludo"


def test_handle_msg():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.ask_ai"
    ) as mock_ask_ai, patch(
        "api.index.send_typing"
    ) as mock_send_typing, patch(
        "api.index.check_global_rate_limit", side_effect=[True, False]
    ) as mock_global_rate_limit, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "time.sleep"
    ) as _mock_sleep:  # Add sleep mock to avoid delays  # noqa: F841

        mock_env.return_value = "testbot"
        mock_ask_ai.return_value = "test response"  # Mock ai response

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        def redis_get(key):
            if key in {"chat_config:123", "chat_config:456"}:
                return json.dumps(CHAT_CONFIG_DEFAULTS)
            return None

        mock_redis.get.side_effect = redis_get
        mock_redis.lrange.return_value = []  # Empty chat history

        mock_redis.get.side_effect = redis_get

        # Test basic message handling
        message = {
            "message_id": "123",
            "chat": {"id": "456", "type": "private"},
            "from": {"id": 9, "first_name": "John", "username": "john123"},
            "text": "/help",
        }

        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once()

        # Test message without command
        message["text"] = "hello bot"
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once()
        mock_send_typing.assert_called_once()
        mock_ask_ai.assert_called_once()

        # Test rate limited message
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        mock_ask_ai.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_typing.assert_called_once()
        mock_ask_ai.assert_not_called()
        assert mock_global_rate_limit.call_count == 2


def test_handle_msg_with_crypto_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.get_prices"
    ) as mock_get_prices:

        mock_env.return_value = "testbot"
        mock_rate_limit.return_value = True
        mock_get_prices.return_value = "BTC: 50000"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        def redis_get(key):
            if key == "chat_config:123":
                return json.dumps(CHAT_CONFIG_DEFAULTS)
            return None

        mock_redis.get.side_effect = redis_get

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/prices btc",
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_get_prices.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_image():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.describe_image_groq"
    ) as mock_describe, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.resize_image_if_needed"
    ) as mock_resize, patch(
        "api.index.encode_image_to_base64"
    ) as mock_encode, patch(
        "api.index.cached_requests"
    ) as mock_requests, patch(
        "api.index.send_typing"
    ) as mock_send_typing, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ):

        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot",
            "TELEGRAM_TOKEN": "test_token",
        }.get(key, default)
        mock_download.return_value = b"image data"
        mock_describe.return_value = "A beautiful landscape"
        mock_resize.return_value = b"resized image data"
        mock_encode.return_value = "base64_encoded_image"
        mock_requests.return_value = {"description": "A beautiful landscape"}

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 11, "first_name": "John", "username": "john"},
            "photo": [{"file_id": "photo_123"}],
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_download.assert_called_once()
        mock_encode.assert_called_once()
        mock_send_msg.assert_called_once()
        mock_send_typing.assert_called()


def test_handle_msg_with_audio():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.transcribe_audio_groq"
    ) as mock_transcribe, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ):

        mock_env.side_effect = lambda key: {
            "TELEGRAM_USERNAME": "testbot",
            "GROQ_API_KEY": "test_key",
        }.get(key)
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = "transcribed text"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 12, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123"},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_transcribe.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_group_audio_without_invocation_skips_auto_transcription():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index._transcribe_audio_file"
    ) as mock_transcribe:

        def env_side_effect(key, default=None):
            env_vars = {
                "TELEGRAM_USERNAME": "testbot",
                "BOT_SYSTEM_PROMPT": "You are a test bot",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key, default)

        mock_env.side_effect = env_side_effect
        mock_transcribe.return_value = ("transcribed text", None)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        def redis_get(key):
            if key == "chat_config:123":
                return json.dumps(CHAT_CONFIG_DEFAULTS)
            return None

        mock_redis.get.side_effect = redis_get
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "group"},
            "from": {"first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123"},
        }

        result = handle_msg(message)

        assert result == "ok"
        mock_transcribe.assert_not_called()
        mock_send_msg.assert_not_called()


def test_handle_msg_with_transcribe_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.handle_transcribe_with_message"
    ) as mock_handle_transcribe, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ):

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_handle_transcribe.return_value = "Transcription result"

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 13, "first_name": "John", "username": "john"},
            "text": "/transcribe",
            "reply_to_message": {"message_id": 2, "voice": {"file_id": "voice_123"}},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_handle_transcribe.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_charges_media_credits():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.handle_transcribe_with_message"
    ) as mock_handle_transcribe, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.credits_db_service.refund_ai_charge"
    ) as mock_refund:

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_rate_limit.return_value = True
        mock_handle_transcribe.return_value = "🎵 Transcripción: todo piola"

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 77, "first_name": "John", "username": "john"},
            "text": "/transcribe",
            "reply_to_message": {"message_id": 2, "voice": {"file_id": "voice_123"}},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_charge.assert_called_once_with(user_id=77, chat_id=None, amount=1)
        mock_refund.assert_not_called()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_refunds_on_unsuccessful_response():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.handle_transcribe_with_message",
        return_value="no pude sacar nada de ese audio, probá más tarde",
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.credits_db_service.refund_ai_charge"
    ) as mock_refund:
        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 2,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 177, "first_name": "John", "username": "john"},
            "text": "/transcribe",
            "reply_to_message": {"message_id": 3, "voice": {"file_id": "voice_123"}},
        }

        result = handle_msg(message)

    assert result == "ok"
    mock_charge.assert_called_once_with(user_id=177, chat_id=None, amount=1)
    mock_refund.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_auto_audio_charges_media_credits():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index._transcribe_audio_file", return_value=("audio transcripto", None)
    ) as mock_transcribe, patch(
        "api.index.should_gordo_respond", return_value=False
    ), patch(
        "api.index.save_message_to_redis"
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge:
        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 88, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123"},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
        mock_charge.assert_called_once_with(user_id=88, chat_id=None, amount=1)
        mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_plus_ai_response_charges_three_requests():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 2
        return "respuesta final"

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index._transcribe_audio_file", return_value=("audio transcripto", None)
    ), patch(
        "api.index.should_gordo_respond", return_value=True
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", side_effect=fake_handle_ai_response
    ), patch(
        "api.index.save_message_to_redis"
    ):
        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 31,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 89, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123"},
        }

        result = handle_msg(message)

    assert result == "ok"
    # 1 por transcripción de audio + 2 por respuesta IA con provider_request_count=2
    assert mock_charge.call_count == 3
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_charges_media_and_response_credits():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.download_telegram_file", return_value=b"img-bytes"
    ) as mock_download, patch(
        "api.index.resize_image_if_needed", return_value=b"img-resized"
    ), patch(
        "api.index.encode_image_to_base64", return_value="abc"
    ), patch(
        "api.index.should_gordo_respond", return_value=True
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", return_value="todo piola"
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 22,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 99, "first_name": "Ana", "username": "ana"},
            "photo": [{"file_id": "img1"}],
        }

        result = handle_msg(message)

    assert result == "ok"
    assert mock_download.called
    # 1 crédito por respuesta IA + 1 crédito por media (imagen/sticker)
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_with_two_provider_requests_charges_three():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 2
        return "todo piola x2"

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.download_telegram_file", return_value=b"img-bytes"
    ) as mock_download, patch(
        "api.index.resize_image_if_needed", return_value=b"img-resized"
    ), patch(
        "api.index.encode_image_to_base64", return_value="abc"
    ), patch(
        "api.index.should_gordo_respond", return_value=True
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", side_effect=fake_handle_ai_response
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 33,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 991, "first_name": "Ana", "username": "ana"},
            "photo": [{"file_id": "img1"}],
        }

        result = handle_msg(message)

    assert result == "ok"
    assert mock_download.called
    # 1 base + 1 media imagen + 1 extra por segundo request de proveedor
    assert mock_charge.call_count == 3
    mock_send_msg.assert_called_once()


def test_handle_msg_search_command_does_not_charge():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.charge_ai_credits"
    ) as mock_charge, patch(
        "api.index.search_command", return_value="resultado web"
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 99,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
            "text": "/buscar btc news",
        }

        result = handle_msg(message)

    assert result == "ok"
    mock_charge.assert_not_called()
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args[0][1] == "resultado web"


def test_handle_msg_ai_flow_charges_each_provider_request():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 2
        return "respuesta ok"

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", side_effect=fake_handle_ai_response
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 199,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 2001, "first_name": "Ana", "username": "ana"},
            "text": "/ask hola",
        }

        result = handle_msg(message)

    assert result == "ok"
    # 1 cobro base + 1 cobro extra por segundo request real al proveedor IA
    assert mock_charge.call_count == 2
    assert mock_send_msg.call_args[0][1] == "respuesta ok"


def test_handle_msg_ai_flow_charges_three_provider_requests():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 3
        return "respuesta ok x3"

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge, patch(
        "api.index.get_chat_history", return_value=[]
    ), patch(
        "api.index.build_ai_messages", return_value=[{"role": "user", "content": "hola"}]
    ), patch(
        "api.index.handle_ai_response", side_effect=fake_handle_ai_response
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 200,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 2002, "first_name": "Ana", "username": "ana"},
            "text": "/ask hola",
        }

        result = handle_msg(message)

    assert result == "ok"
    assert mock_charge.call_count == 3
    assert mock_send_msg.call_args[0][1] == "respuesta ok x3"


def test_handle_msg_transcribe_image_does_not_preprocess_image_or_double_charge():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.handle_transcribe_with_message",
        return_value="🖼️ Descripción: todo piola",
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ) as mock_charge:
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 23,
            "chat": {"id": 556, "type": "private"},
            "from": {"id": 100, "first_name": "Ana", "username": "ana"},
            "text": "/transcribe",
            "reply_to_message": {"message_id": 24, "photo": [{"file_id": "img_reply"}]},
        }

        result = handle_msg(message)

    assert result == "ok"
    # /transcribe maneja la imagen dentro de su flujo; no debería predescargar acá
    mock_download.assert_not_called()
    # Debe cobrar solo una vez por el comando /transcribe
    mock_charge.assert_called_once_with(user_id=100, chat_id=None, amount=1)
    mock_send_msg.assert_called_once()


def test_handle_msg_with_unknown_command():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.check_rate_limit"
    ) as mock_rate_limit, patch(
        "api.index.should_gordo_respond"
    ) as mock_should_respond:

        mock_env.side_effect = lambda key: {"TELEGRAM_USERNAME": "testbot"}.get(key)
        mock_rate_limit.return_value = True
        mock_should_respond.return_value = False

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "group"},
            "from": {"first_name": "John", "username": "john"},
            "text": "/unknown_command",
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_send_msg.assert_not_called()  # Should not send message


def test_handle_msg_with_exception():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.admin_report"
    ) as mock_admin_report, patch("os.environ.get") as mock_env:

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


def test_handle_msg_edge_cases():
    from api.index import handle_msg

    with patch("api.index.config_redis") as mock_config_redis, patch(
        "api.index.send_msg"
    ) as mock_send_msg, patch("os.environ.get") as mock_env, patch(
        "api.index.send_typing"
    ) as mock_send_typing, patch(
        "api.index.gen_random"
    ) as mock_gen_random, patch(
        "api.index.cached_requests"
    ) as mock_cached_requests, patch(
        "api.index.admin_report"
    ) as mock_admin_report, patch(
        "api.index.ask_ai"
    ) as mock_ask_ai, patch(
        "api.index.should_gordo_respond"
    ) as mock_should_respond, patch(
        "api.index.check_global_rate_limit", return_value=True
    ), patch(
        "api.index.credits_db_service.is_configured", return_value=True
    ), patch(
        "api.index.credits_db_service.charge_ai_credits",
        return_value={"ok": True, "source": "user"},
    ), patch(
        "api.index._maybe_grant_onboarding_credits"
    ), patch(
        "time.sleep"
    ) as _mock_sleep:  # noqa: F841

        # Set up mocks
        mock_env.return_value = "testbot"
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_gen_random.return_value = "no boludo"
        mock_cached_requests.return_value = None  # Prevent API calls
        mock_ask_ai.return_value = "test response"
        
        def redis_get(key):
            if key == "chat_config:456":
                return json.dumps(CHAT_CONFIG_DEFAULTS)
            if key.startswith("prices:"):
                return json.dumps({"timestamp": 123, "data": {}})
            return None

        mock_redis.get.side_effect = redis_get
        mock_should_respond.return_value = False  # Don't respond by default

        # Reset all mocks before starting tests
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        mock_ask_ai.reset_mock()
        mock_admin_report.reset_mock()

        # Test empty message
        message = {
            "message_id": "123",
            "chat": {"id": "456", "type": "private"},
            "from": {"id": 14, "first_name": "John", "username": "john123"},
        }
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_not_called()

        # Test message with only whitespace
        message["text"] = "   \n   \t   "
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_not_called()

        # Test message that should get a response
        mock_should_respond.return_value = True
        message["text"] = "test"
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"
        mock_send_msg.assert_called_once_with(
            "456", "test response", "123", reply_markup=None
        )

        # Test message with invalid JSON
        mock_redis.get.side_effect = lambda key: "invalid json"
        mock_send_msg.reset_mock()
        assert handle_msg(message) == "ok"
        mock_redis.get.side_effect = redis_get

        # Test message with missing required fields
        mock_admin_report.reset_mock()  # Reset admin report mock
        message = {"message_id": "123"}  # Missing chat and from fields
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "ok"
        mock_admin_report.assert_not_called()

        # Test message with None values
        mock_admin_report.reset_mock()
        message = {
            "message_id": "123",
            "chat": {"id": None},  # Changed to match error handling structure
            "from": {"username": None},  # Changed to match error handling structure
            "text": None,
        }
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "ok"
        mock_admin_report.assert_not_called()

        # Test message with missing message_id
        mock_admin_report.reset_mock()
        message = {"chat": {"id": "456"}, "from": {"first_name": "John"}}
        mock_send_msg.reset_mock()
        result = handle_msg(message)
        assert result == "ok"
        mock_admin_report.assert_not_called()
