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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.build_topup_keyboard",
            return_value={
                "inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]
            },
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_send_msg.assert_called_once_with(
        "100",
        "elegí cuánto querés cargar:",
        "10",
        reply_markup={
            "inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]
        },
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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"TELEGRAM_USERNAME": "testbot"}.get(
                key, default
            ),
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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index._fetch_balance", return_value=420),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert "42.0" in mock_send_msg.call_args[0][1]
    assert "/topup" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_moves_credits():
    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 1.5",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.transfer_user_to_chat",
            return_value={"ok": True, "user_balance": 285, "chat_balance": 1215},
        ) as mock_transfer,
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_transfer.assert_called_once_with(user_id=55, chat_id=202, amount=15)
    assert "le pasé 1.5 créditos al grupo" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_rejects_more_than_one_decimal():
    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 1.55",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.credits_db_service.transfer_user_to_chat") as mock_transfer,
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_transfer.assert_not_called()
    assert "mandalo bien: /transfer <monto>" in mock_send_msg.call_args[0][1]


def test_handle_msg_printcredits_requires_admin():
    message = {
        "message_id": "12b",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/printcredits 100.0",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"ADMIN_CHAT_ID": "99"}.get(
                key, default
            ),
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert "solo para el admin" in mock_send_msg.call_args[0][1]


def test_handle_msg_printcredits_admin_mints_credits():
    message = {
        "message_id": "12c",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/printcredits 100.0",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.mint_user_credits",
            return_value={"user_balance": 1200},
        ) as mock_mint,
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"ADMIN_CHAT_ID": "99"}.get(
                key, default
            ),
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_mint.assert_called_once_with(user_id=99, amount=1000, actor_user_id=99)
    sent_text = mock_send_msg.call_args[0][1]
    assert "te imprimí 100.0 créditos" in sent_text
    assert "te quedaron 120.0" in sent_text


def test_handle_msg_creditlog_requires_admin():
    message = {
        "message_id": "12d",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/creditlog",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"ADMIN_CHAT_ID": "99"}.get(
                key, default
            ),
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert "solo para el admin" in mock_send_msg.call_args[0][1]


def test_handle_msg_creditlog_admin_shows_recent_settlements():
    message = {
        "message_id": "12e",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/creditlog 2",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.list_recent_ai_settlement_results",
            return_value=[
                {
                    "id": 1,
                    "event_type": "ai_settlement_result",
                    "user_id": 99,
                    "chat_id": 202,
                    "amount": 0,
                    "created_at": "2026-03-11T17:35:10+00:00",
                    "metadata": {
                        "command": "/ask",
                        "reserved_credit_units_total": 20,
                        "settled_credit_units": 10,
                        "refunded_credit_units": 10,
                        "extra_charged_credit_units": 0,
                        "raw_usd_micros": 390,
                        "model_breakdown": [
                            {
                                "model": "qwen/qwen3.6-plus",
                                "usd_micros": 325,
                                "input_tokens": 1000,
                                "input_cached_tokens": 800,
                                "input_non_cached_tokens": 200,
                            },
                            {
                                "model": "qwen/qwen3.6-plus",
                                "usd_micros": 65,
                                "input_tokens": 200,
                                "input_cached_tokens": 100,
                                "input_non_cached_tokens": 100,
                            },
                        ],
                        "tool_breakdown": [
                            {"tool": "web_search", "usd_micros": 8000, "count": 2},
                            {"tool": "python", "usd_micros": 500, "count": 1},
                        ],
                        "billing_segments": [
                            {"kind": "chat"},
                            {"kind": "chat"},
                            {"kind": "chat"},
                            {"kind": "chat", "source": "cache"},
                        ],
                    },
                }
            ],
        ) as mock_creditlog,
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"ADMIN_CHAT_ID": "99"}.get(
                key, default
            ),
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_creditlog.assert_called_once_with(limit=2)
    sent_text = mock_send_msg.call_args[0][1]
    assert "últimas liquidaciones IA" in sent_text
    assert "cmd=/ask" in sent_text
    assert "reservado=2.0 cobrado=1.0 refund=1.0 extra=0.0 deuda=0.0" in sent_text
    assert "requests: chat=3" in sent_text
    assert "cache_hits: chat=1" in sent_text
    assert "cacheados=900 ahorro_cache=0" in sent_text
    assert "qwen/qwen3.6-plus=390" in sent_text
    assert "web_search=8000 (2x)" in sent_text
    assert "python=500 (1x)" in sent_text


def test_handle_msg_creditlog_marks_zero_usage_fallback():
    message = {
        "message_id": "12f",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/creditlog 1",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.list_recent_ai_settlement_results",
            return_value=[
                {
                    "id": 2,
                    "event_type": "ai_settlement_result",
                    "user_id": 99,
                    "chat_id": 202,
                    "amount": 0,
                    "created_at": "2026-03-11T17:35:10+00:00",
                    "metadata": {
                        "command": "/ask",
                        "reserved_credit_units_total": 20,
                        "settled_credit_units": 20,
                        "refunded_credit_units": 0,
                        "extra_charged_credit_units": 0,
                        "raw_usd_micros": 0,
                        "billing_zero_usage_fallback": True,
                        "model_breakdown": [],
                        "tool_breakdown": [],
                    },
                }
            ],
        ),
        patch(
            "os.environ.get",
            side_effect=lambda key, default=None: {"ADMIN_CHAT_ID": "99"}.get(
                key, default
            ),
        ),
    ):
        result = handle_msg(message)

    assert result == "ok"
    sent_text = mock_send_msg.call_args[0][1]
    assert "estado=groq_zero_usage" in sent_text


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

    with (
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.record_star_payment",
            return_value={"inserted": True, "user_balance": 7770},
        ) as mock_record,
        patch("api.index.send_msg") as mock_send_msg,
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_record.assert_called_once()
    assert "777.0" in mock_send_msg.call_args[0][1]


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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index.credits_db_service.refund_ai_charge") as mock_refund,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.ask_ai", return_value="[[AI_FALLBACK]]no boludo"),
        patch("api.index.send_msg", return_value=999) as mock_send,
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index.credits_db_service.refund_ai_charge") as mock_refund,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
        patch("api.index.send_msg", return_value=999),
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert mock_refund.call_count == 1


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

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={
                "ok": False,
                "user_balance_credit_units": 0,
                "chat_balance_credit_units": 0,
            },
        ),
        patch("api.index.gen_random", return_value="no boludo"),
        patch("api.index.send_msg", return_value=1001) as mock_send,
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
        result = handle_msg(message)

    assert result == "ok"
    assert (
        mock_send.call_args[0][1]
        == "no boludo\n\nte quedaste seco de créditos ia, boludo.\nsaldo: 0.0\nmetele /topup si querés que siga laburando"
    )


def test_handle_msg_returns_random_when_ai_billing_backend_is_down(monkeypatch):
    message = {
        "message_id": "15b",
        "chat": {"id": "405", "type": "private"},
        "from": {"id": 78, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.credits_db_service.is_configured", return_value=False),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch(
            "api.index.handle_rate_limit", return_value="no boludo"
        ) as mock_rate_limit,
        patch("api.index.ask_ai") as mock_ask_ai,
        patch("api.index.credits_db_service.charge_ai_credits") as mock_charge,
        patch("api.index.send_msg", return_value=1002) as mock_send,
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_rate_limit.assert_called_once_with("405", message)
    mock_ask_ai.assert_not_called()
    mock_charge.assert_not_called()
    assert mock_send.call_args[0][1] == "no boludo"


def test_handle_rate_limit():
    from api.index import handle_rate_limit

    with (
        patch("api.index.send_typing") as mock_send_typing,
        patch("time.sleep") as mock_sleep,
        patch("api.index.gen_random") as mock_gen_random,
        patch("os.environ.get") as mock_env,
    ):
        chat_id = "123"
        message = {"from": {"first_name": "John"}}
        mock_gen_random.return_value = "no boludo"
        mock_env.return_value = "fake_token"  # Mock TELEGRAM_TOKEN

        response = handle_rate_limit(chat_id, message)

        mock_send_typing.assert_called_once()
        mock_sleep.assert_called_once()
        assert response == "no boludo"


def test_handle_rate_limit_uses_username_when_first_name_is_missing():
    from api.index import handle_rate_limit

    with (
        patch("api.index.send_typing"),
        patch("time.sleep"),
        patch("api.index.gen_random", return_value="no ana_user") as mock_gen_random,
        patch("os.environ.get", return_value="fake_token"),
    ):
        response = handle_rate_limit("123", {"from": {"username": "ana_user"}})

    mock_gen_random.assert_called_once_with("ana_user")
    assert response == "no ana_user"


def test_handle_msg_skips_billing_when_local_rate_limit_hits(monkeypatch):
    from api.index import handle_msg

    message = {
        "message_id": "rl1",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=False),
        patch("api.index.handle_rate_limit", return_value="tranqui"),
        patch("api.index.credits_db_service.charge_ai_credits") as mock_charge,
        patch("api.index.send_msg", return_value=999) as mock_send,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
        result = handle_msg(message)

    assert result == "ok"
    mock_charge.assert_not_called()
    assert mock_send.call_args[0][1] == "tranqui"


def test_handle_msg():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.ask_ai") as mock_ask_ai,
        patch("api.index.send_typing") as mock_send_typing,
        patch(
            "api.index.check_global_rate_limit", side_effect=[True, False]
        ) as mock_global_rate_limit,
        patch("api.index.admin_report"),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("time.sleep") as _mock_sleep,
    ):  # Add sleep mock to avoid delays  # noqa: F841
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot",
            "BOT_SYSTEM_PROMPT": "test prompt",
            "BOT_TRIGGER_WORDS": "bot,assistant,help",
        }.get(key, default)
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
        mock_ask_ai.assert_called_once()

        # Test rate limited message
        mock_send_msg.reset_mock()
        mock_send_typing.reset_mock()
        mock_ask_ai.reset_mock()
        assert handle_msg(message) == "ok"
        mock_ask_ai.assert_not_called()
        assert mock_global_rate_limit.call_count == 2


def test_handle_msg_with_crypto_command():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.check_global_rate_limit") as mock_rate_limit,
        patch("api.index.get_prices") as mock_get_prices,
    ):
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

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.describe_image_groq") as mock_describe,
        patch("api.index.download_telegram_file") as mock_download,
        patch("api.index.resize_image_if_needed") as mock_resize,
        patch("api.index.encode_image_to_base64") as mock_encode,
        patch("api.index.cached_requests") as mock_requests,
        patch("api.index.send_typing") as mock_send_typing,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index._maybe_grant_onboarding_credits"),
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

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index._transcribe_audio_file") as mock_transcribe,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index._maybe_grant_onboarding_credits"),
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot",
            "GROQ_API_KEY": "test_key",
        }.get(key, default)
        mock_transcribe.return_value = (
            "transcribed text",
            None,
            {
                "kind": "transcribe",
                "model": "whisper-large-v3",
                "audio_seconds": 1,
            },
        )

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 12, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123", "duration": 10},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
        mock_send_msg.assert_called_once()


def test_handle_msg_group_audio_without_invocation_skips_auto_transcription():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index._transcribe_audio_file") as mock_transcribe,
    ):

        def env_side_effect(key, default=None):
            env_vars = {
                "TELEGRAM_USERNAME": "testbot",
                "BOT_SYSTEM_PROMPT": "You are a test bot",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key, default)

        mock_env.side_effect = env_side_effect
        mock_transcribe.return_value = ("transcribed text", None, None)

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

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index.handle_transcribe_with_message_result"
        ) as mock_handle_transcribe,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index._maybe_grant_onboarding_credits"),
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)
        mock_handle_transcribe.return_value = (
            "Transcription result",
            [
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                }
            ],
        )

        # Mock Redis instance
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

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

        result = handle_msg(message)
        assert result == "ok"
        mock_handle_transcribe.assert_called_once()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_charges_media_credits():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.check_global_rate_limit") as mock_rate_limit,
        patch(
            "api.index.handle_transcribe_with_message_result"
        ) as mock_handle_transcribe,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.credits_db_service.refund_ai_charge") as mock_refund,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)
        mock_rate_limit.return_value = True
        mock_handle_transcribe.return_value = (
            "🎵 te saqué esto del audio: todo piola",
            [
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                }
            ],
        )

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

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

        result = handle_msg(message)
        assert result == "ok"
        assert mock_charge.call_count == 1
        assert mock_charge.call_args.kwargs["amount"] == 1
        mock_refund.assert_not_called()
        mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_refunds_on_unsuccessful_response():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index.handle_transcribe_with_message_result",
            return_value=("no pude sacar nada de ese audio, probá más tarde", []),
        ),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.credits_db_service.refund_ai_charge") as mock_refund,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

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

        result = handle_msg(message)

    assert result == "ok"
    assert mock_charge.call_count == 1
    assert mock_charge.call_args.kwargs["amount"] == 1
    mock_refund.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_transcribe_command_rejects_audio_without_duration():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index.handle_transcribe_with_message_result"
        ) as mock_handle_transcribe,
        patch("api.index.download_telegram_file", return_value=b"audio-bytes"),
        patch("api.index.measure_audio_duration_seconds", return_value=None),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index.credits_db_service.charge_ai_credits") as mock_charge,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

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
    mock_charge.assert_not_called()
    mock_handle_transcribe.assert_not_called()
    mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_charges_media_credits():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index._transcribe_audio_file",
            return_value=(
                "audio transcripto",
                None,
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 1,
                },
            ),
        ) as mock_transcribe,
        patch("api.index.should_gordo_respond", return_value=False),
        patch("api.index.save_message_to_redis"),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 88, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123", "duration": 10},
        }

        result = handle_msg(message)
        assert result == "ok"
        mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
        assert mock_charge.call_count == 1
        assert mock_charge.call_args.kwargs["amount"] == 1
        mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_rejects_missing_duration():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index._transcribe_audio_file") as mock_transcribe,
        patch("api.index.download_telegram_file", return_value=b"audio-bytes"),
        patch("api.index.measure_audio_duration_seconds", return_value=None),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index.credits_db_service.charge_ai_credits") as mock_charge,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

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
    mock_charge.assert_not_called()
    mock_transcribe.assert_not_called()
    mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_measures_duration_when_missing_in_message():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index._transcribe_audio_file",
            return_value=(
                "audio transcripto",
                None,
                {
                    "kind": "transcribe",
                    "model": "whisper-large-v3",
                    "audio_seconds": 12,
                },
            ),
        ) as mock_transcribe,
        patch(
            "api.index.download_telegram_file", return_value=b"audio-bytes"
        ) as mock_download,
        patch(
            "api.index.measure_audio_duration_seconds", return_value=12.0
        ) as mock_measure,
        patch("api.index.should_gordo_respond", return_value=False),
        patch("api.index.save_message_to_redis"),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

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
    mock_download.assert_called_once_with("voice_123")
    mock_measure.assert_called_once_with(b"audio-bytes")
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    assert mock_charge.call_count == 1
    mock_send_msg.assert_not_called()


def test_handle_msg_auto_audio_plus_ai_response_charges_three_requests():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                }
            ]
        return "respuesta final"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch(
            "api.index._transcribe_audio_file",
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
        patch("api.index.should_gordo_respond", return_value=True),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
        patch("api.index.save_message_to_redis"),
    ):
        mock_env.side_effect = lambda key, default=None: {
            "TELEGRAM_USERNAME": "testbot"
        }.get(key, default)

        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        mock_redis.lrange.return_value = []

        message = {
            "message_id": 31,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 89, "first_name": "John", "username": "john"},
            "voice": {"file_id": "voice_123", "duration": 10},
        }

        result = handle_msg(message)

    assert result == "ok"
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_charges_media_and_response_credits():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch(
            "api.index.download_telegram_file", return_value=b"img-bytes"
        ) as mock_download,
        patch("api.index.resize_image_if_needed", return_value=b"img-resized"),
        patch("api.index.encode_image_to_base64", return_value="abc"),
        patch("api.index.should_gordo_respond", return_value=True),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", return_value="todo piola"),
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


def test_handle_msg_image_conversation_with_two_provider_requests_reserves_base_and_media():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "vision",
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
            ]
        return "todo piola x2"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch(
            "api.index.download_telegram_file", return_value=b"img-bytes"
        ) as mock_download,
        patch("api.index.resize_image_if_needed", return_value=b"img-resized"),
        patch("api.index.encode_image_to_base64", return_value="abc"),
        patch("api.index.should_gordo_respond", return_value=True),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
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
    # El usage real mockeado es mínimo, así que quedan solo las dos reservas iniciales.
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_settles_in_single_batch():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "vision",
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ]
        return "todo piola"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.download_telegram_file", return_value=b"img-bytes"),
        patch("api.index.resize_image_if_needed", return_value=b"img-resized"),
        patch("api.index.encode_image_to_base64", return_value="abc"),
        patch("api.index.should_gordo_respond", return_value=True),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
        patch(
            "api.message_handler.AIMessageBilling.settle_reserved_ai_credits_batch",
            autospec=True,
        ) as mock_settle_batch,
        patch(
            "api.message_handler.AIMessageBilling.settle_reserved_ai_credits",
            autospec=True,
        ) as mock_settle_single,
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        message = {
            "message_id": 44,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 992, "first_name": "Ana", "username": "ana"},
            "photo": [{"file_id": "img1"}],
        }

        result = handle_msg(message)

    assert result == "ok"
    mock_settle_batch.assert_called_once()
    reservations = mock_settle_batch.call_args.args[1]
    assert len(reservations) == 2
    mock_settle_single.assert_not_called()
    mock_send_msg.assert_called_once()


def test_handle_msg_search_command_uses_ai_billing():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ]
        return "resultado web"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.charge_ai_credits") as mock_charge,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "btc news"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
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
    mock_charge.assert_called_once()
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args[0][1] == "resultado web"


def test_handle_msg_search_command_uses_agent_reserve_mode():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "btc news"}],
        ),
        patch(
            "api.index.estimate_ai_base_reserve_credits",
            return_value=(10, {"reserve_mode": "agent", "rate_limit_scope": "chat"}),
        ) as mock_estimate,
        patch("api.index.handle_ai_response", return_value="resultado web"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        result = handle_msg(
            {
                "message_id": 99,
                "chat": {"id": 555, "type": "private"},
                "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
                "text": "/buscar btc news",
            }
        )

    assert result == "ok"
    assert mock_estimate.call_args.kwargs["reserve_mode"] == "agent"
    mock_send_msg.assert_called_once()


def test_handle_msg_ask_command_uses_agent_reserve_mode():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch(
            "api.index.estimate_ai_base_reserve_credits",
            return_value=(10, {"reserve_mode": "agent", "rate_limit_scope": "chat"}),
        ) as mock_estimate,
        patch("api.index.handle_ai_response", return_value="respuesta ok"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
        redis_client.lrange.return_value = []
        mock_config_redis.return_value = redis_client

        result = handle_msg(
            {
                "message_id": 199,
                "chat": {"id": 555, "type": "private"},
                "from": {"id": 2001, "first_name": "Ana", "username": "ana"},
                "text": "/ask hola",
            }
        )

    assert result == "ok"
    assert mock_estimate.call_args.kwargs["reserve_mode"] == "agent"
    mock_send_msg.assert_called_once()


def test_handle_msg_command_reply_to_link_fix_message_is_not_blocked(monkeypatch):
    from api import config as config_module
    from api.index import handle_msg

    config_module.reset_cache()
    chat_config = {
        **CHAT_CONFIG_DEFAULTS,
        "ai_command_followups": False,
        "ignore_link_fix_followups": True,
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(chat_config)
    redis_client.lrange.return_value = []

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setenv("BOT_SYSTEM_PROMPT", "You are a test bot")
    monkeypatch.setenv("BOT_TRIGGER_WORDS", "gordo,test,bot")

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.get_bot_message_metadata", return_value=None),
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "investigá eso"}],
        ),
        patch(
            "api.index.handle_ai_response", return_value="respuesta ok"
        ) as mock_handle_ai_response,
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
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

        result = handle_msg(message)

    assert result == "ok"
    mock_charge.assert_called_once()
    mock_handle_ai_response.assert_called_once()
    mock_send_msg.assert_called_once()
    assert mock_send_msg.call_args[0][1] == "respuesta ok"


def test_handle_msg_ai_flow_settles_with_single_base_reserve_when_usage_is_tiny():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
            ]
        return "respuesta ok"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
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
    assert mock_charge.call_count == 1
    assert mock_send_msg.call_args[0][1] == "respuesta ok"


def test_handle_msg_ai_flow_allows_openrouter_fallback_when_groq_rate_limit_blocks():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=False),
        patch("api.index.should_allow_openrouter_fallback", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch(
            "api.index.handle_ai_response", return_value="respuesta ok"
        ) as mock_handle_ai_response,
        patch("api.index.save_message_to_redis"),
        patch("api.index.save_bot_message_metadata"),
    ):
        mock_config_redis.return_value = MagicMock()
        mock_config_redis.return_value.get.return_value = json.dumps(
            CHAT_CONFIG_DEFAULTS
        )
        message = {
            "message_id": 301,
            "chat": {"id": 777, "type": "group"},
            "from": {"id": 1001, "first_name": "Ana", "username": "ana"},
            "text": "/ask hola",
        }

        result = handle_msg(message)

    assert result == "ok"
    mock_handle_ai_response.assert_called_once()
    mock_send_msg.assert_called_once_with(
        "777", "respuesta ok", "301", reply_markup=None
    )


def test_handle_msg_ai_flow_keeps_single_reserve_for_three_tiny_segments():
    from api.index import handle_msg

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["billing_segments"] = [
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
                {
                    "kind": "chat",
                    "model": "qwen/qwen3.6-plus",
                    "usage": {
                        "input_tokens": 1,
                        "input_non_cached_tokens": 1,
                        "output_tokens": 1,
                    },
                },
            ]
        return "respuesta ok x3"

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
        patch("api.index.get_chat_history", return_value=[]),
        patch(
            "api.index.build_ai_messages",
            return_value=[{"role": "user", "content": "hola"}],
        ),
        patch("api.index.handle_ai_response", side_effect=fake_handle_ai_response),
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
    assert mock_charge.call_count == 1
    assert mock_send_msg.call_args[0][1] == "respuesta ok x3"


def test_handle_msg_transcribe_image_does_not_preprocess_image_or_double_charge():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.download_telegram_file") as mock_download,
        patch(
            "api.index.handle_transcribe_with_message_result",
            return_value=(
                "🖼️ en la imagen veo: todo piola",
                [
                    {
                        "kind": "vision",
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ],
            ),
        ),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ) as mock_charge,
    ):
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
    # /transcribe ahora predescarga la imagen para estimar una reserva precisa
    mock_download.assert_called_once_with("img_reply")
    # Debe cobrar solo una vez por el comando /transcribe
    assert mock_charge.call_count == 1
    assert mock_charge.call_args.kwargs["amount"] == 1
    mock_send_msg.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_vision():
    from api.ai_billing import AIMessageBilling
    from api.message_handler import PreparedMessage, _run_ai_flow

    deps = MagicMock()
    deps.get_chat_history.return_value = []
    deps.build_ai_messages.return_value = [{"role": "user", "content": "hola"}]
    deps.estimate_ai_base_reserve_credits.return_value = (
        1,
        {"rate_limit_scope": "chat", "estimated_rate_limit_tokens": 1},
    )
    deps.check_global_rate_limit.return_value = False
    deps.should_allow_openrouter_fallback.return_value = True
    deps.estimate_image_context_rate_limit_tokens.return_value = 1
    deps.estimate_image_context_reserve_credits.return_value = 1
    deps.handle_ai_response.return_value = "respuesta ok"

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
        command="/ask",
        message={"chat": {"id": 557, "type": "private"}},
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
    deps.handle_ai_response.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_transcribe():
    from api.ai_billing import AIMessageBilling
    from api.message_handler import PreparedMessage, _run_ai_flow

    deps = MagicMock()
    deps.get_chat_history.return_value = []
    deps.build_ai_messages.return_value = [{"role": "user", "content": "hola"}]
    deps.estimate_ai_base_reserve_credits.return_value = (
        1,
        {"rate_limit_scope": "chat", "estimated_rate_limit_tokens": 1},
    )
    deps.check_global_rate_limit.return_value = False
    deps.should_allow_openrouter_fallback.return_value = True
    deps.estimate_image_context_rate_limit_tokens.return_value = 1
    deps.estimate_image_context_reserve_credits.return_value = 1
    deps.handle_ai_response.return_value = "🖼️ en la imagen veo: todo piola"

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
        command="/transcribe",
        message={"chat": {"id": 558, "type": "private"}},
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
    deps.handle_ai_response.assert_called_once()


def test_handle_msg_with_unknown_command():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.check_global_rate_limit") as mock_rate_limit,
        patch("api.index.should_gordo_respond") as mock_should_respond,
    ):
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

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.admin_report") as mock_admin_report,
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


def test_handle_msg_edge_cases():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("os.environ.get") as mock_env,
        patch("api.index.send_typing") as mock_send_typing,
        patch("api.index.gen_random") as mock_gen_random,
        patch("api.index.cached_requests") as mock_cached_requests,
        patch("api.index.admin_report") as mock_admin_report,
        patch("api.index.ask_ai") as mock_ask_ai,
        patch("api.index.should_gordo_respond") as mock_should_respond,
        patch("api.index.check_global_rate_limit", return_value=True),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch(
            "api.index.credits_db_service.charge_ai_credits",
            return_value={"ok": True, "source": "user"},
        ),
        patch("api.index._maybe_grant_onboarding_credits"),
        patch("time.sleep") as _mock_sleep,
    ):  # noqa: F841
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
