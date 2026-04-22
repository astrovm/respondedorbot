
from tests.support import *


def test_handle_msg_topup_private_returns_keyboard():
    from api.message_handler import handle_msg

    message = {
        "message_id": "10",
        "chat": {"id": "100", "type": "private"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "/topup",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_build_keyboard = MagicMock(
        return_value={
            "inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]
        },
    )

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
        build_topup_keyboard=mock_build_keyboard,
        should_gordo_respond=MagicMock(),
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_send_msg.assert_called_once_with(
        "100",
        "elegí cuánto querés cargar:",
        "10",
        reply_markup={
            "inline_keyboard": [[{"text": "pack", "callback_data": "topup:p100"}]]
        },
    )


def test_handle_msg_topup_group_redirects_private(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "10",
        "chat": {"id": "100", "type": "group"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "/topup",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert "@testbot" in mock_send_msg.call_args[0][1]


def test_handle_msg_balance_private_uses_personal_balance(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "11",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/balance",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
        balance_formatter=MagicMock(
            format=MagicMock(
                return_value="tenés 42.0 créditos ia\nsi querés cargar más mandale /topup"
            )
        ),
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert "42.0" in mock_send_msg.call_args[0][1]
    assert "/topup" in mock_send_msg.call_args[0][1]


def test_handle_msg_balance_private_accepts_real_index_formatter(monkeypatch):
    from api.ai_billing import BalanceFormatter
    from api.message_handler import handle_msg

    message = {
        "message_id": "11",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/balance",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.get_balance.return_value = 420
    mock_admin_report = MagicMock()

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
        admin_report=mock_admin_report,
        balance_formatter=BalanceFormatter(mock_credits),
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert "42.0" in mock_send_msg.call_args[0][1]
    mock_admin_report.assert_not_called()


def test_handle_msg_balance_private_uses_balance_formatter_object(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "11",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/balance",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_formatter = MagicMock()
    mock_formatter.format.return_value = "tenés 42.0 créditos ia"

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
        balance_formatter=mock_formatter,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_formatter.format.assert_called_once_with(
        chat_type="private",
        user_id=55,
        chat_id=101,
    )
    assert "42.0" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_moves_credits():
    from api.message_handler import handle_msg

    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 1.5",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_transfer = MagicMock(
        return_value={"ok": True, "user_balance": 285, "chat_balance": 1215}
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.transfer_user_to_chat = mock_transfer

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_transfer.assert_called_once_with(user_id=55, chat_id=202, amount=15)
    assert "le pasé 1.5 créditos al grupo" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_rejects_more_than_one_decimal():
    from api.message_handler import handle_msg

    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 1.55",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_transfer = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.transfer_user_to_chat = mock_transfer

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_transfer.assert_not_called()
    assert "mandalo bien: /transfer <monto>" in mock_send_msg.call_args[0][1]


def test_handle_msg_streamed_response_saves_final_text_to_redis():
    from api.message_handler import handle_msg

    message = {
        "message_id": "10",
        "chat": {"id": "100", "type": "private"},
        "from": {"id": 7, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_save_message = MagicMock()

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        save_message_to_redis=mock_save_message,
        handle_ai_stream=MagicMock(),
    )

    with patch(
        "api.message_handler._run_ai_flow",
        return_value=("__streamed__", True),
    ):
        with patch(
            "api.message_handler._extract_stream_metadata",
            return_value=("777", "hola final"),
        ):
            result = handle_msg(message, deps)

    assert result == "ok"
    mock_send_msg.assert_not_called()
    mock_save_message.assert_any_call(
        "100",
        "bot_777",
        "hola final",
        redis_client,
        role="assistant",
    )


def test_handle_msg_printcredits_requires_admin(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "12b",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/printcredits 100.0",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert "solo para el admin" in mock_send_msg.call_args[0][1]


def test_handle_msg_printcredits_admin_mints_credits(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "12c",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/printcredits 100.0",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_mint = MagicMock(return_value={"user_balance": 1200})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.mint_user_credits = mock_mint

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_mint.assert_called_once_with(user_id=99, amount=1000, actor_user_id=99)
    sent_text = mock_send_msg.call_args[0][1]
    assert "te imprimí 100.0 créditos" in sent_text
    assert "te quedaron 120.0" in sent_text


def test_handle_msg_creditlog_requires_admin(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "12d",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/creditlog",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert "solo para el admin" in mock_send_msg.call_args[0][1]


def test_handle_msg_creditlog_admin_shows_recent_settlements(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "12e",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/creditlog 2",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_creditlog = MagicMock(
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
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.list_recent_ai_settlement_results = mock_creditlog

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

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


def test_handle_msg_creditlog_marks_zero_usage_fallback(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "12f",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/creditlog 1",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_creditlog = MagicMock(
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
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.list_recent_ai_settlement_results = mock_creditlog

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    sent_text = mock_send_msg.call_args[0][1]
    assert "estado=groq_zero_usage" in sent_text


def test_handle_msg_successful_payment_credits_user():
    from api.message_handler import handle_msg

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
    mock_send_msg = MagicMock()
    mock_record = MagicMock(return_value={"inserted": True, "user_balance": 7770})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.record_star_payment = mock_record

    record_called = False

    def fake_handle_payment(msg):
        nonlocal record_called
        record_called = True
        return "ok"

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
        handle_successful_payment_message=fake_handle_payment,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert record_called


def test_handle_msg_refunds_credits_on_internal_ai_fallback(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "14",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send = MagicMock(return_value=999)
    mock_refund = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge
    mock_credits.refund_ai_charge = mock_refund

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["ai_fallback"] = True
        return "no boludo"

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_refund.assert_called_once()
    assert mock_send.call_args[0][1] == "no boludo"


def test_handle_msg_refunds_all_charges_when_fallback_after_multiple_provider_requests(
    monkeypatch,
):
    from api.message_handler import handle_msg

    message = {
        "message_id": "14b",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_refund = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge
    mock_credits.refund_ai_charge = mock_refund

    def fake_handle_ai_response(*args, **kwargs):
        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["provider_request_count"] = 3
            response_meta["ai_fallback"] = True
        return "no boludo"

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=MagicMock(return_value=999),
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        handle_ai_response=fake_handle_ai_response,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_refund.call_count == 1


def test_handle_msg_insufficient_credits_returns_random_plus_topup_hint(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "15",
        "chat": {"id": "405", "type": "private"},
        "from": {"id": 78, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send = MagicMock(return_value=1001)
    mock_charge = MagicMock(
        return_value={
            "ok": False,
            "user_balance_credit_units": 0,
            "chat_balance_credit_units": 0,
        },
    )
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    make_deps, _ = _build_message_handler_deps()
    from api.index import build_insufficient_credits_message as real_build_insuff

    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send,
        check_provider_available=MagicMock(return_value=True),
        credits_db_service=mock_credits,
        gen_random=MagicMock(return_value="no boludo"),
        build_insufficient_credits_message=real_build_insuff,
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert (
        mock_send.call_args[0][1]
        == "no boludo\n\nte quedaste seco de créditos ia, boludo.\nsaldo: 0.0\nmetele /topup si querés que siga laburando"
    )


def test_handle_msg_returns_random_when_ai_billing_backend_is_down(monkeypatch):
    from api.message_handler import handle_msg

    message = {
        "message_id": "15b",
        "chat": {"id": "405", "type": "private"},
        "from": {"id": 78, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send = MagicMock(return_value=1002)
    mock_rate_limit = MagicMock(return_value="no boludo")
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = False

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send,
        handle_rate_limit=mock_rate_limit,
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        ask_ai=MagicMock(),
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    mock_rate_limit.assert_called_once_with("405", message)
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
        mock_env.return_value = "fake_token"

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
    from api.message_handler import handle_msg

    message = {
        "message_id": "rl1",
        "chat": {"id": "404", "type": "private"},
        "from": {"id": 77, "first_name": "Ana", "username": "ana"},
        "text": "/ask hola",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send = MagicMock(return_value=999)
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send,
        check_provider_available=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="tranqui"),
        credits_db_service=mock_credits,
        get_chat_history=MagicMock(return_value=[]),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
    )
    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_send.call_args[0][1] == "tranqui"


def test_handle_msg():
    from api.message_handler import handle_msg

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
    mock_ask_ai = MagicMock(return_value="test response")
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
        ask_ai=mock_ask_ai,
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
    mock_ask_ai.assert_called_once()

    mock_send_msg.reset_mock()
    mock_send_typing.reset_mock()
    mock_ask_ai.reset_mock()
    assert handle_msg(message, deps) == "ok"
    mock_ask_ai.assert_not_called()


def test_handle_msg_with_crypto_command():
    from api.message_handler import handle_msg

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

    with patch("api.index.get_prices", mock_get_prices):
        result = handle_msg(message, deps)
    assert result == "ok"
    mock_get_prices.assert_called_once()
    mock_send_msg.assert_called_once()


def test_handle_msg_with_image(monkeypatch):
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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

    with patch("api.index.ask_ai", return_value="respuesta ok"):
        result = handle_msg(message, deps)
    assert result == "ok"
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    assert mock_charge.call_count == 2
    assert mock_charge.call_args_list[0].kwargs["amount"] == 1
    mock_send_msg.assert_called_once()


def test_handle_msg_auto_audio_rejects_missing_duration(monkeypatch):
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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

    with patch("api.index.ask_ai", return_value="respuesta ok"):
        result = handle_msg(message, deps)

    assert result == "ok"
    mock_download.assert_called_once_with("voice_123")
    mock_measure.assert_called_once_with(b"audio-bytes")
    mock_transcribe.assert_called_once_with("voice_123", use_cache=False)
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_auto_audio_skips_transcription_when_should_not_respond(monkeypatch):
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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
    from api.message_handler import handle_msg

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

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

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


def test_handle_msg_image_conversation_charges_media_and_response_credits(monkeypatch):
    from api.message_handler import handle_msg

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
        handle_ai_response=MagicMock(return_value="todo piola"),
    )
    message = _private_photo_message(message_id=22, chat_id=555, user_id=99)

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_download.called
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_with_two_provider_requests_reserves_base_and_media(
    monkeypatch,
):
    from api.message_handler import handle_msg

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
        handle_ai_response=fake_handle_ai_response,
    )
    message = _private_photo_message(message_id=33, chat_id=555, user_id=991)

    result = handle_msg(message, deps)

    assert result == "ok"
    assert mock_download.called
    assert mock_charge.call_count == 2
    mock_send_msg.assert_called_once()


def test_handle_msg_image_conversation_settles_in_single_batch(monkeypatch):
    from api.message_handler import handle_msg

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

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    with (
        patch(
            "api.message_handler.AIMessageBilling.settle_reserved_ai_credits_batch",
            autospec=True,
        ) as mock_settle_batch,
        patch(
            "api.message_handler.AIMessageBilling.settle_reserved_ai_credits",
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
    from api import config as config_module
    from api.message_handler import handle_msg

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
        handle_ai_response=MagicMock(return_value="respuesta ok"),
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


def test_handle_msg_ai_flow_settles_with_single_base_reserve_when_usage_is_tiny(
    monkeypatch,
):
    from api.message_handler import handle_msg

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

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

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
    from api.message_handler import handle_msg

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_handle_ai_response = MagicMock(return_value="respuesta ok")
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
    mock_send_msg.assert_called_once_with(
        "777", "respuesta ok", "301", reply_markup=None
    )


def test_handle_msg_ai_flow_keeps_single_reserve_for_three_tiny_segments(monkeypatch):
    from api.message_handler import handle_msg

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

    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    redis_client.lrange.return_value = []
    mock_send_msg = MagicMock()
    mock_charge = MagicMock(return_value={"ok": True, "source": "user"})
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.charge_ai_credits = mock_charge

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
    from api.message_handler import handle_msg

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
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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
    from api.ai_billing import AIMessageBilling
    from api.ai_service import build_ai_service
    from api.message_handler import PreparedMessage, _run_ai_flow

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
    from api.ai_billing import AIMessageBilling
    from api.ai_service import build_ai_service
    from api.message_handler import PreparedMessage, _run_ai_flow

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
    from api.command_registry import parse_command as _parse_command
    from api import index as _api_index

    return {
        "config_redis": lambda: redis_client,
        "get_chat_config": lambda _rc, _cid: dict(CHAT_CONFIG_DEFAULTS),
        "initialize_commands": _api_index.initialize_commands,
        "parse_command": _parse_command,
        "should_auto_process_media": _api_index.should_auto_process_media,
        "extract_message_content": _api_index.extract_message_content,
        "replace_links": lambda text: (text, False, []),
        "send_msg": MagicMock(return_value=999),
        "send_animation": MagicMock(return_value=999),
        "delete_msg": MagicMock(),
        "admin_report": MagicMock(),
        "get_bot_message_metadata": MagicMock(return_value=None),
        "save_bot_message_metadata": MagicMock(),
        "build_reply_context_text": MagicMock(return_value=None),
        "build_message_links_context": MagicMock(return_value=""),
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
        "handle_ai_response": _api_index.handle_ai_response,
        "ask_ai": MagicMock(return_value="respuesta ok"),
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


def _build_test_ai_service(flat_defaults):
    from api.ai_service import build_ai_service

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
    from api.message_handler import (
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
            replace_links=flat_defaults["replace_links"],
            should_gordo_respond=flat_defaults["should_gordo_respond"],
            is_group_chat_type=flat_defaults["is_group_chat_type"],
        ),
        io=MessageIODeps(
            send_msg=flat_defaults["send_msg"],
            send_animation=flat_defaults["send_animation"],
            delete_msg=flat_defaults["delete_msg"],
            admin_report=flat_defaults["admin_report"],
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=flat_defaults["get_bot_message_metadata"],
            save_bot_message_metadata=flat_defaults["save_bot_message_metadata"],
            build_reply_context_text=flat_defaults["build_reply_context_text"],
            build_message_links_context=flat_defaults["build_message_links_context"],
            format_user_message=flat_defaults["format_user_message"],
            save_message_to_redis=flat_defaults["save_message_to_redis"],
        ),
        ai=MessageAIDeps(
            ai_service=ai_service,
            balance_formatter=flat_defaults["balance_formatter"],
            ask_ai=flat_defaults["ask_ai"],
            handle_ai_stream=flat_defaults.get("handle_ai_stream", flat_defaults["ask_ai"]),
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


def test_build_message_handler_deps_from_groups_exposes_flat_runtime_contract():
    from api.message_handler import (
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
            replace_links=MagicMock(),
            should_gordo_respond=MagicMock(),
            is_group_chat_type=MagicMock(),
        ),
        io=MessageIODeps(
            send_msg=MagicMock(),
            send_animation=MagicMock(),
            delete_msg=MagicMock(),
            admin_report=MagicMock(),
        ),
        state=MessageStateDeps(
            get_bot_message_metadata=MagicMock(),
            save_bot_message_metadata=MagicMock(),
            build_reply_context_text=MagicMock(),
            build_message_links_context=MagicMock(),
            format_user_message=MagicMock(),
            save_message_to_redis=MagicMock(),
        ),
        ai=MessageAIDeps(
            ai_service=ai_service,
            ask_ai=MagicMock(),
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
    from api.message_handler import _build_message_context

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
    from api.message_handler import (
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
    from api.message_handler import MessageContext, MessageRuntime, PreparedMessage
    from api.message_handler import _resolve_message_intent

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


def test_handle_non_ai_command_summary_builds_valid_prepared_message_and_tuple():
    from api.message_handler import _handle_non_ai_command

    deps = MagicMock()
    commands = {"/resumen": (MagicMock(), False, True)}

    with patch(
        "api.message_handler._run_ai_flow",
        return_value=("resumen listo", True),
    ) as run_ai_flow:
        response = _handle_non_ai_command(
            deps,
            command="/resumen",
            commands=commands,
            sanitized_message_text="focus en crypto",
            message={"message_id": "10"},
            chat_id="123",
            redis_client=MagicMock(),
            billing_helper=MagicMock(),
            user_id=7,
            user_identity="Ana (ana)",
            timezone_offset=-3,
        )

    assert response == ("resumen listo", None, True, "/resumen")
    prepared_message = run_ai_flow.call_args.kwargs["prepared_message"]
    assert (
        prepared_message.message_text
        == "focus en crypto. actualizá el resumen anterior con los mensajes nuevos. incluí todos los temas tratados, quién dijo qué, las conclusiones, las decisiones pendientes y cualquier dato relevante. no seas conciso: sé exhaustivo, detallado y estructurado."
    )
    assert prepared_message.photo_file_id is None
    assert prepared_message.audio_file_id is None


def test_handle_known_command_preserves_ai_flag_from_summary_non_ai_branch():
    from api.message_handler import PreparedMessage, _handle_known_command

    deps = MagicMock()
    commands = {"/resumen": (MagicMock(), False, True)}

    empty_response = (None, None, False, None)
    with patch("api.message_handler._handle_config_command", return_value=empty_response), patch(
        "api.message_handler._handle_topup_command", return_value=empty_response
    ), patch("api.message_handler._handle_balance_command", return_value=empty_response), patch(
        "api.message_handler._handle_transfer_command", return_value=empty_response
    ), patch(
        "api.message_handler._handle_admin_printcredits_command",
        return_value=empty_response,
    ), patch(
        "api.message_handler._handle_admin_creditlog_command",
        return_value=empty_response,
    ), patch(
        "api.message_handler._handle_non_ai_command",
        return_value=("resumen listo", None, True, "/resumen"),
    ):
        response = _handle_known_command(
            deps,
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
            timezone_offset=-3,
        )

    assert response == ("resumen listo", None, True, "/resumen")


def test_message_handler_routes_ai_command_through_known_command_path(monkeypatch):
    from api.message_handler import handle_msg

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    mock_handle_ai_stream = MagicMock(return_value="respuesta ok")

    make_deps, redis_client = _build_message_handler_deps()
    deps = make_deps(
        send_msg=MagicMock(return_value=999),
        handle_ai_stream=mock_handle_ai_stream,
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
    deps.send_msg.assert_called_once_with(
        "555", "respuesta ok", "401", reply_markup=None
    )
    deps.save_bot_message_metadata.assert_called_once_with(
        redis_client,
        "555",
        999,
        {"type": "command", "command": "/ask", "uses_ai": True},
    )


def test_message_handler_ai_command_passes_single_request_object(monkeypatch):
    from api.ai_service import AIConversationRequest
    from api.message_handler import handle_msg

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    ai_service = MagicMock()
    ai_service.run_conversation.return_value = ("respuesta ai", True)

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        ai_service=ai_service,
        send_msg=MagicMock(return_value=999),
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
    deps.send_msg.assert_called_once_with(
        "557", "respuesta ai", "501", reply_markup=None
    )


def test_message_handler_spontaneous_reply_passes_single_request_object(monkeypatch):
    from api.ai_service import AIConversationRequest
    from api.message_handler import handle_msg

    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    ai_service = MagicMock()
    ai_service.run_conversation.return_value = ("respuesta espontanea", True)

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        ai_service=ai_service,
        send_msg=MagicMock(return_value=999),
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
    assert request.handler_func is deps.ask_ai
    assert request.is_spontaneous is True
    deps.send_msg.assert_called_once_with(
        "558", "respuesta espontanea", "502", reply_markup=None
    )


def test_message_handler_stores_user_message_when_bot_should_not_respond(monkeypatch):
    from api.message_handler import handle_msg

    make_deps, redis_client = _build_message_handler_deps()
    deps = make_deps(should_gordo_respond=MagicMock(return_value=False))
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    message = {
        "message_id": 402,
        "chat": {"id": 556, "type": "group"},
        "from": {"id": 1002, "first_name": "Ana", "username": "ana"},
        "text": "hola gordo",
    }

    result = handle_msg(message, deps)

    assert result == "ok"
    deps.send_msg.assert_not_called()
    deps.save_message_to_redis.assert_called_once()
    args, kwargs = deps.save_message_to_redis.call_args
    assert args == ("556", "402", "Ana: hola gordo", redis_client)
    assert kwargs["role"] == "user"
    assert kwargs["user_id"] == "1002"


def test_handle_msg_with_unknown_command():
    from api.message_handler import handle_msg

    mock_config_redis = MagicMock()
    mock_redis = MagicMock()
    mock_config_redis.return_value = mock_redis

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=mock_config_redis,
        send_msg=MagicMock(),
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
    deps.send_msg.assert_not_called()


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


def test_handle_msg_edge_cases(monkeypatch):
    from api.message_handler import handle_msg

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
    mock_ask_ai = MagicMock(return_value="test response")
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
        ask_ai=mock_ask_ai,
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
    mock_send_msg.assert_called_once_with(
        "456", "test response", "123", reply_markup=None
    )

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
