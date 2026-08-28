from tests.support import *
from tests.message_handler_support import _build_message_handler_deps


def test_handle_msg_topup_private_returns_keyboard():
    from api.bot.message_handler import handle_msg

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


def test_pre_checkout_uses_invoice_language_without_loading_config():
    from api.billing.service import BillingService

    credits = MagicMock()
    credits.is_configured.return_value = True
    answer_pre_checkout = MagicMock()
    service = BillingService(
        credits=credits,
        admin_report=MagicMock(),
        telegram=MagicMock(),
        telegram_request=MagicMock(),
        guard_callback=MagicMock(),
        answer_callback=MagicMock(),
        answer_pre_checkout=answer_pre_checkout,
        extract_user_id=MagicMock(),
    )
    query = {
        "id": "pcq-1",
        "from": {"id": 42, "language_code": "es"},
        "invoice_payload": "topup:invalid:42:en",
        "currency": "XTR",
        "total_amount": 1,
    }

    service.handle_pre_checkout(query)

    answer_pre_checkout.assert_called_once_with(
        "pcq-1",
        ok=False,
        error_message="I could not validate this payment",
    )


def test_pre_checkout_old_invoice_uses_telegram_language():
    from api.billing.service import BillingService

    credits = MagicMock()
    credits.is_configured.return_value = True
    answer_pre_checkout = MagicMock()
    service = BillingService(
        credits=credits,
        admin_report=MagicMock(),
        telegram=MagicMock(),
        telegram_request=MagicMock(),
        guard_callback=MagicMock(),
        answer_callback=MagicMock(),
        answer_pre_checkout=answer_pre_checkout,
        extract_user_id=MagicMock(),
    )

    service.handle_pre_checkout(
        {
            "id": "pcq-old",
            "from": {"id": 42, "language_code": "en"},
            "invoice_payload": "invalid",
            "currency": "XTR",
            "total_amount": 1,
        }
    )

    answer_pre_checkout.assert_called_once_with(
        "pcq-old",
        ok=False,
        error_message="I could not validate this payment",
    )


def test_invoice_payload_carries_current_locale():
    from api.billing.callbacks import send_stars_invoice
    from api.i18n import use_locale

    telegram_request = MagicMock(return_value=({"ok": True}, None))

    with use_locale("en"):
        sent = send_stars_invoice(
            chat_id="42",
            user_id=42,
            pack={"id": "p50", "credits": 5_000, "xtr": 25},
            format_credits=lambda amount: f"{amount / 100:.2f}",
            telegram_request=telegram_request,
        )

    assert sent
    assert telegram_request.call_args.kwargs["json_payload"]["payload"] == "topup:p50:42:en"


def test_handle_msg_topup_group_redirects_private(monkeypatch):
    from api.bot.message_handler import handle_msg

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
    from api.bot.message_handler import handle_msg

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


def test_handle_msg_charges_lists_only_calling_users_itemized_history():
    from api.bot.message_handler import handle_msg

    message = {
        "message_id": "11c",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/charges 2",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.list_user_ai_charge_page.return_value = {
        "groups": [
            {
                "cursor_id": 30,
                "created_at": "2026-08-26T17:32:00+00:00",
                "entries": [{
                    "id": 30,
                    "event_type": "ai_settlement_result",
                    "metadata": {
                        "charged_credit_units_total": 8,
                        "payer_scope": "user",
                        "model_breakdown": [
                            {"kind": "chat", "usd_micros": 30},
                        ],
                        "tool_breakdown": [
                            {"tool": "web_search", "count": 1, "usd_micros": 50},
                        ],
                    },
                }],
            },
            {
                "cursor_id": 29,
                "created_at": "2026-08-26T16:00:00+00:00",
                "entries": [{
                    "id": 29,
                    "event_type": "ai_settlement_result",
                    "metadata": {
                        "usage_tag": "auto_audio_media",
                        "charged_credit_units_total": 7,
                        "payer_scope": "chat",
                    },
                }],
            },
        ],
        "has_newer": False,
        "has_older": True,
        "newer_cursor": 30,
        "older_cursor": 29,
    }

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=mock_send_msg,
        credits_db_service=mock_credits,
    )

    assert handle_msg(message, deps) == "ok"

    mock_credits.list_user_ai_charge_page.assert_called_once_with(
        55,
        limit=2,
        cursor_id=None,
        direction="older",
    )
    sent_text = mock_send_msg.call_args[0][1]
    assert sent_text == (
        "Gastos IA\n\n"
        "26/08 14:32 · 0.08 cr\n"
        "  respuesta 0.03 cr\n"
        "  web 0.05 cr\n\n"
        "26/08 13:00 · audio · 0.07 cr · grupo"
    )
    assert mock_send_msg.call_args.kwargs["reply_markup"] == {
        "inline_keyboard": [[{
            "text": "Siguiente ›",
            "callback_data": "chg:55:2:o:29:-180",
        }]]
    }


def test_charge_component_allocation_preserves_rounded_total():
    from api.billing.commands import allocate_charge_components

    allocation = allocate_charge_components(
        1,
        [("respuesta IA", 1), ("búsqueda web", 1), ("transcripción", 1)],
    )

    assert allocation == [
        ("respuesta IA", 1),
        ("búsqueda web", 0),
        ("transcripción", 0),
    ]
    assert sum(units for _label, units in allocation) == 1


def test_charge_history_includes_compaction_pending_reserves_and_split_payers():
    from api.billing.commands import format_user_charge_history

    text = format_user_charge_history(
        [
            {
                "cursor_id": 12,
                "created_at": "2026-08-26T17:00:00+00:00",
                "entries": [
                    {
                        "id": 12,
                        "event_type": "memory_compaction_settlement",
                        "metadata": {
                            "usage_tag": "memory_compaction:1:m1",
                            "actual_credit_units": 2,
                            "source": "user",
                        },
                    },
                    {
                        "id": 11,
                        "event_type": "ai_reserve",
                        "metadata": {
                            "usage_tag": "memory_compaction:1:m2",
                            "charged_credit_units_total": 107,
                            "billing_pending": True,
                            "payer_breakdown": [
                                {"scope": "chat", "credit_units": 100},
                                {"scope": "user", "credit_units": 7},
                            ],
                        },
                    },
                    {
                        "id": 10,
                        "event_type": "ai_settlement_result",
                        "metadata": {
                            "charged_credit_units_total": 8,
                            "payer_breakdown": [
                                {"scope": "user", "credit_units": 3},
                                {"scope": "chat", "credit_units": 5},
                            ],
                        },
                    },
                ],
            },
        ]
    )

    assert text == (
        "Gastos IA\n\n"
        "26/08 17:00 · 1.17 cr · grupo 1.05 · personal 0.12\n"
        "  respuesta 0.08 cr\n"
        "  memoria 0.02 cr\n"
        "  memoria 1.07 cr · pendiente"
    )


def test_charge_history_callback_edits_page_for_requesting_user():
    from api.billing.callbacks import handle_charge_history_callback

    credits = MagicMock()
    credits.list_user_ai_charge_page.return_value = {
        "groups": [
            {
                "cursor_id": 20,
                "created_at": "2026-08-26T17:00:00+00:00",
                "entries": [
                    {
                        "id": 20,
                        "event_type": "ai_settlement_result",
                        "metadata": {"charged_credit_units_total": 4},
                    }
                ],
            }
        ],
        "has_newer": True,
        "has_older": False,
        "newer_cursor": 20,
        "older_cursor": 20,
    }
    edit_message = MagicMock(return_value=True)
    answer_callback = MagicMock()

    handle_charge_history_callback(
        {
            "id": "cb-1",
            "data": "chg:55:2:o:29:-180",
            "from": {"id": 55},
            "message": {"message_id": 8, "chat": {"id": 101}},
        },
        credits_db_service=credits,
        edit_message=edit_message,
        answer_callback=answer_callback,
        admin_report=MagicMock(),
    )

    credits.list_user_ai_charge_page.assert_called_once_with(
        55,
        limit=2,
        cursor_id=29,
        direction="older",
    )
    edit_message.assert_called_once_with(
        "101",
        8,
        "Gastos IA\n\n26/08 14:00 · respuesta · 0.04 cr",
        {
            "inline_keyboard": [[{
                "text": "‹ Anterior",
                "callback_data": "chg:55:2:n:20:-180",
            }]]
        },
    )
    answer_callback.assert_called_once_with("cb-1")


def test_charge_history_callback_rejects_other_users():
    from api.billing.callbacks import handle_charge_history_callback

    credits = MagicMock()
    edit_message = MagicMock()
    answer_callback = MagicMock()

    handle_charge_history_callback(
        {
            "id": "cb-2",
            "data": "chg:55:10:o:29:-180",
            "from": {"id": 99},
            "message": {"message_id": 8, "chat": {"id": -100}},
        },
        credits_db_service=credits,
        edit_message=edit_message,
        answer_callback=answer_callback,
        admin_report=MagicMock(),
    )

    credits.list_user_ai_charge_page.assert_not_called()
    edit_message.assert_not_called()
    answer_callback.assert_called_once_with(
        "cb-2",
        text="este historial no es tuyo",
        show_alert=True,
    )


def test_handle_msg_charges_rejects_public_cursor_argument():
    from api.bot.message_handler import handle_msg

    message = {
        "message_id": "11d",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana"},
        "text": "/charges 10 15364",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    send_msg = MagicMock()
    credits = MagicMock()
    credits.is_configured.return_value = True
    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(
        config_redis=lambda: redis_client,
        send_msg=send_msg,
        credits_db_service=credits,
    )

    assert handle_msg(message, deps) == "ok"

    assert send_msg.call_args[0][1] == "mandalo bien: /charges [cantidad]"
    credits.list_user_ai_charge_page.assert_not_called()


def test_handle_msg_balance_private_accepts_real_index_formatter(monkeypatch):
    from api.billing.ai import BalanceFormatter
    from api.bot.message_handler import handle_msg

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
    mock_credits.get_balance.return_value = 4200
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
    assert "42.00" in mock_send_msg.call_args[0][1]
    mock_admin_report.assert_not_called()


def test_handle_msg_balance_private_uses_balance_formatter_object(monkeypatch):
    from api.bot.message_handler import handle_msg

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


def test_handle_msg_transfer_group_moves_fractional_credits():
    from api.bot.message_handler import handle_msg

    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 0.1",
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
    mock_transfer.assert_called_once_with(
        user_id=55,
        chat_id=202,
        amount=10,
    )
    assert "le pasé 0.10 créditos al grupo" in mock_send_msg.call_args[0][1]


def test_handle_msg_transfer_group_rejects_more_than_two_decimals():
    from api.bot.message_handler import handle_msg

    message = {
        "message_id": "12",
        "chat": {"id": "202", "type": "group"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/transfer 1.555",
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


def test_billing_commands_balance_and_transfer_preserve_responses():
    billing_commands = __import__("api.billing.commands", fromlist=["billing_commands"])
    handle_balance_command = billing_commands.handle_balance_command
    handle_transfer_command = billing_commands.handle_transfer_command

    deps = MagicMock()
    deps.credits_db_service.is_configured.return_value = True
    deps.balance_formatter.format.return_value = "saldo listo"
    deps.is_group_chat_type.return_value = True
    deps.credits_db_service.transfer_user_to_chat.return_value = {
        "ok": False,
        "user_balance": 70,
    }

    balance = handle_balance_command(
        deps,
        command="/balance",
        chat_type="private",
        chat_id="101",
        user_id=55,
        numeric_chat_id=101,
    )
    transfer = handle_transfer_command(
        deps,
        command="/transfer",
        sanitized_message_text="1.5",
        chat_id="202",
        chat_type="group",
        user_id=55,
        numeric_chat_id=202,
    )

    assert balance == ("saldo listo", None, False, "/balance")
    deps.maybe_grant_onboarding_credits.assert_called_once_with(
        deps.credits_db_service,
        deps.admin_report,
        55,
    )
    deps.balance_formatter.format.assert_called_once_with(
        chat_type="private",
        user_id=55,
        chat_id=101,
    )
    assert transfer == (
        "no te alcanza lo tuyo para pasar esa guita al grupo\nte quedan: 0.70",
        None,
        False,
        "/transfer",
    )
    deps.credits_db_service.transfer_user_to_chat.assert_called_once_with(
        user_id=55,
        chat_id=202,
        amount=150,
    )


def test_handle_msg_streamed_response_saves_final_text_to_redis():
    from api.bot.message_handler import handle_msg

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
        "api.bot.message_handler._run_ai_flow",
        return_value=("hola final", True),
    ):
        with patch(
            "api.bot.message_handler.extract_stream_metadata",
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
    from api.bot.message_handler import handle_msg

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
    from api.bot.message_handler import handle_msg

    message = {
        "message_id": "12c",
        "chat": {"id": "202", "type": "private"},
        "from": {"id": 99, "first_name": "Admin", "username": "boss"},
        "text": "/printcredits 100.0",
    }
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(CHAT_CONFIG_DEFAULTS)
    mock_send_msg = MagicMock()
    mock_mint = MagicMock(return_value={"user_balance": 12000})
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
    mock_mint.assert_called_once_with(user_id=99, amount=10000, actor_user_id=99)
    sent_text = mock_send_msg.call_args[0][1]
    assert "te imprimí 100.00 créditos" in sent_text
    assert "te quedaron 120.00" in sent_text


def test_handle_msg_creditlog_requires_admin(monkeypatch):
    from api.bot.message_handler import handle_msg

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
    from api.bot.message_handler import handle_msg

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
                    "reserved_credit_units_total": 200,
                    "settled_credit_units": 100,
                    "refunded_credit_units": 100,
                    "extra_charged_credit_units": 0,
                    "raw_usd_micros": 390,
                    "model_breakdown": [
                        {
                            "model": "~deepseek/deepseek-v4-flash-latest",
                            "usd_micros": 325,
                            "input_tokens": 1000,
                            "input_cached_tokens": 800,
                            "input_non_cached_tokens": 200,
                        },
                        {
                            "model": "~deepseek/deepseek-v4-flash-latest",
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
    assert "reservado=2.00 cobrado=1.00 refund=1.00 extra=0.00 deuda=0.00" in sent_text
    assert "requests: chat=3" in sent_text
    assert "cache_hits: chat=1" in sent_text
    assert "cacheados=900 ahorro_cache=24" in sent_text
    assert "~deepseek/deepseek-v4-flash-latest=390" in sent_text
    assert "web_search=8000 (2x)" in sent_text
    assert "python=500 (1x)" in sent_text


def test_handle_msg_creditlog_marks_zero_usage_fallback(monkeypatch):
    from api.bot.message_handler import handle_msg

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
                    "reserved_credit_units_total": 200,
                    "settled_credit_units": 200,
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


def test_admin_commands_printcredits_and_creditlog_preserve_outputs(monkeypatch):
    admin_commands = __import__("api.admin.commands", fromlist=["admin_commands"])
    handle_admin_creditlog_command = admin_commands.handle_admin_creditlog_command
    handle_admin_printcredits_command = admin_commands.handle_admin_printcredits_command

    monkeypatch.setenv("ADMIN_CHAT_ID", "99")
    deps = MagicMock()
    deps.credits_db_service.is_configured.return_value = True
    deps.credits_db_service.mint_user_credits.return_value = {"user_balance": 12000}
    deps.credits_db_service.list_recent_ai_settlement_results.return_value = [
        {
            "created_at": "2026-03-11T17:35:10+00:00",
            "chat_id": 202,
            "user_id": 99,
            "metadata": {
                "command": "/ask",
                "reserved_credit_units_total": 200,
                "settled_credit_units": 100,
                "refunded_credit_units": 100,
                "model_breakdown": [{"model": "m1", "usd_micros": 5}],
                "tool_breakdown": [{"tool": "web", "usd_micros": 7, "count": 2}],
                "billing_segments": [{"kind": "chat"}, {"kind": "vision"}],
            },
        }
    ]

    printed = handle_admin_printcredits_command(
        deps,
        command="/printcredits",
        sanitized_message_text="100.0",
        chat_id="202",
        user_id=99,
    )
    logged = handle_admin_creditlog_command(
        deps,
        command="/creditlog",
        sanitized_message_text="1",
        chat_id="202",
        user_id=99,
    )

    assert printed == (
        "listo, te imprimí 100.00 créditos\nte quedaron 120.00",
        None,
        False,
        "/printcredits",
    )
    deps.credits_db_service.mint_user_credits.assert_called_once_with(
        user_id=99,
        amount=10000,
        actor_user_id=99,
    )
    assert logged[1:] == (None, False, "/creditlog")
    assert "últimas liquidaciones IA" in logged[0]
    assert "cmd=/ask" in logged[0]
    assert "requests: chat=1, vision=1" in logged[0]
    assert "modelos: m1=5" in logged[0]
    assert "tools: web=7 (2x)" in logged[0]


def test_handle_msg_successful_payment_credits_user():
    from api.bot.message_handler import handle_msg

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
    from api.bot.message_handler import handle_msg

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
        from api.bot.streaming import set_streamed_response_metadata

        response_meta = kwargs.get("response_meta")
        if isinstance(response_meta, dict):
            response_meta["ai_fallback"] = True
        sent_message_id = mock_send(str(args[0]), "no boludo")
        set_streamed_response_metadata(
            str(sent_message_id) if sent_message_id is not None else None,
            "no boludo",
        )
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
    from api.bot.message_handler import handle_msg

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
    from api.bot.message_handler import handle_msg

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
    real_build_insuff = index.app_runtime.billing.build_insufficient_message

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
        == "no boludo\n\nte quedaste seco de créditos ia, boludo.\nsaldo: 0.00\nmetele /topup si querés que siga laburando"
    )


def test_handle_msg_returns_random_when_ai_billing_backend_is_down(monkeypatch):
    from api.bot.message_handler import handle_msg

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
    handle_rate_limit = index.app_runtime.responses.handle_rate_limit

    with (
        patch("api.index.app_runtime.responses._deps.send_typing") as mock_send_typing,
        patch("api.index.app_runtime.responses._deps.sleep") as mock_sleep,
        patch("api.index.app_runtime.responses._deps.gen_random") as mock_gen_random,
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
    handle_rate_limit = index.app_runtime.responses.handle_rate_limit

    with (
        patch("api.index.app_runtime.responses._deps.send_typing"),
        patch("api.index.app_runtime.responses._deps.sleep"),
        patch("api.index.app_runtime.responses._deps.gen_random", return_value="no ana_user") as mock_gen_random,
        patch("os.environ.get", return_value="fake_token"),
    ):
        response = handle_rate_limit("123", {"from": {"username": "ana_user"}})

        mock_gen_random.assert_called_once_with("ana_user")
        assert response == "no ana_user"


def test_handle_msg_skips_billing_when_local_rate_limit_hits(monkeypatch):
    from api.bot.message_handler import handle_msg

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
