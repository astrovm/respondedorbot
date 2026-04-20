from tests.support import *  # noqa: F401,F403


def test_decode_redis_value_variants():
    assert _decode_redis_value(b"value") == "value"
    assert _decode_redis_value("value") == "value"
    assert _decode_redis_value(None) is None


def test_get_chat_config_migrates_bytes_from_redis_when_postgres_misses():
    redis_client = MagicMock(spec=redis.Redis)
    stored_config = {"link_mode": "off"}
    redis_client.get.return_value = json.dumps(stored_config).encode("utf-8")

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch("api.index.chat_config_db_service.get_chat_config", return_value=None),
        patch("api.index.chat_config_db_service.set_chat_config") as mock_set,
    ):
        config = get_chat_config(redis_client, "chat-1")

    assert config["link_mode"] == "off"
    mock_set.assert_called_once_with("chat-1", config)


def test_get_chat_config_migrates_string_from_redis_when_postgres_misses():
    redis_client = MagicMock(spec=redis.Redis)
    stored_config = {"link_mode": "off"}
    redis_client.get.return_value = json.dumps(stored_config)

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch("api.index.chat_config_db_service.get_chat_config", return_value=None),
        patch("api.index.chat_config_db_service.set_chat_config") as mock_set,
    ):
        config = get_chat_config(redis_client, "chat-2")

    assert config["link_mode"] == "off"
    mock_set.assert_called_once_with("chat-2", config)


def test_get_chat_config_uses_postgres_when_available():
    redis_client = MagicMock(spec=redis.Redis)
    pg_config = {
        "link_mode": "reply",
        "ai_random_replies": False,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch(
            "api.index.chat_config_db_service.get_chat_config", return_value=pg_config
        ) as mock_get,
    ):
        config = get_chat_config(redis_client, "chat-4")

    assert config == pg_config
    mock_get.assert_called_once_with("chat-4", CHAT_CONFIG_DEFAULTS)
    redis_client.get.assert_not_called()


def test_get_chat_config_postgres_error_does_not_fallback_to_redis():
    redis_client = MagicMock(spec=redis.Redis)

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch(
            "api.index.chat_config_db_service.get_chat_config",
            side_effect=RuntimeError("pg down"),
        ),
        patch("api.index.admin_report") as mock_admin,
    ):
        config = get_chat_config(redis_client, "chat-5")

    assert config == CHAT_CONFIG_DEFAULTS
    redis_client.get.assert_not_called()
    mock_admin.assert_called_once()


def test_build_config_text_clarifies_group_free_ai_limit_is_messages():
    text = build_config_text(CHAT_CONFIG_DEFAULTS)

    assert "2. seguir charla en comandos" in text
    assert "después de un comando sigo la conversación si me respondés" in text
    assert (
        "ignoro respuestas normales a mensajes automáticos con links arreglados" in text
    )
    assert "5. respuestas random" in text
    assert "a veces respondo solo en el grupo aunque nadie me llame" in text
    assert "6. mensajes gratis por usuario por hora" in text
    assert "cuantos mensajes de ia paga el grupo por usuario cada hora" in text
    assert "\n5\n" in text
    assert "tocá los botones de abajo para cambiar la config" in text


def test_get_chat_config_migration_error_returns_redis_config_and_reports():
    redis_client = MagicMock(spec=redis.Redis)
    redis_client.get.return_value = json.dumps({"link_mode": "off"})

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch("api.index.chat_config_db_service.get_chat_config", return_value=None),
        patch(
            "api.index.chat_config_db_service.set_chat_config",
            side_effect=RuntimeError("pg write failed"),
        ),
        patch("api.index.admin_report") as mock_admin,
    ):
        config = get_chat_config(redis_client, "chat-6")

    assert config["link_mode"] == "off"
    mock_admin.assert_called_once()


def test_get_chat_config_uses_defaults_when_postgres_is_unconfigured():
    redis_client = MagicMock(spec=redis.Redis)

    config = get_chat_config(redis_client, "chat-3")

    assert config == CHAT_CONFIG_DEFAULTS
    redis_client.get.assert_not_called()


def test_get_chat_config_respects_per_call_repository_override():
    from api.chat_settings import get_chat_config as real_get_chat_config

    redis_client = MagicMock(spec=redis.Redis)
    repo_one = MagicMock()
    repo_one.is_configured.return_value = True
    repo_one.get_chat_config.return_value = {"link_mode": "reply"}

    repo_two = MagicMock()
    repo_two.is_configured.return_value = True
    repo_two.get_chat_config.return_value = {"link_mode": "off"}

    first = real_get_chat_config(
        redis_client, "chat-a", chat_config_db_service=repo_one
    )
    second = real_get_chat_config(
        redis_client, "chat-b", chat_config_db_service=repo_two
    )

    assert first["link_mode"] == "reply"
    assert second["link_mode"] == "off"
    repo_one.get_chat_config.assert_called_once_with("chat-a", CHAT_CONFIG_DEFAULTS)
    repo_two.get_chat_config.assert_called_once_with("chat-b", CHAT_CONFIG_DEFAULTS)


def test_is_chat_admin_uses_cache():
    redis_client = MagicMock(spec=redis.Redis)

    with (
        patch("api.index._optional_redis_client", return_value=redis_client),
        patch("api.index.redis_get_json", return_value={"is_admin": True}),
        patch("api.index._telegram_request") as mock_request,
    ):
        assert is_chat_admin("chat-1", 42) is True

    mock_request.assert_not_called()


def test_is_chat_admin_fetches_and_caches():
    redis_client = MagicMock(spec=redis.Redis)

    with (
        patch("api.index._optional_redis_client", return_value=redis_client),
        patch("api.index.redis_get_json", return_value=None),
        patch("api.index._telegram_request") as mock_request,
        patch("api.index.redis_setex_json") as mock_set,
    ):
        mock_request.return_value = (
            {"ok": True, "result": {"status": "administrator"}},
            None,
        )

        assert is_chat_admin("chat-1", 99) is True

    mock_request.assert_called_once()
    mock_set.assert_called_once_with(
        redis_client, ANY, CHAT_ADMIN_STATUS_TTL, {"is_admin": True}
    )


def test_get_bot_message_metadata_decodes_bytes():
    redis_client = MagicMock(spec=redis.Redis)
    metadata = {"foo": "bar"}
    redis_client.get.return_value = json.dumps(metadata).encode("utf-8")

    result = get_bot_message_metadata(redis_client, "chat", 1)

    assert result == metadata


def test_get_bot_message_metadata_decodes_string():
    redis_client = MagicMock(spec=redis.Redis)
    metadata = {"foo": "bar"}
    redis_client.get.return_value = json.dumps(metadata)

    result = get_bot_message_metadata(redis_client, "chat", 1)

    assert result == metadata


def test_get_bot_message_metadata_handles_none():
    redis_client = MagicMock(spec=redis.Redis)
    redis_client.get.return_value = None

    result = get_bot_message_metadata(redis_client, "chat", 1)

    assert result is None

    # Test empty string
    msg_text7 = ""
    expected7 = "y que queres que convierta boludo? mandate texto"
    assert convert_to_command(msg_text7) == expected7


def test_config_redis():
    with patch("redis.Redis") as mock_redis:
        # Test successful connection
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.ping.return_value = True

        redis_client = config_redis(host="test", port=1234, password="pass")
        assert redis_client == mock_instance

        # Test failed connection
        mock_instance.ping.side_effect = Exception("Connection failed")
        try:
            config_redis()
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Connection failed"


def test_load_bot_config_returns_hardcoded_prompt(monkeypatch):
    from api import index

    config_module.reset_cache()

    cfg = index.load_bot_config()
    assert cfg["trigger_words"] == [
        "gordo",
        "respondedor",
        "atendedor",
        "gordito",
        "dogor",
        "bot",
    ]
    assert "sos el gordo" in cfg["system_prompt"]
    assert "Reglas" in cfg["system_prompt"]


def test_load_bot_config_caches_result(monkeypatch):
    from api import index

    config_module.reset_cache()

    cfg_first = index.load_bot_config()
    cfg_second = index.load_bot_config()
    assert cfg_second is cfg_first


def test_handle_callback_query_topup_sends_invoice():
    callback = {
        "id": "cbq_topup",
        "data": "topup:p100",
        "from": {"id": 42},
        "message": {"chat": {"id": 1, "type": "private"}, "message_id": 99},
    }

    with (
        patch(
            "api.index.get_ai_billing_pack",
            return_value={"id": "p100", "credits": 1000, "xtr": 50},
        ),
        patch("api.index.credits_db_service.is_configured", return_value=True),
        patch("api.index._send_stars_invoice", return_value=True) as mock_send_invoice,
        patch("api.index._answer_callback_query") as mock_answer,
        patch("api.index.config_redis") as mock_cfg,
    ):
        handle_callback_query(callback)

    mock_cfg.assert_not_called()
    mock_send_invoice.assert_called_once_with(
        chat_id="1",
        user_id=42,
        pack={"id": "p100", "credits": 1000, "xtr": 50},
    )
    mock_answer.assert_called_once()


def test_handle_msg_blocks_config_for_non_admin_group():
    from api.index import handle_msg

    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index.is_chat_admin", return_value=False),
        patch("api.index._report_unauthorized_config_attempt") as mock_report,
        patch("api.index.handle_config_command") as mock_handle_config,
        patch("api.index.get_chat_config", return_value=CHAT_CONFIG_DEFAULTS),
        patch("api.index.should_gordo_respond", return_value=True),
        patch("os.environ.get") as mock_env,
    ):
        mock_env.return_value = "testbot"

        redis_instance = MagicMock()
        mock_config_redis.return_value = redis_instance

        message = {
            "message_id": 1,
            "chat": {"id": 123, "type": "group"},
            "from": {"id": 5, "first_name": "John"},
            "text": "/config",
        }

        assert handle_msg(message) == "ok"

    mock_send_msg.assert_called_once()
    mock_report.assert_called_once()
    mock_handle_config.assert_not_called()


def test_handle_callback_query_blocks_non_admin():
    with (
        patch("api.index.config_redis") as mock_config_redis,
        patch("api.index.is_chat_admin", return_value=False) as mock_is_admin,
        patch("api.index._report_unauthorized_config_attempt") as mock_report,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index._answer_callback_query") as mock_answer,
        patch("api.index.get_chat_config") as mock_get_chat_config,
        patch("api.index.set_chat_config") as mock_set_chat_config,
    ):
        redis_instance = MagicMock()
        mock_config_redis.return_value = redis_instance

        callback_query = {
            "id": "cb-1",
            "from": {"id": 7, "username": "intruder"},
            "message": {
                "message_id": 99,
                "chat": {"id": 456, "type": "group"},
            },
            "data": "cfg:link:reply",
        }

        handle_callback_query(callback_query)

    mock_is_admin.assert_called_once_with("456", 7, redis_client=redis_instance)
    mock_answer.assert_called_once_with("cb-1")
    mock_send_msg.assert_called_once()
    mock_report.assert_called_once()
    mock_get_chat_config.assert_not_called()
    mock_set_chat_config.assert_not_called()


def test_config_redis_with_env_vars():
    from api.config import config_redis as _config_config_redis

    with patch("redis.Redis") as mock_redis, patch("os.environ.get") as mock_env:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_env.side_effect = lambda key, default=None: {
            "REDIS_HOST": "redis.example.com",
            "REDIS_PORT": "1234",
            "REDIS_PASSWORD": "secret",
        }.get(key, default)

        result = _config_config_redis()

        assert result == mock_instance
        mock_redis.assert_called_with(
            host="redis.example.com",
            port=1234,
            password="secret",
            decode_responses=True,
        )


def test_set_chat_config_updates_link_mode_and_persists():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    config = set_chat_config(redis_client, "123", link_mode="reply")

    redis_client.set.assert_not_called()
    assert config["link_mode"] == "reply"


def test_set_chat_config_turns_off_link_mode():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    set_chat_config(redis_client, "123", link_mode="off")

    redis_client.set.assert_not_called()
    redis_client.delete.assert_not_called()


def test_set_chat_config_persists_to_postgres_when_available():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch("api.index.chat_config_db_service.set_chat_config") as mock_pg_set,
    ):
        config = set_chat_config(redis_client, "123", link_mode="reply")

    assert config["link_mode"] == "reply"
    mock_pg_set.assert_called_once_with("123", config)
    redis_client.set.assert_not_called()


def test_get_chat_config_uses_defaults_when_missing():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    with (
        patch("api.index.chat_config_db_service.is_configured", return_value=True),
        patch("api.index.chat_config_db_service.get_chat_config", return_value=None),
    ):
        config = get_chat_config(redis_client, "123")

    assert config["link_mode"] == "reply"
    assert config["ai_random_replies"] is True
    assert config["ai_command_followups"] is True
    assert config["ignore_link_fix_followups"] is True


def test_build_config_text_and_keyboard_reflect_values():
    config = {
        "link_mode": "delete",
        "ai_random_replies": False,
        "ai_command_followups": False,
        "ignore_link_fix_followups": True,
    }

    text = build_config_text(config)
    assert "config del gordo" in text
    assert "links arreglados" in text
    assert "borra el mensaje original y repostea el link arreglado" in text
    assert (
        "si está activado, a veces respondo solo en el grupo aunque nadie me llame"
        in text
    )
    assert (
        "si está activado, después de un comando sigo la conversación si me respondés"
        in text
    )
    assert (
        "si está activado, ignoro respuestas normales a mensajes automáticos con links arreglados"
        in text
    )
    assert "▫️ desactivado" in text
    assert "tocá los botones de abajo para cambiar la config" in text

    keyboard = build_config_keyboard(config)
    assert keyboard["inline_keyboard"][0][1]["text"] == "✅ borrar link"
    assert keyboard["inline_keyboard"][0][2]["text"] == "▫️ apagado"
    assert keyboard["inline_keyboard"][4][0]["text"] == "▫️ me meto en la charla"
    assert keyboard["inline_keyboard"][1][0]["text"] == "▫️ seguir charla en comandos"
    assert (
        keyboard["inline_keyboard"][2][0]["text"]
        == "✅ ignorar replies a links arreglados"
    )
    assert keyboard["inline_keyboard"][4][0]["callback_data"] == "cfg:random:toggle"
    assert keyboard["inline_keyboard"][5][0]["text"] == "0"
    assert keyboard["inline_keyboard"][5][1]["text"] == "-"
    assert keyboard["inline_keyboard"][5][2]["text"] == "5"
    assert (
        keyboard["inline_keyboard"][5][2]["callback_data"] == "cfg:creditless:current"
    )
    assert keyboard["inline_keyboard"][5][3]["text"] == "+"
    assert keyboard["inline_keyboard"][5][4]["text"] == "∞"


def test_handle_config_command_loads_config():
    redis_client = MagicMock()
    with patch("api.index.config_redis", return_value=redis_client):
        text, keyboard = handle_config_command("123")
    assert "config del gordo" in text
    assert "inline_keyboard" in keyboard
    redis_client.get.assert_not_called()


def test_handle_callback_query_updates_random_toggle():
    redis_client = MagicMock()
    callback = {
        "id": "cbq",
        "data": "cfg:random:toggle",
        "message": {"chat": {"id": 1}, "message_id": 99},
    }
    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch(
            "api.index.get_chat_config",
            return_value={
                "link_mode": "off",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
            },
        ) as mock_get,
        patch(
            "api.index.set_chat_config",
            return_value={
                "link_mode": "off",
                "ai_random_replies": False,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
            },
        ) as mock_set,
        patch("api.index.build_config_text", return_value="text") as mock_text,
        patch(
            "api.index.build_config_keyboard", return_value={"inline_keyboard": []}
        ) as mock_keyboard,
        patch("api.index.edit_message", return_value=True) as mock_edit,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index._answer_callback_query") as mock_answer,
    ):
        handle_callback_query(callback)

    mock_get.assert_called_once_with(redis_client, "1")
    mock_set.assert_called_once_with(redis_client, "1", ai_random_replies=False)
    mock_text.assert_called_once()
    mock_keyboard.assert_called_once()
    mock_edit.assert_called_once_with("1", 99, "text", {"inline_keyboard": []})
    mock_send_msg.assert_not_called()
    mock_answer.assert_called_once_with("cbq")


def test_handle_callback_query_falls_back_when_edit_fails():
    redis_client = MagicMock()
    callback = {
        "id": "cbq",
        "data": "cfg:link:reply",
        "message": {"chat": {"id": 9}, "message_id": 42},
    }
    updated_config = {
        "link_mode": "reply",
        "ai_random_replies": True,
        "ai_command_followups": True,
        "ignore_link_fix_followups": True,
    }

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch(
            "api.index.get_chat_config",
            return_value={**updated_config, "link_mode": "off"},
        ) as mock_get,
        patch("api.index.set_chat_config", return_value=updated_config) as mock_set,
        patch("api.index.build_config_text", return_value="new text") as mock_text,
        patch(
            "api.index.build_config_keyboard", return_value={"inline_keyboard": ["btn"]}
        ) as mock_keyboard,
        patch("api.index.edit_message", return_value=False) as mock_edit,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index._answer_callback_query") as mock_answer,
        patch("api.index._log_config_event") as mock_log,
    ):
        handle_callback_query(callback)

    mock_get.assert_called_once_with(redis_client, "9")
    mock_set.assert_called_once_with(redis_client, "9", link_mode="reply")
    mock_text.assert_called_once_with(updated_config, "")
    mock_keyboard.assert_called_once_with(updated_config, "")
    mock_edit.assert_called_once_with("9", 42, "new text", {"inline_keyboard": ["btn"]})
    mock_log.assert_any_call(
        "Falling back to new config message", {"chat_id": "9", "message_id": 42}
    )
    mock_send_msg.assert_called_once_with(
        "9", "new text", reply_markup={"inline_keyboard": ["btn"]}
    )
    mock_answer.assert_called_once_with("cbq")


def test_handle_callback_query_updates_link_fix_followups_toggle():
    redis_client = MagicMock()
    callback = {
        "id": "cbq",
        "data": "cfg:linkfixfollowups:toggle",
        "message": {"chat": {"id": 1}, "message_id": 99},
    }
    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch(
            "api.index.get_chat_config",
            return_value={
                "link_mode": "off",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
            },
        ) as mock_get,
        patch(
            "api.index.set_chat_config",
            return_value={
                "link_mode": "off",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": False,
            },
        ) as mock_set,
        patch("api.index.build_config_text", return_value="text") as mock_text,
        patch(
            "api.index.build_config_keyboard", return_value={"inline_keyboard": []}
        ) as mock_keyboard,
        patch("api.index.edit_message", return_value=True) as mock_edit,
        patch("api.index.send_msg") as mock_send_msg,
        patch("api.index._answer_callback_query") as mock_answer,
    ):
        handle_callback_query(callback)

    mock_get.assert_called_once_with(redis_client, "1")
    mock_set.assert_called_once_with(redis_client, "1", ignore_link_fix_followups=False)
    mock_text.assert_called_once()
    mock_keyboard.assert_called_once()
    mock_edit.assert_called_once_with("1", 99, "text", {"inline_keyboard": []})
    mock_send_msg.assert_not_called()
    mock_answer.assert_called_once_with("cbq")


def test_save_and_get_bot_message_metadata():
    redis_client = MagicMock()

    save_bot_message_metadata(redis_client, "chat", 100, {"foo": "bar"}, ttl=120)

    setex_call = redis_client.setex.call_args
    assert setex_call[0][0] == "bot_message_meta:chat:100"
    assert setex_call[0][1] == 120
    saved_payload = setex_call[0][2]
    assert json.loads(saved_payload) == {"foo": "bar"}

    redis_client.get.return_value = saved_payload
    metadata = get_bot_message_metadata(redis_client, "chat", 100)
    assert metadata == {"foo": "bar"}


class TestReminderCallback:
    def test_task_delete_routes(self):
        from api.index import handle_callback_query

        callback = {
            "id": "cbq1",
            "data": "task:del:abc123",
            "from": {"id": 42},
            "message": {"chat": {"id": 1, "type": "private"}, "message_id": 99},
        }

        with (
            patch("api.index._answer_callback_query") as mock_answer,
            patch("api.index.handle_task_callback") as mock_handler,
        ):
            handle_callback_query(callback)
            mock_handler.assert_called_once_with(callback)
        mock_answer.assert_not_called()


class TestHandleTaskCallback:
    def test_delete_task(self):
        from api.index import handle_task_callback

        callback = {
            "id": "cbq1",
            "data": "task:del:abc123",
            "from": {"id": 42},
            "message": {"chat": {"id": 1, "type": "private"}, "message_id": 99},
        }

        with (
            patch(
                "api.index._task_list_tasks",
                return_value=[{"id": "abc123", "user_id": 42, "text": "tarea test"}],
            ),
            patch("api.index._task_cancel_task") as mock_cancel,
            patch("api.index._answer_callback_query") as mock_answer,
            patch("api.index.edit_message") as mock_edit,
        ):
            handle_task_callback(callback)

        mock_cancel.assert_called_once_with("abc123")
        mock_answer.assert_called_once_with("cbq1", text="tarea abc123 borrada")
        mock_edit.assert_called_once()

    def test_delete_task_unauthorized(self):
        from api.index import handle_task_callback

        callback = {
            "id": "cbq1",
            "data": "task:del:abc123",
            "from": {"id": 999},
            "message": {"chat": {"id": 1, "type": "group"}, "message_id": 99},
        }

        with (
            patch(
                "api.index._task_list_tasks",
                return_value=[{"id": "abc123", "user_id": 42, "text": "tarea test"}],
            ),
            patch("api.index.config_redis"),
            patch("api.index.is_chat_admin", return_value=False),
            patch("api.index._task_cancel_task") as mock_cancel,
            patch("api.index._answer_callback_query") as mock_answer,
        ):
            handle_task_callback(callback)

        mock_cancel.assert_not_called()
        mock_answer.assert_called_once_with(
            "cbq1",
            text="solo el creador o un admin pueden borrar esta tarea",
            show_alert=True,
        )


class TestTimezoneConfig:
    def test_config_text_includes_timezone(self):
        from api.chat_settings import build_config_text, CHAT_CONFIG_DEFAULTS

        config = dict(CHAT_CONFIG_DEFAULTS)
        config["timezone_offset"] = -3
        text = build_config_text(config)
        assert "zona horaria" in text
        assert "UTC-3" in text

    def test_config_text_different_timezone(self):
        from api.chat_settings import build_config_text

        config = {
            "link_mode": "reply",
            "ai_random_replies": True,
            "ai_command_followups": True,
            "ignore_link_fix_followups": True,
            "timezone_offset": 0,
        }
        text = build_config_text(config)
        assert "UTC" in text

    def test_config_keyboard_includes_timezone_buttons(self):
        from api.chat_settings import build_config_keyboard, CHAT_CONFIG_DEFAULTS

        config = dict(CHAT_CONFIG_DEFAULTS)
        config["timezone_offset"] = -3
        keyboard = build_config_keyboard(config)
        rows = keyboard["inline_keyboard"]

        tz_row = next(
            r for r in rows if any("cfg:timezone:" in btn["callback_data"] for btn in r)
        )
        assert any("🌍" in btn["text"] for btn in tz_row)
        assert any("UTC" in btn["text"] for btn in tz_row)
        assert any("UTC-3" in btn["text"] for btn in tz_row)
        assert any("cfg:timezone:" in btn["callback_data"] for btn in tz_row)
        assert any("➖" in btn["text"] for btn in tz_row)
        assert any("➕" in btn["text"] for btn in tz_row)

    def test_config_keyboard_current_timezone_checked(self):
        from api.chat_settings import build_config_keyboard, CHAT_CONFIG_DEFAULTS

        config = dict(CHAT_CONFIG_DEFAULTS)
        config["timezone_offset"] = -3
        keyboard = build_config_keyboard(config)

        rows = keyboard["inline_keyboard"]
        tz_row = next(
            r for r in rows if any("cfg:timezone:" in btn["callback_data"] for btn in r)
        )
        utc3_btn = next(b for b in tz_row if "UTC-3" in b["text"])
        assert "🌍" in utc3_btn["text"]

    def test_config_keyboard_creditless_stepper_uses_default_five(self):
        from api.chat_settings import build_config_keyboard, CHAT_CONFIG_DEFAULTS

        keyboard = build_config_keyboard(dict(CHAT_CONFIG_DEFAULTS))
        creditless_row = keyboard["inline_keyboard"][5]

        assert [button["text"] for button in creditless_row] == [
            "0",
            "-",
            "5",
            "+",
            "∞",
        ]
        assert [button["callback_data"] for button in creditless_row] == [
            "cfg:creditless:none",
            "cfg:creditless:decrease",
            "cfg:creditless:current",
            "cfg:creditless:increase",
            "cfg:creditless:unlimited",
        ]

    def test_handle_callback_query_updates_timezone(self):
        from api.index import handle_callback_query

        callback = {
            "id": "cbq",
            "data": "cfg:timezone:-5",
            "message": {"chat": {"id": 1}, "message_id": 99},
        }
        with (
            patch("api.index.config_redis") as mock_redis,
            patch(
                "api.index.get_chat_config",
                return_value={
                    "link_mode": "reply",
                    "ai_random_replies": True,
                    "ai_command_followups": True,
                    "ignore_link_fix_followups": True,
                    "timezone_offset": -3,
                },
            ) as mock_get,
            patch(
                "api.index.set_chat_config",
                return_value={
                    "link_mode": "reply",
                    "ai_random_replies": True,
                    "ai_command_followups": True,
                    "ignore_link_fix_followups": True,
                    "timezone_offset": -5,
                },
            ) as mock_set,
            patch("api.index.build_config_text", return_value="text") as mock_text,
            patch(
                "api.index.build_config_keyboard", return_value={"inline_keyboard": []}
            ) as mock_keyboard,
            patch("api.index.edit_message", return_value=True) as mock_edit,
            patch("api.index._answer_callback_query") as mock_answer,
        ):
            handle_callback_query(callback)

        mock_set.assert_called_once()
        call_kwargs = mock_set.call_args.kwargs
        assert call_kwargs.get("timezone_offset") == -5


def test_handle_callback_query_sets_creditless_unlimited():
    from api.index import handle_callback_query

    callback = {
        "id": "cbq",
        "data": "cfg:creditless:unlimited",
        "message": {"chat": {"id": 1}, "message_id": 99},
    }
    with (
        patch("api.index.config_redis") as mock_redis,
        patch(
            "api.index.get_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": 5,
            },
        ),
        patch(
            "api.index.set_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": -1,
            },
        ) as mock_set,
        patch("api.index.build_config_text", return_value="text"),
        patch("api.index.build_config_keyboard", return_value={"inline_keyboard": []}),
        patch("api.index.edit_message", return_value=True),
        patch("api.index._answer_callback_query"),
    ):
        handle_callback_query(callback)

    mock_set.assert_called_once()
    call_kwargs = mock_set.call_args.kwargs
    assert call_kwargs.get("creditless_user_hourly_limit") == -1


def test_handle_callback_query_increases_creditless_limit():
    from api.index import handle_callback_query

    callback = {
        "id": "cbq",
        "data": "cfg:creditless:increase",
        "message": {"chat": {"id": 1}, "message_id": 99},
    }
    with (
        patch("api.index.config_redis"),
        patch(
            "api.index.get_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": 5,
            },
        ),
        patch(
            "api.index.set_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": 6,
            },
        ) as mock_set,
        patch("api.index.build_config_text", return_value="text"),
        patch("api.index.build_config_keyboard", return_value={"inline_keyboard": []}),
        patch("api.index.edit_message", return_value=True),
        patch("api.index._answer_callback_query"),
    ):
        handle_callback_query(callback)

    assert mock_set.call_args.kwargs["creditless_user_hourly_limit"] == 6


def test_handle_callback_query_decrease_clamps_creditless_limit_at_zero():
    from api.index import handle_callback_query

    callback = {
        "id": "cbq",
        "data": "cfg:creditless:decrease",
        "message": {"chat": {"id": 1}, "message_id": 99},
    }
    with (
        patch("api.index.config_redis"),
        patch(
            "api.index.get_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": 0,
            },
        ),
        patch(
            "api.index.set_chat_config",
            return_value={
                "link_mode": "reply",
                "ai_random_replies": True,
                "ai_command_followups": True,
                "ignore_link_fix_followups": True,
                "creditless_user_hourly_limit": 0,
            },
        ) as mock_set,
        patch("api.index.build_config_text", return_value="text"),
        patch("api.index.build_config_keyboard", return_value={"inline_keyboard": []}),
        patch("api.index.edit_message", return_value=True),
        patch("api.index._answer_callback_query"),
    ):
        handle_callback_query(callback)

    assert mock_set.call_args.kwargs["creditless_user_hourly_limit"] == 0
