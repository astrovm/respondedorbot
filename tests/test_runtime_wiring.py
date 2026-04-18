from tests.support import *  # noqa: F401,F403


def test_index_handle_msg_balance_uses_real_wiring(monkeypatch):
    from api import index as _api_index

    message = {
        "message_id": "11",
        "chat": {"id": "101", "type": "private"},
        "from": {"id": 55, "first_name": "Ana", "username": "ana"},
        "text": "/balance",
    }
    redis_client = MagicMock()
    mock_send_msg = MagicMock()
    mock_credits = MagicMock()
    mock_credits.is_configured.return_value = True
    mock_credits.get_balance.return_value = 420

    monkeypatch.setattr(_api_index, "config_redis", lambda: redis_client)
    monkeypatch.setattr(_api_index, "send_msg", mock_send_msg)
    monkeypatch.setattr(_api_index, "credits_db_service", mock_credits)
    monkeypatch.setattr(_api_index, "admin_report", MagicMock())
    monkeypatch.setenv("TELEGRAM_USERNAME", "testbot")

    result = _api_index.handle_msg(message)

    assert result == "ok"
    assert "42.0" in mock_send_msg.call_args[0][1]


def test_index_tasks_command_formats_real_runtime_task_rows(monkeypatch):
    from api import index as _api_index

    monkeypatch.setattr(
        _api_index,
        "_task_list_tasks",
        lambda chat_id: [
            {
                "id": "abc123",
                "text": "recordame algo",
                "user_name": "@astro",
                "interval_seconds": None,
                "trigger_config": None,
                "next_run": "17/04 22:04",
            }
        ],
    )

    text, keyboard = _api_index.tasks_command("123")

    assert "[abc123]" in text
    assert "recordame algo" in text
    assert keyboard == {
        "inline_keyboard": [
            [{"text": "borrar abc123", "callback_data": "task:del:abc123"}]
        ]
    }


def test_task_set_reports_runtime_unavailable_reason():
    from api.tools.registry import execute_tool

    with patch("api.tools.task_set.get_scheduler_runtime_status") as mock_status:
        mock_status.return_value = {
            "ready": False,
            "reason": "storage unavailable",
            "scheduler": True,
            "redis": False,
            "executor": True,
        }

        result = execute_tool(
            "task_set",
            {"text": "algo", "delay_seconds": 1800},
            {"chat_id": "123", "user_name": "u"},
        )

    assert result.output == "no se pudo crear la tarea: storage unavailable"
