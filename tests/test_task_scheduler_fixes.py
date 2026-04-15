"""Tests for task_scheduler fixes: fallback marker stripping, list_tasks
resilience, billing integration, and unknown tool call filtering."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from api.tools.task_scheduler import (
    TASK_REDIS_PREFIX,
    _strip_response_marker,
    list_tasks,
    schedule_task,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_redis_with_task(task_data: dict) -> MagicMock:
    redis_client = MagicMock()
    redis_client.get.return_value = json.dumps(task_data)
    return redis_client


def _credits_ok_mock() -> MagicMock:
    """credits_db_service where charge succeeds."""
    mock_cs = MagicMock()
    mock_cs.is_configured.return_value = True
    mock_cs.charge_ai_credits.return_value = {"ok": True, "source": "user"}
    mock_cs.refund_ai_charge = MagicMock()
    return mock_cs


def _credits_empty_mock() -> MagicMock:
    """credits_db_service where user has no credits."""
    mock_cs = MagicMock()
    mock_cs.is_configured.return_value = True
    mock_cs.charge_ai_credits.return_value = {
        "ok": False,
        "user_balance_credit_units": 0,
        "chat_balance_credit_units": 0,
    }
    return mock_cs


# ---------------------------------------------------------------------------
# _strip_response_marker
# ---------------------------------------------------------------------------


class TestStripResponseMarker:
    def test_strips_marker(self):
        assert _strip_response_marker("[[AI_FALLBACK]]no boludo") == "no boludo"

    def test_strips_marker_with_leading_whitespace(self):
        assert _strip_response_marker("[[AI_FALLBACK]]  hola") == "hola"

    def test_no_marker(self):
        assert _strip_response_marker("respuesta normal") == "respuesta normal"

    def test_empty_string(self):
        assert _strip_response_marker("") == ""

    def test_marker_only(self):
        assert _strip_response_marker("[[AI_FALLBACK]]") == ""

    def test_marker_in_middle_not_stripped(self):
        text = "algo [[AI_FALLBACK]] algo"
        assert _strip_response_marker(text) == text


# ---------------------------------------------------------------------------
# _fire_task -- core behavior
# ---------------------------------------------------------------------------


class TestFireTaskStripsMarker:
    """Verify _fire_task strips [[AI_FALLBACK]] from the sent message."""

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_strips_fallback_marker(self, mock_sched, mock_redis):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "buscar noticias de linux",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.send_msg") as mock_send:
                with patch("api.index.ask_ai", return_value="[[AI_FALLBACK]]no boludo"):
                    _fire_task("abc123")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "[[AI_FALLBACK]]" not in sent_text
        assert "no boludo" in sent_text

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_recurring_strips_fallback_marker(self, mock_sched, mock_redis):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "noticias",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": 3600,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.send_msg") as mock_send:
                with patch(
                    "api.index.ask_ai", return_value="[[AI_FALLBACK]]fallback response"
                ):
                    _fire_task("abc123")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "[[AI_FALLBACK]]" not in sent_text
        assert "fallback response" in sent_text

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_normal_response_unchanged(self, mock_sched, mock_redis):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "buscar noticias",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.send_msg") as mock_send:
                with patch(
                    "api.index.ask_ai",
                    return_value="aca tenes las noticias pedazo de bobi",
                ):
                    _fire_task("abc123")

        sent_text = mock_send.call_args[0][1]
        assert "aca tenes las noticias pedazo de bobi" in sent_text

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_sent_message_includes_tarea_programada_prefix(
        self, mock_sched, mock_redis
    ):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.send_msg") as mock_send:
                with patch("api.index.ask_ai", return_value="dale"):
                    _fire_task("abc123")

        sent_text = mock_send.call_args[0][1]
        assert "tarea programada:" in sent_text


# ---------------------------------------------------------------------------
# _fire_task -- cleanup (one-shot vs recurring)
# ---------------------------------------------------------------------------


class TestFireTaskCleanup:
    """Verify Redis key and job removal behavior for one-shot vs recurring tasks."""

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_deletes_redis_key_on_success(self, mock_sched, mock_redis):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.ask_ai", return_value="dale"):
                with patch("api.index.send_msg"):
                    _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_recurring_does_not_delete_redis_key_on_success(
        self, mock_sched, mock_redis
    ):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "noticias",
                "user_name": "",
                "user_id": 77,
                "interval_seconds": 3600,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_ok_mock()):
            with patch("api.index.ask_ai", return_value="noticias"):
                with patch("api.index.send_msg"):
                    _fire_task("x1")

        redis_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# _fire_task -- billing / auto-deletion rules
# ---------------------------------------------------------------------------


class TestFireTaskBilling:
    """
    Tasks are always charged to the creating user only -- never to the group
    chat balance -- regardless of where the task runs. If the user has no
    credits or cannot be identified, the task is deleted automatically.
    """

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_credits_deletes_one_shot_task(self, mock_sched, mock_redis):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_empty_mock()):
            with patch("api.index.send_msg"):
                _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_credits_deletes_recurring_task(self, mock_sched, mock_redis):
        """Even recurring tasks are deleted when the user runs out of credits."""
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "noticias",
                "user_name": "",
                "user_id": 77,
                "interval_seconds": 3600,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_empty_mock()):
            with patch("api.index.send_msg"):
                _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_user_id_deletes_task(self, mock_sched, mock_redis):
        """If user_id is missing from task data the task cannot be billed and is deleted."""
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "",
                "user_id": None,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        with patch("api.index.credits_db_service", _credits_empty_mock()):
            with patch("api.index.send_msg"):
                _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_billing_uses_private_chat_type_never_charges_group(
        self, mock_sched, mock_redis
    ):
        """
        Even if the task was created in a group chat, billing is done as
        chat_type='private' with numeric_chat_id=None so only the user's
        balance is touched, never the group balance.
        """
        from api.tools.task_scheduler import _fire_task
        from api.ai_billing import AIMessageBilling

        redis_client = _make_redis_with_task(
            {
                "chat_id": "-100999",
                "text": "noticias",
                "user_name": "astro",
                "user_id": 42,
                "interval_seconds": None,
            }
        )
        mock_redis.return_value = redis_client

        captured: list = []
        original_init = AIMessageBilling.__init__

        def capture_init(self, **kwargs):
            captured.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(AIMessageBilling, "__init__", capture_init):
            with patch("api.index.credits_db_service", _credits_ok_mock()):
                with patch("api.index.ask_ai", return_value="ok"):
                    with patch("api.index.send_msg"):
                        _fire_task("x1")

        assert captured, "AIMessageBilling was not instantiated"
        init_kwargs = captured[0]
        assert init_kwargs["chat_type"] == "private"
        assert init_kwargs["numeric_chat_id"] is None


# ---------------------------------------------------------------------------
# list_tasks resilience
# ---------------------------------------------------------------------------


class TestListTasksResilience:
    """list_tasks should show tasks even when APScheduler is unavailable."""

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_lists_tasks_without_scheduler(self, mock_sched, mock_redis):
        mock_sched.return_value = None

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_data = {
            "id": "abc1",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "interval_seconds": None,
            "run_date": "2026-04-15T04:22:00+00:00",
        }
        redis_client.scan_iter.return_value = [f"{TASK_REDIS_PREFIX}abc1"]
        redis_client.get.return_value = json.dumps(task_data)

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["id"] == "abc1"
        assert tasks[0]["text"] == "recordame algo"
        assert tasks[0]["next_run"] == "15/04 01:22"

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_lists_tasks_when_job_missing_from_scheduler(self, mock_sched, mock_redis):
        scheduler = MagicMock()
        scheduler.get_job.return_value = None
        mock_sched.return_value = scheduler

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_data = {
            "id": "abc1",
            "chat_id": "123",
            "text": "tarea",
            "user_name": "",
            "interval_seconds": None,
            "run_date": "2026-04-15T05:00:00+00:00",
        }
        redis_client.scan_iter.return_value = [f"{TASK_REDIS_PREFIX}abc1"]
        redis_client.get.return_value = json.dumps(task_data)

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["next_run"] == "15/04 02:00"

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_uses_scheduler_next_run_when_available(self, mock_sched, mock_redis):
        scheduler = MagicMock()
        job = MagicMock()
        job.next_run_time = "2026-04-15T06:00:00+00:00"
        scheduler.get_job.return_value = job
        mock_sched.return_value = scheduler

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_data = {
            "id": "abc1",
            "chat_id": "123",
            "text": "tarea",
            "user_name": "",
            "interval_seconds": None,
            "run_date": "2026-04-15T05:00:00+00:00",
        }
        redis_client.scan_iter.return_value = [f"{TASK_REDIS_PREFIX}abc1"]
        redis_client.get.return_value = json.dumps(task_data)

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["next_run"] == "15/04 03:00"

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_filters_by_chat_id(self, mock_sched, mock_redis):
        mock_sched.return_value = None

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task1 = json.dumps(
            {
                "id": "a",
                "chat_id": "123",
                "text": "mi tarea",
                "user_name": "",
                "interval_seconds": None,
                "run_date": None,
            }
        )
        task2 = json.dumps(
            {
                "id": "b",
                "chat_id": "999",
                "text": "otra tarea",
                "user_name": "",
                "interval_seconds": None,
                "run_date": None,
            }
        )
        redis_client.scan_iter.return_value = [
            f"{TASK_REDIS_PREFIX}a",
            f"{TASK_REDIS_PREFIX}b",
        ]
        redis_client.get.side_effect = [task1, task2]

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["id"] == "a"

    @patch("api.tools.task_scheduler._get_redis")
    def test_returns_empty_when_redis_unavailable(self, mock_redis):
        mock_redis.return_value = None
        assert list_tasks("123") == []

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_lists_multiple_tasks(self, mock_sched, mock_redis):
        mock_sched.return_value = None

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task1 = json.dumps(
            {
                "id": "a",
                "chat_id": "123",
                "text": "buscar noticias de linux",
                "user_name": "astro",
                "interval_seconds": None,
                "run_date": "2026-04-15T04:22:00+00:00",
            }
        )
        task2 = json.dumps(
            {
                "id": "b",
                "chat_id": "123",
                "text": "bañarse",
                "user_name": "astro",
                "interval_seconds": None,
                "run_date": "2026-04-15T04:24:00+00:00",
            }
        )
        redis_client.scan_iter.return_value = [
            f"{TASK_REDIS_PREFIX}a",
            f"{TASK_REDIS_PREFIX}b",
        ]
        redis_client.get.side_effect = [task1, task2]

        tasks = list_tasks("123")

        assert len(tasks) == 2
        texts = {t["text"] for t in tasks}
        assert "buscar noticias de linux" in texts
        assert "bañarse" in texts


# ---------------------------------------------------------------------------
# schedule_task -- Redis storage
# ---------------------------------------------------------------------------


class TestScheduleTaskStoresRunDate:
    """schedule_task should store run_date in Redis data."""

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_stores_run_date(self, mock_sched, mock_redis):
        scheduler = MagicMock()
        mock_sched.return_value = scheduler

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_id = schedule_task("123", "test", delay_seconds=300)

        assert task_id is not None
        stored_call = redis_client.setex.call_args
        stored_data = json.loads(stored_call[0][2])
        assert stored_data["run_date"] is not None
        assert "T" in stored_data["run_date"]

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_recurring_stores_null_run_date(self, mock_sched, mock_redis):
        scheduler = MagicMock()
        mock_sched.return_value = scheduler

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_id = schedule_task("123", "test", interval_seconds=3600)

        assert task_id is not None
        stored_call = redis_client.setex.call_args
        stored_data = json.loads(stored_call[0][2])
        assert stored_data["run_date"] is None


# ---------------------------------------------------------------------------
# _get_openrouter_ai_response_result -- unknown tool call filtering
# ---------------------------------------------------------------------------


class TestToolCallFiltering:
    """_get_openrouter_ai_response_result should skip unknown tool calls."""

    def _build_tool_call_response(
        self, tool_name, tool_args='{"query": "test"}', content=None
    ):
        fn = MagicMock()
        fn.name = tool_name
        fn.arguments = tool_args

        tc = MagicMock()
        tc.id = "call_1"
        tc.function = fn

        message = MagicMock()
        message.content = content
        message.tool_calls = [tc]

        choice = MagicMock()
        choice.finish_reason = "tool_calls"
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        return response

    def _build_stop_response(self, text="respuesta final"):
        message = MagicMock()
        message.content = text
        message.annotations = []

        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        return response

    def test_skips_openrouter_web_search_and_extracts_content(self):
        from api.index import _get_openrouter_ai_response_result

        tool_response = self._build_tool_call_response(
            "openrouter_web_search",
            content="aca tenes las noticias boludo",
        )
        client = MagicMock()
        client.chat.completions.create.return_value = tool_response

        extra_tools = [
            {
                "type": "function",
                "function": {
                    "name": "task_set",
                    "description": "test",
                    "parameters": {},
                },
            }
        ]

        with patch("api.index._get_openrouter_client", return_value=client):
            result = _get_openrouter_ai_response_result(
                {"role": "system", "content": "sys"},
                [{"role": "user", "content": "noticias linux"}],
                enable_web_search=True,
                extra_tools=extra_tools,
            )

        assert result is not None
        assert result.text == "aca tenes las noticias boludo"

    def test_skips_unknown_tool_and_breaks_when_no_content(self):
        from api.index import _get_openrouter_ai_response_result

        tool_response = self._build_tool_call_response(
            "openrouter_web_search",
            content=None,
        )
        client = MagicMock()
        client.chat.completions.create.return_value = tool_response

        extra_tools = [
            {
                "type": "function",
                "function": {
                    "name": "task_set",
                    "description": "test",
                    "parameters": {},
                },
            }
        ]

        with patch("api.index._get_openrouter_client", return_value=client):
            result = _get_openrouter_ai_response_result(
                {"role": "system", "content": "sys"},
                [{"role": "user", "content": "hola"}],
                extra_tools=extra_tools,
            )

        assert result is None

    def test_executes_known_tool_call(self):
        from api.index import _get_openrouter_ai_response_result

        tool_response = self._build_tool_call_response(
            "calculate",
            tool_args='{"expression": "2+2"}',
        )
        stop_response = self._build_stop_response("el resultado es 4")
        client = MagicMock()
        client.chat.completions.create.side_effect = [tool_response, stop_response]

        extra_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "test",
                    "parameters": {},
                },
            }
        ]

        with patch("api.index._get_openrouter_client", return_value=client):
            result = _get_openrouter_ai_response_result(
                {"role": "system", "content": "sys"},
                [{"role": "user", "content": "cuanto es 2+2"}],
                extra_tools=extra_tools,
            )

        assert result is not None
        assert result.text == "el resultado es 4"
        assert client.chat.completions.create.call_count == 2

    def test_mixed_known_and_unknown_tool_calls(self):
        from api.index import _get_openrouter_ai_response_result

        fn1 = MagicMock()
        fn1.name = "openrouter_web_search"
        fn1.arguments = '{"query": "test"}'
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function = fn1

        fn2 = MagicMock()
        fn2.name = "calculate"
        fn2.arguments = '{"expression": "1+1"}'
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function = fn2

        message = MagicMock()
        message.content = None
        message.tool_calls = [tc1, tc2]

        choice = MagicMock()
        choice.finish_reason = "tool_calls"
        choice.message = message

        tool_response = MagicMock()
        tool_response.choices = [choice]
        tool_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        stop_response = self._build_stop_response("2")
        client = MagicMock()
        client.chat.completions.create.side_effect = [tool_response, stop_response]

        extra_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "test",
                    "parameters": {},
                },
            }
        ]

        with patch("api.index._get_openrouter_client", return_value=client):
            result = _get_openrouter_ai_response_result(
                {"role": "system", "content": "sys"},
                [{"role": "user", "content": "test"}],
                extra_tools=extra_tools,
            )

        assert result is not None
        assert result.text == "2"
        assert client.chat.completions.create.call_count == 2

        second_call_messages = client.chat.completions.create.call_args_list[1]
        messages = second_call_messages.kwargs.get(
            "messages", second_call_messages[1] if len(second_call_messages) > 1 else []
        )
        tool_call_names = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tool_call_names.append(tc["function"]["name"])
        assert "calculate" in tool_call_names
        assert "openrouter_web_search" not in tool_call_names
