"""Tests for task_scheduler fixes: fallback marker stripping, list_tasks
resilience, billing integration, and unknown tool call filtering."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from api.tools.task_scheduler import (
    TASK_REDIS_PREFIX,
    _strip_response_marker,
    get_scheduler,
    list_tasks,
    schedule_task,
    set_task_executor,
    set_redis_client,
    shutdown_scheduler,
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


def _build_executor_for_tests(
    credits_db_service: MagicMock,
    ask_ai_fn: MagicMock,
    send_msg_fn: MagicMock,
    billing_factory=None,
) -> MagicMock:
    """Build a TaskExecutor-like mock injected into the scheduler."""
    from api.task_executor import TaskExecutor

    billing = MagicMock()
    billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
    if billing_factory is None:
        billing_factory = MagicMock(return_value=billing)

    return TaskExecutor(
        ask_ai=ask_ai_fn,
        send_msg=send_msg_fn,
        admin_report=MagicMock(),
        credits_db_service=credits_db_service,
        gen_random_fn=MagicMock(),
        build_insufficient_credits_message_fn=MagicMock(),
        billing_factory=billing_factory,
    )


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

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_strips_fallback_marker(self, mock_sched):
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
        set_redis_client(redis_client)

        mock_ask_ai = MagicMock(return_value="[[AI_FALLBACK]]no boludo")
        mock_send = MagicMock()
        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(credits, mock_ask_ai, mock_send)
        set_task_executor(executor)

        _fire_task("abc123")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "[[AI_FALLBACK]]" not in sent_text
        assert "no boludo" in sent_text

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_recurring_strips_fallback_marker(self, mock_sched):
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
        set_redis_client(redis_client)

        mock_ask_ai = MagicMock(return_value="[[AI_FALLBACK]]fallback response")
        mock_send = MagicMock()
        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(credits, mock_ask_ai, mock_send)
        set_task_executor(executor)

        _fire_task("abc123")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "[[AI_FALLBACK]]" not in sent_text
        assert "fallback response" in sent_text

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_normal_response_unchanged(self, mock_sched):
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
        set_redis_client(redis_client)

        mock_ask_ai = MagicMock(return_value="aca tenes las noticias pedazo de bobi")
        mock_send = MagicMock()
        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(credits, mock_ask_ai, mock_send)
        set_task_executor(executor)

        _fire_task("abc123")

        sent_text = mock_send.call_args[0][1]
        assert "aca tenes las noticias pedazo de bobi" in sent_text

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_sent_message_includes_tarea_programada_prefix(self, mock_sched):
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
        set_redis_client(redis_client)

        mock_ask_ai = MagicMock(return_value="dale")
        mock_send = MagicMock()
        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(credits, mock_ask_ai, mock_send)
        set_task_executor(executor)

        _fire_task("abc123")

        sent_text = mock_send.call_args[0][1]
        assert "tarea programada:" in sent_text


# ---------------------------------------------------------------------------
# _fire_task -- cleanup (one-shot vs recurring)
# ---------------------------------------------------------------------------


class TestFireTaskCleanup:
    """Verify Redis key and job removal behavior for one-shot vs recurring tasks."""

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_deletes_redis_key_on_success(self, mock_sched):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "@testuser",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(
            credits,
            MagicMock(return_value="dale"),
            MagicMock(),
        )
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_recurring_does_not_delete_redis_key_on_success(self, mock_sched):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "noticias",
                "user_name": "@testuser",
                "user_id": 77,
                "interval_seconds": 3600,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_ok_mock()
        executor = _build_executor_for_tests(
            credits,
            MagicMock(return_value="noticias"),
            MagicMock(),
        )
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_not_called()

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_one_shot_refunds_reservation_and_deletes_task_on_fallback(
        self, mock_sched
    ):
        from api.tools.task_scheduler import _fire_task

        credits = _credits_ok_mock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        billing_factory = MagicMock(return_value=billing)

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "@testuser",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        executor = _build_executor_for_tests(
            credits,
            MagicMock(return_value="[[AI_FALLBACK]]dale"),
            MagicMock(),
            billing_factory=billing_factory,
        )
        set_task_executor(executor)

        _fire_task("x1")

        billing.refund_reserved_ai_credits.assert_called_once()
        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")


# ---------------------------------------------------------------------------
# _fire_task -- billing / auto-deletion rules
# ---------------------------------------------------------------------------


class TestFireTaskBilling:
    """
    Tasks are always charged to the creating user only -- never to the group
    chat balance -- regardless of where the task runs. If the user has no
    credits or cannot be identified, the task is deleted automatically.
    """

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_credits_deletes_one_shot_task(self, mock_sched):
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "@testuser",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_empty_mock()
        executor = _build_executor_for_tests(credits, MagicMock(), MagicMock())
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_credits_skips_but_keeps_recurring_task(self, mock_sched):
        """Even if credits fail, recurring tasks are skipped for this run but kept in Redis."""
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "noticias",
                "user_name": "@testuser",
                "user_id": 77,
                "interval_seconds": 3600,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_empty_mock()
        executor = _build_executor_for_tests(credits, MagicMock(), MagicMock())
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_not_called()

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_no_user_id_deletes_task(self, mock_sched):
        """If user_id is missing from task data the task cannot be billed and is deleted."""
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "@testuser",
                "user_id": None,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_empty_mock()
        executor = _build_executor_for_tests(credits, MagicMock(), MagicMock())
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_billing_uses_private_chat_type_never_charges_group(self, mock_sched):
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
        set_redis_client(redis_client)

        credits = _credits_ok_mock()

        captured_init_kwargs = []
        original_init = AIMessageBilling.__init__

        def capture_init(self, **kwargs):
            captured_init_kwargs.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(AIMessageBilling, "__init__", capture_init):
            real_billing = AIMessageBilling(
                credits_db_service=credits,
                admin_reporter=MagicMock(),
                gen_random_fn=MagicMock(),
                build_insufficient_credits_message_fn=MagicMock(),
                maybe_grant_onboarding_credits_fn=MagicMock(),
                command="task",
                chat_id="-100999",
                chat_type="private",
                user_id=42,
                numeric_chat_id=None,
                message={},
            )

        real_billing.reserve_ai_credits = MagicMock(
            return_value=({"reservation": "ok"}, None)
        )
        billing_factory = MagicMock(return_value=real_billing)
        executor = _build_executor_for_tests(
            credits,
            MagicMock(return_value="ok"),
            MagicMock(),
            billing_factory=billing_factory,
        )
        set_task_executor(executor)

        _fire_task("x1")

        assert captured_init_kwargs, "AIMessageBilling was not instantiated"
        init_kwargs = captured_init_kwargs[0]
        assert init_kwargs["chat_type"] == "private"
        assert init_kwargs["numeric_chat_id"] is None


class TestFireTaskBillingUnavailable:
    """Characterization tests: what happens when billing is unavailable."""

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_billing_unavailable_skips_one_shot_task(self, mock_sched):
        """When billing is not configured, one-shot tasks are deleted without calling AI."""
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
        set_redis_client(redis_client)

        credits = MagicMock()
        credits.is_configured.return_value = False
        executor = _build_executor_for_tests(credits, MagicMock(), MagicMock())
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_billing_unavailable_skips_recurring_task(self, mock_sched):
        """Recurring tasks are kept (not deleted) when billing is unavailable."""
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
        set_redis_client(redis_client)

        credits = MagicMock()
        credits.is_configured.return_value = False
        executor = _build_executor_for_tests(credits, MagicMock(), MagicMock())
        set_task_executor(executor)

        _fire_task("x1")

        redis_client.delete.assert_not_called()


class TestFireTaskAIFailure:
    """Characterization tests: what happens when ask_ai raises an exception."""

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_ask_ai_exception_reports_to_admin_and_does_not_send(self, mock_sched):
        """When ask_ai throws, admin is notified and no message is sent."""
        from api.tools.task_scheduler import _fire_task
        from api.task_executor import TaskExecutor

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "buscar info",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_ok_mock()
        mock_ask_ai = MagicMock(side_effect=Exception("network error"))
        mock_send = MagicMock()
        admin_report = MagicMock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        billing_factory = MagicMock(return_value=billing)

        executor = TaskExecutor(
            ask_ai=mock_ask_ai,
            send_msg=mock_send,
            admin_report=admin_report,
            credits_db_service=credits,
            gen_random_fn=MagicMock(),
            build_insufficient_credits_message_fn=MagicMock(),
            billing_factory=billing_factory,
        )
        set_task_executor(executor)

        _fire_task("x1")

        admin_report.assert_called_once()
        call_args = admin_report.call_args[0]
        assert "ask_ai" in call_args[0].lower() or "x1" in call_args[0]
        assert isinstance(call_args[1], Exception)
        mock_send.assert_not_called()
        redis_client.delete.assert_called_once_with(f"{TASK_REDIS_PREFIX}x1")

    @patch("api.tools.task_scheduler.get_scheduler")
    def test_ask_ai_exception_refunds_reserved_credits(self, mock_sched):
        """When ask_ai throws, reserved credits are refunded and never settled."""
        from api.tools.task_scheduler import _fire_task

        redis_client = _make_redis_with_task(
            {
                "chat_id": "123",
                "text": "info",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
            }
        )
        set_redis_client(redis_client)

        credits = _credits_ok_mock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        billing_factory = MagicMock(return_value=billing)
        executor = _build_executor_for_tests(
            credits,
            MagicMock(side_effect=Exception("error")),
            MagicMock(),
            billing_factory=billing_factory,
        )
        set_task_executor(executor)

        _fire_task("x1")

        billing.settle_reserved_ai_credits.assert_not_called()
        billing.refund_reserved_ai_credits.assert_called_once_with(
            {"reservation": "ok"}, reason="task_error"
        )


# ---------------------------------------------------------------------------
# _fire_task -- billing / auto-deletion rules
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

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_lists_tasks_with_stored_timezone_offset(self, mock_sched, mock_redis):
        mock_sched.return_value = None

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_data = {
            "id": "abc1",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "interval_seconds": None,
            "run_date": "2026-04-15T05:00:00+00:00",
            "timezone_offset": -5,
        }
        redis_client.scan_iter.return_value = [f"{TASK_REDIS_PREFIX}abc1"]
        redis_client.get.return_value = json.dumps(task_data)

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["next_run"] == "15/04 00:00"

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_lists_trigger_config_for_cron_tasks(self, mock_sched, mock_redis):
        mock_sched.return_value = None

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_data = {
            "id": "abc1",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "interval_seconds": None,
            "trigger_config": {"type": "cron", "hour": 20, "minute": 30},
            "run_date": None,
        }
        redis_client.scan_iter.return_value = [f"{TASK_REDIS_PREFIX}abc1"]
        redis_client.get.return_value = json.dumps(task_data)

        tasks = list_tasks("123")

        assert len(tasks) == 1
        assert tasks[0]["trigger_config"] == {"type": "cron", "hour": 20, "minute": 30}


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

    @patch("api.tools.task_scheduler._get_redis")
    @patch("api.tools.task_scheduler.get_scheduler")
    def test_stores_timezone_offset(self, mock_sched, mock_redis):
        scheduler = MagicMock()
        mock_sched.return_value = scheduler

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        task_id = schedule_task("123", "test", delay_seconds=300, timezone_offset=-5)

        assert task_id is not None
        stored_call = redis_client.setex.call_args
        stored_data = json.loads(stored_call[0][2])
        assert stored_data["timezone_offset"] == -5


def test_get_scheduler_uses_integer_misfire_grace_time(monkeypatch):
    shutdown_scheduler()
    monkeypatch.setenv("REDIS_PORT", "6379")

    with (
        patch("apscheduler.jobstores.redis.RedisJobStore") as mock_jobstore,
        patch(
            "apscheduler.schedulers.background.BackgroundScheduler"
        ) as mock_scheduler,
    ):
        scheduler = MagicMock()
        mock_scheduler.return_value = scheduler

        result = get_scheduler()

    assert result is scheduler
    mock_jobstore.assert_called_once()
    assert mock_scheduler.call_args.kwargs["job_defaults"]["misfire_grace_time"] == 300
    shutdown_scheduler()


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
