"""Tests for scheduled task execution."""

from __future__ import annotations

from unittest.mock import MagicMock

from api.task_executor import TaskExecutor


def _build_executor(
    *, ask_ai_return_value: str
) -> tuple[TaskExecutor, MagicMock, MagicMock, MagicMock]:
    ask_ai = MagicMock(return_value=ask_ai_return_value)
    send_msg = MagicMock()
    admin_report = MagicMock()
    billing = MagicMock()
    billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
    billing_factory = MagicMock(return_value=billing)
    estimate_ai_base_reserve_credits = MagicMock(return_value=(10, {}))

    executor = TaskExecutor(
        ask_ai=ask_ai,
        send_msg=send_msg,
        admin_report=admin_report,
        credits_db_service=MagicMock(),
        gen_random_fn=MagicMock(),
        build_insufficient_credits_message_fn=MagicMock(),
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        billing_factory=billing_factory,
    )

    return executor, billing, ask_ai, send_msg


class TestTaskExecutor:
    def test_sends_scheduled_ai_message(self):
        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value="hola"
        )

        task = {
            "id": "abc123",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": None,
        }

        should_delete = executor.execute(task)

        assert should_delete is True
        ask_ai.assert_called_once()
        send_msg.assert_called_once_with("123", "astro, tarea programada: hola")
        billing.settle_reserved_ai_credits.assert_called_once()

    def test_passes_stored_task_text_as_is(self):
        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value="hola"
        )

        task = {
            "id": "abc123",
            "chat_id": "123",
            "text": "decime cuanta aura farmeaste hoy",
            "user_name": "astro",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": {"type": "cron", "hour": 20, "minute": 30},
        }

        executor.execute(task)

        ask_ai.assert_called_once()
        sent_prompt = ask_ai.call_args.args[0][0]["content"]
        assert sent_prompt == "decime cuanta aura farmeaste hoy"

    def test_refunds_reserved_credits_on_fallback(self):
        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value="[[AI_FALLBACK]]respuesta"
        )

        task = {
            "id": "abc123",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": None,
        }

        should_delete = executor.execute(task)

        assert should_delete is True
        assert ask_ai.call_count == 2
        send_msg.assert_called_once_with("123", "astro, tarea programada: respuesta")
        billing.refund_reserved_ai_credits.assert_called_once()
        billing.settle_reserved_ai_credits.assert_not_called()

    def test_refunds_reserved_credits_when_ask_ai_raises(self):
        ask_ai = MagicMock(side_effect=RuntimeError("boom"))
        send_msg = MagicMock()
        admin_report = MagicMock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        billing_factory = MagicMock(return_value=billing)
        estimate_ai_base_reserve_credits = MagicMock(return_value=(10, {}))

        executor = TaskExecutor(
            ask_ai=ask_ai,
            send_msg=send_msg,
            admin_report=admin_report,
            credits_db_service=MagicMock(),
            gen_random_fn=MagicMock(),
            build_insufficient_credits_message_fn=MagicMock(),
            estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
            billing_factory=billing_factory,
        )

        task = {
            "id": "abc123",
            "chat_id": "123",
            "text": "recordame algo",
            "user_name": "astro",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": None,
        }

        should_delete = executor.execute(task)

        assert should_delete is True
        admin_report.assert_called_once()
        send_msg.assert_not_called()
        billing.refund_reserved_ai_credits.assert_called_once_with(
            {"reservation": "ok"}, reason="task_error"
        )
        billing.settle_reserved_ai_credits.assert_not_called()
