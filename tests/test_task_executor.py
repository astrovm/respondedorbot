"""Tests for scheduled task execution."""

from __future__ import annotations

from unittest.mock import MagicMock

from api.tasks.executor import TaskExecutor


class _Authorizer:
    def __init__(self, reservations, **_kwargs):
        self.reservations = [item for item in reservations if item]

    def __call__(self, *_args, **_kwargs):
        return None

    def record_provider_segment(self, _segment):
        return None

    def close(self):
        return None


def _configure_authorizer(billing: MagicMock) -> None:
    billing.create_authorizer.side_effect = _Authorizer


def _build_executor(
    *, ask_ai_return_value: str
) -> tuple[TaskExecutor, MagicMock, MagicMock, MagicMock]:
    ask_ai = MagicMock(return_value=ask_ai_return_value)
    send_msg = MagicMock()
    admin_report = MagicMock()
    billing = MagicMock()
    billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
    _configure_authorizer(billing)
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
    def test_sends_explanation_when_task_cannot_reserve_credits(self):
        executor, billing, ask_ai, send_msg = _build_executor(ask_ai_return_value="hola")
        billing.reserve_ai_credits.return_value = (
            None,
            "te quedaste seco de créditos ia, boludo.",
        )

        task = {
            "id": "abc123",
            "chat_id": "-100123",
            "text": "recordame algo",
            "user_name": "@testuser",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": None,
        }

        should_delete = executor.execute(task)

        assert should_delete is True
        ask_ai.assert_not_called()
        send_msg.assert_called_once_with(
            "-100123",
            "@testuser, no pude ejecutar la tarea «recordame algo»:\n"
            "te quedaste seco de créditos ia, boludo.",
        )

    def test_recurring_task_reports_credit_failure_and_is_kept(self):
        executor, billing, ask_ai, send_msg = _build_executor(ask_ai_return_value="hola")
        billing.reserve_ai_credits.return_value = (None, "saldo insuficiente")

        task = {
            "id": "abc123",
            "chat_id": "-100123",
            "text": "mandá las noticias",
            "user_name": "@testuser",
            "user_id": 77,
            "interval_seconds": 3600,
            "trigger_config": {"type": "interval", "seconds": 3600},
        }

        should_delete = executor.execute(task)

        assert should_delete is False
        ask_ai.assert_not_called()
        send_msg.assert_called_once_with(
            "-100123",
            "@testuser, no pude ejecutar la tarea «mandá las noticias»:\nsaldo insuficiente",
        )

    def test_recurring_task_uses_a_new_billing_identity_for_each_execution(self):
        executor, _billing, _ask_ai, _send_msg = _build_executor(
            ask_ai_return_value="hola"
        )
        task = {
            "id": "abc123",
            "chat_id": "-100123",
            "text": "mandá las noticias",
            "user_name": "@testuser",
            "user_id": 77,
            "interval_seconds": 3600,
            "trigger_config": {"type": "interval", "seconds": 3600},
        }

        executor.execute(task)
        executor.execute(task)

        factory_calls = executor._billing_factory.call_args_list
        first_message_id = factory_calls[0].kwargs["message"]["message_id"]
        second_message_id = factory_calls[1].kwargs["message"]["message_id"]
        assert first_message_id.startswith("abc123:")
        assert second_message_id.startswith("abc123:")
        assert first_message_id != second_message_id

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
        send_msg.assert_called_once_with("123", "astro, tarea «recordame algo»:\nhola")
        billing.settle_reserved_ai_credits.assert_called_once()

    def test_passes_stored_task_text_with_formatting_instructions(self):
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
        assert sent_prompt.startswith("decime cuanta aura farmeaste hoy")
        assert "INSTRUCCIONES:" in sent_prompt
        assert "usá lista numerada: 1., 2., 3." in sent_prompt
        assert "dejá una línea en blanco entre cada item numerado" in sent_prompt

    def test_refunds_reserved_credits_on_fallback(self):
        def _fallback_ask_ai(messages, response_meta=None, **_kwargs):
            if response_meta is not None:
                response_meta["ai_fallback"] = True
            return "respuesta"

        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value="respuesta"
        )
        ask_ai.side_effect = _fallback_ask_ai

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
        send_msg.assert_called_once_with("123", "astro, tarea «recordame algo»:\nrespuesta")
        billing.refund_reserved_ai_credits.assert_called_once()
        billing.settle_reserved_ai_credits.assert_not_called()

    def test_settles_provider_usage_before_terminal_fallback(self):
        segment = {"kind": "chat", "usage": {"cost": 0.001}}

        def _fallback_ask_ai(messages, response_meta=None, **_kwargs):
            if response_meta is not None:
                response_meta["ai_fallback"] = True
                response_meta.setdefault("billing_segments", []).append(segment)
            return "respuesta"

        executor, billing, ask_ai, _send_msg = _build_executor(ask_ai_return_value="respuesta")
        ask_ai.side_effect = _fallback_ask_ai

        executor.execute(
            {
                "id": "abc123",
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
                "trigger_config": None,
            }
        )

        billing.settle_reserved_ai_credits.assert_called_once_with(
            {"reservation": "ok"},
            [segment, segment],
            reason="task_fallback_provider_usage",
        )
        billing.refund_reserved_ai_credits.assert_not_called()

    def test_clears_fallback_state_between_task_attempts(self):
        fallback_segment = {"kind": "chat", "usage": {"cost": 0.001}}
        success_segment = {"kind": "chat", "usage": {"cost": 0.002}}
        call_count = 0

        def _ask_ai(messages, response_meta=None, **_kwargs):
            nonlocal call_count
            call_count += 1
            assert response_meta is not None
            if call_count == 1:
                response_meta["ai_fallback"] = True
                response_meta.setdefault("billing_segments", []).append(fallback_segment)
                return "fallback"
            response_meta.setdefault("billing_segments", []).append(success_segment)
            return "respuesta paga"

        executor, billing, ask_ai, send_msg = _build_executor(ask_ai_return_value="respuesta")
        ask_ai.side_effect = _ask_ai

        executor.execute(
            {
                "id": "abc123",
                "chat_id": "123",
                "text": "recordame algo",
                "user_name": "astro",
                "user_id": 77,
                "interval_seconds": None,
                "trigger_config": None,
            }
        )

        assert ask_ai.call_count == 2
        send_msg.assert_called_once_with("123", "astro, tarea «recordame algo»:\nrespuesta paga")
        billing.settle_reserved_ai_credits.assert_called_once_with(
            {"reservation": "ok"},
            [fallback_segment, success_segment],
            reason="task_success",
        )
        billing.refund_reserved_ai_credits.assert_not_called()

    def test_refunds_reserved_credits_when_ask_ai_raises(self):
        ask_ai = MagicMock(side_effect=RuntimeError("boom"))
        send_msg = MagicMock()
        admin_report = MagicMock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        _configure_authorizer(billing)
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

    def test_does_not_admin_report_on_json_decode_error(self):
        import json

        ask_ai = MagicMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        send_msg = MagicMock()
        admin_report = MagicMock()
        billing = MagicMock()
        billing.reserve_ai_credits.return_value = ({"reservation": "ok"}, None)
        _configure_authorizer(billing)
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
        admin_report.assert_not_called()
        send_msg.assert_not_called()
        billing.refund_reserved_ai_credits.assert_called_once_with(
            {"reservation": "ok"}, reason="task_json_error"
        )

    def test_strips_markdown_from_task_response(self):
        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value="**hola**\n## titulo"
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
        send_msg.assert_called_once_with("123", "astro, tarea «recordame algo»:\nhola\ntitulo")

    def test_preserves_spacing_between_numbered_task_items(self):
        executor, billing, ask_ai, send_msg = _build_executor(
            ask_ai_return_value=(
                "1. noticia uno\n"
                "detalle uno\n\n"
                "2. noticia dos\n"
                "detalle dos"
            )
        )

        task = {
            "id": "abc123",
            "chat_id": "123",
            "text": "dame una lista",
            "user_name": "astro",
            "user_id": 77,
            "interval_seconds": None,
            "trigger_config": None,
        }

        should_delete = executor.execute(task)

        assert should_delete is True
        send_msg.assert_called_once_with(
            "123",
            "astro, tarea «dame una lista»:\n"
            "1. noticia uno\n"
            "detalle uno\n\n"
            "2. noticia dos\n"
            "detalle dos",
        )
