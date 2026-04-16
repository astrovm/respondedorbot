"""Execute scheduled AI tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from api.ai_billing import AIMessageBilling


def _strip_response_marker(response: str) -> str:
    marker = "[[AI_FALLBACK]]"
    if response.startswith(marker):
        return response[len(marker) :].lstrip()
    return response


class TaskExecutor:
    def __init__(
        self,
        *,
        ask_ai: Callable[..., str],
        send_msg: Callable[..., Any],
        admin_report: Callable[..., None],
        credits_db_service: Any,
        gen_random_fn: Callable[[str], str],
        build_insufficient_credits_message_fn: Callable[..., str],
        billing_factory: Callable[..., AIMessageBilling] = AIMessageBilling,
    ) -> None:
        self._ask_ai = ask_ai
        self._send_msg = send_msg
        self._admin_report = admin_report
        self._credits_db_service = credits_db_service
        self._gen_random_fn = gen_random_fn
        self._build_insufficient_credits_message_fn = (
            build_insufficient_credits_message_fn
        )
        self._billing_factory = billing_factory

    def execute(self, task: Mapping[str, Any]) -> bool:
        task_id = str(task.get("id", ""))
        chat_id = str(task.get("chat_id", ""))
        text = str(task.get("text", ""))
        user_name = str(task.get("user_name", ""))
        interval = task.get("interval_seconds")
        trigger_cfg = task.get("trigger_config")
        user_id = task.get("user_id")
        should_delete = not interval and not trigger_cfg

        if not chat_id or not text:
            print(f"task_scheduler: {task_id} missing chat_id or text")
            return False

        if not user_name:
            print(f"task_scheduler: {task_id} missing user_name, skipping")
            return False

        display = user_name
        task_message = {"from": {"id": user_id}} if user_id else {}
        billing = self._billing_factory(
            credits_db_service=self._credits_db_service,
            admin_reporter=self._admin_report,
            gen_random_fn=self._gen_random_fn,
            build_insufficient_credits_message_fn=self._build_insufficient_credits_message_fn,
            maybe_grant_onboarding_credits_fn=lambda uid: None,
            command="task",
            chat_id=chat_id,
            chat_type="private",
            user_id=user_id,
            numeric_chat_id=None,
            message=task_message,
        )

        messages = [{"role": "user", "content": text}]
        response_meta: dict[str, Any] = {}
        is_fallback = False

        reserve_meta, reserve_error = billing.reserve_ai_credits(
            "task_ai",
            1000,
            metadata={"task_id": task_id, "chat_id": chat_id},
        )
        if reserve_error:
            print(f"task_scheduler: {task_id} no credits, skipping: {reserve_error}")
            return should_delete

        try:
            print(f"task_scheduler: {task_id} calling ask_ai...")
            response = self._ask_ai(
                messages,
                response_meta=response_meta,
                enable_web_search=True,
                chat_id=chat_id,
                user_name=user_name,
                user_id=user_id,
            )
            is_fallback = response.startswith("[[AI_FALLBACK]]")
            if response:
                response = _strip_response_marker(response)
                self._send_msg(chat_id, f"{display}, tarea programada: {response}")
                print(f"task_scheduler: {task_id} completed successfully")
        except Exception as e:
            print(f"task_scheduler: {task_id} ask_ai failed: {e}")
            self._admin_report(
                f"task_scheduler {task_id} ask_ai error", e, {"chat_id": chat_id}
            )
        else:
            if is_fallback:
                billing.refund_reserved_ai_credits(reserve_meta, reason="task_fallback")
            else:
                segments = list(response_meta.get("billing_segments") or [])
                billing.settle_reserved_ai_credits(
                    reserve_meta,
                    segments,
                    reason="task_success",
                )

        return should_delete


def build_task_executor(
    *,
    ask_ai: Callable[..., str],
    send_msg: Callable[..., Any],
    admin_report: Callable[..., None],
    credits_db_service: Any,
    gen_random_fn: Callable[[str], str],
    build_insufficient_credits_message_fn: Callable[..., str],
    billing_factory: Callable[..., AIMessageBilling] = AIMessageBilling,
) -> TaskExecutor:
    return TaskExecutor(
        ask_ai=ask_ai,
        send_msg=send_msg,
        admin_report=admin_report,
        credits_db_service=credits_db_service,
        gen_random_fn=gen_random_fn,
        build_insufficient_credits_message_fn=build_insufficient_credits_message_fn,
        billing_factory=billing_factory,
    )
