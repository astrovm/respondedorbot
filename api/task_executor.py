"""Execute scheduled AI tasks."""

from __future__ import annotations

import concurrent.futures
import json
from typing import Any, Callable, Dict, List, Mapping, Tuple

from api.ai_billing import AIMessageBilling
from api.ai_pipeline import (
    clean_duplicate_response,
    remove_gordo_prefix,
    strip_markdown_formatting,
)
from api.logging_config import get_logger

logger = get_logger(__name__)

_MAX_FALLBACK_RETRIES = 1
_MAX_EMPTY_RETRIES = 1

_TASK_FORMATTING_INSTRUCTIONS = "\n\n" + "\n".join(
    [
        "INSTRUCCIONES:",
        "- mantené el personaje del gordo",
        "- usá lenguaje coloquial argentino",
        "- respondé en minúsculas, sin emojis, sin punto final",
        "- si la respuesta tiene varios temas, usá listas con viñetas, párrafos cortos y saltos de línea entre secciones",
        "- no la pongas toda en una sola frase: estructurala para que sea fácil de leer",
    ]
)


def _clean_task_response(response: str) -> str:
    response = remove_gordo_prefix(response)
    response = clean_duplicate_response(response)
    response = strip_markdown_formatting(response)
    return response.strip()


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
        estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]],
        billing_factory: Callable[..., AIMessageBilling] = AIMessageBilling,
        max_workers: int = 5,
    ) -> None:
        self._ask_ai = ask_ai
        self._send_msg = send_msg
        self._admin_report = admin_report
        self._credits_db_service = credits_db_service
        self._gen_random_fn = gen_random_fn
        self._build_insufficient_credits_message_fn = (
            build_insufficient_credits_message_fn
        )
        self._estimate_ai_base_reserve_credits = estimate_ai_base_reserve_credits
        self._billing_factory = billing_factory
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="task",
        )

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
            logger.warning("task %s missing chat_id or text", task_id)
            return False

        if not user_name:
            logger.warning("task %s missing user_name, skipping", task_id)
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

        messages = [{"role": "user", "content": text + _TASK_FORMATTING_INSTRUCTIONS}]
        response_meta: dict[str, Any] = {}

        reserve_credits, reserve_meta = self._estimate_ai_base_reserve_credits(
            messages=messages,
        )
        charge_meta, charge_error = billing.reserve_ai_credits(
            "task_ai",
            reserve_credits,
            metadata={"task_id": task_id, "chat_id": chat_id, **reserve_meta},
        )
        if charge_error:
            logger.info("task %s no credits, skipping: %s", task_id, charge_error)
            return should_delete

        fallback_retries = 0
        empty_retries = 0

        while True:
            try:
                logger.info("task %s calling ask_ai", task_id)
                response = self._ask_ai(
                    messages,
                    response_meta=response_meta,
                    enable_web_search=True,
                    chat_id=chat_id,
                    user_name=user_name,
                    user_id=user_id,
                    task_mode=True,
                )

                if not response or not response.strip():
                    if empty_retries < _MAX_EMPTY_RETRIES:
                        empty_retries += 1
                        logger.warning(
                            "task %s empty response, retry %d/%d",
                            task_id, empty_retries, _MAX_EMPTY_RETRIES,
                        )
                        messages.append(
                            {"role": "system", "content": "respondé la tarea, es obligatorio."}
                        )
                        continue
                    logger.warning("task %s empty after retries", task_id)
                    billing.refund_reserved_ai_credits(charge_meta, reason="task_empty")
                    return should_delete

                is_fallback = response_meta.get("ai_fallback", False)
                if is_fallback and fallback_retries < _MAX_FALLBACK_RETRIES:
                    fallback_retries += 1
                    logger.warning(
                        "task %s fallback, retry %d/%d",
                        task_id, fallback_retries, _MAX_FALLBACK_RETRIES,
                    )
                    messages.append(
                        {"role": "system", "content": "tenés que responder. no hay opcion de no responder."}
                    )
                    continue

                response = _clean_task_response(response)
                self._send_msg(chat_id, f"{display}, tarea «{text}»:\n{response}")
                logger.info("task %s completed successfully", task_id)

                if is_fallback:
                    billing.refund_reserved_ai_credits(charge_meta, reason="task_fallback")
                else:
                    segments = list(response_meta.get("billing_segments") or [])
                    billing.settle_reserved_ai_credits(
                        charge_meta,
                        segments,
                        reason="task_success",
                    )
                return should_delete

            except Exception as e:
                is_json_error = isinstance(e, json.JSONDecodeError) or (
                    "JSONDecodeError" in type(e).__name__
                )
                if is_json_error:
                    logger.warning(
                        "task %s ask_ai JSONDecodeError (upstream transient), skipping",
                        task_id,
                    )
                else:
                    logger.error("task %s ask_ai failed: %s", task_id, e, exc_info=True)
                    self._admin_report(
                        f"task_scheduler {task_id} ask_ai error", e, {"chat_id": chat_id}
                    )
                billing.refund_reserved_ai_credits(
                    charge_meta, reason="task_json_error" if is_json_error else "task_error"
                )
                return should_delete

    def execute_many(self, tasks: List[Mapping[str, Any]]) -> List[bool]:
        """Execute multiple tasks in parallel, returning delete flags."""
        if not tasks:
            return []
        futures = {
            self._pool.submit(self.execute, task): task
            for task in tasks
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            task_id = task.get("id", "")
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("task %s parallel execution error: %s", task_id, e, exc_info=True)
                results.append(False)
        return results


def build_task_executor(
    *,
    ask_ai: Callable[..., str],
    send_msg: Callable[..., Any],
    admin_report: Callable[..., None],
    credits_db_service: Any,
    gen_random_fn: Callable[[str], str],
    build_insufficient_credits_message_fn: Callable[..., str],
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]],
    billing_factory: Callable[..., AIMessageBilling] = AIMessageBilling,
    max_workers: int = 5,
) -> TaskExecutor:
    return TaskExecutor(
        ask_ai=ask_ai,
        send_msg=send_msg,
        admin_report=admin_report,
        credits_db_service=credits_db_service,
        gen_random_fn=gen_random_fn,
        build_insufficient_credits_message_fn=build_insufficient_credits_message_fn,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        billing_factory=billing_factory,
        max_workers=max_workers,
    )
