"""AI conversation orchestration service.

Encapsulates the AI response lifecycle: credit reservation, model calls,
media context handling, fallback detection, and billing settlement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.ai_pricing import IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE, credit_units_from_usd_micros


_summary_logger = logging.getLogger(__name__)


def _make_summary_billing_segment(cost_usd_micros: int) -> Dict[str, Any]:
    return {
        "kind": "summary",
        "text": "context compaction",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "billing": {
            "raw_usd_micros": cost_usd_micros,
            "charged_credit_units": credit_units_from_usd_micros(cost_usd_micros),
        },
    }


@dataclass(frozen=True)
class AIService:
    credits_db_service: Any
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]]
    prepare_chat_memory: Callable[..., Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]], int]]
    build_ai_messages: Callable[..., List[Dict[str, Any]]]
    check_provider_available: Callable[..., bool]
    has_openrouter_fallback: Callable[[], bool]
    handle_rate_limit: Callable[[str, Dict[str, Any]], str]
    handle_ai_response: Callable[..., str]
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]]
    estimate_image_context_reserve_credits: Callable[[bytes, str], int]
    stream_summary_command: Callable[[str, Any, str], Any]

    def run_conversation(
        self,
        request: AIConversationRequest,
    ) -> Tuple[str, bool]:
        if not self.credits_db_service.is_configured():
            billing_unavailable = (
                self.handle_rate_limit(request.chat_id, request.message),
                False,
            )
            return ("ok", False) if request.is_spontaneous else billing_unavailable

        chat_history = self.get_chat_history(request.chat_id, request.redis_client)
        reply_to_message = request.message.get("reply_to_message") if request.message else None
        reply_to_message_id = None
        if isinstance(reply_to_message, dict):
            raw_reply_to_id = reply_to_message.get("message_id")
            if raw_reply_to_id is not None:
                reply_to_message_id = str(raw_reply_to_id)
        visible_history, summary_text, retrieved_messages, summary_cost = self.prepare_chat_memory(
            request.redis_client,
            request.chat_id,
            chat_history,
            request.prompt_text,
            reply_to_message_id=reply_to_message_id,
            compaction_threshold=request.compaction_threshold,
            compaction_keep=request.compaction_keep,
        )
        ai_messages = self.build_ai_messages(
            request.message,
            visible_history,
            request.prompt_text,
            request.reply_context_text,
            summary_text=summary_text,
            retrieved_messages=retrieved_messages,
        )

        main_reserve_credits, reserve_meta = self.estimate_ai_base_reserve_credits(
            ai_messages,
            extra_input_tokens=(
                IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE
                if request.prepared_message.resized_image_data
                and request.prepared_message.photo_file_id
                else 0
            ),
        )
        if (
            not self.check_provider_available(scope="chat")
            and not self.has_openrouter_fallback()
        ):
            rate_limit_msg = self.handle_rate_limit(request.chat_id, request.message)
            return ("ok", False) if request.is_spontaneous else (rate_limit_msg, False)
        base_charge_meta, base_charge_error = request.billing_helper.reserve_ai_credits(
            "ai_response_base",
            main_reserve_credits,
            metadata={
                "estimated_prompt_messages": len(ai_messages),
                "summary_cost": summary_cost,
                **reserve_meta,
            },
        )
        if base_charge_error:
            return (
                ("ok", False) if request.is_spontaneous else (base_charge_error, False)
            )

        media_charge_meta: Optional[Dict[str, Any]] = None
        if (
            request.prepared_message.resized_image_data
            and request.prepared_message.photo_file_id
        ):
            image_prompt = "Describe what you see in this image in detail."
            if (
                not self.check_provider_available(scope="vision")
                and not self.has_openrouter_fallback()
            ):
                request.billing_helper.refund_reserved_ai_credits(
                    base_charge_meta, reason="image_context_local_rate_limit"
                )
                rate_limit_msg = self.handle_rate_limit(
                    request.chat_id, request.message
                )
                return (
                    ("ok", False) if request.is_spontaneous else (rate_limit_msg, False)
                )
            media_charge_meta, media_charge_error = (
                request.billing_helper.reserve_ai_credits(
                    "image_context_media",
                    self.estimate_image_context_reserve_credits(
                        request.prepared_message.resized_image_data,
                        image_prompt,
                    ),
                    metadata={"photo_file_id": request.prepared_message.photo_file_id},
                )
            )
            if media_charge_error:
                request.billing_helper.refund_reserved_ai_credits(
                    base_charge_meta, reason="image_context_reserve_failed"
                )
                return (
                    ("ok", False)
                    if request.is_spontaneous
                    else (media_charge_error, False)
                )

        ai_response_meta: Dict[str, Any] = {}
        response_msg = self.handle_ai_response(
            request.chat_id,
            request.handler_func,
            ai_messages,
            image_data=(
                request.prepared_message.resized_image_data
                if request.prepared_message.photo_file_id
                else None
            ),
            image_file_id=request.prepared_message.photo_file_id,
            context_texts=[request.reply_context_text],
            user_identity=request.user_identity,
            response_meta=ai_response_meta,
            user_id=request.user_id,
            timezone_offset=request.timezone_offset,
            reply_to_message_id=request.reply_to_message_id,
        )

        billing_segments = list(ai_response_meta.get("billing_segments") or [])
        if summary_cost > 0:
            billing_segments.insert(0, _make_summary_billing_segment(summary_cost))
        if (
            response_msg == "me quedé reculando y no te pude responder, probá de nuevo"
            or bool(ai_response_meta.get("ai_fallback"))
        ):
            if media_charge_meta:
                request.billing_helper.refund_reserved_ai_credits(
                    media_charge_meta, reason="ai_response_fallback"
                )
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="ai_response_fallback"
            )
            return response_msg, True

        settlement_reservations: List[Optional[Dict[str, Any]]] = [base_charge_meta]
        if media_charge_meta:
            settlement_reservations.append(media_charge_meta)

        request.billing_helper.settle_reserved_ai_credits_batch(
            settlement_reservations,
            billing_segments,
            reason="ai_response_success",
        )

        return response_msg, True

    def run_summary_command_stream(
        self,
        request: SummaryCommandRequest,
        stream_consumer: Callable[[Any], str],
    ) -> Tuple[str, Optional[str], bool]:
        if not self.credits_db_service.is_configured():
            return self.handle_rate_limit(request.chat_id, request.message), None, True

        if (
            not self.check_provider_available(scope="chat")
            and not self.has_openrouter_fallback()
        ):
            return self.handle_rate_limit(request.chat_id, request.message), None, True

        token_iterator, pending_marker = self.stream_summary_command(
            request.chat_id,
            request.redis_client,
            request.prompt_text,
        )

        main_reserve_credits, reserve_meta = self.estimate_ai_base_reserve_credits(
            [{"role": "user", "content": "summary"}],
            extra_input_tokens=0,
        )
        base_charge_meta, base_charge_error = request.billing_helper.reserve_ai_credits(
            "ai_response_base",
            main_reserve_credits,
            metadata={"estimated_prompt_messages": 1, **reserve_meta},
        )
        if base_charge_error:
            return base_charge_error, None, True

        try:
            final_text = stream_consumer(token_iterator)
        except Exception:
            _summary_logger.exception("summary_stream: failed for chat_id=%s", request.chat_id)
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="summary_stream_failed"
            )
            return "no pude generar el resumen", None, True

        request.billing_helper.settle_reserved_ai_credits_batch(
            [base_charge_meta],
            [],
            reason="summary_command_stream_success",
        )

        return final_text, pending_marker, False


@dataclass(frozen=True)
class AIConversationRequest:
    chat_id: str
    message: Dict[str, Any]
    user_id: Optional[int]
    prepared_message: Any
    billing_helper: Any
    prompt_text: str
    reply_context_text: Optional[str]
    user_identity: str
    handler_func: Callable[..., str]
    redis_client: Any
    timezone_offset: int = -3
    is_spontaneous: bool = False
    compaction_threshold: Optional[int] = None
    compaction_keep: Optional[int] = None
    reply_to_message_id: Optional[str] = None


@dataclass(frozen=True)
class SummaryCommandRequest:
    chat_id: str
    message: Dict[str, Any]
    billing_helper: Any
    prompt_text: str
    redis_client: Any


@dataclass(frozen=True)
class SummaryCommandResponse:
    text: str
    is_fallback: bool
    pending_summary: Optional[str]
    pending_marker: Optional[str]


def build_ai_service(
    *,
    credits_db_service: Any,
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]],
    prepare_chat_memory: Callable[..., Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]], int]],
    build_ai_messages: Callable[..., List[Dict[str, Any]]],
    check_provider_available: Callable[..., bool],
    has_openrouter_fallback: Callable[[], bool],
    handle_rate_limit: Callable[[str, Dict[str, Any]], str],
    handle_ai_response: Callable[..., str],
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]],
    estimate_image_context_reserve_credits: Callable[[bytes, str], int],
    stream_summary_command: Callable[[str, Any, str], Any] = lambda _a, _b, _c: (iter([]), None),
) -> AIService:
    return AIService(
        credits_db_service=credits_db_service,
        get_chat_history=get_chat_history,
        prepare_chat_memory=prepare_chat_memory,
        build_ai_messages=build_ai_messages,
        check_provider_available=check_provider_available,
        has_openrouter_fallback=has_openrouter_fallback,
        handle_rate_limit=handle_rate_limit,
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
        stream_summary_command=stream_summary_command,
    )
