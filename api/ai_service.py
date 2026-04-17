"""AI conversation orchestration service.

Encapsulates the AI response lifecycle: credit reservation, model calls,
media context handling, fallback detection, and billing settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.ai_pricing import IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE


@dataclass(frozen=True)
class AIService:
    credits_db_service: Any
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]]
    build_ai_messages: Callable[..., List[Dict[str, Any]]]
    check_provider_available: Callable[..., bool]
    has_openrouter_fallback: Callable[[], bool]
    handle_rate_limit: Callable[[str, Dict[str, Any]], str]
    handle_ai_response: Callable[..., str]
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]]
    estimate_image_context_reserve_credits: Callable[[bytes, str], int]

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
        ai_messages = self.build_ai_messages(
            request.message,
            chat_history,
            request.prompt_text,
            request.reply_context_text,
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
        )

        billing_segments = list(ai_response_meta.get("billing_segments") or [])
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


def build_ai_service(
    *,
    credits_db_service: Any,
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]],
    build_ai_messages: Callable[..., List[Dict[str, Any]]],
    check_provider_available: Callable[..., bool],
    has_openrouter_fallback: Callable[[], bool],
    handle_rate_limit: Callable[[str, Dict[str, Any]], str],
    handle_ai_response: Callable[..., str],
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]],
    estimate_image_context_reserve_credits: Callable[[bytes, str], int],
) -> AIService:
    return AIService(
        credits_db_service=credits_db_service,
        get_chat_history=get_chat_history,
        build_ai_messages=build_ai_messages,
        check_provider_available=check_provider_available,
        has_openrouter_fallback=has_openrouter_fallback,
        handle_rate_limit=handle_rate_limit,
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=estimate_ai_base_reserve_credits,
        estimate_image_context_reserve_credits=estimate_image_context_reserve_credits,
    )
