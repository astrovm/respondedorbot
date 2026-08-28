"""AI conversation orchestration service.

Encapsulates the AI response lifecycle: credit reservation, model calls,
media context handling, fallback detection, and billing settlement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from api.ai.pricing import IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE
from api.billing.authorization import AI_COST_AUTHORIZER_KEY, AI_SEGMENT_RECORDER_KEY
from api.i18n import tr


_summary_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIService:
    credits_db_service: Any
    get_chat_history: Callable[[str, Any], List[Dict[str, Any]]]
    prepare_chat_memory: Callable[..., Any]
    build_ai_messages: Callable[..., List[Dict[str, Any]]]
    check_provider_available: Callable[..., bool]
    has_openrouter_fallback: Callable[[], bool]
    handle_rate_limit: Callable[[str, Dict[str, Any]], str]
    handle_ai_response: Callable[..., str]
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]]
    estimate_image_context_reserve_credits: Callable[[bytes, str], int]
    stream_summary_command: Callable[..., Any]
    schedule_compaction: Callable[[Any, Any], bool]

    @staticmethod
    def _refund_reservations(
        request: AIConversationRequest,
        reservations: Sequence[Optional[Mapping[str, Any]]],
        *,
        reason: str,
    ) -> None:
        for reservation in reservations:
            if reservation:
                request.billing_helper.refund_reserved_ai_credits(reservation, reason=reason)

    @staticmethod
    def _settle_incurred_media(
        request: AIConversationRequest,
        reservation: Optional[Dict[str, Any]],
        billing_segments: List[Mapping[str, Any]],
        *,
        reason: str,
    ) -> None:
        if not reservation:
            return
        if billing_segments:
            request.billing_helper.settle_reserved_ai_credits_batch(
                [reservation],
                billing_segments,
                reason=reason,
            )
            return
        request.billing_helper.refund_reserved_ai_credits(reservation, reason=reason)

    def _prepare_conversation_messages(
        self, request: AIConversationRequest
    ) -> Tuple[List[Dict[str, Any]], Any]:
        chat_history = self.get_chat_history(request.chat_id, request.redis_client)
        reply_to_message = request.message.get("reply_to_message") if request.message else None
        reply_to_message_id = None
        if isinstance(reply_to_message, dict):
            raw_reply_to_id = reply_to_message.get("message_id")
            if raw_reply_to_id is not None:
                reply_to_message_id = str(raw_reply_to_id)
        prepared = self.prepare_chat_memory(
            request.redis_client,
            request.chat_id,
            chat_history,
            request.prompt_text,
            reply_to_message_id=reply_to_message_id,
            compaction_threshold=request.compaction_threshold,
            compaction_keep=request.compaction_keep,
        )
        visible_history, summary_text, retrieved_messages = prepared[:3]
        compaction_plan = prepared[4] if len(prepared) > 4 else None
        messages = self.build_ai_messages(
            request.message,
            visible_history,
            request.prompt_text,
            request.reply_context_text,
            summary_text=summary_text,
            retrieved_messages=retrieved_messages,
            timezone_offset=request.timezone_offset,
        )
        return messages, compaction_plan

    def _reserve_image_context(
        self,
        request: AIConversationRequest,
        existing_charge_meta: Optional[Dict[str, Any]],
        reservations_to_refund: Sequence[Optional[Mapping[str, Any]]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[str, bool]]]:
        if (
            existing_charge_meta
            or not request.prepared_message.resized_image_data
            or not request.prepared_message.photo_file_id
        ):
            return existing_charge_meta, None

        if not self.check_provider_available(scope="vision") and not self.has_openrouter_fallback():
            self._refund_reservations(
                request,
                reservations_to_refund,
                reason="image_context_local_rate_limit",
            )
            rate_limit_msg = self.handle_rate_limit(request.chat_id, request.message)
            response = "ok" if request.is_spontaneous else rate_limit_msg
            return None, (response, False)

        image_prompt = "Describe what you see in this image in detail."
        media_charge_meta, media_charge_error = request.billing_helper.reserve_ai_credits(
            "image_context_media",
            self.estimate_image_context_reserve_credits(
                request.prepared_message.resized_image_data,
                image_prompt,
            ),
            metadata={
                "operation_id": request.billing_helper.operation_id("ai_response"),
                "photo_file_id": request.prepared_message.photo_file_id,
            },
        )
        if not media_charge_error:
            return media_charge_meta, None

        self._refund_reservations(
            request,
            reservations_to_refund,
            reason="image_context_reserve_failed",
        )
        response = "ok" if request.is_spontaneous else media_charge_error
        return None, (response, False)

    def _estimate_full_conversation_reserve(
        self,
        request: AIConversationRequest,
        messages: List[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        full_reserve_credits, reserve_meta = self.estimate_ai_base_reserve_credits(
            messages,
            extra_input_tokens=(
                IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE if request.prepared_message.photo_file_id else 0
            ),
            timezone_offset=request.timezone_offset,
        )
        return full_reserve_credits, {
            "estimated_prompt_messages": len(messages),
            "required_reserve_credits": full_reserve_credits,
            **reserve_meta,
        }

    @staticmethod
    def _start_billing_session(
        request: AIConversationRequest,
        reservations: List[Optional[Dict[str, Any]]],
        initial_segments: List[Mapping[str, Any]],
        model_reserve_credit_units: int,
    ) -> tuple[Any, Dict[str, Any]]:
        session = request.billing_helper.create_authorizer(
            reservations,
            model_reserve_credit_units=model_reserve_credit_units,
        )
        for segment in initial_segments:
            session.record_provider_segment(segment)
        return session, {
            AI_COST_AUTHORIZER_KEY: session,
            AI_SEGMENT_RECORDER_KEY: session.record_provider_segment,
            "billing_segments": initial_segments,
        }

    def _finish_conversation_session(
        self,
        request: AIConversationRequest,
        ai_messages: List[Dict[str, Any]],
        compaction_plan: Any,
        authorizer: Any,
        response_meta: Dict[str, Any],
    ) -> Tuple[str, bool]:
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
            response_meta=response_meta,
            user_id=request.user_id,
            timezone_offset=request.timezone_offset,
            reply_to_message_id=request.reply_to_message_id,
        )

        response_meta.pop(AI_COST_AUTHORIZER_KEY, None)
        response_meta.pop(AI_SEGMENT_RECORDER_KEY, None)
        billing_segments = list(response_meta.get("billing_segments") or [])
        reservations = list(authorizer.reservations)

        if bool(response_meta.get("ai_fallback")):
            if billing_segments:
                request.billing_helper.settle_reserved_ai_credits_batch(
                    reservations,
                    billing_segments,
                    reason="ai_response_provider_usage_before_fallback",
                )
            else:
                self._refund_reservations(
                    request,
                    reservations,
                    reason="ai_response_fallback",
                )
            return response_msg, True

        request.billing_helper.settle_reserved_ai_credits_batch(
            reservations,
            billing_segments,
            reason="ai_response_success",
        )
        try:
            self.schedule_compaction(compaction_plan, request.billing_helper)
        except Exception:
            # Compaction is maintenance. It must not turn a successful answer
            # into an error or add a model call to the foreground path.
            _summary_logger.exception(
                "compaction: failed to schedule chat_id=%s",
                request.chat_id,
            )
        return response_msg, True

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

        admission_messages = [{"role": "user", "content": request.prompt_text}]
        media_charge_meta: Optional[Dict[str, Any]] = (
            dict(request.prepared_message.media_charge_meta)
            if getattr(request.prepared_message, "media_charge_meta", None)
            else None
        )
        media_billing_segments = list(request.prepared_message.media_billing_segments)
        main_reserve_credits, reserve_meta = self.estimate_ai_base_reserve_credits(
            admission_messages,
            extra_input_tokens=(
                IMAGE_CONTEXT_EXTRA_TOKENS_ESTIMATE if request.prepared_message.photo_file_id else 0
            ),
            timezone_offset=request.timezone_offset,
        )
        base_charge_meta, base_charge_error = request.billing_helper.reserve_ai_credits(
            "ai_response_base",
            main_reserve_credits,
            metadata={
                "operation_id": request.billing_helper.operation_id("ai_response"),
                "estimated_prompt_messages": len(admission_messages),
                **reserve_meta,
            },
        )
        if base_charge_error:
            self._settle_incurred_media(
                request,
                media_charge_meta,
                media_billing_segments,
                reason="ai_response_base_reserve_failed",
            )
            return ("ok", False) if request.is_spontaneous else (base_charge_error, False)

        if not self.check_provider_available(scope="chat") and not self.has_openrouter_fallback():
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="chat_provider_unavailable"
            )
            self._settle_incurred_media(
                request,
                media_charge_meta,
                media_billing_segments,
                reason="chat_provider_unavailable",
            )
            rate_limit_msg = self.handle_rate_limit(request.chat_id, request.message)
            return ("ok", False) if request.is_spontaneous else (rate_limit_msg, False)

        try:
            ai_messages, compaction_plan = self._prepare_conversation_messages(request)
        except Exception:
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="ai_request_preparation_failed"
            )
            self._settle_incurred_media(
                request,
                media_charge_meta,
                media_billing_segments,
                reason="ai_request_preparation_failed",
            )
            raise

        full_reserve_credits, full_reserve_meta = self._estimate_full_conversation_reserve(
            request,
            ai_messages,
        )
        context_charge_meta: Optional[Dict[str, Any]] = None
        if full_reserve_credits > main_reserve_credits:
            context_charge_meta, base_charge_error = request.billing_helper.reserve_ai_credits(
                "ai_response_context_extension",
                full_reserve_credits - main_reserve_credits,
                metadata={
                    "operation_id": request.billing_helper.operation_id("ai_response"),
                    **full_reserve_meta,
                },
            )
            if base_charge_error:
                request.billing_helper.refund_reserved_ai_credits(
                    base_charge_meta,
                    reason="ai_response_reserve_adjustment_failed",
                )
                self._settle_incurred_media(
                    request,
                    media_charge_meta,
                    media_billing_segments,
                    reason="ai_response_reserve_adjustment_failed",
                )
                response = "ok" if request.is_spontaneous else base_charge_error
                return response, False

        media_charge_meta, image_error = self._reserve_image_context(
            request,
            media_charge_meta,
            [base_charge_meta, context_charge_meta],
        )
        if image_error:
            return image_error

        settlement_reservations: List[Optional[Dict[str, Any]]] = [
            base_charge_meta,
            context_charge_meta,
        ]
        if media_charge_meta:
            settlement_reservations.append(media_charge_meta)
        authorizer, ai_response_meta = self._start_billing_session(
            request,
            settlement_reservations,
            media_billing_segments,
            sum(
                int(reservation.get("reserved_credit_units", 0) or 0)
                for reservation in (base_charge_meta, context_charge_meta)
                if reservation
            ),
        )
        try:
            return self._finish_conversation_session(
                request,
                ai_messages,
                compaction_plan,
                authorizer,
                ai_response_meta,
            )
        finally:
            authorizer.close()

    @staticmethod
    def _finish_summary_non_success(
        request: SummaryCommandRequest,
        authorizer: Any,
        response_meta: Mapping[str, Any],
        final_text: str,
        pending_marker: Optional[str],
    ) -> Optional[Tuple[str, Optional[str], bool]]:
        if response_meta.get("provider_unavailable"):
            for reservation in authorizer.reservations:
                request.billing_helper.refund_reserved_ai_credits(
                    reservation,
                    reason="summary_provider_unavailable",
                )
            return final_text, pending_marker, False

        if not response_meta.get("ai_fallback"):
            return None

        billing_segments = list(response_meta.get("billing_segments") or [])
        if billing_segments:
            request.billing_helper.settle_reserved_ai_credits_batch(
                authorizer.reservations,
                billing_segments,
                reason="summary_stream_provider_usage_before_fallback",
            )
        else:
            for reservation in authorizer.reservations:
                request.billing_helper.refund_reserved_ai_credits(
                    reservation,
                    reason="summary_stream_fallback",
                )
        return final_text, pending_marker, False

    def run_summary_command_stream(
        self,
        request: SummaryCommandRequest,
        stream_consumer: Callable[[Any], str],
    ) -> Tuple[str, Optional[str], bool]:
        if not self.credits_db_service.is_configured():
            return self.handle_rate_limit(request.chat_id, request.message), None, True

        main_reserve_credits, reserve_meta = self.estimate_ai_base_reserve_credits(
            [{"role": "user", "content": "summary"}],
            extra_input_tokens=0,
        )
        base_charge_meta, base_charge_error = request.billing_helper.reserve_ai_credits(
            "ai_response_base",
            main_reserve_credits,
            metadata={
                "operation_id": request.billing_helper.operation_id("summary"),
                "estimated_prompt_messages": 1,
                **reserve_meta,
            },
        )
        if base_charge_error:
            return base_charge_error, None, True

        if not self.check_provider_available(scope="chat") and not self.has_openrouter_fallback():
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="summary_provider_unavailable"
            )
            return self.handle_rate_limit(request.chat_id, request.message), None, True

        authorizer = None
        try:
            authorizer = request.billing_helper.create_authorizer(
                [base_charge_meta],
                model_reserve_credit_units=int(
                    (base_charge_meta or {}).get("reserved_credit_units", 0) or 0
                ),
            )
            response_meta: dict[str, Any] = {
                AI_COST_AUTHORIZER_KEY: authorizer,
                AI_SEGMENT_RECORDER_KEY: authorizer.record_provider_segment,
            }
            token_iterator, pending_marker = self.stream_summary_command(
                request.chat_id,
                request.redis_client,
                request.prompt_text,
                response_meta=response_meta,
            )
        except Exception:
            _summary_logger.exception(
                "summary_stream: preparation failed for chat_id=%s",
                request.chat_id,
            )
            request.billing_helper.refund_reserved_ai_credits(
                base_charge_meta, reason="summary_preparation_failed"
            )
            if authorizer is not None:
                authorizer.close()
            return tr("summary.error"), None, True

        try:
            try:
                final_text = stream_consumer(token_iterator)
            except Exception:
                _summary_logger.exception(
                    "summary_stream: failed for chat_id=%s", request.chat_id
                )
                billing_segments = list(response_meta.get("billing_segments") or [])
                if billing_segments:
                    request.billing_helper.settle_reserved_ai_credits_batch(
                        authorizer.reservations,
                        billing_segments,
                        reason="summary_stream_provider_usage_before_delivery_failure",
                    )
                else:
                    for reservation in authorizer.reservations:
                        request.billing_helper.refund_reserved_ai_credits(
                            reservation,
                            reason="summary_stream_failed",
                        )
                return tr("summary.error"), None, True

            response_meta.pop(AI_COST_AUTHORIZER_KEY, None)
            response_meta.pop(AI_SEGMENT_RECORDER_KEY, None)

            non_success = self._finish_summary_non_success(
                request,
                authorizer,
                response_meta,
                final_text,
                pending_marker,
            )
            if non_success is not None:
                return non_success

            request.billing_helper.settle_reserved_ai_credits_batch(
                authorizer.reservations,
                response_meta.get("billing_segments", []),
                reason="summary_command_stream_success",
            )

            return final_text, pending_marker, False
        finally:
            authorizer.close()


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
    prepare_chat_memory: Callable[..., Any],
    build_ai_messages: Callable[..., List[Dict[str, Any]]],
    check_provider_available: Callable[..., bool],
    has_openrouter_fallback: Callable[[], bool],
    handle_rate_limit: Callable[[str, Dict[str, Any]], str],
    handle_ai_response: Callable[..., str],
    estimate_ai_base_reserve_credits: Callable[..., Tuple[int, Dict[str, Any]]],
    estimate_image_context_reserve_credits: Callable[[bytes, str], int],
    stream_summary_command: Callable[..., Any] = lambda _a, _b, _c, **_kwargs: (
        iter([]),
        None,
    ),
    schedule_compaction: Callable[[Any, Any], bool] = lambda _plan, _billing: False,
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
        schedule_compaction=schedule_compaction,
    )
