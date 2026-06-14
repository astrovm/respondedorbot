"""Turn AI output into Telegram messages, including streaming fallbacks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from api.ai.pipeline import AIResponseRequest, AIResponseRuntime


def handle_rate_limit(
    chat_id: str,
    message: dict[str, Any],
    *,
    telegram_token: str | None,
    send_typing: Callable[[str, str], None],
    sleep: Callable[[float], None],
    random_delay: Callable[[float, float], float],
    build_random_reply: Callable[..., str],
    gen_random: Callable[..., str],
) -> str:
    if telegram_token:
        send_typing(telegram_token, chat_id)
    sleep(random_delay(0, 1))
    return build_random_reply(
        gen_random,
        cast(Mapping[str, Any], message.get("from") or {}),
    )


def handle_ai_response(
    chat_id: str,
    handler_func: Callable[..., str],
    messages: list[dict[str, Any]],
    *,
    stream_handler: Callable[..., str],
    image_data: bytes | None,
    image_file_id: str | None,
    context_texts: Sequence[str | None] | None,
    user_identity: str | None,
    response_meta: dict[str, Any] | None,
    user_id: int | None,
    timezone_offset: int,
    reply_to_message_id: str | None,
    extract_user_name: Callable[[str | None], str],
    handle_response: Callable[..., str],
    send_typing: Callable[[str, str], None],
    telegram_token: str | None,
    reset_request_count: Callable[[], Any],
    restore_request_count: Callable[[Any], None],
    get_request_count: Callable[[], int],
) -> str:
    """Adapt either a normal or streaming handler to the cleanup pipeline."""

    effective_handler = handler_func
    if handler_func is stream_handler:
        # The generic response pipeline expects a simple ``handler(messages)``.
        # This closure fills in Telegram-specific details for the stream path.
        user_name = extract_user_name(user_identity)

        def run_stream(
            handler_messages: list[dict[str, Any]],
            *,
            response_meta: dict[str, Any] | None = None,
        ) -> str:
            return stream_handler(
                handler_messages,
                response_meta=response_meta,
                chat_id=chat_id,
                user_id=user_id,
                user_name=user_name or None,
                timezone_offset=timezone_offset,
                reply_to_message_id=reply_to_message_id,
                image_data=image_data,
                image_file_id=image_file_id,
            )

        effective_handler = run_stream

    return handle_response(
        AIResponseRequest(
            chat_id=chat_id,
            handler=effective_handler,
            messages=messages,
            image_data=image_data,
            image_file_id=image_file_id,
            context_texts=context_texts,
            user_identity=user_identity,
            response_meta=response_meta,
            user_id=user_id,
            timezone_offset=timezone_offset,
            reply_to_message_id=reply_to_message_id,
        ),
        AIResponseRuntime(
            send_typing=send_typing,
            telegram_token=telegram_token,
            reset_request_count=reset_request_count,
            restore_request_count=restore_request_count,
            get_request_count=get_request_count,
        ),
    )


def send_message_for_stream(
    chat_id: str,
    text: str,
    reply_to_message_id: str | None,
    *,
    send_message: Callable[..., int | None],
    logger: Any,
) -> int | None:
    try:
        return send_message(chat_id, text, reply_to_message_id or "")
    except Exception as error:
        logger.exception(
            "stream: failed to send message chat_id=%s: %s",
            chat_id,
            error,
        )
        return None


def edit_message_for_stream(
    chat_id: str,
    text: str,
    message_id: str,
    *,
    edit_message: Callable[..., Any],
    logger: Any,
) -> None:
    try:
        edit_message(chat_id, int(message_id), text)
    except Exception as error:
        logger.exception(
            "stream: failed to edit message chat_id=%s message_id=%s: %s",
            chat_id,
            message_id,
            error,
        )


def handle_ai_stream_response(
    messages: list[dict[str, Any]],
    *,
    response_meta: dict[str, Any] | None,
    chat_id: str | None,
    user_id: int | None,
    user_name: str | None,
    timezone_offset: int,
    reply_to_message_id: str | None,
    image_data: bytes | None,
    image_file_id: str | None,
    inject_image_context: Callable[..., None],
    telegram_token: str | None,
    send_typing: Callable[[str, str], None],
    ask_ai_stream: Callable[..., Iterator[tuple[str, str]]],
    consume_stream: Callable[..., tuple[str, Any]],
    send_stream_message: Callable[..., int | None],
    edit_stream_message: Callable[..., None],
    ask_ai: Callable[..., str],
    gen_random: Callable[[str], str],
    set_stream_metadata: Callable[[str | None, str], None],
) -> str:
    """Stream tokens to Telegram and retry once without streaming if needed."""

    if not chat_id:
        return "me quedé reculando y no te pude responder, probá de nuevo"

    if image_data:
        inject_image_context(
            messages,
            image_data,
            image_file_id,
            response_meta,
        )

    if telegram_token:
        send_typing(telegram_token, chat_id)

    token_iterator = ask_ai_stream(
        messages,
        chat_id=chat_id,
        user_name=user_name,
        user_id=user_id,
        timezone_offset=timezone_offset,
    )
    try:
        # The consumer creates one Telegram message, then edits it as tokens arrive.
        final_text, message_id = consume_stream(
            chat_id,
            token_iterator,
            send_stream_message,
            edit_stream_message,
            reply_to_message_id=reply_to_message_id,
        )
    except RuntimeError:
        # Telegram editing can fail. Generate the same answer normally and send it
        # as one message so the user still receives a response.
        final_text = ask_ai(
            messages,
            chat_id=chat_id,
            user_name=user_name,
            user_id=user_id,
            timezone_offset=timezone_offset,
            response_meta=response_meta,
        )
        if not final_text.strip():
            final_text = gen_random(user_name or "")
        message_id = send_stream_message(
            chat_id,
            final_text,
            reply_to_message_id,
        )
        set_stream_metadata(
            str(message_id) if message_id is not None else None,
            final_text,
        )

    if response_meta is not None:
        response_meta["billing_segments"] = list(
            cast(
                list[dict[str, Any]],
                response_meta.get("billing_segments") or [],
            )
        )
        response_meta["streamed_text"] = final_text
        response_meta["streamed_message_id"] = message_id
    return final_text


@dataclass
class ResponseServiceDeps:
    """External operations needed to deliver a response."""

    telegram: Any
    ai: Any
    providers: Any
    telegram_token: Callable[[], str | None]
    send_typing: Callable[[str, str], None]
    sleep: Callable[[float], None]
    random_delay: Callable[[float, float], float]
    build_random_reply: Callable[..., str]
    gen_random: Callable[..., str]
    extract_user_name: Callable[[str | None], str]
    handle_response: Callable[..., str]
    consume_stream: Callable[..., tuple[str, Any]]
    set_stream_metadata: Callable[[str | None, str], None]
    edit_message: Callable[..., Any]
    logger: Any


class ResponseService:
    """High-level response API used by the message handler."""

    def __init__(self, deps: ResponseServiceDeps) -> None:
        self._deps = deps
        self.stream_handler = self.handle_stream

    def handle_rate_limit(
        self,
        chat_id: str,
        message: dict[str, Any],
    ) -> str:
        return handle_rate_limit(
            chat_id,
            message,
            telegram_token=self._deps.telegram_token(),
            send_typing=self._deps.send_typing,
            sleep=self._deps.sleep,
            random_delay=self._deps.random_delay,
            build_random_reply=self._deps.build_random_reply,
            gen_random=self._deps.gen_random,
        )

    def handle(
        self,
        chat_id: str,
        handler_func: Callable[..., str],
        messages: list[dict[str, Any]],
        image_data: bytes | None = None,
        image_file_id: str | None = None,
        context_texts: Sequence[str | None] | None = None,
        user_identity: str | None = None,
        response_meta: dict[str, Any] | None = None,
        user_id: int | None = None,
        timezone_offset: int = -3,
        reply_to_message_id: str | None = None,
    ) -> str:
        return handle_ai_response(
            chat_id,
            handler_func,
            messages,
            stream_handler=self.stream_handler,
            extract_user_name=self._deps.extract_user_name,
            handle_response=self._deps.handle_response,
            send_typing=self._deps.send_typing,
            telegram_token=self._deps.telegram_token(),
            reset_request_count=self._deps.providers.reset_request_count,
            restore_request_count=self._deps.providers.restore_request_count,
            get_request_count=self._deps.providers.get_request_count,
            image_data=image_data,
            image_file_id=image_file_id,
            context_texts=context_texts,
            user_identity=user_identity,
            response_meta=response_meta,
            user_id=user_id,
            timezone_offset=timezone_offset,
            reply_to_message_id=reply_to_message_id,
        )

    def send_stream_message(
        self,
        chat_id: str,
        text: str,
        reply_to_message_id: str | None,
    ) -> int | None:
        return send_message_for_stream(
            chat_id,
            text,
            reply_to_message_id,
            send_message=self._deps.telegram.send_message,
            logger=self._deps.logger,
        )

    def edit_stream_message(
        self,
        chat_id: str,
        text: str,
        message_id: str,
    ) -> None:
        edit_message_for_stream(
            chat_id,
            text,
            message_id,
            edit_message=self._deps.edit_message,
            logger=self._deps.logger,
        )

    def handle_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        response_meta: dict[str, Any] | None = None,
        chat_id: str | None = None,
        user_id: int | None = None,
        user_name: str | None = None,
        timezone_offset: int = -3,
        reply_to_message_id: str | None = None,
        image_data: bytes | None = None,
        image_file_id: str | None = None,
        **_: Any,
    ) -> str:
        return handle_ai_stream_response(
            messages,
            response_meta=response_meta,
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
            timezone_offset=timezone_offset,
            reply_to_message_id=reply_to_message_id,
            image_data=image_data,
            image_file_id=image_file_id,
            inject_image_context=self._deps.ai.inject_image_context,
            telegram_token=self._deps.telegram_token(),
            send_typing=self._deps.send_typing,
            ask_ai_stream=self._deps.ai.stream,
            consume_stream=self._deps.consume_stream,
            send_stream_message=self.send_stream_message,
            edit_stream_message=self.edit_stream_message,
            ask_ai=self._deps.ai.ask,
            gen_random=self._deps.gen_random,
            set_stream_metadata=self._deps.set_stream_metadata,
        )
