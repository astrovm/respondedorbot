"""Telegram message streaming support."""

from __future__ import annotations

from contextvars import ContextVar
import logging
import time
from typing import Callable, Iterator, Optional, Protocol, Tuple, cast

from api.billing.authorization import AIAuthorizationDenied
from api.core.rust_bridge import load_rust_bridge


logger = logging.getLogger(__name__)


class _RustTelegramStreamPlanning(Protocol):
    def telegram_stream_should_edit(
        self,
        done: bool,
        has_message_id: bool,
        now_seconds: float,
        last_edit_seconds: float,
        buffer_chars: int,
        sent_chars: int,
        min_edit_interval_seconds: float,
        min_chars_between_edits: int,
    ) -> bool: ...

    def telegram_stream_plan_feed(
        self,
        done: bool,
        has_message_id: bool,
        send_attempted: bool,
        buffer: str,
        sent_text: str,
        token: str,
        now_seconds: float,
        last_edit_seconds: float,
        min_edit_interval_seconds: float,
        min_chars_between_edits: int,
    ) -> tuple[str, str]: ...

    def telegram_stream_plan_finalize(
        self,
        buffer: str,
        sent_text: str,
        has_message_id: bool,
        final_text: str | None,
    ) -> tuple[str, str]: ...


def _load_rust_telegram_stream_planning() -> _RustTelegramStreamPlanning | None:
    module = load_rust_bridge("RUST_TELEGRAM_STREAM_PLANNING_ENABLED")
    if module is None:
        return None
    return cast(_RustTelegramStreamPlanning, module)


def _rust_stream_planning_failed(operation: str) -> None:
    logger.exception(
        "Rust Telegram stream planning failed; using Python fallback: operation=%s",
        operation,
    )

SendMessageFn = Callable[[str, str, Optional[str]], Optional[int]]
EditMessageFn = Callable[[str, str, str], None]


_streamed_response_metadata: ContextVar[Optional[Tuple[Optional[str], str]]] = ContextVar(
    "streamed_response_metadata",
    default=None,
)


def set_streamed_response_metadata(message_id: Optional[str], text: str) -> None:
    _streamed_response_metadata.set((message_id, text))


def extract_stream_metadata() -> Tuple[Optional[str], str]:
    metadata = _streamed_response_metadata.get()
    _streamed_response_metadata.set(None)
    if metadata is None:
        return None, ""
    return metadata


class TelegramMessageStreamer:
    """Stream tokens into a Telegram message via periodic edits."""

    def __init__(
        self,
        chat_id: str,
        send_message_fn: SendMessageFn,
        edit_message_fn: EditMessageFn,
        *,
        min_edit_interval_ms: float = 300.0,
        min_chars_between_edits: int = 15,
        placeholder: str = "...",
        reply_to_message_id: Optional[str] = None,
    ) -> None:
        self._chat_id = chat_id
        self._send_message = send_message_fn
        self._edit_message = edit_message_fn
        self._min_interval = min_edit_interval_ms / 1000.0
        self._min_chars = min_chars_between_edits
        self._placeholder = placeholder
        self._reply_to_message_id = reply_to_message_id
        self._buffer = ""
        self._sent_text = ""
        self._message_id: Optional[str] = None
        self._last_edit_time = 0.0
        self._done = False
        self._send_attempted = False

    def start(self) -> None:
        self._last_edit_time = time.time()

    @property
    def message_id(self) -> Optional[str]:
        return self._message_id

    def _should_edit(self) -> bool:
        if self._done or not self._message_id:
            return False
        now = time.time()
        rust = _load_rust_telegram_stream_planning()
        if rust is not None:
            try:
                return bool(
                    rust.telegram_stream_should_edit(
                        self._done,
                        bool(self._message_id),
                        now,
                        self._last_edit_time,
                        len(self._buffer),
                        len(self._sent_text),
                        self._min_interval,
                        self._min_chars,
                    )
                )
            except Exception:
                _rust_stream_planning_failed("should_edit")
        return self._python_should_edit(now)

    def _python_should_edit(self, now: float) -> bool:
        elapsed = now - self._last_edit_time
        new_chars = len(self._buffer) - len(self._sent_text)
        return elapsed >= self._min_interval and new_chars >= self._min_chars

    @staticmethod
    def _checked_plan(plan: tuple[str, str]) -> tuple[str, str]:
        text, action = plan
        if action not in {"none", "send", "edit"}:
            raise ValueError(f"invalid Telegram stream action: {action}")
        return str(text), action

    def _plan_feed(self, token: str, now: float) -> tuple[str, str]:
        rust = _load_rust_telegram_stream_planning()
        if rust is not None:
            try:
                return self._checked_plan(
                    rust.telegram_stream_plan_feed(
                        self._done,
                        bool(self._message_id),
                        self._send_attempted,
                        self._buffer,
                        self._sent_text,
                        token,
                        now,
                        self._last_edit_time,
                        self._min_interval,
                        self._min_chars,
                    )
                )
            except Exception:
                _rust_stream_planning_failed("feed")

        if self._done:
            return self._buffer, "none"
        buffer = self._buffer + token
        if not self._message_id and not self._send_attempted:
            action = "send"
        elif self._message_id and self._python_should_edit_for_buffer(buffer, now):
            action = "edit"
        else:
            action = "none"
        return buffer, action

    def _python_should_edit_for_buffer(self, buffer: str, now: float) -> bool:
        elapsed = now - self._last_edit_time
        new_chars = len(buffer) - len(self._sent_text)
        return elapsed >= self._min_interval and new_chars >= self._min_chars

    def _plan_finalize(self, final_text: Optional[str]) -> tuple[str, str]:
        rust = _load_rust_telegram_stream_planning()
        if rust is not None:
            try:
                return self._checked_plan(
                    rust.telegram_stream_plan_finalize(
                        self._buffer,
                        self._sent_text,
                        bool(self._message_id),
                        final_text,
                    )
                )
            except Exception:
                _rust_stream_planning_failed("finalize")

        text = final_text if final_text is not None else self._buffer
        if not self._message_id:
            action = "send"
        elif text != self._sent_text:
            action = "edit"
        else:
            action = "none"
        return text, action

    def _do_edit(self) -> None:
        if not self._message_id:
            return
        try:
            self._edit_message(self._chat_id, self._buffer, self._message_id)
            self._sent_text = self._buffer
            self._last_edit_time = time.time()
        except Exception as e:
            print(f"Stream edit error: {e}")

    def feed(self, token: str) -> None:
        if self._done:
            return
        now = time.time() if self._message_id else self._last_edit_time
        self._buffer, action = self._plan_feed(token, now)
        if action == "send":
            self._send_attempted = True
            message_id = self._send_message(
                self._chat_id, self._buffer, self._reply_to_message_id
            )
            self._message_id = str(message_id) if message_id is not None else None
            self._last_edit_time = time.time()
            self._sent_text = self._buffer
        elif action == "edit":
            self._do_edit()

    def finalize(self, final_text: Optional[str] = None) -> str:
        self._done = True
        text, action = self._plan_finalize(final_text)
        if action == "send":
            if not self._send_attempted:
                self._send_attempted = True
            message_id = self._send_message(
                self._chat_id, text, self._reply_to_message_id
            )
            self._message_id = str(message_id) if message_id is not None else None
        elif action == "edit" and self._message_id:
            try:
                self._edit_message(self._chat_id, text, self._message_id)
            except Exception as e:
                print(f"Stream finalize edit error: {e}")
        return text


def stream_to_telegram(
    chat_id: str,
    token_iterator: Iterator[Tuple[str, str]],
    send_message_fn: SendMessageFn,
    edit_message_fn: EditMessageFn,
    *,
    placeholder: str = "...",
    reply_to_message_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Consume a token iterator and stream the result to Telegram.

    Args:
        chat_id: The Telegram chat ID.
        token_iterator: Yields (provider_name, token) tuples.
        send_message_fn: Function to send the initial Telegram message.
        edit_message_fn: Function to edit a Telegram message.
        placeholder: Initial placeholder text.
        reply_to_message_id: Optional message ID to reply to.

    Returns:
        The final accumulated text and Telegram message ID.
    """
    streamer = TelegramMessageStreamer(
        chat_id,
        send_message_fn,
        edit_message_fn,
        placeholder=placeholder,
        reply_to_message_id=reply_to_message_id,
    )
    streamer.start()

    try:
        for _provider_name, token in token_iterator:
            if token:
                streamer.feed(token)
    except AIAuthorizationDenied as error:
        final_text = streamer.finalize(str(error))
        return final_text, streamer.message_id or ""

    final_text = streamer.finalize()
    message_id = streamer.message_id or ""
    if not message_id:
        raise RuntimeError("Failed to send message to Telegram")
    return final_text, message_id


def consume_stream_to_telegram(
    chat_id: str,
    token_iterator: Iterator[Tuple[str, str]],
    send_message_fn: SendMessageFn,
    edit_message_fn: EditMessageFn,
    reply_to_message_id: Optional[str] = None,
) -> Tuple[str, str]:
    final_text, message_id = stream_to_telegram(
        chat_id,
        token_iterator,
        send_message_fn,
        edit_message_fn,
        reply_to_message_id=reply_to_message_id,
    )
    set_streamed_response_metadata(message_id, final_text)
    return final_text, message_id
