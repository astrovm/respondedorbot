"""Telegram message streaming support."""

from __future__ import annotations

from contextvars import ContextVar
import time
from typing import Callable, Iterator, Optional, Tuple


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
        min_edit_interval_ms: float = 400.0,
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
        elapsed = now - self._last_edit_time
        new_chars = len(self._buffer) - len(self._sent_text)
        return elapsed >= self._min_interval and new_chars >= self._min_chars

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
        self._buffer += token
        if not self._message_id and not self._send_attempted:
            self._send_attempted = True
            message_id = self._send_message(
                self._chat_id, self._buffer, self._reply_to_message_id
            )
            self._message_id = str(message_id) if message_id is not None else None
            self._last_edit_time = time.time()
            self._sent_text = self._buffer
        elif self._should_edit():
            self._do_edit()

    def finalize(self, final_text: Optional[str] = None) -> str:
        self._done = True
        text = final_text if final_text is not None else self._buffer
        if not self._message_id and not self._send_attempted:
            self._send_attempted = True
            message_id = self._send_message(
                self._chat_id, text, self._reply_to_message_id
            )
            self._message_id = str(message_id) if message_id is not None else None
        elif self._message_id and text != self._sent_text:
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

    for _provider_name, token in token_iterator:
        if token:
            streamer.feed(token)

    return streamer.finalize(), streamer.message_id or ""
