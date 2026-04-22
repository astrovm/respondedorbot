"""Telegram message streaming support."""

from __future__ import annotations

import time
from typing import Callable, Iterator, Optional, Tuple


SendMessageFn = Callable[[str, str], Optional[int]]
EditMessageFn = Callable[[str, str, str], None]


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
    ) -> None:
        self._chat_id = chat_id
        self._send_message = send_message_fn
        self._edit_message = edit_message_fn
        self._min_interval = min_edit_interval_ms / 1000.0
        self._min_chars = min_chars_between_edits
        self._placeholder = placeholder
        self._buffer = ""
        self._sent_text = ""
        self._message_id: Optional[str] = None
        self._last_edit_time = 0.0
        self._done = False

    def start(self) -> str:
        """Send the initial placeholder message and return its message ID."""
        message_id = self._send_message(self._chat_id, self._placeholder)
        self._message_id = str(message_id) if message_id is not None else None
        self._last_edit_time = time.time()
        return self._message_id or ""

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
        if self._should_edit():
            self._do_edit()

    def finalize(self, final_text: Optional[str] = None) -> str:
        """Return the final text and mark the stream as done."""
        self._done = True
        text = final_text if final_text is not None else self._buffer
        if self._message_id and text != self._sent_text:
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
) -> Tuple[str, str]:
    """Consume a token iterator and stream the result to Telegram.

    Args:
        chat_id: The Telegram chat ID.
        token_iterator: Yields (provider_name, token) tuples.
        send_message_fn: Function to send the initial Telegram message.
        edit_message_fn: Function to edit a Telegram message.
        placeholder: Initial placeholder text.

    Returns:
        The final accumulated text and Telegram message ID.
    """
    streamer = TelegramMessageStreamer(
        chat_id,
        send_message_fn,
        edit_message_fn,
        placeholder=placeholder,
    )
    message_id = streamer.start()

    for _provider_name, token in token_iterator:
        if token:
            streamer.feed(token)

    return streamer.finalize(), message_id
