from tests.support import *


def test_stream_to_telegram_sends_first_token_without_placeholder():
    from api.streaming import stream_to_telegram

    sent_messages: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []

    def send_message(chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Optional[int]:
        sent_messages.append((chat_id, text, reply_to_message_id))
        return 321

    def edit_message(chat_id: str, text: str, message_id: str) -> None:
        edits.append((chat_id, text, message_id))

    final_text, message_id = stream_to_telegram(
        "chat-1",
        iter([("provider", "ho"), ("provider", "la")]),
        send_message,
        edit_message,
    )

    assert final_text == "hola"
    assert message_id == "321"
    assert sent_messages == [("chat-1", "ho", None)]
    assert edits == [("chat-1", "hola", "321")]


def test_stream_to_telegram_passes_reply_to_message_id():
    from api.streaming import stream_to_telegram

    sent_messages: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []

    def send_message(chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Optional[int]:
        sent_messages.append((chat_id, text, reply_to_message_id))
        return 321

    def edit_message(chat_id: str, text: str, message_id: str) -> None:
        edits.append((chat_id, text, message_id))

    final_text, message_id = stream_to_telegram(
        "chat-1",
        iter([("provider", "ho"), ("provider", "la")]),
        send_message,
        edit_message,
        reply_to_message_id="99",
    )

    assert final_text == "hola"
    assert message_id == "321"
    assert sent_messages == [("chat-1", "ho", "99")]
    assert edits == [("chat-1", "hola", "321")]
