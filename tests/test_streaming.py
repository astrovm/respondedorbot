from tests.support import *


def test_stream_to_telegram_sends_placeholder_before_editing():
    from api.streaming import stream_to_telegram

    sent_messages: list[tuple[str, str]] = []
    edits: list[tuple[str, str, str]] = []

    def send_message(chat_id: str, text: str) -> Optional[int]:
        sent_messages.append((chat_id, text))
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
    assert sent_messages == [("chat-1", "...")]
    assert edits == [("chat-1", "hola", "321")]
