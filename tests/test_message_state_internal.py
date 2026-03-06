from api.message_state import build_reply_context_text, format_user_message


def test_build_reply_context_text_describes_non_text_media():
    message = {
        "reply_to_message": {
            "from": {"first_name": "Ana", "username": "ana"},
            "sticker": {"emoji": "🔥"},
        }
    }

    result = build_reply_context_text(
        message,
        extract_message_text_fn=lambda _msg: "",
    )

    assert result == "Ana (ana): un sticker 🔥"


def test_format_user_message_includes_reply_context_when_present():
    message = {"from": {"first_name": "Ana", "username": "ana"}}
    result = format_user_message(message, "hola", "Pepe: ping")
    assert result == "Ana (ana) (en respuesta a Pepe: ping): hola"
