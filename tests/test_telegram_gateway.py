from api.telegram_gateway import TelegramGateway


class FakeTelegramRequest:
    def __init__(self, response):
        self.calls = []
        self.response = response

    def __call__(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.response


def test_send_message_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True, "result": {"message_id": 42}}, None))
    gateway = TelegramGateway(telegram_request=request)

    result = gateway.send_message(
        "123",
        "hola https://polymarket.com/market",
        "10",
        buttons=["https://example.com/app"],
    )

    assert result == 42
    assert request.calls[0][0] == "sendMessage"
    assert request.calls[0][1]["method"] == "POST"
    assert request.calls[0][1]["json_payload"] == {
        "chat_id": "123",
        "text": "hola https://polymarket.com/market",
        "disable_web_page_preview": True,
        "reply_to_message_id": "10",
        "reply_markup": {
            "inline_keyboard": [
                [{"text": "abrir en la app", "url": "https://example.com/app"}]
            ]
        },
    }


def test_delete_message_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True}, None))
    gateway = TelegramGateway(telegram_request=request)

    gateway.delete_message("123", "10")

    assert request.calls[0][0] == "deleteMessage"
    assert request.calls[0][1]["method"] == "GET"
    assert request.calls[0][1]["params"] == {"chat_id": "123", "message_id": "10"}
