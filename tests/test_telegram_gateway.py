from api.telegram_gateway import TelegramGateway
import json


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


def test_send_message_enables_html_for_linked_polymarket_title():
    request = FakeTelegramRequest(({"ok": True, "result": {"message_id": 42}}, None))
    gateway = TelegramGateway(telegram_request=request)

    gateway.send_message(
        "123",
        '<a href="https://polymarket.com/event/test">Election &amp; runoff</a>',
    )

    payload = request.calls[0][1]["json_payload"]
    assert payload["parse_mode"] == "HTML"
    assert payload["disable_web_page_preview"] is True


def test_delete_message_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True}, None))
    gateway = TelegramGateway(telegram_request=request)

    gateway.delete_message("123", "10")

    assert request.calls[0][0] == "deleteMessage"
    assert request.calls[0][1]["method"] == "GET"
    assert request.calls[0][1]["params"] == {"chat_id": "123", "message_id": "10"}


def test_send_photo_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True, "result": {"message_id": 77}}, None))
    gateway = TelegramGateway(telegram_request=request)

    result = gateway.send_photo(
        "123",
        b"png",
        caption="<b>card</b>",
        msg_id="10",
        reply_markup={"inline_keyboard": [[{"text": "x", "callback_data": "sig"}]]},
    )

    assert result == 77
    assert request.calls[0][0] == "sendPhoto"
    payload = request.calls[0][1]["data_payload"]
    assert payload["chat_id"] == "123"
    assert payload["caption"] == "<b>card</b>"
    assert payload["parse_mode"] == "HTML"
    assert payload["reply_to_message_id"] == "10"
    assert json.loads(payload["reply_markup"]) == {
        "inline_keyboard": [[{"text": "x", "callback_data": "sig"}]]
    }
    assert request.calls[0][1]["files"]["photo"][1] == b"png"


def test_send_video_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True, "result": {"message_id": 78}}, None))
    gateway = TelegramGateway(telegram_request=request)

    result = gateway.send_video(
        "123",
        b"video",
        caption="Instagram reel",
        msg_id="10",
        buttons=["https://www.instagram.com/reel/example"],
    )

    assert result == 78
    assert request.calls[0][0] == "sendVideo"
    payload = request.calls[0][1]["data_payload"]
    assert payload["chat_id"] == "123"
    assert payload["caption"] == "Instagram reel"
    assert payload["supports_streaming"] == "true"
    assert payload["reply_to_message_id"] == "10"
    assert json.loads(payload["reply_markup"]) == {
        "inline_keyboard": [
            [
                {
                    "text": "abrir en la app",
                    "url": "https://www.instagram.com/reel/example",
                }
            ]
        ]
    }
    assert request.calls[0][1]["files"]["video"] == (
        "instagram.mp4",
        b"video",
        "video/mp4",
    )
    assert request.calls[0][1]["timeout"] == 60


def test_edit_photo_delegates_to_telegram_request():
    request = FakeTelegramRequest(({"ok": True, "result": {"message_id": 77}}, None))
    gateway = TelegramGateway(telegram_request=request)

    result = gateway.edit_photo(
        "123",
        "77",
        b"png",
        caption="<b>card</b>",
        reply_markup={"inline_keyboard": [[{"text": "x", "callback_data": "sig"}]]},
    )

    assert result is True
    assert request.calls[0][0] == "editMessageMedia"
    payload = request.calls[0][1]["data_payload"]
    assert payload["chat_id"] == "123"
    assert payload["message_id"] == "77"
    assert json.loads(payload["media"]) == {
        "type": "photo",
        "media": "attach://photo",
        "caption": "<b>card</b>",
        "parse_mode": "HTML",
    }
    assert json.loads(payload["reply_markup"]) == {
        "inline_keyboard": [[{"text": "x", "callback_data": "sig"}]]
    }
    assert request.calls[0][1]["files"]["photo"][1] == b"png"


def test_edit_photo_treats_not_modified_as_success():
    request = FakeTelegramRequest(
        (
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message",
            },
            "Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message",
        )
    )
    gateway = TelegramGateway(telegram_request=request)

    result = gateway.edit_photo("123", "77", b"png", caption="<b>card</b>")

    assert result is True
