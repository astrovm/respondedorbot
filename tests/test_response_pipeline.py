from tests.support import *


def test_handle_ai_response_reads_fallback_from_meta(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    meta = {}

    def fake_handler(_messages, response_meta=None):
        if response_meta is not None:
            response_meta["ai_fallback"] = True
        return "no boludo"

    result = handle_ai_response("123", fake_handler, [], response_meta=meta)

    assert result == "no boludo"
    assert meta["ai_fallback"] is True


def test_handle_ai_response_does_not_sleep_before_provider_call(monkeypatch):
    import api.ai.pipeline as ai_pipeline

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)

    def fake_handler(_messages):
        return "respuesta"

    result = handle_ai_response("123", fake_handler, [])

    assert result == "respuesta"
    assert not hasattr(ai_pipeline, "time")
    assert not hasattr(ai_pipeline, "random")


def test_handle_ai_response_returns_fallback_on_empty(monkeypatch, caplog):
    """Empty responses should return a fallback message"""
    import logging

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return ""

    response_meta = {}

    with caplog.at_level(logging.WARNING, logger="api.ai.pipeline"):
        result = handle_ai_response("123", fake_handler, [], response_meta=response_meta)

    assert result == "me quedé reculando y no te pude responder, probá de nuevo"
    assert response_meta["ai_fallback"] is True
    assert any("cleaned response empty" in r.message for r in caplog.records)


def test_handle_ai_response_strips_context_echo(monkeypatch):
    """Responses echoing user context should have the echo stripped"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return "usuario_demo: che gordo dame timba\nAcá va la respuesta posta"

    result = handle_ai_response(
        "123",
        fake_handler,
        [],
        context_texts=["usuario_demo: che gordo dame timba"],
    )

    assert result == "Acá va la respuesta posta"


def test_handle_ai_response_strips_user_identity_prefix(monkeypatch):
    """Responses should not repeat the calling user's identity as a prefix"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return "Test User (handle123): acá está la posta"

    result = handle_ai_response(
        "123", fake_handler, [], user_identity="Test User (handle123)"
    )

    assert result == "acá está la posta"


def test_handle_ai_response_passes_image_data_to_stream_handler(monkeypatch):
    """When handler_func is handle_ai_stream_response, image_data must be forwarded."""
    import api.index

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    received_image_data = []

    def mock_stream_response(
        messages,
        *,
        response_meta=None,
        chat_id=None,
        user_id=None,
        user_name=None,
        timezone_offset=-3,
        reply_to_message_id=None,
        image_data=None,
        image_file_id=None,
    ):
        received_image_data.append(
            {
                "image_data": image_data,
                "image_file_id": image_file_id,
            }
        )
        return "streamed response"

    monkeypatch.setattr(
        "api.index.app_runtime.responses.stream_handler", mock_stream_response
    )

    result = api.index.app_runtime.responses.handle(
        "123",
        api.index.app_runtime.responses.stream_handler,
        [{"role": "user", "content": "hello"}],
        image_data=b"fake_image_bytes",
        image_file_id="photo123",
    )

    assert result == "streamed response"
    assert len(received_image_data) == 1
    assert received_image_data[0]["image_data"] == b"fake_image_bytes"
    assert received_image_data[0]["image_file_id"] == "photo123"


def test_handle_ai_stream_response_injects_image_context(monkeypatch):
    """handle_ai_stream_response must call _inject_image_context when image_data is present."""
    handle_ai_stream_response = index.app_runtime.responses.stream_handler

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)

    injected_calls = []

    def mock_inject_image_context(messages, image_data, image_file_id, response_meta):
        injected_calls.append(
            {
                "messages": messages,
                "image_data": image_data,
                "image_file_id": image_file_id,
                "response_meta": response_meta,
            }
        )

    monkeypatch.setattr("api.index.app_runtime.ai.inject_image_context", mock_inject_image_context)

    def mock_ask_ai_stream(*args, **kwargs):
        return iter([("openrouter", "test")])

    monkeypatch.setattr("api.index.app_runtime.ai.stream", mock_ask_ai_stream)

    monkeypatch.setattr(
        "api.index.app_runtime.responses._deps.consume_stream",
        lambda *args, **kwargs: ("final text", "msg_id"),
    )

    messages = [{"role": "user", "content": "que onda con esta foto"}]
    meta = {}

    result = handle_ai_stream_response(
        messages,
        response_meta=meta,
        chat_id="456",
        image_data=b"image_bytes",
        image_file_id="file456",
    )

    assert len(injected_calls) == 1
    assert injected_calls[0]["image_data"] == b"image_bytes"
    assert injected_calls[0]["image_file_id"] == "file456"
    assert injected_calls[0]["messages"] is messages


def test_handle_ai_stream_response_skips_image_injection_when_no_image_data(monkeypatch):
    """handle_ai_stream_response must not call _inject_image_context when image_data is None."""
    handle_ai_stream_response = index.app_runtime.responses.stream_handler

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)

    injected_calls = []

    def mock_inject_image_context(messages, image_data, image_file_id, response_meta):
        injected_calls.append(True)

    monkeypatch.setattr("api.index.app_runtime.ai.inject_image_context", mock_inject_image_context)

    def mock_ask_ai_stream(*args, **kwargs):
        return iter([("openrouter", "test")])

    monkeypatch.setattr("api.index.app_runtime.ai.stream", mock_ask_ai_stream)

    monkeypatch.setattr(
        "api.index.app_runtime.responses._deps.consume_stream",
        lambda *args, **kwargs: ("final text", "msg_id"),
    )

    result = handle_ai_stream_response(
        [{"role": "user", "content": "hello"}],
        response_meta={},
        chat_id="456",
    )

    assert len(injected_calls) == 0


def test_handle_ai_stream_response_uses_gen_random_when_non_stream_fallback_is_empty(monkeypatch):
    handle_ai_stream_response = index.app_runtime.responses.stream_handler

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.app_runtime.ai.stream", lambda *_, **__: iter([]))
    monkeypatch.setattr(
        "api.index.app_runtime.responses._deps.consume_stream",
        MagicMock(side_effect=RuntimeError("Failed to send message to Telegram")),
    )
    monkeypatch.setattr("api.index.app_runtime.ai.ask", lambda *_, **__: "")
    monkeypatch.setattr("api.index.app_runtime.responses._deps.gen_random", lambda name: f"random:{name}")

    sent_messages = []

    def fake_send_message(chat_id, text, reply_to_message_id=None):
        sent_messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        return 42

    monkeypatch.setattr("api.index.app_runtime.responses.send_stream_message", fake_send_message)

    result = handle_ai_stream_response(
        [{"role": "user", "content": "hello"}],
        response_meta={},
        chat_id="456",
        user_name="astro",
        reply_to_message_id="123",
    )

    assert result == "random:astro"
    assert sent_messages == [
        {
            "chat_id": "456",
            "text": "random:astro",
            "reply_to_message_id": "123",
        }
    ]
