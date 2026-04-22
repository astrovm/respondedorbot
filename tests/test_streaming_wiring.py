from tests.support import *
from tests.test_message_handler import _build_message_handler_deps


def test_handle_ai_stream_response_returns_sentinel_and_stores_stream_metadata(monkeypatch):
    from api.index import handle_ai_stream_response
    from api.message_handler import _STREAMED_SENTINEL

    response_meta: dict[str, Any] = {}
    token_iterator = iter([("openrouter", "ho"), ("openrouter", "la")])

    with patch("api.index.ask_ai_stream", return_value=token_iterator) as ask_stream:
        with patch(
            "api.index.stream_to_telegram",
            return_value=("hola", "777"),
        ) as stream_to_telegram:
            result = handle_ai_stream_response(
                [{"role": "user", "content": "hola"}],
                response_meta=response_meta,
                chat_id="123",
                user_id=55,
                user_name="@ana",
                timezone_offset=-3,
            )

    assert result == _STREAMED_SENTINEL
    ask_stream.assert_called_once_with(
        [{"role": "user", "content": "hola"}],
        chat_id="123",
        user_name="@ana",
        user_id=55,
        timezone_offset=-3,
    )
    stream_to_telegram.assert_called_once()
    assert response_meta["streamed_text"] == "hola"
    assert response_meta["streamed_message_id"] == "777"


def test_finalize_message_response_saves_streamed_text_to_redis():
    from api.message_handler import (
        MessageContext,
        PreparedMessage,
        _STREAMED_SENTINEL,
        _finalize_message_response,
    )

    deps = MagicMock()

    result = _finalize_message_response(
        deps,
        context=MessageContext(
            message_id="42",
            chat_id="555",
            chat_type="private",
            user_identity="Ana (ana)",
            user_id=77,
            numeric_chat_id=555,
        ),
        message={"from": {"first_name": "Ana"}},
        prepared_message=PreparedMessage(
            message_text="hola",
            photo_file_id=None,
            audio_file_id=None,
        ),
        reply_context_text=None,
        redis_client=MagicMock(),
        response_msg=_STREAMED_SENTINEL,
        response_markup=None,
        response_uses_ai=True,
        response_command=None,
        streamed_message_id="777",
        streamed_response_text="hola final",
    )

    assert result == "ok"
    deps.save_message_to_redis.assert_any_call(
        "555",
        "bot_777",
        "hola final",
        ANY,
        role="assistant",
    )
    deps.send_msg.assert_not_called()


def test_handle_known_command_spontaneous_path_uses_run_ai_flow():
    from api.message_handler import PreparedMessage, _handle_known_command

    make_deps, _ = _build_message_handler_deps()
    deps = make_deps(handle_ai_stream=MagicMock(return_value="ignored"))

    with patch(
        "api.message_handler._run_ai_flow",
        return_value=("__streamed__", True),
    ) as run_ai_flow:
        response = _handle_known_command(
            deps,
            commands={},
            command="",
            sanitized_message_text="",
            message={"message_id": "42", "from": {"first_name": "Ana"}},
            chat_id="555",
            chat_type="private",
            user_id=77,
            numeric_chat_id=555,
            prepared_message=PreparedMessage(
                message_text="hola",
                photo_file_id=None,
                audio_file_id=None,
            ),
            billing_helper=MagicMock(),
            reply_context_text=None,
            user_identity="Ana (ana)",
            redis_client=MagicMock(),
            timezone_offset=-3,
        )

    assert response == ("__streamed__", None, True, None)
    assert run_ai_flow.call_args.kwargs["handler_func"] is deps.handle_ai_stream
    assert run_ai_flow.call_args.kwargs["is_spontaneous"] is True
