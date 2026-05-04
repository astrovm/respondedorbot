from tests.support import *
from tests.support import assert_no_raw_tool_syntax
from tests.test_message_handler import _build_message_handler_deps


def test_handle_ai_stream_response_returns_final_text_and_stores_stream_metadata(monkeypatch):
    from api.index import handle_ai_stream_response

    response_meta: dict[str, Any] = {}
    token_iterator = iter([("openrouter", "ho"), ("openrouter", "la")])

    with patch("api.index.ask_ai_stream", return_value=token_iterator) as ask_stream:
        with patch(
            "api.index.consume_stream_to_telegram",
            return_value=("hola", "777"),
        ) as consume_stream_to_telegram:
            result = handle_ai_stream_response(
                [{"role": "user", "content": "hola"}],
                response_meta=response_meta,
                chat_id="123",
                user_id=55,
                user_name="@ana",
                timezone_offset=-3,
            )

    assert result == "hola"
    assert_no_raw_tool_syntax(result)
    ask_stream.assert_called_once_with(
        [{"role": "user", "content": "hola"}],
        chat_id="123",
        user_name="@ana",
        user_id=55,
        timezone_offset=-3,
    )
    consume_stream_to_telegram.assert_called_once()
    assert response_meta["streamed_text"] == "hola"
    assert response_meta["streamed_message_id"] == "777"


def test_finalize_message_response_saves_streamed_text_to_redis():
    from api.message_handler import (
        MessageContext,
        PreparedMessage,
        _finalize_message_response,
    )
    from api.streaming import set_streamed_response_metadata

    set_streamed_response_metadata("777", "hola final")
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
        response_msg="hola final",
        response_markup=None,
        response_uses_ai=True,
        response_command=None,
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
        return_value=("hola final", True),
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

    assert response == ("hola final", None, True, None)
    assert run_ai_flow.call_args.kwargs["handler_func"] is deps.handle_ai_stream
    assert run_ai_flow.call_args.kwargs["is_spontaneous"] is True


def test_handle_ai_stream_response_passes_reply_to_message_id(monkeypatch):
    from api.index import handle_ai_stream_response

    response_meta: dict[str, Any] = {}
    token_iterator = iter([("openrouter", "ho"), ("openrouter", "la")])

    with patch("api.index.ask_ai_stream", return_value=token_iterator):
        with patch(
            "api.index.consume_stream_to_telegram",
            return_value=("hola", "777"),
        ) as consume_stream_to_telegram:
            result = handle_ai_stream_response(
                [{"role": "user", "content": "hola"}],
                response_meta=response_meta,
                chat_id="123",
                user_id=55,
                user_name="@ana",
                timezone_offset=-3,
                reply_to_message_id="99",
            )

    assert result == "hola"
    assert consume_stream_to_telegram.call_args.kwargs["reply_to_message_id"] == "99"


def test_run_ai_flow_uses_user_message_id_not_reply_to_message_id():
    from api.message_handler import _run_ai_flow
    from api.ai_service import AIConversationRequest

    mock_service = MagicMock()
    mock_service.run_conversation.return_value = ("respuesta", True)

    deps = MagicMock()
    deps.ai_service = mock_service

    message = {
        "message_id": 42,
        "reply_to_message": {"message_id": 10},
        "chat": {"id": "456"},
        "from": {"id": 9},
    }

    _run_ai_flow(
        deps,
        chat_id="456",
        message=message,
        user_id=9,
        prepared_message=MagicMock(),
        billing_helper=MagicMock(),
        prompt_text="hola",
        reply_context_text=None,
        user_identity="Ana",
        handler_func=MagicMock(),
        redis_client=MagicMock(),
    )

    request_arg = mock_service.run_conversation.call_args[0][0]
    assert isinstance(request_arg, AIConversationRequest)
    assert request_arg.reply_to_message_id == "42"


def test_ask_ai_stream_forwards_extra_tools_and_tool_context():
    from api.index import ask_ai_stream

    system_message = {"role": "system", "content": "sys"}
    rewritten_messages = [{"role": "user", "content": "hola"}]
    extra_tools = [{"type": "function", "function": {"name": "echo"}}]
    tool_context = {"chat_id": "123"}
    stream_result = iter([("openrouter", "ok")])

    with patch(
        "api.index._build_ai_request",
        return_value=(system_message, rewritten_messages, extra_tools, tool_context),
    ):
        with patch("api.index.stream_with_providers", return_value=stream_result) as stream_call:
            result = ask_ai_stream(
                [{"role": "user", "content": "hola"}],
                enable_web_search=False,
                chat_id="123",
                user_name="@ana",
                user_id=55,
                timezone_offset=-3,
            )

    assert result is stream_result
    stream_call.assert_called_once_with(
        system_message,
        rewritten_messages,
        enable_web_search=False,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )


def test_ask_ai_stream_end_to_end_with_internal_tool():
    from api.ai_pricing import AIUsageResult
    from api.index import ask_ai_stream
    from api.providers.base import ProviderChain

    class FakeToolProvider:
        @property
        def name(self) -> str:
            return "fake-tool-provider"

        def is_available(self) -> bool:
            return True

        def complete(self, system_message, messages, **kwargs):
            if kwargs.get("extra_tools"):
                return AIUsageResult(
                    kind="chat",
                    text="tool result: hola",
                    model="fake",
                    usage={},
                    metadata={},
                )
            return AIUsageResult(
                kind="chat",
                text="normal assistant text",
                model="fake",
                usage={},
                metadata={},
            )

        def stream(self, system_message, messages, **kwargs):
            result = self.complete(system_message, messages, **kwargs)
            if result and result.text:
                for i in range(0, len(result.text), 4):
                    yield result.text[i : i + 4]

    messages = [{"role": "user", "content": "hola"}]
    chain = ProviderChain([FakeToolProvider()])

    with patch(
        "api.index._build_ai_request",
        return_value=(
            {"role": "system", "content": "sys"},
            messages,
            [{"type": "function", "function": {"name": "echo"}}],
            {"chat_id": "123"},
        ),
    ):
        with patch("api.index.get_provider_chain", return_value=chain):
            tokens = list(
                ask_ai_stream(
                    messages,
                    chat_id="123",
                    user_name="@ana",
                    user_id=55,
                    timezone_offset=-3,
                )
            )

    final_text = "".join(token for _, token in tokens)
    assert "tool result: hola" in final_text
    assert_no_raw_tool_syntax(final_text)


def test_ask_ai_stream_end_to_end_with_web_search():
    from api.ai_pricing import AIUsageResult
    from api.index import ask_ai_stream
    from api.providers.base import ProviderChain

    class FakeToolProvider:
        @property
        def name(self) -> str:
            return "fake-tool-provider"

        def is_available(self) -> bool:
            return True

        def complete(self, system_message, messages, **kwargs):
            if kwargs.get("enable_web_search"):
                return AIUsageResult(
                    kind="chat",
                    text="normal assistant text",
                    model="fake",
                    usage={},
                    metadata={},
                )
            return None

        def stream(self, system_message, messages, **kwargs):
            result = self.complete(system_message, messages, **kwargs)
            if result and result.text:
                for i in range(0, len(result.text), 4):
                    yield result.text[i : i + 4]

    messages = [{"role": "user", "content": "hola"}]
    chain = ProviderChain([FakeToolProvider()])

    with patch(
        "api.index._build_ai_request",
        return_value=(
            {"role": "system", "content": "sys"},
            messages,
            [],
            {"chat_id": "123"},
        ),
    ):
        with patch("api.index.get_provider_chain", return_value=chain):
            tokens = list(
                ask_ai_stream(
                    messages,
                    enable_web_search=True,
                    chat_id="123",
                    user_name="@ana",
                    user_id=55,
                    timezone_offset=-3,
                )
            )

    final_text = "".join(token for _, token in tokens)
    assert "normal assistant text" in final_text
    assert_no_raw_tool_syntax(final_text)


def test_handle_ai_stream_response_end_to_end_no_tool_leak():
    from api.index import handle_ai_stream_response

    response_meta: dict[str, Any] = {}
    token_iterator = iter([("openrouter", "tool result: "), ("openrouter", "hola")])

    with patch("api.index.ask_ai_stream", return_value=token_iterator):
        with patch(
            "api.index.consume_stream_to_telegram",
            return_value=("tool result: hola", "777"),
        ) as consume_stream_to_telegram:
            result = handle_ai_stream_response(
                [{"role": "user", "content": "hola"}],
                response_meta=response_meta,
                chat_id="123",
                user_id=55,
                user_name="@ana",
                timezone_offset=-3,
            )

    assert result == "tool result: hola"
    assert_no_raw_tool_syntax(result)
    assert response_meta["streamed_text"] == result
    consume_stream_to_telegram.assert_called_once()


def test_stream_with_providers_forwards_extra_tools_and_tool_context():
    from api.index import stream_with_providers

    chain = MagicMock()
    chain.stream.return_value = iter([("openrouter", "ok")])
    extra_tools = [{"type": "function", "function": {"name": "echo"}}]
    tool_context = {"chat_id": "123"}

    with patch("api.index.get_provider_chain", return_value=chain):
        result = stream_with_providers(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=False,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )
        tokens = list(result)

    assert tokens == [("openrouter", "ok")]
    chain.stream.assert_called_once_with(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
        enable_web_search=False,
        extra_tools=extra_tools,
        tool_context=tool_context,
    )


def test_provider_chain_stream_uses_complete_fallback_for_tool_requests():
    from api.ai_pricing import AIUsageResult
    from api.providers.base import ProviderChain

    class CompleteOnlyProvider:
        def __init__(self, name: str, result_text: Optional[str]) -> None:
            self._name = name
            self._result_text = result_text
            self.complete_calls: list[dict[str, Any]] = []

        @property
        def name(self) -> str:
            return self._name

        def is_available(self) -> bool:
            return True

        def complete(self, system_message, messages, **kwargs):
            self.complete_calls.append(kwargs)
            if self._result_text is None:
                return None
            return AIUsageResult(
                kind="chat",
                text=self._result_text,
                model="test-model",
                usage={},
            )

        def stream(self, system_message, messages, **kwargs):
            result = self.complete(system_message, messages, **kwargs)
            if result and result.text:
                for i in range(0, len(result.text), 4):
                    yield result.text[i : i + 4]

    extra_tools = [{"type": "function", "function": {"name": "echo"}}]
    tool_context = {"chat_id": "123"}
    first = CompleteOnlyProvider("first", None)
    second = CompleteOnlyProvider("second", "hola final")
    chain = ProviderChain([first, second])

    tokens = list(
        chain.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=False,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )
    )

    assert tokens == [("second", ""), ("second", "hola"), ("second", " fin"), ("second", "al")]
    assert first.complete_calls == [
        {
            "enable_web_search": False,
            "extra_tools": extra_tools,
            "tool_context": tool_context,
        }
    ]
    assert second.complete_calls == [
        {
            "enable_web_search": False,
            "extra_tools": extra_tools,
            "tool_context": tool_context,
        }
    ]
