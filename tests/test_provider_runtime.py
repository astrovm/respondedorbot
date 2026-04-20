from types import SimpleNamespace

from tests.support import *


class _FakeChoice:
    def __init__(self, finish_reason, message):
        self.finish_reason = finish_reason
        self.message = message


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )
        self.calls = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_provider_runtime_executes_tool_calls_until_stop():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    tool_calls = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="calc", arguments='{"x": 1}'),
        )
    ]
    first_response = _FakeResponse(
        [
            _FakeChoice(
                "tool_calls",
                SimpleNamespace(content="", tool_calls=tool_calls, annotations=[]),
            )
        ]
    )
    second_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="done", tool_calls=[], annotations=[]),
            )
        ]
    )
    client = _FakeClient([first_response, second_response])
    execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="tool output"))
    tool_runtime = ToolRuntime(
        execute_tool_fn=execute_tool_fn,
        parse_tool_call_arguments_fn=lambda args: {"x": 1},
        tool_registry={"calc": object()},
        print_fn=lambda *_args: None,
    )
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {"type": "web_search"},
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage={},
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda _response: {},
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        tool_runtime,
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
        enable_web_search=False,
        extra_tools=[{"name": "calc"}],
        tool_context={"chat_id": "123"},
    )

    assert result is not None
    assert result.text == "done"
    assert execute_tool_fn.call_count == 1
    assert client.calls[0]["messages"][0]["content"] == "sys"
    assert client.calls[1]["messages"][-1]["role"] == "tool"


def test_provider_runtime_returns_text_when_tool_calls_are_unknown():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "tool_calls",
                SimpleNamespace(
                    content="fallback text",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="unknown", arguments="{}"),
                        )
                    ],
                    annotations=[],
                ),
            )
        ]
    )
    client = _FakeClient([response])
    execute_tool_fn = MagicMock()
    tool_runtime = ToolRuntime(
        execute_tool_fn=execute_tool_fn,
        parse_tool_call_arguments_fn=lambda args: {},
        tool_registry={},
        print_fn=lambda *_args: None,
    )
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {"type": "web_search"},
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage={},
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda _response: {},
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        tool_runtime,
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
        enable_web_search=False,
        extra_tools=[{"name": "calc"}],
        tool_context={},
    )

    assert result is not None
    assert result.text == "fallback text"
    execute_tool_fn.assert_not_called()
