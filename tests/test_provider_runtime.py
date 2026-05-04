from types import SimpleNamespace
import json

import httpx
from openai import APIStatusError

from tests.support import *
from tests.support import assert_no_raw_tool_syntax


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
            function=SimpleNamespace(name="web_fetch", arguments='{"x": 1}'),
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
        tool_registry={"web_fetch": object()},
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
        extra_tools=[{"name": "web_fetch"}],
        tool_context={"chat_id": "123"},
    )

    assert result is not None
    assert result.text == "done"
    assert_no_raw_tool_syntax(result.text)
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
    assert_no_raw_tool_syntax(result.text)
    execute_tool_fn.assert_not_called()


def test_provider_runtime_returns_plain_text_when_tools_never_called():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="plain answer", tool_calls=[], annotations=[]),
            )
        ]
    )
    client = _FakeClient([response])
    execute_tool_fn = MagicMock()
    tool_runtime = ToolRuntime(
        execute_tool_fn=execute_tool_fn,
        parse_tool_call_arguments_fn=lambda args: {},
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
        enable_web_search=True,
        extra_tools=[{"name": "calc"}],
        tool_context={},
    )

    assert result is not None
    assert result.text == "plain answer"
    assert_no_raw_tool_syntax(result.text)
    assert client.calls[0]["tools"] == [{"type": "web_search"}, {"name": "calc"}]
    execute_tool_fn.assert_not_called()


def test_provider_runtime_shared_tool_loop_matches_complete():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    def _build_runtime(responses):
        client = _FakeClient(responses)
        execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="tool output"))
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
            ToolRuntime(
                execute_tool_fn=execute_tool_fn,
                parse_tool_call_arguments_fn=lambda _args: {"x": 1},
                tool_registry={"calc": object(), "web_fetch": object()},
                print_fn=lambda *_args: None,
            ),
        )
        return runtime, execute_tool_fn

    def _tool_then_stop_responses():
        return [
            _FakeResponse(
                [
                    _FakeChoice(
                        "tool_calls",
                        SimpleNamespace(
                            content="",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="calc", arguments='{"x": 1}'
                                    ),
                                )
                            ],
                            annotations=[],
                        ),
                    )
                ]
            ),
            _FakeResponse(
                [
                    _FakeChoice(
                        "tool_calls",
                        SimpleNamespace(
                            content="",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_2",
                                    function=SimpleNamespace(
                                        name="web_fetch",
                                        arguments='{"url": "https://example.com"}',
                                    ),
                                )
                            ],
                            annotations=[],
                        ),
                    )
                ]
            ),
            _FakeResponse(
                [
                    _FakeChoice(
                        "stop",
                        SimpleNamespace(content="done", tool_calls=[], annotations=[]),
                    )
                ]
            ),
        ]

    system_message = {"role": "system", "content": "sys"}
    user_messages = [{"role": "user", "content": "hola"}]

    runtime_from_complete, complete_execute_tool_fn = _build_runtime(
        _tool_then_stop_responses()
    )
    complete_result = runtime_from_complete.complete(
        system_message,
        user_messages,
        enable_web_search=False,
        extra_tools=[{"name": "calc"}, {"name": "web_fetch"}],
        tool_context={"chat_id": "123"},
    )

    runtime_from_helper, helper_execute_tool_fn = _build_runtime(_tool_then_stop_responses())
    helper_result = runtime_from_helper._run_tool_rounds(
        current_messages=list(user_messages),
        system_message=system_message,
        enable_web_search=False,
        extra_tools=[{"name": "calc"}, {"name": "web_fetch"}],
        tool_context={"chat_id": "123"},
    )

    assert complete_result is not None
    assert helper_result is not None
    assert complete_result.text == helper_result.text == "done"
    assert_no_raw_tool_syntax(complete_result.text)
    assert_no_raw_tool_syntax(helper_result.text)
    assert complete_result.metadata == helper_result.metadata
    assert complete_execute_tool_fn.call_count == 2
    assert helper_execute_tool_fn.call_count == 2


def test_provider_runtime_retries_json_decode_errors_then_returns_result():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="done", tool_calls=[], annotations=[]),
            )
        ]
    )
    decode_error = json.JSONDecodeError("Expecting value", "<html>bad gateway</html>", 0)
    client = _FakeClient([decode_error, response])
    admin_report = MagicMock()

    def _create(**kwargs):
        client.calls.append(kwargs)
        next_response = client._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    client.chat.completions.create = _create
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=admin_report,
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
        ToolRuntime(),
    )

    with patch("api.provider_runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is not None
    assert result.text == "done"
    assert len(client.calls) == 2
    sleep.assert_called_once_with(1)
    admin_report.assert_not_called()


def test_provider_runtime_retries_server_status_errors_then_returns_result():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="done", tool_calls=[], annotations=[]),
            )
        ]
    )
    http_response = httpx.Response(
        503,
        text="upstream unavailable",
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )
    status_error = APIStatusError("service unavailable", response=http_response, body=None)
    client = _FakeClient([status_error, response])

    def _create(**kwargs):
        client.calls.append(kwargs)
        next_response = client._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    client.chat.completions.create = _create
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
        ToolRuntime(),
    )

    with patch("api.provider_runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is not None
    assert result.text == "done"
    assert len(client.calls) == 2
    sleep.assert_called_once_with(1)


def test_provider_runtime_does_not_retry_bad_request_and_reports_one_based_round():
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    http_response = httpx.Response(
        400,
        text="bad request",
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )
    status_error = APIStatusError("bad request", response=http_response, body=None)
    client = _FakeClient([status_error])
    admin_report = MagicMock()

    def _create(**kwargs):
        client.calls.append(kwargs)
        next_response = client._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    client.chat.completions.create = _create
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=admin_report,
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
        ToolRuntime(),
    )

    with patch("api.provider_runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is None
    assert len(client.calls) == 1
    sleep.assert_not_called()
    admin_report.assert_called_once()
    assert admin_report.call_args.args[2]["tool_round"] == 1
