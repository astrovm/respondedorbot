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


def test_web_search_sources_ignore_urls_from_other_tools():
    from api.providers.runtime import ProviderRuntime

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "search_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                },
                {
                    "id": "fetch_1",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "search_1",
            "content": '{"results":[]}',
        },
        {
            "role": "tool",
            "tool_call_id": "fetch_1",
            "content": '{"url":"https://example.com/not-a-search-result"}',
        },
    ]

    assert ProviderRuntime._web_search_source_urls(messages) == []


def test_provider_runtime_keeps_direct_search_answer_unchanged():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    search_call = SimpleNamespace(
        id="search_1",
        function=SimpleNamespace(
            name="web_search",
            arguments='{"query":"Autor Ejemplo"}',
        ),
    )
    responses = [
        _FakeResponse(
            [
                _FakeChoice(
                    "tool_calls",
                    SimpleNamespace(
                        content="",
                        tool_calls=[search_call],
                        annotations=[],
                    ),
                )
            ]
        ),
        _FakeResponse(
            [
                _FakeChoice(
                    "stop",
                    SimpleNamespace(
                        content="autor ejemplo es escritor y guionista",
                        tool_calls=[],
                        annotations=[],
                    ),
                )
            ]
        ),
    ]
    client = _FakeClient(responses)
    tool_runtime = ToolRuntime(
        execute_tool_fn=MagicMock(
            return_value=SimpleNamespace(
                output=(
                    '{"results":['
                    '{"url":"https://example.com/profile"},'
                    '{"url":"https://example.com/interview"}]}'
                ),
                metadata={"credits_used": 2},
            )
        ),
        tool_registry={"web_search": object()},
        print_fn=lambda *_args: None,
    )
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {
                "type": "openrouter:web_search",
                "parameters": {"max_uses": 3},
            },
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
    search_schema = {
        "type": "function",
        "function": {"name": "web_search", "parameters": {"type": "object"}},
    }

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "buscá a Autor Ejemplo"}],
        enable_web_search=True,
        extra_tools=[search_schema],
        tool_context={"web_search_enabled": True},
    )

    assert result is not None
    assert result.text == "autor ejemplo es escritor y guionista"
    assert result.metadata["web_search_grounded"] is True
    assert result.metadata["web_search_source_count"] == 2
    assert result.metadata["web_search_citation_count"] == 0
    assert result.metadata["firecrawl_credits_used"] == 2


def test_provider_runtime_executes_tool_calls_until_stop():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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


@pytest.mark.parametrize(
    ("first_usage", "first_annotations", "second_max_uses", "total_requests"),
    [
        ({"server_tool_use": {"web_search_requests": 2}}, [], 1, 3),
        ({}, [{"type": "url_citation"}], 2, 2),
    ],
)
def test_provider_runtime_shares_web_search_budget_across_tool_rounds(
    first_usage,
    first_annotations,
    second_max_uses,
    total_requests,
):
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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
                SimpleNamespace(
                    content="",
                    tool_calls=tool_calls,
                    annotations=first_annotations,
                ),
            )
        ]
    )
    first_response.usage = first_usage
    second_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content="done",
                    tool_calls=[],
                    annotations=[{"type": "url_citation"}],
                ),
            )
        ]
    )
    second_response.usage = {"server_tool_use": {"web_search_requests": 1}}
    client = _FakeClient([first_response, second_response])
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "firecrawl",
                    "max_results": 10,
                    "max_uses": 3,
                    "max_total_results": 30,
                },
            },
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage={},
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda response: response.usage,
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        ToolRuntime(
            execute_tool_fn=MagicMock(return_value=SimpleNamespace(output="2")),
            parse_tool_call_arguments_fn=lambda args: json.loads(args),
            tool_registry={"calc": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "search and calculate"}],
        enable_web_search=True,
        extra_tools=[{"type": "function", "function": {"name": "calc"}}],
    )

    assert result is not None
    assert result.metadata["web_search_requests"] == total_requests
    assert client.calls[0]["tools"] == [
        {"type": "function", "function": {"name": "calc"}}
    ]
    assert client.calls[1]["tools"] == [
        {"type": "function", "function": {"name": "calc"}}
    ]
    assert "extra_body" not in client.calls[0]
    assert "extra_body" not in client.calls[1]


def test_provider_runtime_returns_text_when_tool_calls_are_unknown():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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
    assert client.calls[0]["tools"] == [{"name": "calc"}]
    execute_tool_fn.assert_not_called()


def test_provider_runtime_executes_standalone_pseudo_web_fetch_call():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    first_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content='web_fetch("https://example.com/bts")',
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ]
    )
    second_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="respuesta final", tool_calls=[], annotations=[]),
            )
        ]
    )
    client = _FakeClient([first_response, second_response])
    execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="contenido bts"))
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
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "bts en mexico?"}],
        enable_web_search=False,
        extra_tools=[{"type": "function", "function": {"name": "web_fetch"}}],
        tool_context={"chat_id": "123"},
    )

    assert result is not None
    assert result.text == "respuesta final"
    assert_no_raw_tool_syntax(result.text)
    execute_tool_fn.assert_called_once_with(
        "web_fetch", {"url": "https://example.com/bts"}, {"chat_id": "123"}
    )
    assert client.calls[1]["messages"][-2]["role"] == "assistant"
    assert client.calls[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "web_fetch"
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "pseudo_call_1",
        "content": "contenido bts",
    }


def test_provider_runtime_executes_preamble_then_pseudo_web_fetch_call():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    first_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content=(
                        "ahi me fijo, saco los numeros de memoria no tengo\n"
                        'web_fetch("https://example.com/bts")'
                    ),
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ]
    )
    second_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="respuesta final", tool_calls=[], annotations=[]),
            )
        ]
    )
    client = _FakeClient([first_response, second_response])
    execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="contenido bts"))
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
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "bts en mexico?"}],
        enable_web_search=False,
        extra_tools=[{"type": "function", "function": {"name": "web_fetch"}}],
        tool_context={"chat_id": "123"},
    )

    assert result is not None
    assert result.text == "respuesta final"
    assert_no_raw_tool_syntax(result.text)
    execute_tool_fn.assert_called_once_with(
        "web_fetch", {"url": "https://example.com/bts"}, {"chat_id": "123"}
    )


def test_provider_runtime_executes_dsml_pseudo_web_fetch_call():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    first_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content=(
                        "<｜｜DSML｜｜tool_calls>\n"
                        '<｜｜DSML｜｜invoke name="web_fetch">\n'
                        '<｜｜DSML｜｜parameter name="url" string="true">'
                        "https://nitter.net/test_account/status/1234567890123456789"
                        "</｜｜DSML｜｜parameter>\n"
                        "</｜｜DSML｜｜invoke>\n"
                        "</｜｜DSML｜｜tool_calls>"
                    ),
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ]
    )
    second_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="respuesta final", tool_calls=[], annotations=[]),
            )
        ]
    )
    client = _FakeClient([first_response, second_response])
    execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="tweet"))
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
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "lee esto"}],
        enable_web_search=False,
        extra_tools=[{"type": "function", "function": {"name": "web_fetch"}}],
        tool_context={"chat_id": "123"},
    )

    assert result is not None
    assert result.text == "respuesta final"
    assert_no_raw_tool_syntax(result.text)
    execute_tool_fn.assert_called_once_with(
        "web_fetch",
        {"url": "https://nitter.net/test_account/status/1234567890123456789"},
        {"chat_id": "123"},
    )


def test_provider_runtime_does_not_execute_pseudo_tool_inside_prose():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content='voy a usar web_fetch("https://example.com")',
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ]
    )
    client = _FakeClient([response])
    execute_tool_fn = MagicMock()
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
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
        enable_web_search=False,
        extra_tools=[{"type": "function", "function": {"name": "web_fetch"}}],
        tool_context={},
    )

    assert result is not None
    assert result.text == 'voy a usar web_fetch("https://example.com")'
    execute_tool_fn.assert_not_called()


def test_provider_runtime_does_not_execute_pseudo_tool_when_not_advertised():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(
                    content='web_fetch("https://example.com")',
                    tool_calls=[],
                    annotations=[],
                ),
            )
        ]
    )
    client = _FakeClient([response])
    execute_tool_fn = MagicMock()
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
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
        enable_web_search=False,
        extra_tools=[{"type": "function", "function": {"name": "calculate"}}],
        tool_context={},
    )

    assert result is not None
    assert result.text == 'web_fetch("https://example.com")'
    execute_tool_fn.assert_not_called()


def test_openrouter_stream_uses_tool_runtime_result_without_final_no_tools_call():
    from api.ai.pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider
    from api.tools.runtime import ToolRuntime

    def stream_chunk(content, finish_reason=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    delta=SimpleNamespace(
                        content=content,
                        tool_calls=[],
                        annotations=[],
                    ),
                )
            ],
            usage=None,
        )

    client = _FakeClient(
        [
            iter(
                [
                    stream_chunk('web_fetch("https://example.com/bts")'),
                    stream_chunk(None, "stop"),
                ]
            ),
            iter(
                [
                    stream_chunk("respuesta "),
                    stream_chunk("final", "stop"),
                ]
            ),
        ]
    )
    execute_tool_fn = MagicMock(return_value=SimpleNamespace(output="contenido bts"))
    provider = OpenRouterProvider(
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
        tool_runtime=ToolRuntime(
            execute_tool_fn=execute_tool_fn,
            tool_registry={"web_fetch": object()},
            print_fn=lambda *_args: None,
        ),
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "bts en mexico?"}],
            enable_web_search=False,
            extra_tools=[{"type": "function", "function": {"name": "web_fetch"}}],
            tool_context={"chat_id": "123"},
        )
    )

    assert chunks == ["respuesta ", "final"]
    assert_no_raw_tool_syntax("".join(chunks))
    execute_tool_fn.assert_called_once_with(
        "web_fetch", {"url": "https://example.com/bts"}, {"chat_id": "123"}
    )
    assert client.calls[1]["messages"][-1]["content"] == "contenido bts"
    assert len(client.calls) == 2
    assert "tools" in client.calls[1]


def test_provider_runtime_shared_tool_loop_matches_complete():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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


def _build_retry_runtime(responses, *, extract_usage=lambda _response: {}):
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    client = _FakeClient(responses)
    admin_report = MagicMock()
    request_count = MagicMock()
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=admin_report,
            increment_request_count=request_count,
            build_web_search_tool=lambda: {"type": "web_search"},
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage={},
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=extract_usage,
            primary_model="~deepseek/deepseek-v4-flash-latest",
            max_tool_rounds=5,
        ),
        ToolRuntime(),
    )
    return runtime, client, admin_report, request_count


def test_provider_runtime_returns_none_for_billable_empty_stop():
    response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="", tool_calls=[], annotations=[]),
            )
        ]
    )
    runtime, client, admin_report, request_count = _build_retry_runtime(
        [response],
        extract_usage=lambda _response: {"prompt_tokens": 10},
    )

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "research this"}],
        enable_web_search=True,
        tool_context={"chat_id": "123"},
    )

    assert result is None
    assert len(client.calls) == 1
    assert request_count.call_count == 1
    admin_report.assert_not_called()


@pytest.mark.parametrize(
    ("retryable_finish_reason", "error"),
    [
        (None, None),
        (
            "error",
            {
                "code": 503,
                "metadata": {"error_type": "provider_unavailable"},
            },
        ),
    ],
)
def test_provider_runtime_retries_invalid_finish_reason_then_returns_result(
    retryable_finish_reason,
    error,
):
    from api.ai.pricing import chat_output_token_limit

    incomplete_choice = _FakeChoice(
        retryable_finish_reason,
        SimpleNamespace(content="", tool_calls=[], annotations=[]),
    )
    incomplete_choice.error = error
    incomplete_response = _FakeResponse([incomplete_choice])
    incomplete_response.id = "gen-incomplete"
    incomplete_response._request_id = "req-incomplete"
    incomplete_response.model = "upstream-model"
    incomplete_response.provider = "upstream-provider"
    complete_response = _FakeResponse(
        [
            _FakeChoice(
                "stop",
                SimpleNamespace(content="done", tool_calls=[], annotations=[]),
            )
        ]
    )
    runtime, client, admin_report, request_count = _build_retry_runtime(
        [incomplete_response, complete_response]
    )

    with patch("api.providers.runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "research this"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is not None
    assert result.text == "done"
    assert len(client.calls) == 2
    assert all(
        call["max_tokens"] == chat_output_token_limit("~deepseek/deepseek-v4-flash-latest")
        for call in client.calls
    )
    assert request_count.call_count == 2
    sleep.assert_called_once_with(1)
    admin_report.assert_not_called()


@pytest.mark.parametrize(
    ("finish_reason", "error", "usage"),
    [
        (
            "error",
            {"code": 400, "metadata": {"error_type": "invalid_request"}},
            {},
        ),
        (None, None, {"prompt_tokens": 10}),
    ],
)
def test_provider_runtime_does_not_retry_permanent_or_billable_responses(
    finish_reason,
    error,
    usage,
):
    choice = _FakeChoice(
        finish_reason,
        SimpleNamespace(content="", tool_calls=[], annotations=[]),
    )
    choice.error = error
    response = _FakeResponse([choice])
    runtime, client, admin_report, request_count = _build_retry_runtime(
        [response],
        extract_usage=lambda _response: usage,
    )

    with patch("api.providers.runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "research this"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is None
    assert len(client.calls) == 1
    assert request_count.call_count == 1
    sleep.assert_not_called()
    admin_report.assert_called_once()


def test_provider_runtime_reports_null_finish_reason_after_retries_exhausted():
    responses = []
    for index in range(5):
        choice = _FakeChoice(
            None,
            SimpleNamespace(content="", tool_calls=[], annotations=[]),
        )
        choice.native_finish_reason = "upstream_null"
        response = _FakeResponse([choice])
        response.id = f"gen-{index}"
        response._request_id = f"req-{index}"
        response.model = "upstream-model"
        response.provider = "upstream-provider"
        responses.append(response)

    runtime, client, admin_report, request_count = _build_retry_runtime(responses)

    with patch("api.providers.runtime.time.sleep") as sleep:
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "research this"}],
            enable_web_search=True,
            tool_context={"chat_id": "123"},
        )

    assert result is None
    assert len(client.calls) == 2
    assert request_count.call_count == 2
    assert [call.args[0] for call in sleep.call_args_list] == [1]
    admin_report.assert_called_once()
    assert admin_report.call_args.args[0] == "OpenRouter unexpected finish_reason=None"
    report_context = admin_report.call_args.kwargs["extra_context"]
    assert report_context == {
        "model": "~deepseek/deepseek-v4-flash-latest",
        "enable_web_search": True,
        "tool_round": 1,
        "response_id": "gen-1",
        "request_id": "req-1",
        "response_model": "upstream-model",
        "provider": "upstream-provider",
        "native_finish_reason": "upstream_null",
        "has_content": False,
        "tool_call_count": 0,
    }


def test_provider_runtime_retries_json_decode_errors_then_returns_result():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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

    with patch("api.providers.runtime.time.sleep") as sleep:
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


def test_provider_runtime_keeps_tools_when_json_decode_retries_exhausted():
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    decode_errors = [
        json.JSONDecodeError("Expecting value", "\n         \n", 0)
        for _ in range(5)
    ]
    client = _FakeClient(decode_errors)
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
            build_usage_result=MagicMock(),
            extract_usage_map=lambda _response: {},
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        ToolRuntime(),
    )

    with patch("api.providers.runtime.time.sleep"):
        result = runtime.complete(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=True,
            extra_tools=[{"type": "function", "function": {"name": "calculator"}}],
            tool_context={"chat_id": "123"},
        )

    assert result is None
    assert len(client.calls) == 2
    assert all("tools" in call for call in client.calls)
    admin_report.assert_called_once()
    assert admin_report.call_args.args[2]["provider_error_body"] == (
        " body_len=11 body='\\n         \\n'"
    )


def test_provider_runtime_retries_server_status_errors_then_returns_result():
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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

    with patch("api.providers.runtime.time.sleep") as sleep:
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
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

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

    with patch("api.providers.runtime.time.sleep") as sleep:
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
