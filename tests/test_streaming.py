from tests.support import *
from tests.support import assert_no_raw_tool_syntax


def test_openrouter_stream_uses_native_incremental_streaming_without_tools():
    from types import SimpleNamespace

    from api.providers.openrouter import OpenRouterProvider

    create_calls: list[dict[str, Any]] = []

    def create(**kwargs):
        create_calls.append(kwargs)
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ho"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="la"))]
            ),
        ]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        primary_model="test-model",
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=False,
        )
    )

    assert chunks == ["ho", "la"]
    assert_no_raw_tool_syntax("".join(chunks))
    assert create_calls[0]["stream"] is True
    assert "tools" not in create_calls[0]


def test_openrouter_stream_uses_streaming_path_when_tools_present():
    from types import SimpleNamespace

    from api.ai.pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider

    create_calls: list[dict[str, Any]] = []

    def create(**kwargs):
        create_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hola"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" final"))]),
            ])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="hola final"),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: AIUsageResult(
            kind=kwargs["kind"],
            text=kwargs["text"],
            model=kwargs["model"],
            usage={},
            metadata=kwargs.get("metadata") or {},
        ),
        extract_usage_map=lambda r: {},
        primary_model="test-model",
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=False,
            extra_tools=[{"type": "function", "function": {"name": "echo"}}],
        )
    )

    assert chunks == ["hola", " final"]
    assert len(create_calls) == 1
    assert create_calls[0]["tools"]
    assert create_calls[0]["stream"] is True


def test_openrouter_stream_executes_fragmented_tool_call_and_reports_each_round():
    from types import SimpleNamespace

    from api.ai.pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider
    from api.tools.runtime import ToolRuntime

    create_calls: list[dict[str, Any]] = []
    responses = [
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            annotations=[{"type": "url_citation"}],
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="calculate",
                                        arguments='{"expression":"1',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            annotations=[],
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='+1"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "server_tool_use": {"web_search_requests": 2},
                },
            ),
        ],
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="resultado ",
                            annotations=[{"type": "url_citation"}],
                            tool_calls=[],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content="final",
                            annotations=[],
                            tool_calls=[],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "server_tool_use": {"web_search_requests": 1},
                },
            ),
        ],
    ]

    def create(**kwargs):
        create_calls.append(kwargs)
        return iter(responses.pop(0))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    execute_tool = MagicMock(return_value=SimpleNamespace(output="2"))
    usage_results: list[AIUsageResult] = []
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
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
            usage=kwargs["response"].usage,
            metadata=kwargs.get("metadata") or {},
        ),
        extract_usage_map=lambda response: response.usage,
        primary_model="test-model",
        tool_runtime=ToolRuntime(
            execute_tool_fn=execute_tool,
            tool_registry={"calculate": object()},
            print_fn=lambda *_args: None,
        ),
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "busca y calcula"}],
            enable_web_search=True,
            extra_tools=[
                {
                    "type": "function",
                    "function": {"name": "calculate"},
                }
            ],
            tool_context={"chat_id": "123"},
            on_usage_result=usage_results.append,
        )
    )

    assert chunks == ["resultado final"]
    assert_no_raw_tool_syntax("".join(chunks))
    execute_tool.assert_called_once_with(
        "calculate",
        {"expression": "1+1"},
        {"chat_id": "123"},
    )
    assert create_calls[0]["stream"] is True
    assert create_calls[1]["stream"] is True
    assert create_calls[0]["tools"] == [
        {"type": "function", "function": {"name": "calculate"}}
    ]
    assert create_calls[1]["tools"] == [
        {"type": "function", "function": {"name": "calculate"}}
    ]
    assert "extra_body" not in create_calls[1]
    assert create_calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "2",
    }
    assert [result.usage for result in usage_results] == [
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "server_tool_use": {"web_search_requests": 2},
        },
        {
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "server_tool_use": {"web_search_requests": 1},
        },
    ]
    assert [result.metadata["web_search_requests"] for result in usage_results] == [
        2,
        1,
    ]
    assert all("stream_text_override" not in result.metadata for result in usage_results)


def test_openrouter_stream_uses_web_search_branch_when_enabled():
    from types import SimpleNamespace

    from api.ai.pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider

    create_calls: list[dict[str, Any]] = []
    usage_results: list[AIUsageResult] = []

    def create(**kwargs):
        create_calls.append(kwargs)
        return iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="web"))]),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(content=" answer"),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                },
            ),
        ])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {"type": "web_search"},
        build_usage_result=lambda **kwargs: AIUsageResult(
            kind=kwargs["kind"],
            text=kwargs["text"],
            model=kwargs["model"],
            usage=kwargs["response"].usage,
            metadata=kwargs.get("metadata") or {},
        ),
        extract_usage_map=lambda response: response.usage,
        primary_model="test-model",
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=True,
            extra_tools=[
                {"type": "function", "function": {"name": "web_search"}}
            ],
            on_usage_result=usage_results.append,
        )
    )

    assert chunks == ["web answer"]
    assert_no_raw_tool_syntax("".join(chunks))
    assert create_calls[0]["tools"] == [
        {"type": "function", "function": {"name": "web_search"}}
    ]
    assert usage_results[0].text == "web answer"
    assert "web_search_grounded" not in usage_results[0].metadata


@pytest.mark.parametrize(
    (
        "final_text",
        "search_output",
        "expected_chunks",
        "grounded",
        "source_count",
    ),
    [
        (
            "Perfil: https://example.com/profile",
            '{"results":[{"title":"Persona Ejemplo",'
            '"url":"https://example.com/profile"}]}',
            ["Perfil: https://example.com/profile"],
            True,
            1,
        ),
        (
            "La búsqueda falló y no encontré nada.",
            '{"results":[{"title":"Persona Ejemplo",'
            '"url":"https://example.com/profile"}]}',
            ["La búsqueda falló y no encontré nada."],
            True,
            1,
        ),
        (
            "Consulté https://example.com/not-a-result",
            '{"query":"https://example.com/not-a-result","results":[]}',
            ["Consulté https://example.com/not-a-result"],
            False,
            0,
        ),
        (
            "encontré su perfil en https://social.example/perfil/",
            '{"results":[{"title":"Persona Ejemplo",'
            '"url":"https://social.example/perfil/?lang=es"}]}',
            ["encontré su perfil en https://social.example/perfil/"],
            True,
            1,
        ),
        (
            "respuesta basada en los resultados",
            '{"results":['
            '{"url":"https://example.com/1"},'
            '{"url":"https://example.com/2"},'
            '{"url":"https://example.com/3"},'
            '{"url":"https://example.com/4"}]}',
            ["respuesta basada en los resultados"],
            True,
            4,
        ),
        (
            "",
            '{"results":[{"url":"https://example.com/profile"}]}',
            [],
            False,
            1,
        ),
    ],
)
def test_openrouter_stream_executes_direct_web_search_and_validates_results(
    final_text,
    search_output,
    expected_chunks,
    grounded,
    source_count,
):
    from types import SimpleNamespace

    from api.ai.pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider
    from api.tools.runtime import ToolRuntime

    create_calls = []
    responses = [
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            annotations=[],
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="search_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="web_search",
                                        arguments='{"query":"Persona Ejemplo"}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={"prompt_tokens": 10, "completion_tokens": 2},
            ),
        ],
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content=final_text,
                            annotations=[],
                            tool_calls=[],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={"prompt_tokens": 20, "completion_tokens": 4},
            ),
        ],
    ]

    def create(**kwargs):
        create_calls.append(kwargs)
        return iter(responses.pop(0))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    execute_tool = MagicMock(
        return_value=SimpleNamespace(
            output=search_output,
            metadata={"credits_used": 2},
        )
    )
    usage_results = []
    search_schema = {
        "type": "function",
        "function": {"name": "web_search", "parameters": {"type": "object"}},
    }
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {"type": "openrouter:web_search"},
        build_usage_result=lambda **kwargs: AIUsageResult(
            kind=kwargs["kind"],
            text=kwargs["text"],
            model=kwargs["model"],
            usage=kwargs["response"].usage,
            metadata=kwargs.get("metadata") or {},
        ),
        extract_usage_map=lambda response: response.usage,
        primary_model="test-model",
        tool_runtime=ToolRuntime(
            execute_tool_fn=execute_tool,
            tool_registry={"web_search": object()},
            print_fn=lambda *_args: None,
        ),
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "buscá a Persona Ejemplo"}],
            enable_web_search=True,
            extra_tools=[search_schema],
            on_usage_result=usage_results.append,
        )
    )

    assert chunks == expected_chunks
    execute_tool.assert_called_once_with(
        "web_search",
        {"query": "Persona Ejemplo"},
        {},
    )
    assert create_calls[0]["tools"] == [search_schema]
    assert create_calls[1]["tools"] == [search_schema]
    assert create_calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "search_1",
        "content": search_output,
    }
    assert usage_results[0].metadata["web_search_requests"] == 1
    assert usage_results[0].metadata["firecrawl_credits_used"] == 2
    assert usage_results[-1].metadata["web_search_grounded"] is grounded
    assert usage_results[-1].metadata["web_search_source_count"] == source_count
    assert "web_search_requests" not in usage_results[-1].metadata


def test_openrouter_stream_raises_on_provider_error_chunk():
    from types import SimpleNamespace

    from api.providers.openrouter import OpenRouterProvider

    admin_report = MagicMock()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: iter(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    finish_reason=None,
                                    delta=SimpleNamespace(content="partial"),
                                )
                            ],
                            error=None,
                            usage=None,
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    finish_reason="error",
                                    delta=SimpleNamespace(content=""),
                                )
                            ],
                            error={"code": 502, "message": "provider disconnected"},
                            usage=None,
                        ),
                    ]
                )
            )
        )
    )
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=admin_report,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda response: {},
        primary_model="test-model",
    )

    with pytest.raises(RuntimeError, match="provider disconnected"):
        list(
            provider.stream(
                {"role": "system", "content": "sys"},
                [{"role": "user", "content": "hola"}],
                enable_web_search=False,
            )
        )

    admin_report.assert_called_once()


def test_stream_to_telegram_sends_first_token_without_placeholder():
    from api.bot.streaming import stream_to_telegram

    sent_messages: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []

    def send_message(chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Optional[int]:
        sent_messages.append((chat_id, text, reply_to_message_id))
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
    assert sent_messages == [("chat-1", "ho", None)]
    assert edits == [("chat-1", "hola", "321")]


def test_stream_to_telegram_passes_reply_to_message_id():
    from api.bot.streaming import stream_to_telegram

    sent_messages: list[tuple[str, str, Optional[str]]] = []
    edits: list[tuple[str, str, str]] = []

    def send_message(chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Optional[int]:
        sent_messages.append((chat_id, text, reply_to_message_id))
        return 321

    def edit_message(chat_id: str, text: str, message_id: str) -> None:
        edits.append((chat_id, text, message_id))

    final_text, message_id = stream_to_telegram(
        "chat-1",
        iter([("provider", "ho"), ("provider", "la")]),
        send_message,
        edit_message,
        reply_to_message_id="99",
    )

    assert final_text == "hola"
    assert message_id == "321"
    assert sent_messages == [("chat-1", "ho", "99")]
    assert edits == [("chat-1", "hola", "321")]
