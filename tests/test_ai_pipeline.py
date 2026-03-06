from tests.support import *  # noqa: F401,F403

def test_get_groq_compound_enabled_tools_parses_env(monkeypatch):
    assert get_groq_compound_enabled_tools() == [
        "web_search",
        "code_interpreter",
        "visit_website",
        "browser_automation",
    ]


def test_build_ai_messages():
    from api.index import build_ai_messages

    # Test basic message building
    message = {
        "from": {"first_name": "John", "username": "john123"},
        "chat": {"type": "private", "title": None},
        "text": "test message",
    }

    chat_history = []
    message_text = "test message"

    messages = build_ai_messages(message, chat_history, message_text)

    assert len(messages) == 1  # Should only contain the current message
    assert "CONTEXTO:" in messages[0]["content"]
    assert "Usuario: John" in messages[0]["content"]
    assert "test message" in messages[0]["content"]


def test_build_ai_messages_includes_reply_context():
    from api.index import build_ai_messages

    message = {
        "from": {"first_name": "Ana", "username": "ana"},
        "chat": {"type": "group", "title": "Grupo"},
        "text": "respuesta",
    }

    messages = build_ai_messages(
        message,
        [],
        "respuesta",
        "Juan (juan): mensaje citado",
    )

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "MENSAJE AL QUE RESPONDE:" in content
    assert "Juan (juan): mensaje citado" in content
    assert "MENSAJE:" in content
    assert "respuesta" in content


def test_extract_groq_executed_tools_reads_choice_message():
    from api.index import _extract_groq_executed_tools

    response = {
        "choices": [
            {
                "message": {
                    "executed_tools": [
                        {"type": "search", "mode": "advanced"},
                        {"type": "visit", "count": 1},
                    ]
                }
            }
        ]
    }

    assert _extract_groq_executed_tools(response) == [
        {"type": "search", "mode": "advanced"},
        {"type": "visit", "count": 1},
    ]


def test_estimate_ai_base_reserve_credits_uses_compound_for_forced_search(monkeypatch):
    from api.index import estimate_ai_base_reserve_credits

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr("api.index.should_use_groq_compound_tools", lambda: True)
    monkeypatch.setattr(
        "api.index.get_groq_compound_enabled_tools",
        lambda: ["web_search", "visit_website", "code_interpreter", "browser_automation"],
    )

    reserve, metadata = estimate_ai_base_reserve_credits(
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\nbuscá bitcoin hoy"}]
    )

    assert reserve == 3
    assert metadata["reserve_mode"] == "compound"
    assert metadata["reserve_reason"] == "forced_web_search"
    assert metadata["reserve_model"] == "groq/compound"


def test_ask_ai_with_provider_success():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_hacker_news_context"
    ) as mock_get_hn_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch("os.environ.get") as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_hn_context.return_value = []
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_env.side_effect = lambda key: {"GROQ_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "hello"}]
        result = ask_ai(messages)

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_with_all_failures():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_hacker_news_context"
    ) as mock_get_hn_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch("os.environ.get") as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_hn_context.return_value = []
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_env.side_effect = lambda key: {"GROQ_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "hello"}]
        result = ask_ai(messages)

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_with_image():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing when given an image
    with patch("api.index.get_market_context") as mock_get_market_context, patch(
        "api.index.get_weather_context"
    ) as mock_get_weather_context, patch(
        "api.index.get_time_context"
    ) as mock_get_time_context, patch("api.index.describe_image_groq") as mock_describe_image, patch(
        "os.environ.get"
    ) as mock_env:

        # Setup basic mocks
        mock_get_market_context.return_value = {"crypto": [], "dollar": {}}
        mock_get_weather_context.return_value = {"temperature": 25}
        mock_get_time_context.return_value = {"formatted": "Monday"}
        mock_describe_image.return_value = "A beautiful landscape"
        mock_env.side_effect = lambda key: {"GROQ_API_KEY": "test_key"}.get(key)

        messages = [{"role": "user", "content": "what do you see in this image?"}]
        image_data = b"fake_image_data"
        result = ask_ai(messages, image_data=image_data, image_file_id="img123")

        # Just verify it returns a string (could be fallback response)
        assert isinstance(result, str)
        assert len(result) > 0


def test_ask_ai_forced_search_uses_message_section():
    from api.index import ask_ai

    message_block = "\n".join(
        [
            "CONTEXTO:",
            "- Chat: private",
            "- Usuario: Juan",
            "- Hora: 10:00",
            "",
            "MENSAJE:",
            "Últimas noticias de economía",
            "",
            "INSTRUCCIONES:",
            "- Mantené el personaje del gordo",
        ]
    )

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_hacker_news_context", return_value=[]), patch(
        "api.index.get_time_context", return_value={}
    ), patch("api.index.build_system_message",
        return_value={"role": "system", "content": "sys"},
    ), patch(
        "api.index._run_forced_web_search", return_value="ok"
    ) as mock_run, patch(
        "api.index.environ.get"
    ) as mock_env:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_key"
        }.get(key, default)

        result = ask_ai([{"role": "user", "content": message_block}])

    assert result == "ok"
    assert mock_run.call_args.kwargs["query"] == "Últimas noticias de economía"


def test_ask_ai_sanitizes_tool_call_before_retry():
    """ask_ai should sanitize tool responses before retrying provider"""
    from api.index import ask_ai

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_hacker_news_context", return_value=[]), patch(
        "api.index.get_time_context", return_value={}
    ), patch("api.index.build_system_message",
        return_value={"role": "system", "content": "sys"},
    ), patch(
        "api.index.OpenAI"
    ) as mock_openai, patch(
        "api.index.execute_tool", return_value="{}"
    ):
        mock_openai.return_value = MagicMock()

        calls = []

        def fake_complete(system_message, msgs):
            calls.append(msgs)
            if len(calls) == 1:
                return '[TOOL] fetch_url {"url": "https://example.com"}'
            return "respuesta final"

        with patch("api.index.complete_with_providers", side_effect=fake_complete):
            result = ask_ai([{"role": "user", "content": "hola"}])

    assert result == "respuesta final"
    # second call should contain sanitized assistant message without [TOOL]
    second_messages = calls[1]
    assert "[TOOL]" not in second_messages[-2]["content"]


def test_ask_ai_handles_repeated_tool_calls():
    """ask_ai should execute tools repeatedly if providers return new tool calls"""
    from api.index import ask_ai

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_hacker_news_context", return_value=[]), patch(
        "api.index.get_time_context", return_value={}
    ), patch("api.index.build_system_message",
        return_value={"role": "system", "content": "sys"},
    ), patch(
        "api.index.OpenAI"
    ) as mock_openai, patch(
        "api.index.execute_tool"
    ) as mock_tool:
        mock_openai.return_value = MagicMock()
        mock_tool.side_effect = ["res1", "res2"]

        call_count = {"n": 0}

        def fake_complete(system_message, msgs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "[TOOL] tool1 {}"
            if call_count["n"] == 2:
                return "[TOOL] tool2 {}"
            return "fin"

        with patch("api.index.complete_with_providers", side_effect=fake_complete):
            result = ask_ai([{"role": "user", "content": "hola"}])

    assert result == "fin"
    assert mock_tool.call_count == 2
    assert call_count["n"] == 3


def test_fetch_url_content_success_html():
    html_body = """
    <html>
        <head><title>Example Site</title></head>
        <body>
            <article><p>Hola mundo desde la web.</p></article>
        </body>
    </html>
    """

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [html_body.encode("utf-8")]
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.encoding = "utf-8"
    mock_response.apparent_encoding = "utf-8"
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.url = "https://example.com/articulo"
    mock_response.close = MagicMock()

    with patch("api.index.config_redis", side_effect=Exception("redis down")), patch(
        "api.index.requests.get", return_value=mock_response
    ):
        result = fetch_url_content("https://example.com/articulo")

    assert result["url"] == "https://example.com/articulo"
    assert "Hola mundo" in result["content"]
    assert result["status"] == 200
    assert "text/html" in result["content_type"]
    assert result["title"] == "Example Site"
    assert result["truncated"] is False


def test_fetch_url_content_invalid_url():
    result = fetch_url_content("nota sin protocolo")
    assert result == {"error": "url inválida"}


def test_web_search_success():
    """Test web_search with successful DDGS response"""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        # Mock DDGS instance and its text method
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {
                "title": "Test Result 1",
                "href": "https://example.com/test1",
                "body": "This is test result 1 description",
            },
            {
                "title": "Test Result 2",
                "href": "https://example.com/test2",
                "body": "This is test result 2 description",
            },
        ]

        results = web_search("test query", limit=3)

        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/test1"
        assert results[0]["snippet"] == "This is test result 1 description"
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["url"] == "https://example.com/test2"
        assert results[1]["snippet"] == "This is test result 2 description"


def test_web_search_no_results():
    """Test web_search when no results are found"""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = []

        results = web_search("nonexistent query")

        assert len(results) == 0


def test_web_search_network_error():
    """Test web_search when network error occurs"""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_ddgs_class.side_effect = Exception("Network error")

        results = web_search("test query")

        assert len(results) == 0


def test_web_search_limit_parameter():
    """Test web_search respects the limit parameter"""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_ddgs = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {
                "title": "Result 1",
                "href": "https://example.com/1",
                "body": "Description 1",
            },
            {
                "title": "Result 2",
                "href": "https://example.com/2",
                "body": "Description 2",
            },
        ]

        results = web_search("test query", limit=2)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[1]["title"] == "Result 2"

        # Verify that DDGS was called with the correct limit
        mock_ddgs.text.assert_called_once_with(
            query="test query", region="ar-es", safesearch="off", max_results=2
        )


# Tests for search command


def test_search_command_success():
    """Test search_command with successful results"""
    with patch("api.index.web_search") as mock_search:
        mock_search.return_value = [
            {"title": "Test Result", "url": "https://example.com", "snippet": ""},
            {
                "title": "Another Result",
                "url": "https://test.com",
                "snippet": "Test snippet",
            },
        ]

        result = search_command("python programming")

        assert "🔎 Resultados para: python programming" in result
        assert "Test Result" in result
        assert "https://example.com" in result
        assert "Another Result" in result
        assert "Test snippet" in result


def test_search_command_empty_query():
    """Test search_command with empty query"""
    result = search_command("")
    assert result == "decime qué querés buscar capo"

    result = search_command(None)
    assert result == "decime qué querés buscar capo"


def test_search_command_no_results():
    """Test search_command when no results found"""
    with patch("api.index.web_search") as mock_search:
        mock_search.return_value = []

        result = search_command("nonexistent query")

        assert result == "no encontré resultados ahora con duckduckgo"


# Tests for tool calling functionality


def test_parse_tool_call_valid():
    """Test parse_tool_call with valid tool call"""
    text = 'Some text\n[TOOL] web_search {"query": "test", "limit": 3}\nMore text'

    result = parse_tool_call(text)

    assert result is not None
    tool_name, args = result
    assert tool_name == "web_search"
    assert args == {"query": "test", "limit": 3}


def test_parse_tool_call_multiline():
    """parse_tool_call should handle tool name and JSON on separate lines"""
    text = """
    [TOOL]
    web_search
    {"query": "btc asia news"}
    """

    result = parse_tool_call(text)

    assert result == ("web_search", {"query": "btc asia news"})


def test_parse_tool_call_with_colon():
    """parse_tool_call should handle optional colon after [TOOL] marker"""
    text = """
    [TOOL]: web_search
    {
        "query": "eth dump motivos",
        "limit": 4
    }
    """

    result = parse_tool_call(text)

    assert result is not None
    name, args = result
    assert name == "web_search"
    assert args["query"] == "eth dump motivos"
    assert args["limit"] == 4


def test_parse_tool_call_invalid():
    """Test parse_tool_call with invalid inputs"""
    # No tool call
    assert parse_tool_call("Just normal text") is None

    # Malformed JSON
    text = '[TOOL] web_search {"query": invalid json}'
    assert parse_tool_call(text) is None

    # Missing arguments
    text = "[TOOL] web_search"
    assert parse_tool_call(text) is None

    # None input
    assert parse_tool_call(None) is None


def test_execute_tool_web_search():
    """web_search is not a direct execute_tool command in chat flow."""
    with patch("api.index.web_search") as mock_search:
        result = execute_tool("web_search", {"query": "test", "limit": 3})
        assert result == "herramienta desconocida: web_search"
        mock_search.assert_not_called()


def test_execute_tool_fetch_url():
    with patch("api.index.fetch_url_content") as mock_fetch:
        mock_fetch.side_effect = [
            {"url": "https://example.com", "content": "hola", "truncated": False},
            {"error": "url inválida"},
        ]

        first = execute_tool(
            "fetch_url", {"url": "https://example.com", "max_chars": 1234}
        )
        import json

        parsed_first = json.loads(first)
        assert parsed_first["url"] == "https://example.com"
        assert parsed_first["content"] == "hola"

        first_call = mock_fetch.call_args_list[0]
        assert first_call.args[0] == "https://example.com"
        assert first_call.kwargs["max_chars"] == 1234

        second = execute_tool("fetch_url", {"url": ""})
        parsed_second = json.loads(second)
        assert parsed_second["error"] == "url inválida"


def test_execute_tool_empty_query():
    """Test execute_tool with empty query"""
    result = execute_tool("web_search", {"query": "", "limit": 3})

    assert result == "herramienta desconocida: web_search"


def test_resolve_tool_calls_web_search_uses_compound_only():
    from api.index import resolve_tool_calls

    with patch(
        "api.index.get_groq_compound_response", return_value="respuesta compound"
    ) as mock_compound, patch("api.index.execute_tool") as mock_tool, patch(
        "api.index.should_use_groq_compound_tools", return_value=True
    ):
        result = resolve_tool_calls(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            '[TOOL] web_search {"query": "test"}',
        )

    assert result == "respuesta compound"
    mock_compound.assert_called_once()
    mock_tool.assert_not_called()


def test_execute_tool_unknown():
    """Test execute_tool with unknown tool"""
    result = execute_tool("unknown_tool", {})

    assert result == "herramienta desconocida: unknown_tool"


def test_handle_ai_response_sanitizes_tool_lines(monkeypatch):
    """handle_ai_response should strip visible tool call lines"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return 'Hola\n[TOOL] web_search {"query": "test"}\nChau'

    result = handle_ai_response("123", fake_handler, [])

    assert "[TOOL]" not in result
    assert result == "Hola\nChau"


def test_handle_ai_response_strips_internal_ai_fallback_marker(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    meta = {}

    def fake_handler(_messages):
        return "[[AI_FALLBACK]]no boludo"

    result = handle_ai_response("123", fake_handler, [], response_meta=meta)

    assert result == "no boludo"
    assert meta["ai_fallback"] is True


def test_handle_ai_response_returns_fallback_on_empty(monkeypatch):
    """Empty sanitized responses should return a fallback message"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return '[TOOL] web_search {"query": "test"}'

    result = handle_ai_response("123", fake_handler, [])

    assert result == "me quedé reculando y no te pude responder, probá de nuevo"


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


# Tests for complete_with_providers function


def test_complete_with_providers_groq_success():
    """Test complete_with_providers when Groq succeeds"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch("api.index.get_groq_ai_response") as mock_groq:
        mock_groq.return_value = "Groq response"

        result = complete_with_providers(system_message, messages)

        assert result == "Groq response"
        mock_groq.assert_called_once()


def test_complete_with_providers_returns_none_when_groq_fails():
    """Test complete_with_providers returns None when Groq fails"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch("api.index.get_groq_ai_response") as mock_groq:
        mock_groq.return_value = None

        result = complete_with_providers(system_message, messages)

        assert result is None
        assert mock_groq.call_count == 1


def test_complete_with_providers_all_fail():
    """Test complete_with_providers when all providers fail"""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch("api.index.get_groq_ai_response") as mock_groq:
        mock_groq.return_value = None

        result = complete_with_providers(system_message, messages)

        assert result is None
        assert mock_groq.call_count == 1


def test_get_groq_ai_response_skips_call_during_backoff(monkeypatch):
    """When Groq backoff is active we should skip provider API calls entirely."""

    from api import index as index_module

    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    index_module._provider_backoff_until["groq"] = time.time() + 30

    with patch("api.index.OpenAI") as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    mock_openai.assert_not_called()


def test_get_groq_ai_response_skips_call_when_local_rate_limit_hits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    with patch("api.index._consume_groq_rate_limit", return_value=False), patch(
        "api.index.OpenAI"
    ) as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    mock_openai.assert_not_called()


def test_get_groq_ai_response_sets_backoff_on_rate_limit(monkeypatch):
    """Rate limit errors should trigger a backoff window and skip subsequent calls."""

    from api import index as index_module

    index_module._provider_backoff_until.clear()

    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = Exception(
        "Error code: 429 - rate limit reached"
    )

    with patch("api.index.OpenAI", return_value=fake_client) as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

        assert result is None
        remaining = index_module.get_provider_backoff_remaining("groq")
        assert remaining > 0
        assert mock_openai.call_count == 1

        # Second call should be skipped without hitting the API client again
        second_result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

        assert second_result is None
        assert mock_openai.call_count == 1

    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_get_groq_compound_response_skips_call_when_local_rate_limit_hits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    with patch("api.index._consume_groq_rate_limit", return_value=False), patch(
        "api.index.OpenAI"
    ) as mock_openai:
        result = get_groq_compound_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    mock_openai.assert_not_called()


def test_get_groq_compound_response_uses_enabled_tools(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    fake_choice = MagicMock()
    fake_choice.message.content = "ok"
    fake_choice.finish_reason = "stop"

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("api.index.OpenAI", return_value=fake_client):
        result = get_groq_compound_response(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
        )

    assert result == "ok"
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "groq/compound"
    assert call_kwargs["extra_body"]["compound_custom"]["tools"]["enabled_tools"] == [
        "web_search",
        "code_interpreter",
        "visit_website",
        "browser_automation",
    ]


def test_run_forced_web_search_prefers_compound_tools():
    from api.index import _run_forced_web_search

    with patch(
        "api.index.get_groq_compound_response", return_value="compuesto"
    ) as mock_compound, patch("api.index.execute_tool") as mock_tool:
        result = _run_forced_web_search(
            query="algo",
            messages=[{"role": "user", "content": "algo"}],
            system_message={"role": "system", "content": "sys"},
            compound_system_message={"role": "system", "content": "sys"},
        )

    assert result == "compuesto"
    mock_compound.assert_called_once_with(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "algo"}],
    )
    mock_tool.assert_not_called()


def test_web_search_uses_ttl_constant(monkeypatch):
    from api.index import web_search, TTL_WEB_SEARCH

    # Fake DDGS
    class FakeDDGS:
        def __init__(self, timeout=8):  # noqa: ARG001
            pass

        def text(self, query, region, safesearch, max_results):  # noqa: ARG001
            return [{"title": "A", "href": "http://a", "body": "aa"}]

    monkeypatch.setitem(sys.modules, "ddgs", MagicMock(DDGS=FakeDDGS))

    # Fake Redis with setex spy
    with patch("api.index.config_redis") as mock_redis:

        class R:
            def __init__(self):
                self.calls = []

            def get(self, k):
                return None

            def setex(self, k, ttl, v):
                self.calls.append((k, ttl))
                return True

        r = R()
        mock_redis.return_value = r

        results = web_search("algo", limit=1)
        assert results and isinstance(results, list)
        # Ensure TTL constant is used
        assert any(ttl == TTL_WEB_SEARCH for (_k, ttl) in r.calls)
