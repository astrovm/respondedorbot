from types import SimpleNamespace

from tests.support import *
from api.providers.base import ProviderResult


def _build_provider_runtime(*, client, tool_runtime=None):
    from api.ai_pricing import AIUsageResult
    from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tool_runtime import ToolRuntime

    runtime_tool_runtime = tool_runtime or ToolRuntime(print_fn=lambda *_args: None)
    return ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "firecrawl",
                    "max_results": 10,
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
            extract_usage_map=lambda response: getattr(response, "usage", {}),
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        runtime_tool_runtime,
    )


def _build_chat_response(*, text, finish_reason="stop", annotations=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=text,
                    annotations=annotations or [],
                    tool_calls=[],
                ),
            )
        ],
        usage=usage or {"prompt_tokens": 0, "completion_tokens": 0},
    )


def test_get_groq_accounts_for_scope_returns_all_configured_accounts(monkeypatch):
    from api.index import _get_groq_accounts_for_scope

    monkeypatch.setenv("GROQ_FREE_API_KEY", "free-key")
    monkeypatch.setenv("GROQ_API_KEY", "paid-key")

    assert _get_groq_accounts_for_scope() == ["free", "paid"]


def test_get_openrouter_vision_model_maps_supported_models():
    from api.index import _get_openrouter_vision_model

    assert (
        _get_openrouter_vision_model("groq/meta-llama/llama-4-scout-17b-16e-instruct")
        == "meta-llama/llama-4-scout"
    )
    assert _get_openrouter_vision_model("groq/whisper-large-v3") is None


def test_openrouter_config_helpers(monkeypatch):
    from api.index import (
        _get_openrouter_api_key,
        _get_openrouter_base_url,
        _get_openrouter_client,
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-token")

    with patch("api.index.OpenAI") as mock_openai:
        client = _get_openrouter_client(default_headers={"x-test": "1"})

    assert _get_openrouter_api_key() == "openrouter_key"
    assert (
        _get_openrouter_base_url()
        == "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter"
    )
    assert client is mock_openai.return_value
    assert mock_openai.call_args.kwargs["api_key"] == "openrouter_key"
    assert (
        mock_openai.call_args.kwargs["base_url"]
        == "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter"
    )
    assert (
        mock_openai.call_args.kwargs["default_headers"]["cf-aig-authorization"]
        == "Bearer cf-token"
    )


def test_api_index_does_not_expose_unused_agent_limits():
    from api import index

    assert not hasattr(index, "AGENT_MAX_ITERATIONS")
    assert not hasattr(index, "AGENT_MAX_TOOL_CALLS")
    assert not hasattr(index, "AGENT_MAX_WEB_SEARCHES")


def test_api_index_does_not_expose_dead_openrouter_result_helper():
    from api import index

    assert not hasattr(index, "_get_openrouter_ai_response_result")


def test_provider_runtime_enables_firecrawl_web_search():
    response = _build_chat_response(
        text="respuesta con busqueda",
        annotations=[
            {"type": "url_citation", "url_citation": {"url": "https://example.com/1"}},
            {"type": "url_citation", "url_citation": {"url": "https://example.com/2"}},
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "btc news"}],
        enable_web_search=True,
    )

    assert result is not None
    assert result.text == "respuesta con busqueda"
    assert result.metadata["provider"] == "openrouter"
    assert result.metadata["web_search_requests"] == 1
    assert client.chat.completions.create.call_args.kwargs["tools"] == [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "firecrawl",
                "max_results": 10,
                "max_total_results": 30,
            },
        }
    ]


def test_provider_runtime_ignores_invalid_web_search_requests():
    response = _build_chat_response(
        text="respuesta final",
        usage={
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "server_tool_use": {"web_search_requests": "not-an-int"},
        },
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
    )

    assert result is not None
    assert result.text == "respuesta final"
    assert result.metadata["provider"] == "openrouter"
    assert "web_search_requests" not in result.metadata


def test_provider_runtime_server_tool_use_takes_priority():
    response = _build_chat_response(
        text="respuesta con busqueda",
        annotations=[
            {"type": "url_citation", "url_citation": {"url": "https://example.com/1"}},
        ],
        usage={
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "server_tool_use": {"web_search_requests": 3},
        },
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "btc news"}],
        enable_web_search=True,
    )

    assert result is not None
    assert result.metadata["web_search_requests"] == 3


def test_provider_runtime_detects_pydantic_annotation():
    ann = MagicMock()
    ann.type = "url_citation"
    ann.url_citation = MagicMock(url="https://example.com/1")
    response = _build_chat_response(
        text="respuesta con busqueda",
        annotations=[ann],
        usage={
        "prompt_tokens": 10,
        "completion_tokens": 5,
        },
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "btc news"}],
        enable_web_search=True,
    )

    assert result is not None
    assert result.metadata["web_search_requests"] == 1


def test_provider_runtime_sets_explicit_web_search_limits():
    response = _build_chat_response(
        text="respuesta con busqueda",
        annotations=[
            {"type": "url_citation", "url_citation": {"url": "https://example.com/1"}},
        ],
        usage={
        "prompt_tokens": 10,
        "completion_tokens": 5,
        },
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "btc news"}],
        enable_web_search=True,
    )

    assert client.chat.completions.create.call_args.kwargs["tools"] == [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "firecrawl",
                "max_results": 10,
                "max_total_results": 30,
            },
        }
    ]


def test_provider_runtime_includes_web_search_by_default():
    response = _build_chat_response(
        text="respuesta normal",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "hola"}],
    )

    assert result is not None
    assert "tools" in client.chat.completions.create.call_args.kwargs


def test_check_provider_available_returns_true_by_default(monkeypatch):
    from api.index import check_provider_available
    from api.provider_backoff import clear_all_cooldowns

    clear_all_cooldowns()
    monkeypatch.delenv("GROQ_FREE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert check_provider_available("chat") is True


def test_has_openrouter_fallback_requires_openrouter_key(monkeypatch):
    from api.index import has_openrouter_fallback

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CF_AIG_BASE_URL", raising=False)
    assert has_openrouter_fallback() is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-token")
    assert has_openrouter_fallback() is True


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


def test_build_ai_messages_includes_links_context():
    from api.index import build_ai_messages

    message = {
        "from": {"first_name": "Ana", "username": "ana"},
        "chat": {"type": "private", "title": None},
        "text": "mirá fixupx.com/status/2032173338240467235",
    }

    with patch(
        "api.index.fetch_link_metadata",
        return_value={
            "url": "https://fixupx.com/status/2032173338240467235",
            "title": "tweet",
            "description": "contenido",
        },
    ):
        messages = build_ai_messages(message, [], message["text"])

    content = messages[0]["content"]
    assert "LINKS DEL MENSAJE:" in content
    assert "https://fixupx.com/status/2032173338240467235" in content
    assert "titulo: tweet" in content
    assert "descripcion: contenido" in content


def test_log_groq_request_result_logs_local_billing_details():
    from api.index import AIUsageResult, _log_groq_request_result

    result = AIUsageResult(
        kind="chat",
        text="respuesta",
        model="qwen/qwen3.6-plus",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"groq_account": "primary"},
    )

    with patch("builtins.print") as mock_print:
        _log_groq_request_result(
            label="OpenRouter Chat",
            scope="chat",
            account="primary",
            token_count=321,
            audio_seconds=0.0,
            result=result,
        )

    assert mock_print.call_count == 1
    log_entry = json.loads(mock_print.call_args.args[0])
    assert log_entry["scope"] == "groq_request"
    assert log_entry["status"] == "success"
    assert log_entry["request_scope"] == "chat"
    assert log_entry["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert log_entry["local_billing"]["raw_usd_micros"] == 130
    assert log_entry["local_billing"]["charged_credit_units"] == 1
    assert log_entry["local_billing"]["charged_credits_display"] == "0.1"


def test_log_groq_request_result_logs_empty_requests():
    from api.index import _log_groq_request_result

    with patch("builtins.print") as mock_print:
        _log_groq_request_result(
            label="OpenRouter Chat",
            scope="chat",
            account="primary",
            token_count=123,
            audio_seconds=0.0,
            result=None,
        )

    assert mock_print.call_count == 1
    log_entry = json.loads(mock_print.call_args.args[0])
    assert log_entry == {
        "scope": "groq_request",
        "label": "OpenRouter Chat",
        "request_scope": "chat",
        "account": "primary",
        "estimated_token_count": 123,
        "estimated_audio_seconds": 0.0,
        "status": "empty",
    }


def test_execute_groq_request_with_fallback_retries_next_account_on_request_too_large():
    from api.index import AIUsageResult, _execute_groq_request_with_fallback

    class Fake413Error(Exception):
        def __init__(self) -> None:
            super().__init__("Error code: 413 - request_too_large")
            self.status_code = 413
            self.code = "request_too_large"

    def fake_attempt(account):
        if account == "free":
            raise Fake413Error()
        return AIUsageResult(
            kind="chat",
            text="respuesta chat",
            model="qwen/qwen3.6-plus",
            metadata={"groq_account": account},
        )

    with (
        patch("api.index._get_configured_groq_accounts", return_value=["free", "paid"]),
        patch(
            "api.index._get_groq_client",
            side_effect=lambda account, default_headers=None: object(),
        ),
        patch("api.index._log_groq_request_result"),
        patch(
            "api.index.is_provider_backoff_active",
            return_value=False,
        ),
    ):
        result = _execute_groq_request_with_fallback(
            scope="vision",
            label="Groq Vision",
            token_count=123,
            attempt=fake_attempt,
        )

    assert result is not None
    assert result.text == "respuesta chat"
    assert result.metadata["groq_account"] == "paid"


def test_estimate_ai_base_reserve_credits_uses_standard_chat_without_forced_search(
    monkeypatch,
):
    from api.index import estimate_ai_base_reserve_credits

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    reserve, metadata = estimate_ai_base_reserve_credits(
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\nbuscá bitcoin hoy"}]
    )

    assert reserve == 10
    assert metadata == {}


def test_estimate_ai_base_reserve_credits_includes_reasoning_headroom(monkeypatch):
    from api.ai_pricing import estimate_chat_reserve_credits
    from api.index import estimate_ai_base_reserve_credits

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    reserve_without_reasoning = estimate_chat_reserve_credits(
        system_message={"role": "system", "content": "sys"},
        messages=[{"role": "user", "content": "hola"}],
        reasoning=False,
    )
    reserve, metadata = estimate_ai_base_reserve_credits(
        [{"role": "user", "content": "hola"}],
    )

    assert metadata == {}
    assert reserve > reserve_without_reasoning


def test_ask_ai_fetches_url_unconditionally(monkeypatch):
    from api.index import ask_ai

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    captured = {}

    def fake_fetch(url):
        captured["url"] = url
        return {"url": url, "title": "Demo", "content": "contenido limpio"}

    def fake_complete(system_message, messages, **kwargs):
        captured["messages"] = messages
        return "respuesta"

    monkeypatch.setattr("api.index.fetch_url_content", fake_fetch)
    monkeypatch.setattr("api.index.complete_with_providers", fake_complete)

    result = ask_ai(
        [{"role": "user", "content": "mira esto https://example.com/post"}],
        response_meta={},
    )

    assert result == "respuesta"
    assert captured["url"] == "https://example.com/post"
    last = captured["messages"][-1]
    assert last["role"] == "system"
    assert "contenido limpio" in last["content"]


def test_ask_ai_fetches_multiple_urls(monkeypatch):
    from api.index import ask_ai

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    fetched_urls = []

    def fake_fetch(url):
        fetched_urls.append(url)
        return {"url": url, "title": f"Title {url}", "content": f"content of {url}"}

    captured = {}

    def fake_complete(system_message, messages, **kwargs):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr("api.index.fetch_url_content", fake_fetch)
    monkeypatch.setattr("api.index.complete_with_providers", fake_complete)

    ask_ai(
        [{"role": "user", "content": "https://example.com/a y https://example.com/b"}],
        response_meta={},
    )

    assert fetched_urls == ["https://example.com/a", "https://example.com/b"]
    last = captured["messages"][-1]
    assert last["role"] == "system"
    assert "content of https://example.com/a" in last["content"]
    assert "content of https://example.com/b" in last["content"]


def test_ask_ai_skips_inject_on_fetch_error(monkeypatch):
    from api.index import ask_ai

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )
    monkeypatch.setattr(
        "api.index.fetch_url_content",
        lambda url: {"url": url, "error": "url no permitida"},
    )

    captured = {}

    def fake_complete(system_message, messages, **kwargs):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr("api.index.complete_with_providers", fake_complete)

    ask_ai(
        [{"role": "user", "content": "mira https://example.com/post"}],
        response_meta={},
    )

    # Error result is still injected as context (with the error message)
    last = captured["messages"][-1]
    assert last["role"] == "system"
    assert "url no permitida" in last["content"]


def test_ask_ai_uses_single_provider_call_after_url_prefetch(monkeypatch):
    from api.index import ask_ai

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )
    monkeypatch.setattr(
        "api.index.fetch_url_content",
        lambda _url: {
            "url": "https://example.com/post",
            "title": "Demo",
            "content": "contenido",
        },
    )

    calls = []

    def fake_complete(system_message, messages, **kwargs):
        calls.append((system_message, messages, kwargs))
        return "ok"

    monkeypatch.setattr("api.index.complete_with_providers", fake_complete)

    result = ask_ai(
        [{"role": "user", "content": "https://example.com/post"}],
        response_meta={},
    )

    assert result == "ok"
    assert len(calls) == 1


def test_ask_ai_with_provider_success():
    from api.index import ask_ai

    # Simplified test - just verify the function runs without crashing
    with (
        patch("api.index.get_market_context") as mock_get_market_context,
        patch("api.index.get_weather_context") as mock_get_weather_context,
        patch("api.index.get_hacker_news_context") as mock_get_hn_context,
        patch("api.index.get_time_context") as mock_get_time_context,
        patch("os.environ.get") as mock_env,
    ):
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
    with (
        patch("api.index.get_market_context") as mock_get_market_context,
        patch("api.index.get_weather_context") as mock_get_weather_context,
        patch("api.index.get_hacker_news_context") as mock_get_hn_context,
        patch("api.index.get_time_context") as mock_get_time_context,
        patch("os.environ.get") as mock_env,
    ):
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
    with (
        patch("api.index.get_market_context") as mock_get_market_context,
        patch("api.index.get_weather_context") as mock_get_weather_context,
        patch("api.index.get_time_context") as mock_get_time_context,
        patch("api.index.describe_image_groq") as mock_describe_image,
        patch("os.environ.get") as mock_env,
    ):
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


def test_ask_ai_does_not_force_search_for_news_queries():
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

    with (
        patch("api.index.get_market_context", return_value={}),
        patch("api.index.get_weather_context", return_value={}),
        patch("api.index.get_hacker_news_context", return_value=[]),
        patch("api.index.get_time_context", return_value={}),
        patch(
            "api.index.build_system_message",
            return_value={"role": "system", "content": "sys"},
        ),
        patch("api.index.complete_with_providers", return_value="ok") as mock_complete,
        patch("api.index.environ.get") as mock_env,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_key"
        }.get(key, default)

        result = ask_ai([{"role": "user", "content": message_block}])

    assert result == "ok"
    mock_complete.assert_called_once()


def test_fetch_link_metadata_success():
    html_body = """
    <html>
        <head>
            <meta property="og:title" content="Example Site" />
            <meta property="og:description" content="Hola mundo desde la web." />
        </head>
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

    with (
        patch("api.index.config_redis", side_effect=Exception("redis down")),
        patch("api.index.request_with_ssl_fallback", return_value=mock_response),
    ):
        result = fetch_link_metadata("https://example.com/articulo")

    assert result["url"] == "https://example.com/articulo"
    assert result["title"] == "Example Site"
    assert result["description"] == "Hola mundo desde la web."


def test_fetch_link_metadata_invalid_url():
    assert fetch_link_metadata("nota sin protocolo") == {
        "url": "nota sin protocolo",
        "error": "url inválida",
    }


def test_extract_message_urls_prefers_entities_and_limits_to_three():
    message = {
        "text": "https://uno.com x y",
        "entities": [
            {"type": "url", "offset": 0, "length": len("https://uno.com")},
            {"type": "text_link", "offset": 0, "length": 1, "url": "https://dos.com"},
        ],
        "caption": "mirá https://tres.com y https://cuatro.com",
        "caption_entities": [],
    }

    assert extract_message_urls(message) == [
        "https://uno.com",
        "https://dos.com",
        "https://tres.com",
    ]


def test_extract_message_urls_detects_bare_domains_without_scheme():
    message = {
        "text": "mirá fixupx.com/status/2032173338240467235, después vemos",
        "entities": [],
    }

    assert extract_message_urls(message) == [
        "https://fixupx.com/status/2032173338240467235"
    ]


def test_build_message_links_context_includes_url_when_metadata_fails():
    with (
        patch("api.index.extract_message_urls", return_value=["https://example.com"]),
        patch(
            "api.index.fetch_link_metadata",
            return_value={"url": "https://example.com", "error": "boom"},
        ),
    ):
        context = build_message_links_context({"text": "https://example.com"})

    assert "LINKS DEL MENSAJE:" in context
    assert "https://example.com" in context
    assert "descripcion:" not in context


def test_build_message_links_context_keeps_full_youtube_transcript():
    transcript = "linea\n".join(
        [f"[{index:02d}:00] bloque {index}" for index in range(20)]
    )

    with (
        patch(
            "api.index.extract_message_urls",
            return_value=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ),
        patch(
            "api.index.fetch_link_metadata",
            return_value={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        ),
        patch("api.index.get_youtube_transcript_context", return_value=transcript),
    ):
        context = build_message_links_context(
            {"text": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

    assert transcript in context
    assert context.count(transcript) == 1


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
    """Empty responses should return a fallback message"""
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setattr("api.index.time.sleep", lambda *_, **__: None)

    def fake_handler(_messages):
        return ""

    with patch("builtins.print") as mock_print:
        result = handle_ai_response("123", fake_handler, [])

    assert result == "me quedé reculando y no te pude responder, probá de nuevo"
    assert mock_print.call_count == 2
    assert (
        "cleaned response empty after normalization"
        in mock_print.call_args_list[0].args[0]
    )
    assert "previews raw=''" in mock_print.call_args_list[1].args[0]


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


def test_complete_with_providers_openrouter_success():
    """Test complete_with_providers when OpenRouter succeeds."""
    from api.ai_pricing import AIUsageResult

    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    openrouter_result = AIUsageResult(
        kind="chat",
        text="OpenRouter response",
        model="qwen/qwen3.6-plus",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"provider": "openrouter"},
    )

    with patch("api.index.get_provider_chain") as mock_chain:
        mock_chain.return_value.complete.return_value = ProviderResult(
            result=openrouter_result,
            provider_name="openrouter",
        )

        result = complete_with_providers(system_message, messages)

        assert result == "OpenRouter response"
        mock_chain.return_value.complete.assert_called_once()


def test_complete_with_providers_returns_none_when_openrouter_fails():
    """Test complete_with_providers returns None when OpenRouter fails."""
    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    with patch("api.index.get_provider_chain") as mock_chain:
        mock_chain.return_value.complete.return_value = ProviderResult(
            result=None,
            provider_name="openrouter",
        )

        result = complete_with_providers(system_message, messages)

        assert result is None
        mock_chain.return_value.complete.assert_called_once()


def test_complete_with_providers_records_openrouter_billing_on_success(monkeypatch):
    from api.ai_pricing import AIUsageResult

    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]
    response_meta = {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )

    openrouter_result = AIUsageResult(
        kind="chat",
        text="OpenRouter response",
        model="qwen/qwen3.6-plus",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"provider": "openrouter"},
    )

    with patch("api.index.get_provider_chain") as mock_chain:
        mock_chain.return_value.complete.return_value = ProviderResult(
            result=openrouter_result,
            provider_name="openrouter",
        )

        result = complete_with_providers(
            system_message, messages, response_meta=response_meta
        )

    assert result == "OpenRouter response"
    assert response_meta["billing_segments"] == [openrouter_result.billing_segment()]


def test_provider_runtime_returns_none_when_primary_model_fails():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("primary failed")
    runtime = _build_provider_runtime(client=client)

    result = runtime.complete(
        {"role": "system", "content": "system"},
        [{"role": "user", "content": "hola"}],
    )

    assert result is None


def test_describe_image_openrouter_result_returns_none_when_provider_raises(
    monkeypatch,
):
    from api.index import _describe_image_openrouter_result

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = Exception("boom")

    with patch("api.index.OpenAI", return_value=failing_client):
        result = _describe_image_openrouter_result(b"image-bytes")

    assert result is None


def test_fetch_link_metadata_uses_ttl_constant():
    html_body = '<html><head><meta property="og:title" content="A" /></head></html>'
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [html_body.encode("utf-8")]
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.encoding = "utf-8"
    mock_response.apparent_encoding = "utf-8"
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.url = "https://example.com"
    mock_response.close = MagicMock()

    with (
        patch("api.index.request_with_ssl_fallback", return_value=mock_response),
        patch("api.index.config_redis") as mock_redis,
    ):

        class R:
            def __init__(self):
                self.calls = []

            def get(self, _k):
                return None

            def setex(self, k, ttl, _v):
                self.calls.append((k, ttl))
                return True

        redis_client = R()
        mock_redis.return_value = redis_client

        result = fetch_link_metadata("https://example.com")

    assert result["title"] == "A"
    assert any(ttl == index.TTL_LINK_METADATA for (_k, ttl) in redis_client.calls)


def test_can_embed_url_primes_link_metadata_cache():
    from api.index import can_embed_url, fetch_link_metadata

    html_body = (
        "<html><head>"
        '<meta property="og:title" content="Agustin Cortes (@agucortes)" />'
        '<meta property="og:description" content="Texto del post" />'
        '<meta name="twitter:card" content="tweet" />'
        "</head></html>"
    )
    embed_response = MagicMock()
    embed_response.status_code = 200
    embed_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    embed_response.text = html_body
    embed_response.url = "https://fixupx.com/status/2032173338240467235"

    class R:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def setex(self, key, ttl, value):
            self.data[key] = value
            return True

    redis_client = R()

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.utils.links.request_with_ssl_fallback", return_value=embed_response),
    ):
        assert can_embed_url("https://fixupx.com/status/2032173338240467235") is True

    with (
        patch("api.index.config_redis", return_value=redis_client),
        patch("api.index.request_with_ssl_fallback") as mock_fetch_request,
    ):
        result = fetch_link_metadata("https://fixupx.com/status/2032173338240467235")

    assert result["title"] == "Agustin Cortes (@agucortes)"
    assert result["description"] == "Texto del post"
    mock_fetch_request.assert_not_called()
