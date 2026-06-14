from tests.provider_pipeline_support import *
from tests.provider_pipeline_support import (
    _build_chat_response,
    _build_provider_runtime,
)


def test_openrouter_client_uses_explicit_timeout(monkeypatch):
    from api import index

    captured_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(index.app_runtime.providers, "openai_client_factory", FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("CF_AIG_BASE_URL", raising=False)

    client = index.app_runtime.providers.get_openrouter_client()

    assert client is not None
    assert captured_kwargs["timeout"] == 60.0


def test_strip_markdown_formatting_removes_bold_markers():
    assert strip_markdown_formatting("**hello**") == "hello"
    assert strip_markdown_formatting("__hello__") == "hello"


def test_strip_markdown_formatting_removes_italic_markers():
    assert strip_markdown_formatting("*hello*") == "hello"
    assert strip_markdown_formatting("_hello_") == "hello"


def test_strip_markdown_formatting_removes_header_markers():
    assert strip_markdown_formatting("## Title") == "Title"
    assert strip_markdown_formatting("### Title") == "Title"


def test_strip_markdown_formatting_removes_inline_code_markers():
    assert strip_markdown_formatting("mirá `code`") == "mirá code"


def test_strip_markdown_formatting_removes_code_fences_but_preserves_content():
    text = "antes\n```\nlinea 1\nlinea 2\n```\ndespues"
    assert strip_markdown_formatting(text) == "antes\nlinea 1\nlinea 2\ndespues"


def test_strip_markdown_formatting_keeps_link_text_only():
    assert strip_markdown_formatting("mirá [este link](https://example.com)") == "mirá este link"


def test_strip_markdown_formatting_keeps_image_alt_text_only():
    assert strip_markdown_formatting("mirá ![un gato](https://example.com/cat.png)") == "mirá un gato"


def test_strip_markdown_formatting_removes_horizontal_rules():
    assert strip_markdown_formatting("antes\n---\ndespues\n***") == "antes\ndespues"


def test_strip_markdown_formatting_removes_blockquote_prefix():
    assert strip_markdown_formatting("> citado\nnormal") == "citado\nnormal"


def test_strip_markdown_formatting_removes_list_bullets():
    assert strip_markdown_formatting("- item uno\n* item dos") == "item uno\nitem dos"


def test_strip_markdown_formatting_preserves_underscores_in_urls():
    text = "mirá https://example.com/my_page y user_name"
    assert strip_markdown_formatting(text) == text


def test_strip_markdown_formatting_cleans_mixed_markdown():
    text = "**bold** and *italic*\n## header"
    assert strip_markdown_formatting(text) == "bold and italic\nheader"


def test_strip_markdown_formatting_leaves_non_markdown_text_alone():
    text = "hacé 2 * 3 y dejá snake_case intacto"
    assert strip_markdown_formatting(text) == text


def test_strip_markdown_formatting_returns_empty_string_for_empty_input():
    assert strip_markdown_formatting("") == ""


def test_get_groq_accounts_for_scope_returns_all_configured_accounts(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free-key")
    monkeypatch.setenv("GROQ_API_KEY", "paid-key")

    assert index.app_runtime.providers.get_groq_accounts() == ["free", "paid"]


def test_openrouter_config_helpers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-token")

    with patch("api.index.app_runtime.providers.openai_client_factory") as mock_openai:
        client = index.app_runtime.providers.get_openrouter_client(
            default_headers={"x-test": "1"}
        )

    assert index.app_runtime.providers.get_openrouter_api_key() == "openrouter_key"
    assert (
        index.app_runtime.providers.get_openrouter_base_url()
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
    from api.providers.backoff import clear_all_cooldowns

    clear_all_cooldowns()
    monkeypatch.delenv("GROQ_FREE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert index.app_runtime.providers.is_scope_available("chat") is True


def test_has_openrouter_fallback_requires_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CF_AIG_BASE_URL", raising=False)
    assert index.app_runtime.providers.has_openrouter_fallback() is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/groq"
    )
    monkeypatch.setenv("CF_AIG_TOKEN", "cf-token")
    assert index.app_runtime.providers.has_openrouter_fallback() is True
