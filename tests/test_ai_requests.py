from tests.provider_pipeline_support import *
from tests.provider_pipeline_support import _build_provider_runtime


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


def test_build_ai_messages_preserves_full_telegram_reply_context():
    from api.index import build_ai_messages

    message = {
        "from": {"first_name": "Ana", "username": "ana"},
        "chat": {"type": "group", "title": "Grupo"},
        "text": "¿qué pensás?",
    }
    reply_context = "Polymarket elections\n" + ("candidate odds\n" * 150)

    messages = build_ai_messages(
        message,
        [],
        "¿qué pensás?",
        reply_context,
    )

    assert reply_context in messages[0]["content"]


def test_build_ai_messages_preserves_full_telegram_user_message():
    from api.index import build_ai_messages

    message_text = "question context\n" + ("long question\n" * 150)
    message = {
        "from": {"first_name": "Ana", "username": "ana"},
        "chat": {"type": "group", "title": "Grupo"},
        "text": message_text,
    }

    messages = build_ai_messages(message, [], message_text)

    assert message_text in messages[0]["content"]


def test_build_ai_messages_includes_links_context():
    from api.index import build_ai_messages

    message = {
        "from": {"first_name": "Ana", "username": "ana"},
        "chat": {"type": "private", "title": None},
        "text": "mirá fixupx.com/status/2032173338240467235",
    }

    with (
        patch.object(index._link_service, "fetch_tweet_content", return_value=None),
        patch.object(
            index._link_service,
            "fetch_metadata",
            return_value={
                "url": "https://fixupx.com/status/2032173338240467235",
                "title": "tweet",
                "description": "contenido",
            },
        ),
    ):
        messages = build_ai_messages(message, [], message["text"])

    content = messages[0]["content"]
    assert "LINKS DEL MENSAJE:" in content
    assert "https://fixupx.com/status/2032173338240467235" in content
    assert "titulo: tweet" in content
    assert "descripcion: contenido" in content


def test_log_groq_request_result_logs_local_billing_details():
    from api.index import AIUsageResult

    result = AIUsageResult(
        kind="chat",
        text="respuesta",
        model="deepseek/deepseek-v4-flash",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"groq_account": "primary"},
    )

    with patch("api.index._logger.info") as mock_log:
        index.app_runtime.providers.log_groq_request_result(
            label="OpenRouter Chat",
            scope="chat",
            account="primary",
            token_count=321,
            audio_seconds=0.0,
            result=result,
        )

    assert mock_log.call_count == 1
    log_entry = json.loads(mock_log.call_args.args[1])
    assert log_entry["scope"] == "groq_request"
    assert log_entry["status"] == "success"
    assert log_entry["request_scope"] == "chat"
    assert log_entry["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert log_entry["local_billing"]["raw_usd_micros"] == 100
    assert log_entry["local_billing"]["charged_credit_units"] == 1
    assert log_entry["local_billing"]["charged_credits_display"] == "0.1"


def test_log_groq_request_result_logs_empty_requests():
    with patch("api.index._logger.info") as mock_log:
        index.app_runtime.providers.log_groq_request_result(
            label="OpenRouter Chat",
            scope="chat",
            account="primary",
            token_count=123,
            audio_seconds=0.0,
            result=None,
        )

    assert mock_log.call_count == 1
    log_entry = json.loads(mock_log.call_args.args[1])
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
    from api.index import AIUsageResult

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
            model="deepseek/deepseek-v4-flash",
            metadata={"groq_account": account},
        )

    with (
        patch(
            "api.index.app_runtime.media._deps.get_groq_accounts",
            return_value=["free", "paid"],
        ),
        patch("api.index.app_runtime.media._deps.log_result"),
        patch(
            "api.index.app_runtime.media._deps.is_backoff_active",
            return_value=False,
        ),
    ):
        result = index.app_runtime.media.execute_groq_request(
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
    monkeypatch.setattr("api.index.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    reserve, metadata = estimate_ai_base_reserve_credits(
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\nbuscá bitcoin hoy"}]
    )

    assert reserve == 3
    assert metadata == {}


def test_ask_ai_fetches_url_unconditionally(monkeypatch):
    ask_ai = index.app_runtime.ai.ask

    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.build_system_message",
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
    monkeypatch.setattr("api.index.app_runtime.ai._deps.complete", fake_complete)

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
    ask_ai = index.app_runtime.ai.ask

    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.build_system_message",
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
    monkeypatch.setattr("api.index.app_runtime.ai._deps.complete", fake_complete)

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
    ask_ai = index.app_runtime.ai.ask

    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.build_system_message",
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

    monkeypatch.setattr("api.index.app_runtime.ai._deps.complete", fake_complete)

    ask_ai(
        [{"role": "user", "content": "mira https://example.com/post"}],
        response_meta={},
    )

    # Error result is still injected as context (with the error message)
    last = captured["messages"][-1]
    assert last["role"] == "system"
    assert "url no permitida" in last["content"]


def test_ask_ai_uses_single_provider_call_after_url_prefetch(monkeypatch):
    ask_ai = index.app_runtime.ai.ask

    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.build_system_message",
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

    monkeypatch.setattr("api.index.app_runtime.ai._deps.complete", fake_complete)

    result = ask_ai(
        [{"role": "user", "content": "https://example.com/post"}],
        response_meta={},
    )

    assert result == "ok"
    assert len(calls) == 1


def test_ask_ai_with_provider_success():
    ask_ai = index.app_runtime.ai.ask

    # Simplified test - just verify the function runs without crashing
    with (
        patch("api.index.app_runtime.ai._deps.get_market_context") as mock_get_market_context,
        patch("api.index.app_runtime.ai._deps.get_weather_context") as mock_get_weather_context,
        patch("api.index.app_runtime.ai._deps.get_hacker_news_context") as mock_get_hn_context,
        patch("api.index.app_runtime.ai._deps.get_time_context") as mock_get_time_context,
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
    ask_ai = index.app_runtime.ai.ask

    # Simplified test - just verify the function runs without crashing
    with (
        patch("api.index.app_runtime.ai._deps.get_market_context") as mock_get_market_context,
        patch("api.index.app_runtime.ai._deps.get_weather_context") as mock_get_weather_context,
        patch("api.index.app_runtime.ai._deps.get_hacker_news_context") as mock_get_hn_context,
        patch("api.index.app_runtime.ai._deps.get_time_context") as mock_get_time_context,
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
    ask_ai = index.app_runtime.ai.ask

    # Simplified test - just verify the function runs without crashing when given an image
    with (
        patch("api.index.app_runtime.ai._deps.get_market_context") as mock_get_market_context,
        patch("api.index.app_runtime.ai._deps.get_weather_context") as mock_get_weather_context,
        patch("api.index.app_runtime.ai._deps.get_time_context") as mock_get_time_context,
        patch("api.index.app_runtime.media.describe_image") as mock_describe_image,
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


def test_ask_ai_skips_image_injection_when_image_data_is_none(monkeypatch):
    ask_ai = index.app_runtime.ai.ask

    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_time_context", lambda _offset=-3: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.app_runtime.ai._deps.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.build_system_message",
        lambda _context_data, **_kw: {"role": "system", "content": "sys"},
    )

    inject_calls = []

    def fake_inject_image_context(messages, image_data, image_file_id, response_meta):
        inject_calls.append((messages, image_data, image_file_id, response_meta))

    monkeypatch.setattr("api.index.app_runtime.ai.inject_image_context", fake_inject_image_context)
    monkeypatch.setattr(
        "api.index.app_runtime.ai._deps.complete",
        lambda *_args, **_kwargs: "ok",
    )

    result = ask_ai([{"role": "user", "content": "hola"}], image_data=None)

    assert result == "ok"
    assert inject_calls == []


def test_ask_ai_does_not_force_search_for_news_queries():
    ask_ai = index.app_runtime.ai.ask

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
        patch("api.index.app_runtime.ai._deps.get_market_context", return_value={}),
        patch("api.index.app_runtime.ai._deps.get_weather_context", return_value={}),
        patch("api.index.app_runtime.ai._deps.get_hacker_news_context", return_value=[]),
        patch("api.index.app_runtime.ai._deps.get_time_context", return_value={}),
        patch(
            "api.index.app_runtime.ai._deps.build_system_message",
            return_value={"role": "system", "content": "sys"},
        ),
        patch("api.index.app_runtime.ai._deps.complete", return_value="ok") as mock_complete,
        patch("api.index.environ.get") as mock_env,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_key"
        }.get(key, default)

        result = ask_ai([{"role": "user", "content": message_block}])

    assert result == "ok"
    mock_complete.assert_called_once()


# Tests for complete_with_providers function


def test_complete_with_providers_openrouter_success():
    """Test complete_with_providers when OpenRouter succeeds."""
    from api.ai.pricing import AIUsageResult

    system_message = {"role": "system", "content": "test"}
    messages = [{"role": "user", "content": "hello"}]

    openrouter_result = AIUsageResult(
        kind="chat",
        text="OpenRouter response",
        model="deepseek/deepseek-v4-flash",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"provider": "openrouter"},
    )

    with patch("api.index.app_runtime.providers.get_chain") as mock_chain:
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

    with patch("api.index.app_runtime.providers.get_chain") as mock_chain:
        mock_chain.return_value.complete.return_value = ProviderResult(
            result=None,
            provider_name="openrouter",
        )

        result = complete_with_providers(system_message, messages)

        assert result is None
        mock_chain.return_value.complete.assert_called_once()


def test_complete_with_providers_records_openrouter_billing_on_success(monkeypatch):
    from api.ai.pricing import AIUsageResult

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
        model="deepseek/deepseek-v4-flash",
        usage={"input_tokens": 100, "output_tokens": 50},
        metadata={"provider": "openrouter"},
    )

    with patch("api.index.app_runtime.providers.get_chain") as mock_chain:
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


def test_describe_image_result_returns_none_when_provider_raises(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter"
    )

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = Exception("boom")

    with patch(
        "api.index.app_runtime.media._deps.get_openrouter_client",
        return_value=failing_client,
    ):
        result = index.app_runtime.media.describe_image_result(b"image-bytes")

    assert result is None


def test_describe_image_result_skips_provider_when_image_is_invalid(
    monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter_key")
    monkeypatch.setenv(
        "CF_AIG_BASE_URL", "https://gateway.ai.cloudflare.com/v1/acct/gw/openrouter"
    )

    client = MagicMock()

    with patch(
        "api.index.app_runtime.media._deps.get_openrouter_client",
        return_value=client,
    ):
        result = index.app_runtime.media.describe_image_result(
            b"not-an-image",
            file_id="sticker-file",
        )

    assert result is None
    client.chat.completions.create.assert_not_called()
