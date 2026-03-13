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


def test_log_groq_request_result_logs_local_billing_details():
    from api.index import GroqUsageResult, _log_groq_request_result

    result = GroqUsageResult(
        kind="compound",
        text="respuesta",
        model="groq/compound",
        usage=None,
        usage_breakdown=[
            {
                "model": "openai/gpt-oss-120b",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        ],
        executed_tools=[
            {"type": "search", "mode": "basic", "count": 2},
            {"type": "visit", "count": 1},
        ],
        metadata={"groq_account": "primary"},
    )

    with patch("builtins.print") as mock_print:
        _log_groq_request_result(
            label="Groq Compound",
            scope="compound",
            account="primary",
            token_count=321,
            audio_seconds=0.0,
            default_headers={"Groq-Model-Version": "latest"},
            result=result,
        )

    assert mock_print.call_count == 1
    log_entry = json.loads(mock_print.call_args.args[0])
    assert log_entry["scope"] == "groq_request"
    assert log_entry["status"] == "success"
    assert log_entry["request_scope"] == "compound"
    assert log_entry["default_headers"]["Groq-Model-Version"] == "latest"
    assert log_entry["usage_breakdown"][0]["model"] == "openai/gpt-oss-120b"
    assert log_entry["executed_tools"][0]["type"] == "search"
    assert log_entry["local_billing"]["raw_usd_micros"] == 11_045
    assert log_entry["local_billing"]["charged_credit_units"] == 23
    assert log_entry["local_billing"]["charged_credits_display"] == "2.3"


def test_log_groq_request_result_logs_empty_requests():
    from api.index import _log_groq_request_result

    with patch("builtins.print") as mock_print:
        _log_groq_request_result(
            label="Groq Compound",
            scope="compound",
            account="primary",
            token_count=123,
            audio_seconds=0.0,
            default_headers={"Groq-Model-Version": "latest"},
            result=None,
        )

    assert mock_print.call_count == 1
    log_entry = json.loads(mock_print.call_args.args[0])
    assert log_entry == {
        "scope": "groq_request",
        "label": "Groq Compound",
        "request_scope": "compound",
        "account": "primary",
        "estimated_token_count": 123,
        "estimated_audio_seconds": 0.0,
        "default_headers": {"Groq-Model-Version": "latest"},
        "status": "empty",
    }


def test_execute_groq_request_with_fallback_retries_next_account_on_request_too_large():
    from api.index import GroqUsageResult, _execute_groq_request_with_fallback

    class Fake413Error(Exception):
        def __init__(self) -> None:
            super().__init__("Error code: 413 - request_too_large")
            self.status_code = 413
            self.code = "request_too_large"

    reservations = []

    def fake_reserve(*args, **kwargs):
        token = object()
        reservations.append(token)
        return token

    def fake_attempt(account, _client):
        if account == "free":
            raise Fake413Error()
        return GroqUsageResult(
            kind="compound",
            text="respuesta compound",
            model="groq/compound",
            metadata={"groq_account": account},
        )

    with patch("api.index._get_configured_groq_accounts", return_value=["free", "paid"]), patch(
        "api.index._reserve_groq_rate_limit",
        side_effect=fake_reserve,
    ), patch(
        "api.index._get_groq_client",
        side_effect=lambda account, default_headers=None: object(),
    ), patch(
        "api.index._reconcile_groq_rate_limit"
    ) as mock_reconcile, patch(
        "api.index._log_groq_request_result"
    ), patch(
        "api.index.is_provider_backoff_active",
        return_value=False,
    ):
        result = _execute_groq_request_with_fallback(
            scope="compound",
            label="Groq Compound",
            token_count=123,
            default_headers={"Groq-Model-Version": "latest"},
            attempt=fake_attempt,
        )

    assert result is not None
    assert result.text == "respuesta compound"
    assert result.metadata["groq_account"] == "paid"
    assert mock_reconcile.call_count == 2


def test_execute_groq_request_with_fallback_aborts_for_paid_retry_when_time_is_low():
    from api.index import (
        ForceWebhookRetry,
        _execute_groq_request_with_fallback,
        _webhook_force_paid_retry,
        _webhook_operation_key,
        _webhook_redis_client,
        _webhook_request_started_at,
    )

    class Fake413Error(Exception):
        def __init__(self) -> None:
            super().__init__("Error code: 413 - request_too_large")
            self.status_code = 413
            self.code = "request_too_large"

    class FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None, nx=False):
            if nx and key in self.data:
                return False
            self.data[key] = value
            return True

        def setex(self, key, ttl, value):
            self.data[key] = value
            return True

        def delete(self, key):
            self.data.pop(key, None)
            return 1

    redis_client = FakeRedis()
    started_token = _webhook_request_started_at.set(time.monotonic() - 76.0)
    operation_token = _webhook_operation_key.set("message:1:2")
    redis_token = _webhook_redis_client.set(redis_client)
    force_paid_token = _webhook_force_paid_retry.set(False)

    try:
        with patch("api.index._get_configured_groq_accounts", return_value=["free", "paid"]), patch(
            "api.index._reserve_groq_rate_limit", return_value=object()
        ), patch(
            "api.index._get_groq_client", return_value=object()
        ), patch(
            "api.index._reconcile_groq_rate_limit"
        ), patch(
            "api.index._log_groq_request_result"
        ), patch(
            "api.index.is_provider_backoff_active",
            return_value=False,
        ), patch(
            "api.webhook_state._get_webhook_retry_safety_margin_seconds", return_value=45.0
        ), patch(
            "api.webhook_state._get_webhook_max_runtime_seconds", return_value=120.0
        ):
            with pytest.raises(ForceWebhookRetry):
                _execute_groq_request_with_fallback(
                    scope="compound",
                    label="Groq Compound",
                    token_count=123,
                    default_headers={"Groq-Model-Version": "latest"},
                    attempt=lambda account, _client: (_ for _ in ()).throw(Fake413Error())
                    if account == "free"
                    else None,
                )
    finally:
        _webhook_request_started_at.reset(started_token)
        _webhook_operation_key.reset(operation_token)
        _webhook_redis_client.reset(redis_token)
        _webhook_force_paid_retry.reset(force_paid_token)

    assert "webhook:idempotency:message:1:2:force_paid_retry" in redis_client.data


def test_execute_groq_request_with_fallback_skips_free_when_paid_retry_marker_is_active():
    from api.index import (
        GroqUsageResult,
        _execute_groq_request_with_fallback,
        _webhook_force_paid_retry,
    )

    attempted_accounts = []
    force_paid_token = _webhook_force_paid_retry.set(True)

    try:
        with patch("api.index._get_configured_groq_accounts", return_value=["free", "paid"]), patch(
            "api.index._reserve_groq_rate_limit", return_value=object()
        ), patch(
            "api.index._get_groq_client", return_value=object()
        ), patch(
            "api.index._reconcile_groq_rate_limit"
        ), patch(
            "api.index._log_groq_request_result"
        ), patch(
            "api.index.is_provider_backoff_active",
            return_value=False,
        ):
            result = _execute_groq_request_with_fallback(
                scope="compound",
                label="Groq Compound",
                token_count=123,
                default_headers={"Groq-Model-Version": "latest"},
                attempt=lambda account, _client: (
                    attempted_accounts.append(account)
                    or GroqUsageResult(
                        kind="compound",
                        text="respuesta compound",
                        model="groq/compound",
                        metadata={"groq_account": account},
                    )
                ),
            )
    finally:
        _webhook_force_paid_retry.reset(force_paid_token)

    assert attempted_accounts == ["paid"]
    assert result is not None
    assert result.metadata["groq_account"] == "paid"


def test_execute_groq_request_with_fallback_marks_paid_preference_when_escalating_from_free():
    from api.index import (
        GroqUsageResult,
        _execute_groq_request_with_fallback,
        _webhook_force_paid_retry,
        _webhook_operation_key,
        _webhook_redis_client,
        _webhook_request_started_at,
    )

    class Fake413Error(Exception):
        def __init__(self) -> None:
            super().__init__("Error code: 413 - request_too_large")
            self.status_code = 413
            self.code = "request_too_large"

    class FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None, nx=False):
            if nx and key in self.data:
                return False
            self.data[key] = value
            return True

        def setex(self, key, ttl, value):
            self.data[key] = value
            return True

        def delete(self, key):
            self.data.pop(key, None)
            return 1

    redis_client = FakeRedis()
    attempted_accounts = []
    started_token = _webhook_request_started_at.set(time.monotonic())
    operation_token = _webhook_operation_key.set("message:1:3")
    redis_token = _webhook_redis_client.set(redis_client)
    force_paid_token = _webhook_force_paid_retry.set(False)

    try:
        with patch("api.index._get_configured_groq_accounts", return_value=["free", "paid"]), patch(
            "api.index._reserve_groq_rate_limit", return_value=object()
        ), patch(
            "api.index._get_groq_client", return_value=object()
        ), patch(
            "api.index._reconcile_groq_rate_limit"
        ), patch(
            "api.index._log_groq_request_result"
        ), patch(
            "api.index.is_provider_backoff_active",
            return_value=False,
        ), patch(
            "api.webhook_state._get_webhook_retry_safety_margin_seconds", return_value=45.0
        ), patch(
            "api.webhook_state._get_webhook_max_runtime_seconds", return_value=120.0
        ):
            result = _execute_groq_request_with_fallback(
                scope="compound",
                label="Groq Compound",
                token_count=123,
                default_headers={"Groq-Model-Version": "latest"},
                attempt=lambda account, _client: (
                    attempted_accounts.append(account)
                    or (
                        (_ for _ in ()).throw(Fake413Error())
                        if account == "free"
                        else GroqUsageResult(
                            kind="compound",
                            text="respuesta compound",
                            model="groq/compound",
                            metadata={"groq_account": account},
                        )
                    )
                ),
            )
    finally:
        _webhook_request_started_at.reset(started_token)
        _webhook_operation_key.reset(operation_token)
        _webhook_redis_client.reset(redis_token)
        _webhook_force_paid_retry.reset(force_paid_token)

    assert result is not None
    assert attempted_accounts == ["free", "paid"]
    assert (
        "webhook:idempotency:message:1:3:force_paid_retry" in redis_client.data
    )


def test_estimate_ai_base_reserve_credits_uses_standard_chat_without_forced_search(monkeypatch):
    from api.index import estimate_ai_base_reserve_credits

    monkeypatch.setattr("api.index.get_market_context", lambda: {})
    monkeypatch.setattr("api.index.get_weather_context", lambda: {})
    monkeypatch.setattr("api.index.get_time_context", lambda: {"formatted": "Friday"})
    monkeypatch.setattr("api.index.get_hacker_news_context", lambda: [])
    monkeypatch.setattr(
        "api.index.build_system_message",
        lambda _context_data, include_tools=True: {"role": "system", "content": "sys"},
    )
    monkeypatch.setattr("api.index.should_use_groq_compound_tools", lambda: True)
    monkeypatch.setattr(
        "api.index.build_compound_system_message",
        lambda: {"role": "system", "content": "compound"},
    )
    monkeypatch.setattr(
        "api.index.get_groq_compound_enabled_tools",
        lambda: ["web_search", "visit_website", "code_interpreter", "browser_automation"],
    )

    reserve, metadata = estimate_ai_base_reserve_credits(
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\nbuscá bitcoin hoy"}]
    )

    assert reserve == 2
    assert metadata["reserve_mode"] == "chat"
    assert metadata["reserve_reason"] == "standard_chat"
    assert metadata["reserve_model"] == "moonshotai/kimi-k2-instruct-0905"


def test_normalize_search_query_replaces_stale_year_for_current_pricing_query():
    from api.index import _normalize_search_query

    result = _normalize_search_query(
        "costo pañales 3 años argentina 2024 promedio",
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\ncuanto sale comprar pañales por 3 años en promedio?"}],
        now=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    assert result == "costo pañales 3 años argentina 2026 promedio"


def test_normalize_search_query_appends_current_year_for_current_pricing_query():
    from api.index import _normalize_search_query

    result = _normalize_search_query(
        "costo pañales 3 años argentina promedio",
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\ncuanto sale comprar pañales por 3 años en promedio?"}],
        now=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    assert result == "costo pañales 3 años argentina promedio 2026"


def test_normalize_search_query_preserves_explicit_user_year():
    from api.index import _normalize_search_query

    result = _normalize_search_query(
        "costo pañales 3 años argentina 2024 promedio",
        [{"role": "user", "content": "CONTEXTO:\nMENSAJE:\ncuanto salia comprar pañales por 3 años en 2024?"}],
        now=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )

    assert result == "costo pañales 3 años argentina 2024 promedio"


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

    with patch("api.index.get_market_context", return_value={}), patch(
        "api.index.get_weather_context", return_value={}
    ), patch("api.index.get_hacker_news_context", return_value=[]), patch(
        "api.index.get_time_context", return_value={}
    ), patch("api.index.build_system_message",
        return_value={"role": "system", "content": "sys"},
    ), patch(
        "api.index.complete_with_providers", return_value="ok"
    ) as mock_complete, patch(
        "api.index._run_compound_task", return_value="forced"
    ) as mock_run, patch(
        "api.index.environ.get"
    ) as mock_env:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_key"
        }.get(key, default)

        result = ask_ai([{"role": "user", "content": message_block}])

    assert result == "ok"
    mock_run.assert_not_called()
    mock_complete.assert_called_once()


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
                return '[TOOL] compound {"task": "trae contexto"}'
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


def test_search_command_success():
    with patch("api.index.should_use_groq_compound_tools", return_value=True), patch(
        "api.index._run_compound_task", return_value="respuesta compound"
    ) as mock_run:
        result = search_command(
            [{"role": "user", "content": "CONTEXTO:\n\nMENSAJE:\npython programming\n\nINSTRUCCIONES:"}],
            response_meta={},
        )

    assert result == "respuesta compound"
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["persona_pass"] is False
    assert mock_run.call_args.kwargs["task"] == "python programming"


def test_search_command_empty_query():
    result = search_command([])
    assert result == "decime qué querés buscar capo"

    result = search_command([{"role": "user", "content": ""}])
    assert result == "decime qué querés buscar capo"


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

    with patch("api.index.config_redis", side_effect=Exception("redis down")), patch(
        "api.index.request_with_ssl_fallback", return_value=mock_response
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


def test_build_message_links_context_includes_url_when_metadata_fails():
    with patch("api.index.extract_message_urls", return_value=["https://example.com"]), patch(
        "api.index.fetch_link_metadata",
        return_value={"url": "https://example.com", "error": "boom"},
    ):
        context = build_message_links_context({"text": "https://example.com"})

    assert "LINKS DEL MENSAJE:" in context
    assert "https://example.com" in context
    assert "descripcion:" not in context


# Tests for tool calling functionality


def test_parse_tool_call_valid():
    """Test parse_tool_call with valid tool call"""
    text = 'Some text\n[TOOL] compound {"task": "test", "limit": 3}\nMore text'

    result = parse_tool_call(text)

    assert result is not None
    tool_name, args = result
    assert tool_name == "compound"
    assert args == {"task": "test", "limit": 3}


def test_parse_tool_call_multiline():
    """parse_tool_call should handle tool name and JSON on separate lines"""
    text = """
    [TOOL]
    compound
    {"task": "btc asia news"}
    """

    result = parse_tool_call(text)

    assert result == ("compound", {"task": "btc asia news"})


def test_parse_tool_call_with_colon():
    """parse_tool_call should handle optional colon after [TOOL] marker"""
    text = """
    [TOOL]: compound
    {
        "task": "eth dump motivos",
        "limit": 4
    }
    """

    result = parse_tool_call(text)

    assert result is not None
    name, args = result
    assert name == "compound"
    assert args["task"] == "eth dump motivos"
    assert args["limit"] == 4


def test_parse_tool_call_invalid():
    """Test parse_tool_call with invalid inputs"""
    # No tool call
    assert parse_tool_call("Just normal text") is None

    # Malformed JSON
    text = '[TOOL] compound {"task": invalid json}'
    assert parse_tool_call(text) is None

    # Missing arguments
    text = "[TOOL] compound"
    assert parse_tool_call(text) is None

    # None input
    assert parse_tool_call(None) is None


def test_execute_tool_compound():
    with patch("api.index.should_use_groq_compound_tools", return_value=True), patch(
        "api.index._run_compound_task", return_value="resultado compound"
    ) as mock_run:
        result = execute_tool("compound", {"task": "btc hoy"})

    assert result == "resultado compound"
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["persona_pass"] is False


def test_execute_tool_empty_task():
    assert execute_tool("compound", {"task": ""}) == "task vacío"


def test_resolve_tool_calls_compound_reinjects_for_persona_pass():
    from api.index import resolve_tool_calls

    with patch(
        "api.index.execute_tool", return_value="resultado compound"
    ) as mock_tool, patch(
        "api.index.complete_with_providers", return_value="respuesta final"
    ) as mock_complete:
        result = resolve_tool_calls(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            '[TOOL] compound {"task": "test"}',
        )

    assert result == "respuesta final"
    mock_tool.assert_called_once_with("compound", {"task": "test"})
    mock_complete.assert_called_once()
    assert "RESULTADO DE HERRAMIENTA" in mock_complete.call_args.args[1][-1]["content"]


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

    with patch("builtins.print") as mock_print:
        result = handle_ai_response("123", fake_handler, [])

    assert result == "me quedé reculando y no te pude responder, probá de nuevo"
    assert mock_print.call_count == 2
    assert "cleaned response empty after normalization" in mock_print.call_args_list[0].args[0]
    assert "previews raw='[TOOL] web_search" in mock_print.call_args_list[1].args[0]


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
    index_module._provider_backoff_until[
        index_module._get_groq_backoff_key(index_module.GROQ_PAID_ACCOUNT, "chat")
    ] = time.time() + 30

    with patch("api.index.OpenAI") as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    mock_openai.assert_not_called()


def test_get_groq_ai_response_skips_call_when_local_rate_limit_hits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    with patch("api.index._reserve_groq_rate_limit", return_value=None), patch(
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
        remaining = index_module.get_provider_backoff_remaining(
            index_module._get_groq_backoff_key(index_module.GROQ_PAID_ACCOUNT, "chat")
        )
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


def test_get_groq_ai_response_uses_retry_after_header_for_scope_backoff(monkeypatch):
    from api import index as index_module

    class RateLimitError(Exception):
        def __init__(self):
            super().__init__("Error code: 429 - rate limit reached")
            self.status_code = 429
            self.response = MagicMock(headers={"retry-after": "30"})

    index_module._provider_backoff_until.clear()
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RateLimitError()

    with patch("api.index.OpenAI", return_value=fake_client):
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    remaining = index_module.get_provider_backoff_remaining(
        index_module._get_groq_backoff_key(index_module.GROQ_PAID_ACCOUNT, "chat")
    )
    assert 0 < remaining <= 30


def test_get_groq_ai_response_prefers_free_account(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")

    free_choice = MagicMock()
    free_choice.message.content = "free wins"
    free_choice.finish_reason = "stop"

    free_response = MagicMock()
    free_response.choices = [free_choice]

    free_client = MagicMock()
    free_client.chat.completions.create.return_value = free_response

    with patch("api.index.OpenAI", return_value=free_client) as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result == "free wins"
    assert mock_openai.call_count == 1
    assert mock_openai.call_args.kwargs["api_key"] == "free_key"


def test_get_groq_ai_response_falls_back_to_paid_when_free_local_limit_hits(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")

    paid_choice = MagicMock()
    paid_choice.message.content = "paid fallback"
    paid_choice.finish_reason = "stop"

    paid_response = MagicMock()
    paid_response.choices = [paid_choice]

    paid_client = MagicMock()
    paid_client.chat.completions.create.return_value = paid_response

    def reserve_side_effect(account, scope, **kwargs):
        if account == "free" and scope == "chat":
            return None
        return {
            "account": account,
            "scope": scope,
            "request_count": 1,
            "token_count": 100,
            "audio_seconds": 0,
        }

    with patch("api.index._reserve_groq_rate_limit", side_effect=reserve_side_effect), patch(
        "api.index.OpenAI", return_value=paid_client
    ) as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result == "paid fallback"
    assert mock_openai.call_count == 1
    assert mock_openai.call_args.kwargs["api_key"] == "paid_key"


def test_get_groq_ai_response_falls_back_to_paid_after_free_429(monkeypatch):
    from api import index as index_module

    index_module._provider_backoff_until.clear()
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")

    free_client = MagicMock()
    free_client.chat.completions.create.side_effect = Exception(
        "Error code: 429 - rate limit reached"
    )

    paid_choice = MagicMock()
    paid_choice.message.content = "paid fallback"
    paid_choice.finish_reason = "stop"

    paid_response = MagicMock()
    paid_response.choices = [paid_choice]

    paid_client = MagicMock()
    paid_client.chat.completions.create.return_value = paid_response

    with patch("api.index.OpenAI", side_effect=[free_client, paid_client]) as mock_openai:
        result = get_groq_ai_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result == "paid fallback"
    assert mock_openai.call_count == 2
    assert mock_openai.call_args_list[0].kwargs["api_key"] == "free_key"
    assert mock_openai.call_args_list[1].kwargs["api_key"] == "paid_key"
    assert index_module.get_provider_backoff_remaining(
        index_module._get_groq_backoff_key(index_module.GROQ_FREE_ACCOUNT, "chat")
    ) > 0
    assert (
        index_module.get_provider_backoff_remaining(
            index_module._get_groq_backoff_key(index_module.GROQ_PAID_ACCOUNT, "chat")
        )
        == 0
    )


def test_get_groq_compound_response_skips_call_when_local_rate_limit_hits(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")

    with patch("api.index._reserve_groq_rate_limit", return_value=None), patch(
        "api.index.OpenAI"
    ) as mock_openai:
        result = get_groq_compound_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is None
    mock_openai.assert_not_called()


def test_get_groq_compound_response_falls_back_to_paid_after_free_429(monkeypatch):
    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_key")

    free_client = MagicMock()
    free_client.chat.completions.create.side_effect = Exception(
        "Error code: 429 - rate limit reached"
    )

    paid_choice = MagicMock()
    paid_choice.message.content = "compound fallback"
    paid_choice.finish_reason = "stop"

    paid_response = MagicMock()
    paid_response.choices = [paid_choice]

    paid_client = MagicMock()
    paid_client.chat.completions.create.return_value = paid_response

    with patch("api.index.OpenAI", side_effect=[free_client, paid_client]) as mock_openai:
        result = get_groq_compound_response(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "hola"}],
        )

    assert result == "compound fallback"
    assert mock_openai.call_count == 2
    assert mock_openai.call_args_list[0].kwargs["api_key"] == "free_key"
    assert mock_openai.call_args_list[1].kwargs["api_key"] == "paid_key"


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


def test_get_groq_compound_response_result_uses_cache(monkeypatch):
    from api.index import _get_groq_compound_response_result

    cached_segment = {
        "kind": "compound",
        "text": "respuesta cacheada",
        "model": "groq/compound",
        "usage": None,
        "usage_breakdown": [
            {
                "model": "openai/gpt-oss-120b",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        ],
        "executed_tools": [{"type": "search", "count": 1}],
        "audio_seconds": None,
        "cached": False,
        "source": "groq",
        "metadata": {"groq_account": "free"},
    }

    monkeypatch.setattr("api.index._optional_redis_client", lambda **kwargs: object())
    monkeypatch.setattr("api.index.redis_get_json", lambda client, key: cached_segment)

    with patch("api.index._execute_groq_request_with_fallback") as mock_execute:
        result = _get_groq_compound_response_result(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is not None
    assert result.text == "respuesta cacheada"
    assert result.cached is True
    assert result.source == "cache"
    assert result.metadata["compound_cache_hit"] is True
    mock_execute.assert_not_called()


def test_get_groq_compound_response_result_caches_successful_response(monkeypatch):
    from api.index import (
        GroqUsageResult,
        TTL_GROQ_COMPOUND_CACHE,
        _get_groq_compound_response_result,
    )

    monkeypatch.setattr("api.index._optional_redis_client", lambda **kwargs: object())
    monkeypatch.setattr("api.index.redis_get_json", lambda client, key: None)

    stored = {}

    def _fake_setex_json(client, key, ttl, value):
        stored["key"] = key
        stored["ttl"] = ttl
        stored["value"] = value
        return True

    monkeypatch.setattr("api.index.redis_setex_json", _fake_setex_json)

    with patch("api.index._execute_groq_request_with_fallback") as mock_execute:
        mock_execute.return_value = GroqUsageResult(
            kind="compound",
            text="compound ok",
            model="groq/compound",
            usage=None,
            usage_breakdown=[],
            executed_tools=[],
            metadata={"groq_account": "paid"},
        )
        result = _get_groq_compound_response_result(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
        )

    assert result is not None
    assert stored["ttl"] == TTL_GROQ_COMPOUND_CACHE
    assert stored["value"]["text"] == "compound ok"
    assert stored["value"]["source"] == "groq"


def test_run_compound_task_uses_compound_as_source_for_main_model():
    from api.index import _run_compound_task

    with patch(
        "api.index.get_groq_compound_response", return_value="compuesto"
    ) as mock_compound, patch(
        "api.index.complete_with_providers", return_value="respuesta final"
    ) as mock_complete:
        result = _run_compound_task(
            task="algo",
            messages=[{"role": "user", "content": "algo"}],
            system_message={"role": "system", "content": "sys"},
            compound_system_message={"role": "system", "content": "sys"},
        )

    assert result == "respuesta final"
    mock_compound.assert_called_once_with(
        {"role": "system", "content": "sys"},
        [{"role": "user", "content": "algo"}],
    )
    mock_complete.assert_called_once()
    complete_args, complete_kwargs = mock_complete.call_args
    assert complete_args[0]["role"] == "system"
    assert complete_kwargs == {}
    assert complete_args[1][0] == {"role": "user", "content": "algo"}
    assert "FUENTE COMPOUND" in complete_args[1][-1]["content"]
    assert "compuesto" in complete_args[1][-1]["content"]


def test_disable_tools_in_system_message_removes_tool_section():
    from api.index import _disable_tools_in_system_message

    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "base\n\nHERRAMIENTAS DISPONIBLES:\n- web_search\n\nCÓMO LLAMAR HERRAMIENTAS:\n[TOOL] ...",
            }
        ],
    }

    result = _disable_tools_in_system_message(system_message)

    text = result["content"][0]["text"]
    assert "HERRAMIENTAS DISPONIBLES" not in text
    assert "[TOOL]" not in text
    assert "No llames herramientas" in text

def test_run_compound_task_logs_compound_source_text():
    from api.index import _run_compound_task

    with patch(
        "api.index.get_groq_compound_response", return_value="compuesto"
    ), patch(
        "api.index.complete_with_providers", return_value="respuesta final"
    ), patch("builtins.print") as mock_print:
        result = _run_compound_task(
            task="algo",
            messages=[{"role": "user", "content": "algo"}],
            system_message={"role": "system", "content": "sys"},
            compound_system_message={"role": "system", "content": "sys"},
        )

    assert result == "respuesta final"
    printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "_run_compound_task: compound source " in printed
    assert "_run_compound_task: compound source text >>>" in printed
    assert "compuesto" in printed


def test_run_compound_task_logs_when_persona_pass_returns_empty():
    from api.index import _run_compound_task

    with patch(
        "api.index.get_groq_compound_response", return_value="compuesto"
    ), patch(
        "api.index.complete_with_providers", return_value=None
    ), patch("builtins.print") as mock_print:
        result = _run_compound_task(
            task="algo",
            messages=[{"role": "user", "content": "algo"}],
            system_message={"role": "system", "content": "sys"},
            compound_system_message={"role": "system", "content": "sys"},
        )

    assert result == "compuesto"
    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "persona pass returned empty" in printed


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

    with patch("api.index.request_with_ssl_fallback", return_value=mock_response), patch(
        "api.index.config_redis"
    ) as mock_redis:
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
