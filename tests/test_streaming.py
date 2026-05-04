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


def test_openrouter_stream_uses_complete_path_when_tools_present():
    from api.ai_pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider

    client = MagicMock()
    client.chat.completions.create.side_effect = AssertionError("native stream path used")
    provider = OpenRouterProvider(
        get_client=lambda: client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        primary_model="test-model",
    )
    provider._runtime.complete = MagicMock(
        return_value=AIUsageResult(
            kind="chat",
            text="hola final",
            model="test-model",
            usage={},
        )
    )

    chunks = list(
        provider.stream(
            {"role": "system", "content": "sys"},
            [{"role": "user", "content": "hola"}],
            enable_web_search=False,
            extra_tools=[{"type": "function", "function": {"name": "echo"}}],
        )
    )

    assert chunks == ["hola", " fin", "al"]
    provider._runtime.complete.assert_called_once()
    client.chat.completions.create.assert_not_called()


def test_openrouter_stream_uses_web_search_branch_when_enabled():
    from types import SimpleNamespace

    from api.ai_pricing import AIUsageResult
    from api.providers.openrouter import OpenRouterProvider

    create_calls: list[dict[str, Any]] = []

    def create(**kwargs):
        create_calls.append(kwargs)
        return iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="web"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" answer"))]),
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
            enable_web_search=True,
        )
    )

    assert chunks == ["web", " answer"]
    assert_no_raw_tool_syntax("".join(chunks))
    assert create_calls[0]["tools"] == [{"type": "web_search"}]


def test_stream_to_telegram_sends_first_token_without_placeholder():
    from api.streaming import stream_to_telegram

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
    from api.streaming import stream_to_telegram

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
