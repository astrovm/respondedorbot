from tests.support import *

from api.providers.base import ProviderChain
from api.providers.openrouter import OpenRouterProvider
from api.providers.groq import GroqChatProvider


class FakeProvider:
    def __init__(self, name: str, available: bool = True, result_text: str = ""):
        self._name = name
        self._available = available
        self._result_text = result_text

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def complete(self, system_message, messages, **kwargs):
        from api.ai_pricing import AIUsageResult

        return AIUsageResult(
            kind="chat",
            text=self._result_text,
            model="fake",
            usage={},
            metadata={},
        )


class FakeUnavailableProvider:
    @property
    def name(self) -> str:
        return "unavailable"

    def is_available(self) -> bool:
        return False

    def complete(self, system_message, messages, **kwargs):
        return None


def test_provider_chain_tries_providers_in_order():
    p1 = FakeProvider("p1", available=False, result_text="")
    p2 = FakeProvider("p2", result_text="success")
    chain = ProviderChain([p1, p2])

    result = chain.complete({"role": "system", "content": "test"}, [])

    assert result.result is not None
    assert result.result.text == "success"
    assert result.provider_name == "p2"
    assert result.fallback_used is False


def test_provider_chain_uses_first_available():
    p1 = FakeProvider("p1", result_text="first")
    p2 = FakeProvider("p2", result_text="second")
    chain = ProviderChain([p1, p2])

    result = chain.complete({"role": "system", "content": "test"}, [])

    assert result.result is not None
    assert result.result.text == "first"
    assert result.provider_name == "p1"
    assert result.fallback_used is False


def test_provider_chain_returns_none_when_all_fail():
    p1 = FakeProvider("p1", available=False)
    p2 = FakeProvider("p2", available=False)
    chain = ProviderChain([p1, p2])

    result = chain.complete({"role": "system", "content": "test"}, [])

    assert result.result is None
    assert result.provider_name == "none"


def test_provider_chain_has_any_available():
    chain1 = ProviderChain([FakeUnavailableProvider(), FakeProvider("p")])
    chain2 = ProviderChain([FakeUnavailableProvider(), FakeUnavailableProvider()])

    assert chain1.has_any_available() is True
    assert chain2.has_any_available() is False


def test_openrouter_provider_is_available_when_client_exists():
    def get_client():
        return MagicMock()

    provider = OpenRouterProvider(
        get_client=get_client,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        primary_model="test-model",
    )

    assert provider.is_available() is True
    assert provider.name == "openrouter"


def test_openrouter_provider_not_available_when_client_none():
    provider = OpenRouterProvider(
        get_client=lambda: None,
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        primary_model="test-model",
    )

    assert provider.is_available() is False


def test_groq_provider_not_available_when_cooled_down():
    from api.provider_backoff import mark_provider_cooldown

    mark_provider_cooldown("groq:test", 60)

    provider = GroqChatProvider(
        get_client=lambda: MagicMock(),
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        backoff_key="groq:test",
    )

    assert provider.is_available() is False


def test_groq_provider_available_when_not_cooled_down():
    from api.provider_backoff import clear_provider_cooldown

    clear_provider_cooldown("groq:test2")

    provider = GroqChatProvider(
        get_client=lambda: MagicMock(),
        admin_report=lambda *a, **k: None,
        increment_request_count=lambda: None,
        build_usage_result=lambda **kwargs: MagicMock(),
        extract_usage_map=lambda r: {},
        backoff_key="groq:test2",
    )

    assert provider.is_available() is True
