from tests.support import *


def test_get_market_context_success():
    from api.index import get_market_context

    with (
        patch("api.index.app_runtime.cache.request") as mock_cached,
        patch("api.index.clean_crypto_data") as mock_clean,
        patch("os.environ.get") as mock_env,
    ):
        # Mock crypto response
        crypto_response = {
            "data": {"data": [{"symbol": "BTC", "quote": {"USD": {"price": 50000}}}]}
        }

        # Mock dollar response
        dollar_response = {
            "data": {"oficial": {"price": 1000}, "blue": {"price": 1200}}
        }

        def mock_requests_side_effect(url, *_args, **_kwargs):
            if "coinmarketcap" in url:
                return crypto_response
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_clean.return_value = [{"symbol": "BTC", "price": 50000}]
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" in result
        assert "dollar" in result
        assert result["crypto"] == [{"symbol": "BTC", "price": 50000}]
        assert result["dollar"] == {"oficial": {"price": 1000}, "blue": {"price": 1200}}


def test_get_market_context_crypto_fail():
    from api.index import get_market_context

    with (
        patch("api.index.app_runtime.cache.request") as mock_cached,
        patch("os.environ.get") as mock_env,
    ):
        # Mock dollar response only
        dollar_response = {"data": {"oficial": {"price": 1000}}}

        def mock_requests_side_effect(url, *_args, **_kwargs):
            if "coinmarketcap" in url:
                return None  # Crypto fails
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" not in result
        assert "dollar" in result


def test_get_market_context_all_fail():
    from api.index import get_market_context

    with (
        patch("api.index.app_runtime.cache.request") as mock_cached,
        patch("os.environ.get") as mock_env,
        patch("api.index.app_runtime.bcra.get_cached_variables") as mock_get_bcra,
        patch("api.index.app_runtime.bcra.fetch_latest_variables") as mock_fetch_bcra,
        patch("api.index.app_runtime.bcra.cache_variables") as mock_cache_bcra,
    ):
        mock_cached.return_value = None
        mock_env.return_value = "test_api_key"
        mock_get_bcra.return_value = None
        mock_fetch_bcra.return_value = None
        mock_cache_bcra.return_value = None

        result = get_market_context()

        assert result == {}


def test_get_weather_context_success():
    from api.index import get_weather_context

    with (
        patch("api.index.get_weather") as mock_weather,
        patch("api.index.get_weather_description") as mock_description,
    ):
        mock_weather.return_value = {"temperature": 25.0, "weather_code": 0}
        mock_description.return_value = "cielo despejado"

        result = get_weather_context()

        assert result is not None
        assert result["temperature"] == 25.0
        assert result["weather_code"] == 0
        assert result["description"] == "cielo despejado"
        mock_description.assert_called_once_with(0)


def test_get_weather_context_fail():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.return_value = None

        result = get_weather_context()

        assert result is None


def test_get_weather_context_exception():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.side_effect = Exception("Weather API error")

        result = get_weather_context()

        assert result is None


def test_get_time_context():
    from api.index import get_time_context
    from datetime import datetime, timezone, timedelta

    with patch("api.index.datetime") as mock_datetime:
        # Mock a fixed time
        fixed_time = datetime(2024, 1, 15, 14, 30, 0)
        buenos_aires_tz = timezone(timedelta(hours=-3))
        fixed_time_ba = fixed_time.replace(tzinfo=buenos_aires_tz)

        mock_datetime.now.return_value = fixed_time_ba
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        result = get_time_context(timezone_offset=-3)

        assert "datetime" in result
        assert "formatted" in result
        assert result["datetime"] == fixed_time_ba


def test_get_hacker_news_context_success():
    from api.index import get_hacker_news_context

    sample_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <rss version=\"2.0\">
      <channel>
        <item>
          <title>Historia Uno</title>
          <link>https://example.com/uno</link>
          <description><![CDATA[
            <p>Article URL: <a href=\"https://example.com/uno\">https://example.com/uno</a></p>
            <p>Comments URL: <a href=\"https://news.ycombinator.com/item?id=1\">https://news.ycombinator.com/item?id=1</a></p>
            <p>Points: 123</p>
            <p># Comments: 45</p>
          ]]></description>
        </item>
        <item>
          <title>Historia Dos</title>
          <link>https://example.com/dos</link>
          <description><![CDATA[
            <p>Comments URL: <a href=\"https://news.ycombinator.com/item?id=2\">https://news.ycombinator.com/item?id=2</a></p>
            <p>Points: 456</p>
            <p># Comments: 78</p>
          ]]></description>
        </item>
      </channel>
    </rss>"""

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    with (
        patch("api.index.app_runtime.config.redis", side_effect=RuntimeError("no redis")),
        patch(
            "api.utils.http.requests.get", return_value=DummyResponse(sample_xml)
        ) as mock_get,
    ):
        items = get_hacker_news_context(limit=2)

    assert len(items) == 2
    assert items[0]["title"] == "Historia Uno"
    assert items[0]["points"] == 123
    assert items[0]["comments"] == 45
    assert items[0]["comments_url"].endswith("id=1")
    assert items[1]["title"] == "Historia Dos"
    assert items[1]["points"] == 456
    assert items[1]["comments"] == 78
    mock_get.assert_called_once()


def test_get_hacker_news_context_uses_cache():
    from api.index import get_hacker_news_context

    cached_items = [
        {
            "title": "Cacheada",
            "url": "https://cached.example",
            "points": 50,
            "comments": 5,
            "comments_url": "https://news.ycombinator.com/item?id=3",
        }
    ]

    with (
        patch("api.index.app_runtime.config.redis", return_value=object()),
        patch("api.index.redis_get_json", return_value=cached_items) as mock_cache,
        patch("api.index.http_client.get") as mock_get,
    ):
        items = get_hacker_news_context(limit=1)

    assert items == cached_items[:1]
    mock_cache.assert_called_once()
    mock_get.assert_not_called()


def test_get_fallback_response():
    from api.index import get_fallback_response

    messages = [{"role": "user", "content": "hello"}]

    result = get_fallback_response(messages)

    # Should return a string (one of many predefined fallback responses)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) < 50  # Reasonable length for a fallback response


def test_build_system_message():
    from api.index import build_system_message

    # Reset global cache to ensure clean state
    config_module.reset_cache()

    context = {
        "market": {
            "crypto": [{"symbol": "BTC", "price": 50000}],
            "dollar": {"oficial": {"price": 1000}},
        },
        "weather": {
            "temperature": 25,
            "apparent_temperature": 26,
            "precipitation_probability": 10,
            "description": "cielo despejado",
            "cloud_cover": 20,
            "visibility": 10000,
        },
        "time": {"formatted": "Monday 15/01/2024"},
        "hacker_news": [
            {
                "title": "Nueva feature",
                "url": "https://example.com/hn",
                "points": 321,
                "comments": 42,
                "comments_url": "https://news.ycombinator.com/item?id=1",
            }
        ],
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "Sos el gordo, un bot argentino de prueba.",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0
        content_text = result["content"][0]["text"]
        assert "argentin" in content_text.lower()
        assert "gordo" in content_text.lower()
        assert "hacker news" in content_text.lower()
        assert "CAPACIDADES DEL BOT" in content_text
        assert "$ticker" in content_text
        assert "/buscar y /search no existen" in content_text


def test_build_system_message_empty_context():
    from api.index import build_system_message

    # Reset global cache to ensure clean state
    config_module.reset_cache()

    context = {
        "market": {},
        "weather": None,
        "time": {"formatted": "Monday"},
        "hacker_news": [],
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "You are a test bot assistant.",
                "BOT_TRIGGER_WORDS": "test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        # Should still have base personality
        content_text = result["content"][0]["text"]
        assert len(content_text) > 100


def test_build_system_message_task_tool_instructions_preserve_perspective():
    from api.index import build_system_message

    context = {
        "market": {},
        "weather": None,
        "time": {"formatted": "Monday"},
        "hacker_news": [],
    }

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "task_set",
                "description": "Create a scheduled task.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    result = build_system_message(context, tools_active=True, tool_schemas=tool_schemas)

    content_text = result["content"][0]["text"]
    assert (
        "task_set.text debe contener solo el contenido a ejecutar despues"
        in content_text
    )
    assert "no reescribas pronombres ni cambies sujeto" in content_text
    assert "no incluyas tiempo ni frecuencia en text" in content_text
    assert 'text="decime cuanta aura farmeaste hoy"' in content_text
    assert 'text="deci fumareeemooss"' in content_text
