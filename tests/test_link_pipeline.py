from tests.support import *


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

    service = make_link_service(
        optional_redis_client=lambda: None,
        request_fn=MagicMock(return_value=mock_response),
    )
    result = service.fetch_metadata("https://example.com/articulo")

    assert result["url"] == "https://example.com/articulo"
    assert result["title"] == "Example Site"
    assert result["description"] == "Hola mundo desde la web."


def test_get_hacker_news_context_uses_fallback_url():
    from api.index import get_hacker_news_context

    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Fallback Story</title>
          <link>https://example.com/fallback</link>
          <description><![CDATA[
            <p>Comments URL: <a href="https://news.ycombinator.com/item?id=99">https://news.ycombinator.com/item?id=99</a></p>
            <p>Points: 42</p>
            <p># Comments: 7</p>
          ]]></description>
        </item>
      </channel>
    </rss>"""

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    with patch(
        "api.index.request_with_ssl_fallback",
        side_effect=[
            requests.RequestException("primary timeout"),
            DummyResponse(sample_xml),
        ],
    ) as mock_get, patch("api.index.app_runtime.config.redis", side_effect=RuntimeError("no redis")):
        items = get_hacker_news_context(limit=1)

    assert items == [
        {
            "title": "Fallback Story",
            "url": "https://example.com/fallback",
            "points": 42,
            "comments": 7,
            "comments_url": "https://news.ycombinator.com/item?id=99",
        }
    ]
    assert mock_get.call_count == 2


def test_fetch_link_metadata_invalid_url():
    assert link_service.fetch_metadata("nota sin protocolo") == {
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

    assert link_service.extract_message_urls(message) == [
        "https://uno.com",
        "https://dos.com",
        "https://tres.com",
    ]


def test_extract_message_urls_detects_bare_domains_without_scheme():
    message = {
        "text": "mirá fixupx.com/status/2032173338240467235, después vemos",
        "entities": [],
    }

    assert link_service.extract_message_urls(message) == [
        "https://fixupx.com/status/2032173338240467235"
    ]


def test_build_message_links_context_includes_url_when_metadata_fails():
    with (
        patch.object(
            link_service,
            "extract_message_urls",
            return_value=["https://example.com"],
        ),
        patch.object(
            link_service,
            "fetch_metadata",
            return_value={"url": "https://example.com", "error": "boom"},
        ),
    ):
        context = link_service.build_context({"text": "https://example.com"})

    assert "LINKS DEL MENSAJE:" in context
    assert "https://example.com" in context
    assert "descripcion:" not in context


def test_build_message_links_context_keeps_full_youtube_transcript():
    transcript = "linea\n".join(
        [f"[{index:02d}:00] bloque {index}" for index in range(20)]
    )

    with (
        patch.object(
            link_service,
            "extract_message_urls",
            return_value=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ),
        patch.object(
            link_service,
            "fetch_metadata",
            return_value={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        ),
        patch.object(link_service, "fetch_transcript", return_value=transcript),
    ):
        context = link_service.build_context(
            {"text": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

    assert transcript in context
    assert context.count(transcript) == 1


def test_build_ai_request_reuses_stable_context(monkeypatch):
    from api import index

    calls = {"market": 0, "weather": 0, "hn": 0}

    def market_context():
        calls["market"] += 1
        return {}

    def weather_context():
        calls["weather"] += 1
        return {}

    def hacker_news_context():
        calls["hn"] += 1
        return []

    monkeypatch.setattr(index.app_runtime.ai._deps, "get_market_context", market_context)
    monkeypatch.setattr(index.app_runtime.ai._deps, "get_weather_context", weather_context)
    monkeypatch.setattr(
        index.app_runtime.ai._deps,
        "get_hacker_news_context",
        hacker_news_context,
    )
    monkeypatch.setattr(
        index.app_runtime.ai._deps,
        "get_time_context",
        lambda _offset=-3: {"formatted": "Monday 12:00"},
    )
    monkeypatch.setattr(
        index.app_runtime.ai._deps,
        "fetch_urls",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        index.app_runtime.ai._deps,
        "get_tool_schemas",
        lambda *_args, **_kwargs: [],
    )

    index.app_runtime.ai._stable_context_cache.clear()

    index.app_runtime.ai.build_request([{"role": "user", "content": "hola"}], enable_web_search=False)
    index.app_runtime.ai.build_request([{"role": "user", "content": "chau"}], enable_web_search=False)

    assert calls == {"market": 1, "weather": 1, "hn": 1}


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

    class R:
        def __init__(self):
            self.calls = []

        def get(self, _k):
            return None

        def set(self, key, _value, *, ex=None):
            self.calls.append((key, ex))
            return True

    redis_client = R()
    service = make_link_service(
        optional_redis_client=lambda: redis_client,
        request_fn=MagicMock(return_value=mock_response),
    )
    result = service.fetch_metadata("https://example.com")

    assert result["title"] == "A"
    assert any(ttl == index.TTL_LINK_METADATA for (_k, ttl) in redis_client.calls)


def test_can_embed_url_primes_link_metadata_cache():
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

        def set(self, key, value, *, ex=None):
            self.data[key] = value
            return True

    redis_client = R()

    service = make_link_service(
        optional_redis_client=lambda: redis_client,
        request_fn=MagicMock(),
    )
    with patch(
        "api.utils.links.request_with_ssl_fallback",
        return_value=embed_response,
    ):
        assert service.can_embed(
            "https://fixupx.com/status/2032173338240467235"
        ) is True

    result = service.fetch_metadata(
        "https://fixupx.com/status/2032173338240467235"
    )

    assert result["title"] == "Agustin Cortes (@agucortes)"
    assert result["description"] == "Texto del post"
    service.request_fn.assert_not_called()


def test_build_message_links_context_uses_tweet_content_before_generic_metadata():
    with (
        patch.object(
            link_service,
            "fetch_tweet_content",
            return_value={
                "url": "https://x.com/sentdefender/status/2048202539770802483",
                "author": "OSINTdefender",
                "date": "Apr 26, 2026",
                "text": "Reports of shots fired were unfounded.",
            },
        ) as mock_tweet,
        patch.object(link_service, "fetch_metadata") as mock_metadata,
    ):
        context = link_service.build_context(
            {"text": "https://fixupx.com/status/2048202539770802483"}
        )

    assert "https://x.com/sentdefender/status/2048202539770802483" in context
    assert "autor: OSINTdefender" in context
    assert "tweet: Reports of shots fired were unfounded." in context
    mock_tweet.assert_called_once_with("https://fixupx.com/status/2048202539770802483")
    mock_metadata.assert_not_called()
