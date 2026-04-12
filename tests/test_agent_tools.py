from tests.support import *  # noqa: F401,F403


class _FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
        body: bytes = b"",
        encoding: str = "utf-8",
        apparent_encoding: str = "utf-8",
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.encoding = encoding
        self.apparent_encoding = apparent_encoding
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 4096):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index : index + chunk_size]

    def close(self) -> None:
        return None


def test_fetch_url_content_extracts_main_text_from_html():
    from api.agent_tools import fetch_url_content

    html = b"""
    <html>
      <head>
        <title>Example Title</title>
        <style>.hidden { display: none; }</style>
        <script>window.alert('x');</script>
      </head>
      <body>
        <main>
          <h1>Hola</h1>
          <p>Este es el contenido principal.</p>
        </main>
      </body>
    </html>
    """

    with patch(
        "api.agent_tools.request_with_ssl_fallback",
        return_value=_FakeResponse(
            url="https://example.com/final",
            body=html,
        ),
    ):
        result = fetch_url_content("https://example.com/start")

    assert result == {
        "url": "https://example.com/final",
        "status": 200,
        "content_type": "text/html; charset=utf-8",
        "title": "Example Title",
        "content": "Example Title Hola Este es el contenido principal.",
        "truncated": False,
    }


def test_fetch_url_content_rejects_private_network_urls():
    from api.agent_tools import fetch_url_content

    result = fetch_url_content("http://127.0.0.1:8000/secret")

    assert result == {
        "url": "http://127.0.0.1:8000/secret",
        "error": "url no permitida",
    }


def test_fetch_url_content_truncates_large_pages():
    from api.agent_tools import fetch_url_content

    repeated = ("palabra " * 4000).encode()
    html = (
        b"<html><head><title>Largo</title></head><body><p>"
        + repeated
        + b"</p></body></html>"
    )

    with patch(
        "api.agent_tools.request_with_ssl_fallback",
        return_value=_FakeResponse(
            url="https://example.com/large",
            body=html,
        ),
    ):
        result = fetch_url_content("https://example.com/large")

    assert result["url"] == "https://example.com/large"
    assert result["status"] == 200
    assert result["content_type"] == "text/html; charset=utf-8"
    assert result["title"] == "Largo"
    assert result["truncated"] is True
    assert len(result["content"]) <= 12003
    assert result["content"].startswith("Largo palabra palabra palabra")
