from __future__ import annotations

import ipaddress
import re
import socket
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

from requests.exceptions import RequestException

from api.utils.http import request_with_ssl_fallback

FETCH_TIMEOUT = 8
FETCH_MAX_BYTES = 262144
FETCH_MAX_CHARS = 12000
FETCH_MAX_REDIRECTS = 5


def normalize_http_url(raw_url: str) -> Optional[str]:
    """Normalize raw URL strings to HTTP/HTTPS form without fragments.

    Prepends https:// for scheme-less inputs. Rejects URLs with spaces in
    the netloc or non-HTTP/S schemes.
    """
    candidate = str(raw_url or "").strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    if not parsed.scheme:
        parsed = urlparse(f"https://{candidate}")

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    if any(char.isspace() for char in parsed.netloc):
        return None

    return urlunparse(parsed._replace(fragment=""))


def _is_private_host(hostname: Optional[str]) -> bool:
    host = str(hostname or "").strip().lower()
    if not host:
        return True
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _hostname_resolves_private(hostname: Optional[str]) -> bool:
    host = str(hostname or "").strip()
    if not host:
        return True
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for entry in addresses:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        if _is_private_host(sockaddr[0]):
            return True
    return False


def _is_blocked_url(url: str) -> bool:
    hostname = urlparse(url).hostname
    return _is_private_host(hostname) or _hostname_resolves_private(hostname)


def _extract_redirect_url(base_url: str, response: Any) -> Optional[str]:
    location = str(response.headers.get("Location", "") or "").strip()
    if not location:
        return None
    redirected = normalize_http_url(urljoin(base_url, location))
    if not redirected:
        return None
    return redirected


def _decode_response_body(
    body: bytes, encoding: Optional[str], fallback: Optional[str]
) -> str:
    try:
        return body.decode(encoding or fallback or "utf-8", errors="replace")
    except Exception:
        return body.decode("utf-8", errors="replace")


class HtmlTextExtractor(HTMLParser):
    """Extract visible text and title from HTML documents."""

    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "li",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._buffer: List[str] = []
        self._skip_depth = 0
        self._title_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._in_title = True
            return
        if tag_lower in self._BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag_lower == "title":
            self._in_title = False
            return
        if tag_lower in self._BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = unescape(data)
        if self._in_title:
            self._title_parts.append(text)
            # Title text is intentionally also buffered so get_text() includes it.
        cleaned = text.strip()
        if not cleaned:
            return
        if self._buffer and not self._buffer[-1].endswith((" ", "\n")):
            self._buffer.append(" ")
        self._buffer.append(cleaned)

    def get_text(self) -> str:
        raw = "".join(self._buffer)
        collapsed = re.sub(r"[ \t]+", " ", raw)
        collapsed = re.sub(r"\n\s*", "\n", collapsed)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    def get_title(self) -> Optional[str]:
        title = "".join(self._title_parts).strip()
        return re.sub(r"\s+", " ", title) or None


def extract_text_from_html(html: str) -> Tuple[Optional[str], str]:
    """Return (title, visible_text) extracted from an HTML string."""
    parser = HtmlTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return parser.get_title(), parser.get_text()


def _truncate_content(content: str) -> tuple[str, bool]:
    if len(content) <= FETCH_MAX_CHARS:
        return content, False
    return content[:FETCH_MAX_CHARS].rstrip(), True


def fetch_url_content(url: str) -> Dict[str, Any]:
    normalized_url = normalize_http_url(url)
    if not normalized_url:
        return {"url": str(url or "").strip(), "error": "url no permitida"}

    if _is_blocked_url(normalized_url):
        return {"url": normalized_url, "error": "url no permitida"}

    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    current_url = normalized_url
    response = None
    for _ in range(FETCH_MAX_REDIRECTS + 1):
        try:
            response = request_with_ssl_fallback(
                current_url,
                allow_redirects=False,
                timeout=FETCH_TIMEOUT,
                stream=True,
                headers=request_headers,
            )
            response.raise_for_status()
        except RequestException:
            return {"url": current_url, "error": "no se pudo obtener la url"}

        redirected_url = _extract_redirect_url(current_url, response)
        if not redirected_url:
            break
        try:
            response.close()
        except Exception:
            pass
        if _is_blocked_url(redirected_url):
            return {"url": redirected_url, "error": "url no permitida"}
        current_url = redirected_url
    else:
        return {"url": current_url, "error": "no se pudo obtener la url"}

    if response is None or getattr(response, "status_code", 0) >= 300:
        return {"url": current_url, "error": "no se pudo obtener la url"}

    final_url = current_url
    content_type = ""
    status_code = getattr(response, "status_code", None)
    body = b""
    body_truncated = False
    try:
        maybe_url = normalize_http_url(str(getattr(response, "url", "") or ""))
        if maybe_url:
            final_url = maybe_url
        if _is_blocked_url(final_url):
            return {"url": final_url, "error": "url no permitida"}

        content_type = str(response.headers.get("Content-Type", "")).lower()
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            remaining = FETCH_MAX_BYTES - total
            if remaining <= 0:
                body_truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                body_truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        body = b"".join(chunks)
    finally:
        try:
            response.close()
        except Exception:
            pass

    decoded = _decode_response_body(
        body,
        getattr(response, "encoding", None),
        getattr(response, "apparent_encoding", None),
    )
    is_html = "html" in content_type or b"<html" in body[:400].lower()
    if is_html:
        title, content = extract_text_from_html(decoded)
    else:
        title = None
        content = re.sub(r"\s+", " ", unescape(decoded)).strip()

    content, text_truncated = _truncate_content(content)
    return {
        "url": final_url,
        "status": status_code,
        "content_type": content_type,
        "title": title,
        "content": content,
        "truncated": body_truncated or text_truncated,
    }
