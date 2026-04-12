from __future__ import annotations

import ipaddress
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from requests.exceptions import RequestException

from api.utils.http import request_with_ssl_fallback

FETCH_TIMEOUT = 8
FETCH_MAX_BYTES = 262144
FETCH_MAX_CHARS = 12000


def _normalize_http_url(url: str) -> Optional[str]:
    raw = str(url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunparse(parsed)


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


def _decode_response_body(
    body: bytes, encoding: Optional[str], fallback: Optional[str]
) -> str:
    try:
        return body.decode(encoding or fallback or "utf-8", errors="replace")
    except Exception:
        return body.decode("utf-8", errors="replace")


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignore_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style"}:
            self._ignore_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def _compress_whitespace(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", unescape(str(value or ""))).strip()


def _extract_html_content(html: str) -> tuple[Optional[str], str]:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    title = _compress_whitespace(" ".join(parser.title_parts)) or None
    text = _compress_whitespace(" ".join(parser.text_parts))
    return title, text


def _truncate_content(content: str) -> tuple[str, bool]:
    if len(content) <= FETCH_MAX_CHARS:
        return content, False
    return content[:FETCH_MAX_CHARS].rstrip(), True


def fetch_url_content(url: str) -> Dict[str, Any]:
    normalized_url = _normalize_http_url(url)
    if not normalized_url:
        return {"url": str(url or "").strip(), "error": "url no permitida"}

    if _is_private_host(urlparse(normalized_url).hostname):
        return {"url": normalized_url, "error": "url no permitida"}

    try:
        response = request_with_ssl_fallback(
            normalized_url,
            allow_redirects=True,
            timeout=FETCH_TIMEOUT,
            stream=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
    except RequestException:
        return {"url": normalized_url, "error": "no se pudo obtener la url"}

    final_url = normalized_url
    content_type = ""
    status_code = getattr(response, "status_code", None)
    body = b""
    body_truncated = False
    try:
        maybe_url = _normalize_http_url(str(getattr(response, "url", "") or ""))
        if maybe_url:
            final_url = maybe_url
        if _is_private_host(urlparse(final_url).hostname):
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
        title, content = _extract_html_content(decoded)
    else:
        title = None
        content = _compress_whitespace(decoded)

    content, text_truncated = _truncate_content(content)
    return {
        "url": final_url,
        "status": status_code,
        "content_type": content_type,
        "title": title,
        "content": content,
        "truncated": body_truncated or text_truncated,
    }
