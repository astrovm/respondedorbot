from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from requests.exceptions import RequestException

from api.agent_tools import normalize_http_url
from api.utils import local_cache_get, update_local_cache


class HtmlMetadataExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._title_parts: List[str] = []
        self._in_title = False
        self.title: Optional[str] = None
        self.description: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attrs_map = {str(key).lower(): value for key, value in attrs}
        if tag_lower == "title":
            self._in_title = True
            return
        if tag_lower != "meta":
            return

        property_name = str(attrs_map.get("property") or "").strip().lower()
        meta_name = str(attrs_map.get("name") or "").strip().lower()
        content = str(attrs_map.get("content") or "").strip()
        if not content:
            return

        normalized_content = re.sub(r"\s+", " ", unescape(content)).strip()
        if not normalized_content:
            return

        if property_name in {"og:title", "twitter:title"} and not self.title:
            self.title = normalized_content
        elif (
            property_name in {"og:description", "twitter:description"}
            and not self.description
        ):
            self.description = normalized_content
        elif meta_name == "description" and not self.description:
            self.description = normalized_content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if not self._in_title:
            return
        text = re.sub(r"\s+", " ", unescape(data)).strip()
        if text:
            self._title_parts.append(text)

    def finalize(self) -> Tuple[Optional[str], Optional[str]]:
        title = self.title or re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()
        return title or None, self.description or None


def extract_html_metadata(html_text: str) -> Tuple[Optional[str], Optional[str]]:
    parser = HtmlMetadataExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass
    return parser.finalize()


def truncate_link_metadata_text(
    text: Optional[str], limit: int = 280
) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def cache_link_metadata(
    raw_url: str,
    metadata: Mapping[str, Any],
    *,
    local_cache: Dict[str, Dict[str, Any]],
    ttl: int,
) -> None:
    normalized = normalize_http_url(raw_url)
    if not normalized:
        return

    cache_payload = {
        "url": str(metadata.get("url") or normalized).strip() or normalized,
        "status": metadata.get("status"),
        "content_type": str(metadata.get("content_type") or ""),
        "title": truncate_link_metadata_text(metadata.get("title"), limit=160),
        "description": truncate_link_metadata_text(metadata.get("description"), limit=280),
    }
    cache_store = local_cache.setdefault(normalized, {})
    update_local_cache(cache_store, cache_payload, ttl, 0)


def fetch_link_metadata(
    raw_url: str,
    *,
    local_cache: Dict[str, Dict[str, Any]],
    ttl: int,
    max_bytes: int,
    optional_redis_client: Callable[[], Optional[Any]],
    hash_cache_key: Callable[[str, Mapping[str, Any]], str],
    request_fn: Callable[..., Any],
    redis_get_json_fn: Callable[[Any, str], Any],
    redis_setex_json_fn: Callable[[Any, str, int, Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    normalized = normalize_http_url(raw_url)
    if not normalized:
        return {"url": str(raw_url or "").strip(), "error": "url inválida"}

    local_cached, is_fresh, _ = local_cache_get(
        local_cache.setdefault(normalized, {}),
        allow_stale=False,
    )
    if is_fresh and isinstance(local_cached, dict):
        return local_cached

    redis_client = optional_redis_client()
    cache_key = hash_cache_key("link_metadata", {"url": normalized})
    if redis_client is not None:
        try:
            cached = redis_get_json_fn(redis_client, cache_key)
            if isinstance(cached, dict):
                cache_link_metadata(
                    normalized, cached, local_cache=local_cache, ttl=ttl
                )
                return cached
        except Exception:
            redis_client = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }

    response = None
    try:
        response = request_fn(
            normalized,
            headers=headers,
            timeout=8,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
    except RequestException as error:
        return {"url": normalized, "error": error.__class__.__name__}

    final_url = normalized
    content_type = ""
    status_code = getattr(response, "status_code", None)
    encoding: Optional[str] = None
    apparent_encoding: Optional[str] = None
    content_bytes = b""
    try:
        maybe_url = normalize_http_url(str(getattr(response, "url", "") or ""))
        if maybe_url:
            final_url = maybe_url
        content_type = str(response.headers.get("Content-Type", "")).lower()
        encoding = response.encoding
        apparent_encoding = getattr(response, "apparent_encoding", None)
        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
        content_bytes = b"".join(chunks)
    finally:
        try:
            response.close()
        except Exception:
            pass

    result: Dict[str, Any] = {
        "url": final_url,
        "status": status_code,
        "content_type": content_type or "",
        "title": None,
        "description": None,
    }

    if content_bytes:
        sample_lower = content_bytes[:400].lower()
        if "html" in content_type or b"<html" in sample_lower or b"<!doctype" in sample_lower:
            try:
                text_body = content_bytes.decode(
                    encoding or apparent_encoding or "utf-8", errors="replace"
                )
            except Exception:
                text_body = content_bytes.decode("utf-8", errors="replace")
            title, description = extract_html_metadata(text_body)
            result["title"] = truncate_link_metadata_text(title, limit=160)
            result["description"] = truncate_link_metadata_text(description, limit=280)

    if redis_client is not None:
        try:
            redis_setex_json_fn(redis_client, cache_key, ttl, result)
        except Exception:
            pass

    cache_link_metadata(normalized, result, local_cache=local_cache, ttl=ttl)
    return result
