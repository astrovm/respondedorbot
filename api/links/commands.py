from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from requests.exceptions import RequestException

from api.links.agent_tools import normalize_http_url
from api.utils import local_cache_get, update_local_cache
from api.utils.bounded_http import decode_bounded_body, read_bounded_response


@dataclass(frozen=True, slots=True)
class LinkMetadataDeps:
    local_cache: Dict[str, Dict[str, Any]]
    ttl: int
    max_bytes: int
    optional_redis_client: Callable[[], Optional[Any]]
    hash_cache_key: Callable[[str, Mapping[str, Any]], str]
    request: Callable[..., Any]
    redis_get_json: Callable[[Any, str], Any]
    redis_setex_json: Callable[[Any, str, int, Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class MetadataCacheLookup:
    cache_key: str
    redis_client: Any | None
    value: Dict[str, Any] | None = None


def _load_metadata_cache(
    normalized: str,
    deps: LinkMetadataDeps,
) -> MetadataCacheLookup:
    local_cached, is_fresh, _ = local_cache_get(
        deps.local_cache.setdefault(normalized, {}),
        allow_stale=False,
    )
    cache_key = deps.hash_cache_key("link_metadata", {"url": normalized})
    if is_fresh and isinstance(local_cached, dict):
        return MetadataCacheLookup(cache_key, None, local_cached)

    redis_client = deps.optional_redis_client()
    if redis_client is None:
        return MetadataCacheLookup(cache_key, None)
    try:
        cached = deps.redis_get_json(redis_client, cache_key)
    except Exception:
        return MetadataCacheLookup(cache_key, None)
    if not isinstance(cached, dict):
        return MetadataCacheLookup(cache_key, redis_client)
    cache_link_metadata(
        normalized,
        cached,
        local_cache=deps.local_cache,
        ttl=deps.ttl,
    )
    return MetadataCacheLookup(cache_key, redis_client, cached)


def _fetch_metadata_response(
    normalized: str,
    deps: LinkMetadataDeps,
) -> tuple[Any | None, str | None]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    }
    try:
        response = deps.request(
            normalized,
            headers=headers,
            timeout=8,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        return response, None
    except RequestException as error:
        return None, error.__class__.__name__


def _metadata_result(
    normalized: str,
    response: Any,
    max_bytes: int,
) -> Dict[str, Any]:
    bounded = read_bounded_response(
        response,
        max_bytes=max_bytes,
        fallback_url=normalized,
    )
    result: Dict[str, Any] = {
        "url": normalize_http_url(bounded.url) or normalized,
        "status": bounded.status,
        "content_type": bounded.content_type,
        "title": None,
        "description": None,
    }
    sample = bounded.body[:400].lower()
    is_html = (
        "html" in bounded.content_type
        or b"<html" in sample
        or b"<!doctype" in sample
    )
    if bounded.body and is_html:
        title, description = extract_html_metadata(
            decode_bounded_body(bounded)
        )
        result["title"] = truncate_link_metadata_text(title, limit=160)
        result["description"] = truncate_link_metadata_text(
            description,
            limit=280,
        )
    return result


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
    deps: LinkMetadataDeps,
) -> Dict[str, Any]:
    normalized = normalize_http_url(raw_url)
    if not normalized:
        return {"url": str(raw_url or "").strip(), "error": "url inválida"}

    cache = _load_metadata_cache(normalized, deps)
    if cache.value is not None:
        return cache.value
    response, error = _fetch_metadata_response(normalized, deps)
    if response is None:
        return {"url": normalized, "error": error or "RequestException"}

    result = _metadata_result(normalized, response, deps.max_bytes)
    if cache.redis_client is not None:
        try:
            deps.redis_setex_json(
                cache.redis_client,
                cache.cache_key,
                deps.ttl,
                result,
            )
        except Exception:
            pass

    cache_link_metadata(
        normalized,
        result,
        local_cache=deps.local_cache,
        ttl=deps.ttl,
    )
    return result
