from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from html import unescape
from logging import Logger
from typing import Any, Protocol, cast
from xml.etree import ElementTree as ET

import redis
from requests.exceptions import RequestException

from api.core.rust_bridge import load_rust_bridge

RedisFactory = Callable[[], redis.Redis | None]
RedisJsonGetter = Callable[[redis.Redis, str], Any]
RedisJsonSetter = Callable[..., bool]
RequestGetter = Callable[..., Any]


class _RustHackerNews(Protocol):
    def normalize_hacker_news_item(self, title: str, url: str, description: str) -> str: ...


logger = logging.getLogger(__name__)


def _load_rust_hacker_news() -> _RustHackerNews | None:
    module = load_rust_bridge("RUST_HACKER_NEWS_ENABLED")
    if module is None:
        return None
    return cast(_RustHackerNews, module)


def get_hacker_news_context(
    limit: int,
    *,
    max_items: int,
    cache_key: str,
    cache_ttl: int,
    primary_url: str,
    fallback_url: str,
    redis_factory: RedisFactory,
    redis_get_json: RedisJsonGetter,
    redis_setex_json: RedisJsonSetter,
    request_get: RequestGetter,
    logger: Logger,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = max_items
    limit = max(1, min(limit, max_items))

    redis_client = redis_factory()
    cached_items: list[dict[str, Any]] | None = None
    if redis_client:
        cached = redis_get_json(redis_client, cache_key)
        if isinstance(cached, list):
            cached_items = cached
            if cached_items:
                return cached_items[:limit]

    response = _fetch_feed(
        primary_url=primary_url,
        fallback_url=fallback_url,
        request_get=request_get,
        logger=logger,
    )
    if response is None:
        return (cached_items or [])[:limit]

    try:
        items = _parse_feed(response.text, max_items=max_items)
        if redis_client and items:
            try:
                redis_setex_json(redis_client, cache_key, cache_ttl, items)
            except Exception as error:
                logger.warning(
                    "get_hacker_news_context: failed to cache RSS items: %s",
                    error,
                )
        return items[:limit] if items else (cached_items or [])[:limit]
    except ET.ParseError:
        logger.exception("Error parsing Hacker News RSS")
    except Exception:
        logger.exception("get_hacker_news_context failed")
    return (cached_items or [])[:limit]


def _fetch_feed(
    *,
    primary_url: str,
    fallback_url: str,
    request_get: RequestGetter,
    logger: Logger,
) -> Any | None:
    try:
        response = request_get(primary_url, timeout=10)
        response.raise_for_status()
        return response
    except RequestException:
        logger.warning("HN RSS primary failed, trying fallback")
    try:
        response = request_get(fallback_url, timeout=10)
        response.raise_for_status()
        return response
    except RequestException:
        logger.exception("Error fetching Hacker News RSS (both URLs)")
        return None


def _parse_feed(response_text: str, *, max_items: int) -> list[dict[str, Any]]:
    root = ET.fromstring(response_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict[str, Any]] = []
    rust = _load_rust_hacker_news()
    for item_element in channel.findall("item"):
        title = unescape(str(item_element.findtext("title", "") or "")).strip()
        if not title:
            continue
        description = item_element.findtext("description", "") or ""
        url = str(item_element.findtext("link", "") or "").strip()
        item = None
        if rust is not None:
            try:
                item = json.loads(rust.normalize_hacker_news_item(title, url, description))
            except Exception:
                logger.exception("Rust Hacker News item normalization failed; using Python")
        if item is None:
            item = {
                "title": title,
                "url": url,
                "points": _extract_int(r"Points:\s*(\d+)", description),
                "comments": _extract_int(r"# Comments:\s*(\d+)", description),
                "comments_url": _extract_text(
                    r'Comments URL: <a href="([^"]+)"', description
                ),
            }
        items.append(item)
        if len(items) >= max_items:
            break
    return items


def _extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_text(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""
