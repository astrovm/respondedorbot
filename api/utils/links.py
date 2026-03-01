"""Helpers to normalise and inspect social media links."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse

import requests
from requests.exceptions import RequestException

from api.utils.http import request_with_ssl_fallback

EMBED_REQUEST_TIMEOUT = 10

ALTERNATIVE_FRONTENDS: Set[str] = {
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kkinstagram.com",
    "eeinstagram.com",
    "rxddit.com",
}

ORIGINAL_FRONTENDS: Set[str] = {
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
}

__all__ = [
    "ALTERNATIVE_FRONTENDS",
    "ORIGINAL_FRONTENDS",
    "is_social_frontend",
    "can_embed_url",
    "url_is_embedable",
    "replace_links",
]


_TWITTER_HOSTS: Set[str] = {
    "twitter.com",
    "x.com",
    "xcancel.com",
}


def _is_twitter_user_profile(parsed: ParseResult) -> bool:
    """Return ``True`` when *parsed* points to a Twitter/X user profile."""

    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host not in _TWITTER_HOSTS:
        return False

    path = parsed.path.strip("/")
    if not path:
        return False

    segments = [segment.lstrip("@") for segment in path.split("/") if segment]
    if not segments:
        return False

    lower_segments = [segment.lower() for segment in segments]
    if "status" in lower_segments:
        return False

    first_segment = lower_segments[0]
    reserved = {
        "home",
        "share",
        "intent",
        "i",
        "search",
        "explore",
        "notifications",
        "messages",
        "settings",
        "compose",
        "privacy",
        "tos",
    }
    if first_segment in reserved:
        return False

    if len(lower_segments) == 1:
        return True

    profile_subpages = {"with_replies", "media", "likes"}
    return len(lower_segments) == 2 and lower_segments[1] in profile_subpages


def _normalize_twitter_status_path(parsed: ParseResult) -> ParseResult:
    """Normalize Twitter/X status URLs that include the /i/ prefix."""

    path = parsed.path or ""
    normalized = re.sub(
        r"^/i/(?:web/)?status/",
        "/status/",
        path,
        flags=re.IGNORECASE,
    )
    if normalized == path:
        return parsed
    return parsed._replace(path=normalized)


def is_social_frontend(host: str) -> bool:
    """Return True if *host* matches a known social media frontend."""
    host = host.lower()
    frontends = ALTERNATIVE_FRONTENDS | ORIGINAL_FRONTENDS
    return any(host == domain or host.endswith(f".{domain}") for domain in frontends)


def can_embed_url(url: str) -> bool:
    """Return True when the target page exposes Telegram-compatible OpenGraph metadata."""
    parsed = urlparse(url)
    eeinstagram_preview = _eeinstagram_preview_check(parsed, url)
    if eeinstagram_preview is False:
        return False
    if eeinstagram_preview is True:
        return True
    headers = {"User-Agent": "TelegramBot (like TwitterBot)"}
    try:
        response = request_with_ssl_fallback(
            url,
            allow_redirects=True,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        print(f"[EMBED] {url} request failed: {exc}")
        return False

    if response.status_code >= 400:
        print(f"[EMBED] {url} returned status {response.status_code}")
        return False

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type.startswith(("image/", "video/", "audio/")):
        print(f"[EMBED] {url} served direct media type {content_type}")
        return True
    if "text/html" not in content_type:
        print(f"[EMBED] {url} content-type {content_type} not embeddable")
        return False

    html = response.text[:20000]

    class MetaParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: Dict[str, str] = {}

        def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
            if tag != "meta":
                return
            attrs_dict: Dict[str, Optional[str]] = dict(attrs)
            key_candidate = attrs_dict.get("property") or attrs_dict.get("name")
            if not key_candidate:
                return
            key_lower = key_candidate.lower()
            if key_lower.startswith("og:") or key_lower.startswith("twitter:"):
                content = (attrs_dict.get("content") or "").strip()
                if content:
                    self.tags[key_lower] = content

    parser = MetaParser()
    parser.feed(html)
    meta_tags = parser.tags
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    is_eeinstagram_host = host == "eeinstagram.com" or host.endswith(".eeinstagram.com")

    has_title = "og:title" in meta_tags
    has_image = "og:image" in meta_tags
    has_video = "og:video" in meta_tags

    if (has_title or (is_eeinstagram_host and (has_image or has_video))) and (has_image or has_video):
        detail = ", ".join(
            f"{key}={value[:80]}" for key, value in meta_tags.items()
        )
        print(f"[EMBED] {url} has embed metadata: {detail}")
        return True

    missing_fields: List[str] = []
    if not has_title and not (is_eeinstagram_host and (has_image or has_video)):
        missing_fields.append("og:title")
    if not (has_image or has_video):
        missing_fields.append("og:image or og:video")
    missing_detail = ", ".join(missing_fields)
    print(f"[EMBED] {url} missing required metadata: {missing_detail}")
    return False


def _eeinstagram_preview_check(
    parsed: ParseResult, url: str
) -> Optional[bool]:
    """Return embed eligibility from the eeinstagram HEAD response when available."""

    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if not (host == "eeinstagram.com" or host.endswith(".eeinstagram.com")):
        return None

    path_segments = [segment for segment in parsed.path.lower().split("/") if segment]
    if not path_segments or path_segments[0] not in {"p", "reel", "reels"}:
        return None

    headers = {"User-Agent": "TelegramBot 1.0"}
    try:
        response = request_with_ssl_fallback(
            url,
            method="head",
            allow_redirects=False,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        print(f"[EMBED] {url} HEAD request failed: {exc}")
        return False

    status_code = response.status_code
    if status_code == 405:
        print(f"[EMBED] {url} HEAD not allowed, falling back to GET metadata check")
        return None
    if status_code >= 400:
        print(f"[EMBED] {url} HEAD returned status {status_code}")
        return False

    if 300 <= status_code < 400:
        location = response.headers.get("Location", "")
        if not location:
            print(f"[EMBED] {url} HEAD redirect missing location")
            return False
        print(f"[EMBED] {url} HEAD redirect {status_code} to {location[:120]}")
        return True

    return None


def url_is_embedable(url: str) -> bool:
    """Backwards-compatible wrapper around :func:`can_embed_url`."""
    return can_embed_url(url)


def replace_links(
    text: str, embed_checker: Optional[Callable[[str], bool]] = None
) -> Tuple[str, bool, List[str]]:
    """Replace known social links with their privacy-friendly alternatives."""

    patterns = [
        (r"(https?://)(?:www\.)?twitter\.com([^\s]*)", r"\1fxtwitter.com\2"),
        (r"(https?://)(?:www\.)?x\.com([^\s]*)", r"\1fixupx.com\2"),
        (r"(https?://)(?:www\.)?xcancel\.com([^\s]*)", r"\1fixupx.com\2"),
        (r"(https?://)(?:www\.)?bsky\.app([^\s]*)", r"\1fxbsky.app\2"),
        (r"(https?://)(?:www\.)?instagram\.com([^\s]*)", r"\1kkinstagram.com\2"),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)reddit\.com([^\s]*)",
            r"\1\2rxddit.com\3",
        ),
    ]

    changed = False
    original_links: List[str] = []
    checker = embed_checker or url_is_embedable

    def make_sub(repl: str):
        def _sub(match: re.Match) -> str:
            original = match.group(0)
            parsed_original = urlparse(original)
            if _is_twitter_user_profile(parsed_original):
                return original
            replaced = match.expand(repl)
            parsed = urlparse(replaced)
            parsed = _normalize_twitter_status_path(parsed)
            cleaned = parsed._replace(query="", fragment="")
            replaced_full = urlunparse(cleaned)
            fallback_replaced_full: Optional[str] = None
            replaced_host = cleaned.netloc.lower().split(":", 1)[0]
            if replaced_host.startswith("www."):
                replaced_host = replaced_host[4:]
            if replaced_host == "kkinstagram.com":
                fallback_cleaned = cleaned._replace(netloc="eeinstagram.com")
                fallback_replaced_full = urlunparse(fallback_cleaned)

            if checker(replaced_full):
                nonlocal changed
                changed = True
                cleaned_original = parsed_original._replace(query="", fragment="")
                original_links.append(urlunparse(cleaned_original))
                print(f"[LINK] replacing {original} with {replaced_full}")
                return replaced_full

            if fallback_replaced_full and checker(fallback_replaced_full):
                changed = True
                cleaned_original = parsed_original._replace(query="", fragment="")
                original_links.append(urlunparse(cleaned_original))
                print(
                    f"[LINK] replacing {original} with fallback {fallback_replaced_full}"
                )
                return fallback_replaced_full

            print(f"[LINK] cannot embed {replaced_full}, keeping {original}")
            return original

        return _sub

    new_text = text
    for pattern, repl in patterns:
        new_text = re.sub(pattern, make_sub(repl), new_text, flags=re.IGNORECASE)

    url_pattern = re.compile(r"(https?://[^\s]+)")

    def strip_tracking(match: re.Match) -> str:
        url = match.group(0)
        parsed = urlparse(url)
        if is_social_frontend(parsed.netloc):
            cleaned = parsed._replace(query="", fragment="")
            return urlunparse(cleaned)
        return url

    new_text = url_pattern.sub(strip_tracking, new_text)

    return new_text, changed, original_links
