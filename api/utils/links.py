"""Helpers to normalise and inspect social media links."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse

import requests
from requests.exceptions import RequestException

from api.utils.http import request_with_ssl_fallback

ALTERNATIVE_FRONTENDS: Set[str] = {
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kkinstagram.com",
    "rxddit.com",
    "vxtiktok.com",
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


def is_social_frontend(host: str) -> bool:
    """Return True if *host* matches a known social media frontend."""
    host = host.lower()
    frontends = ALTERNATIVE_FRONTENDS | ORIGINAL_FRONTENDS
    return any(host == domain or host.endswith(f".{domain}") for domain in frontends)


def can_embed_url(url: str) -> bool:
    """Return True when the target page exposes OpenGraph/Twitter metadata."""
    headers = {"User-Agent": "TelegramBot (like TwitterBot)"}
    try:
        response = request_with_ssl_fallback(
            url, allow_redirects=True, timeout=5, headers=headers
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
            self.tags: List[Tuple[str, str]] = []

        def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
            if tag != "meta":
                return
            attrs_dict: Dict[str, Optional[str]] = dict(attrs)
            key_candidate = attrs_dict.get("property") or attrs_dict.get("name")
            if not key_candidate:
                return
            key = key_candidate
            key_lower = key.lower()
            if key_lower.startswith("og:") or key_lower.startswith("twitter:"):
                content = attrs_dict.get("content") or ""
                self.tags.append((key, content))

    parser = MetaParser()
    parser.feed(html)
    meta_tags = parser.tags
    has_meta = bool(meta_tags)
    if not has_meta:
        print(f"[EMBED] {url} missing og/twitter meta tags")
    else:
        detail = ", ".join(f"{key}={value[:80]}" for key, value in meta_tags)
        print(f"[EMBED] {url} has embed metadata: {detail}")
    return has_meta


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
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)tiktok\.com([^\s]*)",
            r"\1\2vxtiktok.com\3",
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
            cleaned = parsed._replace(query="", fragment="")
            replaced_full = urlunparse(cleaned)
            if checker(replaced_full):
                nonlocal changed
                changed = True
                cleaned_original = parsed_original._replace(query="", fragment="")
                original_links.append(urlunparse(cleaned_original))
                print(f"[LINK] replacing {original} with {replaced_full}")
                return replaced_full
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
