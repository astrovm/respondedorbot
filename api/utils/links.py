"""Helpers to normalise and inspect social media links."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse

from requests.exceptions import RequestException

from api.utils.http import request_with_ssl_fallback

EMBED_REQUEST_TIMEOUT = 10
TELEGRAM_PREVIEW_USER_AGENT = "TelegramBot (like TwitterBot)"

ALTERNATIVE_FRONTENDS: Set[str] = {
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kksave.com",
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
    "TELEGRAM_PREVIEW_USER_AGENT",
    "inspect_embed_url",
    "is_social_frontend",
    "can_embed_url",
    "url_is_embedable",
    "replace_links",
    "extract_tweet_urls",
    "resolve_tweet_url",
    "fetch_tweet_via_oembed",
    "fetch_tweet_content",
    "fetch_tweet_text",
]


_TWITTER_HOSTS: Set[str] = {
    "twitter.com",
    "x.com",
    "xcancel.com",
}

_TWITTER_FRONTEND_HOSTS: Set[str] = _TWITTER_HOSTS | {
    "fixupx.com",
    "fxtwitter.com",
}


def _normalized_host(parsed: ParseResult) -> str:
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_twitter_user_profile(parsed: ParseResult) -> bool:
    """Return ``True`` when *parsed* points to a Twitter/X user profile."""

    host = _normalized_host(parsed)
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


def inspect_embed_url(url: str) -> Dict[str, Any]:
    """Inspect whether the target page exposes Telegram-compatible preview metadata."""

    parsed = urlparse(url)
    eeinstagram_preview = _eeinstagram_preview_check(parsed, url)
    if eeinstagram_preview is False:
        return {
            "embeddable": False,
            "url": url,
            "status": None,
            "content_type": "",
            "title": None,
            "description": None,
        }
    if eeinstagram_preview is True:
        return {
            "embeddable": True,
            "url": url,
            "status": None,
            "content_type": "",
            "title": None,
            "description": None,
        }
    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
    try:
        response = request_with_ssl_fallback(
            url,
            allow_redirects=True,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        print(f"[EMBED] {url} request failed: {exc}")
        return {
            "embeddable": False,
            "url": url,
            "status": None,
            "content_type": "",
            "title": None,
            "description": None,
            "error": exc.__class__.__name__,
        }

    if response.status_code >= 400:
        print(f"[EMBED] {url} returned status {response.status_code}")
        return {
            "embeddable": False,
            "url": str(getattr(response, "url", "") or url),
            "status": response.status_code,
            "content_type": str(response.headers.get("Content-Type", "")).lower(),
            "title": None,
            "description": None,
        }

    content_type = response.headers.get("Content-Type", "").lower()
    normalized_url = str(getattr(response, "url", "") or url).strip() or url
    if content_type.startswith(("image/", "video/", "audio/")):
        print(f"[EMBED] {url} served direct media type {content_type}")
        return {
            "embeddable": True,
            "url": normalized_url,
            "status": response.status_code,
            "content_type": content_type,
            "title": None,
            "description": None,
        }
    if "text/html" not in content_type:
        print(f"[EMBED] {url} content-type {content_type} not embeddable")
        return {
            "embeddable": False,
            "url": normalized_url,
            "status": response.status_code,
            "content_type": content_type,
            "title": None,
            "description": None,
        }

    html = response.text[:20000]

    class MetaParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: Dict[str, str] = {}

        def handle_starttag(
            self, tag: str, attrs: List[Tuple[str, Optional[str]]]
        ) -> None:
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
    title = meta_tags.get("og:title") or meta_tags.get("twitter:title")
    description = meta_tags.get("og:description") or meta_tags.get(
        "twitter:description"
    )
    canonical_url = meta_tags.get("og:url")
    host = _normalized_host(parsed)
    is_eeinstagram_host = host == "eeinstagram.com" or host.endswith(".eeinstagram.com")

    has_title = "og:title" in meta_tags or "twitter:title" in meta_tags
    has_description = (
        "og:description" in meta_tags or "twitter:description" in meta_tags
    )
    has_og_image = "og:image" in meta_tags
    has_og_video = "og:video" in meta_tags
    has_twitter_image = "twitter:image" in meta_tags
    has_twitter_video = any(
        key in meta_tags for key in ("twitter:player", "twitter:player:stream")
    )
    has_image = has_og_image or has_twitter_image
    has_video = has_og_video or has_twitter_video
    has_card = "twitter:card" in meta_tags

    has_preview_text = has_title or has_description
    has_preview_media = has_image or has_video
    has_eeinstagram_media = has_og_image or has_og_video

    if (has_preview_text and (has_preview_media or has_card)) or (
        is_eeinstagram_host and has_eeinstagram_media
    ):
        detail = ", ".join(f"{key}={value[:80]}" for key, value in meta_tags.items())
        print(f"[EMBED] {url} has embed metadata: {detail}")
        return {
            "embeddable": True,
            "url": normalized_url,
            "status": response.status_code,
            "content_type": content_type,
            "title": title,
            "description": description,
            "canonical_url": canonical_url,
        }

    missing_fields: List[str] = []
    if not has_preview_text and not (is_eeinstagram_host and has_eeinstagram_media):
        missing_fields.append(
            "og:title/twitter:title or og:description/twitter:description"
        )
    if not (has_preview_media or has_card) and not is_eeinstagram_host:
        missing_fields.append(
            "og:image/twitter:image or og:video/twitter:player or twitter:card"
        )
    if not has_eeinstagram_media and is_eeinstagram_host:
        missing_fields.append("og:image or og:video")
    missing_detail = ", ".join(missing_fields)
    print(f"[EMBED] {url} missing required metadata: {missing_detail}")
    return {
        "embeddable": False,
        "url": normalized_url,
        "status": response.status_code,
        "content_type": content_type,
        "title": title,
        "description": description,
        "canonical_url": canonical_url,
    }


def can_embed_url(
    url: str,
    *,
    metadata_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> bool:
    """Return True when the target page exposes Telegram-compatible OpenGraph metadata."""

    metadata = inspect_embed_url(url)
    if metadata_sink is not None:
        metadata_sink(metadata)
    return bool(metadata.get("embeddable"))


def _eeinstagram_preview_check(parsed: ParseResult, url: str) -> Optional[bool]:
    """Return embed eligibility from the eeinstagram HEAD response when available."""

    host = _normalized_host(parsed)
    if not (host == "eeinstagram.com" or host.endswith(".eeinstagram.com")):
        return None

    path_segments = [segment for segment in parsed.path.lower().split("/") if segment]
    if not path_segments or path_segments[0] not in {"p", "reel", "reels"}:
        return None

    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
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
    """Return whether a URL should keep its current form for embeds."""
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
        (r"(https?://)(?:www\.)?instagram\.com([^\s]*)", r"\1kksave.com\2"),
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
            if replaced_host == "kksave.com":
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


TWITTER_STATUS_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com|fixupx\.com|fxtwitter\.com|xcancel\.com)/(\w+)/status/(\d+)"
)

TWITTER_STATUS_ID_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com|fixupx\.com|fxtwitter\.com|xcancel\.com)/status/(\d+)"
)


def extract_tweet_urls(text: str) -> List[Tuple[str, str, str]]:
    """Extract tweet URLs from text and return list of (username, status_id, original_url)."""
    matches = TWITTER_STATUS_REGEX.findall(text.lower())
    results = []
    for username, status_id in matches:
        full_url = f"https://x.com/{username}/status/{status_id}"
        results.append((username, status_id, full_url))
    return results


def resolve_tweet_url(url: str) -> Optional[str]:
    """Return a canonical x.com status URL for supported Twitter/X frontends."""

    parsed = urlparse(url)
    if _normalized_host(parsed) not in _TWITTER_FRONTEND_HOSTS:
        return None

    direct_match = TWITTER_STATUS_REGEX.search(url.lower())
    if direct_match:
        username, status_id = direct_match.groups()
        return f"https://x.com/{username}/status/{status_id}"

    id_only_match = TWITTER_STATUS_ID_REGEX.search(url.lower())
    if not id_only_match:
        return None

    metadata = inspect_embed_url(url)
    canonical_url = str(metadata.get("canonical_url") or metadata.get("url") or "")
    canonical_match = TWITTER_STATUS_REGEX.search(canonical_url.lower())
    if canonical_match:
        username, status_id = canonical_match.groups()
        return f"https://x.com/{username}/status/{status_id}"

    return None


def _resolve_tco_redirect(url: str) -> Optional[str]:
    """Resolve a t.co short URL to see if it points to a tweet."""
    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
    try:
        response = request_with_ssl_fallback(
            url,
            allow_redirects=False,
            timeout=5,
            headers=headers,
        )
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "")
            if location:
                return location
    except RequestException:
        pass
    return None


def fetch_tweet_via_oembed(url: str) -> Optional[Dict[str, Any]]:
    """Fetch tweet content via Twitter oEmbed API."""
    normalized_url = url
    if "fixupx.com" in url.lower() or "fxtwitter.com" in url.lower():
        match = TWITTER_STATUS_REGEX.search(url.lower())
        if match:
            username, status_id = match.groups()
            normalized_url = f"https://x.com/{username}/status/{status_id}"
    elif "twitter.com" in url.lower():
        match = TWITTER_STATUS_REGEX.search(url.lower())
        if match:
            username, status_id = match.groups()
            normalized_url = f"https://x.com/{username}/status/{status_id}"

    oembed_url = (
        f"https://publish.twitter.com/oembed?url={normalized_url}&omit_script=true"
    )
    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
    try:
        response = request_with_ssl_fallback(
            oembed_url,
            allow_redirects=True,
            timeout=10,
            headers=headers,
        )
        if response.status_code == 200:
            return response.json()
    except RequestException:
        pass
    return None


def fetch_tweet_content(url: str) -> Optional[Dict[str, str]]:
    """Fetch readable tweet content for a Twitter/X/front-end URL."""

    canonical_url = resolve_tweet_url(url)
    if not canonical_url:
        return None

    oembed_data = fetch_tweet_via_oembed(canonical_url)
    if not oembed_data:
        return {
            "url": canonical_url,
            "error": "no se pudo leer el tweet",
        }

    html = str(oembed_data.get("html") or "")
    tweet_text = _extract_text_from_oembed_html(html)
    author_name = str(oembed_data.get("author_name") or "").strip()
    author_url = str(oembed_data.get("author_url") or "").strip()
    date = _extract_date_from_oembed_html(html)

    return {
        "url": canonical_url,
        "author": author_name,
        "author_url": author_url,
        "date": date,
        "text": tweet_text,
    }


def fetch_tweet_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extract tweet URLs from text and fetch their content via oEmbed.

    Returns a tuple of (summary_text, tweets_list) where tweets_list contains
    dicts with 'username', 'text', 'date', and optionally 'quoted' (quote tweet).
    """
    tweet_urls = extract_tweet_urls(text)
    if not tweet_urls:
        return "", []

    tweets_data = []
    for username, status_id, original_url in tweet_urls:
        oembed_data = fetch_tweet_via_oembed(original_url)
        if not oembed_data:
            continue

        html = oembed_data.get("html", "")
        tweet_text = _extract_text_from_oembed_html(html)
        author_name = oembed_data.get("author_name", username)
        date = _extract_date_from_oembed_html(html)

        tweets_data.append(
            {
                "username": username,
                "author": author_name,
                "text": tweet_text,
                "date": date,
                "url": original_url,
                "quoted": None,
            }
        )

    if not tweets_data:
        return "", []

    for i, tweet in enumerate(tweets_data):
        links = _extract_links_from_oembed_html(tweet["text"])
        for link in links:
            resolved = _resolve_tco_redirect(link)
            if resolved and (
                "twitter.com" in resolved.lower() or "x.com" in resolved.lower()
            ):
                quote_match = TWITTER_STATUS_REGEX.search(resolved.lower())
                if quote_match:
                    quote_username, quote_status_id = quote_match.groups()
                    quote_url = (
                        f"https://x.com/{quote_username}/status/{quote_status_id}"
                    )
                    quote_oembed = fetch_tweet_via_oembed(quote_url)
                    if quote_oembed:
                        quote_html = quote_oembed.get("html", "")
                        quote_text = _extract_text_from_oembed_html(quote_html)
                        quote_author = quote_oembed.get("author_name", quote_username)
                        quote_date = _extract_date_from_oembed_html(quote_html)
                        tweets_data[i]["quoted"] = {
                            "username": quote_username,
                            "author": quote_author,
                            "text": quote_text,
                            "date": quote_date,
                            "url": quote_url,
                        }
                    break

    summary = _format_tweets_summary(tweets_data)
    return summary, tweets_data


def _extract_text_from_oembed_html(html: str) -> str:
    """Extract readable text from oEmbed HTML blockquote."""
    p_match = re.search(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    if p_match:
        text = p_match.group(1)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return ""


def _extract_date_from_oembed_html(html: str) -> str:
    """Extract date from oEmbed HTML."""
    date_match = re.search(r">(\w+\s+\d+,\s+\d{4})<", html)
    if date_match:
        return date_match.group(1)
    return ""


def _extract_links_from_oembed_html(text: str) -> List[str]:
    """Extract all links from oEmbed tweet text."""
    url_pattern = re.compile(r"https?://[^\s<]+")
    return url_pattern.findall(text)


def _format_tweets_summary(tweets: List[Dict[str, Any]]) -> str:
    """Format tweets into a readable summary."""
    if not tweets:
        return ""

    lines = []
    for tweet in tweets:
        author = tweet.get("author", tweet.get("username", ""))
        text = tweet.get("text", "")
        date = tweet.get("date", "")
        quote = tweet.get("quoted")

        line = f"📱 @{author}"
        if date:
            line += f" · {date}"
        lines.append(line)
        lines.append(f'"{text}"')

        if quote:
            lines.append("")
            lines.append(
                f"↪️ Quote de @{quote.get('author', quote.get('username', ''))}:"
            )
            lines.append(f'"{quote.get("text", "")}"')

        lines.append("")

    return "\n".join(lines).strip()
