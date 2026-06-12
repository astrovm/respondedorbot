"""Helpers to normalise and inspect social media links."""

from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

from requests.exceptions import RequestException

from api.logging_config import get_logger
from api.utils.http import request_with_ssl_fallback


logger = get_logger(__name__)

EMBED_REQUEST_TIMEOUT = 10
EEINSTAGRAM_PROBE_ATTEMPTS = 3
EEINSTAGRAM_PROBE_BACKOFF_SECONDS = 0.25
TELEGRAM_PREVIEW_USER_AGENT = "TelegramBot (like TwitterBot)"
TELEGRAM_REMOTE_VIDEO_MAX_BYTES = 20_000_000
TELEGRAM_MULTIPART_VIDEO_MAX_BYTES = 50_000_000

ALTERNATIVE_FRONTENDS: Set[str] = {
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "eeinstagram.com",
    "vxinstagram.com",
    "kkinstagram.com",
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

REPLACEABLE_FRONTENDS: Set[str] = {
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
}

__all__ = [
    "ALTERNATIVE_FRONTENDS",
    "ORIGINAL_FRONTENDS",
    "REPLACEABLE_FRONTENDS",
    "TELEGRAM_PREVIEW_USER_AGENT",
    "inspect_embed_url",
    "is_social_frontend",
    "has_replaceable_link",
    "can_embed_url",
    "url_is_embedable",
    "replace_links",
    "download_oversized_instagram_video",
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


def has_replaceable_link(text: str) -> bool:
    """Return True when text contains a link handled by replace_links."""
    for match in re.finditer(r"https?://[^\s]+", text or ""):
        parsed = urlparse(match.group(0).strip("()[]{}<>\"'.,;!?"))
        host = _normalized_host(parsed)
        if any(
            host == domain or host.endswith(f".{domain}")
            for domain in REPLACEABLE_FRONTENDS
        ):
            return True
    return False


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
    is_eeinstagram_host = _is_eeinstagram_host(parsed)
    try:
        response = _request_embed_url(
            url,
            retry=is_eeinstagram_host,
            allow_redirects=True,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        logger.warning("embed: request failed url=%s error=%s", url, exc)
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
        logger.info("embed: returned status url=%s status=%s", url, response.status_code)
        return {
            "embeddable": False,
            "url": str(getattr(response, "url", "") or url),
            "status": response.status_code,
            "content_type": str(response.headers.get("Content-Type", "")).lower(),
            "title": None,
            "description": None,
        }

    content_type = response.headers.get("Content-Type", "").lower()
    response_url = getattr(response, "url", "")
    normalized_url = (
        response_url.strip()
        if isinstance(response_url, str) and response_url.strip()
        else url
    )
    if content_type.startswith(("image/", "video/", "audio/")):
        logger.info("embed: direct media url=%s content_type=%s", url, content_type)
        return {
            "embeddable": True,
            "url": normalized_url,
            "status": response.status_code,
            "content_type": content_type,
            "title": None,
            "description": None,
            "media_url": normalized_url,
            "media_content_type": content_type,
            "media_size": _response_content_length(response),
        }
    if "text/html" not in content_type:
        logger.info("embed: not embeddable url=%s content_type=%s", url, content_type)
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
    is_eeinstagram_host = _is_eeinstagram_host(parsed)

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
        media_probe: Dict[str, Any] = {}
        if _is_instagram_frontend_host(parsed):
            media_reference = (
                meta_tags.get("og:video")
                or meta_tags.get("twitter:player:stream")
                or meta_tags.get("og:image")
                or meta_tags.get("twitter:image")
            )
            probed_media = (
                _probe_instagram_media(urljoin(normalized_url, media_reference))
                if media_reference
                else None
            )
            if not probed_media:
                logger.info(
                    "embed: instagram media not ready url=%s media=%s",
                    url,
                    media_reference,
                )
                return {
                    "embeddable": False,
                    "url": normalized_url,
                    "status": response.status_code,
                    "content_type": content_type,
                    "title": title,
                    "description": description,
                    "canonical_url": canonical_url,
                }
            media_probe = probed_media
        detail = ", ".join(f"{key}={value[:80]}" for key, value in meta_tags.items())
        logger.info("embed: metadata found url=%s metadata=%s", url, detail)
        return {
            "embeddable": True,
            "url": normalized_url,
            "status": response.status_code,
            "content_type": content_type,
            "title": title,
            "description": description,
            "canonical_url": canonical_url,
            **media_probe,
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
    logger.info("embed: missing metadata url=%s missing=%s", url, missing_detail)
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

    if not _is_eeinstagram_host(parsed):
        return None

    path_segments = [segment for segment in parsed.path.lower().split("/") if segment]
    if not path_segments or path_segments[0] not in {"p", "reel", "reels"}:
        return None

    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
    try:
        response = _request_embed_url(
            url,
            retry=True,
            method="head",
            allow_redirects=False,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        logger.warning("embed: HEAD request failed url=%s error=%s", url, exc)
        return None

    status_code = response.status_code
    if status_code == 405:
        logger.info("embed: HEAD not allowed, falling back to GET url=%s", url)
        return None
    if _should_retry_embed_response(response):
        logger.info(
            "embed: HEAD transient status exhausted, falling back to GET url=%s status=%s",
            url,
            status_code,
        )
        return None
    if status_code >= 400:
        logger.info("embed: HEAD returned status url=%s status=%s", url, status_code)
        return False

    if 300 <= status_code < 400:
        location = response.headers.get("Location", "")
        if not location:
            logger.info("embed: HEAD redirect missing location url=%s", url)
            return False
        if not _probe_instagram_media(urljoin(url, location)):
            logger.info("embed: HEAD redirect media not ready url=%s", url)
            return False
        logger.info(
            "embed: HEAD redirect url=%s status=%s location=%s",
            url,
            status_code,
            location[:120],
        )
        return True

    return None


def _is_eeinstagram_host(parsed: ParseResult) -> bool:
    host = _normalized_host(parsed)
    return host == "eeinstagram.com" or host.endswith(".eeinstagram.com")


def _is_instagram_frontend_host(parsed: ParseResult) -> bool:
    host = _normalized_host(parsed)
    return (
        _is_eeinstagram_host(parsed)
        or host == "vxinstagram.com"
        or host.endswith(".vxinstagram.com")
        or host == "kkinstagram.com"
        or host.endswith(".kkinstagram.com")
    )


def _response_content_length(response: Any) -> Optional[int]:
    raw_length = str(response.headers.get("Content-Length", "")).strip()
    try:
        return int(raw_length) if raw_length else None
    except ValueError:
        return None


def _probe_instagram_media(url: str) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
    response = None
    try:
        response = _request_embed_url(
            url,
            retry=True,
            allow_redirects=True,
            stream=True,
            timeout=EMBED_REQUEST_TIMEOUT,
            headers=headers,
        )
    except RequestException as exc:
        logger.warning("embed: media request failed url=%s error=%s", url, exc)
        return None

    try:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        is_ready = response.status_code < 400 and content_type.startswith(
            ("image/", "video/", "audio/")
        )
        logger.info(
            "embed: media probe url=%s status=%s content_type=%s ready=%s",
            url,
            response.status_code,
            content_type,
            is_ready,
        )
        if not is_ready:
            return None
        response_url = getattr(response, "url", "")
        normalized_url = (
            response_url.strip()
            if isinstance(response_url, str) and response_url.strip()
            else url
        )
        return {
            "media_url": normalized_url,
            "media_content_type": content_type,
            "media_size": _response_content_length(response),
        }
    finally:
        response.close()


def _request_embed_url(url: str, *, retry: bool, **kwargs: Any) -> Any:
    attempts = EEINSTAGRAM_PROBE_ATTEMPTS if retry else 1
    last_exc: Optional[RequestException] = None
    for attempt in range(attempts):
        try:
            response = request_with_ssl_fallback(url, **kwargs)
        except RequestException as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            _sleep_before_embed_retry(url, attempt, error=exc)
            continue

        if not _should_retry_embed_response(response) or attempt == attempts - 1:
            return response
        _sleep_before_embed_retry(url, attempt, status_code=response.status_code)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("embed request retry loop exhausted")


def _should_retry_embed_response(response: Any) -> bool:
    status_code = int(getattr(response, "status_code", 0) or 0)
    return status_code == 429 or status_code >= 500


def _sleep_before_embed_retry(
    url: str,
    attempt: int,
    *,
    error: Optional[BaseException] = None,
    status_code: Optional[int] = None,
) -> None:
    delay = EEINSTAGRAM_PROBE_BACKOFF_SECONDS * (2**attempt)
    if error is not None:
        logger.info(
            "embed: retrying request url=%s attempt=%d delay=%.2f error=%s",
            url,
            attempt + 2,
            delay,
            error,
        )
    else:
        logger.info(
            "embed: retrying request url=%s attempt=%d delay=%.2f status=%s",
            url,
            attempt + 2,
            delay,
            status_code,
        )
    time.sleep(delay)


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
        (r"(https?://)(?:www\.)?instagram\.com([^\s]*)", r"\1eeinstagram.com\2"),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)reddit\.com([^\s]*)",
            r"\1\2rxddit.com\3",
        ),
    ]

    changed = False
    original_links: List[str] = []
    checker = embed_checker or url_is_embedable

    def make_sub(repl: str) -> Callable[[re.Match[str]], str]:
        def _sub(match: re.Match[str]) -> str:
            original = match.group(0)
            parsed_original = urlparse(original)
            if _is_twitter_user_profile(parsed_original):
                return original
            replaced = match.expand(repl)
            parsed = urlparse(replaced)
            parsed = _normalize_twitter_status_path(parsed)
            cleaned = parsed._replace(query="", fragment="")
            replaced_full = urlunparse(cleaned)
            candidates = [replaced_full]
            if _is_eeinstagram_host(cleaned):
                candidates.extend(
                    [
                        urlunparse(cleaned._replace(netloc="vxinstagram.com")),
                        urlunparse(cleaned._replace(netloc="kkinstagram.com")),
                    ]
                )

            for candidate in candidates:
                if checker(candidate):
                    nonlocal changed
                    changed = True
                    cleaned_original = parsed_original._replace(query="", fragment="")
                    original_links.append(urlunparse(cleaned_original))
                    logger.info(
                        "link: replacing original=%s replacement=%s",
                        original,
                        candidate,
                    )
                    if _is_instagram_frontend_host(urlparse(candidate)):
                        cache_bucket = int(time.time() // 3600)
                        return urlunparse(
                            urlparse(candidate)._replace(query=f"tg={cache_bucket}")
                        )
                    return candidate
                logger.info("link: cannot embed replacement=%s", candidate)

            logger.info(
                "link: no embeddable replacement keeping=%s", original
            )
            return original

        return _sub

    new_text = text
    for pattern, repl in patterns:
        new_text = re.sub(pattern, make_sub(repl), new_text, flags=re.IGNORECASE)

    url_pattern = re.compile(r"(https?://[^\s]+)")

    def strip_tracking(match: re.Match[str]) -> str:
        url = match.group(0)
        parsed = urlparse(url)
        if is_social_frontend(parsed.netloc):
            if _is_instagram_frontend_host(parsed) and re.fullmatch(
                r"tg=\d+", parsed.query
            ):
                return urlunparse(parsed._replace(fragment=""))
            cleaned = parsed._replace(query="", fragment="")
            return urlunparse(cleaned)
        return url

    new_text = url_pattern.sub(strip_tracking, new_text)

    return new_text, changed, original_links


def download_oversized_instagram_video(text: str) -> Optional[bytes]:
    """Download a validated Instagram video that exceeds Telegram's URL limit."""

    for match in re.finditer(r"https?://[^\s]+", text or ""):
        raw_url = match.group(0).strip("()[]{}<>\"'.,;!?")
        parsed = urlparse(raw_url)
        if not _is_instagram_frontend_host(parsed):
            continue

        clean_url = urlunparse(parsed._replace(query="", fragment=""))
        metadata = inspect_embed_url(clean_url)
        media_url = str(metadata.get("media_url") or "").strip()
        media_type = str(metadata.get("media_content_type") or "").lower()
        media_size = metadata.get("media_size")
        if (
            not metadata.get("embeddable")
            or not media_url
            or not media_type.startswith("video/")
            or not isinstance(media_size, int)
            or media_size <= TELEGRAM_REMOTE_VIDEO_MAX_BYTES
            or media_size > TELEGRAM_MULTIPART_VIDEO_MAX_BYTES
        ):
            return None

        headers = {"User-Agent": TELEGRAM_PREVIEW_USER_AGENT}
        response = None
        try:
            response = request_with_ssl_fallback(
                media_url,
                allow_redirects=True,
                stream=True,
                timeout=30,
                headers=headers,
            )
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if not content_type.startswith("video/"):
                return None
            buffer = BytesIO()
            total = 0
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > TELEGRAM_MULTIPART_VIDEO_MAX_BYTES:
                    return None
                buffer.write(chunk)
            if total <= TELEGRAM_REMOTE_VIDEO_MAX_BYTES:
                return None
            return buffer.getvalue()
        except RequestException as exc:
            logger.warning("embed: oversized video download failed url=%s error=%s", media_url, exc)
            return None
        finally:
            if response is not None:
                response.close()

    return None


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
            payload = response.json()
            return cast(Dict[str, Any], payload) if isinstance(payload, dict) else None
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
