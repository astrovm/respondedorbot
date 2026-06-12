from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from logging import Logger
from typing import Any

UrlNormalizer = Callable[[str], str | None]
MetadataFetcher = Callable[[str], dict[str, Any]]
TweetFetcher = Callable[[str], dict[str, Any] | None]
TextTruncator = Callable[..., str | None]
VideoIdExtractor = Callable[[str], str | None]
TranscriptFetcher = Callable[[str], str]


def utf16_slice(text: str, offset: int, length: int) -> str:
    if not text or length <= 0:
        return ""
    encoded = text.encode("utf-16-le")
    start = max(0, int(offset)) * 2
    end = max(start, start + max(0, int(length)) * 2)
    try:
        return encoded[start:end].decode("utf-16-le", errors="ignore")
    except Exception:
        return ""


def normalize_detected_url(
    raw_url: str,
    *,
    normalize_url: UrlNormalizer,
) -> str | None:
    candidate = str(raw_url or "").strip().rstrip(".,;:!?)\"]}'")
    return normalize_url(candidate) if candidate else None


def extract_urls_from_entities(
    source_text: str,
    entities: Any,
    *,
    normalize_url: UrlNormalizer,
) -> list[str]:
    urls: list[str] = []
    if not source_text or not isinstance(entities, Sequence):
        return urls
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        entity_type = str(entity.get("type") or "").strip().lower()
        if entity_type == "text_link":
            candidate = str(entity.get("url") or "").strip()
        elif entity_type == "url":
            try:
                candidate = utf16_slice(
                    source_text,
                    int(entity.get("offset") or 0),
                    int(entity.get("length") or 0),
                ).strip()
            except (TypeError, ValueError):
                continue
        else:
            continue
        normalized = normalize_url(candidate)
        if normalized:
            urls.append(normalized)
    return urls


def extract_message_urls(
    message: Mapping[str, Any],
    *,
    url_pattern: re.Pattern[str],
    max_links: int,
    normalize_url: UrlNormalizer,
    extract_entities: Callable[[str, Any], list[str]],
) -> list[str]:
    candidates: list[str] = []
    for text_key, entities_key in (
        ("text", "entities"),
        ("caption", "caption_entities"),
    ):
        source_text = str(message.get(text_key) or "")
        if not source_text:
            continue
        candidates.extend(extract_entities(source_text, message.get(entities_key)))
        for match in url_pattern.finditer(source_text):
            normalized = normalize_url(match.group(1))
            if normalized:
                candidates.append(normalized)

    unique_urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_urls.append(candidate)
        if len(unique_urls) >= max_links:
            break
    return unique_urls


def build_message_links_context(
    message: Mapping[str, Any],
    *,
    extract_urls: Callable[[Mapping[str, Any]], list[str]],
    fetch_metadata: MetadataFetcher,
    fetch_tweet: TweetFetcher,
    truncate_text: TextTruncator,
    extract_video_id: VideoIdExtractor,
    fetch_transcript: TranscriptFetcher,
    logger: Logger,
    format_log_context: Callable[[Mapping[str, Any]], str],
) -> str:
    urls = extract_urls(message)
    if not urls:
        return ""
    logger.info(
        "links context: extracted urls count=%d urls=%s%s",
        len(urls),
        urls,
        format_log_context(_message_log_context(message)),
    )

    lines = ["LINKS DEL MENSAJE:"]
    transcripts: list[str] = []
    for index, url in enumerate(urls, 1):
        tweet = fetch_tweet(url)
        if tweet:
            final_url = str(tweet.get("url") or url).strip() or url
            lines.append(f"{index}. {final_url}")
            if tweet.get("error"):
                lines.append(f"error: {tweet['error']}")
                continue
            for label, key, limit in (
                ("autor", "author", 160),
                ("fecha", "date", 80),
                ("tweet", "text", 500),
            ):
                value = truncate_text(tweet.get(key), limit=limit)
                if value:
                    lines.append(f"{label}: {value}")
            continue

        metadata = fetch_metadata(url)
        final_url = str(metadata.get("url") or url).strip() or url
        lines.append(f"{index}. {final_url}")
        title = truncate_text(metadata.get("title"), limit=160)
        description = truncate_text(metadata.get("description"), limit=280)
        if title:
            lines.append(f"titulo: {title}")
        if description:
            lines.append(f"descripcion: {description}")

        video_id = extract_video_id(final_url)
        if video_id:
            transcript = fetch_transcript(video_id)
            if transcript:
                transcripts.append(transcript)

    if transcripts:
        lines.append("")
        lines.extend(transcripts)
    return "\n".join(lines)


def _message_log_context(message: Mapping[str, Any]) -> dict[str, Any]:
    chat = message.get("chat")
    sender = message.get("from")
    return {
        "source": "build_message_links_context",
        "chat_id": chat.get("id") if isinstance(chat, Mapping) else message.get("chat_id"),
        "chat_title": chat.get("title") if isinstance(chat, Mapping) else None,
        "message_id": message.get("message_id"),
        "user_id": sender.get("id") if isinstance(sender, Mapping) else None,
        "user_name": sender.get("username") if isinstance(sender, Mapping) else None,
    }
