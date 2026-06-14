from __future__ import annotations

import re
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from api.links.agent_tools import normalize_http_url
from api.links import commands as link_commands
from api.links import context as link_context
from api.utils import links


MESSAGE_URL_PATTERN = re.compile(
    r"(?i)\b("
    r"(?:https?://|www\.)[^\s<>()]+"
    r"|"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s<>()]*)?"
    r")"
)


class LinkServiceProtocol(Protocol):
    def replace(self, text: str) -> Tuple[str, bool, List[str]]: ...

    def download_oversized_instagram_video(self, text: str) -> Optional[bytes]: ...

    def build_context(self, message: Mapping[str, Any]) -> str: ...


@dataclass
class LinkService:
    optional_redis_client: Callable[[], Optional[Any]]
    hash_cache_key: Callable[[str, Mapping[str, Any]], str]
    request_fn: Callable[..., Any]
    redis_get_json: Callable[[Any, str], Any]
    redis_setex_json: Callable[[Any, str, int, Mapping[str, Any]], Any]
    extract_video_id: Callable[[str], Optional[str]]
    fetch_transcript: Callable[[str], str]
    logger: Logger
    format_log_context: Callable[[Mapping[str, Any]], str]
    metadata_ttl: int = 300
    metadata_max_bytes: int = 64_000
    max_links: int = 3
    local_metadata_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    url_pattern: re.Pattern[str] = field(
        default_factory=lambda: MESSAGE_URL_PATTERN
    )

    def can_embed(self, url: str) -> bool:
        return links.can_embed_url(
            url,
            metadata_sink=lambda metadata: self.cache_metadata(url, metadata),
        )

    def replace(self, text: str) -> Tuple[str, bool, List[str]]:
        return links.replace_links(text, embed_checker=self.can_embed)

    def download_oversized_instagram_video(self, text: str) -> Optional[bytes]:
        return links.download_oversized_instagram_video(text)

    def fetch_tweet_content(self, url: str) -> Optional[Dict[str, str]]:
        return links.fetch_tweet_content(url)

    def normalize_detected_url(self, raw_url: str) -> Optional[str]:
        return link_context.normalize_detected_url(
            raw_url,
            normalize_url=normalize_http_url,
        )

    def extract_urls_from_entities(self, source_text: str, entities: Any) -> List[str]:
        return link_context.extract_urls_from_entities(
            source_text,
            entities,
            normalize_url=self.normalize_detected_url,
        )

    def extract_message_urls(self, message: Mapping[str, Any]) -> List[str]:
        return link_context.extract_message_urls(
            message,
            url_pattern=self.url_pattern,
            max_links=self.max_links,
            normalize_url=self.normalize_detected_url,
            extract_entities=self.extract_urls_from_entities,
        )

    def cache_metadata(self, raw_url: str, metadata: Mapping[str, Any]) -> None:
        link_commands.cache_link_metadata(
            raw_url,
            metadata,
            local_cache=self.local_metadata_cache,
            ttl=self.metadata_ttl,
        )

    def fetch_metadata(self, raw_url: str) -> Dict[str, Any]:
        return link_commands.fetch_link_metadata(
            raw_url,
            deps=link_commands.LinkMetadataDeps(
                local_cache=self.local_metadata_cache,
                ttl=self.metadata_ttl,
                max_bytes=self.metadata_max_bytes,
                optional_redis_client=self.optional_redis_client,
                hash_cache_key=self.hash_cache_key,
                request=self.request_fn,
                redis_get_json=self.redis_get_json,
                redis_setex_json=self.redis_setex_json,
            ),
        )

    def build_context(self, message: Mapping[str, Any]) -> str:
        return link_context.build_message_links_context(
            message,
            extract_urls=self.extract_message_urls,
            fetch_metadata=self.fetch_metadata,
            fetch_tweet=self.fetch_tweet_content,
            truncate_text=link_commands.truncate_link_metadata_text,
            extract_video_id=self.extract_video_id,
            fetch_transcript=self.fetch_transcript,
            logger=self.logger,
            format_log_context=self.format_log_context,
        )


__all__ = ["LinkService", "LinkServiceProtocol"]
