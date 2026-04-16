"""Utility helpers for the Respondedor bot."""

from api.utils.formatting import (
    fmt_num,
    fmt_signed_pct,
    parse_date_string,
    parse_monetary_number,
    to_es_number,
    to_ddmmyy,
)
from api.utils.caching import now_utc_iso, update_local_cache, local_cache_get
from api.utils.http import request_with_ssl_fallback
from api.utils.links import (
    ALTERNATIVE_FRONTENDS,
    ORIGINAL_FRONTENDS,
    can_embed_url,
    is_social_frontend,
    replace_links,
    url_is_embedable,
    extract_tweet_urls,
    fetch_tweet_via_oembed,
    fetch_tweet_text,
)
from api.utils.youtube_transcript import (
    extract_youtube_video_id,
    is_youtube_url,
    fetch_youtube_transcript,
    format_youtube_transcript_for_context,
    get_youtube_transcript_context,
)

__all__ = [
    "fmt_num",
    "fmt_signed_pct",
    "parse_date_string",
    "parse_monetary_number",
    "to_es_number",
    "to_ddmmyy",
    "now_utc_iso",
    "update_local_cache",
    "local_cache_get",
    "request_with_ssl_fallback",
    "ALTERNATIVE_FRONTENDS",
    "ORIGINAL_FRONTENDS",
    "can_embed_url",
    "is_social_frontend",
    "replace_links",
    "url_is_embedable",
    "extract_tweet_urls",
    "fetch_tweet_via_oembed",
    "fetch_tweet_text",
    "extract_youtube_video_id",
    "is_youtube_url",
    "fetch_youtube_transcript",
    "format_youtube_transcript_for_context",
    "get_youtube_transcript_context",
]
