"""YouTube transcript extraction utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

__all__ = [
    "extract_youtube_video_id",
    "is_youtube_url",
    "fetch_youtube_transcript",
    "format_youtube_transcript_for_context",
    "get_youtube_transcript_context",
]


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

YOUTUBE_REGEX = re.compile(
    r"(?:https?://)?(?:"
    r"(?:www\.)?youtube\.com(?:/[^/\s]+)*"
    r"|"
    r"youtu\.be"
    r")"
    r"(?:/[^\s]*)?"
)


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract the 11-character video ID from a YouTube URL.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - Shorts URLs, live URLs, etc.
    """
    if not url:
        return None

    parsed_url = re.search(
        r"(?:v=|/v/|/embed/|/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        url,
    )
    if parsed_url:
        return parsed_url.group(1)

    return None


def is_youtube_url(url: str) -> bool:
    """Return True if the URL is a YouTube video URL."""
    if not url:
        return False

    video_id = extract_youtube_video_id(url)
    if video_id:
        return True

    url_lower = url.lower()
    return any(host in url_lower for host in YOUTUBE_HOSTS)


def fetch_youtube_transcript(
    video_id: str,
    languages: List[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Fetch transcript for a YouTube video.

    Args:
        video_id: The 11-character YouTube video ID.
        languages: Preferred languages in order of preference. Defaults to ['en', 'es'].

    Returns:
        Tuple of (transcript_list, error_message). transcript_list is None if
        fetching failed. error_message is None on success.
    """
    if languages is None:
        languages = ["en", "es"]

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(languages)
        except Exception:
            try:
                transcript = transcript_list.find_transcript(["en"])
            except Exception:
                try:
                    transcript = transcript_list.find_transcript(["es"])
                except Exception:
                    try:
                        transcript = transcript_list.find_generated_transcript(
                            languages
                        )
                    except Exception:
                        return None, "no transcript found"

        fetched_transcript = transcript.fetch()
        return fetched_transcript, None

    except VideoUnavailable:
        return None, "video unavailable"
    except TranscriptsDisabled:
        return None, "transcripts disabled"
    except NoTranscriptFound:
        return None, "no transcript found"
    except CouldNotRetrieveTranscript as e:
        print(f"[YOUTUBE_TRANSCRIPT] CouldNotRetrieveTranscript: {e}")
        return None, "could not retrieve transcript"
    except Exception as e:
        print(f"[YOUTUBE_TRANSCRIPT] Unexpected error: {e}")
        return None, f"error: {str(e)}"


def format_youtube_transcript_for_context(
    transcript: List[Dict[str, Any]],
    include_timestamps: bool = True,
) -> str:
    """Format a YouTube transcript for AI context.

    Args:
        transcript: List of transcript segments from youtube-transcript-api.
        include_timestamps: If True, include [MM:SS] timestamps before each segment.

    Returns:
        Formatted transcript string.
    """
    if not transcript:
        return ""

    lines = []
    for segment in transcript:
        start = segment.get("start", 0)
        text = segment.get("text", "").strip()

        if not text:
            continue

        if include_timestamps:
            minutes = int(start) // 60
            seconds = int(start) % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            lines.append(f"{timestamp} {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def get_youtube_transcript_context(
    video_id: str,
    languages: List[str] = None,
) -> str:
    """Get formatted transcript context for a YouTube video.

    Args:
        video_id: The 11-character YouTube video ID.
        languages: Preferred languages in order of preference.

    Returns:
        Formatted transcript string, or empty string if fetching failed.
    """
    transcript, error = fetch_youtube_transcript(video_id, languages)

    if error or not transcript:
        print(
            f"[YOUTUBE_TRANSCRIPT] Failed to fetch transcript for {video_id}: {error}"
        )
        return ""

    formatted = format_youtube_transcript_for_context(transcript)

    if not formatted:
        return ""

    return f"TRANSCRIPCION DEL VIDEO:\n{formatted}"
