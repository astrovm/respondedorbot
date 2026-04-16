import pytest
from unittest.mock import patch, MagicMock

from api.utils.youtube_transcript import (
    extract_youtube_video_id,
    fetch_youtube_transcript,
    format_youtube_transcript_for_context,
    get_youtube_transcript_context,
)


class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert (
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_short_youtu_be_url(self):
        assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embedded_url(self):
        assert (
            extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_short_url(self):
        assert (
            extract_youtube_video_id("https://youtube.com/shorts/dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_youtube_url_with_extra_params(self):
        assert (
            extract_youtube_video_id(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
            )
            == "dQw4w9WgXcQ"
        )

    def test_mobile_url(self):
        assert (
            extract_youtube_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_no_url_returns_none(self):
        assert extract_youtube_video_id("") is None
        assert extract_youtube_video_id(None) is None

    def test_non_youtube_url_returns_none(self):
        assert extract_youtube_video_id("https://www.google.com") is None
        assert extract_youtube_video_id("https://vimeo.com/123456789") is None


class TestFormatTranscriptForContext:
    def test_format_with_timestamps(self):
        transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.5, "text": "This is a test"},
            {"start": 120.3, "text": "Two minutes in"},
        ]
        result = format_youtube_transcript_for_context(transcript)

        assert "[00:00] Hello world" in result
        assert "[00:05] This is a test" in result
        assert "[02:00] Two minutes in" in result

    def test_format_without_timestamps(self):
        transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.5, "text": "This is a test"},
        ]
        result = format_youtube_transcript_for_context(
            transcript, include_timestamps=False
        )

        assert "Hello world" in result
        assert "This is a test" in result
        assert "[00:00]" not in result

    def test_empty_transcript(self):
        assert format_youtube_transcript_for_context([]) == ""

    def test_skips_empty_text_segments(self):
        transcript = [
            {"start": 0.0, "text": "Hello"},
            {"start": 1.0, "text": ""},
            {"start": 2.0, "text": "  "},
            {"start": 3.0, "text": "World"},
        ]
        result = format_youtube_transcript_for_context(transcript)

        assert "Hello" in result
        assert "World" in result
        assert "[00:01]" not in result
        assert "[00:02]" not in result


class TestFetchYouTubeTranscript:
    @patch("api.utils.youtube_transcript.YouTubeTranscriptApi")
    def test_successful_fetch(self, mock_yta):
        mock_transcript_list = MagicMock()
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.0, "text": "Test transcript"},
        ]
        mock_transcript_list.find_transcript.return_value = mock_transcript
        mock_yta.list_transcripts.return_value = mock_transcript_list

        transcript, error = fetch_youtube_transcript("dQw4w9WgXcQ")

        assert transcript is not None
        assert len(transcript) == 2
        assert error is None
        mock_yta.list_transcripts.assert_called_once_with("dQw4w9WgXcQ")

    @patch("api.utils.youtube_transcript.YouTubeTranscriptApi")
    def test_no_transcript_found(self, mock_yta):
        from youtube_transcript_api._errors import NoTranscriptFound

        mock_yta.list_transcripts.side_effect = NoTranscriptFound(
            "No transcript available"
        )

        transcript, error = fetch_youtube_transcript("dQw4w9WgXcQ")

        assert transcript is None
        assert error == "no transcript found"

    @patch("api.utils.youtube_transcript.YouTubeTranscriptApi")
    def test_video_unavailable(self, mock_yta):
        from youtube_transcript_api._errors import VideoUnavailable

        mock_yta.list_transcripts.side_effect = VideoUnavailable("Video not found")

        transcript, error = fetch_youtube_transcript("invalid_video_id")

        assert transcript is None
        assert error == "video unavailable"


class TestGetYouTubeTranscriptContext:
    @patch("api.utils.youtube_transcript.fetch_youtube_transcript")
    def test_returns_formatted_context(self, mock_fetch):
        mock_fetch.return_value = (
            [
                {"start": 0.0, "text": "Hello world"},
                {"start": 5.0, "text": "This is a test"},
            ],
            None,
        )

        result = get_youtube_transcript_context("dQw4w9WgXcQ")

        assert "TRANSCRIPCION DEL VIDEO:" in result
        assert "[00:00] Hello world" in result
        assert "[00:05] This is a test" in result

    @patch("api.utils.youtube_transcript.fetch_youtube_transcript")
    def test_returns_empty_on_error(self, mock_fetch):
        mock_fetch.return_value = (None, "no transcript found")

        result = get_youtube_transcript_context("dQw4w9WgXcQ")

        assert result == ""

    @patch("api.utils.youtube_transcript.fetch_youtube_transcript")
    def test_returns_empty_on_empty_transcript(self, mock_fetch):
        mock_fetch.return_value = ([], None)

        result = get_youtube_transcript_context("dQw4w9WgXcQ")

        assert result == ""
