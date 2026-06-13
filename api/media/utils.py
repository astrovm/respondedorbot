from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from importlib import import_module
from typing import Any, Callable, cast

from api.core.logging import get_logger

logger = get_logger(__name__)
MutagenFile = cast(Callable[[Any], Any], import_module("mutagen").File)


def measure_audio_duration_seconds(audio_data: bytes) -> float | None:
    if not audio_data:
        return None

    try:
        parsed = MutagenFile(io.BytesIO(audio_data))
        length = getattr(getattr(parsed, "info", None), "length", None)
        if isinstance(length, (int, float)) and length > 0:
            return float(length)
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate > 0 and frame_count > 0:
                return float(frame_count) / float(frame_rate)
    except Exception:
        pass
    return None


def extract_audio_from_video(video_data: bytes) -> bytes | None:
    if not video_data:
        return None
    try:
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4") as video_file,
            tempfile.NamedTemporaryFile(suffix=".ogg") as audio_file,
        ):
            video_file.write(video_data)
            video_file.flush()
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_file.name,
                    "-vn",
                    "-acodec",
                    "libopus",
                    "-b:a",
                    "64k",
                    audio_file.name,
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                print(f"ffmpeg failed: {result.stderr[:500]!r}")
                return None
            audio_file.seek(0)
            audio_bytes = audio_file.read()
            return audio_bytes or None
    except Exception:
        logger.exception("Error extracting audio from video")
        return None
