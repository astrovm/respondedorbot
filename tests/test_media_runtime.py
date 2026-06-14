from tests.support import *


def test_handle_transcribe_with_message_no_reply():
    from api.index import handle_transcribe_with_message

    message = {"message_id": 1, "chat": {"id": 123}, "text": "/transcribe"}

    result = handle_transcribe_with_message(message)
    assert (
        result
        == "respondeme un audio, video, imagen o sticker y te digo qué carajo hay ahí"
    )


def test_handle_transcribe_with_message_voice_cached():
    from api.index import handle_transcribe_with_message

    with patch("api.index.app_runtime.media_cache.get_transcription") as mock_cached:
        mock_cached.return_value = "cached voice transcription"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 te saqué esto del audio: cached voice transcription"
        mock_cached.assert_called_once_with("voice123")


def test_handle_transcribe_with_message_voice_download_success():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_transcription") as mock_cached,
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
        patch(
            "api.index.app_runtime.media._deps.measure_duration", return_value=1.0
        ) as mock_measure,
        patch("api.index.app_runtime.media.transcribe_audio_result") as mock_transcribe,
    ):
        mock_cached.return_value = None
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = MagicMock(
            text="new transcription",
            kind="transcribe",
            audio_seconds=1.0,
            billing_segment=lambda: {"kind": "transcribe", "audio_seconds": 1.0},
        )

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 te saqué esto del audio: new transcription"
        mock_download.assert_called_once_with("voice123")
        mock_measure.assert_called_once_with(b"audio data")
        mock_transcribe.assert_called_once_with(
            b"audio data", "voice123", use_cache=True
        )


def test_handle_transcribe_with_message_voice_download_fail():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_transcription") as mock_cached,
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
    ):
        mock_cached.return_value = None
        mock_download.return_value = None

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"voice": {"file_id": "voice123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "no pude bajar el audio, mandalo de nuevo"


def test_handle_transcribe_with_message_audio_success():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_transcription") as mock_cached,
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
        patch(
            "api.index.app_runtime.media._deps.measure_duration", return_value=1.0
        ) as mock_measure,
        patch("api.index.app_runtime.media.transcribe_audio_result") as mock_transcribe,
    ):
        mock_cached.return_value = None
        mock_download.return_value = b"audio data"
        mock_transcribe.return_value = MagicMock(
            text="audio transcription",
            kind="transcribe",
            audio_seconds=1.0,
            billing_segment=lambda: {"kind": "transcribe", "audio_seconds": 1.0},
        )

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"audio": {"file_id": "audio123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎵 te saqué esto del audio: audio transcription"
        mock_measure.assert_called_once_with(b"audio data")


def test_handle_transcribe_with_message_photo_cached():
    from api.index import handle_transcribe_with_message

    with patch("api.index.app_runtime.media_cache.get_description") as mock_cached:
        mock_cached.return_value = "cached image description"

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"photo": [{"file_id": "photo123"}]},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🖼️ en la imagen veo: cached image description"
        mock_cached.assert_called_once_with("photo123")


def test_handle_transcribe_with_message_photo_success():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_description") as mock_cached,
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
        patch("api.index.app_runtime.media.describe_image_result") as mock_describe,
    ):
        mock_cached.return_value = None
        mock_download.return_value = b"image data"
        mock_describe.return_value = MagicMock(
            text="image description",
            kind="vision",
            billing_segment=lambda: {"kind": "vision"},
        )

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"photo": [{"file_id": "photo123"}]},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🖼️ en la imagen veo: image description"
        mock_download.assert_called_once_with("photo123")
        mock_describe.assert_called_once_with(
            b"image data",
            "describí lo que ves en esta imagen en detalle, en minúsculas, sin emojis, sin markdown, en lenguaje coloquial argentino",
            "photo123",
        )


def test_handle_transcribe_with_message_sticker_success():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_description") as mock_cached,
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
        patch("api.index.app_runtime.media.describe_image_result") as mock_describe,
    ):
        mock_cached.return_value = None
        mock_download.return_value = b"sticker data"
        mock_describe.return_value = MagicMock(
            text="sticker description",
            kind="vision",
            billing_segment=lambda: {"kind": "vision"},
        )

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {"sticker": {"file_id": "sticker123"}},
        }

        result = handle_transcribe_with_message(message)
        assert result == "🎨 en el sticker veo: sticker description"


def test_handle_transcribe_with_message_animated_sticker_uses_thumbnail():
    from api.index import handle_transcribe_with_message

    with (
        patch("api.index.app_runtime.media_cache.get_description", return_value=None),
        patch("api.index.app_runtime.telegram.download_file") as mock_download,
        patch("api.index.app_runtime.images.resize", return_value=b"thumb image"),
        patch("api.index.app_runtime.media.describe_image_result") as mock_describe,
    ):
        mock_download.return_value = b"thumb image"
        mock_describe.return_value = MagicMock(
            text="animated sticker description",
            kind="vision",
            billing_segment=lambda: {"kind": "vision"},
        )

        message = {
            "message_id": 1,
            "chat": {"id": 123},
            "text": "/transcribe",
            "reply_to_message": {
                "sticker": {
                    "file_id": "animated_sticker",
                    "is_animated": True,
                    "thumbnail": {"file_id": "sticker_thumb"},
                }
            },
        }

        result = handle_transcribe_with_message(message)

    assert result == "🎨 en el sticker veo: animated sticker description"
    mock_download.assert_called_once_with("sticker_thumb")


def test_handle_transcribe_with_message_no_media():
    from api.index import handle_transcribe_with_message

    message = {
        "message_id": 1,
        "chat": {"id": 123},
        "text": "/transcribe",
        "reply_to_message": {"text": "just text"},
    }

    result = handle_transcribe_with_message(message)
    assert result == "ese mensaje no tiene audio, video, imagen ni sticker para laburar"


def test_handle_transcribe_with_message_exception():
    from api.index import handle_transcribe_with_message
    from typing import cast, Dict, Any

    # Malformed message that causes exception
    message = cast(Dict[str, Any], None)

    result = handle_transcribe_with_message(message)
    assert result == "se trabó el /transcribe, probá más tarde"


def test_handle_transcribe():
    from api.index import handle_transcribe

    result = handle_transcribe()
    assert result == "el /transcribe se usa respondiendo a un audio, imagen o sticker"


def test_download_telegram_file_success():
    """Test download_telegram_file with successful download"""
    with (
        patch("api.bot.telegram.environ.get") as mock_env,
        patch("api.index.telegram_gateway._telegram_request") as mock_telegram_request,
        patch("api.services.http_client.get") as mock_get,
    ):
        mock_env.side_effect = lambda key, default=None: (
            "test_token" if key == "TELEGRAM_TOKEN" else default
        )
        mock_telegram_request.return_value = (
            {"ok": True, "result": {"file_path": "photos/file_123.jpg"}},
            None,
        )

        # Mock file download response
        mock_file_download = MagicMock()
        mock_file_download.content = b"fake image data"
        mock_file_download.raise_for_status.return_value = None
        mock_get.return_value = mock_file_download

        result = index.app_runtime.telegram.download_file("test_file_id")

        assert result == b"fake image data"
        mock_telegram_request.assert_called_once_with(
            "getFile",
            method="GET",
            params={"file_id": "test_file_id"},
            log_errors=False,
        )
        mock_get.assert_called_once_with(
            "https://api.telegram.org/file/bottest_token/photos/file_123.jpg",
            timeout=30,
        )


def test_download_telegram_file_api_error():
    """Test download_telegram_file with API error"""
    with (
        patch("api.bot.telegram.environ.get") as mock_env,
        patch("api.services.http_client.get") as mock_get,
    ):
        mock_env.side_effect = lambda key, default=None: (
            "test_token" if key == "TELEGRAM_TOKEN" else default
        )

        # Mock failed API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "File not found"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = index.app_runtime.telegram.download_file("invalid_file_id")

        assert result is None


def test_download_telegram_file_network_error():
    """Test download_telegram_file with network error"""
    with (
        patch("api.bot.telegram.environ.get") as mock_env,
        patch("api.services.http_client.get") as mock_get,
    ):
        mock_env.side_effect = lambda key, default=None: (
            "test_token" if key == "TELEGRAM_TOKEN" else default
        )
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = index.app_runtime.telegram.download_file("test_file_id")

        assert result is None


def test_encode_image_to_base64_success():
    """Test encode_image_to_base64 with valid image data"""
    encode_image_to_base64 = index.app_runtime.images.encode
    import base64

    test_data = b"fake image bytes"
    expected = base64.b64encode(test_data).decode("utf-8")

    result = encode_image_to_base64(test_data)

    assert result == expected
    assert isinstance(result, str)


def test_encode_image_to_base64_empty():
    """Test encode_image_to_base64 with empty data"""
    encode_image_to_base64 = index.app_runtime.images.encode

    result = encode_image_to_base64(b"")

    assert result == ""


def test_resize_image_if_needed_no_resize():
    """Test resize_image_if_needed always returns WEBP even when no resize needed"""
    resize_image_if_needed = index.app_runtime.images.resize

    with (
        patch("api.index.app_runtime.images.image_module") as mock_image_module,
        patch("api.index.io.BytesIO") as mock_bytesio,
    ):
        mock_image = MagicMock()
        mock_image.size = (200, 150)
        mock_image.mode = "RGB"
        mock_image_module.open.return_value = mock_image

        mock_output_buffer = MagicMock()
        mock_output_buffer.getvalue.return_value = b"webp bytes"
        mock_bytesio.return_value = mock_output_buffer

        test_data = b"small image data"
        result = resize_image_if_needed(test_data, max_size=512)

        assert result == b"webp bytes"
        mock_image.save.assert_called_once()


def test_resize_image_if_needed_with_resize():
    """Test resize_image_if_needed when image needs resizing"""
    resize_image_if_needed = index.app_runtime.images.resize

    with (
        patch("api.index.app_runtime.images.image_module") as mock_image_module,
        patch("api.index.io.BytesIO") as mock_bytesio,
    ):
        # Mock large image that needs resizing
        mock_image = MagicMock()
        mock_image.size = (1024, 768)  # Larger than max_size of 512
        mock_image.mode = "RGB"

        # Mock resized image
        mock_resized = MagicMock()
        mock_image.resize.return_value = mock_resized

        # Mock output buffer
        mock_output_buffer = MagicMock()
        mock_output_buffer.getvalue.return_value = b"resized image data"
        mock_bytesio.return_value = mock_output_buffer

        mock_image_module.open.return_value = mock_image
        mock_image_module.Resampling.LANCZOS = "LANCZOS"

        test_data = b"large image data"
        result = resize_image_if_needed(test_data, max_size=512)

        assert result == b"resized image data"
        mock_image.resize.assert_called_once()
        mock_resized.save.assert_called_once()


def test_resize_image_if_needed_rgba_conversion():
    """Test resize_image_if_needed with RGBA image conversion"""
    resize_image_if_needed = index.app_runtime.images.resize

    with (
        patch("api.index.app_runtime.images.image_module") as mock_image_module,
        patch("api.index.io.BytesIO") as mock_bytesio,
    ):
        # Mock large RGBA image that needs conversion
        mock_image = MagicMock()
        mock_image.size = (1024, 768)
        mock_image.mode = "RGBA"

        # Mock converted image
        mock_converted = MagicMock()
        mock_image.convert.return_value = mock_converted

        # Mock resized image that also has RGBA mode
        mock_resized = MagicMock()
        mock_resized.mode = "RGBA"
        mock_resized_converted = MagicMock()
        mock_resized.convert.return_value = mock_resized_converted
        mock_image.resize.return_value = mock_resized

        # Mock output buffer
        mock_output_buffer = MagicMock()
        mock_output_buffer.getvalue.return_value = b"converted resized image"
        mock_bytesio.return_value = mock_output_buffer

        mock_image_module.open.return_value = mock_image
        mock_image_module.Resampling.LANCZOS = "LANCZOS"

        result = resize_image_if_needed(b"rgba image data")

        assert result == b"converted resized image"
        mock_resized.convert.assert_called_once_with("RGB")


def test_resize_image_if_needed_import_error():
    """Test resize_image_if_needed when PIL is not available"""
    resize_image_if_needed = index.app_runtime.images.resize

    with patch("api.index.app_runtime.images.image_module") as mock_image_module:
        mock_image_module.open.side_effect = ImportError("PIL not available")

        test_data = b"image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on ImportError


def test_resize_image_if_needed_processing_error():
    """Test resize_image_if_needed with image processing error"""
    resize_image_if_needed = index.app_runtime.images.resize

    with patch("api.index.app_runtime.images.image_module") as mock_image_module:
        mock_image_module.open.side_effect = Exception("Invalid image format")

        test_data = b"corrupted image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on error


def test_describe_image_groq_success():
    """Test describe_image_groq with successful OpenRouter response"""
    describe_image_groq = index.app_runtime.media.describe_image

    choice = MagicMock()
    choice.message.content = "A beautiful landscape with mountains"

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response

    with (
        patch("api.index.app_runtime.media._deps.get_openrouter_client", return_value=client),
        patch("api.index.app_runtime.images.prepare", return_value=(b"image_data", "image/webp")),
    ):
        result = describe_image_groq(b"image_data")

    assert result == "A beautiful landscape with mountains"
    client.chat.completions.create.assert_called_once()


def test_describe_image_groq_api_error():
    """Test describe_image_groq with OpenRouter API error"""
    describe_image_groq = index.app_runtime.media.describe_image

    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API error")

    with patch("api.index.app_runtime.media._deps.get_openrouter_client", return_value=client):
        result = describe_image_groq(b"image_data")

    assert result is None


def test_describe_image_groq_no_client():
    """Test describe_image_groq when OpenRouter client is unavailable"""
    describe_image_groq = index.app_runtime.media.describe_image

    with patch("api.index.app_runtime.media._deps.get_openrouter_client", return_value=None):
        result = describe_image_groq(b"image_data")

    assert result is None


def test_transcribe_audio_groq_success():
    """Test transcribe_audio_groq with successful transcription"""
    transcribe_audio_groq = index.app_runtime.media.transcribe_audio

    with (
        patch("api.index.environ.get") as mock_env,
        patch("api.index.app_runtime.media._deps.get_groq_native_client") as mock_groq,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_FREE_API_KEY": "test_api_key",
        }.get(key, default)

        fake_response = MagicMock()
        fake_response.text = "Hello, this is a test audio transcription"

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_response
        mock_groq.return_value = fake_client

        result = transcribe_audio_groq(b"audio_data")

        assert result == "Hello, this is a test audio transcription"
        fake_client.audio.transcriptions.create.assert_called_once()


def test_transcribe_audio_groq_network_error():
    """Test transcribe_audio_groq with network error"""
    transcribe_audio_groq = index.app_runtime.media.transcribe_audio

    with (
        patch("api.index.environ.get") as mock_env,
        patch("api.index.app_runtime.media._deps.get_groq_native_client") as mock_groq,
    ):
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_FREE_API_KEY": "test_api_key",
        }.get(key, default)

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.side_effect = Exception("Network error")
        mock_groq.return_value = fake_client

        result = transcribe_audio_groq(b"audio_data")

        assert result is None


def test_transcribe_audio_groq_skips_call_when_provider_in_cooldown():
    transcribe_audio_groq = index.app_runtime.media.transcribe_audio
    from api.providers.backoff import mark_provider_cooldown, clear_all_cooldowns

    clear_all_cooldowns()
    mark_provider_cooldown("groq:free:transcribe", 300)
    mark_provider_cooldown("groq:paid:transcribe", 300)

    with (
        patch("api.index.environ.get") as mock_env,
        patch("api.index.app_runtime.media._deps.get_groq_native_client", return_value=None),
    ):
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_FREE_API_KEY": "test_api_key",
        }.get(key, default)

        result = transcribe_audio_groq(b"audio_data")

        assert result is None

    clear_all_cooldowns()


def test_transcribe_audio_groq_falls_back_to_paid_after_free_429(monkeypatch):
    transcribe_audio_groq = index.app_runtime.media.transcribe_audio

    monkeypatch.setenv("GROQ_FREE_API_KEY", "free_api_key")
    monkeypatch.setenv("GROQ_API_KEY", "paid_api_key")

    free_client = MagicMock()
    free_client.audio.transcriptions.create.side_effect = Exception(
        "Error code: 429 - rate limit reached"
    )

    paid_choice = MagicMock()
    paid_choice.text = "transcribed text"

    paid_client = MagicMock()
    paid_client.audio.transcriptions.create.return_value = paid_choice

    with patch(
        "api.index.app_runtime.media._deps.get_groq_native_client", side_effect=[free_client, paid_client]
    ) as mock_groq:
        result = transcribe_audio_groq(b"audio_data")

    assert result is not None
    assert mock_groq.call_count == 2
    assert mock_groq.call_args_list[0].args == ("free",)
    assert mock_groq.call_args_list[1].args == ("paid",)


# Tests for video transcription support


def test_extract_message_content_video():
    from api.index import extract_message_content

    message = {"text": "", "video": {"file_id": "vid123"}}
    text, photo_id, audio_id = extract_message_content(message)
    assert audio_id == "vid123"


def test_extract_message_content_video_note():
    from api.index import extract_message_content

    message = {"text": "", "video_note": {"file_id": "vn456"}}
    text, photo_id, audio_id = extract_message_content(message)
    assert audio_id == "vn456"


def test_extract_message_content_video_in_reply():
    from api.index import extract_message_content

    message = {
        "text": "/transcribe",
        "reply_to_message": {"video": {"file_id": "vid_reply"}},
    }
    text, photo_id, audio_id = extract_message_content(message)
    assert audio_id == "vid_reply"


def test_extract_message_content_video_note_in_reply():
    from api.index import extract_message_content

    message = {
        "text": "/transcribe",
        "reply_to_message": {"video_note": {"file_id": "vn_reply"}},
    }
    text, photo_id, audio_id = extract_message_content(message)
    assert audio_id == "vn_reply"


def test_extract_message_content_audio_takes_priority_over_video():
    from api.index import extract_message_content

    message = {
        "text": "",
        "audio": {"file_id": "aud1"},
        "video": {"file_id": "vid1"},
    }
    text, photo_id, audio_id = extract_message_content(message)
    assert audio_id == "aud1"


def test_extract_message_content_animated_sticker_uses_thumbnail():
    from api.index import extract_message_content

    message = {
        "sticker": {
            "file_id": "animated_sticker",
            "is_animated": True,
            "thumbnail": {"file_id": "sticker_thumb"},
        }
    }

    text, photo_id, audio_id = extract_message_content(message)

    assert photo_id == "sticker_thumb"
    assert audio_id is None


def test_transcribe_file_by_id_video_fallback():
    transcribe_file_by_id = index.app_runtime.media.transcribe_file

    with (
        patch("api.index.app_runtime.media_cache.get_transcription", return_value=None),
        patch("api.index.app_runtime.telegram.download_file", return_value=b"video bytes"),
        patch("api.index.app_runtime.media._deps.measure_duration") as mock_measure,
        patch("api.index.app_runtime.media._deps.extract_audio") as mock_extract,
        patch("api.index.app_runtime.media.transcribe_audio_result") as mock_transcribe,
    ):
        # First call (video bytes) returns None, second call (extracted audio) returns 5.0
        mock_measure.side_effect = [None, 5.0]
        mock_extract.return_value = b"extracted audio"
        mock_transcribe.return_value = MagicMock(
            text="transcribed from video",
            audio_seconds=5.0,
            billing_segment=lambda: {"kind": "transcribe", "audio_seconds": 5.0},
        )

        text, error, billing = transcribe_file_by_id("file123", use_cache=False)
        assert text == "transcribed from video"
        assert error is None
        mock_extract.assert_called_once_with(b"video bytes")


def test_transcribe_file_by_id_video_extraction_fails():
    transcribe_file_by_id = index.app_runtime.media.transcribe_file

    with (
        patch("api.index.app_runtime.media_cache.get_transcription", return_value=None),
        patch("api.index.app_runtime.telegram.download_file", return_value=b"bad data"),
        patch("api.index.app_runtime.media._deps.measure_duration", return_value=None),
        patch("api.index.app_runtime.media._deps.extract_audio", return_value=None),
    ):
        text, error, billing = transcribe_file_by_id("file123", use_cache=False)
        assert text is None
        assert error == "transcribe"


def test_extract_audio_duration_seconds_video():
    from api.bot.message_handler import _extract_audio_duration_seconds

    message = {"video": {"duration": 42}}
    assert _extract_audio_duration_seconds(message) == 42.0


def test_extract_audio_duration_seconds_video_note():
    from api.bot.message_handler import _extract_audio_duration_seconds

    message = {"video_note": {"duration": 15}}
    assert _extract_audio_duration_seconds(message) == 15.0


def test_extract_audio_duration_seconds_video_in_reply():
    from api.bot.message_handler import _extract_audio_duration_seconds

    message = {"reply_to_message": {"video": {"duration": 30}}}
    assert _extract_audio_duration_seconds(message) == 30.0
