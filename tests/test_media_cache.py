from tests.support import *


@pytest.mark.parametrize(
    "prefix,file_id,cached_value",
    [
        ("audio_transcription", "audio_file", "cached transcription text"),
        ("image_description", "image_file", "cached image description"),
    ],
)
def test_get_cached_media_success(prefix, file_id, cached_value):
    from api.media.cache import get_cached_media

    mock_redis = make_mock_config_redis()
    mock_redis.get.return_value = cached_value

    result = get_cached_media(
        prefix,
        file_id,
        redis_factory=lambda: mock_redis,
        logger=MagicMock(),
    )

    assert result == cached_value
    mock_redis.get.assert_called_once_with(f"{prefix}:{file_id}")


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_get_cached_media_not_found(prefix):
    from api.media.cache import get_cached_media

    mock_redis = make_mock_config_redis()
    mock_redis.get.return_value = None

    result = get_cached_media(
        prefix,
        "test_file_id",
        redis_factory=lambda: mock_redis,
        logger=MagicMock(),
    )

    assert result is None
    mock_redis.get.assert_called_once_with(f"{prefix}:test_file_id")


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_get_cached_media_exception(prefix):
    from api.media.cache import get_cached_media

    def fail_redis():
        raise Exception("Redis error")

    result = get_cached_media(
        prefix,
        "test_file_id",
        redis_factory=fail_redis,
        logger=MagicMock(),
    )

    assert result is None


@pytest.mark.parametrize(
    "prefix,file_id,text,ttl",
    [
        ("audio_transcription", "audio_file", "transcription text", 3600),
        ("image_description", "image_file", "image description", 7200),
    ],
)
def test_cache_media_success(prefix, file_id, text, ttl):
    from api.media.cache import cache_media

    mock_redis = make_mock_config_redis()

    cache_media(
        prefix,
        file_id,
        text,
        ttl,
        redis_factory=lambda: mock_redis,
        logger=MagicMock(),
    )

    mock_redis.setex.assert_called_once_with(f"{prefix}:{file_id}", ttl, text)


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_cache_media_exception(prefix):
    from api.media.cache import cache_media

    def fail_redis():
        raise Exception("Redis error")

    cache_media(
        prefix,
        "test_file_id",
        "payload",
        3600,
        redis_factory=fail_redis,
        logger=MagicMock(),
    )


def test_get_cached_transcription_delegates_to_helper():
    get_cached_transcription = index.app_runtime.media_cache.get_transcription

    with patch("api.index.app_runtime.media_cache.get") as mock_get:
        mock_get.return_value = "cached transcription text"

        result = get_cached_transcription("test_file_id")

        assert result == "cached transcription text"
        mock_get.assert_called_once_with("audio_transcription", "test_file_id")


def test_get_cached_description_delegates_to_helper():
    get_cached_description = index.app_runtime.media_cache.get_description

    with patch("api.index.app_runtime.media_cache.get") as mock_get:
        mock_get.return_value = "cached image description"

        result = get_cached_description("test_file_id")

        assert result == "cached image description"
        mock_get.assert_called_once_with("image_description", "test_file_id")


def test_cache_transcription_delegates_default_ttl():
    cache_transcription = index.app_runtime.media_cache.cache_transcription

    with patch("api.index.app_runtime.media_cache.set") as mock_cache:
        cache_transcription("test_file_id", "transcription text")

        mock_cache.assert_called_once_with(
            "audio_transcription", "test_file_id", "transcription text", TTL_MEDIA_CACHE
        )


def test_cache_transcription_delegates_custom_ttl():
    cache_transcription = index.app_runtime.media_cache.cache_transcription

    with patch("api.index.app_runtime.media_cache.set") as mock_cache:
        cache_transcription("test_file_id", "transcription text", 3600)

        mock_cache.assert_called_once_with(
            "audio_transcription", "test_file_id", "transcription text", 3600
        )


def test_cache_description_delegates_default_ttl():
    cache_description = index.app_runtime.media_cache.cache_description

    with patch("api.index.app_runtime.media_cache.set") as mock_cache:
        cache_description("test_file_id", "image description")

        mock_cache.assert_called_once_with(
            "image_description", "test_file_id", "image description", TTL_MEDIA_CACHE
        )


def test_cache_description_delegates_custom_ttl():
    cache_description = index.app_runtime.media_cache.cache_description

    with patch("api.index.app_runtime.media_cache.set") as mock_cache:
        cache_description("test_file_id", "image description", 7200)

        mock_cache.assert_called_once_with(
            "image_description", "test_file_id", "image description", 7200
        )


def test_get_cache_history_success():
    get_cache_history = index.app_runtime.cache.get_history
    import json
    from datetime import datetime, timedelta
    from api.services.maintenance import request_cache_history_key

    with patch("api.cache.service.datetime") as mock_datetime:
        mock_redis = MagicMock()
        test_data = {"data": "test", "timestamp": "2024-01-01"}
        mock_redis.get.return_value = json.dumps(test_data)

        # Mock datetime.now() to return a fixed time
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        result = get_cache_history(1, "test_hash", mock_redis)

        assert result == test_data
        expected_timestamp = (fixed_time - timedelta(hours=1)).strftime("%Y-%m-%d-%H")
        mock_redis.get.assert_called_once_with(
            request_cache_history_key(expected_timestamp, "test_hash")
        )


def test_get_cache_history_not_found():
    get_cache_history = index.app_runtime.cache.get_history

    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    result = get_cache_history(1, "test_hash", mock_redis)

    assert result is None


def test_get_cache_history_invalid_data():
    get_cache_history = index.app_runtime.cache.get_history
    import json

    mock_redis = MagicMock()
    test_data = {"data": "test"}  # Missing timestamp
    mock_redis.get.return_value = json.dumps(test_data)

    result = get_cache_history(1, "test_hash", mock_redis)

    assert result is None
