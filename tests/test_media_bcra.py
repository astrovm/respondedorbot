from tests.support import *  # noqa: F401,F403

@pytest.mark.parametrize(
    "prefix,file_id,cached_value",
    [
        ("audio_transcription", "audio_file", "cached transcription text"),
        ("image_description", "image_file", "cached image description"),
    ],
)
def test_get_cached_media_success(prefix, file_id, cached_value):
    from api.index import _get_cached_media

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = cached_value

        result = _get_cached_media(prefix, file_id)

        assert result == cached_value
        mock_redis.get.assert_called_once_with(f"{prefix}:{file_id}")


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_get_cached_media_not_found(prefix):
    from api.index import _get_cached_media

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = None

        result = _get_cached_media(prefix, "test_file_id")

        assert result is None
        mock_redis.get.assert_called_once_with(f"{prefix}:test_file_id")


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_get_cached_media_exception(prefix):
    from api.index import _get_cached_media

    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")

        result = _get_cached_media(prefix, "test_file_id")

        assert result is None


@pytest.mark.parametrize(
    "prefix,file_id,text,ttl",
    [
        ("audio_transcription", "audio_file", "transcription text", 3600),
        ("image_description", "image_file", "image description", 7200),
    ],
)
def test_cache_media_success(prefix, file_id, text, ttl):
    from api.index import _cache_media

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        _cache_media(prefix, file_id, text, ttl)

        mock_redis.setex.assert_called_once_with(f"{prefix}:{file_id}", ttl, text)


@pytest.mark.parametrize(
    "prefix",
    ["audio_transcription", "image_description"],
)
def test_cache_media_exception(prefix):
    from api.index import _cache_media

    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")

        # Should not raise exception, just print error
        _cache_media(prefix, "test_file_id", "payload", 3600)


def test_get_cached_transcription_delegates_to_helper():
    from api.index import get_cached_transcription

    with patch("api.index._get_cached_media") as mock_get:
        mock_get.return_value = "cached transcription text"

        result = get_cached_transcription("test_file_id")

        assert result == "cached transcription text"
        mock_get.assert_called_once_with("audio_transcription", "test_file_id")


def test_get_cached_description_delegates_to_helper():
    from api.index import get_cached_description

    with patch("api.index._get_cached_media") as mock_get:
        mock_get.return_value = "cached image description"

        result = get_cached_description("test_file_id")

        assert result == "cached image description"
        mock_get.assert_called_once_with("image_description", "test_file_id")


def test_cache_transcription_delegates_default_ttl():
    from api.index import cache_transcription

    with patch("api.index._cache_media") as mock_cache:
        cache_transcription("test_file_id", "transcription text")

        mock_cache.assert_called_once_with(
            "audio_transcription", "test_file_id", "transcription text", TTL_MEDIA_CACHE
        )


def test_cache_transcription_delegates_custom_ttl():
    from api.index import cache_transcription

    with patch("api.index._cache_media") as mock_cache:
        cache_transcription("test_file_id", "transcription text", 3600)

        mock_cache.assert_called_once_with(
            "audio_transcription", "test_file_id", "transcription text", 3600
        )


def test_cache_description_delegates_default_ttl():
    from api.index import cache_description

    with patch("api.index._cache_media") as mock_cache:
        cache_description("test_file_id", "image description")

        mock_cache.assert_called_once_with(
            "image_description", "test_file_id", "image description", TTL_MEDIA_CACHE
        )


def test_cache_description_delegates_custom_ttl():
    from api.index import cache_description

    with patch("api.index._cache_media") as mock_cache:
        cache_description("test_file_id", "image description", 7200)

        mock_cache.assert_called_once_with(
            "image_description", "test_file_id", "image description", 7200
        )


def test_get_cache_history_success():
    from api.index import get_cache_history
    import json
    from datetime import datetime, timedelta

    with patch("api.index.datetime") as mock_datetime:
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
        mock_redis.get.assert_called_once_with(expected_timestamp + "test_hash")


def test_get_cache_history_not_found():
    from api.index import get_cache_history

    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    result = get_cache_history(1, "test_hash", mock_redis)

    assert result is None


def test_get_cache_history_invalid_data():
    from api.index import get_cache_history
    import json

    mock_redis = MagicMock()
    test_data = {"data": "test"}  # Missing timestamp
    mock_redis.get.return_value = json.dumps(test_data)

    result = get_cache_history(1, "test_hash", mock_redis)

    assert result is None


def test_get_cached_bcra_variables_success():
    from api.index import get_cached_bcra_variables
    import json

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}
        payload = {"data": test_data, "fetched_at": "2025-09-20T00:00:00+00:00"}
        mock_redis.get.return_value = json.dumps(payload)

        result = get_cached_bcra_variables()

        assert result == test_data
        mock_redis.get.assert_called_once_with("bcra_variables")


def test_get_cached_bcra_variables_not_found():
    from api.index import get_cached_bcra_variables

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        mock_redis.get.return_value = None

        result = get_cached_bcra_variables()

        assert result is None
        mock_redis.get.assert_called_once_with("bcra_variables")


def test_get_cached_bcra_variables_exception():
    from api.index import get_cached_bcra_variables

    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")

        result = get_cached_bcra_variables()

        assert result is None


def test_cache_bcra_variables_success():
    from api.index import cache_bcra_variables
    import json

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}

        cache_bcra_variables(test_data, 600)

        assert mock_redis.setex.call_count == 1
        args, _ = mock_redis.setex.call_args
        assert args[0] == "bcra_variables"
        assert args[1] == 600
        payload = json.loads(args[2])
        assert payload["data"] == test_data
        assert "fetched_at" in payload

        keys_written = [call[0][0] for call in mock_redis.set.call_args_list]
        assert "bcra_variables:last_success" in keys_written
        for args, _ in mock_redis.set.call_args_list:
            if args[0] == "bcra_variables:last_success":
                last_success_payload = json.loads(args[1])
                assert last_success_payload["data"] == test_data
                assert "fetched_at" in last_success_payload
                break


def test_cache_bcra_variables_default_ttl():
    from api.index import cache_bcra_variables
    import json

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        test_data = {"base_monetaria": "1000000"}

        cache_bcra_variables(test_data)

        args, _ = mock_redis.setex.call_args
        assert args[0] == "bcra_variables"
        assert args[1] == 300
        payload = json.loads(args[2])
        assert payload["data"] == test_data
        assert "fetched_at" in payload


def test_cache_bcra_variables_exception():
    from api.index import cache_bcra_variables

    with patch("api.index.config_redis") as mock_config_redis:
        mock_config_redis.side_effect = Exception("Redis error")
        test_data = {"base_monetaria": "1000000"}

        # Should not raise exception, just print error
        cache_bcra_variables(test_data)


def test_get_or_refresh_bcra_variables_returns_stale_on_failure():
    from api import index as index_module
    from api.index import cache_bcra_variables, get_or_refresh_bcra_variables

    test_data = {
        "Tipo de cambio mayorista": {"value": "1000", "date": "01/09/2025"}
    }

    with patch("api.index.config_redis") as mock_config_redis:
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis
        cache_bcra_variables(test_data, ttl=1)

    bcra_service._bcra_local_cache["expires_at"] = time.time() - 10
    bcra_service._bcra_local_cache["stale_until"] = time.time() + 60

    with patch("api.index.bcra_fetch_latest_variables") as mock_fetch, patch(
        "api.index.config_redis"
    ) as mock_config_redis:
        mock_fetch.return_value = None
        mock_config_redis.side_effect = Exception("Redis down")

        result = get_or_refresh_bcra_variables()

    assert result is not None
    assert result.get("_meta", {}).get("stale") is True
    assert "Tipo de cambio mayorista" in result


def test_get_currency_band_limits_reuses_stale_after_failure():
    from api import index as index_module
    from api.index import get_currency_band_limits

    band_data = {"lower": 950.0, "upper": 1200.0, "date": "22/09/2025"}

    with patch("api.index.fetch_currency_band_limits") as mock_fetch, patch(
        "api.index.config_redis"
    ) as mock_config_redis:
        mock_fetch.return_value = band_data
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        result = get_currency_band_limits()

    assert result == band_data

    bcra_service._currency_band_local_cache["expires_at"] = time.time() - 5
    bcra_service._currency_band_local_cache["stale_until"] = time.time() + 300

    with patch("api.index.fetch_currency_band_limits") as mock_fetch, patch(
        "api.index.config_redis"
    ) as mock_config_redis:
        mock_fetch.return_value = None
        mock_config_redis.side_effect = Exception("Redis offline")

        fallback_result = get_currency_band_limits()
        mock_fetch.assert_called_once()

    assert fallback_result == band_data

    with patch("api.index.fetch_currency_band_limits") as mock_fetch_again, patch(
        "api.index.config_redis"
    ) as mock_config_redis:
        mock_fetch_again.return_value = None
        mock_config_redis.side_effect = Exception("Redis offline")

        second_result = get_currency_band_limits()
        mock_fetch_again.assert_not_called()

    assert second_result == band_data


def test_get_currency_band_limits_discards_future_cache():
    from api.index import get_currency_band_limits
    from api.services import bcra as bcra_service
    from datetime import datetime, timedelta

    today = datetime.now(bcra_service.BA_TZ).date()
    future = today + timedelta(days=1)

    future_cache = {
        "lower": 975.0,
        "upper": 1480.0,
        "date": future.strftime("%d/%m/%y"),
        "date_iso": future.isoformat(),
    }

    with patch(
        "api.services.bcra._get_cached_currency_band_entry",
        return_value=(future_cache, {"is_fresh": True, "fetched_at": None}),
    ) as mock_cached, patch("api.index.fetch_currency_band_limits") as mock_fetch, patch(
        "api.index.config_redis"
    ) as mock_config_redis:
        mock_fetch.return_value = {
            "lower": 940.0,
            "upper": 1475.0,
            "date": today.strftime("%d/%m/%y"),
            "date_iso": today.isoformat(),
        }
        mock_config_redis.return_value = MagicMock()

        result = get_currency_band_limits()

    mock_cached.assert_called_once()
    mock_fetch.assert_called_once()
    assert result == mock_fetch.return_value


def test_handle_transcribe_with_message_no_reply():
    from api.index import handle_transcribe_with_message

    message = {"message_id": 1, "chat": {"id": 123}, "text": "/transcribe"}

    result = handle_transcribe_with_message(message)
    assert result == "respondeme un audio, imagen o sticker y te digo qué carajo hay ahí"


def test_handle_transcribe_with_message_voice_cached():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_transcription") as mock_cached:
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

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.measure_audio_duration_seconds", return_value=1.0
    ) as mock_measure, patch(
        "api.index._transcribe_audio_groq_result"
    ) as mock_transcribe:

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

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download:

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

    with patch("api.index.get_cached_transcription") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch(
        "api.index.measure_audio_duration_seconds", return_value=1.0
    ) as mock_measure, patch(
        "api.index._transcribe_audio_groq_result"
    ) as mock_transcribe:

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

    with patch("api.index.get_cached_description") as mock_cached:
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

    with patch("api.index.get_cached_description") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch("api.index.resize_image_if_needed") as mock_resize, patch(
        "api.index._describe_image_groq_result"
    ) as mock_describe:

        mock_cached.return_value = None
        mock_download.return_value = b"image data"
        mock_resize.return_value = b"resized image data"
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
        mock_resize.assert_called_once_with(b"image data")
        mock_describe.assert_called_once_with(
            b"resized image data",
            "Describe what you see in this image in detail.",
            "photo123",
        )


def test_handle_transcribe_with_message_sticker_success():
    from api.index import handle_transcribe_with_message

    with patch("api.index.get_cached_description") as mock_cached, patch(
        "api.index.download_telegram_file"
    ) as mock_download, patch("api.index.resize_image_if_needed") as mock_resize, patch(
        "api.index._describe_image_groq_result"
    ) as mock_describe:

        mock_cached.return_value = None
        mock_download.return_value = b"sticker data"
        mock_resize.return_value = b"resized sticker data"
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


def test_handle_transcribe_with_message_no_media():
    from api.index import handle_transcribe_with_message

    message = {
        "message_id": 1,
        "chat": {"id": 123},
        "text": "/transcribe",
        "reply_to_message": {"text": "just text"},
    }

    result = handle_transcribe_with_message(message)
    assert result == "ese mensaje no tiene audio, imagen ni sticker para laburar"


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


def test_format_bcra_variables_empty():
    from api.index import format_bcra_variables

    result = format_bcra_variables({})
    assert result == "No se pudieron obtener las variables del BCRA"

    from typing import cast, Dict, Any

    result = format_bcra_variables(cast(Dict[str, Any], None))
    assert result == "No se pudieron obtener las variables del BCRA"


def test_format_bcra_variables_with_data():
    from api.index import format_bcra_variables
    from unittest.mock import patch

    variables = {
        "base_monetaria_total": {"value": "5.000.000,50", "date": "15/01/2025"},
        "inflacion_mensual": {"value": "5,2", "date": "15/01/2025"},
        "inflacion_interanual": {"value": "150,5", "date": "15/01/2025"},
        "inflacion_esperada": {"value": "3,1", "date": "15/01/2025"},
        "tasa_tamar": {"value": "45,0", "date": "15/01/2025"},
        "tasa_badlar": {"value": "40,5", "date": "15/01/2025"},
        "dolar_minorista_compra": {"value": "1.200,50", "date": "15/01/2025"},
        "dolar_minorista_venta": {"value": "1.250,75", "date": "15/01/2025"},
        "dolar_mayorista": {"value": "1.180,25", "date": "15/01/2025"},
        "uva": {"value": "500,75", "date": "15/01/2025"},
        "cer": {"value": "0,45", "date": "15/01/2025"},
        "reservas_int": {"value": "25.000", "date": "15/01/2025"},
    }

    with patch("api.index.get_latest_itcrm_value_and_date") as mock_itcrm, patch(
        "api.index.get_currency_band_limits"
    ) as mock_bands:
        mock_itcrm.return_value = (123.45, "01/02/25")
        mock_bands.return_value = None
        result = format_bcra_variables(variables)
    assert "📊 Variables principales BCRA" in result
    assert "15/01/25" in result  # Date should be formatted
    assert "TCRM" in result  # Should include current TCRM line
    assert "01/02/25" in result  # Should include TCRM date from sheet


def test_format_bcra_variables_includes_currency_bands():
    from api.index import format_bcra_variables, fmt_num
    from unittest.mock import patch

    variables = {
        "Tipo de cambio mayorista de referencia": {
            "value": "1.100,00",
            "date": "15/09/2025",
        }
    }

    with patch("api.index.get_currency_band_limits") as mock_bands, patch(
        "api.index.get_latest_itcrm_value_and_date"
    ) as mock_itcrm:
        mock_bands.return_value = {
            "lower": 950.12,
            "upper": 1460.34,
            "date": "15/09/25",
        }
        mock_itcrm.return_value = None
        result = format_bcra_variables(variables)

    expected_lower = fmt_num(950.12, 2)
    expected_upper = fmt_num(1460.34, 2)
    assert "📏 Bandas cambiarias" in result
    assert f"piso ${expected_lower}" in result
    assert f"techo ${expected_upper}" in result
    assert "15/09/25" in result


def test_get_country_risk_summary_fallback():
    from api.services import bcra as bcra_service
    import pytest

    original_cached = bcra_service._cached_requests_fn
    original_redis = bcra_service._redis_factory_fn
    original_admin = bcra_service._admin_reporter_fn
    original_history = bcra_service._cache_history_fn

    # Configure the service with a cached_requests implementation that fails,
    # forcing the direct HTTP fallback path.
    bcra_service.configure(
        cached_requests=lambda *args, **kwargs: None,
        redis_factory=lambda *args, **kwargs: MagicMock(),
    )

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "weightedSpreadBps": "685.21",
                "deltas": {"oneDay": "-12.3"},
                "valuationDate": "2025-10-29T15:34:00Z",
            }

    mock_get = None
    try:
        with patch(
            "api.services.bcra.requests.get", return_value=DummyResponse()
        ) as patched_get:
            mock_get = patched_get
            summary = bcra_service.get_country_risk_summary()
    finally:
        bcra_service._cached_requests_fn = original_cached
        bcra_service._redis_factory_fn = original_redis
        bcra_service._admin_reporter_fn = original_admin
        bcra_service._cache_history_fn = original_history

    assert summary is not None
    assert summary["value_bps"] == pytest.approx(685.21)
    assert summary["delta_one_day"] == pytest.approx(-12.3)
    assert summary["valuation_label"] == "29/10 12:34"
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs.get("verify") is False


def test_parse_currency_band_rows_skips_future_dates():
    from api.index import _parse_currency_band_rows
    from datetime import date
    import pytest

    rows = [
        ["18/9/2025", "900,00", "1.400,00"],
        ["19/9/2025", "948,44", "1.475,32"],
        ["30/9/2025", "944,96", "1.480,72"],
    ]

    parsed = _parse_currency_band_rows(rows, today=date(2025, 9, 19))

    assert parsed is not None
    assert parsed["date"] == "19/09/25"
    assert parsed["lower"] == 948.44
    assert parsed["upper"] == 1475.32
    assert pytest.approx(parsed["lower_change_pct"], rel=1e-6) == ((948.44 - 900.0) / 900.0) * 100
    assert pytest.approx(parsed["upper_change_pct"], rel=1e-6) == ((1475.32 - 1400.0) / 1400.0) * 100


def test_fetch_currency_band_limits_uses_principales_variables():
    from api.index import fetch_currency_band_limits, BA_TZ

    today = datetime.now(BA_TZ).date()
    previous = today - timedelta(days=1)
    future = today + timedelta(days=1)

    lower_payload = {
        "status": 200,
        "results": [
            {
                "detalle": [
                    {"fecha": future.isoformat(), "valor": 950.0},
                    {"fecha": today.isoformat(), "valor": 944.32},
                    {"fecha": previous.isoformat(), "valor": 930.0},
                ]
            }
        ],
    }

    upper_payload = {
        "status": 200,
        "results": [
            {
                "detalle": [
                    {"fecha": future.isoformat(), "valor": 1490.0},
                    {"fecha": today.isoformat(), "valor": 1481.70},
                    {"fecha": previous.isoformat(), "valor": 1470.0},
                ]
            }
        ],
    }

    with patch("api.index.bcra_list_variables") as mock_list, patch(
        "api.index.bcra_api_get"
    ) as mock_api:
        mock_list.return_value = [
            {
                "idVariable": 1188,
                "descripcion": "Régimen de bandas cambiarias  (Límite superior $1.400 + 1% mensual)",
            },
            {
                "idVariable": 1187,
                "descripcion": "Régimen de bandas cambiarias (Límite inferior $1.000 - 1% mensual)",
            },
        ]

        def side_effect(path, params=None, ttl=None):
            if "1187" in path:
                return lower_payload
            if "1188" in path:
                return upper_payload
            return None

        mock_api.side_effect = side_effect

        result = fetch_currency_band_limits()

    assert result is not None
    assert result["date"] == today.strftime("%d/%m/%y")
    assert result["lower"] == pytest.approx(944.32)
    assert result["upper"] == pytest.approx(1481.70)
    assert result["lower_change_pct"] == pytest.approx(((944.32 - 930.0) / 930.0) * 100)
    assert result["upper_change_pct"] == pytest.approx(((1481.70 - 1470.0) / 1470.0) * 100)


def test_handle_bcra_variables_cached():
    from api.index import handle_bcra_variables

    with patch("api.index.get_or_refresh_bcra_variables") as mock_get, patch(
        "api.index.format_bcra_variables"
    ) as mock_format:

        mock_get.return_value = {"test": "data"}
        mock_format.return_value = "formatted variables"

        result = handle_bcra_variables()
        assert result == "formatted variables"
        mock_get.assert_called_once()
        mock_format.assert_called_once_with({"test": "data"})


def test_handle_bcra_variables_scrape_fresh():
    from api.index import handle_bcra_variables

    with patch("api.index.get_or_refresh_bcra_variables") as mock_get, patch(
        "api.index.format_bcra_variables"
    ) as mock_format:

        mock_get.return_value = {"scraped": "data"}
        mock_format.return_value = "formatted scraped data"

        result = handle_bcra_variables()
        assert result == "formatted scraped data"
        mock_get.assert_called_once()
        mock_format.assert_called_once_with({"scraped": "data"})


def test_handle_bcra_variables_no_data():
    from api.index import handle_bcra_variables

    with patch("api.index.get_or_refresh_bcra_variables") as mock_get:
        mock_get.return_value = None

        result = handle_bcra_variables()
        assert (
            result
            == "No pude obtener las variables del BCRA en este momento, probá más tarde"
        )


def test_handle_bcra_variables_exception():
    from api.index import handle_bcra_variables

    with patch("api.index.get_or_refresh_bcra_variables") as mock_get:
        mock_get.side_effect = Exception("Cache error")

        result = handle_bcra_variables()
        assert result == "error al obtener las variables del BCRA"


def test_calculate_tcrm_100_success():
    from api.index import calculate_tcrm_100
    from unittest.mock import MagicMock

    with patch(
        "api.index.get_latest_itcrm_value_and_date"
    ) as mock_itcrm_details, patch("api.index.config_redis") as mock_cfg, patch(
        "api.index.redis_get_json"
    ) as mock_redis_get:
        mock_itcrm_details.return_value = (120.0, "01/01/24")

        # Simulate cached mayorista for 2024-01-01
        def redis_side_effect(_client, key):  # noqa: ARG001
            if key == "bcra_mayorista:2024-01-01":
                return {"value": 1000.0}
            return None

        mock_cfg.return_value = MagicMock()
        mock_redis_get.side_effect = redis_side_effect

        result = calculate_tcrm_100()
        assert result == 1000.0 * 100 / 120.0


def test_calculate_tcrm_100_caches_missing_mayorista():
    from api.index import calculate_tcrm_100

    cache = {}

    with patch(
        "api.index.get_latest_itcrm_value_and_date"
    ) as mock_itcrm_details, patch("api.index.config_redis") as mock_cfg, patch(
        "api.index.redis_get_json"
    ) as mock_get_json, patch("api.index.redis_setex_json") as mock_setex_json, patch(
        "api.index.bcra_get_value_for_date"
    ) as mock_bcra:
        mock_itcrm_details.return_value = (120.0, "01/01/24")
        redis_client = MagicMock()
        mock_cfg.return_value = redis_client

        def get_json_side_effect(_client, key):  # noqa: ARG001
            return cache.get(key)

        def setex_side_effect(_client, key, ttl, value):  # noqa: ARG001
            cache[key] = value
            return True

        mock_get_json.side_effect = get_json_side_effect
        mock_setex_json.side_effect = setex_side_effect
        mock_bcra.return_value = None

        result = calculate_tcrm_100()
        assert result is None

        sentinel_key = "bcra_mayorista:2024-01-01"
        assert sentinel_key in cache
        assert cache[sentinel_key]["missing"] is True
        mock_bcra.assert_called_once()

        mock_bcra.reset_mock()

        result_again = calculate_tcrm_100()
        assert result_again is None
        mock_bcra.assert_not_called()


def test_get_cached_tcrm_100_backfills_missing_history():
    import pytest
    from api.index import get_cached_tcrm_100
    from unittest.mock import MagicMock
    from typing import Any, List

    with patch("api.index.config_redis") as mock_cfg, patch(
        "api.index.redis_get_json"
    ) as mock_get_json, patch("api.index.redis_set_json") as mock_set_json, patch(
        "api.index.calculate_tcrm_100"
    ) as mock_calc, patch("api.index.get_latest_itcrm_value_and_date") as mock_latest:
        redis_client = MagicMock()
        redis_client.get.return_value = None
        mock_cfg.return_value = redis_client
        mock_set_json.return_value = True
        mock_latest.return_value = (150.0, "02/01/24")

        def redis_get_json_side_effect(_client, key):  # noqa: ARG001
            if key == "tcrm_100":
                return None
            if key == "latest_itcrm_details":
                return {"value": 150.0, "date": "2024-01-02"}
            if key == "bcra_mayorista:2024-01-02":
                return {"value": 95.0}
            return None

        mock_get_json.side_effect = redis_get_json_side_effect

        current_calls = 0
        history_calls: List[Any] = []

        def calc_side_effect(target_date=None):
            nonlocal current_calls
            if target_date is None:
                current_calls += 1
                return 200.0
            history_calls.append(target_date)
            return 100.0

        mock_calc.side_effect = calc_side_effect

        value, change = get_cached_tcrm_100()

        assert value == 200.0
        assert change is not None
        assert pytest.approx(change, rel=1e-9) == 100.0
        assert current_calls == 1
        assert len(history_calls) == 1
        history_dt = history_calls[0]
        history_key = history_dt.strftime("%Y-%m-%d-%H") + "tcrm_100"

        assert any(
            call.args[1] == history_key and call.args[2]["data"] == 100.0
            for call in mock_set_json.call_args_list
        )


def test_get_cached_tcrm_100_skips_fetch_when_mayorista_missing():
    from api.index import get_cached_tcrm_100

    cache = {
        "bcra_mayorista:2024-01-02": {"missing": True, "timestamp": 1234567890}
    }

    with patch("api.index.config_redis") as mock_cfg, patch(
        "api.index.redis_get_json"
    ) as mock_get_json, patch("api.index.get_cache_history") as mock_history, patch(
        "api.index.bcra_get_value_for_date"
    ) as mock_bcra, patch("api.index.calculate_tcrm_100") as mock_calc:
        redis_client = MagicMock()
        redis_client.get.return_value = None
        mock_cfg.return_value = redis_client
        mock_history.return_value = None

        def get_json_side_effect(_client, key):  # noqa: ARG001
            if key == "tcrm_100":
                return None
            if key == "latest_itcrm_details":
                return {"date": "2024-01-02"}
            return cache.get(key)

        mock_get_json.side_effect = get_json_side_effect

        value, change = get_cached_tcrm_100()

        assert value is None
        assert change is None
        mock_bcra.assert_not_called()
        mock_calc.assert_called_once_with(target_date=ANY)


def test_sort_dollar_rates_with_tcrm():
    from api.index import sort_dollar_rates

    dollar_rates = {
        "data": {
            "mayorista": {"price": 90.0, "variation": 0.5},
            "oficial": {"price": 100.0, "variation": 0.5},
            "tarjeta": {"price": 150.0, "variation": 0.5},
            "mep": {
                "al30": {
                    "24hs": {"price": 120.0, "variation": 0.5},
                    "ci": {"price": 120.0, "variation": 0.5},
                }
            },
            "ccl": {
                "al30": {
                    "24hs": {"price": 130.0, "variation": 0.5},
                    "ci": {"price": 130.0, "variation": 0.5},
                }
            },
            "blue": {"ask": 140.0, "variation": 0.5},
            "cripto": {
                "ccb": {"ask": 145.0, "variation": 0.5},
                "usdc": {"ask": 146.0, "variation": 0.5},
                "usdt": {"ask": 147.0, "variation": 0.5},
            },
        }
    }

    result = sort_dollar_rates(dollar_rates, 160.0, 0.5)

    assert any(
        r["name"] == "TCRM 100" and r["price"] == 160.0 and r["history"] == 0.5
        for r in result
    )


def test_format_dollar_rates_includes_currency_band_limits():
    from api.index import format_dollar_rates, fmt_num

    dollar_rates = [
        {"name": "Oficial", "price": 1000.5, "history": 0.5},
    ]

    band_limits = {
        "lower": 950.12,
        "upper": 1460.34,
        "date": "15/09/25",
        "lower_change_pct": 0.5,
        "upper_change_pct": -0.25,
    }

    result = format_dollar_rates(dollar_rates, 24, band_limits)

    lines = result.splitlines()
    assert lines[0] == "Banda piso: 950.12 (+0.5% 24hs)"
    assert lines[1] == "Oficial: 1000.5 (+0.5% 24hs)"
    assert lines[2] == "Banda techo: 1460.34 (-0.25% 24hs)"


def test_download_telegram_file_success():
    """Test download_telegram_file with successful download"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"

        # Mock file info response
        mock_file_info = MagicMock()
        mock_file_info.json.return_value = {
            "ok": True,
            "result": {"file_path": "photos/file_123.jpg"},
        }
        mock_file_info.raise_for_status.return_value = None

        # Mock file download response
        mock_file_download = MagicMock()
        mock_file_download.content = b"fake image data"
        mock_file_download.raise_for_status.return_value = None

        # Configure side effect for two different calls
        mock_get.side_effect = [mock_file_info, mock_file_download]

        result = download_telegram_file("test_file_id")

        assert result == b"fake image data"
        assert mock_get.call_count == 2

        # Verify file info call
        info_call = mock_get.call_args_list[0]
        assert "getFile" in info_call[0][0]
        assert info_call[1]["params"]["file_id"] == "test_file_id"

        # Verify file download call
        download_call = mock_get.call_args_list[1]
        assert "photos/file_123.jpg" in download_call[0][0]


def test_download_telegram_file_api_error():
    """Test download_telegram_file with API error"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"

        # Mock failed API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "File not found"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_telegram_file("invalid_file_id")

        assert result is None


def test_download_telegram_file_network_error():
    """Test download_telegram_file with network error"""
    from api.index import download_telegram_file

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.requests.get"
    ) as mock_get:
        mock_env.return_value = "test_token"
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = download_telegram_file("test_file_id")

        assert result is None


def test_encode_image_to_base64_success():
    """Test encode_image_to_base64 with valid image data"""
    from api.index import encode_image_to_base64
    import base64

    test_data = b"fake image bytes"
    expected = base64.b64encode(test_data).decode("utf-8")

    result = encode_image_to_base64(test_data)

    assert result == expected
    assert isinstance(result, str)


def test_encode_image_to_base64_empty():
    """Test encode_image_to_base64 with empty data"""
    from api.index import encode_image_to_base64

    result = encode_image_to_base64(b"")

    assert result == ""


def test_resize_image_if_needed_no_resize():
    """Test resize_image_if_needed when image is already small enough"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        # Mock small image
        mock_image = MagicMock()
        mock_image.size = (200, 150)  # Smaller than max_size of 512
        mock_image_module.open.return_value = mock_image

        test_data = b"small image data"
        result = resize_image_if_needed(test_data, max_size=512)

        assert result == test_data  # Should return original data unchanged


def test_resize_image_if_needed_with_resize():
    """Test resize_image_if_needed when image needs resizing"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module, patch(
        "api.index.io.BytesIO"
    ) as mock_bytesio:
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
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module, patch(
        "api.index.io.BytesIO"
    ) as mock_bytesio:
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
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        mock_image_module.open.side_effect = ImportError("PIL not available")

        test_data = b"image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on ImportError


def test_resize_image_if_needed_processing_error():
    """Test resize_image_if_needed with image processing error"""
    from api.index import resize_image_if_needed

    with patch("api.index.Image") as mock_image_module:
        mock_image_module.open.side_effect = Exception("Invalid image format")

        test_data = b"corrupted image data"
        result = resize_image_if_needed(test_data)

        assert result == test_data  # Should return original data on error


def test_describe_image_groq_success():
    """Test describe_image_groq with successful API response"""
    from api.index import describe_image_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.OpenAI"
    ) as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        fake_response = MagicMock()
        fake_response.output_text = "A beautiful landscape with mountains"

        fake_client = MagicMock()
        fake_client.responses.create.return_value = fake_response
        mock_openai.return_value = fake_client

        result = describe_image_groq(b"image_data")

        assert result == "A beautiful landscape with mountains"
        fake_client.responses.create.assert_called_once()


def test_describe_image_groq_api_error():
    """Test describe_image_groq with API error"""
    from api.index import describe_image_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.OpenAI"
    ) as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        fake_client = MagicMock()
        fake_client.responses.create.side_effect = Exception("API error")
        mock_openai.return_value = fake_client

        result = describe_image_groq(b"image_data")

        assert result is None


def test_describe_image_groq_skips_call_when_local_rate_limit_hits():
    from api.index import describe_image_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index._reserve_groq_rate_limit", return_value=None
    ), patch("api.index.OpenAI") as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        result = describe_image_groq(b"image_data")

        assert result is None
        mock_openai.assert_not_called()


def test_transcribe_audio_groq_success():
    """Test transcribe_audio_groq with successful transcription"""
    from api.index import transcribe_audio_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.OpenAI"
    ) as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        fake_response = MagicMock()
        fake_response.text = "Hello, this is a test audio transcription"

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_response
        mock_openai.return_value = fake_client

        result = transcribe_audio_groq(b"audio_data")

        assert result == "Hello, this is a test audio transcription"
        fake_client.audio.transcriptions.create.assert_called_once()


def test_transcribe_audio_groq_network_error():
    """Test transcribe_audio_groq with network error"""
    from api.index import transcribe_audio_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index.OpenAI"
    ) as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.side_effect = Exception("Network error")
        mock_openai.return_value = fake_client

        result = transcribe_audio_groq(b"audio_data")

        assert result is None


def test_transcribe_audio_groq_skips_call_when_local_rate_limit_hits():
    from api.index import transcribe_audio_groq

    with patch("api.index.environ.get") as mock_env, patch(
        "api.index._reserve_groq_rate_limit", return_value=None
    ), patch("api.index.OpenAI") as mock_openai:
        mock_env.side_effect = lambda key, default=None: {
            "GROQ_API_KEY": "test_api_key",
        }.get(key, default)

        result = transcribe_audio_groq(b"audio_data")

        assert result is None
        mock_openai.assert_not_called()


# Tests for web search functionality
