from tests.support import *


def _bcra_service_patches():
    """Patch BCRA service to bypass _require_configured and use mock redis."""
    mock_redis = MagicMock()
    return (
        patch("api.services.bcra._require_configured", return_value=None),
        patch("api.services.bcra._config_redis", return_value=mock_redis),
        mock_redis,
    )


def test_get_cached_bcra_variables_success():
    get_cached_bcra_variables = index.app_runtime.bcra.get_cached_variables
    import json

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}
        payload = {"data": test_data, "fetched_at": "2025-09-20T00:00:00+00:00"}
        mock_redis.get.return_value = json.dumps(payload)

        result = get_cached_bcra_variables()

        assert result == test_data


def test_get_cached_bcra_variables_not_found():
    get_cached_bcra_variables = index.app_runtime.bcra.get_cached_variables

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        mock_redis.get.return_value = None

        result = get_cached_bcra_variables()

        assert result is None


def test_get_cached_bcra_variables_exception():
    get_cached_bcra_variables = index.app_runtime.bcra.get_cached_variables

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        mock_redis.get.side_effect = Exception("Redis error")

        result = get_cached_bcra_variables()

        assert result is None


def test_cache_bcra_variables_success():
    cache_bcra_variables = index.app_runtime.bcra.cache_variables
    import json
    from api.services.maintenance import last_success_ttl

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        test_data = {"base_monetaria": "1000000", "inflacion_mensual": "5.2"}

        cache_bcra_variables(test_data, 600)

        assert mock_redis.set.call_count == 2
        first_args, first_kwargs = mock_redis.set.call_args_list[0]
        assert first_args[0] == "bcra_variables"
        assert first_kwargs["ex"] == 600
        payload = json.loads(first_args[1])
        assert payload["data"] == test_data
        assert "fetched_at" in payload

        last_success_args, last_success_kwargs = mock_redis.set.call_args_list[1]
        assert last_success_args[0] == "bcra_variables:last_success"
        assert last_success_kwargs["ex"] == last_success_ttl(600, 6 * 3600)
        last_success_payload = json.loads(last_success_args[1])
        assert last_success_payload["data"] == test_data
        assert "fetched_at" in last_success_payload


def test_cache_bcra_variables_default_ttl():
    cache_bcra_variables = index.app_runtime.bcra.cache_variables
    import json

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        test_data = {"base_monetaria": "1000000"}

        cache_bcra_variables(test_data)

        first_args, first_kwargs = mock_redis.set.call_args_list[0]
        assert first_args[0] == "bcra_variables"
        assert first_kwargs["ex"] == 300
        payload = json.loads(first_args[1])
        assert payload["data"] == test_data
        assert "fetched_at" in payload


def test_cache_bcra_variables_exception():
    cache_bcra_variables = index.app_runtime.bcra.cache_variables

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        mock_redis.set.side_effect = Exception("Redis error")
        test_data = {"base_monetaria": "1000000"}

        cache_bcra_variables(test_data)


def test_get_or_refresh_bcra_variables_returns_stale_on_failure():
    cache_bcra_variables = index.app_runtime.bcra.cache_variables
    get_or_refresh_bcra_variables = index.app_runtime.bcra.get_or_refresh_variables

    test_data = {"Tipo de cambio mayorista": {"value": "1000", "date": "01/09/2025"}}

    p_req, p_red, mock_redis = _bcra_service_patches()
    with p_req, p_red:
        cache_bcra_variables(test_data, ttl=1)

    bcra_service._bcra_local_cache["expires_at"] = time.time() - 10
    bcra_service._bcra_local_cache["stale_until"] = time.time() + 60

    with (
        patch("api.index.app_runtime.bcra.fetch_latest_variables") as mock_fetch,
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
    ):
        mock_fetch.return_value = None
        mock_config_redis.side_effect = Exception("Redis down")

        result = get_or_refresh_bcra_variables()

    assert result is not None
    assert result.get("_meta", {}).get("stale") is True
    assert "Tipo de cambio mayorista" in result


def test_get_currency_band_limits_reuses_stale_after_failure():
    get_currency_band_limits = index.app_runtime.bcra.get_currency_band_limits

    band_data = {"lower": 950.0, "upper": 1200.0, "date": "22/09/2025"}

    with (
        patch("api.index.app_runtime.bcra.fetch_currency_band_limits") as mock_fetch,
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
    ):
        mock_fetch.return_value = band_data
        mock_redis = MagicMock()
        mock_config_redis.return_value = mock_redis

        result = get_currency_band_limits()

    assert result == band_data

    bcra_service._currency_band_local_cache["expires_at"] = time.time() - 5
    bcra_service._currency_band_local_cache["stale_until"] = time.time() + 300

    with (
        patch("api.index.app_runtime.bcra.fetch_currency_band_limits") as mock_fetch,
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
    ):
        mock_fetch.return_value = None
        mock_config_redis.side_effect = Exception("Redis offline")

        fallback_result = get_currency_band_limits()
        mock_fetch.assert_called_once()

    assert fallback_result == band_data

    with (
        patch("api.index.app_runtime.bcra.fetch_currency_band_limits") as mock_fetch_again,
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
    ):
        mock_fetch_again.return_value = None
        mock_config_redis.side_effect = Exception("Redis offline")

        second_result = get_currency_band_limits()
        mock_fetch_again.assert_not_called()

    assert second_result == band_data


def test_get_currency_band_limits_discards_future_cache():
    get_currency_band_limits = index.app_runtime.bcra.get_currency_band_limits
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

    with (
        patch(
            "api.services.bcra._get_cached_currency_band_entry",
            return_value=(future_cache, {"is_fresh": True, "fetched_at": None}),
        ) as mock_cached,
        patch("api.index.app_runtime.bcra.fetch_currency_band_limits") as mock_fetch,
        patch("api.index.app_runtime.config.redis") as mock_config_redis,
    ):
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


def test_format_bcra_variables_empty():
    format_bcra_variables = index.app_runtime.bcra.format_variables

    result = format_bcra_variables({})
    assert result == "No se pudieron obtener las variables del BCRA"

    from typing import cast, Dict, Any

    result = format_bcra_variables(cast(Dict[str, Any], None))
    assert result == "No se pudieron obtener las variables del BCRA"


def test_format_bcra_variables_with_data():
    format_bcra_variables = index.app_runtime.bcra.format_variables
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

    with (
        patch("api.index.app_runtime.bcra.get_latest_itcrm_value_and_date") as mock_itcrm,
        patch("api.index.app_runtime.bcra.get_currency_band_limits") as mock_bands,
        patch("api.index.app_runtime.bcra.get_country_risk_summary", return_value=None),
    ):
        mock_itcrm.return_value = (123.45, "01/02/25")
        mock_bands.return_value = None
        result = format_bcra_variables(variables)
    assert "variables principales bcra" in result
    assert "15/01/25" in result  # Date should be formatted
    assert "tcrm" in result  # Should include current TCRM line
    assert "01/02/25" in result  # Should include TCRM date from sheet


def test_format_bcra_variables_includes_currency_bands():
    from api.index import fmt_num
    format_bcra_variables = index.app_runtime.bcra.format_variables
    from unittest.mock import patch

    variables = {
        "Tipo de cambio mayorista de referencia": {
            "value": "1.100,00",
            "date": "15/09/2025",
        }
    }

    with (
        patch("api.index.app_runtime.bcra.get_currency_band_limits") as mock_bands,
        patch("api.index.app_runtime.bcra.get_latest_itcrm_value_and_date") as mock_itcrm,
        patch("api.index.app_runtime.bcra.get_country_risk_summary", return_value=None),
    ):
        mock_bands.return_value = {
            "lower": 950.12,
            "upper": 1460.34,
            "date": "15/09/25",
        }
        mock_itcrm.return_value = None
        result = format_bcra_variables(variables)

    expected_lower = fmt_num(950.12, 2)
    expected_upper = fmt_num(1460.34, 2)
    assert "bandas cambiarias" in result
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
    assert (
        pytest.approx(parsed["lower_change_pct"], rel=1e-6)
        == ((948.44 - 900.0) / 900.0) * 100
    )
    assert (
        pytest.approx(parsed["upper_change_pct"], rel=1e-6)
        == ((1475.32 - 1400.0) / 1400.0) * 100
    )


def test_fetch_currency_band_limits_uses_principales_variables():
    from api.index import BA_TZ
    fetch_currency_band_limits = index.app_runtime.bcra.fetch_currency_band_limits

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

    with (
        patch("api.index.app_runtime.bcra.list_variables") as mock_list,
        patch("api.index.app_runtime.bcra.api_get") as mock_api,
    ):
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
    assert result["upper_change_pct"] == pytest.approx(
        ((1481.70 - 1470.0) / 1470.0) * 100
    )


def test_handle_bcra_variables_cached():
    handle_bcra_variables = index.app_runtime.dollar.handle_bcra_variables

    with (
        patch("api.index.app_runtime.dollar.get_bcra_variables") as mock_get,
        patch("api.index.app_runtime.dollar.format_bcra_variables") as mock_format,
    ):
        mock_get.return_value = {"test": "data"}
        mock_format.return_value = "formatted variables"

        result = handle_bcra_variables()
        assert result == "formatted variables"
        mock_get.assert_called_once()
        mock_format.assert_called_once_with({"test": "data"})


def test_handle_bcra_variables_scrape_fresh():
    handle_bcra_variables = index.app_runtime.dollar.handle_bcra_variables

    with (
        patch("api.index.app_runtime.dollar.get_bcra_variables") as mock_get,
        patch("api.index.app_runtime.dollar.format_bcra_variables") as mock_format,
    ):
        mock_get.return_value = {"scraped": "data"}
        mock_format.return_value = "formatted scraped data"

        result = handle_bcra_variables()
        assert result == "formatted scraped data"
        mock_get.assert_called_once()
        mock_format.assert_called_once_with({"scraped": "data"})


def test_handle_bcra_variables_no_data():
    handle_bcra_variables = index.app_runtime.dollar.handle_bcra_variables

    with patch("api.index.app_runtime.dollar.get_bcra_variables") as mock_get:
        mock_get.return_value = None

        result = handle_bcra_variables()
        assert (
            result
            == "No pude obtener las variables del BCRA en este momento, probá más tarde"
        )


def test_handle_bcra_variables_exception():
    handle_bcra_variables = index.app_runtime.dollar.handle_bcra_variables

    with patch("api.index.app_runtime.dollar.get_bcra_variables") as mock_get:
        mock_get.side_effect = Exception("Cache error")

        result = handle_bcra_variables()
        assert result == "error al obtener las variables del BCRA"


def test_calculate_tcrm_100_success():
    calculate_tcrm_100 = index.app_runtime.bcra.calculate_tcrm_100
    from unittest.mock import MagicMock

    with (
        patch("api.index.app_runtime.bcra.get_latest_itcrm_value_and_date") as mock_itcrm_details,
        patch("api.index.app_runtime.config.redis") as mock_cfg,
        patch("api.services.bcra.redis_get_json") as mock_redis_get,
    ):
        mock_itcrm_details.return_value = (120.0, "01/01/24")

        # Simulate cached mayorista for 2024-01-01
        def redis_side_effect(_client, key):
            if key == "bcra_mayorista:2024-01-01":
                return {"value": 1000.0}
            return None

        mock_cfg.return_value = MagicMock()
        mock_redis_get.side_effect = redis_side_effect

        result = calculate_tcrm_100()
        assert result == 1000.0 * 100 / 120.0


def test_calculate_tcrm_100_caches_missing_mayorista():
    calculate_tcrm_100 = index.app_runtime.bcra.calculate_tcrm_100

    cache = {}

    with (
        patch("api.index.app_runtime.bcra.get_latest_itcrm_value_and_date") as mock_itcrm_details,
        patch("api.index.app_runtime.config.redis") as mock_cfg,
        patch("api.services.bcra.redis_get_json") as mock_get_json,
        patch("api.services.bcra.redis_setex_json") as mock_setex_json,
        patch("api.index.app_runtime.bcra.get_value_for_date") as mock_bcra,
    ):
        mock_itcrm_details.return_value = (120.0, "01/01/24")
        redis_client = MagicMock()
        mock_cfg.return_value = redis_client

        def get_json_side_effect(_client, key):
            return cache.get(key)

        def setex_side_effect(_client, key, ttl, value):
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
    get_cached_tcrm_100 = index.app_runtime.bcra.get_cached_tcrm_100
    from unittest.mock import MagicMock
    from typing import Any, List

    with (
        patch("api.index.app_runtime.config.redis") as mock_cfg,
        patch("api.services.bcra.redis_get_json") as mock_get_json,
        patch("api.services.bcra.redis_set_json") as mock_set_json,
        patch("api.index.app_runtime.bcra.calculate_tcrm_100") as mock_calc,
        patch("api.index.app_runtime.bcra.get_latest_itcrm_value_and_date") as mock_latest,
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = None
        mock_cfg.return_value = redis_client
        mock_set_json.return_value = True
        mock_latest.return_value = (150.0, "02/01/24")

        def redis_get_json_side_effect(_client, key):
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
    get_cached_tcrm_100 = index.app_runtime.bcra.get_cached_tcrm_100

    cache = {"bcra_mayorista:2024-01-02": {"missing": True, "timestamp": 1234567890}}

    with (
        patch("api.index.app_runtime.config.redis") as mock_cfg,
        patch("api.services.bcra.redis_get_json") as mock_get_json,
        patch("api.index.app_runtime.cache.get_history") as mock_history,
        patch("api.index.app_runtime.bcra.get_value_for_date") as mock_bcra,
        patch("api.index.app_runtime.bcra.calculate_tcrm_100") as mock_calc,
    ):
        redis_client = MagicMock()
        redis_client.get.return_value = None
        mock_cfg.return_value = redis_client
        mock_history.return_value = None

        def get_json_side_effect(_client, key):
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
    from api.markets.dollar import format_dollar_rates

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
