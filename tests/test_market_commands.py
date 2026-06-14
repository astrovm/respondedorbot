from tests.support import *


def test_get_rulo():
    dolar_data = {
        "oficial": {"price": 1440},
        "blue": {"ask": 1450, "bid": 1430},
        "mep": {
            "al30": {
                "24hs": {"price": 1462.32},
                "ci": {"price": 1459.73},
            },
            "gd30": {
                "24hs": {"price": 1460.0},
            },
        },
    }

    usd_usdt_data = {
        "buenbit": {"ask": 1.031, "totalAsk": 1.031},
        "xapo": {"ask": 1.002, "totalAsk": 1.004},
    }

    usdt_ars_data = {
        "buenbit": {"bid": 1458.44, "totalBid": 1458.44},
        "ripio": {"bid": 1448.75, "totalBid": 1448.75},
    }

    def fake_cached_requests(api_url, *_args, **_kwargs):
        if "dolar" in api_url:
            return {"data": dolar_data}
        if "USDT/USD" in api_url:
            return {"data": usd_usdt_data}
        if "USDT/ARS" in api_url:
            return {"data": usdt_ars_data}
        raise AssertionError(f"unexpected url {api_url}")

    with patch("api.index.app_runtime.cache.request", side_effect=fake_cached_requests):
        result = get_rulo()

    assert result.startswith("Rulos desde Oficial (precio oficial: 1.440 ARS/USD)")
    assert "Inversión base: 1.000 USD → 1.440.000 ARS" in result
    assert "- MEP (AL30 CI)" in result
    assert "  • Resultado: 1.000 USD → 1.459.730 ARS" in result
    assert "  • Ganancia: +19.730 ARS" in result
    assert "- Blue" in result
    assert "  • Ganancia: -10.000 ARS" in result
    assert "- USDT" in result
    assert "Tramos: USD→USDT BUENBIT, USDT→ARS BUENBIT" in result


def test_build_rulo_message_preserves_arbitrage_output():
    from api.markets.rulo import build_rulo_message

    dolar_data = {
        "oficial": {"price": 1440},
        "blue": {"ask": 1450, "bid": 1430},
        "mep": {"al30": {"ci": {"price": 1459.73}}},
    }
    usd_usdt_data = {
        "buenbit": {"ask": 1.031, "totalAsk": 1.031},
        "xapo": {"ask": 1.002, "totalAsk": 1.004},
    }
    usdt_ars_data = {
        "buenbit": {"bid": 1458.44, "totalBid": 1458.44},
        "ripio": {"bid": 1448.75, "totalBid": 1448.75},
    }

    result = build_rulo_message(dolar_data, usd_usdt_data, usdt_ars_data)

    assert result.startswith("Rulos desde Oficial (precio oficial: 1.440 ARS/USD)")
    assert "- MEP (AL30 CI)" in result
    assert "  • Ganancia: +19.730 ARS" in result
    assert "- Blue" in result
    assert "  • Ganancia: -10.000 ARS" in result
    assert "- USDT" in result
    assert "Tramos: USD→USDT BUENBIT, USDT→ARS BUENBIT" in result


def test_sort_dollar_rates_uses_hourly_history_and_tcrm():
    from api.markets.dollar_commands import sort_dollar_rates

    dollar_rates = {
        "data": {
            "mayorista": {"price": 1400, "variation": 1.0},
            "oficial": {"price": 1420, "variation": 2.0},
            "tarjeta": {"price": 1988, "variation": 3.0},
            "mep": {"al30": {"ci": {"price": 1450, "variation": 4.0}}},
            "ccl": {"al30": {"ci": {"price": 1460, "variation": 5.0}}},
            "blue": {"ask": 1430, "variation": 6.0},
            "cripto": {
                "ccb": {"ask": 1470, "variation": 7.0},
                "usdc": {"ask": 1480, "variation": 8.0},
                "usdt": {"ask": 1490, "variation": 9.0},
            },
        },
        "history": {
            "data": {
                "mayorista": {"price": 1300},
                "oficial": {"price": 1400},
                "tarjeta": {"price": 1900},
                "mep": {"al30": {"ci": {"price": 1400}}},
                "ccl": {"al30": {"ci": {"price": 1400}}},
                "blue": {"ask": 1400},
                "cripto": {
                    "ccb": {"ask": 1400},
                    "usdc": {"ask": 1400},
                    "usdt": {"ask": 1400},
                },
            }
        },
    }

    result = sort_dollar_rates(dollar_rates, 1410, -0.5, hours_ago=6)

    assert [entry["name"] for entry in result[:3]] == [
        "Mayorista",
        "TCRM 100",
        "Oficial",
    ]
    assert result[0] == {
        "name": "Mayorista",
        "price": 1400,
        "history": pytest.approx(7.6923076923),
    }
    assert result[1] == {"name": "TCRM 100", "price": 1410, "history": -0.5}


def test_get_weather():
    from api.index import get_weather

    # Create a fixed datetime for testing
    current_time = datetime(2024, 1, 1, 12, 0)  # Create naive datetime first

    with (
        patch("api.index.datetime") as mock_datetime,
        patch("api.index.app_runtime.cache.request") as mock_cached_requests,
    ):
        # Set up datetime mock to handle timezone
        class MockDatetime:
            @classmethod
            def now(cls, tz=None):
                if tz:
                    return current_time.replace(tzinfo=tz)
                return current_time

            @classmethod
            def fromisoformat(cls, timestamp):
                return datetime.fromisoformat(timestamp)

        mock_datetime.now = MockDatetime.now
        mock_datetime.fromisoformat = MockDatetime.fromisoformat
        mock_datetime.datetime = datetime
        mock_datetime.timezone = timezone
        mock_datetime.timedelta = timedelta

        # Test successful weather fetch
        mock_cached_requests.return_value = {
            "data": {
                "hourly": {
                    "time": [
                        "2024-01-01T12:00",  # Current hour
                        "2024-01-01T13:00",  # Next hour
                        "2024-01-01T14:00",  # Future hours...
                    ],
                    "apparent_temperature": [25.5, 26.0, 26.5],
                    "precipitation_probability": [30, 35, 40],
                    "weather_code": [0, 1, 2],
                    "cloud_cover": [50, 55, 60],
                    "visibility": [10000, 9000, 8000],
                }
            }
        }

        weather = get_weather()
        assert weather is not None
        assert weather["apparent_temperature"] == 25.5
        assert weather["precipitation_probability"] == 30
        assert weather["weather_code"] == 0
        assert weather["cloud_cover"] == 50
        assert weather["visibility"] == 10000

        # Test failed weather fetch
        mock_cached_requests.return_value = None
        assert get_weather() == {}


def test_get_prices_basic():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        # Mock the API response with some basic cryptocurrency data
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {"USD": {"price": 50000.0, "percent_change_24h": 5.25}},
                },
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "quote": {"USD": {"price": 2500.0, "percent_change_24h": -2.5}},
                },
            ]
        }

        # Test basic price query
        result = get_prices("")
        assert result is not None
        assert "BTC: 50000" in result
        assert "ETH: 2500" in result
        assert "+5.25%" in result
        assert "-2.5%" in result


def test_get_prices_amount_conversion():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "USDT",
                    "name": "Tether",
                    "quote": {"HKD": {"price": 7.8, "percent_change_24h": 0.1}},
                }
            ]
        }

        result = get_prices("2000 usdt in hkd")

        assert result is not None
        assert "USDT =" in result
        assert "HKD" in result
        assert "15600" in result


def test_get_prices_amount_conversion_accepts_to_and_en():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "USDT",
                    "name": "Tether",
                    "quote": {"HKD": {"price": 7.8, "percent_change_24h": 0.1}},
                }
            ]
        }

        to_result = get_prices("2000 usdt to hkd")
        en_result = get_prices("2000 usdt en hkd")

        assert to_result == "2000 USDT = 15600 HKD"
        assert en_result == "2000 USDT = 15600 HKD"


def test_get_prices_amount_conversion_reverse_accepts_a():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.side_effect = [
            {"data": []},
            {
                "data": [
                    {
                        "symbol": "USDT",
                        "name": "Tether",
                        "quote": {"HKD": {"price": 7.8, "percent_change_24h": 0.1}},
                    }
                ]
            },
        ]

        result = get_prices("2000 hkd a usdt")

        assert result == "2000 HKD = 256.41025641 USDT"


def test_get_prices_amount_conversion_reverse():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.side_effect = [
            {"data": []},
            {
                "data": [
                    {
                        "symbol": "USDT",
                        "name": "Tether",
                        "quote": {"HKD": {"price": 7.8, "percent_change_24h": 0.1}},
                    }
                ]
            },
        ]

        result = get_prices("2000 hkd in usdt")

        assert result == "2000 HKD = 256.41025641 USDT"


def test_get_prices_amount_conversion_invalid_symbol():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {"USD": {"price": 50000.0, "percent_change_24h": 5.25}},
                }
            ]
        }

        result = get_prices("2000 notacoin in usd")
        assert result == "no laburo con esos ponzis boludo"


def test_get_prices_amount_conversion_unsupported_pair_still_fails():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.side_effect = [
            {"data": []},
            {"data": []},
        ]

        result = get_prices("2000 hkd in usdt")
        assert result == "no laburo con esos ponzis boludo"


def test_get_prices_existing_paths_unchanged():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:

        def mock_get_prices_side_effect(currency):
            return {
                "data": [
                    {
                        "symbol": "BTC",
                        "name": "Bitcoin",
                        "quote": {
                            "USD": {"price": 50000.0, "percent_change_24h": 5.25},
                            "EUR": {"price": 46000.0, "percent_change_24h": 5.25},
                        },
                    },
                    {
                        "symbol": "ETH",
                        "name": "Ethereum",
                        "quote": {
                            "USD": {"price": 2500.0, "percent_change_24h": -2.5},
                            "EUR": {"price": 2200.0, "percent_change_24h": -2.5},
                        },
                    },
                    {
                        "symbol": "SOL",
                        "name": "Solana",
                        "quote": {
                            "USD": {"price": 120.0, "percent_change_24h": 1.25},
                            "EUR": {"price": 100.0, "percent_change_24h": 1.25},
                        },
                    },
                ]
            }

        mock_get_prices.side_effect = mock_get_prices_side_effect

        top_n_result = get_prices("10")
        list_result = get_prices("btc,eth")
        in_result = get_prices("in eur")

        assert top_n_result is not None
        assert "BTC:" in top_n_result

        assert list_result is not None
        assert "BTC:" in list_result
        assert "ETH:" in list_result
        assert "SOL:" not in list_result

        assert in_result is not None
        assert "EUR" in in_result


def test_get_prices_stablecoins_expands_known_symbols():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "USDT",
                    "name": "Tether",
                    "quote": {"USD": {"price": 1.0, "percent_change_24h": 0.01}},
                },
                {
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "quote": {"USD": {"price": 0.999, "percent_change_24h": -0.01}},
                },
            ]
        }

        result = get_prices("stables")

    assert result is not None
    assert "USDT" in result
    assert "USDC" in result


def test_get_prices_rejects_unsupported_timeframe_without_api_call():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        result = get_prices("btc 2h")

    assert result == "timeframe '2h' no soportado, uso: 1h, 24h, 7d, 30d"
    mock_get_prices.assert_not_called()


def test_get_prices_ignores_non_numeric_limit_tokens():
    get_prices = index.app_runtime.prices.get_prices

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        mock_get_prices.return_value = {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {"USD": {"price": 50000.0, "percent_change_24h": 1.0}},
                },
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "quote": {"USD": {"price": 2500.0, "percent_change_24h": 2.0}},
                },
            ]
        }

        result = get_prices("2,btc")

    assert result is not None
    assert "BTC" in result


def test_get_weather_description():
    from api.index import get_weather_description

    # Test various weather codes
    assert get_weather_description(0) == "despejado"
    assert get_weather_description(1) == "mayormente despejado"
    assert get_weather_description(2) == "parcialmente nublado"
    assert get_weather_description(3) == "nublado"
    assert get_weather_description(45) == "neblina"
    assert get_weather_description(61) == "lluvia leve"
    assert get_weather_description(95) == "tormenta"

    # Test unknown code
    assert get_weather_description(9999) == "clima raro"


def test_get_dollar_rates_basic():
    get_dollar_rates = index.app_runtime.dollar.get_rates

    with (
        patch("api.index.app_runtime.cache.request") as mock_cached_requests,
        patch("api.index.app_runtime.dollar.get_tcrm") as mock_tcrm,
        patch("api.index.app_runtime.dollar.get_band_limits", return_value=None),
    ):
        # Mock the API response with dollar rate data
        mock_cached_requests.return_value = {
            "data": {
                "mayorista": {"price": 90.0, "variation": 0.25},
                "oficial": {"price": 100.0, "variation": 0.5},
                "tarjeta": {"price": 150.0, "variation": 0.75},
                "mep": {
                    "al30": {
                        "24hs": {"price": 200.0, "variation": 1.25},
                        "ci": {"price": 200.0, "variation": 1.25},
                    }
                },
                "ccl": {
                    "al30": {
                        "24hs": {"price": 210.0, "variation": 1.5},
                        "ci": {"price": 210.0, "variation": 1.5},
                    }
                },
                "blue": {"ask": 220.0, "variation": 2.0},
                "cripto": {
                    "ccb": {"ask": 230.0, "variation": 2.5},
                    "usdc": {"ask": 235.0, "variation": 2.75},
                    "usdt": {"ask": 240.0, "variation": 3.0},
                },
            }
        }
        mock_tcrm.return_value = (130.0, 1.0)

        result = get_dollar_rates()
        assert result is not None
        assert "Mayorista: 90" in result
        assert "Oficial: 100" in result
        assert "Tarjeta: 150" in result
        assert "MEP: 200" in result
        assert "CCL: 210" in result
        assert "Blue: 220" in result
        assert "Bitcoin: 230" in result
        assert "USDC: 235" in result
        assert "USDT: 240" in result
        assert "TCRM 100: 130 (+1% 24hs)" in result


def test_get_dollar_rates_api_failure():
    get_dollar_rates = index.app_runtime.dollar.get_rates
    import pytest

    with (
        patch("api.index.app_runtime.cache.request") as mock_cached_requests,
        patch("api.index.app_runtime.dollar.get_tcrm") as mock_tcrm,
    ):
        # Mock API failure
        mock_cached_requests.return_value = None
        mock_tcrm.return_value = (None, None)

        # The function should raise an exception when API fails
        with pytest.raises(TypeError):
            get_dollar_rates()


def test_get_dollar_rates_unsupported_timeframe():
    get_dollar_rates = index.app_runtime.dollar.get_rates

    result = get_dollar_rates("7d")
    assert result is not None
    assert "7d" in result
    assert "no soportado" in result


def test_get_devo_with_fee_only():
    get_devo = index.app_runtime.dollar.get_devo

    with patch("api.index.app_runtime.cache.request") as mock_cached_requests:
        # Mock the API response
        mock_cached_requests.return_value = {
            "data": {
                "oficial": {"price": 100.0},
                "tarjeta": {"price": 150.0},
                "cripto": {"usdt": {"ask": 200.0, "bid": 190.0}},
            }
        }

        result = get_devo("0.5")
        assert result is not None
        assert "ganancia: 62.68%" in result


def test_get_devo_with_fee_and_amount():
    get_devo = index.app_runtime.dollar.get_devo

    with patch("api.index.app_runtime.cache.request") as mock_cached_requests:
        # Mock the API response
        mock_cached_requests.return_value = {
            "data": {
                "oficial": {"price": 100.0},
                "tarjeta": {"price": 150.0},
                "cripto": {"usdt": {"ask": 200.0, "bid": 190.0}},
            }
        }

        result = get_devo("0.5, 100")
        assert result is not None
        assert "100 USD Tarjeta" in result
        assert "Ganarias" in result


def test_get_devo_invalid_input():
    get_devo = index.app_runtime.dollar.get_devo

    result = get_devo("invalid")
    assert "uso: /devo <fee_porcentaje>[, <monto_compra>]" in result


def test_satoshi_basic():
    satoshi = index.app_runtime.dollar.satoshi

    with patch("api.index.app_runtime.dollar.get_btc_price") as mock_get_price:
        # Mock the API response for both USD and ARS
        def mock_get_price_side_effect(currency):
            if currency == "USD":
                return 50000.0
            if currency == "ARS":
                return 10000000.0
            return None

        mock_get_price.side_effect = mock_get_price_side_effect

        result = satoshi()
        assert result is not None
        assert "1 satoshi = $0.00050000 USD" in result
        assert "1 satoshi = $0.1000 ARS" in result
        assert "$1 USD = 2,000 sats" in result
        assert "$1 ARS = 10.000 sats" in result


def test_powerlaw_basic():
    from api.index import powerlaw

    with patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices:
        # Mock the API response for BTC price
        mock_get_prices.return_value = {
            "data": [
                {"quote": {"USD": {"price": 50000.0}}},
            ]
        }

        result = powerlaw()
        assert result is not None
        assert "segun power law btc deberia estar en" in result


def test_rainbow_basic():
    from api.index import rainbow

    with (
        patch("api.index.app_runtime.prices.get_api_prices") as mock_get_prices,
        patch("api.index.datetime") as mock_datetime,
    ):
        # Mock the API response for BTC price
        mock_get_prices.return_value = {
            "data": [
                {"quote": {"USD": {"price": 50000.0}}},
            ]
        }

        # Mock the current date to a fixed value for consistent calculations
        mock_datetime.now.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = rainbow()
        assert result is not None
        assert "segun rainbow chart btc deberia estar en" in result
