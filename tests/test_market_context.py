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

    with patch("api.index.cached_requests", side_effect=fake_cached_requests):
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
    from api.rulo_commands import build_rulo_message

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
    from api.dollar_commands import sort_dollar_rates

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
        patch("api.index.cached_requests") as mock_cached_requests,
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        mock_get_prices.side_effect = [
            {"data": []},
            {"data": []},
        ]

        result = get_prices("2000 hkd in usdt")
        assert result == "no laburo con esos ponzis boludo"


def test_get_prices_existing_paths_unchanged():
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:

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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        result = get_prices("btc 2h")

    assert result == "timeframe '2h' no soportado, uso: 1h, 24h, 7d, 30d"
    mock_get_prices.assert_not_called()


def test_get_prices_ignores_non_numeric_limit_tokens():
    from api.index import get_prices

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
    from api.index import get_dollar_rates

    with (
        patch("api.index.cached_requests") as mock_cached_requests,
        patch("api.index.get_cached_tcrm_100") as mock_tcrm,
        patch("api.index.get_currency_band_limits", return_value=None),
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
    from api.index import get_dollar_rates
    import pytest

    with (
        patch("api.index.cached_requests") as mock_cached_requests,
        patch("api.index.get_cached_tcrm_100") as mock_tcrm,
    ):
        # Mock API failure
        mock_cached_requests.return_value = None
        mock_tcrm.return_value = (None, None)

        # The function should raise an exception when API fails
        with pytest.raises(TypeError):
            get_dollar_rates()


def test_get_dollar_rates_unsupported_timeframe():
    from api.index import get_dollar_rates

    result = get_dollar_rates("7d")
    assert result is not None
    assert "7d" in result
    assert "no soportado" in result


def test_get_devo_with_fee_only():
    from api.index import get_devo

    with patch("api.index.cached_requests") as mock_cached_requests:
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
    from api.index import get_devo

    with patch("api.index.cached_requests") as mock_cached_requests:
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
    from api.index import get_devo

    result = get_devo("invalid")
    assert "uso: /devo <fee_porcentaje>[, <monto_compra>]" in result


def test_satoshi_basic():
    from api.index import satoshi

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
        # Mock the API response for both USD and ARS
        def mock_get_prices_side_effect(currency):
            if currency == "USD":
                return {
                    "data": [
                        {"quote": {"USD": {"price": 50000.0}}},
                    ]
                }
            elif currency == "ARS":
                return {
                    "data": [
                        {"quote": {"ARS": {"price": 10000000.0}}},
                    ]
                }
            return None

        mock_get_prices.side_effect = mock_get_prices_side_effect

        result = satoshi()
        assert result is not None
        assert "1 satoshi = $0.00050000 USD" in result
        assert "1 satoshi = $0.1000 ARS" in result
        assert "$1 USD = 2,000 sats" in result
        assert "$1 ARS = 10.000 sats" in result


def test_powerlaw_basic():
    from api.index import powerlaw

    with patch("api.index.get_api_or_cache_prices") as mock_get_prices:
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
        patch("api.index.get_api_or_cache_prices") as mock_get_prices,
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


def test_get_market_context_success():
    from api.index import get_market_context

    with (
        patch("api.index.cached_requests") as mock_cached,
        patch("api.index.clean_crypto_data") as mock_clean,
        patch("os.environ.get") as mock_env,
    ):
        # Mock crypto response
        crypto_response = {
            "data": {"data": [{"symbol": "BTC", "quote": {"USD": {"price": 50000}}}]}
        }

        # Mock dollar response
        dollar_response = {
            "data": {"oficial": {"price": 1000}, "blue": {"price": 1200}}
        }

        def mock_requests_side_effect(url, *_args, **_kwargs):
            if "coinmarketcap" in url:
                return crypto_response
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_clean.return_value = [{"symbol": "BTC", "price": 50000}]
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" in result
        assert "dollar" in result
        assert result["crypto"] == [{"symbol": "BTC", "price": 50000}]
        assert result["dollar"] == {"oficial": {"price": 1000}, "blue": {"price": 1200}}


def test_get_market_context_crypto_fail():
    from api.index import get_market_context

    with (
        patch("api.index.cached_requests") as mock_cached,
        patch("os.environ.get") as mock_env,
    ):
        # Mock dollar response only
        dollar_response = {"data": {"oficial": {"price": 1000}}}

        def mock_requests_side_effect(url, *_args, **_kwargs):
            if "coinmarketcap" in url:
                return None  # Crypto fails
            elif "criptoya" in url:
                return dollar_response
            return None

        mock_cached.side_effect = mock_requests_side_effect
        mock_env.return_value = "test_api_key"

        result = get_market_context()

        assert "crypto" not in result
        assert "dollar" in result


def test_get_market_context_all_fail():
    from api.index import get_market_context

    with (
        patch("api.index.cached_requests") as mock_cached,
        patch("os.environ.get") as mock_env,
        patch("api.index.get_cached_bcra_variables") as mock_get_bcra,
        patch("api.index.bcra_fetch_latest_variables") as mock_fetch_bcra,
        patch("api.index.cache_bcra_variables") as mock_cache_bcra,
    ):
        mock_cached.return_value = None
        mock_env.return_value = "test_api_key"
        mock_get_bcra.return_value = None
        mock_fetch_bcra.return_value = None
        mock_cache_bcra.return_value = None

        result = get_market_context()

        assert result == {}


def test_get_weather_context_success():
    from api.index import get_weather_context

    with (
        patch("api.index.get_weather") as mock_weather,
        patch("api.index.get_weather_description") as mock_description,
    ):
        mock_weather.return_value = {"temperature": 25.0, "weather_code": 0}
        mock_description.return_value = "cielo despejado"

        result = get_weather_context()

        assert result is not None
        assert result["temperature"] == 25.0
        assert result["weather_code"] == 0
        assert result["description"] == "cielo despejado"
        mock_description.assert_called_once_with(0)


def test_get_weather_context_fail():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.return_value = None

        result = get_weather_context()

        assert result is None


def test_get_weather_context_exception():
    from api.index import get_weather_context

    with patch("api.index.get_weather") as mock_weather:
        mock_weather.side_effect = Exception("Weather API error")

        result = get_weather_context()

        assert result is None


def test_get_time_context():
    from api.index import get_time_context
    from datetime import datetime, timezone, timedelta

    with patch("api.index.datetime") as mock_datetime:
        # Mock a fixed time
        fixed_time = datetime(2024, 1, 15, 14, 30, 0)
        buenos_aires_tz = timezone(timedelta(hours=-3))
        fixed_time_ba = fixed_time.replace(tzinfo=buenos_aires_tz)

        mock_datetime.now.return_value = fixed_time_ba
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        result = get_time_context()

        assert "datetime" in result
        assert "formatted" in result
        assert result["datetime"] == fixed_time_ba


def test_get_hacker_news_context_success():
    from api.index import get_hacker_news_context

    sample_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <rss version=\"2.0\">
      <channel>
        <item>
          <title>Historia Uno</title>
          <link>https://example.com/uno</link>
          <description><![CDATA[
            <p>Article URL: <a href=\"https://example.com/uno\">https://example.com/uno</a></p>
            <p>Comments URL: <a href=\"https://news.ycombinator.com/item?id=1\">https://news.ycombinator.com/item?id=1</a></p>
            <p>Points: 123</p>
            <p># Comments: 45</p>
          ]]></description>
        </item>
        <item>
          <title>Historia Dos</title>
          <link>https://example.com/dos</link>
          <description><![CDATA[
            <p>Comments URL: <a href=\"https://news.ycombinator.com/item?id=2\">https://news.ycombinator.com/item?id=2</a></p>
            <p>Points: 456</p>
            <p># Comments: 78</p>
          ]]></description>
        </item>
      </channel>
    </rss>"""

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    with (
        patch("api.index.config_redis", side_effect=RuntimeError("no redis")),
        patch(
            "api.index.requests.get", return_value=DummyResponse(sample_xml)
        ) as mock_get,
    ):
        items = get_hacker_news_context(limit=2)

    assert len(items) == 2
    assert items[0]["title"] == "Historia Uno"
    assert items[0]["points"] == 123
    assert items[0]["comments"] == 45
    assert items[0]["comments_url"].endswith("id=1")
    assert items[1]["title"] == "Historia Dos"
    assert items[1]["points"] == 456
    assert items[1]["comments"] == 78
    mock_get.assert_called_once()


def test_get_hacker_news_context_uses_cache():
    from api.index import get_hacker_news_context

    cached_items = [
        {
            "title": "Cacheada",
            "url": "https://cached.example",
            "points": 50,
            "comments": 5,
            "comments_url": "https://news.ycombinator.com/item?id=3",
        }
    ]

    with (
        patch("api.index.config_redis", return_value=object()),
        patch("api.index.redis_get_json", return_value=cached_items) as mock_cache,
        patch("api.index.http_client.get") as mock_get,
    ):
        items = get_hacker_news_context(limit=1)

    assert items == cached_items[:1]
    mock_cache.assert_called_once()
    mock_get.assert_not_called()


def test_get_fallback_response():
    from api.index import get_fallback_response

    messages = [{"role": "user", "content": "hello"}]

    result = get_fallback_response(messages)

    # Should return a string (one of many predefined fallback responses)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result) < 50  # Reasonable length for a fallback response


def test_build_system_message():
    from api.index import build_system_message

    # Reset global cache to ensure clean state
    config_module.reset_cache()

    context = {
        "market": {
            "crypto": [{"symbol": "BTC", "price": 50000}],
            "dollar": {"oficial": {"price": 1000}},
        },
        "weather": {
            "temperature": 25,
            "apparent_temperature": 26,
            "precipitation_probability": 10,
            "description": "cielo despejado",
            "cloud_cover": 20,
            "visibility": 10000,
        },
        "time": {"formatted": "Monday 15/01/2024"},
        "hacker_news": [
            {
                "title": "Nueva feature",
                "url": "https://example.com/hn",
                "points": 321,
                "comments": 42,
                "comments_url": "https://news.ycombinator.com/item?id=1",
            }
        ],
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "Sos el gordo, un bot argentino de prueba.",
                "BOT_TRIGGER_WORDS": "gordo,test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0
        content_text = result["content"][0]["text"]
        assert "argentin" in content_text.lower()
        assert "gordo" in content_text.lower()
        assert "hacker news" in content_text.lower()


def test_build_system_message_empty_context():
    from api.index import build_system_message

    # Reset global cache to ensure clean state
    config_module.reset_cache()

    context = {
        "market": {},
        "weather": None,
        "time": {"formatted": "Monday"},
        "hacker_news": [],
    }

    with patch("os.environ.get") as mock_env:
        # Mock environment variables
        def env_side_effect(key):
            env_vars = {
                "BOT_SYSTEM_PROMPT": "You are a test bot assistant.",
                "BOT_TRIGGER_WORDS": "test,bot",
            }
            return env_vars.get(key)

        mock_env.side_effect = env_side_effect

        result = build_system_message(context)

        assert result["role"] == "system"
        assert "content" in result
        assert isinstance(result["content"], list)
        # Should still have base personality
        content_text = result["content"][0]["text"]
        assert len(content_text) > 100


def test_build_system_message_task_tool_instructions_preserve_perspective():
    from api.index import build_system_message

    context = {
        "market": {},
        "weather": None,
        "time": {"formatted": "Monday"},
        "hacker_news": [],
    }

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "task_set",
                "description": "Create a scheduled task.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    result = build_system_message(context, tools_active=True, tool_schemas=tool_schemas)

    content_text = result["content"][0]["text"]
    assert (
        "task_set.text debe contener solo el contenido a ejecutar despues"
        in content_text
    )
    assert "no reescribas pronombres ni cambies sujeto" in content_text
    assert "no incluyas tiempo ni frecuencia en text" in content_text
    assert 'text="decime cuanta aura farmeaste hoy"' in content_text
    assert 'text="deci fumareeemooss"' in content_text


def test_clean_crypto_data():
    from api.index import clean_crypto_data

    raw_data = [
        {
            "id": 1,
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 50000.123456,
                    "volume_24h": 1000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.5,
                    "percent_change_7d": 5.0,
                    "percent_change_30d": 10.0,
                    "market_cap": 1000000000000,
                    "market_cap_dominance": 45.5,
                }
            },
        }
    ]

    result = clean_crypto_data(raw_data)

    assert len(result) == 1
    assert result[0]["symbol"] == "BTC"
    assert result[0]["name"] == "Bitcoin"
    assert result[0]["slug"] == "bitcoin"
    assert result[0]["quote"]["USD"]["price"] == 50000.123456


def test_format_market_info():
    from api.index import format_market_info

    market_data = {
        "crypto": [{"symbol": "BTC", "price": 50000, "change_24h": 2.5}],
        "dollar": {"oficial": {"price": 1000}, "blue": {"price": 1200}},
    }

    result = format_market_info(market_data)

    assert "PRECIOS DE CRIPTOS:" in result
    assert "DOLARES:" in result
    assert "BTC" in result
    assert "50.000" in result or "50000" in result
    assert "1.000" in result or "1000" in result


def test_market_commands_format_market_info_compact_summary():
    from api.market_commands import format_market_info

    market_data = {
        "crypto": [
            {
                "symbol": "BTC",
                "quote": {
                    "USD": {
                        "price": 50000,
                        "changes": {"24h": 2.5},
                        "dominance": 52.25,
                    }
                },
            }
        ],
        "dollar": {
            "oficial": {"price": 1000},
            "blue": {"ask": 1200, "bid": 1180},
            "mep": {"al30": {"ci": {"price": 1150}}},
            "tarjeta": {"price": 1600},
            "cripto": {"usdt": {"ask": 1230, "bid": 1220}},
        },
    }

    result = format_market_info(market_data)

    assert result == "\n".join(
        [
            "PRECIOS DE CRIPTOS:",
            "- BTC: 50000 usd (+2.5 24h), dom 52.2%",
            "DOLARES:",
            "- oficial: 1000",
            "- blue: 1200 (bid 1180)",
            "- mep al30 ci: 1150",
            "- tarjeta: 1600",
            "- usdt: 1230 (bid 1220)",
        ]
    )


def test_format_market_info_empty_dict():
    from api.index import format_market_info

    result = format_market_info({})

    assert result == ""


def test_format_weather_info():
    from api.index import format_weather_info

    weather_data = {
        "temperature": 25.5,
        "apparent_temperature": 26.0,
        "precipitation_probability": 10,
        "description": "cielo despejado",
        "cloud_cover": 20,
        "visibility": 10000,
    }

    result = format_weather_info(weather_data)

    assert "26" in result  # apparent_temperature
    assert "cielo despejado" in result
    assert "10%" in result  # precipitation_probability


def test_format_weather_info_empty():
    from api.index import format_weather_info

    result = format_weather_info({})
    assert "?" in result or "sin datos" in result


def test_sort_dollar_rates_success():
    """Test sort_dollar_rates with valid dollar rates data"""
    from api.index import sort_dollar_rates

    dollar_rates = {
        "data": {
            "mayorista": {"price": 900.00, "variation": 1.0},
            "oficial": {"price": 1000.50, "variation": 1.2},
            "tarjeta": {"price": 1600.75, "variation": -0.8},
            "mep": {
                "al30": {
                    "24hs": {"price": 1050.25, "variation": 0.5},
                    "ci": {"price": 1050.25, "variation": 0.5},
                }
            },
            "ccl": {
                "al30": {
                    "24hs": {"price": 1075.80, "variation": 0.7},
                    "ci": {"price": 1075.80, "variation": 0.7},
                }
            },
            "blue": {"ask": 1200.00, "variation": 2.1},
            "cripto": {
                "ccb": {"ask": 1150.90, "variation": 1.8},
                "usdc": {"ask": 1140.30, "variation": 1.5},
                "usdt": {"ask": 1145.60, "variation": 1.6},
            },
        }
    }

    result = sort_dollar_rates(dollar_rates)

    assert len(result) == 9
    assert result[0]["name"] == "Mayorista"
    assert result[0]["price"] == 900.00
    assert result[-1]["name"] == "Tarjeta"
    assert result[-1]["price"] == 1600.75
    # Verify sorting by price
    for i in range(len(result) - 1):
        assert result[i]["price"] <= result[i + 1]["price"]


def test_sort_dollar_rates_with_none_variations():
    """Test sort_dollar_rates with None variation values"""
    from api.index import sort_dollar_rates

    dollar_rates = {
        "data": {
            "mayorista": {"price": 900.00, "variation": None},
            "oficial": {"price": 1000.50, "variation": None},
            "tarjeta": {"price": 1600.75, "variation": None},
            "mep": {
                "al30": {
                    "24hs": {"price": 1050.25, "variation": None},
                    "ci": {"price": 1050.25, "variation": None},
                }
            },
            "ccl": {
                "al30": {
                    "24hs": {"price": 1075.80, "variation": None},
                    "ci": {"price": 1075.80, "variation": None},
                }
            },
            "blue": {"ask": 1200.00, "variation": None},
            "cripto": {
                "ccb": {"ask": 1150.90, "variation": None},
                "usdc": {"ask": 1140.30, "variation": None},
                "usdt": {"ask": 1145.60, "variation": None},
            },
        }
    }

    result = sort_dollar_rates(dollar_rates)

    assert len(result) == 9
    for rate in result:
        assert rate["history"] is None


def test_sort_dollar_rates_non24h_no_history_gives_none_variations():
    from api.index import sort_dollar_rates

    dollar_rates = {
        "timestamp": 1000,
        "data": {
            "mayorista": {"price": 1100.0, "variation": 1.0},
            "oficial": {"price": 1100.0, "variation": 1.0},
            "tarjeta": {"price": 1870.0, "variation": 1.0},
            "mep": {"al30": {"ci": {"price": 1180.0, "variation": 2.0}}},
            "ccl": {"al30": {"ci": {"price": 1200.0, "variation": 2.0}}},
            "blue": {"ask": 1190.0, "bid": 1170.0, "variation": 1.5},
            "cripto": {
                "ccb": {"ask": 1220.0, "bid": 1200.0, "variation": 1.0},
                "usdc": {"ask": 1210.0, "bid": 1190.0, "variation": 0.5},
                "usdt": {"ask": 1215.0, "bid": 1195.0, "variation": 0.8},
            },
        },
    }
    result = sort_dollar_rates(dollar_rates, hours_ago=1)
    assert all(r["history"] is None for r in result)


def test_format_dollar_rates_shows_no_history_footer():
    from api.index import format_dollar_rates

    rates = [
        {"name": "Blue", "price": 1200.0, "history": None},
        {"name": "Oficial", "price": 1100.0, "history": None},
    ]
    result = format_dollar_rates(rates, hours_ago=6)
    assert result is not None
    assert "sin datos historicos" in result
    assert "6hs" in result


def test_format_dollar_rates_no_footer_for_24h():
    from api.index import format_dollar_rates

    rates = [{"name": "Blue", "price": 1200.0, "history": None}]
    result = format_dollar_rates(rates, hours_ago=24)
    assert "sin datos historicos" not in result


def test_get_prices_unsupported_timeframe():
    from api.index import get_prices

    result = get_prices("BTC 6h")
    assert result is not None
    assert "6h" in result
    assert "no soportado" in result


def test_get_prices_7d_uses_correct_cmc_field():
    from api.index import get_prices

    mock_response = {
        "timestamp": 1000,
        "data": {
            "data": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "quote": {
                        "USD": {
                            "price": 90000.0,
                            "percent_change_24h": 1.0,
                            "percent_change_7d": 5.5,
                        }
                    },
                }
            ]
        },
    }
    with patch("api.index.cached_requests", return_value=mock_response):
        result = get_prices("BTC 7d")
    assert result is not None
    assert "5.5" in result or "+5.5" in result
    assert "7d" in result


def test_format_dollar_rates_with_positive_variations():
    """Test format_dollar_rates with positive variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": 1.2},
        {"name": "Blue", "price": 1200.00, "history": 2.1},
        {"name": "MEP", "price": 1050.25, "history": 0.5},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = [
        "Oficial: 1000.5 (+1.2% 24hs)",
        "MEP: 1050.25 (+0.5% 24hs)",
        "Blue: 1200 (+2.1% 24hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_with_negative_variations():
    """Test format_dollar_rates with negative variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Tarjeta", "price": 1600.75, "history": -0.8},
        {"name": "CCL", "price": 1075.80, "history": -1.5},
    ]

    result = format_dollar_rates(dollar_rates, 12)

    expected_lines = ["CCL: 1075.8 (-1.5% 12hs)", "Tarjeta: 1600.75 (-0.8% 12hs)"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_with_none_variations():
    """Test format_dollar_rates with None variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": None},
        {"name": "Blue", "price": 1200.00, "history": None},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = ["Oficial: 1000.5", "Blue: 1200"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_mixed_variations():
    """Test format_dollar_rates with mixed variation values"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": 1.2},
        {"name": "Blue", "price": 1200.00, "history": None},
        {"name": "Tarjeta", "price": 1600.75, "history": -0.8},
    ]

    result = format_dollar_rates(dollar_rates, 6)

    expected_lines = [
        "Oficial: 1000.5 (+1.2% 6hs)",
        "Blue: 1200",
        "Tarjeta: 1600.75 (-0.8% 6hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_zero_decimal_formatting():
    """Test format_dollar_rates decimal formatting for whole numbers"""
    from api.index import format_dollar_rates

    dollar_rates = [
        {"name": "Test1", "price": 1000.00, "history": 0.00},
        {"name": "Test2", "price": 1200.10, "history": 1.00},
        {"name": "Test3", "price": 1500.50, "history": -2.50},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = [
        "Test1: 1000 (+0% 24hs)",
        "Test2: 1200.1 (+1% 24hs)",
        "Test3: 1500.5 (-2.5% 24hs)",
    ]
    assert result == "\n".join(expected_lines)


def test_clean_crypto_data_success():
    """Test clean_crypto_data with valid crypto data"""
    from api.index import clean_crypto_data

    cryptos = [
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 45000.50,
                    "volume_24h": 25000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.1,
                    "percent_change_7d": -1.8,
                    "percent_change_30d": 15.2,
                    "market_cap": 877500000000,
                    "market_cap_dominance": 42.5,
                }
            },
        }
    ]

    result = clean_crypto_data(cryptos)

    assert len(result) == 1
    crypto = result[0]
    assert crypto["name"] == "Bitcoin"
    assert crypto["symbol"] == "BTC"
    assert crypto["slug"] == "bitcoin"
    assert crypto["supply"]["max"] == 21000000
    assert crypto["supply"]["circulating"] == 19500000
    assert crypto["quote"]["USD"]["price"] == 45000.50
    assert crypto["quote"]["USD"]["changes"]["24h"] == 2.1


def test_clean_crypto_data_multiple_cryptos():
    """Test clean_crypto_data with multiple cryptocurrencies"""
    from api.index import clean_crypto_data

    cryptos = [
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "slug": "bitcoin",
            "max_supply": 21000000,
            "circulating_supply": 19500000,
            "total_supply": 19500000,
            "infinite_supply": False,
            "quote": {
                "USD": {
                    "price": 45000.50,
                    "volume_24h": 25000000000,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 2.1,
                    "percent_change_7d": -1.8,
                    "percent_change_30d": 15.2,
                    "market_cap": 877500000000,
                    "market_cap_dominance": 42.5,
                }
            },
        },
        {
            "name": "Ethereum",
            "symbol": "ETH",
            "slug": "ethereum",
            "max_supply": None,
            "circulating_supply": 120000000,
            "total_supply": 120000000,
            "infinite_supply": True,
            "quote": {
                "USD": {
                    "price": 3000.25,
                    "volume_24h": 15000000000,
                    "percent_change_1h": -0.2,
                    "percent_change_24h": 1.8,
                    "percent_change_7d": -3.2,
                    "percent_change_30d": 8.5,
                    "market_cap": 360000000000,
                    "market_cap_dominance": 18.2,
                }
            },
        },
    ]

    result = clean_crypto_data(cryptos)

    assert len(result) == 2
    assert result[0]["name"] == "Bitcoin"
    assert result[1]["name"] == "Ethereum"
    assert result[1]["supply"]["max"] is None
    assert result[1]["supply"]["infinite"] is True


def test_format_market_info_with_crypto_and_dollar():
    """Test format_market_info with both crypto and dollar data"""
    from api.index import format_market_info

    market = {
        "crypto": [
            {"name": "Bitcoin", "price": 45000.50},
            {"name": "Ethereum", "price": 3000.25},
        ],
        "dollar": [
            {"name": "Oficial", "price": 1000.50},
            {"name": "Blue", "price": 1200.00},
        ],
    }

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" in result
    assert "DOLARES:" in result
    assert "BITCOIN" in result or "BTC" in result
    assert "oficial" in result.lower()


def test_format_market_info_crypto_only():
    """Test format_market_info with only crypto data"""
    from api.index import format_market_info

    market = {"crypto": [{"name": "Bitcoin", "price": 45000.50}]}

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" in result
    assert "DOLARES:" not in result
    assert "BITCOIN" in result or "BTC" in result


def test_format_market_info_dollar_only():
    """Test format_market_info with only dollar data"""
    from api.index import format_market_info

    market = {"dollar": [{"name": "Oficial", "price": 1000.50}]}

    result = format_market_info(market)

    assert "PRECIOS DE CRIPTOS:" not in result
    assert "DOLARES:" in result
    assert "oficial" in result.lower()


def test_format_market_info_empty():
    """Test format_market_info with empty market data"""
    from api.index import format_market_info

    market = {}

    result = format_market_info(market)

    assert result == ""


def test_get_weather_description_clear():
    """Test get_weather_description for clear weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(0) == "despejado"
    assert get_weather_description(1) == "mayormente despejado"
    assert get_weather_description(2) == "parcialmente nublado"
    assert get_weather_description(3) == "nublado"


def test_get_weather_description_rain():
    """Test get_weather_description for rain weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(61) == "lluvia leve"
    assert get_weather_description(63) == "lluvia moderada"
    assert get_weather_description(65) == "lluvia intensa"


def test_get_weather_description_storm():
    """Test get_weather_description for storm weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(95) == "tormenta"
    assert get_weather_description(96) == "tormenta con granizo leve"
    assert get_weather_description(99) == "tormenta con granizo intenso"


def test_get_weather_description_unknown():
    """Test get_weather_description for unknown weather codes"""
    from api.index import get_weather_description

    assert get_weather_description(999) == "clima raro"
    assert get_weather_description(-1) == "clima raro"


def test_get_oil_price_falls_back_to_stooq_quote_endpoint_when_daily_is_empty():
    brent_json = {
        "chart": {
            "result": [{
                "indicators": {"quote": [{"close": [107.6, 98.15]}]}
            }],
            "error": None,
        }
    }
    wti_json = {
        "chart": {
            "result": [{
                "indicators": {"quote": [{"close": [106.75, 95.45]}]}
            }],
            "error": None,
        }
    }

    with patch("api.index.cached_requests") as mock_get:
        mock_get.side_effect = [{"data": brent_json}, {"data": wti_json}]
        result = get_oil_price()

    assert "Brent: 98.15 USD (-8.78% 24hs)" in result
    assert "WTI: 95.45 USD (-10.59% 24hs)" in result


def test_get_oil_price_uses_cached_requests(monkeypatch):
    from api import index

    calls = []

    def fake_cached_requests(api_url, parameters, headers, expiration_time, *args, **kwargs):
        calls.append((api_url, parameters, headers, expiration_time))
        return {
            "data": {
                "chart": {
                    "result": [
                        {"indicators": {"quote": [{"close": [100.0, 101.5]}]}}
                    ]
                }
            }
        }

    monkeypatch.setattr(index, "cached_requests", fake_cached_requests)

    result = index.get_oil_price()

    assert "Brent" in result
    assert "WTI" in result
    assert len(calls) == 2


def test_get_oil_price_returns_error_when_all_sources_fail():
    with patch("api.index.cached_requests", return_value=None):
        result = get_oil_price()

    assert result == "no pude traer el precio del petróleo boludo"


def test_cached_requests_uses_shared_http_get(monkeypatch):
    from api import index

    class Redis:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def setex(self, key, ttl, value):
            self.store[key] = value
            return True

    class Response:
        text = '{"ok": true}'

        def raise_for_status(self):
            return None

    calls = []
    monkeypatch.setattr(index, "config_redis", lambda: Redis())
    monkeypatch.setattr(
        index.http_client,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    result = index.cached_requests("https://example.test", None, None, 10)

    assert result["data"] == {"ok": True}
    assert calls


def test_get_dollar_rates_returns_formatted_snapshot(monkeypatch):
    from api import index

    class FakeCache:
        def get(self, **kwargs):
            return index.StaleCacheResult(value="cached dolar", status="fresh")

    monkeypatch.setattr(index, "_get_dollar_snapshot_cache", lambda: FakeCache())
    monkeypatch.setattr(
        index,
        "_build_dollar_rates_text",
        lambda *_args, **_kwargs: "live dolar",
        raising=False,
    )

    assert index.get_dollar_rates("") == "cached dolar"


def test_get_dollar_rates_keeps_invalid_timeframe_without_cache(monkeypatch):
    from api import index

    called = False

    def fail_cache():
        nonlocal called
        called = True
        raise AssertionError("cache should not be used")

    monkeypatch.setattr(index, "_get_dollar_snapshot_cache", fail_cache, raising=False)

    result = index.get_dollar_rates("2h")

    assert "no soportado" in result
    assert called is False


def test_background_refresh_scheduler_uses_bounded_executor(monkeypatch):
    from api import index

    submitted = []

    class DummyExecutor:
        def submit(self, fn):
            submitted.append(fn)
            return object()

    monkeypatch.setattr(index, "_BACKGROUND_REFRESH_EXECUTOR", DummyExecutor())

    index._schedule_background_refresh(lambda: None)

    assert len(submitted) == 1


def test_refresh_price_caches_submits_independent_jobs(monkeypatch):
    from concurrent.futures import Future
    from api import index

    submitted = []

    class Executor:
        def submit(self, fn):
            submitted.append(fn)
            f = Future()
            f.set_result(None)
            return f

    monkeypatch.setattr(index, "_BACKGROUND_REFRESH_EXECUTOR", Executor())

    index.refresh_price_caches()

    assert len(submitted) == 4
