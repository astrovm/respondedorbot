from tests.support import *


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
    from api.markets.context import format_market_info

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
    from api.markets.dollar import format_dollar_rates

    rates = [
        {"name": "Blue", "price": 1200.0, "history": None},
        {"name": "Oficial", "price": 1100.0, "history": None},
    ]
    result = format_dollar_rates(rates, hours_ago=6)
    assert result is not None
    assert "sin datos historicos" in result
    assert "6hs" in result


def test_format_dollar_rates_no_footer_for_24h():
    from api.markets.dollar import format_dollar_rates

    rates = [{"name": "Blue", "price": 1200.0, "history": None}]
    result = format_dollar_rates(rates, hours_ago=24)
    assert "sin datos historicos" not in result


def test_get_prices_unsupported_timeframe():
    get_prices = index.app_runtime.prices.get_prices

    result = get_prices("BTC 6h")
    assert result is not None
    assert "6h" in result
    assert "no soportado" in result


def test_get_prices_7d_uses_correct_cmc_field():
    get_prices = index.app_runtime.prices.get_prices

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
    with patch("api.index.app_runtime.cache.request", return_value=mock_response):
        result = get_prices("BTC 7d")
    assert result is not None
    assert "5.5" in result or "+5.5" in result
    assert "7d" in result


def test_format_dollar_rates_with_positive_variations():
    """Test format_dollar_rates with positive variation values"""
    from api.markets.dollar import format_dollar_rates

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
    from api.markets.dollar import format_dollar_rates

    dollar_rates = [
        {"name": "Tarjeta", "price": 1600.75, "history": -0.8},
        {"name": "CCL", "price": 1075.80, "history": -1.5},
    ]

    result = format_dollar_rates(dollar_rates, 12)

    expected_lines = ["CCL: 1075.8 (-1.5% 12hs)", "Tarjeta: 1600.75 (-0.8% 12hs)"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_with_none_variations():
    """Test format_dollar_rates with None variation values"""
    from api.markets.dollar import format_dollar_rates

    dollar_rates = [
        {"name": "Oficial", "price": 1000.50, "history": None},
        {"name": "Blue", "price": 1200.00, "history": None},
    ]

    result = format_dollar_rates(dollar_rates, 24)

    expected_lines = ["Oficial: 1000.5", "Blue: 1200"]
    assert result == "\n".join(expected_lines)


def test_format_dollar_rates_mixed_variations():
    """Test format_dollar_rates with mixed variation values"""
    from api.markets.dollar import format_dollar_rates

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
    from api.markets.dollar import format_dollar_rates

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

    with patch("api.index.app_runtime.cache.request") as mock_get:
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

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)

    result = index.app_runtime.stocks.get_oil_price()

    assert "Brent" in result
    assert "WTI" in result
    assert len(calls) == 2


def test_get_oil_price_returns_error_when_all_sources_fail():
    with patch("api.index.app_runtime.cache.request", return_value=None):
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
    monkeypatch.setattr(index.app_runtime.config, "redis", lambda: Redis())
    monkeypatch.setattr(
        index.http_client,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    result = index.app_runtime.cache.request("https://example.test", None, None, 10)

    assert result["data"] == {"ok": True}
    assert calls


def test_get_dollar_rates_returns_formatted_snapshot(monkeypatch):
    from api import index

    class FakeCache:
        def get(self, **kwargs):
            return index.StaleCacheResult(value="cached dolar", status="fresh")

    monkeypatch.setattr(
        index.app_runtime.dollar,
        "get_snapshot_cache",
        lambda: FakeCache(),
    )
    monkeypatch.setattr(
        index.app_runtime.dollar,
        "build_rates_text",
        lambda *_args, **_kwargs: "live dolar",
    )

    assert index.app_runtime.dollar.get_rates("") == "cached dolar"


def test_get_dollar_rates_keeps_invalid_timeframe_without_cache(monkeypatch):
    from api import index

    called = False

    def fail_cache():
        nonlocal called
        called = True
        raise AssertionError("cache should not be used")

    monkeypatch.setattr(
        index.app_runtime.dollar,
        "get_snapshot_cache",
        fail_cache,
    )

    result = index.app_runtime.dollar.get_rates("2h")

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
