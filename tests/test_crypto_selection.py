from unittest.mock import MagicMock, call

from api.markets.crypto import get_prices
from api.markets.stocks import StockQuote


CHANGE_FIELDS = {"24h": "percent_change_24h"}


def _coin(symbol, price, *, coin_id):
    return {
        "id": coin_id,
        "symbol": symbol,
        "name": symbol,
        "slug": symbol.lower(),
        "quote": {
            "USD": {"price": price, "percent_change_24h": 0},
            "BTC": {"price": price, "percent_change_24h": 0},
        },
    }


def test_missing_quotes_are_fetched_in_two_batches():
    fetch_prices = MagicMock(return_value={"data": []})
    fetch_quotes = MagicMock(
        side_effect=[
            {"1": _coin("ZZZ", 1.0, coin_id=1)},
            {"2": _coin("YYY", 2.0, coin_id=2)},
        ]
    )

    result = get_prices(
        "zzz,yyy",
        change_fields=CHANGE_FIELDS,
        fetch_prices=fetch_prices,
        fetch_quotes=fetch_quotes,
    )

    assert result is not None
    assert "ZZZ:" in result
    assert "YYY:" in result
    assert fetch_quotes.call_args_list == [
        call(["ZZZ", "YYY"], "USD"),
        call(["yyy"], "USD", by_slug=True),
    ]


def test_space_separated_symbols_select_multiple_assets():
    coins = [_coin("AAA", 1.0, coin_id=1), _coin("BBB", 2.0, coin_id=2)]

    result = get_prices(
        "aaa bbb",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": coins}),
        fetch_quotes=MagicMock(),
    )

    assert result is not None
    assert "AAA:" in result
    assert "BBB:" in result


def test_missing_asset_is_fetched_when_another_asset_is_listed():
    listed = [_coin("AAA", 1.0, coin_id=1)]
    fetched = _coin("ZZZ", 2.0, coin_id=2)
    fetch_quotes = MagicMock(return_value={"2": fetched})

    result = get_prices(
        "aaa zzz",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": listed}),
        fetch_quotes=fetch_quotes,
    )

    assert result is not None
    assert "AAA:" in result
    assert "ZZZ:" in result
    fetch_quotes.assert_called_once_with(["ZZZ"], "USD")


def test_unresolved_asset_is_reported_with_partial_results():
    listed = [_coin("AAA", 1.0, coin_id=1)]
    fetch_quotes = MagicMock(return_value={})

    result = get_prices(
        "aaa zzz",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": listed}),
        fetch_quotes=fetch_quotes,
    )

    assert result is not None
    assert "AAA:" in result
    assert "no encontré estos activos: ZZZ" in result
    assert fetch_quotes.call_args_list == [
        call(["ZZZ"], "USD"),
        call(["zzz"], "USD", by_slug=True),
    ]


def test_satoshi_formatting_does_not_mutate_cached_quote():
    coin = _coin("BTC", 1.0, coin_id=1)

    result = get_prices(
        "btc in sats",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": [coin]}),
        fetch_quotes=MagicMock(),
    )

    assert result is not None
    assert "100000000 SATS" in result
    assert coin["quote"]["BTC"]["price"] == 1.0


def _stock(symbol: str, price: float = 123.45) -> StockQuote:
    return StockQuote(symbol, symbol, price, "USD", "Synthetic", 1.25)


def test_unified_prices_fall_back_to_stock_quote():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "nvda",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: 123.45 USD (+1.25% 24h)"
    lookup_stocks.assert_called_once_with("NVDA")


def test_unified_prices_accept_stock_cashtag():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "$NVDA",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: 123.45 USD (+1.25% 24h)"
    lookup_stocks.assert_called_once_with("NVDA")


def test_crypto_prices_accept_crypto_cashtag():
    result = get_prices(
        "$BTC",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": [_coin("BTC", 50000, coin_id=1)]}),
        fetch_quotes=MagicMock(),
    )

    assert result is not None
    assert "BTC: 50000 USD" in result


def test_crypto_amount_conversion_accepts_cashtags():
    result = get_prices(
        "2 $BTC in $USD",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": [_coin("BTC", 50000, coin_id=1)]}),
        fetch_quotes=MagicMock(),
    )

    assert result == "2 BTC = 100000 USD"


def test_unified_prices_support_mixed_crypto_and_stocks():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "btc nvda",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": [_coin("BTC", 50000, coin_id=1)]}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result is not None
    assert "BTC: 50000 USD" in result
    assert "NVDA: 123.45 USD" in result
    lookup_stocks.assert_called_once_with("NVDA")


def test_unified_prices_resolve_company_name_as_one_stock_query():
    quote = _stock("MELI")
    lookup_stocks = MagicMock(return_value=[("Mercado Libre", quote)])

    result = get_prices(
        "Mercado Libre",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "MELI: 123.45 USD (+1.25% 24h)"
    lookup_stocks.assert_called_once_with("Mercado Libre")


def test_unified_prices_preserve_company_name_in_mixed_comma_query():
    lookup_stocks = MagicMock(return_value=[("Mercado Libre", _stock("MELI"))])

    result = get_prices(
        "btc, Mercado Libre",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": [_coin("BTC", 50000, coin_id=1)]}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result is not None
    assert "BTC: 50000 USD" in result
    assert "MELI: 123.45 USD" in result
    lookup_stocks.assert_called_once_with("Mercado Libre")


def test_stock_scope_bypasses_crypto_for_symbol_collisions():
    fetch_prices = MagicMock()
    lookup_stocks = MagicMock(return_value=[("META", _stock("META"))])

    result = get_prices(
        "stock:META",
        change_fields=CHANGE_FIELDS,
        fetch_prices=fetch_prices,
        fetch_quotes=MagicMock(),
        lookup_stocks=lookup_stocks,
    )

    assert result == "META: 123.45 USD (+1.25% 24h)"
    fetch_prices.assert_not_called()


def test_stock_scope_reports_partial_missing_symbols():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA")), ("ZZZ", None)])

    result = get_prices(
        "stock:NVDA,ZZZ",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(),
        fetch_quotes=MagicMock(),
        lookup_stocks=lookup_stocks,
    )

    assert result == ("NVDA: 123.45 USD (+1.25% 24h)\nno encontré estos activos: ZZZ")


def test_stock_scope_rejects_currency_conversion():
    lookup_stocks = MagicMock()

    result = get_prices(
        "stock:NVDA in EUR",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(),
        fetch_quotes=MagicMock(),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: las acciones solo soportan moneda nativa y variación 24h"
    lookup_stocks.assert_not_called()


def test_unified_stock_fallback_rejects_currency_conversion():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "NVDA in EUR",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: las acciones solo soportan moneda nativa y variación 24h"


def test_unified_stock_fallback_accepts_explicit_24h_timeframe():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "NVDA 24h",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: 123.45 USD (+1.25% 24h)"


def test_unified_stock_fallback_rejects_non_24h_timeframe():
    lookup_stocks = MagicMock(return_value=[("NVDA", _stock("NVDA"))])

    result = get_prices(
        "NVDA 7d",
        change_fields={**CHANGE_FIELDS, "7d": "percent_change_7d"},
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "NVDA: las acciones solo soportan moneda nativa y variación 24h"


def test_crypto_scope_does_not_fall_back_to_stock():
    lookup_stocks = MagicMock()

    result = get_prices(
        "crypto:META",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value={"data": []}),
        fetch_quotes=MagicMock(return_value={}),
        lookup_stocks=lookup_stocks,
    )

    assert result == "no encontré estos activos: META"
    lookup_stocks.assert_not_called()


def test_stock_fallback_survives_crypto_provider_failure():
    result = get_prices(
        "nvda",
        change_fields=CHANGE_FIELDS,
        fetch_prices=MagicMock(return_value=None),
        fetch_quotes=MagicMock(),
        lookup_stocks=MagicMock(return_value=[("nvda", _stock("NVDA"))]),
    )

    assert result == "NVDA: 123.45 USD (+1.25% 24h)"
