from unittest.mock import MagicMock, call

from api.markets.crypto import get_prices


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
