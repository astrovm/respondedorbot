from unittest.mock import MagicMock

from api.token_signals import (
    TokenAddress,
    TokenSignal,
    choose_best_pair,
    detect_token_address,
    format_signal_caption,
    fetch_signal,
    handle_token_signal_callback,
    handle_token_signal_message,
    render_signal_chart,
)


SOL_MINT = "J8PSdNP3QewKq2Z1JJJFDMaqF7KcaiJhR7gbr5KZpump"
EVM_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def _pair(address=SOL_MINT):
    return {
        "chainId": "solana",
        "url": "https://dexscreener.com/solana/pair",
        "pairAddress": "pair1",
        "baseToken": {"address": address, "name": "Tung Tung", "symbol": "TRIPLET"},
        "priceUsd": "0.0106",
        "priceChange": {"h1": 5.1, "h24": 2.3},
        "marketCap": 10580000,
        "volume": {"h24": 851700},
        "liquidity": {"usd": 246700},
        "txns": {"h1": {"buys": 122, "sells": 148}},
        "pairCreatedAt": 1710000000000,
    }


def _candles():
    return [
        [1, 1.0, 2.0, 0.8, 1.5, 1000],
        [2, 1.5, 2.5, 1.2, 1.3, 1000],
        [3, 1.3, 1.8, 1.0, 1.7, 1000],
    ]


def test_detect_token_address_accepts_solana_and_evm():
    assert detect_token_address(SOL_MINT) == TokenAddress(
        "solana", "solana", "SOL", SOL_MINT
    )
    assert detect_token_address(EVM_ADDRESS) == TokenAddress(
        "ethereum", "eth", "ETH", EVM_ADDRESS
    )


def test_detect_token_address_rejects_extra_words():
    assert detect_token_address(f"buy {SOL_MINT}") is None
    assert detect_token_address("not-a-token") is None


def test_choose_best_pair_uses_liquidity_then_volume():
    low = {"liquidity": {"usd": 10}, "volume": {"h24": 999}}
    high = {"liquidity": {"usd": 100}, "volume": {"h24": 1}}

    assert choose_best_pair([low, high]) is high


def test_format_signal_caption_contains_phanes_style_fields():
    signal = TokenSignal(
        TokenAddress("solana", "solana", "SOL", SOL_MINT),
        _pair(),
        _candles(),
    )

    caption = format_signal_caption(signal)

    assert "Tung Tung" in caption
    assert "$TRIPLET" in caption
    assert "#SOL" in caption
    assert "📊 <b>Stats</b>" in caption
    assert "DEF" in caption
    assert "DS" in caption
    assert "ATH   <b>$2.50B" in caption


def test_format_signal_caption_uses_candles_for_missing_pair_age():
    signal = TokenSignal(
        TokenAddress("solana", "solana", "SOL", SOL_MINT),
        {**_pair(), "pairCreatedAt": None},
        _candles(),
    )

    caption = format_signal_caption(signal)

    assert "#SOL | <i>" in caption
    assert "#SOL | <i>?</i>" not in caption


def test_render_signal_chart_returns_png():
    signal = TokenSignal(
        TokenAddress("solana", "solana", "SOL", SOL_MINT),
        _pair(),
        _candles(),
    )

    output = render_signal_chart(signal, width=420, height=300)

    assert output.startswith(b"\x89PNG")
    assert len(output) > 1000


def test_fetch_signal_uses_first_pair_with_candles(monkeypatch):
    import api.token_signals as token_signals

    pairs = [
        {"pairAddress": "bad", "liquidity": {"usd": 1000}, "volume": {"h24": 1}},
        {"pairAddress": "good", "liquidity": {"usd": 100}, "volume": {"h24": 1}},
    ]

    monkeypatch.setattr(token_signals, "fetch_token_pairs", lambda *_args: pairs)
    monkeypatch.setattr(
        token_signals,
        "fetch_ohlcv",
        lambda _redis, _token, pair_address: _candles() if pair_address == "good" else [],
    )

    signal = fetch_signal(
        MagicMock(),
        TokenAddress("ethereum", "eth", "ETH", EVM_ADDRESS),
    )

    assert signal is not None
    assert signal.pair["pairAddress"] == "good"
    assert len(signal.candles) == 3


def test_handle_token_signal_message_sends_photo_and_skips_ai(monkeypatch):
    import api.token_signals as token_signals

    redis_client = MagicMock()
    send_photo = MagicMock(return_value=55)
    monkeypatch.setattr(
        token_signals,
        "fetch_signal",
        lambda _redis, token: TokenSignal(token, _pair(), _candles()),
    )
    message = {
        "message_id": 10,
        "chat": {"id": 100, "type": "group"},
        "from": {"id": 7},
        "text": SOL_MINT,
    }

    handled = handle_token_signal_message(
        message,
        redis_client=redis_client,
        send_photo=send_photo,
        admin_report=MagicMock(),
    )

    assert handled is True
    send_photo.assert_called_once()
    assert send_photo.call_args.kwargs["msg_id"] == "10"
    assert redis_client.setex.called
    assert "source_message_id" in redis_client.setex.call_args.args[2]


def test_handle_token_signal_message_does_not_handle_failed_send(monkeypatch):
    import api.token_signals as token_signals

    monkeypatch.setattr(
        token_signals,
        "fetch_signal",
        lambda _redis, token: TokenSignal(token, _pair(), _candles()),
    )

    handled = handle_token_signal_message(
        {
            "message_id": 10,
            "chat": {"id": 100, "type": "group"},
            "from": {"id": 7},
            "text": SOL_MINT,
        },
        redis_client=MagicMock(),
        send_photo=MagicMock(return_value=None),
        admin_report=MagicMock(),
    )

    assert handled is False


def test_handle_token_signal_callback_deletes_for_requester():
    redis_client = MagicMock()
    redis_client.get.return_value = (
        '{"chat_id":"100","message_id":55,"requester_id":"7",'
        '"chain_id":"solana","network":"solana","tag":"SOL","address":"'
        + SOL_MINT
        + '"}'
    )
    delete_msg = MagicMock()
    answer = MagicMock()

    handled = handle_token_signal_callback(
        {
            "id": "cb1",
            "data": "sig:del:abc",
            "from": {"id": 7},
            "message": {"message_id": 55, "chat": {"id": 100, "type": "group"}},
        },
        redis_client=redis_client,
        delete_msg=delete_msg,
        send_photo=MagicMock(),
        is_chat_admin=MagicMock(return_value=False),
        answer_callback_query=answer,
        admin_report=MagicMock(),
    )

    assert handled is True
    delete_msg.assert_called_once_with("100", "55")
    answer.assert_called_once()


def test_handle_token_signal_callback_refresh_keeps_source_reply(monkeypatch):
    import api.token_signals as token_signals

    redis_client = MagicMock()
    redis_client.get.return_value = (
        '{"chat_id":"100","message_id":55,"source_message_id":"10",'
        '"requester_id":"7","chain_id":"solana","network":"solana",'
        '"tag":"SOL","address":"'
        + SOL_MINT
        + '"}'
    )
    send_photo = MagicMock(return_value=56)
    delete_msg = MagicMock()
    monkeypatch.setattr(
        token_signals,
        "fetch_signal",
        lambda _redis, token: TokenSignal(token, _pair(), _candles()),
    )

    handled = handle_token_signal_callback(
        {
            "id": "cb1",
            "data": "sig:ref:abc",
            "from": {"id": 7},
            "message": {"message_id": 55, "chat": {"id": 100, "type": "group"}},
        },
        redis_client=redis_client,
        delete_msg=delete_msg,
        send_photo=send_photo,
        is_chat_admin=MagicMock(return_value=False),
        answer_callback_query=MagicMock(),
        admin_report=MagicMock(),
    )

    assert handled is True
    assert send_photo.call_args.kwargs["msg_id"] == "10"
    delete_msg.assert_called_once_with("100", "55")
