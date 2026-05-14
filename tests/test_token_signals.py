from unittest.mock import MagicMock

from api.token_signals import (
    TokenAddress,
    TokenSignal,
    choose_best_pair,
    choose_symbol_pair,
    detect_token_address,
    detect_token_symbol,
    fetch_pump_metadata,
    format_signal_caption,
    fetch_signal,
    fetch_signal_by_symbol,
    build_signal_keyboard,
    has_usable_chart,
    handle_token_signal_callback,
    handle_token_signal_message,
    render_or_fetch_signal_photo,
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


def _pump_meta():
    return {
        "image_uri": "https://example.com/token.png",
        "twitter": "https://x.com/token",
        "telegram": "https://t.me/token",
        "website": "https://token.test",
        "created_timestamp": 1710000000000,
        "real_token_reserves": 489_800_000_000_000,
        "total_supply": 999_900_000_000_000,
        "ath_market_cap": 51600,
        "ath_market_cap_timestamp": 1710100000000,
        "complete": False,
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


def test_detect_token_symbol_accepts_cashtag_only():
    assert detect_token_symbol("$glorp") == "glorp"
    assert detect_token_symbol("$GLORP") == "glorp"
    assert detect_token_symbol("buy $glorp") is None
    assert detect_token_symbol("glorp") is None


def test_choose_best_pair_uses_liquidity_then_volume():
    low = {"liquidity": {"usd": 10}, "volume": {"h24": 999}}
    high = {"liquidity": {"usd": 100}, "volume": {"h24": 1}}

    assert choose_best_pair([low, high]) is high


def test_choose_symbol_pair_prefers_exact_supported_symbol():
    unrelated = {
        "chainId": "solana",
        "baseToken": {"address": SOL_MINT, "symbol": "NOPE"},
        "liquidity": {"usd": 1000000},
    }
    glorp = {
        "chainId": "solana",
        "baseToken": {"address": "FkBF9u1upwEMUPxnXjcydxxVSxgr8f3k1YXbz7G7bmtA", "symbol": "glorp"},
        "liquidity": {"usd": 1000},
    }

    assert choose_symbol_pair([unrelated, glorp], "glorp") is glorp


def test_format_signal_caption_contains_phanes_style_fields():
    signal = TokenSignal(
        TokenAddress("solana", "solana", "SOL", SOL_MINT),
        _pair(),
        _candles(),
        supply=999_900_000,
    )

    caption = format_signal_caption(signal)

    assert "Tung Tung" in caption
    assert "$TRIPLET" in caption
    assert "J8P...pump" in caption
    assert f"<code>{SOL_MINT}</code>" not in caption
    assert "#SOL" in caption
    assert "📊 <b>Stats</b>" in caption
    assert "├ LP    <b>$123.3K</b>" in caption
    assert "├ Sup   <b>999.9M/999.9M</b>" in caption
    assert "DEF" in caption
    assert "DS" in caption
    assert "ATH   <b>$2.50B" in caption
    assert "TIP:" not in caption
    assert "Axiom!" not in caption
    assert "👁️" not in caption


def test_format_signal_caption_adds_pump_and_socials_when_real_data_exists():
    signal = TokenSignal(
        TokenAddress("solana", "solana", "SOL", SOL_MINT),
        _pair(),
        _candles(),
        supply=999_900_000,
        socials={"X": "https://x.com/token", "TG": "https://t.me/token", "Web": "https://token.test"},
        pump=_pump_meta(),
    )

    caption = format_signal_caption(signal)

    assert "#SOL (Pump @ 38%)" in caption
    assert "🔗 <b>Socials</b>" in caption
    assert "X" in caption
    assert "TG" in caption
    assert "Web" in caption
    assert "ATH   <b>$51.6K" in caption


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


def test_has_usable_chart_rejects_missing_and_flat_candles():
    token = TokenAddress("solana", "solana", "SOL", SOL_MINT)

    assert has_usable_chart(TokenSignal(token, _pair(), [])) is False
    assert has_usable_chart(TokenSignal(token, _pair(), [[1, 1, 1, 1, 1]] * 5)) is False
    assert has_usable_chart(TokenSignal(token, _pair(), _candles() + _candles())) is True


def test_render_or_fetch_signal_photo_uses_image_fallback(monkeypatch):
    import api.token_signals as token_signals

    token = TokenAddress("solana", "solana", "SOL", SOL_MINT)
    signal = TokenSignal(token, _pair(), [], token_image_url="https://example.com/img.png")
    monkeypatch.setattr(token_signals, "download_token_image", lambda _signal: b"image")

    assert render_or_fetch_signal_photo(signal) == b"image"


def test_download_token_image_normalizes_to_png(monkeypatch):
    import io
    from PIL import Image
    import api.token_signals as token_signals

    source = io.BytesIO()
    Image.new("RGB", (4, 4), "#ff0000").save(source, format="JPEG")

    class Response:
        content = source.getvalue()
        headers = {"content-type": "image/jpeg"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(token_signals.http_client, "get", lambda *_args, **_kwargs: Response())
    output = token_signals.download_token_image(
        TokenSignal(
            TokenAddress("solana", "solana", "SOL", SOL_MINT),
            _pair(),
            [],
            token_image_url="https://example.com/image.jpg",
        )
    )

    assert output is not None
    assert output.startswith(b"\x89PNG")


def test_fetch_pump_metadata_uses_api(monkeypatch):
    import api.token_signals as token_signals

    redis_client = MagicMock()
    redis_client.get.return_value = None
    monkeypatch.setattr(token_signals, "_json_get", lambda _url: _pump_meta())

    metadata = fetch_pump_metadata(
        redis_client,
        TokenAddress("solana", "solana", "SOL", "F3A1baCgv4TF79TSjdMTvpMDtNv8DJvHZwNc9DG8pump"),
    )

    assert metadata is not None
    assert metadata["twitter"] == "https://x.com/token"
    assert redis_client.setex.called


def test_build_signal_keyboard_uses_native_copy_text_button():
    token = TokenAddress("solana", "solana", "SOL", SOL_MINT)

    keyboard = build_signal_keyboard("abc", token, _pair())

    copy_button = keyboard["inline_keyboard"][0][2]
    assert copy_button == {"text": "📋", "copy_text": {"text": SOL_MINT}}


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


def test_fetch_signal_by_symbol_resolves_pair_token(monkeypatch):
    import api.token_signals as token_signals

    pair = {
        "chainId": "solana",
        "pairAddress": "pair",
        "baseToken": {
            "address": "FkBF9u1upwEMUPxnXjcydxxVSxgr8f3k1YXbz7G7bmtA",
            "name": "glorp",
            "symbol": "glorp",
        },
        "liquidity": {"usd": 52500},
        "volume": {"h24": 7400},
    }
    monkeypatch.setattr(token_signals, "search_token_pairs", lambda *_args: [pair])
    monkeypatch.setattr(token_signals, "fetch_ohlcv", lambda *_args: _candles())

    signal = fetch_signal_by_symbol(MagicMock(), "glorp")

    assert signal is not None
    assert signal.token.address == "FkBF9u1upwEMUPxnXjcydxxVSxgr8f3k1YXbz7G7bmtA"
    assert signal.pair is pair


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


def test_handle_token_signal_message_accepts_cashtag(monkeypatch):
    import api.token_signals as token_signals

    redis_client = MagicMock()
    send_photo = MagicMock(return_value=55)
    monkeypatch.setattr(
        token_signals,
        "fetch_signal_by_symbol",
        lambda _redis, symbol: TokenSignal(
            TokenAddress("solana", "solana", "SOL", SOL_MINT),
            _pair(),
            _candles(),
        ),
    )

    handled = handle_token_signal_message(
        {
            "message_id": 10,
            "chat": {"id": 100, "type": "group"},
            "from": {"id": 7},
            "text": "$glorp",
        },
        redis_client=redis_client,
        send_photo=send_photo,
        admin_report=MagicMock(),
    )

    assert handled is True
    send_photo.assert_called_once()


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
        edit_photo=MagicMock(),
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
    send_photo = MagicMock()
    edit_photo = MagicMock(return_value=True)
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
        edit_photo=edit_photo,
        is_chat_admin=MagicMock(return_value=False),
        answer_callback_query=MagicMock(),
        admin_report=MagicMock(),
    )

    assert handled is True
    send_photo.assert_not_called()
    edit_photo.assert_called_once()
    assert edit_photo.call_args.args[:2] == ("100", "55")
    delete_msg.assert_not_called()


def test_handle_token_signal_callback_refresh_uses_short_address(monkeypatch):
    import api.token_signals as token_signals

    redis_client = MagicMock()
    redis_client.get.return_value = (
        '{"chat_id":"100","message_id":55,"source_message_id":"10",'
        '"requester_id":"7","chain_id":"solana","network":"solana",'
        '"tag":"SOL","address":"'
        + SOL_MINT
        + '"}'
    )
    send_photo = MagicMock()
    edit_photo = MagicMock(return_value=True)
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
        delete_msg=MagicMock(),
        send_photo=send_photo,
        edit_photo=edit_photo,
        is_chat_admin=MagicMock(return_value=False),
        answer_callback_query=MagicMock(),
        admin_report=MagicMock(),
    )

    assert handled is True
    send_photo.assert_not_called()
    caption = edit_photo.call_args.kwargs["caption"]
    assert "J8P...pump" in caption
