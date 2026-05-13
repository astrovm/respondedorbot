from __future__ import annotations

import html
import io
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

from api.services import http_client
from api.services.redis_helpers import redis_get_json, redis_setex_json

SIGNAL_STATE_TTL = 3600
_SOLANA_ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}(?:pump)?$")
_EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


@dataclass(frozen=True)
class TokenAddress:
    chain_id: str
    network: str
    tag: str
    address: str


@dataclass(frozen=True)
class TokenSignal:
    token: TokenAddress
    pair: Mapping[str, Any]
    candles: Sequence[Sequence[float]]


def detect_token_address(text: str) -> Optional[TokenAddress]:
    candidate = (text or "").strip()
    if not candidate or any(char.isspace() for char in candidate):
        return None
    if _EVM_ADDRESS_RE.fullmatch(candidate):
        return TokenAddress("ethereum", "eth", "ETH", candidate.lower())
    if _SOLANA_ADDRESS_RE.fullmatch(candidate):
        return TokenAddress("solana", "solana", "SOL", candidate)
    return None


def signal_state_key(signal_id: str) -> str:
    return f"token_signal:{signal_id}"


def _json_get(url: str, *, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    response = http_client.get(url, params=params, timeout=8)
    response.raise_for_status()
    return response.json()


def _cache_get(redis_client: Any, key: str) -> Optional[Any]:
    return redis_get_json(redis_client, key)


def _cache_set(redis_client: Any, key: str, ttl: int, value: Any) -> None:
    redis_setex_json(redis_client, key, ttl, value)


def fetch_token_pairs(redis_client: Any, token: TokenAddress) -> List[Mapping[str, Any]]:
    cache_key = f"token_signal:pairs:{token.chain_id}:{token.address}"
    cached = _cache_get(redis_client, cache_key)
    if isinstance(cached, list):
        return [item for item in cached if isinstance(item, Mapping)]

    url = f"https://api.dexscreener.com/token-pairs/v1/{token.chain_id}/{token.address}"
    data = _json_get(url)
    pairs = data if isinstance(data, list) else []
    _cache_set(redis_client, cache_key, 30, pairs)
    return [item for item in pairs if isinstance(item, Mapping)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_best_pair(pairs: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not pairs:
        return None
    return max(
        pairs,
        key=lambda pair: (
            _as_float((pair.get("liquidity") or {}).get("usd")),
            _as_float((pair.get("volume") or {}).get("h24")),
        ),
    )


def fetch_ohlcv(redis_client: Any, token: TokenAddress, pair_address: str) -> List[List[float]]:
    cache_key = f"token_signal:ohlcv:{token.network}:{pair_address}:hour"
    cached = _cache_get(redis_client, cache_key)
    if isinstance(cached, list):
        return [item for item in cached if isinstance(item, list)]

    url = (
        "https://api.geckoterminal.com/api/v2/networks/"
        f"{token.network}/pools/{pair_address}/ohlcv/hour"
    )
    try:
        data = _json_get(
            url,
            params={"aggregate": 4, "limit": 60, "currency": "usd"},
        )
    except Exception:
        return []
    attributes = ((data or {}).get("data") or {}).get("attributes") or {}
    candles = attributes.get("ohlcv_list") or []
    _cache_set(redis_client, cache_key, 60, candles)
    return [item for item in candles if isinstance(item, list)]


def fetch_signal(redis_client: Any, token: TokenAddress) -> Optional[TokenSignal]:
    pairs = sorted(
        fetch_token_pairs(redis_client, token),
        key=lambda pair: (
            _as_float((pair.get("liquidity") or {}).get("usd")),
            _as_float((pair.get("volume") or {}).get("h24")),
        ),
        reverse=True,
    )
    if not pairs:
        return None

    fallback = pairs[0]
    for pair in pairs:
        pair_address = str(pair.get("pairAddress") or "")
        if not pair_address:
            continue
        candles = fetch_ohlcv(redis_client, token, pair_address)
        if candles:
            return TokenSignal(token=token, pair=pair, candles=candles)

    return TokenSignal(token=token, pair=fallback, candles=[])


def _fmt_money(value: Any, *, price: bool = False) -> str:
    number = _as_float(value)
    if price:
        if number >= 1:
            return f"${number:,.3f}".rstrip("0").rstrip(".")
        if number >= 0.01:
            return f"${number:,.4f}".rstrip("0").rstrip(".")
        return f"${number:.8f}".rstrip("0").rstrip(".")
    abs_number = abs(number)
    if abs_number >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B".replace(".00", "")
    if abs_number >= 1_000_000:
        return f"${number / 1_000_000:.2f}M".replace(".00", "")
    if abs_number >= 1_000:
        return f"${number / 1_000:.1f}K".replace(".0", "")
    return f"${number:,.0f}"


def _fmt_pct(value: Any) -> str:
    number = _as_float(value)
    sign = "+" if number >= 0 else ""
    if abs(number) < 0.05:
        return "+0%"
    return f"{sign}{number:.1f}%".replace(".0%", "%")


def _age_from_ms(value: Any) -> str:
    created_ms = _as_float(value)
    if created_ms <= 0:
        return "?"
    seconds = max(0, int(time.time() - created_ms / 1000))
    days = seconds // 86400
    if days >= 365:
        return f"{days // 365}y"
    if days >= 1:
        return f"{days}d"
    hours = seconds // 3600
    if hours >= 1:
        return f"{hours}h"
    return f"{max(1, seconds // 60)}m"


def _age_from_candles(candles: Sequence[Sequence[float]]) -> str:
    timestamps = [int(_as_float(candle[0])) for candle in candles if candle]
    if not timestamps:
        return "?"
    seconds = max(0, int(time.time() - min(timestamps)))
    days = seconds // 86400
    if days >= 365:
        return f"{days // 365}y"
    if days >= 1:
        return f"{days}d"
    hours = seconds // 3600
    if hours >= 1:
        return f"{hours}h"
    return f"{max(1, seconds // 60)}m"


def _short_address(token: TokenAddress) -> str:
    if token.chain_id == "ethereum":
        return f"{token.address[:4]}...{token.address[-4:]}"
    return token.address


def _txns(pair: Mapping[str, Any], key: str) -> Tuple[int, int]:
    bucket = (pair.get("txns") or {}).get(key) or {}
    return int(_as_float(bucket.get("buys"))), int(_as_float(bucket.get("sells")))


def _chain_explorer(token: TokenAddress) -> str:
    if token.chain_id == "ethereum":
        return f"https://etherscan.io/address/{token.address}"
    return f"https://solscan.io/token/{token.address}"


def _defined_url(token: TokenAddress) -> str:
    return f"https://www.defined.fi/{token.network}/{token.address}"


def _gt_url(token: TokenAddress, pair: Mapping[str, Any]) -> str:
    pair_address = str(pair.get("pairAddress") or token.address)
    return f"https://www.geckoterminal.com/{token.network}/pools/{pair_address}"


def _x_search_url(token: TokenAddress, symbol: str, pair: Mapping[str, Any]) -> str:
    pair_address = str(pair.get("pairAddress") or "")
    query = f"(${symbol} OR {token.address}"
    if pair_address:
        query += f" OR url:{pair_address}"
    query += ")"
    return f"https://x.com/search?f=live&q={quote(query)}&src=typed_query"


def _html_link(label: str, url: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def _link_rows(token: TokenAddress, pair: Mapping[str, Any], symbol: str) -> List[str]:
    ds_url = str(pair.get("url") or "")
    primary = [
        _html_link("DEF", _defined_url(token)),
        _html_link("DS", ds_url) if ds_url else "DS",
        _html_link("GT", _gt_url(token, pair)),
        _html_link("EXP", _chain_explorer(token)),
        _html_link("Xs", _x_search_url(token, symbol, pair)),
    ]
    if token.chain_id == "ethereum":
        trade = [
            _html_link("GM", f"https://gmgn.ai/eth/token/{token.address}"),
            _html_link("OKX", f"https://web3.okx.com/token/ethereum/{token.address}"),
            _html_link("PHO", f"https://photon.tinyastro.io/en/r/@respondedor/{token.address}"),
        ]
    else:
        trade = [
            _html_link("GM", f"https://gmgn.ai/sol/token/{token.address}"),
            _html_link("AXI", f"https://axiom.trade/t/{token.address}"),
            _html_link("TRO", f"https://t.me/menelaus_trojanbot?start={token.address}"),
            _html_link("BLO", f"https://t.me/BloomSolana_bot?start=ca_{token.address}"),
            _html_link("PHO", f"https://photon-sol.tinyastro.io/en/r/@respondedor/{token.address}"),
        ]
    return ["•".join(primary), "•".join(trade)]


def _ath(candles: Sequence[Sequence[float]]) -> Tuple[float, Optional[int]]:
    highs: List[Tuple[float, int]] = []
    for candle in candles:
        if len(candle) >= 4:
            highs.append((_as_float(candle[2]), int(_as_float(candle[0]))))
    if not highs:
        return 0.0, None
    return max(highs, key=lambda item: item[0])


def _ath_market_cap(
    candles: Sequence[Sequence[float]],
    *,
    current_price: float,
    current_market_cap: float,
) -> Tuple[float, Optional[int]]:
    ath_price, ath_ts = _ath(candles)
    if ath_price <= 0 or current_price <= 0 or current_market_cap <= 0:
        return ath_price, ath_ts
    return current_market_cap * (ath_price / current_price), ath_ts


def format_signal_caption(signal: TokenSignal) -> str:
    pair = signal.pair
    token = signal.token
    base = pair.get("baseToken") or {}
    name = str(base.get("name") or "Token")
    symbol = str(base.get("symbol") or "TOKEN").upper()
    price = pair.get("priceUsd")
    market_cap = pair.get("marketCap") or pair.get("fdv")
    volume = (pair.get("volume") or {}).get("h24")
    liquidity = (pair.get("liquidity") or {}).get("usd")
    price_change = (pair.get("priceChange") or {}).get("h24")
    one_hour = (pair.get("priceChange") or {}).get("h1")
    buys, sells = _txns(pair, "h1")
    current_price = _as_float(price)
    current_market_cap = _as_float(market_cap)
    ath_value, ath_ts = _ath_market_cap(
        signal.candles,
        current_price=current_price,
        current_market_cap=current_market_cap,
    )
    ath_line = "?"
    if ath_value > 0:
        drawdown_base = current_market_cap if current_market_cap > 0 else current_price
        drawdown = (
            ((drawdown_base - ath_value) / ath_value) * 100
            if drawdown_base
            else 0
        )
        age_days = ""
        if ath_ts:
            age = max(0, int(time.time() - ath_ts))
            age_days = f" / {max(1, age // 86400)}d"
        ath_line = f"{_fmt_money(ath_value)} ({_fmt_pct(drawdown)}{age_days})"

    age_text = _age_from_ms(pair.get("pairCreatedAt"))
    if age_text == "?":
        age_text = _age_from_candles(signal.candles)

    rows = [
        f"💊 <b>{html.escape(name)}</b> (${html.escape(symbol)})",
        f"├ <code>{html.escape(_short_address(token))}</code>",
        f"└ #{token.tag} | <i>{age_text}</i> | 👁️0",
        "",
        "📊 <b>Stats</b>",
        f"├ USD   <b>{_fmt_money(price, price=True)}</b> ({_fmt_pct(price_change)})",
        f"├ MC    <b>{_fmt_money(market_cap)}</b>",
        f"├ Vol   <b>{_fmt_money(volume)}</b>",
        f"├ LP    <b>{_fmt_money(liquidity)}</b>",
        f"├ 1H    <b>{_fmt_pct(one_hour)}</b> 🟩 {buys} 🟥 {sells}",
        f"└ ATH   <b>{ath_line}</b>",
        "",
        *_link_rows(token, pair, symbol),
    ]
    if token.chain_id == "solana":
        rows.extend(["", "⏏️ <b>TIP:</b> <i>Save fees on Axiom!</i>"])
    return "\n".join(rows)


def _font(size: int, bold: bool = False) -> Any:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_signal_chart(signal: TokenSignal, *, width: int = 1280, height: int = 900) -> bytes:
    image = Image.new("RGB", (width, height), "#070912")
    draw = ImageDraw.Draw(image)
    margin_left, margin_right, margin_top, margin_bottom = 56, 92, 74, 82
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    pair = signal.pair
    base = pair.get("baseToken") or {}
    symbol = str(base.get("symbol") or "TOKEN").upper()
    price = _fmt_money(pair.get("priceUsd"), price=True)
    change = _fmt_pct((pair.get("priceChange") or {}).get("h24"))
    mc = _fmt_money(pair.get("marketCap") or pair.get("fdv"))
    title = f"{symbol} (4H) Price: {price} ({change}) • MC: {mc}"
    draw.text((24, 22), title, fill="#dce7f4", font=_font(24, True))

    candles = [c for c in signal.candles if len(c) >= 5]
    candles = list(reversed(candles)) if candles and candles[0][0] > candles[-1][0] else candles
    if not candles:
        draw.text((width // 2 - 120, height // 2), "no chart data", fill="#8da1b6", font=_font(32, True))
    else:
        highs = [_as_float(c[2]) for c in candles]
        lows = [_as_float(c[3]) for c in candles]
        min_v, max_v = min(lows), max(highs)
        if math.isclose(min_v, max_v):
            min_v *= 0.99
            max_v *= 1.01
        pad = (max_v - min_v) * 0.08
        min_v -= pad
        max_v += pad

        def y_for(value: float) -> int:
            return int(margin_top + (max_v - value) / (max_v - min_v) * chart_h)

        for i in range(6):
            y = margin_top + int(chart_h * i / 5)
            draw.line((margin_left, y, width - margin_right, y), fill="#111827", width=1)
        for i in range(7):
            x = margin_left + int(chart_w * i / 6)
            draw.line((x, margin_top, x, height - margin_bottom), fill="#111827", width=1)

        step = chart_w / max(1, len(candles))
        body_w = max(4, int(step * 0.58))
        for idx, candle in enumerate(candles):
            _ts, open_v, high_v, low_v, close_v = [_as_float(v) for v in candle[:5]]
            x = int(margin_left + idx * step + step / 2)
            color = "#12b8a6" if close_v >= open_v else "#ff4356"
            draw.line((x, y_for(low_v), x, y_for(high_v)), fill=color, width=2)
            top = y_for(max(open_v, close_v))
            bottom = y_for(min(open_v, close_v))
            if top == bottom:
                bottom += 2
            draw.rectangle((x - body_w // 2, top, x + body_w // 2, bottom), fill=color)

        current = _as_float(pair.get("priceUsd")) or _as_float(candles[-1][4])
        current_y = y_for(current)
        draw.line((margin_left, current_y, width - margin_right, current_y), fill="#00b894", width=1)
        label_font = _font(18, True)
        label_bbox = draw.textbbox((0, 0), price, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_left = min(width - margin_right + 8, width - label_width - 20)
        draw.rectangle((label_left, current_y - 24, width - 8, current_y + 24), fill="#0e8f7d")
        draw.text((label_left + 6, current_y - 18), price, fill="#eafff9", font=label_font)

        ath_value, _ath_ts = _ath(candles)
        if ath_value:
            ath_y = y_for(ath_value)
            draw.text((width - margin_right - 170, max(margin_top, ath_y - 28)), f"{_fmt_money(ath_value, price=True)} ATH", fill="#36e0c3", font=_font(20, True))

    draw.rectangle((margin_left, margin_top, width - margin_right, height - margin_bottom), outline="#2a3442", width=2)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def build_signal_keyboard(signal_id: str, token: TokenAddress, pair: Mapping[str, Any]) -> Dict[str, Any]:
    copy_url = f"https://t.me/share/url?url={quote(token.address)}"
    ds_url = str(pair.get("url") or _defined_url(token))
    return {
        "inline_keyboard": [
            [
                {"text": "🗑", "callback_data": f"sig:del:{signal_id}"},
                {"text": "🔄", "callback_data": f"sig:ref:{signal_id}"},
                {"text": "📋", "url": copy_url},
                {"text": "DS", "url": ds_url},
            ]
        ]
    }


def handle_token_signal_message(
    message: Mapping[str, Any],
    *,
    redis_client: Any,
    send_photo: Callable[..., Optional[int]],
    admin_report: Callable[..., None],
) -> bool:
    token = detect_token_address(str(message.get("text") or ""))
    if token is None:
        return False
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    message_id = str(message.get("message_id") or "")
    requester_id = str((message.get("from") or {}).get("id") or "")
    if not chat_id or not message_id:
        return False
    try:
        signal = fetch_signal(redis_client, token)
        if signal is None:
            return False
        signal_id = uuid.uuid4().hex[:12]
        chart = render_signal_chart(signal)
        caption = format_signal_caption(signal)
        sent_id = send_photo(
            chat_id,
            chart,
            caption=caption,
            msg_id=message_id,
            reply_markup=build_signal_keyboard(signal_id, token, signal.pair),
        )
        if sent_id is None:
            return False
        redis_setex_json(
            redis_client,
            signal_state_key(signal_id),
            SIGNAL_STATE_TTL,
            {
                "chat_id": chat_id,
                "message_id": sent_id,
                "source_message_id": message_id,
                "requester_id": requester_id,
                "chain_id": token.chain_id,
                "network": token.network,
                "tag": token.tag,
                "address": token.address,
            },
        )
        return True
    except Exception as error:
        admin_report("token signal failed", error, {"chat_id": chat_id, "token": token.address})
        return False


def handle_token_signal_callback(
    callback_query: Mapping[str, Any],
    *,
    redis_client: Any,
    delete_msg: Callable[[str, str], None],
    send_photo: Callable[..., Optional[int]],
    is_chat_admin: Callable[..., bool],
    answer_callback_query: Callable[..., None],
    admin_report: Callable[..., None],
) -> bool:
    data = str(callback_query.get("data") or "")
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != "sig":
        return False
    action, signal_id = parts[1], parts[2]
    state = redis_get_json(redis_client, signal_state_key(signal_id))
    callback_id = str(callback_query.get("id") or "")
    if not isinstance(state, Mapping):
        answer_callback_query(callback_id, text="card vencida", show_alert=True)
        return True

    message = callback_query.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or state.get("chat_id") or "")
    message_id = str(message.get("message_id") or state.get("message_id") or "")
    user_id = str((callback_query.get("from") or {}).get("id") or "")
    chat_type = str(chat.get("type") or "")
    allowed = user_id == str(state.get("requester_id") or "")
    if not allowed and chat_type in {"group", "supergroup"}:
        allowed = bool(is_chat_admin(chat_id, user_id, redis_client=redis_client))
    if not allowed:
        answer_callback_query(callback_id, text="solo quien lo pidió o admin", show_alert=True)
        return True

    if action == "del":
        delete_msg(chat_id, message_id)
        answer_callback_query(callback_id, text="borrado")
        return True
    if action != "ref":
        answer_callback_query(callback_id)
        return True

    token = TokenAddress(
        str(state.get("chain_id") or ""),
        str(state.get("network") or ""),
        str(state.get("tag") or ""),
        str(state.get("address") or ""),
    )
    try:
        signal = fetch_signal(redis_client, token)
        if signal is None:
            answer_callback_query(callback_id, text="sin datos", show_alert=True)
            return True
        sent_id = send_photo(
            chat_id,
            render_signal_chart(signal),
            caption=format_signal_caption(signal),
            msg_id=str(state.get("source_message_id") or ""),
            reply_markup=build_signal_keyboard(signal_id, token, signal.pair),
        )
        if sent_id is None:
            answer_callback_query(callback_id, text="falló refresh", show_alert=True)
            return True
        delete_msg(chat_id, message_id)
        new_state = dict(state)
        new_state["message_id"] = sent_id
        redis_setex_json(redis_client, signal_state_key(signal_id), SIGNAL_STATE_TTL, new_state)
        answer_callback_query(callback_id, text="refrescado")
        return True
    except Exception as error:
        admin_report("token signal refresh failed", error, {"chat_id": chat_id, "signal_id": signal_id})
        answer_callback_query(callback_id, text="falló refresh", show_alert=True)
        return True
