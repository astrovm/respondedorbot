from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from html import escape
from typing import Any

import pycountry

from api.utils import fmt_num

EVENTS_URL = "https://gamma-api.polymarket.com/events"
MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
MIDPOINTS_URL = "https://clob.polymarket.com/midpoints"
GLOBAL_ELECTIONS_TAG = "global-elections"
GLOBAL_ELECTIONS_LIMIT = 10
WORLD_CUP_SERIES_ID = 11433
WORLD_CUP_LIMIT = 10
WORLD_CUP_FETCH_LIMIT = 100
WORLD_CUP_WINNER_SLUG = "world-cup-winner"
WORLD_CUP_WINNER_LIMIT = 5
COUNTRY_NAME_ALIASES = {
    "bosnia-herzegovina": "BA",
    "england": "GB",
    "ir iran": "IR",
    "korea republic": "KR",
    "scotland": "GB",
    "uk": "GB",
}

CachedRequest = Callable[..., dict[str, Any] | None]
LivePriceFetcher = Callable[[str], tuple[float, int | None] | None]
LivePricesFetcher = Callable[[Sequence[str]], dict[str, float]]
EventFetcher = Callable[[str], tuple[dict[str, Any], int | None] | None]
OutcomesGetter = Callable[..., list[tuple[str, float]]]
CountryFormatter = Callable[[str], str]
TimezoneFactory = Callable[[int], timezone]


def fetch_live_price(
    token_id: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> tuple[float, int | None] | None:
    if not token_id:
        return None

    response = cached_request(
        MIDPOINT_URL,
        {"token_id": token_id},
        None,
        cache_ttl,
        verify_ssl=False,
    )
    midpoint_data = response.get("data") if response else None
    if not isinstance(midpoint_data, dict):
        return None
    raw_midpoint = midpoint_data.get("mid")
    if not isinstance(raw_midpoint, (str, int, float)):
        return None
    try:
        price = float(raw_midpoint)
    except (TypeError, ValueError):
        return None
    return price, None


def fetch_live_prices(
    token_ids: Sequence[str],
    *,
    http_post: Callable[..., Any],
) -> dict[str, float]:
    unique_token_ids = list(dict.fromkeys(token_id for token_id in token_ids if token_id))
    if not unique_token_ids:
        return {}

    try:
        response = http_post(
            MIDPOINTS_URL,
            json=[{"token_id": token_id} for token_id in unique_token_ids],
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    prices: dict[str, float] = {}
    for token_id, raw_price in payload.items():
        if not isinstance(raw_price, (str, int, float)):
            continue
        try:
            prices[str(token_id)] = float(raw_price)
        except (TypeError, ValueError):
            continue
    return prices


def fetch_event(
    slug: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> tuple[dict[str, Any], int | None] | None:
    response = cached_request(
        EVENTS_URL,
        {"slug": slug},
        None,
        cache_ttl,
    )
    events = response.get("data") if response else None
    if not isinstance(events, list) or not events:
        return None
    assert response is not None
    timestamp = response.get("timestamp")
    return events[0], timestamp if isinstance(timestamp, int) else None


def format_event_section(
    event: dict[str, Any],
    header: str,
    filter_prefixes: Sequence[str],
    *,
    fetch_live: LivePriceFetcher,
) -> tuple[list[str], int | None] | None:
    odds: list[tuple[str, float]] = []
    latest_stream_timestamp: int | None = None

    for market in event.get("markets") or []:
        raw_outcomes = market.get("outcomes")
        raw_prices = market.get("outcomePrices")
        raw_token_ids = market.get("clobTokenIds")
        if not raw_outcomes or not raw_prices:
            continue
        try:
            outcomes = json.loads(raw_outcomes)
            prices = json.loads(raw_prices)
            token_ids = json.loads(raw_token_ids) if raw_token_ids else None
        except (TypeError, json.JSONDecodeError):
            continue
        if not outcomes or not prices:
            continue
        try:
            yes_index = outcomes.index("Yes")
        except ValueError:
            yes_index = 0
        if yes_index >= len(prices):
            continue

        yes_price: float | None = None
        yes_timestamp: int | None = None
        if token_ids and yes_index < len(token_ids):
            live = fetch_live(token_ids[yes_index])
            if live:
                yes_price, yes_timestamp = live
        if yes_price is None:
            try:
                yes_price = float(prices[yes_index])
            except (TypeError, ValueError):
                continue
        if yes_timestamp is not None:
            latest_stream_timestamp = max(
                latest_stream_timestamp or yes_timestamp,
                yes_timestamp,
            )

        title = (
            market.get("groupItemTitle")
            or market.get("question")
            or market.get("slug")
        )
        if title:
            odds.append((str(title), max(0.0, min(yes_price, 1.0)) * 100))

    if not odds:
        return None
    odds.sort(key=lambda item: item[1], reverse=True)
    filtered = [
        item
        for item in odds
        if any(
            item[0].strip().upper().startswith(prefix.upper())
            for prefix in filter_prefixes
        )
    ]
    lines = [header, ""]
    for title, probability in filtered or odds:
        decimals = 2 if probability < 10 else 1
        lines.append(f"- {title}: {fmt_num(probability, decimals)}%")
    return lines, latest_stream_timestamp


def event_top_outcomes(
    event: dict[str, Any],
    limit: int = 2,
    *,
    fetch_live: LivePriceFetcher | None = None,
) -> list[tuple[str, float]]:
    event_outcomes: list[tuple[str, float]] = []
    for market in event.get("markets") or []:
        if market.get("active") is False or market.get("closed") is True:
            continue
        try:
            outcomes = json.loads(market.get("outcomes") or "[]")
            prices = json.loads(market.get("outcomePrices") or "[]")
            yes_index = outcomes.index("Yes")
        except (
            AttributeError,
            TypeError,
            ValueError,
            IndexError,
            json.JSONDecodeError,
        ):
            continue
        probability: float | None = None
        if fetch_live:
            try:
                token_ids = json.loads(market.get("clobTokenIds") or "[]")
            except (TypeError, json.JSONDecodeError):
                token_ids = []
            if yes_index < len(token_ids):
                live = fetch_live(str(token_ids[yes_index]))
                if live:
                    probability = live[0] * 100
        if probability is None:
            try:
                probability = float(prices[yes_index]) * 100
            except (TypeError, ValueError, IndexError):
                continue
        title = (
            market.get("groupItemTitle")
            or market.get("question")
            or market.get("slug")
        )
        if title:
            event_outcomes.append(
                (str(title), max(0.0, min(probability, 100.0)))
            )
    event_outcomes.sort(key=lambda item: item[1], reverse=True)
    return event_outcomes[:limit]


def format_usd_compact(value: float) -> str:
    for divisor, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if value >= divisor:
            return f"US${fmt_num(value / divisor, 1)}{suffix}"
    return f"US${fmt_num(value, 0)}"


def country_flag(country_code: str) -> str:
    code = country_code.upper()
    if len(code) != 2 or not code.isalpha():
        return ""
    return "".join(chr(127397 + ord(char)) for char in code)


def country_code_from_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.replace("_", " ").strip()).casefold()
    alias = COUNTRY_NAME_ALIASES.get(normalized)
    if alias:
        return alias
    try:
        return str(pycountry.countries.lookup(normalized.replace("-", " ")).alpha_2)
    except LookupError:
        return ""


def event_country_flag(event: dict[str, Any]) -> str:
    for tag in event.get("tags") or []:
        code = (
            country_code_from_name(str(tag.get("slug") or ""))
            if isinstance(tag, dict)
            else ""
        )
        if code:
            return country_flag(code)
    return ""


def flagged_country_name(name: str) -> str:
    flag = country_flag(country_code_from_name(name))
    return f"{flag} {name}" if flag else name


def get_global_elections(
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
    get_top_outcomes: OutcomesGetter,
    fetch_live_prices: LivePricesFetcher,
    get_event_flag: Callable[[dict[str, Any]], str],
    format_liquidity: Callable[[float], str],
) -> str:
    response = cached_request(
        EVENTS_URL,
        {
            "limit": GLOBAL_ELECTIONS_LIMIT,
            "active": "true",
            "closed": "false",
            "tag_slug": GLOBAL_ELECTIONS_TAG,
            "order": "liquidity",
            "ascending": "false",
        },
        None,
        cache_ttl,
    )
    events = response.get("data") if response else None
    if not isinstance(events, list) or not events:
        return "No pude traer las elecciones desde Polymarket"

    events.sort(key=lambda event: float(event.get("liquidity") or 0), reverse=True)
    token_ids: list[str] = []
    for event in events[:GLOBAL_ELECTIONS_LIMIT]:
        for market in event.get("markets") or []:
            if market.get("active") is False or market.get("closed") is True:
                continue
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                market_token_ids = json.loads(market.get("clobTokenIds") or "[]")
                yes_index = outcomes.index("Yes")
            except (
                AttributeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if yes_index < len(market_token_ids):
                token_ids.append(str(market_token_ids[yes_index]))
    live_prices = fetch_live_prices(token_ids)

    def fetch_live(token_id: str) -> tuple[float, int | None] | None:
        price = live_prices.get(token_id)
        return (price, None) if price is not None else None

    lines = ["Polymarket - Global elections by liquidity"]
    for event in events[:GLOBAL_ELECTIONS_LIMIT]:
        title, slug = event.get("title"), event.get("slug")
        if not title or not slug:
            continue
        try:
            liquidity = float(event.get("liquidity") or 0)
        except (TypeError, ValueError):
            liquidity = 0
        outcomes = []
        for outcome_title, probability in get_top_outcomes(
            event,
            fetch_live=fetch_live,
        ):
            decimals = 2 if probability < 10 else 1
            outcomes.append(
                f"{escape(outcome_title)} {fmt_num(probability, decimals)}%"
            )
        details = [f"Liquidity {format_liquidity(liquidity)}"]
        end_date = str(event.get("endDate") or "")[:10]
        if end_date:
            details.append(f"Closes {end_date}")
        flag = get_event_flag(event)
        display_title = f"{flag} {title}" if flag else str(title)
        event_url = f"https://polymarket.com/event/{slug}"
        lines.extend(
            [
                "",
                f'<a href="{escape(event_url, quote=True)}">'
                f"{escape(display_title)}</a>",
            ]
        )
        if outcomes:
            lines.append(" | ".join(outcomes))
        lines.append(" | ".join(details))
    return (
        "\n".join(lines)
        if len(lines) > 1
        else "No pude traer las elecciones desde Polymarket"
    )


def get_world_cup_games(
    timezone_offset: int,
    *,
    fetch_winner_event: EventFetcher,
    cached_request: CachedRequest,
    cache_ttl: int,
    get_top_outcomes: OutcomesGetter,
    fetch_live: LivePriceFetcher,
    format_country: CountryFormatter,
    make_timezone: TimezoneFactory,
) -> str:
    winner_event = fetch_winner_event(WORLD_CUP_WINNER_SLUG)
    response = cached_request(
        EVENTS_URL,
        {
            "limit": WORLD_CUP_FETCH_LIMIT,
            "active": "true",
            "closed": "false",
            "series_id": WORLD_CUP_SERIES_ID,
            "order": "endDate",
            "ascending": "true",
        },
        None,
        cache_ttl,
    )
    events = response.get("data") if response else None
    if not isinstance(events, list) or not events:
        return "Could not fetch World Cup games from Polymarket"

    pattern = re.compile(r"^fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$")
    games = [
        event
        for event in events
        if pattern.fullmatch(str(event.get("slug") or ""))
    ]
    games.sort(key=lambda event: str(event.get("endDate") or ""))
    lines = ["Polymarket - World Cup"]

    if winner_event:
        event, _timestamp = winner_event
        winner_outcomes = []
        for title, probability in get_top_outcomes(
            event, limit=WORLD_CUP_WINNER_LIMIT
        ):
            decimals = 2 if probability < 10 else 1
            winner_outcomes.append(
                f"{escape(format_country(title))} "
                f"{fmt_num(probability, decimals)}%"
            )
        if winner_outcomes:
            lines.extend(
                [
                    "",
                    f'<a href="https://polymarket.com/event/{WORLD_CUP_WINNER_SLUG}">'
                    "World Cup Winner</a>",
                    " | ".join(winner_outcomes),
                ]
            )

    games_by_date: dict[str, list[tuple[str, str]]] = {}
    chat_timezone = make_timezone(timezone_offset)
    timezone_label = f"UTC{timezone_offset:+d}" if timezone_offset else "UTC"
    for event in games[:WORLD_CUP_LIMIT]:
        title, slug = event.get("title"), event.get("slug")
        if not title or not slug:
            continue
        outcomes = get_top_outcomes(event, limit=3, fetch_live=fetch_live)
        probabilities = dict(outcomes)
        favorite = outcomes[0][0] if outcomes else ""
        teams = []
        for team_name in [part.strip() for part in str(title).split(" vs. ")]:
            team_probability = probabilities.get(team_name)
            if team_probability is None:
                continue
            decimals = 2 if team_probability < 10 else 1
            label = (
                f"{format_country(team_name)} "
                f"{fmt_num(team_probability, decimals)}%"
            )
            teams.append(f"[{label}]" if team_name == favorite else label)

        event_url = f"https://polymarket.com/sports/world-cup/{slug}"
        linked_title = (
            f'<a href="{escape(event_url, quote=True)}">'
            f"{escape(' vs. '.join(teams))}</a>"
        )
        date_string, time_string = _format_kickoff(
            str(event.get("endDate") or ""),
            chat_timezone,
            timezone_label,
        )
        games_by_date.setdefault(date_string, []).append(
            (linked_title, time_string)
        )

    for date_string, daily_games in games_by_date.items():
        lines.extend([""] if date_string == "Unknown Date" else ["", date_string])
        for linked_title, time_string in daily_games:
            lines.append(linked_title)
            if time_string:
                lines.append(time_string)
            lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return (
        "\n".join(lines)
        if len(lines) > 1
        else "Could not fetch World Cup games from Polymarket"
    )


def _format_kickoff(
    end_date: str,
    chat_timezone: timezone,
    timezone_label: str,
) -> tuple[str, str]:
    try:
        kickoff = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        local = kickoff.astimezone(chat_timezone)
        return (
            local.strftime("%a, %B %d").replace(" 0", " "),
            local.strftime(f"%H:%M {timezone_label}"),
        )
    except ValueError:
        if end_date:
            return end_date[:10], end_date[11:16].replace("T", " ")
        return "Unknown Date", ""
