"""Fetch and format Polymarket events and prices."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Any

import pycountry

from api.cache.service import CacheService
from api.markets.world_cup_goals import (
    MatchScore,
    TEAM_NAME_ALIASES,
    WORLD_CUP_TEAM_RANKING,
    canonical_team_name,
    fetch_scoreboard_scores,
    WorldCupScoreboard,
    team_name_es,
)
from api.services import http_client
from api.utils import fmt_num

EVENTS_URL = "https://gamma-api.polymarket.com/events"
MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
MIDPOINTS_URL = "https://clob.polymarket.com/midpoints"
GLOBAL_ELECTIONS_TAG = "global-elections"
GLOBAL_ELECTIONS_LIMIT = 10
WORLD_CUP_SERIES_ID = 11433
WORLD_CUP_LIMIT = 10
WORLD_CUP_CANDIDATE_LIMIT = 30
WORLD_CUP_TEAM_SCOREBOARD_DAYS_BEFORE = 30
WORLD_CUP_TEAM_SCOREBOARD_DAYS_AFTER = 40
WORLD_CUP_TEAM_SCOREBOARD_LIMIT = 300
WORLD_CUP_FETCH_LIMIT = 100
WORLD_CUP_FETCH_MAX_PAGES = 20
WORLD_CUP_WINNER_SLUG = "world-cup-winner"
WORLD_CUP_WINNER_LIMIT = 5
WORLD_CUP_MATCH_MARKET_MIN_EDGE = 5.0
WORLD_CUP_GAME_SLUG_PATTERN = re.compile(
    r"^fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$"
)
WORLD_CUP_SCORE_TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z"
)
WORLD_CUP_MINUTE_TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z")
WORLD_CUP_WINNER_PLACEHOLDER_PATTERN = re.compile(
    r"^(Round of 32|Round of 16|Quarterfinal|Semifinal) (\d+) Winner$"
)
WORLD_CUP_OFFICIAL_MATCH_BY_ESPN_EVENT_ID = {
    "760486": 73,
    "760489": 74,
    "760488": 75,
    "760487": 76,
    "760492": 77,
    "760490": 78,
    "760491": 79,
    "760495": 80,
    "760494": 81,
    "760493": 82,
    "760496": 83,
    "760497": 84,
    "760498": 85,
    "760500": 86,
    "760501": 87,
    "760499": 88,
    "760503": 89,
    "760502": 90,
    "760504": 91,
    "760505": 92,
    "760506": 93,
    "760507": 94,
    "760509": 95,
    "760508": 96,
    "760510": 97,
    "760511": 98,
    "760512": 99,
    "760513": 100,
    "760514": 101,
    "760515": 102,
    "760517": 104,
}
WORLD_CUP_OFFICIAL_MATCH_SOURCES = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}
SPANISH_WEEKDAYS = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")
SPANISH_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
COUNTRY_NAME_ALIASES = {
    "bosnia-herzegovina": "BA",
    "cabo verde": "CV",
    "cape verde": "CV",
    "congo dr": "CD",
    "cote d'ivoire": "CI",
    "dr congo": "CD",
    "england": "GB",
    "ir iran": "IR",
    "ivory coast": "CI",
    "korea republic": "KR",
    "scotland": "GB",
    "turkey": "TR",
    "turkiye": "TR",
    "uk": "GB",
    "wales": "GB",
}
REGIONAL_COUNTRY_FLAGS = {
    "england": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "scotland": "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "wales": "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f",
}

CachedRequest = Callable[..., dict[str, Any] | None]
LivePriceFetcher = Callable[[str], tuple[float, int | None] | None]
LivePricesFetcher = Callable[[Sequence[str]], dict[str, float]]
EventFetcher = Callable[[str], tuple[dict[str, Any], int | None] | None]
CountryFormatter = Callable[[str], str]
TimezoneFactory = Callable[[int], timezone]
ScoreFetcher = Callable[[], Mapping[str, MatchScore]]


@dataclass(frozen=True)
class WorldCupSelectedMatch:
    match: MatchScore
    projected_team: str = ""
    projection_token: str = ""
    projection_predicted: bool = False
    token_predictions: Mapping[str, str] | None = None


@dataclass(frozen=True)
class WorldCupPrediction:
    team: str
    source: str


@dataclass(frozen=True)
class MarketQuote:
    title: str
    probability: float
    token_id: str | None


def _decode_market_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return loaded if isinstance(loaded, list) else []


def normalize_event_quotes(
    event: dict[str, Any],
    *,
    active_only: bool = True,
    default_first_outcome: bool = False,
) -> tuple[MarketQuote, ...]:
    quotes: list[MarketQuote] = []
    for market in event.get("markets") or []:
        if not isinstance(market, dict):
            continue
        if active_only and (
            market.get("active") is False or market.get("closed") is True
        ):
            continue
        outcomes = _decode_market_list(market.get("outcomes"))
        prices = _decode_market_list(market.get("outcomePrices"))
        token_ids = _decode_market_list(market.get("clobTokenIds"))
        try:
            yes_index = outcomes.index("Yes")
        except ValueError:
            if not default_first_outcome:
                continue
            yes_index = 0
        if yes_index >= len(prices):
            continue
        try:
            probability = float(prices[yes_index])
        except (TypeError, ValueError):
            continue
        title = (
            market.get("groupItemTitle")
            or market.get("question")
            or market.get("slug")
        )
        if not title:
            continue
        token_id = (
            str(token_ids[yes_index])
            if yes_index < len(token_ids) and token_ids[yes_index]
            else None
        )
        quotes.append(
            MarketQuote(
                title=str(title),
                probability=max(0.0, min(probability, 1.0)),
                token_id=token_id,
            )
        )
    return tuple(quotes)


def _top_outcomes(
    quotes: Sequence[MarketQuote],
    *,
    limit: int,
    fetch_live: LivePriceFetcher | None = None,
) -> list[tuple[str, float]]:
    outcomes: list[tuple[str, float]] = []
    for quote in quotes:
        probability = quote.probability
        if fetch_live and quote.token_id:
            live = fetch_live(quote.token_id)
            if live:
                probability = max(0.0, min(live[0], 1.0))
        outcomes.append((quote.title, probability * 100))
    outcomes.sort(key=lambda item: item[1], reverse=True)
    return outcomes[:limit]


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

    quotes = normalize_event_quotes(
        event,
        active_only=False,
        default_first_outcome=True,
    )
    for quote in quotes:
        probability = quote.probability
        yes_timestamp: int | None = None
        if quote.token_id:
            live = fetch_live(quote.token_id)
            if live:
                probability, yes_timestamp = live
        if yes_timestamp is not None:
            latest_stream_timestamp = max(
                latest_stream_timestamp or yes_timestamp,
                yes_timestamp,
            )
        odds.append(
            (quote.title, max(0.0, min(probability, 1.0)) * 100)
        )

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
    return _top_outcomes(
        normalize_event_quotes(event),
        limit=limit,
        fetch_live=fetch_live,
    )


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


def country_flag_from_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.replace("_", " ").strip()).casefold()
    regional_flag = REGIONAL_COUNTRY_FLAGS.get(normalized)
    if regional_flag:
        return regional_flag
    return country_flag(country_code_from_name(name))


def event_country_flag(event: dict[str, Any]) -> str:
    for tag in event.get("tags") or []:
        flag = (
            country_flag_from_name(str(tag.get("slug") or ""))
            if isinstance(tag, dict)
            else ""
        )
        if flag:
            return flag
    return ""


def flagged_country_name(name: str) -> str:
    flag = country_flag_from_name(name)
    display_name = team_name_es(name)
    return f"{flag} {display_name}" if flag else display_name


def get_global_elections(
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
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
    selected_events = events[:GLOBAL_ELECTIONS_LIMIT]
    normalized_events = [
        (event, normalize_event_quotes(event))
        for event in selected_events
    ]
    token_ids = [
        quote.token_id
        for _event, quotes in normalized_events
        for quote in quotes
        if quote.token_id
    ]
    live_prices = fetch_live_prices(token_ids)

    def fetch_live(token_id: str) -> tuple[float, int | None] | None:
        price = live_prices.get(token_id)
        return (price, None) if price is not None else None

    lines = ["Polymarket - Global elections by liquidity"]
    for event, quotes in normalized_events:
        title, slug = event.get("title"), event.get("slug")
        if not title or not slug:
            continue
        try:
            liquidity = float(event.get("liquidity") or 0)
        except (TypeError, ValueError):
            liquidity = 0
        outcomes = []
        for outcome_title, probability in _top_outcomes(
            quotes,
            limit=2,
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
    fetch_live: LivePriceFetcher,
    format_country: CountryFormatter,
    make_timezone: TimezoneFactory,
    fetch_scores: ScoreFetcher,
    team_query: str = "",
) -> str:
    selected_team = _world_cup_team_from_query(team_query)
    if team_query and not selected_team:
        return "No encontré ese país en el Mundial"
    winner_event = fetch_winner_event(WORLD_CUP_WINNER_SLUG)
    try:
        live_scores = fetch_scores()
    except Exception:
        live_scores = {}
    match_scores, selected_matches = _initial_world_cup_match_selection(
        live_scores,
        selected_team=selected_team,
    )
    events = _fetch_available_world_cup_events(
        cached_request=cached_request,
        cache_ttl=cache_ttl,
        match_scores=match_scores,
    )
    if not winner_event and not events and not live_scores:
        return "No pude traer los partidos del Mundial desde Polymarket"

    games = [event for event in events if _is_world_cup_game_event(event)]
    games.sort(key=lambda event: str(event.get("endDate") or ""))
    events_by_match = _world_cup_events_by_match(games)
    if selected_team:
        selected_matches = _select_world_cup_matches(
            live_scores,
            team_query=team_query,
            events_by_match=events_by_match,
            fetch_live=fetch_live,
            winner_event=winner_event,
        )
    lines = _world_cup_header_lines(
        selected_team=selected_team,
        winner_event=winner_event,
        format_country=format_country,
    )

    games_by_date: dict[str, list[tuple[str, str]]] = {}
    chat_timezone = make_timezone(timezone_offset)
    timezone_label = f"UTC{timezone_offset:+d}" if timezone_offset else "UTC"
    rendered_games = 0
    for selected in selected_matches:
        match = selected.match
        formatted_event = _format_world_cup_game(
            match,
            event=events_by_match.get(_match_key(match.home_team, match.away_team)),
            fetch_live=fetch_live,
            format_country=format_country,
            chat_timezone=chat_timezone,
            timezone_label=timezone_label,
            projected_team=selected.projected_team,
            projection_token=selected.projection_token,
            projection_predicted=selected.projection_predicted,
            token_predictions=selected.token_predictions or {},
        )
        if formatted_event is None:
            continue
        date_string, linked_title, time_string = formatted_event
        games_by_date.setdefault(date_string, []).append((linked_title, time_string))
        rendered_games += 1
        if rendered_games >= WORLD_CUP_LIMIT:
            break

    for date_string, daily_games in games_by_date.items():
        lines.extend([""] if date_string == "Fecha desconocida" else ["", date_string])
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
        else "No pude traer los partidos del Mundial desde Polymarket"
    )


def _fetch_available_world_cup_events(
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
    match_scores: Sequence[MatchScore],
) -> list[dict[str, Any]]:
    if not match_scores:
        return []
    return _fetch_world_cup_events(
        cached_request=cached_request,
        cache_ttl=cache_ttl,
        selected_matches=match_scores,
    )


def _initial_world_cup_match_selection(
    live_scores: Mapping[str, MatchScore],
    *,
    selected_team: str,
) -> tuple[list[MatchScore], list[WorldCupSelectedMatch]]:
    if selected_team:
        return sorted(live_scores.values(), key=lambda match: match.start_time), []
    selected_matches = _select_world_cup_matches(live_scores)
    return [selected.match for selected in selected_matches], selected_matches


def _world_cup_header_lines(
    *,
    selected_team: str,
    winner_event: tuple[dict[str, Any], int | None] | None,
    format_country: CountryFormatter,
) -> list[str]:
    title = "Polymarket - Mundial"
    if selected_team:
        title = f"Polymarket - Mundial: {team_name_es(selected_team)}"
    lines = [title]
    if winner_event and not selected_team:
        lines.extend(_format_world_cup_winner_lines(winner_event, format_country))
    return lines


def _format_world_cup_winner_lines(
    winner_event: tuple[dict[str, Any], int | None],
    format_country: CountryFormatter,
) -> list[str]:
    event, _timestamp = winner_event
    winner_outcomes = []
    for title, probability in _top_outcomes(
        normalize_event_quotes(event),
        limit=WORLD_CUP_WINNER_LIMIT,
    ):
        decimals = 2 if probability < 10 else 1
        winner_outcomes.append(
            f"{escape(format_country(title))} "
            f"{fmt_num(probability, decimals)}%"
        )
    if not winner_outcomes:
        return []
    return [
        "",
        f'<a href="https://polymarket.com/event/{WORLD_CUP_WINNER_SLUG}">'
        "Campeón del Mundial</a>",
        " | ".join(winner_outcomes),
    ]


def _fetch_world_cup_events(
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
    selected_matches: Sequence[MatchScore],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    fetch_start, fetch_end = _world_cup_fetch_bounds(selected_matches)
    target_keys = {
        _match_key(match.home_team, match.away_team)
        for match in selected_matches
        if match.state != "post"
    }
    found_keys: set[frozenset[str]] = set()
    for page in range(WORLD_CUP_FETCH_MAX_PAGES):
        parameters: dict[str, Any] = {
            "limit": WORLD_CUP_FETCH_LIMIT,
            "offset": page * WORLD_CUP_FETCH_LIMIT,
            "active": "true",
            "series_id": WORLD_CUP_SERIES_ID,
            "order": "endDate",
            "ascending": "true",
        }
        if fetch_start:
            parameters["end_date_min"] = fetch_start
        if fetch_end:
            parameters["end_date_max"] = fetch_end
        response = cached_request(
            EVENTS_URL,
            parameters,
            None,
            cache_ttl,
        )
        page_events = response.get("data") if response else None
        if not isinstance(page_events, list) or not page_events:
            break
        for event in page_events:
            if not isinstance(event, dict):
                continue
            events.append(event)
            if _is_world_cup_game_event(event):
                match_key = _world_cup_event_match_key(event)
                if match_key is not None:
                    found_keys.add(match_key)
        if target_keys and target_keys <= found_keys:
            break
        if len(page_events) < WORLD_CUP_FETCH_LIMIT:
            break
    return events


def _world_cup_fetch_bounds(
    matches: Sequence[MatchScore],
) -> tuple[str | None, str | None]:
    start_times = [
        _normalize_world_cup_fetch_time(match.start_time)
        for match in matches
        if WORLD_CUP_SCORE_TIME_PATTERN.fullmatch(match.start_time)
    ]
    if not start_times:
        return None, None
    return min(start_times), max(start_times)


def _normalize_world_cup_fetch_time(start_time: str) -> str:
    if WORLD_CUP_MINUTE_TIME_PATTERN.fullmatch(start_time):
        return start_time[:-1] + ":00Z"
    return start_time


def _normalize_team_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", ascii_text.casefold())


def _world_cup_team_from_query(query: str) -> str:
    normalized = _normalize_team_query(query)
    if not normalized:
        return ""
    candidates: dict[str, str] = {}
    for team in WORLD_CUP_TEAM_RANKING:
        candidates[_normalize_team_query(team)] = team
        candidates[_normalize_team_query(team_name_es(team))] = team
    for alias, canonical in TEAM_NAME_ALIASES.items():
        candidates[_normalize_team_query(alias)] = canonical
    return candidates.get(normalized, "")


def _is_world_cup_game_event(event: Mapping[str, Any]) -> bool:
    return WORLD_CUP_GAME_SLUG_PATTERN.fullmatch(str(event.get("slug") or "")) is not None


def _select_world_cup_matches(
    scores: Mapping[str, MatchScore],
    *,
    team_query: str = "",
    events_by_match: Mapping[frozenset[str], dict[str, Any]] | None = None,
    fetch_live: LivePriceFetcher | None = None,
    winner_event: tuple[dict[str, Any], int | None] | None = None,
) -> list[WorldCupSelectedMatch]:
    selected_team = _world_cup_team_from_query(team_query)
    if selected_team:
        return _select_team_world_cup_matches(
            scores,
            selected_team,
            events_by_match=events_by_match or {},
            fetch_live=fetch_live,
            winner_event=winner_event,
        )

    matches = sorted(scores.values(), key=lambda match: match.start_time)
    finished = [match for match in matches if match.state == "post"]
    active_or_future = [match for match in matches if match.state != "post"]
    return [
        WorldCupSelectedMatch(match)
        for match in finished[-2:] + active_or_future[:WORLD_CUP_CANDIDATE_LIMIT]
    ]


def _select_team_world_cup_matches(
    scores: Mapping[str, MatchScore],
    selected_team: str,
    *,
    events_by_match: Mapping[frozenset[str], dict[str, Any]],
    fetch_live: LivePriceFetcher | None,
    winner_event: tuple[dict[str, Any], int | None] | None,
) -> list[WorldCupSelectedMatch]:
    team_key = _score_key(selected_team)
    matches = sorted(scores.values(), key=lambda match: match.start_time)
    round_winner_tokens = _world_cup_round_winner_tokens(matches)
    predicted_winners = _world_cup_predicted_winners(
        matches,
        events_by_match=events_by_match,
        fetch_live=fetch_live,
        winner_event=winner_event,
    )
    selected: list[WorldCupSelectedMatch] = [
        WorldCupSelectedMatch(
            match,
            token_predictions=_world_cup_token_predictions(
                match,
                predicted_winners=predicted_winners,
                round_winner_tokens=round_winner_tokens,
            ),
        )
        for match in matches
        if team_key in {_score_key(match.home_team), _score_key(match.away_team)}
    ]
    selected.extend(
        _project_team_world_cup_path(
            matches,
            selected_team,
            round_winner_tokens=round_winner_tokens,
            predicted_winners=predicted_winners,
        )
    )
    return selected


def _project_team_world_cup_path(
    matches: Sequence[MatchScore],
    selected_team: str,
    *,
    round_winner_tokens: Mapping[str, str],
    predicted_winners: Mapping[str, WorldCupPrediction],
) -> list[WorldCupSelectedMatch]:
    token = ""
    path_team = selected_team
    token_is_predicted = False
    current_event_id = ""
    for match in matches:
        if _score_key(selected_team) not in {
            _score_key(match.home_team),
            _score_key(match.away_team),
        }:
            continue
        token = round_winner_tokens.get(match.event_id, "")
        token_is_predicted = False
        prediction = predicted_winners.get(match.event_id)
        if prediction:
            path_team = prediction.team
            token_is_predicted = match.state != "post"
        current_event_id = match.event_id
    if not token:
        return []

    projected: list[WorldCupSelectedMatch] = []
    match_by_event_id = {match.event_id: match for match in matches}
    next_match_ids = _world_cup_next_match_ids(matches)
    seen_event_ids = {
        match.event_id
        for match in matches
        if _score_key(selected_team) in {
            _score_key(match.home_team),
            _score_key(match.away_team),
        }
    }
    while current_event_id:
        next_event_id = next_match_ids.get(current_event_id)
        if next_event_id is None or next_event_id in seen_event_ids:
            break
        target_match = match_by_event_id.get(next_event_id)
        if target_match is None:
            break
        token = _world_cup_projection_token(
            current_event_id,
            target_match,
            projected_team=path_team,
            round_winner_tokens=round_winner_tokens,
        )
        if not token:
            break
        token_predictions = {
            placeholder: predicted
            for placeholder, predicted in _world_cup_token_predictions(
                target_match,
                predicted_winners=predicted_winners,
                round_winner_tokens=round_winner_tokens,
            ).items()
            if placeholder != token
        }
        projected.append(
            WorldCupSelectedMatch(
                target_match,
                projected_team=path_team,
                projection_token=token,
                projection_predicted=token_is_predicted,
                token_predictions=token_predictions,
            )
        )
        prediction = predicted_winners.get(target_match.event_id)
        current_event_id = target_match.event_id
        seen_event_ids.add(target_match.event_id)
        if prediction:
            path_team = prediction.team
            token_is_predicted = target_match.state != "post"
        else:
            token_is_predicted = False
    return projected


def _world_cup_round_winner_tokens(
    matches: Sequence[MatchScore],
) -> dict[str, str]:
    return _world_cup_official_winner_tokens(matches)


def _world_cup_official_winner_tokens(
    matches: Sequence[MatchScore],
) -> dict[str, str]:
    tokens: dict[str, str] = {}
    match_number_by_event_id = _world_cup_match_numbers_by_event_id(matches)
    event_id_by_match_number = {
        match_number: event_id
        for event_id, match_number in match_number_by_event_id.items()
    }
    match_by_event_id = {match.event_id: match for match in matches}
    for target_number, source_numbers in WORLD_CUP_OFFICIAL_MATCH_SOURCES.items():
        target_event_id = event_id_by_match_number.get(target_number)
        if target_event_id is None:
            continue
        target_match = match_by_event_id.get(target_event_id)
        if target_match is None:
            continue
        target_teams = (target_match.home_team, target_match.away_team)
        if len(target_teams) != len(source_numbers):
            continue
        for source_number, team_name in zip(source_numbers, target_teams):
            if not WORLD_CUP_WINNER_PLACEHOLDER_PATTERN.fullmatch(team_name):
                continue
            source_event_id = event_id_by_match_number.get(source_number)
            if source_event_id is not None:
                tokens[source_event_id] = team_name
    return tokens


def _world_cup_next_match_ids(matches: Sequence[MatchScore]) -> dict[str, str]:
    next_match_ids: dict[str, str] = {}
    match_number_by_event_id = _world_cup_match_numbers_by_event_id(matches)
    event_id_by_match_number = {
        match_number: event_id
        for event_id, match_number in match_number_by_event_id.items()
    }
    for target_number, source_numbers in WORLD_CUP_OFFICIAL_MATCH_SOURCES.items():
        target_event_id = event_id_by_match_number.get(target_number)
        if target_event_id is None:
            continue
        source_event_ids = [
            event_id_by_match_number.get(source_number)
            for source_number in source_numbers
        ]
        if any(source_event_id is None for source_event_id in source_event_ids):
            continue
        for source_event_id in source_event_ids:
            if source_event_id is not None:
                next_match_ids[source_event_id] = target_event_id
    return next_match_ids


def _world_cup_match_numbers_by_event_id(
    matches: Sequence[MatchScore],
) -> dict[str, int]:
    return {
        match.event_id: match_number
        for match in matches
        if (
            match_number := WORLD_CUP_OFFICIAL_MATCH_BY_ESPN_EVENT_ID.get(
                match.event_id,
            )
        )
    }


def _world_cup_placeholder_team_names(match: MatchScore) -> tuple[str, ...]:
    return tuple(
        team_name
        for team_name in (match.home_team, match.away_team)
        if WORLD_CUP_WINNER_PLACEHOLDER_PATTERN.fullmatch(team_name)
    )


def _world_cup_projection_token(
    source_event_id: str,
    target_match: MatchScore,
    *,
    projected_team: str,
    round_winner_tokens: Mapping[str, str],
) -> str:
    token = round_winner_tokens.get(source_event_id, "")
    if token:
        return token
    placeholders = _world_cup_placeholder_team_names(target_match)
    if len(placeholders) != 1:
        return ""
    projected_key = _score_key(projected_team)
    if any(
        _score_key(team_name) == projected_key
        for team_name in (target_match.home_team, target_match.away_team)
        if not WORLD_CUP_WINNER_PLACEHOLDER_PATTERN.fullmatch(team_name)
    ):
        return ""
    return placeholders[0]


def _world_cup_predicted_winners(
    matches: Sequence[MatchScore],
    *,
    events_by_match: Mapping[frozenset[str], dict[str, Any]],
    fetch_live: LivePriceFetcher | None,
    winner_event: tuple[dict[str, Any], int | None] | None,
) -> dict[str, WorldCupPrediction]:
    winners: dict[str, WorldCupPrediction] = {}
    token_by_match_id = {
        winner_token: event_id
        for event_id, winner_token in _world_cup_round_winner_tokens(matches).items()
    }
    title_strengths = _world_cup_title_strengths(winner_event, fetch_live)
    for match in matches:
        prediction = _world_cup_match_winner(
            match,
            event=events_by_match.get(_match_key(match.home_team, match.away_team)),
            fetch_live=fetch_live,
            predicted_winners=winners,
            token_by_match_id=token_by_match_id,
            title_strengths=title_strengths,
        )
        if prediction:
            winners[match.event_id] = prediction
    return winners


def _world_cup_match_winner(
    match: MatchScore,
    *,
    event: dict[str, Any] | None,
    fetch_live: LivePriceFetcher | None,
    predicted_winners: Mapping[str, WorldCupPrediction],
    token_by_match_id: Mapping[str, str],
    title_strengths: Mapping[str, float],
) -> WorldCupPrediction | None:
    if match.winner_team:
        return WorldCupPrediction(match.winner_team, "espn_result")
    score_winner = _final_winner(_match_display(match))
    if match.state == "post" and score_winner:
        return WorldCupPrediction(score_winner, "espn_result")
    if event is not None and fetch_live is not None:
        event_winner = _world_cup_event_winner(
            match,
            event=event,
            fetch_live=fetch_live,
        )
        if event_winner:
            return event_winner
    return _world_cup_strength_winner(
        match,
        predicted_winners=predicted_winners,
        token_by_match_id=token_by_match_id,
        title_strengths=title_strengths,
    )


def _world_cup_event_winner(
    match: MatchScore,
    *,
    event: dict[str, Any],
    fetch_live: LivePriceFetcher,
) -> WorldCupPrediction | None:
    quotes = normalize_event_quotes(event)
    team_names = [match.home_team, match.away_team]
    quotes_by_team = _world_cup_team_quotes(
        quotes,
        team_names,
        fetch_live=fetch_live,
    )
    if len(quotes_by_team) != 2:
        return None
    favorite, favorite_probability = max(quotes_by_team.items(), key=lambda item: item[1])
    draw_probability = _world_cup_draw_probability(
        quotes,
        team_names,
        fetch_live=fetch_live,
    )
    challengers = [probability for team, probability in quotes_by_team.items() if team != favorite]
    if draw_probability is not None:
        challengers.append(draw_probability)
    if challengers and favorite_probability - max(challengers) < WORLD_CUP_MATCH_MARKET_MIN_EDGE:
        return None
    return WorldCupPrediction(favorite, "match_market")


def _world_cup_strength_winner(
    match: MatchScore,
    *,
    predicted_winners: Mapping[str, WorldCupPrediction],
    token_by_match_id: Mapping[str, str],
    title_strengths: Mapping[str, float],
) -> WorldCupPrediction | None:
    if not title_strengths:
        return None
    teams = _world_cup_resolved_match_teams(
        match,
        predicted_winners=predicted_winners,
        token_by_match_id=token_by_match_id,
    )
    if len(teams) != 2:
        return None
    first_strength = _world_cup_team_strength(teams[0], title_strengths)
    second_strength = _world_cup_team_strength(teams[1], title_strengths)
    if first_strength is None or second_strength is None:
        return None
    if first_strength == second_strength:
        return None
    winner = teams[0] if first_strength > second_strength else teams[1]
    return WorldCupPrediction(winner, "winner_market")


def _world_cup_resolved_match_teams(
    match: MatchScore,
    *,
    predicted_winners: Mapping[str, WorldCupPrediction],
    token_by_match_id: Mapping[str, str],
) -> list[str]:
    teams = []
    for team_name in (match.home_team, match.away_team):
        source_event_id = token_by_match_id.get(team_name)
        if source_event_id is None:
            teams.append(team_name)
            continue
        prediction = predicted_winners.get(source_event_id)
        if prediction:
            teams.append(prediction.team)
    return teams


def _world_cup_team_strength(
    team_name: str,
    title_strengths: Mapping[str, float],
) -> float | None:
    key = _score_key(canonical_team_name(team_name))
    return title_strengths.get(key)


def _world_cup_title_strengths(
    winner_event: tuple[dict[str, Any], int | None] | None,
    fetch_live: LivePriceFetcher | None,
) -> dict[str, float]:
    if winner_event is None or fetch_live is None:
        return {}
    event, _timestamp = winner_event
    strengths: dict[str, float] = {}
    for quote in normalize_event_quotes(event):
        key = _score_key(canonical_team_name(quote.title))
        strengths[key] = _quote_probability_percent(quote, fetch_live=fetch_live)
    return strengths


def _world_cup_token_predictions(
    match: MatchScore,
    *,
    predicted_winners: Mapping[str, WorldCupPrediction],
    round_winner_tokens: Mapping[str, str],
) -> dict[str, str]:
    predictions: dict[str, str] = {}
    token_by_match_id = {
        winner_token: event_id
        for event_id, winner_token in round_winner_tokens.items()
    }
    for team_name in (match.home_team, match.away_team):
        source_event_id = token_by_match_id.get(team_name)
        if source_event_id is None:
            continue
        prediction = predicted_winners.get(source_event_id)
        if prediction:
            predictions[team_name] = prediction.team
    return predictions


def _world_cup_events_by_match(
    games: Sequence[dict[str, Any]],
) -> dict[frozenset[str], dict[str, Any]]:
    events_by_match: dict[frozenset[str], dict[str, Any]] = {}
    for event in games:
        key = _world_cup_event_match_key(event)
        if key is None:
            continue
        if key not in events_by_match or event.get("closed") is not True:
            events_by_match[key] = event
    return events_by_match


def _world_cup_event_match_key(event: Mapping[str, Any]) -> frozenset[str] | None:
    title = event.get("title")
    if not title:
        return None
    team_names = [part.strip() for part in str(title).split(" vs. ")]
    if len(team_names) != 2:
        return None
    return _match_key(team_names[0], team_names[1])


def _match_key(first_team: str, second_team: str) -> frozenset[str]:
    return frozenset({_score_key(first_team), _score_key(second_team)})


def _format_world_cup_game(
    match: MatchScore,
    *,
    event: dict[str, Any] | None,
    fetch_live: LivePriceFetcher,
    format_country: CountryFormatter,
    chat_timezone: timezone,
    timezone_label: str,
    projected_team: str = "",
    projection_token: str = "",
    projection_predicted: bool = False,
    token_predictions: Mapping[str, str] | None = None,
) -> tuple[str, str, str] | None:
    teams = []
    team_names = [match.home_team, match.away_team]
    scores = _match_display(match)
    quotes_by_team: dict[str, float] = {}
    favorite = _final_winner(scores) if scores.state == "post" else ""
    if scores.state != "post" and event is not None:
        quotes = normalize_event_quotes(event)
        quotes_by_team = _world_cup_team_quotes(
            quotes,
            team_names,
            fetch_live=fetch_live,
        )
        if len(quotes_by_team) == len(team_names):
            favorite, favorite_probability = max(
                quotes_by_team.items(),
                key=lambda item: item[1],
                default=("", 0.0),
            )
            draw_probability = _world_cup_draw_probability(
                quotes,
                team_names,
                fetch_live=fetch_live,
            )
            if draw_probability is not None and draw_probability > favorite_probability:
                favorite = ""
    for team_name in team_names:
        score = scores.scores.get(team_name)
        if score is None:
            team_probability = quotes_by_team.get(team_name)
            if team_probability is None:
                label = _format_world_cup_team_label(
                    team_name,
                    projected_team=projected_team,
                    projection_token=projection_token,
                    projection_predicted=projection_predicted,
                    token_predictions=token_predictions or {},
                    format_country=format_country,
                )
            else:
                decimals = 2 if team_probability < 10 else 1
                probability = f"{fmt_num(team_probability, decimals)}%"
                label = f"{format_country(team_name)} {probability}"
        elif scores.state == "post":
            label = f"{format_country(team_name)} {score}"
        else:
            team_probability = quotes_by_team.get(team_name)
            if team_probability is None:
                label = f"{format_country(team_name)} {score}"
            else:
                decimals = 2 if team_probability < 10 else 1
                probability = f"{fmt_num(team_probability, decimals)}%"
                label = f"{format_country(team_name)} {score} ({probability})"
        teams.append(f"[{label}]" if team_name == favorite else label)

    title = escape(" vs. ".join(teams))
    slug = event.get("slug") if event else None
    linked_title = (
        f'<a href="{escape(f"https://polymarket.com/sports/world-cup/{slug}", quote=True)}">'
        f"{title}</a>"
        if slug
        else title
    )
    date_string, time_string = _format_kickoff(
        match.start_time,
        chat_timezone,
        timezone_label,
    )
    time_string = _format_match_time(time_string, scores)
    return date_string, linked_title, time_string


def _format_world_cup_team_label(
    team_name: str,
    *,
    projected_team: str,
    projection_token: str,
    projection_predicted: bool,
    token_predictions: Mapping[str, str],
    format_country: CountryFormatter,
) -> str:
    if projection_token and team_name == projection_token:
        if projection_predicted:
            return f"{format_country(projected_team)} (pronóstico)"
        return f"{format_country(projected_team)} (si avanza)"
    predicted_team = token_predictions.get(team_name)
    if predicted_team:
        return f"{format_country(predicted_team)} (pronóstico)"
    match = WORLD_CUP_WINNER_PLACEHOLDER_PATTERN.fullmatch(team_name)
    if match:
        return f"Ganador {match.group(1)} {match.group(2)}"
    return format_country(team_name)


def _world_cup_team_quotes(
    quotes: Sequence[MarketQuote],
    team_names: Sequence[str],
    *,
    fetch_live: LivePriceFetcher,
) -> dict[str, float]:
    team_by_key = {_score_key(team_name): team_name for team_name in team_names}
    probabilities: dict[str, float] = {}
    for quote in quotes:
        team_name = team_by_key.get(_score_key(quote.title))
        if team_name is None:
            continue
        probabilities[team_name] = _quote_probability_percent(
            quote,
            fetch_live=fetch_live,
        )
    return probabilities


def _world_cup_draw_probability(
    quotes: Sequence[MarketQuote],
    team_names: Sequence[str],
    *,
    fetch_live: LivePriceFetcher,
) -> float | None:
    draw_keys = {
        _score_key(f"Draw ({team_names[0]} vs. {team_names[1]})"),
        _score_key(f"Draw ({team_names[1]} vs. {team_names[0]})"),
    }
    for quote in quotes:
        if _score_key(quote.title) in draw_keys:
            return _quote_probability_percent(quote, fetch_live=fetch_live)
    return None


def _quote_probability_percent(
    quote: MarketQuote,
    *,
    fetch_live: LivePriceFetcher,
) -> float:
    probability = quote.probability
    if quote.token_id:
        live = fetch_live(quote.token_id)
        if live:
            probability = max(0.0, min(live[0], 1.0))
    return probability * 100


def _score_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonical_team_name(name).casefold())


@dataclass(frozen=True)
class MatchDisplay:
    scores: dict[str, int]
    state: str
    display_clock: str


def _match_display(match: MatchScore) -> MatchDisplay:
    if match.state not in {"in", "post"}:
        return MatchDisplay({}, match.state, match.display_clock)
    return MatchDisplay(
        {
            match.home_team: match.home_score,
            match.away_team: match.away_score,
        },
        match.state,
        match.display_clock,
    )


def _final_winner(scores: MatchDisplay) -> str:
    if len(scores.scores) != 2:
        return ""
    first_team, second_team = scores.scores
    first_score = scores.scores[first_team]
    second_score = scores.scores[second_team]
    if first_score == second_score:
        return ""
    return first_team if first_score > second_score else second_team


def _format_match_time(time_string: str, scores: MatchDisplay) -> str:
    if scores.state == "post":
        return "Final"
    if scores.state == "in" and scores.display_clock:
        return (
            f"{scores.display_clock} · {time_string}"
            if time_string
            else scores.display_clock
        )
    return time_string


def _format_kickoff(
    end_date: str,
    chat_timezone: timezone,
    timezone_label: str,
) -> tuple[str, str]:
    try:
        kickoff = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        local = kickoff.astimezone(chat_timezone)
        return (
            (
                f"{SPANISH_WEEKDAYS[local.weekday()]}, "
                f"{local.day} de {SPANISH_MONTHS[local.month - 1]}"
            ),
            local.strftime(f"%H:%M {timezone_label}"),
        )
    except ValueError:
        if end_date:
            return end_date[:10], end_date[11:16].replace("T", " ")
        return "Fecha desconocida", ""


class PolymarketService:
    def __init__(
        self,
        *,
        cache: CacheService,
        cache_ttl: int,
        stream_cache_ttl: int,
        make_timezone: TimezoneFactory,
        scoreboard: WorldCupScoreboard | None = None,
    ) -> None:
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._stream_cache_ttl = stream_cache_ttl
        self._make_timezone = make_timezone
        self._scoreboard = scoreboard or WorldCupScoreboard(
            fetch_scores=fetch_scoreboard_scores
        )

    def fetch_live_price(self, token_id: str) -> tuple[float, int | None] | None:
        return fetch_live_price(
            token_id,
            cached_request=self._cache.request,
            cache_ttl=self._stream_cache_ttl,
        )

    def fetch_live_prices(self, token_ids: Sequence[str]) -> dict[str, float]:
        return fetch_live_prices(token_ids, http_post=http_client.post)

    def fetch_event(self, slug: str) -> tuple[dict[str, Any], int | None] | None:
        return fetch_event(
            slug,
            cached_request=self._cache.request,
            cache_ttl=self._cache_ttl,
        )

    def format_event_section(
        self,
        event: dict[str, Any],
        header: str,
        filter_prefixes: Sequence[str],
    ) -> tuple[list[str], int | None] | None:
        return format_event_section(
            event,
            header,
            filter_prefixes,
            fetch_live=self.fetch_live_price,
        )

    def event_top_outcomes(
        self,
        event: dict[str, Any],
        limit: int = 2,
        *,
        fetch_live: LivePriceFetcher | None = None,
    ) -> list[tuple[str, float]]:
        return event_top_outcomes(
            event,
            limit,
            fetch_live=fetch_live,
        )

    def get_global_elections(self) -> str:
        return get_global_elections(
            cached_request=self._cache.request,
            cache_ttl=self._cache_ttl,
            fetch_live_prices=self.fetch_live_prices,
            get_event_flag=event_country_flag,
            format_liquidity=format_usd_compact,
        )

    def get_world_cup_games(
        self,
        timezone_offset: int = -3,
        team_query: str = "",
    ) -> str:
        fetch_scores = self._scoreboard.get_scores
        if team_query:
            fetch_scores = self._fetch_team_world_cup_scores
        return get_world_cup_games(
            timezone_offset,
            fetch_winner_event=self.fetch_event,
            cached_request=self._cache.request,
            cache_ttl=self._cache_ttl,
            fetch_live=self.fetch_live_price,
            format_country=flagged_country_name,
            make_timezone=self._make_timezone,
            fetch_scores=fetch_scores,
            team_query=team_query,
        )

    def _fetch_team_world_cup_scores(self) -> dict[str, MatchScore]:
        bracket_scores = fetch_scoreboard_scores(
            days_before=WORLD_CUP_TEAM_SCOREBOARD_DAYS_BEFORE,
            days_after=WORLD_CUP_TEAM_SCOREBOARD_DAYS_AFTER,
            limit=WORLD_CUP_TEAM_SCOREBOARD_LIMIT,
        )
        return _overlay_live_scores(bracket_scores, self._scoreboard.get_scores())


def _overlay_live_scores(
    bracket_scores: Mapping[str, MatchScore],
    live_scores: Mapping[str, MatchScore],
) -> dict[str, MatchScore]:
    return {
        event_id: live_scores.get(event_id, match)
        for event_id, match in bracket_scores.items()
    }


__all__ = ["MatchScore", "PolymarketService", "fetch_scoreboard_scores"]
