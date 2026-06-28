"""Fetch and format Polymarket events and prices."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Any

import pycountry

from api.cache.service import CacheService
from api.markets.world_cup_goals import (
    MatchScore,
    canonical_team_name,
    fetch_scoreboard_scores,
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
WORLD_CUP_FETCH_LIMIT = 100
WORLD_CUP_FETCH_MAX_PAGES = 20
WORLD_CUP_WINNER_SLUG = "world-cup-winner"
WORLD_CUP_WINNER_LIMIT = 5
WORLD_CUP_GAME_SLUG_PATTERN = re.compile(
    r"^fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$"
)
WORLD_CUP_SCORE_TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z"
)
WORLD_CUP_MINUTE_TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z")
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
    "dr congo": "CD",
    "england": "GB",
    "ir iran": "IR",
    "ivory coast": "CI",
    "korea republic": "KR",
    "scotland": "GB",
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
) -> str:
    winner_event = fetch_winner_event(WORLD_CUP_WINNER_SLUG)
    try:
        live_scores = fetch_scores()
    except Exception:
        live_scores = {}
    selected_matches = _select_world_cup_matches(live_scores)
    events = (
        _fetch_world_cup_events(
            cached_request=cached_request,
            cache_ttl=cache_ttl,
            selected_matches=selected_matches,
        )
        if selected_matches
        else []
    )
    if not winner_event and not events and not live_scores:
        return "No pude traer los partidos del Mundial desde Polymarket"

    games = [event for event in events if _is_world_cup_game_event(event)]
    games.sort(key=lambda event: str(event.get("endDate") or ""))
    lines = ["Polymarket - Mundial"]

    if winner_event:
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
        if winner_outcomes:
            lines.extend(
                [
                    "",
                    f'<a href="https://polymarket.com/event/{WORLD_CUP_WINNER_SLUG}">'
                    "Campeón del Mundial</a>",
                    " | ".join(winner_outcomes),
                ]
            )

    games_by_date: dict[str, list[tuple[str, str]]] = {}
    chat_timezone = make_timezone(timezone_offset)
    timezone_label = f"UTC{timezone_offset:+d}" if timezone_offset else "UTC"
    events_by_match = _world_cup_events_by_match(games)
    rendered_games = 0
    for match in selected_matches:
        formatted_event = _format_world_cup_game(
            match,
            event=events_by_match.get(_match_key(match.home_team, match.away_team)),
            fetch_live=fetch_live,
            format_country=format_country,
            chat_timezone=chat_timezone,
            timezone_label=timezone_label,
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


def _is_world_cup_game_event(event: Mapping[str, Any]) -> bool:
    return WORLD_CUP_GAME_SLUG_PATTERN.fullmatch(str(event.get("slug") or "")) is not None


def _select_world_cup_matches(
    scores: Mapping[str, MatchScore],
) -> list[MatchScore]:
    matches = sorted(scores.values(), key=lambda match: match.start_time)
    finished = [match for match in matches if match.state == "post"]
    active_or_future = [match for match in matches if match.state != "post"]
    return finished[-2:] + active_or_future[:WORLD_CUP_CANDIDATE_LIMIT]


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
                label = format_country(team_name)
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
    ) -> None:
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._stream_cache_ttl = stream_cache_ttl
        self._make_timezone = make_timezone

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

    def get_world_cup_games(self, timezone_offset: int = -3) -> str:
        return get_world_cup_games(
            timezone_offset,
            fetch_winner_event=self.fetch_event,
            cached_request=self._cache.request,
            cache_ttl=self._cache_ttl,
            fetch_live=self.fetch_live_price,
            format_country=flagged_country_name,
            make_timezone=self._make_timezone,
            fetch_scores=fetch_scoreboard_scores,
        )


__all__ = ["PolymarketService"]
