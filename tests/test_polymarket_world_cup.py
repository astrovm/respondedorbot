from datetime import timedelta, timezone
from unittest.mock import MagicMock

from api import index
from api.markets.world_cup_goals import (
    MatchScore,
    TEAM_NAME_ALIASES,
    TEAM_NAMES_ES,
    WORLD_CUP_TEAM_RANKING,
    team_name_es,
)
from api.markets.polymarket import country_flag_from_name, flagged_country_name
from api.markets import polymarket as polymarket_commands


def _match_score(
    event_id: str,
    home_team: str,
    away_team: str,
    home_score: int = 0,
    away_score: int = 0,
    *,
    state: str = "pre",
    display_clock: str = "0'",
    start_time: str = "2026-06-11T19:00:00Z",
    round_slug: str = "",
    winner_team: str = "",
) -> MatchScore:
    return MatchScore(
        event_id=event_id,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        state=state,
        display_clock=display_clock,
        start_time=start_time,
        round_slug=round_slug,
        winner_team=winner_team,
    )


def test_fetch_live_price_uses_clob_midpoint():
    captured = {}

    def fake_cached_request(url, parameters, *_args, **kwargs):
        captured.update({"url": url, "parameters": parameters, **kwargs})
        return {"data": {"mid": "0.525"}}

    result = polymarket_commands.fetch_live_price(
        "token-123",
        cached_request=fake_cached_request,
        cache_ttl=5,
    )

    assert result == (0.525, None)
    assert captured == {
        "url": "https://clob.polymarket.com/midpoint",
        "parameters": {"token_id": "token-123"},
        "verify_ssl": False,
    }


def test_get_polymarket_world_cup_games_filters_props_and_formats_kickoff(
    monkeypatch,
):
    captured = []
    events = [
        {
            "title": "Mexico vs. South Africa - Exact Score",
            "slug": "fifwc-mex-rsa-2026-06-11-exact-score",
            "endDate": "2026-06-11T19:00:00Z",
            "markets": [],
        },
        {
            "title": "Korea Republic vs. Czechia",
            "slug": "fifwc-kr-cze-2026-06-11",
            "endDate": "2026-06-12T02:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Korea Republic",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.55", "0.45"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Czechia",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.25", "0.75"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
        {
            "title": "Mexico vs. South Africa",
            "slug": "fifwc-mex-rsa-2026-06-11",
            "endDate": "2026-06-11T19:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Mexico",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.695", "0.305"]',
                    "clobTokenIds": '["mex-yes", "mex-no"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Draw (Mexico vs. South Africa)",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.205", "0.795"]',
                    "clobTokenIds": '["draw-yes", "draw-no"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "South Africa",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.105", "0.895"]',
                    "clobTokenIds": '["rsa-yes", "rsa-no"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
    ]

    winner = {
        "title": "World Cup Winner",
        "slug": "world-cup-winner",
        "markets": [
            {
                "groupItemTitle": name,
                "outcomes": '["Yes", "No"]',
                "outcomePrices": f'["{price}", "{1 - price}"]',
                "active": True,
                "closed": False,
            }
            for name, price in [
                ("Spain", 0.18),
                ("France", 0.16),
                ("Brazil", 0.14),
                ("England", 0.12),
                ("Argentina", 0.10),
                ("Germany", 0.08),
            ]
        ],
    }

    def fake_cached_requests(_url, parameters, *_args):
        captured.append(parameters)
        if parameters == {"slug": "world-cup-winner"}:
            return {"data": [winner], "timestamp": 1}
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        index.app_runtime.polymarket,
        "fetch_live_price",
        lambda token_id: {
            "mex-yes": (0.72, 1),
            "draw-yes": (0.18, 1),
            "rsa-yes": (0.10, 1),
        }.get(token_id),
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "game-1": _match_score(
                "game-1",
                "South Africa",
                "Mexico",
                1,
                2,
                state="in",
                display_clock="35'",
                start_time="2026-06-11T19:00:00Z",
            ),
            "game-2": _match_score(
                "game-2",
                "Korea Republic",
                "Czechia",
                state="pre",
                display_clock="0'",
                start_time="2026-06-12T02:00:00Z",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert captured == [
        {"slug": "world-cup-winner"},
        {
            "limit": 100,
            "offset": 0,
            "active": "true",
            "series_id": 11433,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": "2026-06-11T19:00:00Z",
            "end_date_max": "2026-06-12T02:00:00Z",
        },
    ]
    assert result.startswith("Polymarket - Mundial")
    assert "Campeón del Mundial</a>" in result
    assert (
        "🇪🇸 España 18% | 🇫🇷 Francia 16% | 🇧🇷 Brasil 14% | "
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e"
        "\U000e0067\U000e007f Inglaterra 12% | 🇦🇷 Argentina 10%"
    ) in result
    assert "Germany" not in result
    assert "World Cup" not in result
    assert result.index("fifwc-mex-rsa-2026-06-11") < result.index(
        "fifwc-kr-cze-2026-06-11"
    )
    assert "Exact Score" not in result
    assert "🇿🇦 Sudáfrica 1 (10%) vs. [🇲🇽 México 2 (72%)]" in result
    assert "Mexico 2-1 South Africa" not in result
    assert "· live" not in result
    assert "Mexico 69.5%" not in result
    assert "Draw 18%" not in result
    assert "jue, 11 de junio\n<a href=" in result
    assert "35' · 16:00 UTC-3\n\n<a href=" in result
    assert (
        '<a href="https://polymarket.com/sports/world-cup/'
        'fifwc-mex-rsa-2026-06-11">🇿🇦 Sudáfrica 1 (10%) vs. '
        "[🇲🇽 México 2 (72%)]</a>"
    ) in result


def test_get_polymarket_world_cup_games_handles_empty_response(monkeypatch):
    monkeypatch.setattr(
        index.app_runtime.cache,
        "request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        dict,
    )

    assert (
        index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)
        == "No pude traer los partidos del Mundial desde Polymarket"
    )


def test_get_polymarket_world_cup_games_shows_final_without_polymarket_event(
    monkeypatch,
):
    monkeypatch.setattr(
        index.app_runtime.cache,
        "request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "game-1": _match_score(
                "game-1",
                "Ghana",
                "Panama",
                1,
                0,
                state="post",
                display_clock="FT",
                start_time="2026-06-18T02:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "[🇬🇭 Ghana 1] vs. 🇵🇦 Panamá 0" in result
    assert "polymarket.com/sports/world-cup" not in result
    assert "\nFinal" in result


def test_get_polymarket_world_cup_games_marks_team_when_team_leads_draw(
    monkeypatch,
):
    event = {
        "title": "Team A vs. Team B",
        "slug": "fifwc-a-b-2026-06-11",
        "endDate": "2026-06-11T19:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Draw (Team A vs. Team B)",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            {
                "groupItemTitle": "Team A",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.35", "0.65"]',
            },
            {
                "groupItemTitle": "Team B",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.25", "0.75"]',
            },
        ],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": [event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "game-1": _match_score(
                "game-1",
                "Team A",
                "Team B",
                state="pre",
                display_clock="0'",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert ">[Team A 35%] vs. Team B 25%</a>" in result
    assert "[Draw]" not in result


def test_get_polymarket_world_cup_games_does_not_mark_team_when_draw_leads(
    monkeypatch,
):
    event = {
        "title": "Saudi Arabia vs. Uruguay",
        "slug": "fifwc-ksa-ury-2026-06-15",
        "endDate": "2026-06-15T22:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Saudi Arabia",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.0005", "0.9995"]',
            },
            {
                "groupItemTitle": "Draw (Saudi Arabia vs. Uruguay)",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.9995", "0.0005"]',
            },
            {
                "groupItemTitle": "Uruguay",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.0005", "0.9995"]',
            },
        ],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": [event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "game-1": _match_score(
                "game-1",
                "Saudi Arabia",
                "Uruguay",
                1,
                1,
                state="in",
                display_clock="75'",
                start_time="2026-06-15T22:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "🇸🇦 Arabia Saudita 1 (0.05%) vs. 🇺🇾 Uruguay 1 (0.05%)" in result
    assert "75' · 19:00 UTC-3" in result
    assert "[🇸🇦 Arabia Saudita" not in result
    assert "[🇺🇾 Uruguay" not in result
    assert "Draw" not in result


def test_get_polymarket_world_cup_games_shows_final_with_closed_team_market(
    monkeypatch,
):
    event = {
        "title": "Belgium vs. Egypt",
        "slug": "fifwc-bel-egy-2026-06-15",
        "endDate": "2026-06-15T19:00:00Z",
        "closed": True,
        "markets": [],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": [event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Belgium",
                "Egypt",
                1,
                1,
                state="post",
                display_clock="FT",
                start_time="2026-06-15T19:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "🇧🇪 Bélgica 1 vs. 🇪🇬 Egipto 1" in result
    assert "fifwc-bel-egy-2026-06-15" in result
    assert "[🇧🇪 Bélgica" not in result
    assert "[🇪🇬 Egipto" not in result
    assert "\nFinal" in result


def test_get_polymarket_world_cup_games_uses_later_pages_for_recent_matches(
    monkeypatch,
):
    old_events = [
        {
            "title": f"Old Team {index_} vs. Old Rival {index_}",
            "slug": f"fifwc-old{index_}-rvl{index_}-2026-06-11",
            "endDate": "2026-06-11T19:00:00Z",
            "closed": True,
            "markets": [],
        }
        for index_ in range(100)
    ]
    recent_event = {
        "title": "Team A vs. Team B",
        "slug": "fifwc-a-b-2026-06-17",
        "endDate": "2026-06-18T02:00:00Z",
        "closed": True,
        "markets": [],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        if parameters["offset"] == 0:
            return {"data": old_events}
        return {"data": [recent_event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Team A",
                "Team B",
                2,
                1,
                state="post",
                display_clock="FT",
                start_time="2026-06-18T02:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "fifwc-a-b-2026-06-17" in result
    assert "[Team A 2] vs. Team B 1" in result
    assert "fifwc-old0-rvl0-2026-06-11" not in result


def test_get_polymarket_world_cup_games_pages_until_selected_match_is_found(
    monkeypatch,
):
    filler_events = [
        {
            "title": f"Prop Team {index_} vs. Prop Rival {index_} - Total Corners",
            "slug": f"fifwc-prop{index_}-rvl{index_}-2026-06-26-total-corners",
            "endDate": "2026-06-27T00:00:00Z",
            "markets": [],
        }
        for index_ in range(100)
    ]
    target_event = {
        "title": "Uruguay vs. Spain",
        "slug": "fifwc-ury-esp-2026-06-26",
        "endDate": "2026-06-27T00:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Uruguay",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.20", "0.80"]',
                "active": True,
                "closed": False,
            },
            {
                "groupItemTitle": "Spain",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.62", "0.38"]',
                "active": True,
                "closed": False,
            },
        ],
    }
    captured_offsets = []

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        captured_offsets.append(parameters["offset"])
        assert parameters["end_date_min"] == "2026-06-27T00:00:00Z"
        assert parameters["end_date_max"] == "2026-06-27T00:00:00Z"
        if parameters["offset"] < 500:
            return {"data": filler_events}
        return {"data": [target_event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Uruguay",
                "Spain",
                state="pre",
                display_clock="0'",
                start_time="2026-06-27T00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert captured_offsets == [0, 100, 200, 300, 400, 500]
    assert "fifwc-ury-esp-2026-06-26" in result
    assert "🇺🇾 Uruguay 20% vs. [🇪🇸 España 62%]" in result


def test_get_polymarket_world_cup_games_uses_later_candidates_when_odds_missing(
    monkeypatch,
):
    def event(index_):
        return {
            "title": f"Future {index_} vs. Rival {index_}",
            "slug": f"fifwc-fut{index_}-riv{index_}-2026-06-29",
            "endDate": f"2026-06-29T{index_:02d}:00:00Z",
            "markets": [
                {
                    "groupItemTitle": f"Future {index_}",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.60", "0.40"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": f"Rival {index_}",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.30", "0.70"]',
                    "active": True,
                    "closed": False,
                },
            ],
        }

    events = [
        {
            "title": "Final A vs. Final B",
            "slug": "fifwc-fina-finb-2026-06-28",
            "endDate": "2026-06-28T01:00:00Z",
            "closed": True,
            "markets": [],
        },
        {
            "title": "Final C vs. Final D",
            "slug": "fifwc-finc-find-2026-06-28",
            "endDate": "2026-06-28T02:00:00Z",
            "closed": True,
            "markets": [],
        },
        *(event(index_) for index_ in range(2, 11)),
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": events}

    scores = {
        "final-1": _match_score(
            "final-1",
            "Final A",
            "Final B",
            1,
            0,
            state="post",
            display_clock="FT",
            start_time="2026-06-28T01:00:00Z",
        ),
        "final-2": _match_score(
            "final-2",
            "Final C",
            "Final D",
            2,
            0,
            state="post",
            display_clock="FT",
            start_time="2026-06-28T02:00:00Z",
        ),
        "missing-odds": _match_score(
            "missing-odds",
            "Future 1",
            "Rival 1",
            state="pre",
            start_time="2026-06-29T01:00:00Z",
        ),
    }
    scores.update(
        {
            f"future-{index_}": _match_score(
                f"future-{index_}",
                f"Future {index_}",
                f"Rival {index_}",
                state="pre",
                start_time=f"2026-06-29T{index_:02d}:00:00Z",
            )
            for index_ in range(2, 11)
        }
    )

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr("api.markets.polymarket.fetch_scoreboard_scores", lambda: scores)

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert result.count("polymarket.com/sports/world-cup/") == 9
    assert "Future 1 vs. Rival 1" in result
    assert "fifwc-fut1-riv1-2026-06-29" not in result
    assert "fifwc-fut8-riv8-2026-06-29" in result
    assert "fifwc-fut9-riv9-2026-06-29" not in result


def test_get_polymarket_world_cup_games_matches_provider_aliases(monkeypatch):
    events = [
        {
            "title": "Cabo Verde vs. Côte d'Ivoire",
            "slug": "fifwc-cvi-civ-2026-06-26",
            "endDate": "2026-06-26T21:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Cabo Verde",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.36", "0.64"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Côte d'Ivoire",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.44", "0.56"]',
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Cape Verde",
                "Ivory Coast",
                state="pre",
                display_clock="0'",
                start_time="2026-06-26T21:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "fifwc-cvi-civ-2026-06-26" in result
    assert "🇨🇻 Cabo Verde 36% vs. [🇨🇮 Costa de Marfil 44%]" in result
    assert "Cape Verde" not in result
    assert "Ivory Coast" not in result


def test_get_polymarket_world_cup_games_shows_scheduled_match_without_polymarket_event(
    monkeypatch,
):
    monkeypatch.setattr(
        index.app_runtime.cache,
        "request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Mexico",
                "Ecuador",
                state="pre",
                display_clock="0'",
                start_time="2026-07-01T01:00:00Z",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "🇲🇽 México vs. 🇪🇨 Ecuador" in result
    assert "22:00 UTC-3" in result
    assert "polymarket.com/sports/world-cup" not in result
    assert "%" not in result


def test_get_polymarket_world_cup_games_filters_country_and_projects_path(
    monkeypatch,
):
    monkeypatch.setattr(
        index.app_runtime.cache,
        "request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "760480": _match_score(
                "760480",
                "Argentina",
                "Jordan",
                3,
                1,
                state="post",
                display_clock="FT",
                start_time="2026-06-28T02:00:00Z",
            ),
            "760499": _match_score(
                "760499",
                "Australia",
                "Egypt",
                state="pre",
                start_time="2026-07-03T18:00:00Z",
                round_slug="round-of-32",
            ),
            "760500": _match_score(
                "760500",
                "Argentina",
                "Cape Verde",
                state="pre",
                start_time="2026-07-03T22:00:00Z",
                round_slug="round-of-32",
            ),
            "760501": _match_score(
                "760501",
                "Colombia",
                "Ghana",
                state="pre",
                start_time="2026-07-04T01:30:00Z",
                round_slug="round-of-32",
            ),
            "760508": _match_score(
                "760508",
                "Round of 32 1 Winner",
                "Round of 32 2 Winner",
                state="pre",
                start_time="2026-07-07T20:00:00Z",
                round_slug="round-of-16",
            ),
            "760513": _match_score(
                "760513",
                "Round of 16 1 Winner",
                "Round of 16 4 Winner",
                state="pre",
                start_time="2026-07-12T01:00:00Z",
                round_slug="quarterfinals",
            ),
            "760515": _match_score(
                "760515",
                "Quarterfinal 1 Winner",
                "Quarterfinal 2 Winner",
                state="pre",
                start_time="2026-07-15T19:00:00Z",
                round_slug="semifinals",
            ),
            "760517": _match_score(
                "760517",
                "Semifinal 1 Winner",
                "Semifinal 2 Winner",
                state="pre",
                start_time="2026-07-19T19:00:00Z",
                round_slug="final",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(
        timezone_offset=-3,
        team_query="argentina",
    )

    assert result.startswith("Polymarket - Mundial: Argentina")
    assert "[🇦🇷 Argentina 3] vs. 🇯🇴 Jordania 1" in result
    assert "🇦🇷 Argentina vs. 🇨🇻 Cabo Verde" in result
    assert "Ganador Round of 32 1 vs. 🇦🇷 Argentina (si avanza)" in result
    assert "🇦🇷 Argentina (si avanza) vs. Ganador Round of 16 4" in result
    assert "🇦🇷 Argentina (si avanza) vs. Ganador Quarterfinal 2" in result
    assert "🇦🇷 Argentina (si avanza) vs. Ganador Semifinal 2" in result
    assert "Australia" not in result
    assert "Colombia" not in result


def test_get_polymarket_world_cup_games_stops_when_country_is_predicted_out(
    monkeypatch,
):
    events = [
        {
            "title": "Argentina vs. Cape Verde",
            "slug": "fifwc-arg-cvi-2026-07-03",
            "endDate": "2026-07-03T22:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Argentina",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.35", "0.65"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Cape Verde",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.55", "0.45"]',
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "760500": _match_score(
                "760500",
                "Argentina",
                "Cape Verde",
                state="pre",
                start_time="2026-07-03T22:00:00Z",
                round_slug="round-of-32",
            ),
            "760508": _match_score(
                "760508",
                "Round of 32 1 Winner",
                "Round of 32 2 Winner",
                state="pre",
                start_time="2026-07-07T20:00:00Z",
                round_slug="round-of-16",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(
        timezone_offset=-3,
        team_query="argentina",
    )

    assert "🇦🇷 Argentina 35% vs. [🇨🇻 Cabo Verde 55%]" in result
    assert "si avanza" not in result
    assert "Round of 32" not in result


def test_get_polymarket_world_cup_games_resolves_predicted_placeholder_opponent(
    monkeypatch,
):
    events = [
        {
            "title": "Argentina vs. Cape Verde",
            "slug": "fifwc-arg-cvi-2026-07-03",
            "endDate": "2026-07-03T22:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Argentina",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.80", "0.20"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Cape Verde",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.10", "0.90"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
        {
            "title": "Colombia vs. Ghana",
            "slug": "fifwc-col-gha-2026-07-03",
            "endDate": "2026-07-04T01:30:00Z",
            "markets": [
                {
                    "groupItemTitle": "Colombia",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.70", "0.30"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Ghana",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.20", "0.80"]',
                    "active": True,
                    "closed": False,
                },
            ],
        },
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "760500": _match_score(
                "760500",
                "Argentina",
                "Cape Verde",
                state="pre",
                start_time="2026-07-03T22:00:00Z",
                round_slug="round-of-32",
            ),
            "760501": _match_score(
                "760501",
                "Colombia",
                "Ghana",
                state="pre",
                start_time="2026-07-04T01:30:00Z",
                round_slug="round-of-32",
            ),
            "760508": _match_score(
                "760508",
                "Round of 32 1 Winner",
                "Round of 32 2 Winner",
                state="pre",
                start_time="2026-07-07T20:00:00Z",
                round_slug="round-of-16",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(
        timezone_offset=-3,
        team_query="argentina",
    )

    assert "🇦🇷 Argentina (pronóstico) vs. 🇨🇴 Colombia (pronóstico)" in result


def test_get_polymarket_world_cup_games_projects_deeper_with_winner_market(
    monkeypatch,
):
    winner_event = {
        "title": "World Cup Winner",
        "slug": "world-cup-winner",
        "markets": [
            {
                "groupItemTitle": "Argentina",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.22", "0.78"]',
                "active": True,
                "closed": False,
            },
            {
                "groupItemTitle": "Switzerland",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.02", "0.98"]',
                "active": True,
                "closed": False,
            },
            {
                "groupItemTitle": "Algeria",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.01", "0.99"]',
                "active": True,
                "closed": False,
            },
            {
                "groupItemTitle": "Mexico",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.03", "0.97"]',
                "active": True,
                "closed": False,
            },
            {
                "groupItemTitle": "Cape Verde",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.005", "0.995"]',
                "active": True,
                "closed": False,
            },
        ],
    }
    events = [
        {
            "title": "Argentina vs. Cape Verde",
            "slug": "fifwc-arg-cvi-2026-07-03",
            "endDate": "2026-07-03T22:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Argentina",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.80", "0.20"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Cape Verde",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.10", "0.90"]',
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return {"data": [winner_event]}
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "760498": _match_score(
                "760498",
                "Switzerland",
                "Algeria",
                state="pre",
                start_time="2026-07-03T03:00:00Z",
                round_slug="round-of-32",
            ),
            "760500": _match_score(
                "760500",
                "Argentina",
                "Cape Verde",
                state="pre",
                start_time="2026-07-03T22:00:00Z",
                round_slug="round-of-32",
            ),
            "760508": _match_score(
                "760508",
                "Round of 32 1 Winner",
                "Round of 32 2 Winner",
                state="pre",
                start_time="2026-07-07T20:00:00Z",
                round_slug="round-of-16",
            ),
            "760513": _match_score(
                "760513",
                "Round of 16 1 Winner",
                "Mexico",
                state="pre",
                start_time="2026-07-12T01:00:00Z",
                round_slug="quarterfinals",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(
        timezone_offset=-3,
        team_query="argentina",
    )

    assert "🇨🇭 Suiza (pronóstico) vs. 🇦🇷 Argentina (pronóstico)" in result
    assert "🇦🇷 Argentina (pronóstico) vs. 🇲🇽 México" in result


def test_get_polymarket_world_cup_games_accepts_spanish_country_query(monkeypatch):
    monkeypatch.setattr(
        index.app_runtime.cache,
        "request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda **_kwargs: {
            "game-1": _match_score(
                "game-1",
                "South Korea",
                "Japan",
                1,
                0,
                state="post",
                display_clock="FT",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(
        timezone_offset=-3,
        team_query="corea del sur",
    )

    assert result.startswith("Polymarket - Mundial: Corea del Sur")
    assert "[🇰🇷 Corea del Sur 1] vs. 🇯🇵 Japón 0" in result


def test_get_polymarket_world_cup_games_rejects_unknown_country(monkeypatch):
    fetch_scores = MagicMock(return_value={})
    result = polymarket_commands.get_world_cup_games(
        -3,
        fetch_winner_event=lambda _slug: None,
        cached_request=lambda *_args, **_kwargs: None,
        cache_ttl=0,
        fetch_live=lambda _token_id: None,
        format_country=flagged_country_name,
        make_timezone=lambda offset: timezone(timedelta(hours=offset)),
        fetch_scores=fetch_scores,
        team_query="narnia",
    )

    assert result == "No encontré ese país en el Mundial"
    fetch_scores.assert_not_called()


def test_get_polymarket_world_cup_games_omits_scheduled_zero_zero_scores(
    monkeypatch,
):
    event = {
        "title": "Team A vs. Team B",
        "slug": "fifwc-a-b-2026-06-11",
        "endDate": "2026-06-11T19:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Team A",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.60", "0.40"]',
            },
            {
                "groupItemTitle": "Team B",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            },
        ],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": [event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Team A",
                "Team B",
                0,
                0,
                state="pre",
                display_clock="0'",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "Team A 0-0 Team B" not in result


def test_get_polymarket_world_cup_games_omits_odds_for_final_scores(
    monkeypatch,
):
    event = {
        "title": "Team A vs. Team B",
        "slug": "fifwc-a-b-2026-06-11",
        "endDate": "2026-06-11T19:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Team A",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.60", "0.40"]',
            },
            {
                "groupItemTitle": "Team B",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.40", "0.60"]',
            },
        ],
    }

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": [event]}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Team A",
                "Team B",
                2,
                1,
                state="post",
                display_clock="90'+5'",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert ">[Team A 2] vs. Team B 1</a>" in result
    assert "Team A 2 (60%)" not in result
    assert "Team B 1 (40%)" not in result
    assert "\nFinal" in result


def test_get_polymarket_world_cup_games_keeps_only_two_latest_final_matches(
    monkeypatch,
):
    def event(slug, title, end_date, first_team, second_team):
        return {
            "title": title,
            "slug": slug,
            "endDate": end_date,
            "markets": [
                {
                    "groupItemTitle": first_team,
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.60", "0.40"]',
                },
                {
                    "groupItemTitle": second_team,
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.30", "0.70"]',
                },
            ],
        }

    events = [
        event(
            "fifwc-a-b-2026-06-11",
            "Team A vs. Team B",
            "2026-06-11T19:00:00Z",
            "Team A",
            "Team B",
        ),
        event(
            "fifwc-c-d-2026-06-11",
            "Team C vs. Team D",
            "2026-06-11T22:00:00Z",
            "Team C",
            "Team D",
        ),
        event(
            "fifwc-e-f-2026-06-12",
            "Team E vs. Team F",
            "2026-06-12T01:00:00Z",
            "Team E",
            "Team F",
        ),
        event(
            "fifwc-g-h-2026-06-12",
            "Team G vs. Team H",
            "2026-06-12T04:00:00Z",
            "Team G",
            "Team H",
        ),
    ]

    def fake_cached_requests(_url, parameters, *_args):
        if parameters == {"slug": "world-cup-winner"}:
            return None
        return {"data": events}

    monkeypatch.setattr(index.app_runtime.cache, "request", fake_cached_requests)
    monkeypatch.setattr(
        "api.markets.polymarket.fetch_scoreboard_scores",
        lambda: {
            "game-1": _match_score(
                "game-1",
                "Team A",
                "Team B",
                1,
                0,
                state="post",
                display_clock="FT",
                start_time="2026-06-11T19:00:00Z",
            ),
            "game-2": _match_score(
                "game-2",
                "Team C",
                "Team D",
                2,
                1,
                state="post",
                display_clock="FT",
                start_time="2026-06-11T22:00:00Z",
            ),
            "game-3": _match_score(
                "game-3",
                "Team E",
                "Team F",
                3,
                2,
                state="post",
                display_clock="FT",
                start_time="2026-06-12T01:00:00Z",
            ),
            "game-4": _match_score(
                "game-4",
                "Team G",
                "Team H",
                state="pre",
                display_clock="0'",
                start_time="2026-06-12T04:00:00Z",
            ),
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "fifwc-a-b-2026-06-11" not in result
    assert "fifwc-c-d-2026-06-11" in result
    assert "fifwc-e-f-2026-06-12" in result
    assert "fifwc-g-h-2026-06-12" in result
    assert result.index("fifwc-c-d-2026-06-11") < result.index(
        "fifwc-e-f-2026-06-12"
    )
    assert result.index("fifwc-e-f-2026-06-12") < result.index(
        "fifwc-g-h-2026-06-12"
    )


def test_country_flags_use_iso_data_with_sports_aliases():
    assert flagged_country_name("Romania") == "🇷🇴 Romania"
    assert flagged_country_name("Türkiye") == "🇹🇷 Turquía"
    assert flagged_country_name("Korea Republic") == "🇰🇷 Corea del Sur"
    assert flagged_country_name("IR Iran") == "🇮🇷 Irán"
    assert flagged_country_name("Congo DR") == "🇨🇩 Congo"
    assert flagged_country_name("DR Congo") == "🇨🇩 Congo"
    assert flagged_country_name("Ivory Coast") == "🇨🇮 Costa de Marfil"
    assert flagged_country_name("Côte d'Ivoire") == "🇨🇮 Costa de Marfil"
    assert flagged_country_name("Cabo Verde") == "🇨🇻 Cabo Verde"
    assert flagged_country_name("Cape Verde") == "🇨🇻 Cabo Verde"
    assert flagged_country_name("New Zealand") == "🇳🇿 Nueva Zelanda"
    assert flagged_country_name("Bosnia-Herzegovina") == (
        "🇧🇦 Bosnia y Herzegovina"
    )
    assert flagged_country_name("England") == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e"
        "\U000e0067\U000e007f Inglaterra"
    )
    assert flagged_country_name("Scotland") == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063"
        "\U000e0074\U000e007f Escocia"
    )
    assert flagged_country_name("Wales") == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c"
        "\U000e0073\U000e007f Wales"
    )
    assert flagged_country_name("UK") == "🇬🇧 UK"


def test_all_ranked_world_cup_teams_have_flags_and_spanish_display_names():
    assert set(TEAM_NAMES_ES) == set(WORLD_CUP_TEAM_RANKING)
    for team in WORLD_CUP_TEAM_RANKING:
        flag = country_flag_from_name(team)
        assert flag, team
        assert flagged_country_name(team) == f"{flag} {team_name_es(team)}"


def test_all_provider_team_aliases_have_flags_and_canonical_display_names():
    for alias, canonical in TEAM_NAME_ALIASES.items():
        flag = country_flag_from_name(alias)
        assert flag, alias
        assert flagged_country_name(alias) == f"{flag} {team_name_es(canonical)}"
