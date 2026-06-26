from api import index
from api.markets.world_cup_goals import MatchScore
from api.markets.polymarket import flagged_country_name
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
        lambda: {
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
        lambda: {
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
        lambda: {
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
        lambda: {
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
