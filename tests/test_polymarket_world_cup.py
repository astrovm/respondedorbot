from api import index
from api.markets.world_cup_goals import MatchScore
from api.markets.polymarket import flagged_country_name
from api.markets import polymarket as polymarket_commands


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
            "markets": [],
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
            "game-1": MatchScore(
                event_id="game-1",
                home_team="South Africa",
                away_team="Mexico",
                home_score=1,
                away_score=2,
                state="in",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert captured == [
        {"slug": "world-cup-winner"},
        {
        "limit": 100,
        "active": "true",
        "closed": "false",
        "series_id": 11433,
        "order": "endDate",
        "ascending": "true",
        },
    ]
    assert "World Cup Winner</a>" in result
    assert (
        "🇪🇸 España 18% | 🇫🇷 Francia 16% | 🇧🇷 Brasil 14% | "
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e"
        "\U000e0067\U000e007f Inglaterra 12% | 🇦🇷 Argentina 10%"
    ) in result
    assert "Germany" not in result
    assert "Next 8 games" not in result
    assert result.index("fifwc-mex-rsa-2026-06-11") < result.index(
        "fifwc-kr-cze-2026-06-11"
    )
    assert "Exact Score" not in result
    assert "[🇲🇽 México 2 (72%)] vs. 🇿🇦 Sudáfrica 1 (10%)" in result
    assert "Mexico 2-1 South Africa" not in result
    assert "· live" not in result
    assert "Mexico 69.5%" not in result
    assert "Draw 18%" not in result
    assert "Thu, June 11\n<a href=" in result
    assert "16:00 UTC-3\n\n<a href=" in result
    assert (
        '<a href="https://polymarket.com/sports/world-cup/'
        'fifwc-mex-rsa-2026-06-11">[🇲🇽 México 2 (72%)] vs. '
        "🇿🇦 Sudáfrica 1 (10%)</a>"
    ) in result


def test_get_polymarket_world_cup_games_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index.app_runtime.cache, "request", lambda *_args, **_kwargs: None)

    assert (
        index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)
        == "Could not fetch World Cup games from Polymarket"
    )


def test_get_polymarket_world_cup_games_does_not_mark_draw_as_favorite(
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
                "outcomePrices": '["0.40", "0.60"]',
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

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert ">Team A 35% vs. Team B 25%</a>" in result
    assert "[Draw]" not in result


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
            "game-1": MatchScore(
                event_id="game-1",
                home_team="Team A",
                away_team="Team B",
                home_score=0,
                away_score=0,
                state="pre",
            )
        },
    )

    result = index.app_runtime.polymarket.get_world_cup_games(timezone_offset=-3)

    assert "Team A 0-0 Team B" not in result


def test_country_flags_use_iso_data_with_sports_aliases():
    assert flagged_country_name("Romania") == "🇷🇴 Romania"
    assert flagged_country_name("Türkiye") == "🇹🇷 Turquía"
    assert flagged_country_name("Korea Republic") == "🇰🇷 Corea del Sur"
    assert flagged_country_name("IR Iran") == "🇮🇷 Irán"
    assert flagged_country_name("Congo DR") == "🇨🇩 Congo"
    assert flagged_country_name("Ivory Coast") == "🇨🇮 Costa de Marfil"
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
