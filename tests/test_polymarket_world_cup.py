from api import index


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
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Draw (Mexico vs. South Africa)",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.205", "0.795"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "groupItemTitle": "South Africa",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.105", "0.895"]',
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

    monkeypatch.setattr(index, "cached_requests", fake_cached_requests)

    result = index.get_polymarket_world_cup_games()

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
        "🇪🇸 Spain 18% | 🇫🇷 France 16% | 🇧🇷 Brazil 14% | "
        "🇬🇧 England 12% | 🇦🇷 Argentina 10%"
    ) in result
    assert "Germany" not in result
    assert "Next 8 games" not in result
    assert result.index("fifwc-mex-rsa-2026-06-11") < result.index(
        "fifwc-kr-cze-2026-06-11"
    )
    assert "Exact Score" not in result
    assert "[🇲🇽 Mexico 69.5%] vs. 🇿🇦 South Africa 10.5%" in result
    assert "Draw 20.5%" not in result
    assert "2026-06-11 16:00 UTC-3" in result
    assert (
        '<a href="https://polymarket.com/sports/world-cup/'
        'fifwc-mex-rsa-2026-06-11">[🇲🇽 Mexico 69.5%] vs. '
        "🇿🇦 South Africa 10.5%</a>"
    ) in result


def test_get_polymarket_world_cup_games_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index, "cached_requests", lambda *_args, **_kwargs: None)

    assert (
        index.get_polymarket_world_cup_games()
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

    monkeypatch.setattr(index, "cached_requests", fake_cached_requests)

    result = index.get_polymarket_world_cup_games()

    assert ">Team A 35% vs. Team B 25%</a>" in result
    assert "[Draw]" not in result


def test_country_flags_use_iso_data_with_sports_aliases():
    assert index._flagged_country_name("Romania") == "🇷🇴 Romania"
    assert index._flagged_country_name("Türkiye") == "🇹🇷 Türkiye"
    assert index._flagged_country_name("Korea Republic") == "🇰🇷 Korea Republic"
    assert index._flagged_country_name("Bosnia-Herzegovina") == (
        "🇧🇦 Bosnia-Herzegovina"
    )
