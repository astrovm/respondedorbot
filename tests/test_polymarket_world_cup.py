from api import index


def test_get_polymarket_world_cup_games_filters_props_and_formats_kickoff(
    monkeypatch,
):
    captured = {}
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

    def fake_cached_requests(_url, parameters, *_args):
        captured.update(parameters)
        return {"data": events}

    monkeypatch.setattr(index, "cached_requests", fake_cached_requests)

    result = index.get_polymarket_world_cup_games()

    assert captured == {
        "limit": 100,
        "active": "true",
        "closed": "false",
        "series_id": 11433,
        "order": "endDate",
        "ascending": "true",
    }
    assert result.index("Mexico vs. South Africa") < result.index(
        "Korea Republic vs. Czechia"
    )
    assert "Exact Score" not in result
    assert "Mexico 69.5% | Draw 20.5% | South Africa 10.5%" in result
    assert "2026-06-11 16:00 UTC-3" in result
    assert (
        '<a href="https://polymarket.com/sports/world-cup/'
        'fifwc-mex-rsa-2026-06-11">Mexico vs. South Africa</a>'
    ) in result


def test_get_polymarket_world_cup_games_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index, "cached_requests", lambda *_args, **_kwargs: None)

    assert (
        index.get_polymarket_world_cup_games()
        == "Could not fetch World Cup games from Polymarket"
    )
