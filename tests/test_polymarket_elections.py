from api import index


def test_get_polymarket_global_elections_requests_and_formats_top_liquidity(
    monkeypatch,
):
    captured = {}
    events = [
        {
            "title": "Lower liquidity election",
            "slug": "lower-election",
            "liquidity": 1_000,
            "endDate": "2026-08-01T00:00:00Z",
            "markets": [],
        },
        {
            "title": "Higher liquidity election",
            "slug": "higher-election",
            "liquidity": 2_500_000,
            "endDate": "2027-04-30T00:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Candidate A",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.42", "0.58"]',
                },
                {
                    "groupItemTitle": "Candidate B",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.61", "0.39"]',
                },
                {
                    "groupItemTitle": "Candidate C",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.20", "0.80"]',
                },
                {
                    "groupItemTitle": "Inactive placeholder",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.99", "0.01"]',
                    "active": False,
                    "closed": False,
                },
                {
                    "groupItemTitle": "Closed candidate",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.98", "0.02"]',
                    "active": True,
                    "closed": True,
                },
            ],
        },
    ]

    def fake_cached_requests(_url, parameters, *_args):
        captured.update(parameters)
        return {"data": events}

    monkeypatch.setattr(index, "cached_requests", fake_cached_requests)

    result = index.get_polymarket_global_elections()

    assert captured == {
        "limit": 10,
        "active": "true",
        "closed": "false",
        "tag_slug": "global-elections",
        "order": "liquidity",
        "ascending": "false",
    }
    assert result.index("Higher liquidity election") < result.index(
        "Lower liquidity election"
    )
    assert "Candidate B 61% | Candidate A 42%" in result
    assert "Candidate C" not in result
    assert "Inactive placeholder" not in result
    assert "Closed candidate" not in result
    assert "Liquidity US$2.5M | Closes 2027-04-30" in result
    assert (
        '<a href="https://polymarket.com/event/higher-election">'
        "Higher liquidity election</a>"
    ) in result
    assert "\nhttps://polymarket.com/event/higher-election" not in result


def test_get_polymarket_global_elections_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index, "cached_requests", lambda *_args, **_kwargs: None)

    assert (
        index.get_polymarket_global_elections()
        == "No pude traer las elecciones desde Polymarket"
    )
