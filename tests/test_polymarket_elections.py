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
    assert result.index("- Candidate B: 61%") < result.index("- Candidate A: 42%")
    assert "Candidate C" not in result
    assert "- Liquidez: US$2.5M" in result
    assert "- Cierre: 2027-04-30" in result
    assert "https://polymarket.com/event/higher-election" in result


def test_get_polymarket_global_elections_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index, "cached_requests", lambda *_args, **_kwargs: None)

    assert (
        index.get_polymarket_global_elections()
        == "No pude traer las elecciones desde Polymarket"
    )
