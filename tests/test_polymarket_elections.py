from api import index
from api import polymarket_commands


def test_fetch_live_prices_uses_clob_midpoints():
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"token-a": "0.72", "token-b": "invalid"}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse()

    result = polymarket_commands.fetch_live_prices(
        ["token-a", "token-b", "token-a"],
        http_post=fake_post,
    )

    assert result == {"token-a": 0.72}
    assert captured == {
        "url": "https://clob.polymarket.com/midpoints",
        "json": [{"token_id": "token-a"}, {"token_id": "token-b"}],
        "timeout": 5,
    }


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
            "tags": [{"slug": "united-states"}],
            "liquidity": 2_500_000,
            "endDate": "2027-04-30T00:00:00Z",
            "markets": [
                {
                    "groupItemTitle": "Candidate A",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.42", "0.58"]',
                    "clobTokenIds": '["candidate-a", "candidate-a-no"]',
                },
                {
                    "groupItemTitle": "Candidate B",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.61", "0.39"]',
                    "clobTokenIds": '["candidate-b", "candidate-b-no"]',
                },
                {
                    "groupItemTitle": "Candidate C",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.20", "0.80"]',
                    "clobTokenIds": '["candidate-c", "candidate-c-no"]',
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
    monkeypatch.setattr(
        index,
        "_fetch_polymarket_live_prices",
        lambda token_ids: {
            "candidate-a": 0.72,
            "candidate-b": 0.55,
        },
    )

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
    assert "Candidate A 72% | Candidate B 55%" in result
    assert "Candidate B 61%" not in result
    assert "Candidate C" not in result
    assert "Inactive placeholder" not in result
    assert "Closed candidate" not in result
    assert "Liquidity US$2.5M | Closes 2027-04-30" in result
    assert (
        '<a href="https://polymarket.com/event/higher-election">'
        "🇺🇸 Higher liquidity election</a>"
    ) in result
    assert "\nhttps://polymarket.com/event/higher-election" not in result


def test_get_polymarket_global_elections_handles_empty_response(monkeypatch):
    monkeypatch.setattr(index, "cached_requests", lambda *_args, **_kwargs: None)

    assert (
        index.get_polymarket_global_elections()
        == "No pude traer las elecciones desde Polymarket"
    )


def test_event_country_flag_resolves_new_standard_country_tag():
    assert index._event_country_flag({"tags": [{"slug": "armenia"}]}) == "🇦🇲"


def test_event_country_flag_resolves_uk_regional_tags():
    assert index._event_country_flag({"tags": [{"slug": "england"}]}) == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e"
        "\U000e0067\U000e007f"
    )
    assert index._event_country_flag({"tags": [{"slug": "scotland"}]}) == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063"
        "\U000e0074\U000e007f"
    )
    assert index._event_country_flag({"tags": [{"slug": "wales"}]}) == (
        "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c"
        "\U000e0073\U000e007f"
    )
    assert index._event_country_flag({"tags": [{"slug": "uk"}]}) == "🇬🇧"
