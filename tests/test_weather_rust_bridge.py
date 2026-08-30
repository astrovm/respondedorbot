from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from unittest.mock import MagicMock

import pytest

from api.markets import weather


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FixedDatetime:
        return cls(2026, 1, 2, 10, 30, tzinfo=tz)


class _FakeRustWeather:
    def __init__(self, *, fail_location: bool = False, fail_hour: bool = False) -> None:
        self.fail_location = fail_location
        self.fail_hour = fail_hour
        self.location_calls: list[tuple[list[str], list[str]]] = []
        self.hour_calls: list[tuple[list[str], str | None, str]] = []

    def select_weather_location(
        self, qualifier_keys: list[str], candidate_keys: list[str]
    ) -> int | None:
        self.location_calls.append((qualifier_keys, candidate_keys))
        if self.fail_location:
            raise ValueError("synthetic location failure")
        return 1

    def select_weather_hour(
        self,
        forecast_hours: list[str],
        provider_hour: str | None,
        local_hour: str,
    ) -> int | None:
        self.hour_calls.append((forecast_hours, provider_hour, local_hour))
        if self.fail_hour:
            raise ValueError("synthetic hour failure")
        return 0


def _responses() -> list[dict[str, object]]:
    return [
        {
            "data": {
                "results": [
                    {
                        "name": "Example City",
                        "country": "Otherland",
                        "latitude": 1.0,
                        "longitude": 2.0,
                    },
                    {
                        "name": "Example City",
                        "country": "Exampleland",
                        "latitude": 3.0,
                        "longitude": 4.0,
                    },
                ]
            }
        },
        {
            "data": {
                "current": {"time": "2026-01-02T10:00"},
                "hourly": {
                    "time": ["2026-01-02T10:00"],
                    "apparent_temperature": [19.5],
                    "precipitation_probability": [20],
                    "weather_code": [1],
                    "cloud_cover": [30],
                    "visibility": [15000],
                },
            }
        },
    ]


def test_weather_uses_rust_selection_without_changing_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustWeather()
    monkeypatch.setattr(weather, "_load_rust_weather_selector", lambda: rust)
    cached_request = MagicMock(side_effect=_responses())

    result = weather.get_weather(
        "Example City, Exampleland",
        cached_request=cached_request,
        cache_ttl=300,
        local_timezone=UTC,
        datetime_type=_FixedDatetime,
        logger=MagicMock(),
    )

    assert result["location"] == "Example City, Exampleland"
    assert result["apparent_temperature"] == 19.5
    assert cached_request.call_count == 2
    assert rust.location_calls == [
        (["exampleland"], [" otherland ", " exampleland "])
    ]
    assert rust.hour_calls == [
        (["2026-01-02T10"], "2026-01-02T10", "2026-01-02T10")
    ]


@pytest.mark.parametrize("failure", ["location", "hour"])
def test_weather_falls_back_to_python_after_rust_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    rust = _FakeRustWeather(
        fail_location=failure == "location",
        fail_hour=failure == "hour",
    )
    monkeypatch.setattr(weather, "_load_rust_weather_selector", lambda: rust)

    result = weather.get_weather(
        "Example City, Exampleland",
        cached_request=MagicMock(side_effect=_responses()),
        cache_ttl=300,
        local_timezone=UTC,
        datetime_type=_FixedDatetime,
        logger=MagicMock(),
    )

    assert result["location"] == "Example City, Exampleland"
    assert result["weather_code"] == 1
