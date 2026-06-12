from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from logging import Logger
from typing import Any

CachedRequest = Callable[..., dict[str, Any] | None]
WeatherGetter = Callable[[], dict[str, Any] | None]
DescriptionGetter = Callable[[int], str]


def get_weather(
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
    local_timezone: timezone,
    datetime_type: type[datetime],
    logger: Logger,
) -> dict[str, Any]:
    try:
        response = cached_request(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": -34.5429,
                "longitude": -58.7119,
                "hourly": (
                    "apparent_temperature,precipitation_probability,"
                    "weather_code,cloud_cover,visibility"
                ),
                "timezone": "auto",
                "forecast_days": 2,
            },
            None,
            cache_ttl,
        )
        if not response or "data" not in response:
            return {}

        hourly = response["data"]["hourly"]
        current_time = datetime_type.now(local_timezone)
        current_index = None
        for index, timestamp in enumerate(hourly["time"]):
            forecast_time = datetime_type.fromisoformat(timestamp)
            if (
                forecast_time.year == current_time.year
                and forecast_time.month == current_time.month
                and forecast_time.day == current_time.day
                and forecast_time.hour == current_time.hour
            ):
                current_index = index
                break

        if current_index is None:
            return {}
        return {
            "apparent_temperature": hourly["apparent_temperature"][current_index],
            "precipitation_probability": hourly["precipitation_probability"][
                current_index
            ],
            "weather_code": hourly["weather_code"][current_index],
            "cloud_cover": hourly["cloud_cover"][current_index],
            "visibility": hourly["visibility"][current_index],
        }
    except Exception:
        logger.exception("Error getting weather")
        return {}


def get_weather_context(
    *,
    get_weather_data: WeatherGetter,
    get_description: DescriptionGetter,
    logger: Logger,
) -> dict[str, Any] | None:
    try:
        weather = get_weather_data()
        if weather:
            weather["description"] = get_description(weather["weather_code"])
        return weather
    except Exception:
        logger.exception("Error fetching weather data")
        return None
