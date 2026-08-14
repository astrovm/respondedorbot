from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from logging import Logger
from typing import Any
import unicodedata

CachedRequest = Callable[..., dict[str, Any] | None]
WeatherGetter = Callable[[str], dict[str, Any] | None]
DescriptionGetter = Callable[[int], str]

DEFAULT_WEATHER_LOCATION = "Buenos Aires"
_DEFAULT_LOCATION = {
    "name": DEFAULT_WEATHER_LOCATION,
    "country": "Argentina",
    "latitude": -34.6037,
    "longitude": -58.3816,
}
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _search_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _resolve_location(
    location: str,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
) -> dict[str, Any] | None:
    query = location.strip() or DEFAULT_WEATHER_LOCATION
    if query.casefold() in {"buenos aires", "caba"}:
        return dict(_DEFAULT_LOCATION)

    parts = [part.strip() for part in query.split(",") if part.strip()]
    search_name = parts[0] if len(parts) > 1 else query
    qualifiers = parts[1:]

    def search(name: str) -> list[dict[str, Any]]:
        response = cached_request(
            _GEOCODING_URL,
            {"name": name, "count": 10, "language": "es", "format": "json"},
            None,
            cache_ttl,
        )
        payload = response.get("data", {}) if response else {}
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return [result for result in results if isinstance(result, dict)]

    results = search(search_name)
    if not results and not qualifiers and " " in query:
        shorter_name, qualifier = query.rsplit(maxsplit=1)
        results = search(shorter_name)
        qualifiers = [qualifier]
    if not results:
        return None

    wanted = [_search_key(qualifier) for qualifier in qualifiers]
    if wanted:

        def score(result: dict[str, Any]) -> int:
            candidate = _search_key(
                " ".join(
                    str(result.get(key) or "") for key in ("admin1", "country", "country_code")
                )
            )
            return sum(qualifier in candidate for qualifier in wanted)

        best = max(results, key=score)
        if score(best):
            return best
    return results[0]


def _location_label(location: dict[str, Any]) -> str:
    parts = [location.get("name"), location.get("admin1"), location.get("country")]
    return ", ".join(dict.fromkeys(str(part) for part in parts if part))


def get_weather(
    location: str = DEFAULT_WEATHER_LOCATION,
    *,
    cached_request: CachedRequest,
    cache_ttl: int,
    local_timezone: timezone,
    datetime_type: type[datetime],
    logger: Logger,
) -> dict[str, Any]:
    try:
        resolved = _resolve_location(
            location,
            cached_request=cached_request,
            cache_ttl=cache_ttl,
        )
        if not resolved:
            return {}
        response = cached_request(
            _FORECAST_URL,
            {
                "latitude": resolved["latitude"],
                "longitude": resolved["longitude"],
                "current": "weather_code",
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
        current_timestamp = response["data"].get("current", {}).get("time")
        current_time = datetime_type.now(local_timezone)
        current_index = None
        for index, timestamp in enumerate(hourly["time"]):
            if current_timestamp and timestamp[:13] == current_timestamp[:13]:
                current_index = index
                break
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
            "location": _location_label(resolved),
            "apparent_temperature": hourly["apparent_temperature"][current_index],
            "precipitation_probability": hourly["precipitation_probability"][current_index],
            "weather_code": hourly["weather_code"][current_index],
            "cloud_cover": hourly["cloud_cover"][current_index],
            "visibility": hourly["visibility"][current_index],
        }
    except Exception:
        logger.exception("Error getting weather")
        return {}


def get_weather_context(
    location: str = DEFAULT_WEATHER_LOCATION,
    *,
    get_weather_data: WeatherGetter,
    get_description: DescriptionGetter,
    logger: Logger,
) -> dict[str, Any] | None:
    try:
        weather = get_weather_data(location)
        if weather:
            weather["description"] = get_description(weather["weather_code"])
        return weather
    except Exception:
        logger.exception("Error fetching weather data")
        return None
