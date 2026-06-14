"""TCRM calculation and cache coordination."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import redis

from api.core.logging import get_logger
from api.utils import parse_date_string, parse_monetary_number, to_ddmmyy

logger = get_logger(__name__)

BA_TZ = timezone(timedelta(hours=-3))
TCRM_CACHE_KEY = "tcrm_100"

RedisFactory = Callable[[], redis.Redis]
RedisJsonGetter = Callable[[redis.Redis, str], Any]
RedisJsonSetter = Callable[..., Any]
WholesaleGetter = Callable[[str, str], float | None]
MissingWholesaleWriter = Callable[..., None]
LatestITCRMGetter = Callable[[], tuple[float, str] | None]
HistoricalITCRMGetter = Callable[[datetime], tuple[float, datetime] | None]
TCRMCalculatorFn = Callable[..., float | None]
CacheHistoryGetter = Callable[[int, str, redis.Redis], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class TCRMCalculationDeps:
    redis_factory: RedisFactory
    get_json: RedisJsonGetter
    set_json: RedisJsonSetter
    setex_json: RedisJsonSetter | None
    get_wholesale_rate: WholesaleGetter
    cache_missing_wholesale: MissingWholesaleWriter
    get_latest_itcrm: LatestITCRMGetter
    get_historical_itcrm: HistoricalITCRMGetter


@dataclass(frozen=True, slots=True)
class TCRMCacheDeps:
    redis_factory: RedisFactory
    get_json: RedisJsonGetter
    set_json: RedisJsonSetter
    setex_json: RedisJsonSetter | None
    calculate: TCRMCalculatorFn
    get_latest_itcrm: LatestITCRMGetter
    get_wholesale_rate: WholesaleGetter
    cache_missing_wholesale: MissingWholesaleWriter
    get_history: CacheHistoryGetter


def _normalize_target_date(value: str | datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return parse_date_string(str(value))


def _parse_itcrm(
    target_date: datetime | None,
    deps: TCRMCalculationDeps,
) -> tuple[float, datetime] | None:
    if target_date is not None:
        return deps.get_historical_itcrm(target_date)

    details = deps.get_latest_itcrm()
    if not details:
        return None
    raw_value, raw_date = details
    parsed_date = parse_date_string(raw_date)
    if parsed_date is None:
        return None
    try:
        return float(raw_value), parsed_date
    except (TypeError, ValueError):
        return None


def _redis_or_none(factory: RedisFactory) -> redis.Redis | None:
    try:
        return factory()
    except Exception:
        return None


def _parse_cached_number(value: Any) -> float | None:
    if value is None:
        return None
    parsed = parse_monetary_number(value)
    return float(parsed) if parsed is not None else None


class TCRMCalculator:
    """Compute the nominal wholesale rate that would set ITCRM to 100."""

    def __init__(self, deps: TCRMCalculationDeps) -> None:
        self._deps = deps

    def _cached_wholesale(
        self,
        client: redis.Redis,
        date_key: str,
    ) -> tuple[float | None, bool]:
        try:
            cached = self._deps.get_json(client, f"bcra_mayorista:{date_key}")
        except Exception:
            return None, False
        if not isinstance(cached, Mapping):
            return None, False
        if cached.get("missing"):
            return None, True
        return _parse_cached_number(cached.get("value")), False

    def _fetch_wholesale(
        self,
        date_key: str,
        client: redis.Redis | None,
    ) -> float | None:
        fetched = self._deps.get_wholesale_rate(
            "tipo de cambio mayorista",
            date_key,
        )
        if fetched is not None:
            return float(fetched)
        self._deps.cache_missing_wholesale(
            date_key,
            client,
            redis_setex_json_fn=self._deps.setex_json,
        )
        return None

    def calculate(
        self,
        target_date: str | datetime | date | None = None,
    ) -> float | None:
        normalized_target = _normalize_target_date(target_date)
        if target_date is not None and normalized_target is None:
            return None

        itcrm = _parse_itcrm(normalized_target, self._deps)
        if itcrm is None:
            return None
        itcrm_value, itcrm_date = itcrm
        date_key = itcrm_date.date().isoformat()
        client = _redis_or_none(self._deps.redis_factory)

        wholesale: float | None = None
        if client is not None:
            wholesale, known_missing = self._cached_wholesale(client, date_key)
            if known_missing:
                return None
        if wholesale is None:
            wholesale = self._fetch_wholesale(date_key, client)
        if wholesale is None:
            return None

        if client is not None:
            try:
                self._deps.set_json(
                    client,
                    f"bcra_mayorista:{date_key}",
                    {"value": wholesale, "date": to_ddmmyy(date_key)},
                )
            except Exception:
                pass
        return wholesale * 100 / itcrm_value


def _cached_itcrm_date(
    client: redis.Redis,
    deps: TCRMCacheDeps,
) -> datetime | None:
    cached = deps.get_json(client, "latest_itcrm_details")
    if isinstance(cached, Mapping) and "date" in cached:
        return parse_date_string(str(cached.get("date", "")))
    details = deps.get_latest_itcrm()
    return parse_date_string(details[1]) if details else None


class TCRMCacheService:
    """Coordinate current TCRM freshness and optional historical comparison."""

    def __init__(self, deps: TCRMCacheDeps) -> None:
        self._deps = deps

    def _has_current_wholesale(self, client: redis.Redis) -> bool:
        try:
            itcrm_date = _cached_itcrm_date(client, self._deps)
            if itcrm_date is None:
                return False
            date_key = itcrm_date.date().isoformat()
            cached = self._deps.get_json(
                client,
                f"bcra_mayorista:{date_key}",
            )
            if isinstance(cached, Mapping):
                if cached.get("missing"):
                    return False
                if _parse_cached_number(cached.get("value")) is not None:
                    return True

            fetched = self._deps.get_wholesale_rate(
                "tipo de cambio mayorista",
                date_key,
            )
            if fetched is not None:
                return True
            self._deps.cache_missing_wholesale(
                date_key,
                client,
                redis_setex_json_fn=self._deps.setex_json,
            )
        except Exception:
            return False
        return False

    def _history(
        self,
        client: redis.Redis,
        hours_ago: int,
    ) -> Mapping[str, Any] | None:
        try:
            history = self._deps.get_history(hours_ago, TCRM_CACHE_KEY, client)
        except Exception:
            history = None
        if history is not None or not hours_ago:
            return history

        try:
            history_dt = datetime.now(BA_TZ) - timedelta(hours=hours_ago)
            history_key = history_dt.strftime("%Y-%m-%d-%H") + TCRM_CACHE_KEY
            if client.get(history_key):
                return None
            value = self._deps.calculate(target_date=history_dt)
            if value is None:
                return None
            history = {
                "timestamp": int(
                    history_dt.replace(
                        minute=0,
                        second=0,
                        microsecond=0,
                    ).timestamp()
                ),
                "data": value,
            }
            self._deps.set_json(client, history_key, history)
            return history
        except Exception:
            return None

    def _compute_and_store(
        self,
        client: redis.Redis,
        timestamp: int,
    ) -> float | None:
        value = self._deps.calculate()
        if value is None:
            return None
        payload = {"timestamp": timestamp, "data": value}
        self._deps.set_json(client, TCRM_CACHE_KEY, payload)
        current_hour = datetime.now(BA_TZ).strftime("%Y-%m-%d-%H")
        self._deps.set_json(client, current_hour + TCRM_CACHE_KEY, payload)
        return value

    def get(
        self,
        hours_ago: int = 24,
        expiration_time: int = 300,
    ) -> tuple[float | None, float | None]:
        try:
            client = self._deps.redis_factory()
            cached = self._deps.get_json(client, TCRM_CACHE_KEY)
            timestamp = int(time.time())
            history = self._history(client, hours_ago)
            history_value = history.get("data") if history else None

            if not self._has_current_wholesale(client):
                current = None
            elif not isinstance(cached, Mapping):
                current = self._compute_and_store(client, timestamp)
            else:
                cache_age = timestamp - int(cached["timestamp"])
                current = (
                    self._compute_and_store(client, timestamp)
                    if cache_age > expiration_time
                    else cached.get("data")
                )

            change = None
            if current is not None and history_value:
                change = ((float(current) / float(history_value)) - 1) * 100
            return (
                float(current) if current is not None else None,
                change,
            )
        except Exception as exc:
            logger.error("Error getting cached TCRM 100: %s", exc)
            return None, None
