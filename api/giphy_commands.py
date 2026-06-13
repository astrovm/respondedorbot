from __future__ import annotations

import json
import random
from collections.abc import Callable
from logging import Logger
from os import environ
import redis

from api.config_runtime import ConfigRuntime
from api.services import http_client
from api.services.maintenance import GIPHY_STALE_TTL
from api.services.redis_helpers import redis_get_json

GIPHY_API_URL = "https://api.giphy.com/v1/gifs"
TTL_GIPHY_POOL = 86400

GIPHY_GM_TERMS = [
    "good morning",
    "buenos dias",
    "morning coffee",
    "rise and shine",
]
GIPHY_GN_TERMS = [
    "good night",
    "buenas noches",
    "sweet dreams",
    "go to sleep",
]

RedisFactory = Callable[[], redis.Redis | None]
PoolFetcher = Callable[[str], list[str]]
PoolGetter = Callable[[str], list[str]]
GifGetter = Callable[[str], str | None]


def fetch_giphy_pool(category: str, *, logger: Logger) -> list[str]:
    """Fetch a pool of GIF URLs from Giphy for a category."""
    api_key = environ.get("GIPHY_API_KEY")
    if not api_key:
        return []

    terms = GIPHY_GM_TERMS if category == "gm" else GIPHY_GN_TERMS
    urls: list[str] = []

    for term in terms:
        try:
            params = {
                "api_key": api_key,
                "q": term,
                "limit": 25,
                "offset": random.randint(0, 100),
                "rating": "g",
            }
            response = http_client.get(
                f"{GIPHY_API_URL}/search", params=params, timeout=5
            )
            response.raise_for_status()
            data = response.json()
            for gif in data.get("data", []):
                url = gif.get("images", {}).get("original", {}).get("url")
                if url:
                    urls.append(url)
        except Exception:
            logger.exception("Error fetching Giphy pool for %s", category)

    return urls


def get_giphy_pool(
    category: str,
    *,
    redis_factory: RedisFactory,
    fetch_pool: PoolFetcher,
    logger: Logger,
) -> list[str]:
    """Return the current pool, falling back to stale cached data."""
    pool_key = f"giphy_pool:{category}"
    stale_key = f"giphy_pool_stale:{category}"
    redis_client = redis_factory()

    if redis_client:
        try:
            cached = redis_get_json(redis_client, pool_key)
            if isinstance(cached, list):
                return [str(url) for url in cached]
        except Exception:
            logger.exception("Error reading Giphy pool from cache")

    urls = fetch_pool(category)

    if urls and redis_client:
        try:
            encoded_urls = json.dumps(urls)
            redis_client.setex(pool_key, TTL_GIPHY_POOL, encoded_urls)
            redis_client.setex(stale_key, GIPHY_STALE_TTL, encoded_urls)
        except Exception:
            logger.exception("Error caching Giphy pool")
    elif not urls and redis_client:
        try:
            stale = redis_get_json(redis_client, stale_key)
            if isinstance(stale, list):
                logger.warning("Giphy API failed, using stale pool for %s", category)
                return [str(url) for url in stale]
        except Exception:
            logger.exception("Error reading stale Giphy pool")

    return urls


def get_random_gif(category: str, *, get_pool: PoolGetter) -> str | None:
    pool = get_pool(category)
    return random.choice(pool) if pool else None


def get_good_morning(*, get_gif: GifGetter) -> str:
    return get_gif("gm") or "buen día boludo"


def get_good_night(*, get_gif: GifGetter) -> str:
    return get_gif("gn") or "buenas noches boludo"


class GiphyService:
    def __init__(self, *, config: ConfigRuntime, logger: Logger) -> None:
        self._config = config
        self._logger = logger

    def fetch_pool(self, category: str) -> list[str]:
        return fetch_giphy_pool(category, logger=self._logger)

    def get_pool(self, category: str) -> list[str]:
        return get_giphy_pool(
            category,
            redis_factory=self._config.optional_redis,
            fetch_pool=self.fetch_pool,
            logger=self._logger,
        )

    def get_random_gif(self, category: str) -> str | None:
        return get_random_gif(category, get_pool=self.get_pool)

    def get_good_morning(self) -> str:
        return get_good_morning(get_gif=self.get_random_gif)

    def get_good_night(self) -> str:
        return get_good_night(get_gif=self.get_random_gif)


__all__ = ["GiphyService"]
