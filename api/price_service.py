from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Mapping, Optional

from api.cache_service import CacheService
from api.crypto_commands import get_prices

_CHANGE_FIELDS = {
    "1h": "percent_change_1h",
    "24h": "percent_change_24h",
    "7d": "percent_change_7d",
    "30d": "percent_change_30d",
}


class PriceService:
    def __init__(
        self,
        *,
        cache: CacheService,
        environment: Mapping[str, str],
        logger: Logger,
        cache_ttl: int,
    ) -> None:
        self._cache = cache
        self._environment = environment
        self._logger = logger
        self._cache_ttl = cache_ttl

    def get_api_prices(
        self,
        convert_to: str,
        limit: Optional[int] = None,
        hourly_cache: bool = False,
    ) -> Optional[Dict[str, Any]]:
        parameters = {"start": "1", "limit": "100", "convert": convert_to}
        if isinstance(limit, int) and limit > 0:
            parameters["limit"] = str(limit)
        response = self._cache.request(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
            parameters,
            {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": self._environment.get("COINMARKETCAP_KEY"),
            },
            self._cache_ttl,
            hourly_cache,
        )
        return response["data"] if response else None

    def fetch_quotes(
        self,
        identifiers: List[str],
        convert_to: str = "USD",
        by_slug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        param_key = "slug" if by_slug else "symbol"
        response = self._cache.request(
            "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest",
            {param_key: ",".join(identifiers), "convert": convert_to},
            {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": self._environment.get("COINMARKETCAP_KEY"),
            },
            self._cache_ttl,
        )
        return response["data"] if response else None

    def get_btc_price(self, convert_to: str = "USD") -> Optional[float]:
        try:
            data = self.get_api_prices(convert_to)
            if not data or "data" not in data or not data["data"]:
                return None
            first = data["data"][0]
            price_info = first.get("quote", {}).get(convert_to)
            if not price_info:
                return None
            return float(price_info.get("price"))
        except (KeyError, TypeError, ValueError):
            return None
        except Exception as error:
            self._logger.exception(
                "get_btc_price failed for convert_to=%s: %s",
                convert_to,
                error,
            )
            return None

    def get_prices(self, msg_text: str) -> Optional[str]:
        return get_prices(
            msg_text,
            change_fields=_CHANGE_FIELDS,
            fetch_prices=self.get_api_prices,
            fetch_quotes=self.fetch_quotes,
        )


__all__ = ["PriceService"]
