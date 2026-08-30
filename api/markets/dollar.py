from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from logging import Logger
from typing import Any, Protocol, cast

from api.cache.service import CacheService
from api.core.config_runtime import ConfigRuntime
from api.core.rust_bridge import load_rust_bridge
from api.i18n import tr
from api.markets.rulo import build_rulo_message
from api.services.stale_cache import StaleCache
from api.markets.dollar_commands import sort_dollar_rates
from api.utils import fmt_num, fmt_signed_pct

DollarFetcher = Callable[..., dict[str, Any] | None]
TextBuilder = Callable[[int], str | None]
CacheGetter = Callable[[], Any]
RefreshScheduler = Callable[[Callable[[], None]], None]
TcrmGetter = Callable[[int], tuple[float | None, float | None]]
RatesSorter = Callable[..., list[dict[str, Any]]]
BandGetter = Callable[[], dict[str, Any] | None]
RatesFormatter = Callable[
    [list[dict[str, Any]], int, dict[str, Any] | None],
    str,
]
CachedRequest = Callable[..., dict[str, Any] | None]
RuloBuilder = Callable[..., str]
PriceGetter = Callable[[str], float | None]
logger = logging.getLogger(__name__)


class _RustSatoshiFormatter(Protocol):
    def format_satoshi_quote(self, price_usd: float, price_ars: float) -> str: ...


class _RustDevoCalculator(Protocol):
    def parse_devo_input(self, message_text: str) -> tuple[str, float, float]: ...

    def calculate_devo(
        self,
        fee: float,
        purchase: float,
        official: float,
        card: float,
        usdt_ask: float,
        usdt_bid: float,
    ) -> str: ...


def _load_rust_satoshi_formatter() -> _RustSatoshiFormatter | None:
    module = load_rust_bridge("RUST_SATOSHI_ENABLED")
    if module is None:
        return None
    return cast(_RustSatoshiFormatter, module)


def _load_rust_devo_calculator() -> _RustDevoCalculator | None:
    module = load_rust_bridge("RUST_DEVO_ENABLED")
    if module is None:
        return None
    return cast(_RustDevoCalculator, module)


def format_dollar_rates(
    dollar_rates: list[dict[str, Any]],
    hours_ago: int,
    band_limits: dict[str, Any] | None = None,
) -> str:
    rates = list(dollar_rates)
    if band_limits:
        for label, key in (("Banda piso", "lower"), ("Banda techo", "upper")):
            value = band_limits.get(key)
            if not isinstance(value, (int, float)):
                continue
            history = band_limits.get(f"{key}_change_pct")
            rates.append(
                {
                    "name": label,
                    "price": float(value),
                    "history": history if isinstance(history, (int, float)) else None,
                }
            )

    rates.sort(key=lambda item: item.get("price", 0))
    lines = []
    for dollar in rates:
        line = f"{dollar['name']}: {fmt_num(dollar['price'], 2)}"
        if dollar["history"] is not None:
            line += f" ({fmt_signed_pct(dollar['history'], 2)}% {hours_ago}hs)"
        lines.append(line)
    if hours_ago != 24 and all(rate.get("history") is None for rate in rates):
        lines.append(f"\n({tr('market.no_history', hours=hours_ago)})")
    return "\n".join(lines)


def get_dollar_rates(
    msg_text: str,
    *,
    timeframes: Mapping[str, int],
    get_cache: CacheGetter,
    build_text: TextBuilder,
    cache_ttl: int,
    stale_grace: int,
    schedule_refresh: RefreshScheduler,
    logger: Logger,
) -> str | None:
    _, timeframe = _parse_timeframe(msg_text, timeframes)
    if timeframe is None and msg_text.strip():
        token = msg_text.strip().lower()
        if re.fullmatch(r"\d+[hd]", token):
            return tr(
                "market.timeframe_invalid",
                timeframe=token,
                valid=", ".join(timeframes),
            )
    hours_ago = timeframes.get(timeframe, 24) if timeframe else 24
    cache_key = f"market:dolar:formatted:{hours_ago}"
    try:
        result = get_cache().get(
            key=cache_key,
            lock_key=f"{cache_key}:lock",
            ttl=cache_ttl,
            stale_grace=stale_grace,
            refresh=lambda: build_text(hours_ago),
            schedule_refresh=schedule_refresh,
        )
        value = result.value
        return str(value) if value is not None else None
    except Exception:
        logger.exception("dollar snapshot cache failed")
        return build_text(hours_ago)


def build_dollar_rates_text(
    hours_ago: int,
    *,
    fetch_dollars: DollarFetcher,
    get_tcrm: TcrmGetter,
    sort_rates: RatesSorter,
    get_band_limits: BandGetter,
    format_rates: RatesFormatter,
) -> str | None:
    dollars = fetch_dollars(
        hourly_cache=True,
        get_history=hours_ago if hours_ago != 24 else 0,
    )
    tcrm_value, tcrm_history = get_tcrm(hours_ago)
    sorted_rates = sort_rates(
        dollars,
        tcrm_value,
        tcrm_history,
        hours_ago=hours_ago,
    )
    band_limits = get_band_limits()
    if band_limits and hours_ago != 24:
        band_limits = {
            key: value for key, value in band_limits.items() if not key.endswith("_change_pct")
        }
    return format_rates(sorted_rates, hours_ago, band_limits)


def _parse_devo_python(msg_text: str) -> tuple[str, float, float]:
    purchase = 0.0
    if "," in msg_text:
        numbers = msg_text.replace(" ", "").split(",")
        fee = float(numbers[0]) / 100
        if len(numbers) > 1:
            purchase = float(numbers[1])
    else:
        fee = float(msg_text) / 100
    if fee != fee or fee > 1 or purchase != purchase or purchase < 0:
        return "input_error", 0.0, 0.0
    return "valid", fee, purchase


def _calculate_devo_python(
    fee: float,
    purchase: float,
    official: float,
    card: float,
    usdt_ask: float,
    usdt_bid: float,
) -> dict[str, Any]:
    usdt = (usdt_ask + usdt_bid) / 2
    profit = -(fee * usdt + official - usdt) / card
    result: dict[str, Any] = {
        "profit": fmt_num(profit * 100, 2),
        "fee": fmt_num(fee * 100, 2),
        "official": fmt_num(official, 2),
        "usdt": fmt_num(usdt, 2),
        "card": fmt_num(card, 2),
    }
    if purchase > 0:
        purchase_ars = purchase * card
        purchase_usdt = purchase_ars / usdt
        profit_ars = purchase_ars * profit
        profit_usdt = profit_ars / usdt
        result["purchase"] = {
            "usd": fmt_num(purchase, 2),
            "ars": fmt_num(purchase_ars, 2),
            "usdt": fmt_num(purchase_usdt, 2),
            "profit_ars": fmt_num(profit_ars, 2),
            "profit_usdt": fmt_num(profit_usdt, 2),
            "total_ars": fmt_num(purchase_ars + profit_ars, 2),
            "total_usdt": fmt_num(purchase_usdt + profit_usdt, 2),
        }
    return result


def _render_devo(result: Mapping[str, Any]) -> str:
    message = tr(
        "market.dollar.devo_summary",
        profit=result["profit"],
        fee=result["fee"],
        official=result["official"],
        usdt=result["usdt"],
        card=result["card"],
    )
    purchase = result.get("purchase")
    if isinstance(purchase, Mapping):
        return tr(
            "market.dollar.devo_purchase",
            usd=purchase["usd"],
            ars=purchase["ars"],
            usdt=purchase["usdt"],
            profit_ars=purchase["profit_ars"],
            profit_usdt=purchase["profit_usdt"],
            total_ars=purchase["total_ars"],
            total_usdt=purchase["total_usdt"],
            summary=message,
        )
    return message


def _parse_devo(rust: _RustDevoCalculator | None, msg_text: str) -> tuple[str, float, float]:
    if rust is None:
        return _parse_devo_python(msg_text)
    try:
        return rust.parse_devo_input(msg_text)
    except Exception as error:
        logger.warning(
            "Rust devo parser failed; using Python fallback: error_type=%s",
            type(error).__name__,
        )
        return _parse_devo_python(msg_text)


def _calculate_devo(
    rust: _RustDevoCalculator | None,
    fee: float,
    purchase: float,
    official: float,
    card: float,
    usdt_ask: float,
    usdt_bid: float,
) -> Mapping[str, Any]:
    if rust is None:
        return _calculate_devo_python(fee, purchase, official, card, usdt_ask, usdt_bid)
    try:
        result = json.loads(
            rust.calculate_devo(fee, purchase, official, card, usdt_ask, usdt_bid)
        )
        if not isinstance(result, Mapping):
            raise ValueError("Rust devo result is not a mapping")
        return result
    except Exception as error:
        logger.warning(
            "Rust devo calculation failed; using Python fallback: error_type=%s",
            type(error).__name__,
        )
        return _calculate_devo_python(fee, purchase, official, card, usdt_ask, usdt_bid)


def get_devo(msg_text: str, *, fetch_dollars: DollarFetcher) -> str:
    try:
        rust = _load_rust_devo_calculator()
        kind, fee, purchase = _parse_devo(rust, msg_text)
        if kind == "usage":
            return tr("market.dollar.devo_usage")
        if kind == "input_error":
            return tr("market.dollar.input_error")
        if kind != "valid":
            raise ValueError("unknown devo parse result")

        dollars = fetch_dollars()
        if not dollars or "data" not in dollars:
            return tr("market.dollar.load_error")
        data = dollars["data"]
        official = float(data["oficial"]["price"])
        card = float(data["tarjeta"]["price"])
        usdt_ask = float(data["cripto"]["usdt"]["ask"])
        usdt_bid = float(data["cripto"]["usdt"]["bid"])
        result = _calculate_devo(rust, fee, purchase, official, card, usdt_ask, usdt_bid)
        return _render_devo(result)
    except ValueError:
        return tr("market.dollar.devo_usage")


def get_rulo(
    *,
    fetch_dollars: DollarFetcher,
    cached_request: CachedRequest,
    cache_ttl: int,
    build_message: RuloBuilder,
) -> str:
    usd_amount = 1000.0
    amount = str(int(usd_amount))
    dollars = fetch_dollars()
    if not dollars or "data" not in dollars:
        return tr("market.dollar.load_error")
    usd_usdt = cached_request(
        f"https://criptoya.com/api/USDT/USD/{amount}",
        None,
        None,
        cache_ttl,
        True,
    )
    usdt_ars = cached_request(
        f"https://criptoya.com/api/USDT/ARS/{amount}",
        None,
        None,
        cache_ttl,
        True,
    )
    return build_message(
        dollars["data"],
        usd_usdt.get("data") if usd_usdt and "data" in usd_usdt else None,
        usdt_ars.get("data") if usdt_ars and "data" in usdt_ars else None,
        usd_amount=usd_amount,
    )


def _format_satoshi_python(price_usd: float, price_ars: float) -> str:
    return f"""1 satoshi = ${price_usd / 100_000_000:.8f} USD
1 satoshi = ${price_ars / 100_000_000:.4f} ARS

$1 USD = {int(100_000_000 / price_usd):,} sats
$1 ARS = {100_000_000 / price_ars:.3f} sats"""


def satoshi(*, get_btc_price: PriceGetter, logger: Logger) -> str:
    try:
        price_usd = get_btc_price("USD")
        price_ars = get_btc_price("ARS")
        if price_usd is None:
            return tr("market.dollar.btc_usd_error")
        if price_ars is None:
            return tr("market.dollar.btc_ars_error")
        rust = _load_rust_satoshi_formatter()
        if rust is not None:
            try:
                return rust.format_satoshi_quote(price_usd, price_ars)
            except Exception as error:
                logger.warning(
                    "Rust satoshi formatter failed; using Python fallback: error_type=%s",
                    type(error).__name__,
                )
        return _format_satoshi_python(price_usd, price_ars)
    except TypeError, ValueError, ZeroDivisionError:
        return tr("market.dollar.btc_error")
    except Exception as error:
        logger.exception("satoshi failed: %s", error)
        return tr("market.dollar.btc_error")


def handle_bcra_variables(
    *,
    get_variables: Callable[[], dict[str, Any] | None],
    format_variables: Callable[[dict[str, Any]], str],
    logger: Logger,
) -> str:
    try:
        variables = get_variables()
        if not variables:
            return tr("market.bcra.load_error")
        return format_variables(variables)
    except Exception:
        logger.exception("Error handling BCRA variables")
        return tr("market.bcra.error")


def _parse_timeframe(
    msg_text: str,
    valid: Mapping[str, int],
) -> tuple[str, str | None]:
    parts = msg_text.strip().rsplit(None, 1)
    if parts and parts[-1].lower() in valid:
        timeframe = parts[-1].lower()
        return (parts[0].strip() if len(parts) > 1 else ""), timeframe
    return msg_text.strip(), None


class DollarService:
    def __init__(
        self,
        *,
        cache: CacheService,
        config: ConfigRuntime,
        logger: Logger,
        timeframes: Mapping[str, int],
        cache_ttl: int,
        stale_grace: int,
        schedule_refresh: RefreshScheduler,
        get_tcrm: TcrmGetter,
        get_band_limits: BandGetter,
        get_btc_price: PriceGetter,
        get_bcra_variables: Callable[[], dict[str, Any] | None],
        format_bcra_variables: Callable[[dict[str, Any]], str],
    ) -> None:
        self._cache = cache
        self._config = config
        self._logger = logger
        self._timeframes = dict(timeframes)
        self._cache_ttl = cache_ttl
        self._stale_grace = stale_grace
        self._schedule_refresh = schedule_refresh
        self.get_tcrm = get_tcrm
        self.get_band_limits = get_band_limits
        self.get_btc_price = get_btc_price
        self.get_bcra_variables = get_bcra_variables
        self.format_bcra_variables = format_bcra_variables
        self._snapshot_cache: StaleCache | None = None

    def fetch_dollar_data(
        self,
        *,
        hourly_cache: bool = True,
        get_history: int = 0,
    ) -> dict[str, Any] | None:
        return self._cache.request(
            "https://criptoya.com/api/dolar",
            None,
            None,
            self._cache_ttl,
            hourly_cache,
            get_history,
        )

    def get_snapshot_cache(self) -> StaleCache:
        if self._snapshot_cache is None:
            self._snapshot_cache = StaleCache(redis_client=self._config.redis())
        return self._snapshot_cache

    def format_rates(
        self,
        dollar_rates: list[dict[str, Any]],
        hours_ago: int,
        band_limits: dict[str, Any] | None = None,
    ) -> str:
        return format_dollar_rates(dollar_rates, hours_ago, band_limits)

    def build_rates_text(self, hours_ago: int) -> str | None:
        return build_dollar_rates_text(
            hours_ago,
            fetch_dollars=self.fetch_dollar_data,
            get_tcrm=self.get_tcrm,
            sort_rates=sort_dollar_rates,
            get_band_limits=self.get_band_limits,
            format_rates=self.format_rates,
        )

    def get_rates(self, msg_text: str = "") -> str | None:
        return get_dollar_rates(
            msg_text,
            timeframes=self._timeframes,
            get_cache=self.get_snapshot_cache,
            build_text=self.build_rates_text,
            cache_ttl=self._cache_ttl,
            stale_grace=self._stale_grace,
            schedule_refresh=self._schedule_refresh,
            logger=self._logger,
        )

    def get_devo(self, msg_text: str) -> str:
        return get_devo(msg_text, fetch_dollars=self.fetch_dollar_data)

    def get_rulo(self) -> str:
        return get_rulo(
            fetch_dollars=self.fetch_dollar_data,
            cached_request=self._cache.request,
            cache_ttl=self._cache_ttl,
            build_message=build_rulo_message,
        )

    def satoshi(self) -> str:
        return satoshi(get_btc_price=self.get_btc_price, logger=self._logger)

    def handle_bcra_variables(self) -> str:
        return handle_bcra_variables(
            get_variables=self.get_bcra_variables,
            format_variables=self.format_bcra_variables,
            logger=self._logger,
        )


__all__ = ["DollarService"]
