from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from logging import Logger
from typing import Any

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
                    "history": history
                    if isinstance(history, (int, float))
                    else None,
                }
            )

    rates.sort(key=lambda item: item.get("price", 0))
    lines = []
    for dollar in rates:
        line = f"{dollar['name']}: {fmt_num(dollar['price'], 2)}"
        if dollar["history"] is not None:
            line += (
                f" ({fmt_signed_pct(dollar['history'], 2)}% {hours_ago}hs)"
            )
        lines.append(line)
    if hours_ago != 24 and all(rate.get("history") is None for rate in rates):
        lines.append(f"\n(sin datos historicos para {hours_ago}hs todavia)")
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
            return (
                f"timeframe '{token}' no soportado, uso: "
                f"{', '.join(timeframes)}"
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
            key: value
            for key, value in band_limits.items()
            if not key.endswith("_change_pct")
        }
    return format_rates(sorted_rates, hours_ago, band_limits)


def get_devo(msg_text: str, *, fetch_dollars: DollarFetcher) -> str:
    try:
        fee = 0.0
        purchase = 0.0
        if "," in msg_text:
            numbers = msg_text.replace(" ", "").split(",")
            fee = float(numbers[0]) / 100
            if len(numbers) > 1:
                purchase = float(numbers[1])
        else:
            fee = float(msg_text) / 100

        if fee != fee or fee > 1 or purchase != purchase or purchase < 0:
            return (
                "mandá bien los datos capo: fee entre 0 y 100, "
                "y monto de compra positivo"
            )
        dollars = fetch_dollars()
        if not dollars or "data" not in dollars:
            return "no pude traer cotizaciones del dólar boludo"

        data = dollars["data"]
        usdt = (
            float(data["cripto"]["usdt"]["ask"])
            + float(data["cripto"]["usdt"]["bid"])
        ) / 2
        official = float(data["oficial"]["price"])
        card = float(data["tarjeta"]["price"])
        profit = -(fee * usdt + official - usdt) / card
        message = f"""ganancia: {fmt_num(profit * 100, 2)}%

comisión: {fmt_num(fee * 100, 2)}%
oficial: {fmt_num(official, 2)}
usdt: {fmt_num(usdt, 2)}
tarjeta: {fmt_num(card, 2)}"""

        if purchase > 0:
            purchase_ars = purchase * card
            purchase_usdt = purchase_ars / usdt
            profit_ars = purchase_ars * profit
            profit_usdt = profit_ars / usdt
            message = f"""{fmt_num(purchase, 2)} USD Tarjeta = {fmt_num(purchase_ars, 2)} ARS = {fmt_num(purchase_usdt, 2)} USDT
Ganarias {fmt_num(profit_ars, 2)} ARS / {fmt_num(profit_usdt, 2)} USDT
Total: {fmt_num(purchase_ars + profit_ars, 2)} ARS / {fmt_num(purchase_usdt + profit_usdt, 2)} USDT

{message}"""
        return message
    except ValueError:
        return "uso: /devo <fee_porcentaje>[, <monto_compra>]"


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
        return "error consiguiendo cotizaciones del dólar"
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


def satoshi(*, get_btc_price: PriceGetter, logger: Logger) -> str:
    try:
        price_usd = get_btc_price("USD")
        price_ars = get_btc_price("ARS")
        if price_usd is None:
            return "no pude traer el precio de btc en usd"
        if price_ars is None:
            return "no pude traer el precio de btc en ars"
        return f"""1 satoshi = ${price_usd / 100_000_000:.8f} USD
1 satoshi = ${price_ars / 100_000_000:.4f} ARS

$1 USD = {int(100_000_000 / price_usd):,} sats
$1 ARS = {100_000_000 / price_ars:.3f} sats"""
    except (TypeError, ValueError, ZeroDivisionError):
        return "no pude conseguir el precio de btc boludo"
    except Exception as error:
        logger.exception("satoshi failed: %s", error)
        return "no pude conseguir el precio de btc boludo"


def handle_bcra_variables(
    *,
    get_variables: Callable[[], dict[str, Any] | None],
    format_variables: Callable[[dict[str, Any]], str],
    logger: Logger,
) -> str:
    try:
        variables = get_variables()
        if not variables:
            return (
                "No pude obtener las variables del BCRA en este momento, "
                "probá más tarde"
            )
        return format_variables(variables)
    except Exception:
        logger.exception("Error handling BCRA variables")
        return "error al obtener las variables del BCRA"


def _parse_timeframe(
    msg_text: str,
    valid: Mapping[str, int],
) -> tuple[str, str | None]:
    parts = msg_text.strip().rsplit(None, 1)
    if parts and parts[-1].lower() in valid:
        timeframe = parts[-1].lower()
        return (parts[0].strip() if len(parts) > 1 else ""), timeframe
    return msg_text.strip(), None
