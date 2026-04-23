from __future__ import annotations

from typing import Any, Dict, Optional


def _pct_change(current: float, historical: float) -> Optional[float]:
    try:
        h = float(historical)
        if h != 0:
            return ((float(current) - h) / h) * 100
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def sort_dollar_rates(
    dollar_rates,
    tcrm_100: Optional[float] = None,
    tcrm_history: Optional[float] = None,
    hours_ago: int = 24,
):
    dollars = dollar_rates["data"]

    hist_dollars: Optional[Dict[str, Any]] = None
    if hours_ago != 24:
        history_entry = dollar_rates.get("history")
        if isinstance(history_entry, dict) and "data" in history_entry:
            hist_dollars = history_entry["data"]

    def _var(
        current_price: float, criptoya_variation, *hist_path: str
    ) -> Optional[float]:
        if hours_ago == 24:
            return criptoya_variation
        if hist_dollars is None:
            return None
        try:
            obj: Any = hist_dollars
            for key in hist_path:
                obj = obj[key]
            return _pct_change(current_price, float(obj))
        except (KeyError, TypeError, ValueError):
            return None

    sorted_dollar_rates = [
        {
            "name": "Mayorista",
            "price": dollars["mayorista"]["price"],
            "history": _var(
                dollars["mayorista"]["price"],
                dollars["mayorista"]["variation"],
                "mayorista",
                "price",
            ),
        },
        {
            "name": "Oficial",
            "price": dollars["oficial"]["price"],
            "history": _var(
                dollars["oficial"]["price"],
                dollars["oficial"]["variation"],
                "oficial",
                "price",
            ),
        },
        {
            "name": "Tarjeta",
            "price": dollars["tarjeta"]["price"],
            "history": _var(
                dollars["tarjeta"]["price"],
                dollars["tarjeta"]["variation"],
                "tarjeta",
                "price",
            ),
        },
        {
            "name": "MEP",
            "price": dollars["mep"]["al30"]["ci"]["price"],
            "history": _var(
                dollars["mep"]["al30"]["ci"]["price"],
                dollars["mep"]["al30"]["ci"]["variation"],
                "mep",
                "al30",
                "ci",
                "price",
            ),
        },
        {
            "name": "CCL",
            "price": dollars["ccl"]["al30"]["ci"]["price"],
            "history": _var(
                dollars["ccl"]["al30"]["ci"]["price"],
                dollars["ccl"]["al30"]["ci"]["variation"],
                "ccl",
                "al30",
                "ci",
                "price",
            ),
        },
        {
            "name": "Blue",
            "price": dollars["blue"]["ask"],
            "history": _var(
                dollars["blue"]["ask"], dollars["blue"]["variation"], "blue", "ask"
            ),
        },
        {
            "name": "Bitcoin",
            "price": dollars["cripto"]["ccb"]["ask"],
            "history": _var(
                dollars["cripto"]["ccb"]["ask"],
                dollars["cripto"]["ccb"]["variation"],
                "cripto",
                "ccb",
                "ask",
            ),
        },
        {
            "name": "USDC",
            "price": dollars["cripto"]["usdc"]["ask"],
            "history": _var(
                dollars["cripto"]["usdc"]["ask"],
                dollars["cripto"]["usdc"]["variation"],
                "cripto",
                "usdc",
                "ask",
            ),
        },
        {
            "name": "USDT",
            "price": dollars["cripto"]["usdt"]["ask"],
            "history": _var(
                dollars["cripto"]["usdt"]["ask"],
                dollars["cripto"]["usdt"]["variation"],
                "cripto",
                "usdt",
                "ask",
            ),
        },
    ]

    if tcrm_100 is not None:
        sorted_dollar_rates.append(
            {"name": "TCRM 100", "price": tcrm_100, "history": tcrm_history}
        )

    sorted_dollar_rates.sort(key=lambda x: x["price"])

    return sorted_dollar_rates
