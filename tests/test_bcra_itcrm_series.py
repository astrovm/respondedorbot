from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.services import bcra


class _Sheet:
    def __init__(self, rows):
        self._rows = rows
        self.max_row = len(rows)

    def cell(self, *, row, column):
        return SimpleNamespace(value=self._rows[row - 1][column - 1])


def test_fetch_itcrm_series_normalizes_sorts_and_deduplicates(monkeypatch):
    sheet = _Sheet([
        ("fecha", "valor"),
        ("03/01/2024", "103,5"),
        (date(2024, 1, 1), 100),
        (datetime(2024, 1, 2), "101.25"),
        (date(2024, 1, 2), 102),
    ])
    response = MagicMock(content=b"xlsx")
    monkeypatch.setattr(bcra.requests, "get", MagicMock(return_value=response))
    monkeypatch.setattr(
        bcra,
        "load_workbook",
        MagicMock(return_value=SimpleNamespace(active=sheet, worksheets=[sheet])),
    )

    series = bcra._fetch_itcrm_series()

    assert series == bcra.ITCRMSeries(
        dates=(date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)),
        values=(100.0, 102.0, 103.5),
    )


def test_itcrm_series_loader_reuses_fresh_cache(monkeypatch):
    series = bcra.ITCRMSeries((date(2024, 1, 1),), (100.0,))
    fetch = MagicMock(return_value=series)
    monkeypatch.setattr(bcra, "_fetch_itcrm_series", fetch)
    monkeypatch.setattr(bcra, "_ITCRM_SERIES_CACHE", None)
    monkeypatch.setattr(bcra, "_ITCRM_SERIES_CACHED_AT", 0.0)

    assert bcra._load_itcrm_series() is series
    assert bcra._load_itcrm_series() is series
    fetch.assert_called_once()


def test_historical_itcrm_lookup_uses_nearest_prior_date(monkeypatch):
    series = bcra.ITCRMSeries(
        dates=(date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 10)),
        values=(100.0, 105.0, 110.0),
    )
    monkeypatch.setattr(bcra, "_load_itcrm_series", lambda: series)

    assert bcra._get_itcrm_value_for_date(datetime(2024, 1, 7)) == (
        105.0,
        datetime(2024, 1, 5),
    )
    assert bcra._get_itcrm_value_for_date(datetime(2023, 12, 31)) is None
    assert bcra.get_latest_itcrm_details() == (110.0, "10/01/24")
