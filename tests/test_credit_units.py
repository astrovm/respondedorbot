import json
from pathlib import Path

import pytest

from api.billing import credit_units as credit_units_module
from api.billing.credit_units import (
    CREDIT_SCALE,
    format_credit_units,
    parse_credit_units,
    rescale_credit_units,
    whole_credits_to_units,
)


def _contract():
    path = Path(__file__).parents[1] / "contracts" / "credit_units.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_credit_units_use_hundredth_precision():
    assert CREDIT_SCALE == 100
    assert whole_credits_to_units(3) == 300
    assert parse_credit_units("0.01") == 1
    assert parse_credit_units("1.55") == 155
    assert parse_credit_units("0.001") is None


def test_credit_units_always_display_two_decimals():
    assert format_credit_units(0) == "0.00"
    assert format_credit_units(1) == "0.01"
    assert format_credit_units(-155) == "-1.55"


def test_legacy_tenth_units_rescale_to_hundredths():
    assert rescale_credit_units(15, 10) == 150
    assert rescale_credit_units(150, CREDIT_SCALE) == 150
    assert rescale_credit_units(15, None) == 150


def test_python_credit_units_match_shared_contract(monkeypatch):
    monkeypatch.delenv("RUST_CREDIT_UNITS_ENABLED", raising=False)
    contract = _contract()

    for case in contract["parse"]:
        assert parse_credit_units(case["input"]) == case["expected"]
    for case in contract["format"]:
        assert format_credit_units(case["units"]) == case["expected"]
    for case in contract["rescale"]:
        if case["error"] is None:
            assert rescale_credit_units(case["units"], case["source_scale"]) == case["expected"]
        else:
            with pytest.raises(ValueError, match=case["error"]):
                rescale_credit_units(case["units"], case["source_scale"])
    for case in contract["whole"]:
        assert whole_credits_to_units(case["credits"]) == case["expected"]


def test_enabled_rust_credit_units_are_authoritative(monkeypatch):
    class FakeRustCreditUnits:
        def whole_credits_to_units(self, credits):
            return credits * 100

        def rescale_credit_units(self, units, source_scale):
            return units * (100 // source_scale)

        def parse_credit_units(self, value):
            return 155 if value == "1.55" else None

        def format_credit_units(self, units):
            return f"rust:{units}"

    monkeypatch.setenv("RUST_CREDIT_UNITS_ENABLED", "true")
    monkeypatch.setattr(
        credit_units_module,
        "_load_rust_credit_units",
        lambda: FakeRustCreditUnits(),
    )

    assert whole_credits_to_units(3) == 300
    assert rescale_credit_units(15, 10) == 150
    assert parse_credit_units("1.55") == 155
    assert format_credit_units(155) == "rust:155"


def test_rust_credit_unit_failure_falls_back_to_python(monkeypatch, caplog):
    class FailingRustCreditUnits:
        def whole_credits_to_units(self, _credits):
            raise RuntimeError("synthetic failure")

        def rescale_credit_units(self, _units, _source_scale):
            raise RuntimeError("synthetic failure")

        def parse_credit_units(self, _value):
            raise RuntimeError("synthetic failure")

        def format_credit_units(self, _units):
            raise RuntimeError("synthetic failure")

    monkeypatch.setenv("RUST_CREDIT_UNITS_ENABLED", "1")
    monkeypatch.setattr(
        credit_units_module,
        "_load_rust_credit_units",
        lambda: FailingRustCreditUnits(),
    )

    assert whole_credits_to_units(3) == 300
    assert rescale_credit_units(15, 10) == 150
    assert parse_credit_units("1.55") == 155
    assert format_credit_units(155) == "1.55"
    assert len(caplog.records) == 4
