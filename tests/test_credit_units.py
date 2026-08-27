from api.billing.credit_units import (
    CREDIT_SCALE,
    format_credit_units,
    parse_credit_units,
    rescale_credit_units,
    whole_credits_to_units,
)


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
