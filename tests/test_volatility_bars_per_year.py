"""Unit tests for ``volatility.bars_per_year``.

Pin the ``days_convention`` parameter: 'calendar' (365 days/year, 24/7
markets) and 'business' (252 trading days/year), and the hard rename from
the old 'crypto'/'tradfi' values (rejected loudly, no silent aliases).

Run from the repo root:  python -m pytest tests/test_volatility_bars_per_year.py -v
"""

import math

import pytest

from volatility import bars_per_year


def test_calendar_daily_is_365():
    assert math.isclose(bars_per_year('1d', 'calendar'), 365.0)


def test_business_daily_is_252():
    assert math.isclose(bars_per_year('1d', 'business'), 252.0)


def test_calendar_4h_is_365_x_6():
    assert math.isclose(bars_per_year('4h', 'calendar'), 365.0 * 6)


def test_business_1h_is_252_x_24():
    assert math.isclose(bars_per_year('1h', 'business'), 252.0 * 24)


def test_old_value_crypto_rejected():
    """Hard rename guard: the pre-rename 'crypto' value must fail loudly."""
    with pytest.raises(ValueError, match="days_convention"):
        bars_per_year('1d', 'crypto')


def test_old_value_tradfi_rejected():
    """Hard rename guard: the pre-rename 'tradfi' value must fail loudly."""
    with pytest.raises(ValueError, match="days_convention"):
        bars_per_year('1d', 'tradfi')


def test_unknown_days_convention_rejected():
    with pytest.raises(ValueError, match="days_convention"):
        bars_per_year('1d', 'lunar')
