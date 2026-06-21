"""
Unit tests for ``portfolio/_margin.py`` (MarginModel family).

Pins the formulas of the default ``PortfolioMarginModel`` (universal-leverage
cross-margin), its validation, the price==0 / negative-price edges of the
inverse, exact equivalence with the legacy ``abs(qty*price)/leverage`` math,
and that the ``SingleMarginModel`` scaffold raises until implemented.

Run from the repo root:  pytest tests/test_margin.py -v
"""

import math

import pytest

from portfolio import MarginModel, PortfolioMarginModel, SingleMarginModel


# ──────────────────────────────────────────────
# PortfolioMarginModel — forward margin
# ──────────────────────────────────────────────

def test_initial_margin_is_rate_times_abs_notional():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # abs(10 * 100) * 0.1 = 100
    assert m.initial_margin('BTC', 10.0, 100.0) == pytest.approx(100.0)


def test_maintenance_margin_is_rate_times_abs_notional():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.05)
    # abs(10 * 100) * 0.05 = 50
    assert m.maintenance_margin('BTC', 10.0, 100.0) == pytest.approx(50.0)


def test_maintenance_margin_defaults_to_zero():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    assert m.maintenance_margin('BTC', 10.0, 100.0) == 0.0


def test_margin_uses_abs_for_short_and_negative_price():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.05)
    # Short position: qty negative, margin is a positive magnitude.
    assert m.initial_margin('BTC', -10.0, 100.0) == pytest.approx(100.0)
    # Negative price (WTI 2020): still positive magnitude.
    assert m.initial_margin('CL', 10.0, -37.0) == pytest.approx(37.0)
    assert m.maintenance_margin('CL', 10.0, -37.0) == pytest.approx(18.5)


# ──────────────────────────────────────────────
# PortfolioMarginModel — inverse (max_abs_position)
# ──────────────────────────────────────────────

def test_max_abs_position_inverts_initial_margin():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # budget=100, price=100, rate=0.1 -> 100 / (100 * 0.1) = 10
    assert m.max_abs_position('BTC', 100.0, 100.0) == pytest.approx(10.0)


def test_max_abs_position_zero_price_returns_inf():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # A zero-priced position locks no margin -> no position bound.
    assert m.max_abs_position('BTC', 100.0, 0.0) == math.inf


def test_max_abs_position_uses_abs_for_negative_price():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    assert m.max_abs_position('CL', 100.0, -100.0) == pytest.approx(10.0)


# ──────────────────────────────────────────────
# from_leverage + legacy exact-equivalence
# ──────────────────────────────────────────────

def test_from_leverage_sets_inverse_rate():
    m = PortfolioMarginModel.from_leverage(10.0)
    assert m.initial_margin_rate == pytest.approx(0.1)
    assert m.maintenance_margin_rate == 0.0


def test_from_leverage_carries_maintenance_rate():
    m = PortfolioMarginModel.from_leverage(10.0, maintenance_margin_rate=0.05)
    assert m.maintenance_margin_rate == pytest.approx(0.05)


@pytest.mark.parametrize("leverage,qty,price", [
    (1.0, 3.0, 100.0),
    (10.0, -2.5, 250.0),
    (4.0, 1.0, -37.0),
])
def test_initial_margin_matches_legacy_leverage_formula(leverage, qty, price):
    """PortfolioMarginModel.from_leverage reproduces abs(qty*price)/leverage exactly."""
    m = PortfolioMarginModel.from_leverage(leverage)
    assert m.initial_margin('X', qty, price) == pytest.approx(abs(qty * price) / leverage)


def test_max_abs_position_matches_legacy_inverse():
    """The inverse reproduces budget * leverage / abs(price) exactly."""
    leverage, budget, price = 10.0, 500.0, 250.0
    m = PortfolioMarginModel.from_leverage(leverage)
    assert m.max_abs_position('X', budget, price) == pytest.approx(
        budget * leverage / abs(price)
    )


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

@pytest.mark.parametrize("bad_rate", [0.0, -0.1, 1.5, float('nan')])
def test_initial_margin_rate_out_of_range_raises(bad_rate):
    with pytest.raises(ValueError):
        PortfolioMarginModel(initial_margin_rate=bad_rate)


@pytest.mark.parametrize("bad_mm", [-0.01, 0.2, float('nan')])
def test_maintenance_margin_rate_out_of_range_raises(bad_mm):
    # initial_margin_rate=0.1, so 0.2 > initial and -0.01 < 0 both invalid.
    with pytest.raises(ValueError):
        PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=bad_mm)


def test_maintenance_equal_to_initial_is_allowed():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.1)
    assert m.maintenance_margin('BTC', 1.0, 100.0) == pytest.approx(10.0)


@pytest.mark.parametrize("bad_leverage", [0.0, -5.0])
def test_from_leverage_non_positive_raises(bad_leverage):
    with pytest.raises(ValueError):
        PortfolioMarginModel.from_leverage(bad_leverage)


# ──────────────────────────────────────────────
# Interface + SingleMarginModel stub
# ──────────────────────────────────────────────

def test_portfolio_margin_model_is_a_margin_model():
    assert isinstance(PortfolioMarginModel(initial_margin_rate=0.1), MarginModel)


def test_single_margin_model_is_a_margin_model():
    assert isinstance(SingleMarginModel(), MarginModel)


def test_single_margin_model_methods_raise_not_implemented():
    m = SingleMarginModel(
        initial_margin_rates={'BTC': 0.1},
        maintenance_margin_rates={'BTC': 0.05},
    )
    with pytest.raises(NotImplementedError):
        m.initial_margin('BTC', 1.0, 100.0)
    with pytest.raises(NotImplementedError):
        m.maintenance_margin('BTC', 1.0, 100.0)
    with pytest.raises(NotImplementedError):
        m.max_abs_position('BTC', 100.0, 100.0)
