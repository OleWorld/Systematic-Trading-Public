"""
Unit tests for ``portfolio/_margin.py`` (MarginModel family).

Pins the formulas of the default ``PortfolioMarginModel`` (universal-leverage
cross-margin), its validation, the price==0 / negative-price edges of the
inverse, exact equivalence with the legacy ``abs(qty*price)/leverage`` math,
and the contract-multiplier (``point_value``) scaling.

The margin methods take ``(quantity, price, point_value=1.0)`` — no ``symbol``
arg: per-symbol margin is now expressed by giving each instrument its own
model instance in the ``InstrumentConfig`` registry, so the old per-symbol
``SingleMarginModel`` stub is gone.

Run from the repo root:  pytest tests/test_margin.py -v
"""

import math

import pytest

from portfolio import MarginModel, PortfolioMarginModel


# ──────────────────────────────────────────────
# PortfolioMarginModel — forward margin
# ──────────────────────────────────────────────

def test_initial_margin_is_rate_times_abs_notional():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # abs(10 * 100) * 0.1 = 100
    assert m.initial_margin(10.0, 100.0) == pytest.approx(100.0)


def test_maintenance_margin_is_rate_times_abs_notional():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.05)
    # abs(10 * 100) * 0.05 = 50
    assert m.maintenance_margin(10.0, 100.0) == pytest.approx(50.0)


def test_maintenance_margin_defaults_to_zero():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    assert m.maintenance_margin(10.0, 100.0) == 0.0


def test_margin_uses_abs_for_short_and_negative_price():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.05)
    # Short position: qty negative, margin is a positive magnitude.
    assert m.initial_margin(-10.0, 100.0) == pytest.approx(100.0)
    # Negative price (WTI 2020): still positive magnitude.
    assert m.initial_margin(10.0, -37.0) == pytest.approx(37.0)
    assert m.maintenance_margin(10.0, -37.0) == pytest.approx(18.5)


# ──────────────────────────────────────────────
# point_value (contract multiplier) scaling
# ──────────────────────────────────────────────

def test_initial_margin_scales_by_point_value():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # abs(2 * 1000 * 50) * 0.1 = 10_000
    assert m.initial_margin(2.0, 50.0, 1000.0) == pytest.approx(10_000.0)


def test_maintenance_margin_scales_by_point_value():
    m = PortfolioMarginModel(initial_margin_rate=0.1, maintenance_margin_rate=0.05)
    # abs(2 * 1000 * 50) * 0.05 = 5_000
    assert m.maintenance_margin(2.0, 50.0, 1000.0) == pytest.approx(5_000.0)


def test_point_value_defaults_to_one():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    assert m.initial_margin(3.0, 100.0) == m.initial_margin(3.0, 100.0, 1.0)


# ──────────────────────────────────────────────
# PortfolioMarginModel — inverse (max_abs_position)
# ──────────────────────────────────────────────

def test_max_abs_position_inverts_initial_margin():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # budget=100, price=100, rate=0.1 -> 100 / (100 * 0.1) = 10
    assert m.max_abs_position(100.0, 100.0) == pytest.approx(10.0)


def test_max_abs_position_inverts_initial_margin_with_point_value():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # budget=10_000, price=50, pv=1000, rate=0.1 -> 10_000 / (1000*50*0.1) = 2
    assert m.max_abs_position(10_000.0, 50.0, 1000.0) == pytest.approx(2.0)
    # round-trip: that position locks exactly the budget.
    assert m.initial_margin(2.0, 50.0, 1000.0) == pytest.approx(10_000.0)


def test_max_abs_position_zero_price_returns_inf():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    # A zero-priced position locks no margin -> no position bound.
    assert m.max_abs_position(100.0, 0.0) == math.inf
    assert m.max_abs_position(100.0, 0.0, 1000.0) == math.inf


def test_max_abs_position_uses_abs_for_negative_price():
    m = PortfolioMarginModel(initial_margin_rate=0.1)
    assert m.max_abs_position(100.0, -100.0) == pytest.approx(10.0)


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
    """PortfolioMarginModel.from_leverage reproduces abs(qty*price)/leverage exactly (pv=1)."""
    m = PortfolioMarginModel.from_leverage(leverage)
    assert m.initial_margin(qty, price) == pytest.approx(abs(qty * price) / leverage)


def test_max_abs_position_matches_legacy_inverse():
    """The inverse reproduces budget * leverage / abs(price) exactly (pv=1)."""
    leverage, budget, price = 10.0, 500.0, 250.0
    m = PortfolioMarginModel.from_leverage(leverage)
    assert m.max_abs_position(budget, price) == pytest.approx(
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
    assert m.maintenance_margin(1.0, 100.0) == pytest.approx(10.0)


@pytest.mark.parametrize("bad_leverage", [0.0, -5.0])
def test_from_leverage_non_positive_raises(bad_leverage):
    with pytest.raises(ValueError):
        PortfolioMarginModel.from_leverage(bad_leverage)


@pytest.mark.parametrize("sub_unit_leverage", [0.5, 0.99])
def test_from_leverage_below_one_raises_leverage_worded(sub_unit_leverage):
    """0 < leverage < 1 implies initial_margin_rate > 1 (posting more than 100%
    of notional), which this model disallows. The guard must reject it in
    from_leverage with a leverage-worded message naming the bad value — not leak
    the derived-rate ValueError from __post_init__, which talks about a rate the
    caller never supplied."""
    with pytest.raises(ValueError) as exc:
        PortfolioMarginModel.from_leverage(sub_unit_leverage)
    msg = str(exc.value)
    assert "leverage" in msg
    assert str(sub_unit_leverage) in msg
    assert "initial_margin_rate" not in msg


# ──────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────

def test_portfolio_margin_model_is_a_margin_model():
    assert isinstance(PortfolioMarginModel(initial_margin_rate=0.1), MarginModel)


def test_single_margin_model_is_removed():
    """SingleMarginModel is superseded by per-instrument registry models."""
    with pytest.raises(ImportError):
        from portfolio import SingleMarginModel  # noqa: F401
