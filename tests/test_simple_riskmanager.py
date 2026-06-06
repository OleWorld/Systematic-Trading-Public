"""
Unit tests for ``SimpleRiskManager`` after the forecast-aware redesign.

Uses minimal FakePortfolio + FakeStrategy stubs so the risk-manager
logic is tested in isolation. Pin:
- forecast sign → trade direction
- forecast == 0 → flatten
- already-at-target → no order
- forming bars are skipped
- sizing modes (fixed_notional, fixed_quantity, fixed_equity_pct)
- negative-price handling via ``abs(price)``
- price=None or price=0 → no order
- size_mode validation

Run from the repo root:  pytest tests/test_simple_riskmanager.py -v
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from event import BarEvent, OrderType, Direction
from riskmanager import RiskManager, SimpleRiskManager


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakePortfolio:
    """Captures submit_order calls; configurable price / balance / positions."""

    def __init__(self, price: Optional[float] = 100.0,
                 balance: float = 1_000_000.0,
                 positions: Optional[Dict[str, float]] = None):
        self._price = price
        self._balance = balance
        self.positions: Dict[str, float] = positions if positions is not None else {}
        self.submitted: List[Dict[str, Any]] = []

    def get_price(self, symbol: str) -> Optional[float]:
        return self._price

    def calculate_balance(self) -> float:
        return self._balance

    def submit_order(self, symbol, quantity, direction, timestamp,
                     order_type, price=None):
        self.submitted.append({
            'symbol': symbol,
            'quantity': quantity,
            'direction': direction,
            'timestamp': timestamp,
            'order_type': order_type,
            'price': price,
        })
        return None


class FakeStrategy:
    """Returns a configurable per-symbol forecast."""

    def __init__(self, forecasts: Optional[Dict[str, float]] = None):
        self._forecasts: Dict[str, float] = (
            dict(forecasts) if forecasts is not None else {}
        )

    def set(self, symbol: str, forecast: float) -> None:
        self._forecasts[symbol] = forecast

    def get_forecast(self, symbol: str) -> float:
        return self._forecasts.get(symbol, 0.0)


def _bar(symbol: str = 'BTC', is_forming: bool = False,
         ts: Optional[datetime] = None) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else datetime(2026, 1, 1, 12, 0, 0),
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        period='1h', is_forming=is_forming,
    )


# ──────────────────────────────────────────────
# update_bar — skip conditions
# ──────────────────────────────────────────────

def test_update_bar_skips_forming_bars():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar(is_forming=True))
    assert pf.submitted == []


def test_update_bar_skips_when_price_is_none():
    pf = FakePortfolio(price=None, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_update_bar_skips_when_price_is_zero():
    pf = FakePortfolio(price=0.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_update_bar_no_order_when_already_at_target():
    """target_qty = 100 (10_000 / 100); current_qty = 100 → trade_qty ≈ 0."""
    pf = FakePortfolio(price=100.0, positions={'BTC': 100.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    assert pf.submitted == []


# ──────────────────────────────────────────────
# update_bar — direction inference from forecast sign
# ──────────────────────────────────────────────

def test_positive_forecast_from_flat_submits_buy():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 50.0})       # any > 0 should drive long
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['symbol'] == 'BTC'
    assert call['direction'] == Direction.BUY
    assert math.isclose(call['quantity'], 100.0)
    assert call['order_type'] == OrderType.MKT


def test_negative_forecast_from_flat_submits_sell():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': -50.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.SELL
    assert math.isclose(call['quantity'], 100.0)


def test_zero_forecast_with_long_position_flattens():
    pf = FakePortfolio(price=100.0, positions={'BTC': 50.0})
    strat = FakeStrategy({'BTC': 0.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.SELL
    assert math.isclose(call['quantity'], 50.0)


def test_zero_forecast_with_short_position_flattens():
    pf = FakePortfolio(price=100.0, positions={'BTC': -50.0})
    strat = FakeStrategy({'BTC': 0.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.BUY
    assert math.isclose(call['quantity'], 50.0)


def test_zero_forecast_zero_position_no_order():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 0.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_positive_forecast_flips_short_to_long():
    """Existing short position; forecast turns positive → buy enough to
    cover the short AND open the long."""
    pf = FakePortfolio(price=100.0, positions={'BTC': -50.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    # target_qty = +100, current = -50, trade = +150 BUY.
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.BUY
    assert math.isclose(call['quantity'], 150.0)


def test_negative_forecast_flips_long_to_short():
    pf = FakePortfolio(price=100.0, positions={'BTC': 50.0})
    strat = FakeStrategy({'BTC': -100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    # target_qty = -100, current = +50, trade = -150 SELL.
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.SELL
    assert math.isclose(call['quantity'], 150.0)


def test_no_signal_type_kwarg_passed():
    """The new SimpleRiskManager doesn't pass signal_type — it doesn't
    exist on the new Protocol surface. Confirm the call kwargs."""
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    call = pf.submitted[0]
    assert 'signal_type' not in call               # not captured by FakePortfolio
    assert call['order_type'] == OrderType.MKT


# ──────────────────────────────────────────────
# Sizing modes
# ──────────────────────────────────────────────

def test_size_mode_fixed_notional():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    assert math.isclose(pf.submitted[0]['quantity'], 100.0)  # 10k / 100


def test_size_mode_fixed_quantity():
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_quantity',
                             position_size=7.5)
    rm.update_bar(_bar())
    assert math.isclose(pf.submitted[0]['quantity'], 7.5)


def test_size_mode_fixed_equity_pct():
    pf = FakePortfolio(price=100.0, balance=500_000.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_equity_pct',
                             position_size=0.10)
    rm.update_bar(_bar())
    # equity * pct / price = 500_000 * 0.10 / 100 = 500
    assert math.isclose(pf.submitted[0]['quantity'], 500.0)


# ──────────────────────────────────────────────
# Negative-price handling
# ──────────────────────────────────────────────

def test_negative_price_uses_abs_for_notional():
    """Negative-priced instruments (e.g. WTI 2020): direction comes from
    the forecast sign; quantity divides by ``abs(price)``."""
    pf = FakePortfolio(price=-5.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_notional',
                             position_size=10_000.0)
    rm.update_bar(_bar())
    call = pf.submitted[0]
    assert call['direction'] == Direction.BUY
    assert math.isclose(call['quantity'], 2_000.0)             # 10k / abs(-5)


def test_negative_price_quantity_mode_unaffected():
    pf = FakePortfolio(price=-5.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': -100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_quantity',
                             position_size=7.5)
    rm.update_bar(_bar())
    assert math.isclose(pf.submitted[0]['quantity'], 7.5)
    assert pf.submitted[0]['direction'] == Direction.SELL


# ──────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────

def test_constructor_rejects_invalid_size_mode():
    pf = FakePortfolio()
    strat = FakeStrategy()
    with pytest.raises(ValueError, match="size_mode"):
        SimpleRiskManager(pf, strat, size_mode='bogus')


# ──────────────────────────────────────────────
# RiskManager ABC — cannot be instantiated
# ──────────────────────────────────────────────

def test_riskmanager_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        RiskManager()  # type: ignore[abstract]
