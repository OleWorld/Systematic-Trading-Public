"""
Unit tests for ``SimpleRiskManager``.

Named ``test_riskmanager_simple`` to match the
``test_riskmanager_vol_targeting`` sibling (the ``test_riskmanager_<variant>``
convention). Two sections, mirroring that file's layout:

1. **Behavior / order submission** — forecast sign → trade direction,
   forecast == 0 → flatten, already-at-target → no order, forming bars
   skipped, sizing modes (fixed_quantity / fixed_notional /
   fixed_equity_pct), negative-price ``abs(price)`` handling, flip
   short↔long, constructor + ABC validation.
2. **Per-bar diagnostics (``get_records`` / ``_records`` / ``skip_reason``)**
   — empty frame for unknown symbol / forming bar, one row per completed
   bar, and the ``skip_reason`` ladder
   (``'no_price'`` / ``'warmup_forecast'`` / ``'at_target'`` / ``None``).

Uses minimal FakePortfolio + FakeStrategy stubs so the risk-manager logic
is tested in isolation.

Run from the repo root:  pytest tests/test_riskmanager_simple.py -v
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
                 balance: float = 100_000.0,
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
            'symbol': symbol, 'quantity': quantity, 'direction': direction,
            'timestamp': timestamp, 'order_type': order_type, 'price': price,
        })
        return None


class FakeStrategy:
    """Forecast oracle + symbol_list source."""

    def __init__(self, forecasts: Optional[Dict[str, float]] = None,
                 symbol_list: Optional[List[str]] = None):
        self._forecasts: Dict[str, float] = (
            dict(forecasts) if forecasts is not None else {}
        )
        self.symbol_list: List[str] = (
            list(symbol_list) if symbol_list is not None else ['BTC']
        )

    def get_forecast(self, symbol: str) -> Optional[float]:
        return self._forecasts.get(symbol, 0.0)


def _bar(symbol: str = 'BTC', is_forming: bool = False,
         ts: Optional[datetime] = None) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else datetime(2026, 1, 1, 12, 0, 0),
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        period='1d', is_forming=is_forming,
    )


def _make(
    *,
    forecast: Optional[float] = 1.0,
    price: Optional[float] = 100.0,
    balance: float = 100_000.0,
    positions: Optional[Dict[str, float]] = None,
    size_mode: str = 'fixed_notional',
    position_size: float = 10_000.0,
):
    pf = FakePortfolio(price=price, balance=balance,
                       positions=positions if positions is not None else {'BTC': 0.0})
    strat = FakeStrategy({'BTC': forecast}, symbol_list=['BTC'])
    rm = SimpleRiskManager(portfolio=pf, strategy=strat,
                           size_mode=size_mode, position_size=position_size)
    return pf, strat, rm


# ══════════════════════════════════════════════
# SECTION 1 — Behavior / order submission
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# update_bar — skip conditions
# (forming-bar skip is pinned in Section 2 by
# test_get_records_empty_on_forming_bar, which asserts strictly more:
# no order submitted AND no diagnostic row recorded.)
# ──────────────────────────────────────────────

def test_update_bar_skips_when_price_is_zero():
    # Exact zero is treated like a missing price (margin scaling would
    # divide by it) → 'no_price' skip, no order.
    pf = FakePortfolio(price=0.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat)
    rm.update_bar(_bar())
    assert pf.submitted == []
    assert rm.get_records('BTC').iloc[0]['skip_reason'] == 'no_price'


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

def test_default_size_mode_is_fixed_quantity():
    """Futures-first default: size in contracts, not notional dollars."""
    pf = FakePortfolio(price=100.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat)
    assert rm.size_mode == 'fixed_quantity'


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


def test_negative_price_equity_pct_mode_uses_abs():
    """fixed_equity_pct also divides by ``abs(price)`` — the sign comes
    from the forecast, never from the price."""
    pf = FakePortfolio(price=-5.0, balance=500_000.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 100.0})
    rm = SimpleRiskManager(pf, strat, size_mode='fixed_equity_pct',
                           position_size=0.10)
    rm.update_bar(_bar())
    call = pf.submitted[0]
    assert call['direction'] == Direction.BUY
    # equity * pct / abs(price) = 500_000 * 0.10 / 5 = 10_000
    assert math.isclose(call['quantity'], 10_000.0)


# ──────────────────────────────────────────────
# Construction validation / ABC
# ──────────────────────────────────────────────

def test_constructor_rejects_invalid_size_mode():
    pf = FakePortfolio()
    strat = FakeStrategy()
    with pytest.raises(ValueError, match="size_mode"):
        SimpleRiskManager(pf, strat, size_mode='bogus')


def test_riskmanager_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        RiskManager()  # type: ignore[abstract]


# ══════════════════════════════════════════════
# SECTION 2 — Per-bar diagnostics (get_records / skip_reason)
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# get_records — empty / forming / one row per bar
# ──────────────────────────────────────────────

def test_get_records_empty_for_unknown_symbol():
    _, _, rm = _make()
    assert rm.get_records('UNKNOWN').empty


def test_get_records_empty_on_forming_bar():
    pf, _, rm = _make()
    rm.update_bar(_bar(is_forming=True))
    assert rm.get_records('BTC').empty
    assert pf.submitted == []


def test_get_records_records_one_row_per_completed_bar():
    _, _, rm = _make(positions={'BTC': 100.0})       # already at target → at_target
    rm.update_bar(_bar(ts=datetime(2026, 1, 1), is_forming=True))
    rm.update_bar(_bar(ts=datetime(2026, 1, 2)))
    rm.update_bar(_bar(ts=datetime(2026, 1, 3)))
    df = rm.get_records('BTC')
    assert len(df) == 2
    assert list(df.index) == [datetime(2026, 1, 2), datetime(2026, 1, 3)]


# ──────────────────────────────────────────────
# Per-bar diagnostic rows
# ──────────────────────────────────────────────

def test_get_records_no_price_skip():
    pf, _, rm = _make(price=None)
    rm.update_bar(_bar())
    df = rm.get_records('BTC')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['skip_reason'] == 'no_price'
    assert not row['submitted']
    assert row['target_qty'] is None
    assert row['trade_qty'] is None
    assert row['price'] is None
    assert pf.submitted == []


def test_get_records_warmup_forecast_skip():
    # get_forecast returns None (strategy warming up) → skip before the
    # sign logic, which would raise on None (``None > 0`` is a TypeError).
    pf, _, rm = _make(forecast=None, positions={'BTC': 0.0})
    rm.update_bar(_bar())
    df = rm.get_records('BTC')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['skip_reason'] == 'warmup_forecast'
    assert not row['submitted']
    assert row['target_qty'] is None
    assert row['trade_qty'] is None
    # price is resolved before the forecast check, so it's recorded.
    assert row['price'] == 100.0
    assert pf.submitted == []


def test_get_records_at_target_skip():
    # fixed_notional=10_000, price=100 → target_qty = 100 (sign +1)
    pf, _, rm = _make(positions={'BTC': 100.0})
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'at_target'
    assert not row['submitted']
    assert row['target_qty'] == 100.0
    assert abs(row['trade_qty']) < 1e-9
    assert row['price'] == 100.0
    assert pf.submitted == []


def test_get_records_submitted_row():
    # fixed_notional=10_000, price=100, current=0 → target=100, trade=100, submit
    pf, _, rm = _make(positions={'BTC': 0.0})
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['submitted']
    assert row['skip_reason'] is None
    assert row['target_qty'] == 100.0
    assert row['trade_qty'] == 100.0
    assert row['current_qty'] == 0.0
    assert row['price'] == 100.0
    assert row['size_mode'] == 'fixed_notional'
    assert row['position_size'] == 10_000.0
    assert row['forecast'] == 1.0
    assert len(pf.submitted) == 1
    assert pf.submitted[0]['quantity'] == 100.0


def test_get_records_flat_target_when_forecast_zero():
    # forecast=0 → target_qty=0.0 from _compute_target_qty (valid flat target);
    # current_qty=0 → trade_qty=0 → 'at_target' via post-target ladder.
    pf, _, rm = _make(forecast=0.0, positions={'BTC': 0.0})
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['target_qty'] == 0.0
    assert row['skip_reason'] == 'at_target'
    assert not row['submitted']
    assert pf.submitted == []
