"""
Unit tests for ``SimpleRiskManager`` — focused on the per-bar
diagnostic records (``_records`` / ``get_records``) added with the
base-class consolidation.

Pin:
- Empty DataFrame for unknown symbols / before any completed bar.
- Forming bars produce no row.
- ``'no_price'`` skip when ``portfolio.get_price`` returns ``None``.
- ``'at_target'`` skip when realized position already matches target
  (including the forecast=0 / current_qty=0 flat-from-flat case).
- Happy path: submitted row populates target_qty, trade_qty, price.
- One row per completed bar.

Run from the repo root:  pytest tests/test_riskmanager_simple.py -v
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from event import BarEvent
from riskmanager import SimpleRiskManager


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakePortfolio:
    """Configurable price + balance + positions; captures submitted orders."""

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

    def get_forecast(self, symbol: str) -> float:
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
    forecast: float = 1.0,
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
