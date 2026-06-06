"""
Characterization unit tests for `execution._backtest`.

Uses a minimal FakeQueue stub so BacktestExecution's fill timing,
limit-order satisfaction, and pending-order management are exercised in
isolation. The execution module is intentionally decoupled from the
portfolio: it does not perform margin or solvency checks, so these tests
do not need a portfolio double.

Run from the repo root:  pytest tests/test_execution.py -v
"""

import math
from datetime import datetime
from typing import Any, List, Optional, Tuple

import pytest

from event import (
    BarEvent,
    Direction,
    FillEvent,
    OrderEvent,
    OrderType,
)
from execution import (
    BacktestExecution,
    CommissionModel,
    ExecutionHandler,
    LiveExecution,
    SlippageModel,
)


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakeQueue:
    """Captures every put() for inspection."""

    def __init__(self):
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


# ──────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────

DEFAULT_TS = datetime(2026, 1, 1, 12, 0, 0)


def _bar(symbol: str = 'BTC', ts: Optional[datetime] = None,
         open: float = 100.0, high: float = 101.0,
         low: float = 99.0, close: float = 100.5,
         volume: float = 1.0) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else DEFAULT_TS,
        open=open, high=high, low=low, close=close, volume=volume,
        period='1h', is_forming=False,
    )


def _order(symbol: str = 'BTC', qty: float = 1.0,
           direction: Direction = Direction.BUY,
           order_type: OrderType = OrderType.MKT,
           price: Optional[float] = None,
           order_id: Optional[str] = None,
           ts: Optional[datetime] = None) -> OrderEvent:
    kwargs = dict(
        symbol=symbol, order_type=order_type, quantity=qty, direction=direction,
        price=price, timestamp=ts if ts is not None else DEFAULT_TS,
    )
    if order_id is not None:
        kwargs['order_id'] = order_id
    return OrderEvent(**kwargs)


def _new_execution(
    fill_on: str = 'signal_close',
    slippage: Tuple[str, float] = ('pct', 0.0),
    commission_rate: float = 0.0,
    exchange_name: str = 'BACKTEST',
) -> Tuple[BacktestExecution, FakeQueue]:
    """Build a BacktestExecution with a fake queue. Zero-cost by default."""
    q = FakeQueue()
    ex = BacktestExecution(
        events_queue=q,
        slippage_model=SlippageModel(mode=slippage[0], value=slippage[1]),
        commission_model=CommissionModel(rate=commission_rate),
        fill_on=fill_on,
        exchange_name=exchange_name,
    )
    return ex, q


# ──────────────────────────────────────────────
# SlippageModel
# ──────────────────────────────────────────────

def test_slippage_invalid_mode_raises():
    with pytest.raises(ValueError):
        SlippageModel(mode='bogus', value=0.001)


def test_slippage_pct_buy_adds_to_price():
    s = SlippageModel(mode='pct', value=0.001)
    assert math.isclose(s.apply(100.0, Direction.BUY), 100.1)


def test_slippage_pct_sell_subtracts_from_price():
    s = SlippageModel(mode='pct', value=0.001)
    assert math.isclose(s.apply(100.0, Direction.SELL), 99.9)


def test_slippage_absolute_buy_adds_fixed_value():
    s = SlippageModel(mode='absolute', value=0.5)
    assert math.isclose(s.apply(100.0, Direction.BUY), 100.5)


def test_slippage_absolute_sell_subtracts_fixed_value():
    s = SlippageModel(mode='absolute', value=0.5)
    assert math.isclose(s.apply(100.0, Direction.SELL), 99.5)


def test_slippage_apply_rejects_non_enum_direction():
    s = SlippageModel(mode='pct', value=0.001)
    with pytest.raises(ValueError):
        s.apply(100.0, "BUY")  # string is not a Direction enum member


def test_slippage_pct_buy_at_negative_price_pushes_fill_higher():
    # WTI 2020 case: BUY at -$37 with 0.1% pct should fill at a *higher*
    # number (less negative), i.e. worse for the buyer. Pre-fix the slip
    # term was negative and the fill went the wrong way.
    s = SlippageModel(mode='pct', value=0.001)
    assert math.isclose(s.apply(-37.0, Direction.BUY), -37.0 + 0.037)


def test_slippage_pct_sell_at_negative_price_pushes_fill_lower():
    s = SlippageModel(mode='pct', value=0.001)
    assert math.isclose(s.apply(-37.0, Direction.SELL), -37.0 - 0.037)


# ──────────────────────────────────────────────
# CommissionModel
# ──────────────────────────────────────────────

def test_commission_calculate_uses_abs_notional():
    c = CommissionModel(rate=0.001)
    # negative quantity must still produce a positive commission
    assert math.isclose(c.calculate(-2.0, 100.0), 0.2)


def test_commission_calculate_zero_rate_returns_zero():
    c = CommissionModel(rate=0.0)
    assert c.calculate(1.0, 100.0) == 0.0


# ──────────────────────────────────────────────
# ExecutionHandler ABC
# ──────────────────────────────────────────────

def test_execution_handler_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ExecutionHandler()  # type: ignore[abstract]


# ──────────────────────────────────────────────
# BacktestExecution — construction
# ──────────────────────────────────────────────

def test_init_invalid_fill_on_raises():
    with pytest.raises(ValueError):
        _new_execution(fill_on='bogus')


def test_init_accepts_both_valid_fill_on_values():
    ex1, _ = _new_execution(fill_on='signal_close')
    ex2, _ = _new_execution(fill_on='next_open')
    assert ex1.fill_on == 'signal_close'
    assert ex2.fill_on == 'next_open'


def test_init_seeds_empty_pending_orders_and_current_bars():
    ex, _ = _new_execution()
    assert ex.pending_orders == {}
    assert ex._current_bars == {}


def test_init_does_not_hold_portfolio_reference():
    """
    Decoupling regression: BacktestExecution must not require nor hold a
    portfolio reference. The constructor signature has been simplified;
    confirm there's no `portfolio` attribute.
    """
    ex, _ = _new_execution()
    assert not hasattr(ex, 'portfolio')


# ──────────────────────────────────────────────
# execute_order — signal_close
# ──────────────────────────────────────────────

def test_signal_close_mkt_fills_at_bar_close():
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.5))
    ex.execute_order(_order(qty=1.0, direction=Direction.BUY,
                            order_type=OrderType.MKT))
    assert len(q.items) == 1
    fill = q.items[0]
    assert isinstance(fill, FillEvent)
    assert math.isclose(fill.fill_notional, 1.0 * 100.5)
    assert ex.pending_orders == {}


def test_signal_close_lmt_buy_satisfied_fills_at_limit_price():
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=101.0, low=98.0, close=100.5))
    # limit = 99. bar.low = 98 -> satisfied. Fill at limit, NOT open/close/low.
    ex.execute_order(_order(qty=2.0, direction=Direction.BUY,
                            order_type=OrderType.LMT, price=99.0))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 2.0 * 99.0)
    assert ex.pending_orders == {}


def test_signal_close_lmt_sell_satisfied_fills_at_limit_price():
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=102.0, low=99.0, close=100.5))
    ex.execute_order(_order(qty=1.0, direction=Direction.SELL,
                            order_type=OrderType.LMT, price=101.0))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 101.0)
    assert ex.pending_orders == {}


def test_signal_close_lmt_buy_not_satisfied_falls_to_pending():
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.5))
    order = _order(qty=1.0, direction=Direction.BUY,
                   order_type=OrderType.LMT, price=98.0, order_id='L1')
    ex.execute_order(order)
    assert q.items == []
    assert 'L1' in ex.pending_orders


def test_signal_close_lmt_sell_not_satisfied_falls_to_pending():
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.5))
    order = _order(qty=1.0, direction=Direction.SELL,
                   order_type=OrderType.LMT, price=102.0, order_id='L2')
    ex.execute_order(order)
    assert q.items == []
    assert 'L2' in ex.pending_orders


def test_signal_close_no_current_bar_raises_runtime_error():
    ex, _ = _new_execution(fill_on='signal_close')
    with pytest.raises(RuntimeError):
        ex.execute_order(_order())  # no bar observed first


def test_signal_close_does_not_apply_gap_favorable_pricing():
    # On the SIGNAL bar, LMT BUY fills strictly at the limit price — gap-favorable
    # min(limit, bar.open) only applies to orders already resting on the book
    # at a LATER bar's open. Pin that distinction here.
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=95.0, high=101.0, low=94.0, close=100.0))
    ex.execute_order(_order(qty=1.0, direction=Direction.BUY,
                            order_type=OrderType.LMT, price=100.0))
    # limit=100 satisfied (low<=100). Fills at 100, not at min(100, 95)=95.
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 100.0)


# ──────────────────────────────────────────────
# execute_order — next_open
# ──────────────────────────────────────────────

def test_next_open_mkt_queues_without_fill():
    ex, q = _new_execution(fill_on='next_open')
    ex.update_bar(_bar())  # even with a bar observed
    order = _order(qty=1.0, order_type=OrderType.MKT, order_id='M1')
    ex.execute_order(order)
    assert q.items == []
    assert 'M1' in ex.pending_orders


def test_next_open_lmt_queues_without_fill():
    # Even when this bar's range would satisfy the limit, next_open queues it.
    ex, q = _new_execution(fill_on='next_open')
    ex.update_bar(_bar(open=100.0, high=101.0, low=98.0, close=100.5))
    order = _order(qty=1.0, direction=Direction.BUY,
                   order_type=OrderType.LMT, price=99.0, order_id='L1')
    ex.execute_order(order)
    assert q.items == []
    assert 'L1' in ex.pending_orders


def test_next_open_execute_order_without_current_bar_does_not_raise():
    ex, q = _new_execution(fill_on='next_open')
    # Should NOT raise the RuntimeError that signal_close mode raises.
    ex.execute_order(_order(order_id='X'))
    assert 'X' in ex.pending_orders


# ──────────────────────────────────────────────
# update_bar — fills pending orders
# ──────────────────────────────────────────────

def test_update_bar_fills_pending_mkt_at_bar_open():
    ex, q = _new_execution(fill_on='next_open')
    order = _order(qty=1.0, direction=Direction.BUY, order_type=OrderType.MKT,
                   order_id='M1')
    ex.execute_order(order)
    ex.update_bar(_bar(open=123.0, high=124.0, low=122.0, close=123.5))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 123.0)
    assert 'M1' not in ex.pending_orders


def test_update_bar_fills_pending_lmt_buy_with_gap_favorable_pricing():
    ex, q = _new_execution(fill_on='next_open')
    order = _order(qty=1.0, direction=Direction.BUY,
                   order_type=OrderType.LMT, price=100.0, order_id='L1')
    ex.execute_order(order)
    # Gap-down open: bar.open=95, bar.low=94. limit=100 satisfied; fill at min(100, 95)=95.
    ex.update_bar(_bar(open=95.0, high=96.0, low=94.0, close=95.5))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 95.0)
    assert 'L1' not in ex.pending_orders


def test_update_bar_fills_pending_lmt_sell_with_gap_favorable_pricing():
    ex, q = _new_execution(fill_on='next_open')
    order = _order(qty=1.0, direction=Direction.SELL,
                   order_type=OrderType.LMT, price=100.0, order_id='L1')
    ex.execute_order(order)
    # Gap-up open: bar.open=105, bar.high=106. limit=100 satisfied; fill at max(100, 105)=105.
    ex.update_bar(_bar(open=105.0, high=106.0, low=104.5, close=105.5))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 105.0)


def test_update_bar_pending_lmt_not_satisfied_stays_pending():
    """LMT order never fills — stays pending across many bars, no FillEvent."""
    ex, q = _new_execution(fill_on='signal_close')
    # Place LMT BUY at 50 when market is trading ~100 — nowhere near the book.
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.5))
    order = _order(qty=1.0, direction=Direction.BUY,
                   order_type=OrderType.LMT, price=50.0, order_id='FAR')
    ex.execute_order(order)
    assert 'FAR' in ex.pending_orders

    # Feed many bars whose low never drops to 50.
    for close in (100.0, 102.0, 98.0, 105.0, 95.0):
        ex.update_bar(_bar(open=close, high=close + 1.0, low=close - 1.0,
                           close=close))
    assert q.items == []
    assert 'FAR' in ex.pending_orders  # persisted indefinitely, no TTL


def test_update_bar_ignores_orders_for_other_symbols():
    ex, q = _new_execution(fill_on='next_open')
    btc_order = _order(symbol='BTC', order_id='B1', order_type=OrderType.MKT)
    ex.execute_order(btc_order)
    # ETH bar arrives — must not touch the BTC pending order.
    ex.update_bar(_bar(symbol='ETH'))
    assert q.items == []
    assert 'B1' in ex.pending_orders


def test_update_bar_fills_only_matching_symbol_leaves_others_pending():
    ex, q = _new_execution(fill_on='next_open')
    ex.execute_order(_order(symbol='BTC', order_id='B1',
                            order_type=OrderType.MKT))
    ex.execute_order(_order(symbol='ETH', order_id='E1',
                            order_type=OrderType.MKT))
    ex.update_bar(_bar(symbol='BTC', open=100.0))
    assert len(q.items) == 1
    assert q.items[0].symbol == 'BTC'
    assert 'B1' not in ex.pending_orders
    assert 'E1' in ex.pending_orders


def test_update_bar_stores_current_bar_per_symbol():
    ex, _ = _new_execution()
    ex.update_bar(_bar(symbol='BTC', close=100.0))
    ex.update_bar(_bar(symbol='ETH', close=2000.0))
    assert ex._current_bars['BTC'].close == 100.0
    assert ex._current_bars['ETH'].close == 2000.0


def test_update_bar_fills_multiple_pending_orders_in_one_bar():
    ex, q = _new_execution(fill_on='next_open')
    ex.execute_order(_order(symbol='BTC', order_id='A',
                            order_type=OrderType.MKT, qty=1.0))
    ex.execute_order(_order(symbol='BTC', order_id='B',
                            order_type=OrderType.MKT, qty=2.0))
    ex.update_bar(_bar(symbol='BTC', open=100.0))
    assert len(q.items) == 2
    assert ex.pending_orders == {}


def test_update_bar_removes_filled_orders_from_pending():
    ex, q = _new_execution(fill_on='next_open')
    ex.execute_order(_order(order_id='A', order_type=OrderType.MKT))
    assert 'A' in ex.pending_orders
    ex.update_bar(_bar(open=100.0))
    assert 'A' not in ex.pending_orders


def test_update_bar_preserves_unfilled_pending_across_many_bars():
    """Pin: no TTL — a pending LMT with an unreachable price sits forever."""
    ex, _ = _new_execution(fill_on='next_open')
    ex.execute_order(_order(qty=1.0, direction=Direction.BUY,
                            order_type=OrderType.LMT, price=1.0,
                            order_id='GHOST'))
    for _ in range(20):
        ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.5))
    assert 'GHOST' in ex.pending_orders


# ──────────────────────────────────────────────
# _emit_fill — slippage / commission / unconditional fill
# ──────────────────────────────────────────────

def test_emit_fill_applies_slippage_to_base_price():
    ex, q = _new_execution(fill_on='signal_close', slippage=('pct', 0.001))
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.0))
    ex.execute_order(_order(qty=1.0, direction=Direction.BUY,
                            order_type=OrderType.MKT))
    # BUY slips up: 100 * (1 + 0.001) = 100.1
    assert math.isclose(q.items[0].fill_notional, 1.0 * 100.1)


def test_emit_fill_calls_commission_at_slipped_price():
    # commission rate 0.001 on notional = qty * slipped_price
    ex, q = _new_execution(fill_on='signal_close',
                           slippage=('pct', 0.001), commission_rate=0.001)
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.0))
    ex.execute_order(_order(qty=2.0, direction=Direction.BUY,
                            order_type=OrderType.MKT))
    # slipped = 100.1; commission = |2 * 100.1| * 0.001 = 0.2002
    assert math.isclose(q.items[0].commission, 0.2002)


def test_emit_fill_writes_fill_event_with_expected_fields():
    ex, q = _new_execution(fill_on='signal_close',
                           exchange_name='TESTEX', commission_rate=0.0004)
    bar_ts = datetime(2026, 3, 15, 9, 30, 0)
    ex.update_bar(_bar(symbol='BTC', ts=bar_ts,
                       open=100.0, high=101.0, low=99.0, close=100.5))
    ex.execute_order(_order(symbol='BTC', qty=1.5, direction=Direction.SELL,
                            order_type=OrderType.MKT, order_id='ORD-42'))
    assert len(q.items) == 1
    f = q.items[0]
    assert f.timestamp == bar_ts
    assert f.symbol == 'BTC'
    assert f.exchange == 'TESTEX'
    assert math.isclose(f.quantity, 1.5)
    assert f.direction == Direction.SELL
    assert math.isclose(f.fill_notional, 1.5 * 100.5)
    assert math.isclose(f.commission, 1.5 * 100.5 * 0.0004)
    assert f.order_id == 'ORD-42'


def test_emit_fill_uses_configured_exchange_name():
    ex, q = _new_execution(fill_on='signal_close', exchange_name='FOOBAR')
    ex.update_bar(_bar())
    ex.execute_order(_order())
    assert q.items[0].exchange == 'FOOBAR'


def test_emit_fill_uses_bar_timestamp_not_order_timestamp():
    ex, q = _new_execution(fill_on='signal_close')
    bar_ts = datetime(2026, 6, 1, 0, 0, 0)
    order_ts = datetime(2025, 1, 1, 0, 0, 0)
    ex.update_bar(_bar(ts=bar_ts))
    ex.execute_order(_order(ts=order_ts))
    assert q.items[0].timestamp == bar_ts


def test_emit_fill_fills_at_full_requested_quantity():
    """
    Decoupling regression: execution no longer scales fills based on
    portfolio margin. Whatever quantity the order asks for, that's what
    the fill emits.
    """
    ex, q = _new_execution(fill_on='signal_close')
    ex.update_bar(_bar(open=100.0, high=101.0, low=99.0, close=100.0))
    ex.execute_order(_order(qty=42.0, direction=Direction.BUY,
                            order_type=OrderType.MKT))
    assert len(q.items) == 1
    assert math.isclose(q.items[0].quantity, 42.0)


# ──────────────────────────────────────────────
# LiveExecution (stub)
# ──────────────────────────────────────────────

def test_live_execution_execute_order_is_noop():
    le = LiveExecution()
    assert le.execute_order(_order()) is None


def test_live_execution_update_bar_is_noop():
    le = LiveExecution()
    assert le.update_bar(_bar()) is None
