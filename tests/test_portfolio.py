"""
Characterization unit tests for `portfolio.py` (BacktestPortfolio).

Uses minimal FakeQueue / FakeDataHandler stubs so the portfolio's margin
math, fill bookkeeping, and equity accounting are exercised in isolation,
independent of execution.py and riskmanager.py. These tests pin the
current behavior of `BacktestPortfolio` and must pass both before and
after any future refactor of the margin / fill path.

Run from the repo root:  pytest tests/test_portfolio.py -v
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from event import (
    BarEvent,
    Direction,
    FillEvent,
    OrderEvent,
    OrderType,
)
from portfolio import BacktestPortfolio, Portfolio


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakeQueue:
    """Captures every put() for inspection."""

    def __init__(self):
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class FakeDataHandler:
    """Minimal data handler — get_latest_bars returns a canned frame per symbol."""

    def __init__(self, frames: Optional[Dict[str, pd.DataFrame]] = None):
        self._frames = frames if frames is not None else {}

    def get_latest_bars(self, symbol: str, n: int) -> pd.DataFrame:
        frame = self._frames.get(symbol)
        if frame is None:
            return pd.DataFrame()
        return frame.tail(n)


# ──────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────

DEFAULT_TS = datetime(2026, 1, 1, 12, 0, 0)


def _new_portfolio(
    symbols: Tuple[str, ...] = ('BTC',),
    capital: float = 100_000.0,
    leverage: float = 1.0,
    prices: Optional[Dict[str, float]] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[BacktestPortfolio, FakeQueue, FakeDataHandler]:
    """Build a BacktestPortfolio with fake queue / data handler. Seeds latest prices if given."""
    q = FakeQueue()
    dh = FakeDataHandler(frames=frames)
    pf = BacktestPortfolio(
        events_queue=q,
        data_handler=dh,
        symbol_list=list(symbols),
        initial_capital=capital,
        leverage=leverage,
    )
    if prices:
        for sym, px in prices.items():
            pf._latest_prices[sym] = px
    return pf, q, dh


def _bar(symbol: str = 'BTC', ts: Optional[datetime] = None,
         close: float = 100.0) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else DEFAULT_TS,
        open=close, high=close, low=close, close=close, volume=1.0,
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


def _fill(symbol: str = 'BTC', qty: float = 1.0,
          direction: Direction = Direction.BUY,
          fill_price: float = 100.0,
          commission: float = 0.0,
          order_id: Optional[str] = None,
          ts: Optional[datetime] = None) -> FillEvent:
    return FillEvent(
        timestamp=ts if ts is not None else DEFAULT_TS,
        symbol=symbol,
        exchange='SIM',
        quantity=qty,
        direction=direction,
        fill_notional=qty * fill_price,
        commission=commission,
        order_id=order_id,
    )


# ──────────────────────────────────────────────
# ABC instantiation guard
# ──────────────────────────────────────────────

def test_portfolio_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Portfolio()  # type: ignore[abstract]


# ──────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────

def test_init_seeds_cash_and_zero_positions():
    pf, _, _ = _new_portfolio(symbols=('BTC', 'ETH'), capital=50_000.0)
    assert pf.cash == 50_000.0
    assert pf.account_balance == 50_000.0
    assert pf.available_balance == 50_000.0
    for sym in ('BTC', 'ETH'):
        assert pf.positions[sym] == 0.0
        assert pf.avg_cost[sym] == 0.0
        assert pf.margin_requirements[sym] == 0.0
        assert pf.realized_pnl[sym] == 0.0
        assert pf.unrealized_pnl[sym] == 0.0
    assert pf.equity_curve == []
    assert pf.trade_log == []
    assert pf.order_log == []
    assert pf.pending_orders == {}


def test_init_stores_leverage_and_symbol_list():
    pf, _, _ = _new_portfolio(symbols=('BTC', 'ETH'), capital=1.0, leverage=5.0)
    assert pf.leverage == 5.0
    assert pf.symbol_list == ['BTC', 'ETH']
    assert pf.initial_capital == 1.0


# ──────────────────────────────────────────────
# update_bar / equity snapshots
# ──────────────────────────────────────────────

def test_update_bar_appends_equity_curve_row():
    pf, _, _ = _new_portfolio()
    pf.update_bar(_bar(close=123.0))
    assert len(pf.equity_curve) == 1
    row = pf.equity_curve[0]
    assert row['timestamp'] == DEFAULT_TS
    assert row['symbol'] == 'BTC'
    assert row['cash'] == 100_000.0
    assert row['account_balance'] == 100_000.0
    assert row['available_balance'] == 100_000.0
    assert row['positions'] == {'BTC': 0.0}
    assert row['margin_requirements'] == {'BTC': 0.0}
    assert row['unrealized_pnl'] == {'BTC': 0.0}
    assert row['realized_pnl'] == {'BTC': 0.0}


def test_update_bar_snapshots_cumulative_realized_pnl_per_symbol():
    """Each equity_curve row freezes the cumulative realized_pnl dict at that bar.

    After a profitable round trip on BTC, the next bar's snapshot carries
    the booked amount; a subsequent fill-less bar leaves the snapshot
    unchanged (cumulative, not per-bar delta).
    """
    pf, _, _ = _new_portfolio()
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    pf.update_fill(_fill(qty=1.0, direction=Direction.SELL, fill_price=110.0))
    pf.update_bar(_bar(close=110.0))
    pf.update_bar(_bar(close=115.0, ts=datetime(2026, 1, 1, 13, 0, 0)))
    assert pf.realized_pnl['BTC'] == 10.0
    assert pf.equity_curve[0]['realized_pnl'] == {'BTC': 10.0}
    assert pf.equity_curve[1]['realized_pnl'] == {'BTC': 10.0}
    # Snapshot is a copy — mutating realized_pnl later doesn't rewrite history.
    pf.realized_pnl['BTC'] = 999.0
    assert pf.equity_curve[0]['realized_pnl'] == {'BTC': 10.0}


def test_update_bar_recalculates_margin_from_open_position():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=2.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    # After fill at 100: margin = 1 * 100 / 2 = 50. After bar at 110: margin = 1 * 110 / 2 = 55.
    pf.update_bar(_bar(close=110.0))
    assert math.isclose(pf.margin_requirements['BTC'], 55.0)


def test_update_bar_recalculates_unrealized_pnl():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    pf.update_bar(_bar(close=110.0))
    assert math.isclose(pf.unrealized_pnl['BTC'], 10.0)
    assert math.isclose(pf.account_balance, pf.cash + 10.0)
    # available = balance - position_margin (110). No reserved margin.
    assert math.isclose(pf.available_balance, pf.account_balance - 110.0)


def test_update_bar_records_zero_returns_on_first_flat_bar():
    """First bar on an unfilled portfolio: balance == initial_capital, so both
    simple_return and log_return are exactly zero."""
    pf, _, _ = _new_portfolio()
    pf.update_bar(_bar(close=100.0))
    row = pf.equity_curve[0]
    assert row['simple_return'] == 0.0
    assert row['log_return'] == 0.0


def test_update_bar_records_simple_period_return_between_bars():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    ts1 = datetime(2026, 1, 1, 12, 0, 0)
    ts2 = datetime(2026, 1, 1, 13, 0, 0)
    pf.update_bar(_bar(ts=ts1, close=100.0))
    pf.update_bar(_bar(ts=ts2, close=110.0))
    balance_0 = pf.equity_curve[0]['account_balance']
    balance_1 = pf.equity_curve[1]['account_balance']
    expected = (balance_1 - balance_0) / balance_0
    assert math.isclose(pf.equity_curve[1]['simple_return'], expected)


def test_update_bar_records_log_return_between_bars():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    ts1 = datetime(2026, 1, 1, 12, 0, 0)
    ts2 = datetime(2026, 1, 1, 13, 0, 0)
    pf.update_bar(_bar(ts=ts1, close=100.0))
    pf.update_bar(_bar(ts=ts2, close=110.0))
    balance_0 = pf.equity_curve[0]['account_balance']
    balance_1 = pf.equity_curve[1]['account_balance']
    expected = math.log(balance_1 / balance_0)
    assert math.isclose(pf.equity_curve[1]['log_return'], expected)


@pytest.mark.parametrize('prior_balance', [0.0, -1.0])
def test_update_bar_returns_are_nan_when_prior_balance_is_nonpositive(prior_balance):
    """When the previous row's account_balance is zero or negative the
    returns are undefined; both columns must be NaN rather than crashing.

    Parametrized over the boundary (0.0) and a clearly negative value so a
    future regression flipping the guard from ``> 0`` to ``>= 0`` would be
    caught.
    """
    pf, _, _ = _new_portfolio()
    pf.equity_curve.append({
        'timestamp': datetime(2026, 1, 1, 11, 0, 0),
        'account_balance': prior_balance,
    })
    pf.update_bar(_bar(close=100.0))
    new_row = pf.equity_curve[-1]
    assert math.isnan(new_row['simple_return'])
    assert math.isnan(new_row['log_return'])


def test_update_bar_log_return_is_nan_when_current_balance_nonpositive():
    """Prior balance > 0 but a catastrophic price drives the current
    ``account_balance`` to zero or negative. ``simple_return`` must remain
    finite (the literal arithmetic is well-defined), but ``log_return``
    must be NaN — ``log`` of a non-positive number is undefined.
    """
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0))
    # First bar at the entry price — locks in a positive prior_balance.
    ts1 = datetime(2026, 1, 1, 12, 0, 0)
    pf.update_bar(_bar(ts=ts1, close=100.0))
    prior_balance = pf.equity_curve[-1]['account_balance']
    assert prior_balance > 0  # sanity: prior_balance > 0 path will be taken

    # Second bar at a deeply negative price. With 1 unit long at avg=100,
    # unrealized PnL ≈ 1 * (-10_000 - 100) = -10_100, driving
    # account_balance ≈ initial_capital + unrealized ≈ -100 ≤ 0.
    ts2 = datetime(2026, 1, 1, 13, 0, 0)
    pf.update_bar(_bar(ts=ts2, close=-10_000.0))
    new_row = pf.equity_curve[-1]
    assert new_row['account_balance'] <= 0  # confirms target branch reached
    assert math.isfinite(new_row['simple_return'])
    assert math.isnan(new_row['log_return'])


# ──────────────────────────────────────────────
# _dir_sign & enum dispatch
# ──────────────────────────────────────────────

def test_dir_sign_buy_returns_positive_one():
    assert BacktestPortfolio._dir_sign(Direction.BUY) == 1.0


def test_dir_sign_sell_returns_negative_one():
    assert BacktestPortfolio._dir_sign(Direction.SELL) == -1.0


def test_dir_sign_invalid_direction_raises():
    with pytest.raises(ValueError):
        BacktestPortfolio._dir_sign("BUY")  # string is not a Direction enum member


# ──────────────────────────────────────────────
# _apply_fill_to_position (pure math)
# ──────────────────────────────────────────────

def test_apply_fill_buy_on_flat_opens_long():
    pf, _, _ = _new_portfolio()
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.BUY, fill_price=100.0, fill_notional=200.0)
    assert new_pos == 2.0
    assert new_avg == 100.0
    assert pnl == 0.0


def test_apply_fill_buy_adds_to_long_weighted_avg_cost():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = 2.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.BUY, fill_price=120.0, fill_notional=240.0)
    # (100*2 + 240) / 4 = 440/4 = 110
    assert new_pos == 4.0
    assert math.isclose(new_avg, 110.0)
    assert pnl == 0.0


def test_apply_fill_sell_on_flat_opens_short():
    pf, _, _ = _new_portfolio()
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.SELL, fill_price=100.0, fill_notional=200.0)
    assert new_pos == -2.0
    assert new_avg == 100.0
    assert pnl == 0.0


def test_apply_fill_sell_adds_to_short_weighted_avg_cost():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = -2.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.SELL, fill_price=80.0, fill_notional=160.0)
    # (100*2 + 160) / 4 = 360/4 = 90
    assert new_pos == -4.0
    assert math.isclose(new_avg, 90.0)
    assert pnl == 0.0


def test_apply_fill_sell_closes_long_realizes_pnl():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = 3.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.SELL, fill_price=110.0, fill_notional=220.0)
    # pnl = (110 - 100) * 2 = 20, remaining long keeps old avg_cost
    assert new_pos == 1.0
    assert new_avg == 100.0
    assert math.isclose(pnl, 20.0)


def test_apply_fill_buy_closes_short_realizes_pnl():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = -3.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.BUY, fill_price=90.0, fill_notional=180.0)
    # pnl = (100 - 90) * 2 = 20, remaining short keeps old avg_cost
    assert new_pos == -1.0
    assert new_avg == 100.0
    assert math.isclose(pnl, 20.0)


def test_apply_fill_sell_closes_long_exactly_zeroes_position():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = 2.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=2.0, direction=Direction.SELL, fill_price=110.0, fill_notional=220.0)
    assert new_pos == 0.0
    assert new_avg == 0.0
    assert math.isclose(pnl, 20.0)


def test_apply_fill_buy_overfills_short_flips_to_long():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=3.0, direction=Direction.BUY, fill_price=90.0, fill_notional=270.0)
    # Cover 1 short at 90: pnl = (100-90)*1 = 10. Remaining 2 open as long at fill price.
    assert new_pos == 2.0
    assert new_avg == 90.0
    assert math.isclose(pnl, 10.0)


def test_apply_fill_sell_overfills_long_flips_to_short():
    pf, _, _ = _new_portfolio()
    pf.positions['BTC'] = 1.0
    pf.avg_cost['BTC'] = 100.0
    new_pos, new_avg, pnl = pf._apply_fill_to_position(
        'BTC', qty=3.0, direction=Direction.SELL, fill_price=110.0, fill_notional=330.0)
    # Close 1 long at 110: pnl = (110-100)*1 = 10. Remaining 2 open as short at fill price.
    assert new_pos == -2.0
    assert new_avg == 110.0
    assert math.isclose(pnl, 10.0)


def test_apply_fill_invalid_direction_raises():
    pf, _, _ = _new_portfolio()
    with pytest.raises(ValueError):
        pf._apply_fill_to_position(
            'BTC', qty=1.0, direction="BUY", fill_price=100.0, fill_notional=100.0)


# ──────────────────────────────────────────────
# update_fill (full state update)
# ──────────────────────────────────────────────

def test_update_fill_open_long_only_reduces_cash_by_commission():
    pf, _, _ = _new_portfolio(capital=10_000.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0, commission=0.5))
    assert math.isclose(pf.cash, 9_999.5)  # no notional flow, only commission
    assert pf.positions['BTC'] == 1.0
    assert pf.avg_cost['BTC'] == 100.0
    assert pf.realized_pnl['BTC'] == 0.0


def test_update_fill_close_long_realizes_pnl_into_cash():
    pf, _, _ = _new_portfolio(capital=10_000.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0, commission=0.5))
    pf.update_fill(_fill(qty=1.0, direction=Direction.SELL, fill_price=110.0, commission=0.5))
    # cash: 10_000 - 0.5 (open) + 10.0 (realized) - 0.5 (close) = 10_009.0
    assert math.isclose(pf.cash, 10_009.0)
    assert pf.positions['BTC'] == 0.0
    assert math.isclose(pf.realized_pnl['BTC'], 10.0)


def test_total_commission_starts_at_zero():
    """total_commission is initialised to 0.0 before any fills are processed."""
    pf, _, _ = _new_portfolio(capital=10_000.0)
    assert pf.total_commission == 0.0


def test_total_commission_accumulates_across_fills():
    """Each fill's commission adds to the running account-level total_commission."""
    pf, _, _ = _new_portfolio(capital=10_000.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0, commission=0.5))
    pf.update_fill(_fill(qty=1.0, direction=Direction.SELL, fill_price=110.0, commission=0.75))
    pf.update_fill(_fill(qty=2.0, direction=Direction.SELL, fill_price=105.0, commission=1.25))
    assert math.isclose(pf.total_commission, 2.5)


def test_update_fill_appends_trade_log_row():
    pf, _, _ = _new_portfolio(capital=10_000.0)
    pf.update_fill(_fill(qty=2.0, direction=Direction.BUY, fill_price=100.0,
                         commission=1.0, order_id='ord-1'))
    assert len(pf.trade_log) == 1
    row = pf.trade_log[0]
    assert row['symbol'] == 'BTC'
    assert row['direction'] == Direction.BUY.value
    assert row['quantity'] == 2.0
    assert math.isclose(row['fill_price'], 100.0)
    assert math.isclose(row['fill_notional'], 200.0)
    assert row['commission'] == 1.0
    assert row['realized_pnl'] == 0.0
    assert row['position_after'] == 2.0
    assert math.isclose(row['cash_after'], 9_999.0)
    assert row['order_id'] == 'ord-1'


def test_update_fill_updates_margin_requirements_at_fill_price():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=4.0)
    pf.update_fill(_fill(qty=2.0, direction=Direction.BUY, fill_price=100.0))
    # margin = |2 * 100| / 4 = 50
    assert math.isclose(pf.margin_requirements['BTC'], 50.0)


def test_update_fill_releases_reserved_margin_on_matching_order_id():
    pf, q, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None
    assert order.order_id in pf.pending_orders
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0,
                         order_id=order.order_id))
    assert order.order_id not in pf.pending_orders


def test_update_fill_unknown_order_id_leaves_pending_orders_intact():
    pf, _, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0,
                         order_id='not-a-real-id'))
    assert order.order_id in pf.pending_orders


# ──────────────────────────────────────────────
# submit_order & validation
# ──────────────────────────────────────────────

def test_submit_order_mkt_with_price_raises_value_error():
    pf, _, _ = _new_portfolio(prices={'BTC': 100.0})
    with pytest.raises(ValueError):
        pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                        timestamp=DEFAULT_TS, order_type=OrderType.MKT, price=100.0)


def test_submit_order_zero_quantity_returns_none():
    pf, q, _ = _new_portfolio(prices={'BTC': 100.0})
    result = pf.submit_order('BTC', quantity=0.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert result is None
    assert q.items == []
    assert pf.order_log == []
    assert pf.pending_orders == {}


def test_submit_order_mkt_with_no_cached_price_returns_none():
    pf, q, _ = _new_portfolio()  # no prices, no frames -> get_price returns None
    result = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert result is None
    assert q.items == []


def test_submit_order_lmt_with_nan_price_returns_none():
    pf, q, _ = _new_portfolio(prices={'BTC': 100.0})
    result = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.LMT,
                             price=float('nan'))
    assert result is None
    assert q.items == []


def test_submit_order_lmt_with_zero_price_returns_none():
    # Exact zero is rejected (margin scaling would divide-by-zero). Negative
    # prices are allowed — see test_submit_order_lmt_with_negative_price_succeeds.
    pf, q, _ = _new_portfolio(prices={'BTC': 100.0})
    result = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.LMT,
                             price=0.0)
    assert result is None
    assert q.items == []


def test_submit_order_lmt_with_negative_price_succeeds():
    # WTI-2020-style negative LMT prices must flow through; downstream margin
    # uses ``abs(price)`` so reservations stay positive.
    pf, q, _ = _new_portfolio(capital=1_000_000.0, prices={'BTC': 100.0})
    result = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.LMT,
                             price=-37.0)
    assert isinstance(result, OrderEvent)
    assert result.price == -37.0


def test_submit_order_success_enqueues_event_and_logs():
    pf, q, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert isinstance(order, OrderEvent)
    assert order.quantity == 1.0
    assert order.direction == Direction.BUY
    assert order.order_type == OrderType.MKT
    assert len(q.items) == 1 and q.items[0] is order
    assert order.order_id in pf.pending_orders
    assert len(pf.order_log) == 1
    log_row = pf.order_log[0]
    assert log_row['symbol'] == 'BTC'
    assert log_row['order_type'] == OrderType.MKT.value
    assert log_row['direction'] == Direction.BUY.value
    assert log_row['quantity'] == 1.0
    assert log_row['order_id'] == order.order_id
    assert 'signal_type' not in log_row


def test_submit_order_scales_quantity_when_insufficient_margin():
    # capital=1000, leverage=1, price=100 -> max new pos |qty|=10
    pf, q, _ = _new_portfolio(capital=1_000.0, leverage=1.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=20.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None
    assert math.isclose(order.quantity, 10.0)
    assert q.items[0] is order


def test_submit_order_zero_approvable_quantity_returns_none():
    pf, q, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    pf.cash = 0.0  # drain wallet so available balance is 0
    result = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                             timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert result is None
    assert q.items == []
    assert pf.pending_orders == {}


# ──────────────────────────────────────────────
# Margin calculations
# ──────────────────────────────────────────────

def test_calculate_new_margin_uses_new_position_notional():
    pf, _, _ = _new_portfolio(leverage=2.0)
    # baseline pos=0 -> BUY 2 @ 100 -> new_pos = 2 -> margin = 2*100/2 = 100
    assert math.isclose(
        pf._calculate_new_margin(pos=0.0, qty=2.0, direction=Direction.BUY,
                                 price=100.0),
        100.0,
    )


def test_calculate_new_margin_uniform_for_mkt_and_lmt():
    """
    The LMT worst-case clause (max(abs(new_pos), qty)) has been removed.
    Both MKT and LMT now use abs(new_pos) * price / leverage.
    """
    pf, _, _ = _new_portfolio(leverage=1.0)
    # pos=-1, qty=2, BUY -> new_pos = 1 -> margin = 1 * 100 / 1 = 100 (no worst-case 200)
    assert math.isclose(
        pf._calculate_new_margin(pos=-1.0, qty=2.0, direction=Direction.BUY,
                                 price=100.0),
        100.0,
    )


def test_reserved_margin_split_formula_walks_pending_in_fifo_order():
    """
    Reserved margin is computed per-order via the risk-reducing /
    risk-increasing split, walked in FIFO order so each order's split is
    measured against the running projected position that includes earlier
    pending orders.

    Setup: pos=+1 (margin=100), pending BUY LMT 1 @ 100 then SELL MKT 1.

    FIFO walk:
      - BUY 1 against running_pos=+1: dir_sign=+1, reducing_capacity=0,
        increasing=1 -> 100 reserved. running_pos becomes +2.
      - SELL 1 against running_pos=+2: dir_sign=-1, reducing_capacity=2,
        reducing=1, increasing=0 -> 0 reserved. running_pos becomes +1.

    Total reserved = 100. Even though the SELL would offset the BUY at
    fill time, the SELL might not fill (or fill later); the conservative
    reservation keeps the BUY's risk-increasing exposure committed.
    """
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.margin_requirements['BTC'] = 100.0
    risk_up = _order(qty=1.0, direction=Direction.BUY, order_type=OrderType.LMT,
                     price=100.0, order_id='up')
    risk_down = _order(qty=1.0, direction=Direction.SELL, order_type=OrderType.MKT,
                       order_id='down')
    pf.pending_orders = {'up': risk_up, 'down': risk_down}
    assert math.isclose(pf._reserved_margin(), 100.0)


def test_reserved_margin_sums_net_increase_for_opening_orders():
    """
    Two pending BUYs that both add to the position contribute the full
    aggregate increase as reserved margin (each is purely risk-increasing
    against the running projection).
    """
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.margin_requirements['BTC'] = 100.0
    a = _order(qty=1.0, direction=Direction.BUY, order_type=OrderType.MKT, order_id='A')
    b = _order(qty=2.0, direction=Direction.BUY, order_type=OrderType.MKT, order_id='B')
    pf.pending_orders = {'A': a, 'B': b}
    # FIFO walk: A increasing=1 (margin 100), B increasing=2 (margin 200). Total = 300.
    assert math.isclose(pf._reserved_margin(), 300.0)


def test_reserved_margin_lmt_uses_limit_price_not_mark():
    """LMT pending orders reserve at limit price, not current safe price."""
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    a = _order(qty=1.0, direction=Direction.BUY, order_type=OrderType.LMT,
               price=80.0, order_id='A')
    pf.pending_orders = {'A': a}
    # increasing=1 at limit 80 -> reserved = 80 (not 100).
    assert math.isclose(pf._reserved_margin(), 80.0)


def test_reserved_margin_excludes_given_order_id():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    a = _order(qty=1.0, direction=Direction.BUY, order_id='A')
    b = _order(qty=1.0, direction=Direction.BUY, order_id='B')
    pf.pending_orders = {'A': a, 'B': b}
    # Each order contributes 100 (MKT BUY from flat -> increasing=1).
    assert math.isclose(pf._reserved_margin(), 200.0)
    assert math.isclose(pf._reserved_margin(exclude_order_id='A'), 100.0)
    assert math.isclose(pf._reserved_margin(exclude_order_id='B'), 100.0)


def test_calculate_available_balance_matches_expected_formula():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    pf.margin_requirements['BTC'] = 200.0
    # No unrealized pnl, no pending orders. balance = cash = 10_000; used = 200.
    assert math.isclose(pf.calculate_available_balance(), 9_800.0)


# ──────────────────────────────────────────────
# Decoupling regression: validate_fill is removed
# ──────────────────────────────────────────────

def test_validate_fill_method_no_longer_exists():
    """
    Decoupling regression: BacktestExecution no longer calls
    portfolio.validate_fill, and the method itself has been removed.
    Pinning its absence prevents accidental reintroduction.
    """
    pf, _, _ = _new_portfolio()
    assert not hasattr(pf, 'validate_fill')


# ──────────────────────────────────────────────
# Same-bar flip — projected-position margin check (C1 regression)
# ──────────────────────────────────────────────

def test_submit_order_same_bar_flip_does_not_overleverage_account():
    """
    REGRESSION (audit C1): same-bar FLATTEN + OPEN_OPPOSITE must not exceed
    the configured leverage cap.

    Setup: capital=100, leverage=1.0, currently long 1 BTC @ 100 (margin 100).
    Strategy emits FLATTEN (SELL 1) + OPEN_SHORT (SELL 2) at the same bar.
    Without projecting position through pending orders, both orders pass the
    margin check because each is evaluated against stale `positions[BTC] = +1`,
    yielding final position short 2 (notional 200) on 100 of capital — i.e.
    2x the configured 1x cap. The fix must scale ORDER2 down so the achievable
    end-state position respects the leverage cap.
    """
    pf, _, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0  # |1 * 100| / 1

    order1 = pf.submit_order(
        'BTC', quantity=1.0, direction=Direction.SELL,
        timestamp=DEFAULT_TS, order_type=OrderType.MKT,
    )
    assert order1 is not None and order1.quantity == 1.0  # risk-reducing, passes

    order2 = pf.submit_order(
        'BTC', quantity=2.0, direction=Direction.SELL,
        timestamp=DEFAULT_TS, order_type=OrderType.MKT,
    )
    assert order2 is not None
    # ORDER1 (pending) projects position to 0; ORDER2 of qty=2 would take it
    # to short 2 (margin 200) on capital 100 — over the 1x cap. Must scale to 1.
    assert math.isclose(order2.quantity, 1.0), (
        f"ORDER2 should be scaled to 1.0 to respect leverage cap, got {order2.quantity}"
    )


# ──────────────────────────────────────────────
# Projection asymmetry — LMT pendings excluded, MKT pendings included
# ──────────────────────────────────────────────

def test_submit_order_pending_lmt_does_not_pre_credit_margin_freedom():
    """
    REGRESSION: a pending LMT order is a conditional fill (price may never
    reach the limit), so the projected-position margin check must NOT
    credit its hypothetical margin freeing when sizing a new order.

    Setup: BTC @ 100, leverage 10x (so 1 BTC needs $10 margin).
      - Realized: short 5 BTC, margin = $50.
      - Pending LMT BUY 2 @ 100 (would reduce short if it fills).
      - Cash $40, unrealized 0 -> account_balance $40, available -$10.

    A new MKT SELL 4 must be REJECTED. If LMT were projected (the old
    behavior), baseline pos would be -3 instead of -5 and the scaler
    would approve a SELL of 1 — leaving the account at short 6 / margin
    $60 if the LMT never fills. Pinning rejection prevents regression.
    """
    pf, q, _ = _new_portfolio(capital=40.0, leverage=10.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = -5.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 50.0  # |-5 * 100| / 10
    lmt_buy_2 = _order(qty=2.0, direction=Direction.BUY,
                       order_type=OrderType.LMT, price=100.0, order_id='LMT_BUY_2')
    pf.pending_orders = {'LMT_BUY_2': lmt_buy_2}

    # Sanity: available is the negative-cushion situation we care about.
    assert pf.calculate_available_balance() == pytest.approx(-10.0)

    order = pf.submit_order(
        'BTC', quantity=4.0, direction=Direction.SELL,
        timestamp=DEFAULT_TS, order_type=OrderType.MKT,
    )
    assert order is None, (
        f"MKT SELL 4 must be rejected when an LMT BUY 2 is pending and "
        f"available is negative; got {order}"
    )
    # Only the original LMT remains in the queue — no new order entered.
    assert list(pf.pending_orders.keys()) == ['LMT_BUY_2']


def test_submit_order_pending_mkt_still_projects_for_same_bar_flip():
    """
    Mirror of the LMT test above: a pending **MKT** order IS projected
    (MKT fills on the next eligible bar by construction), so its freeing
    effect on margin is credited when sizing a new order. Documents the
    asymmetric semantics — MKT projects, LMT doesn't.

    Setup: same as above but the pending order is MKT BUY 2 instead of
    LMT. Projected pos is -3; the new MKT SELL 4 is sized against pos=-3
    and scales down to 1 (not 0) because freed margin from the MKT BUY
    can be reused.
    """
    pf, q, _ = _new_portfolio(capital=40.0, leverage=10.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = -5.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 50.0
    mkt_buy_2 = _order(qty=2.0, direction=Direction.BUY,
                       order_type=OrderType.MKT, order_id='MKT_BUY_2')
    pf.pending_orders = {'MKT_BUY_2': mkt_buy_2}

    assert pf.calculate_available_balance() == pytest.approx(-10.0)

    order = pf.submit_order(
        'BTC', quantity=4.0, direction=Direction.SELL,
        timestamp=DEFAULT_TS, order_type=OrderType.MKT,
    )
    assert order is not None, (
        "MKT SELL 4 should be partially approved when a pending MKT BUY 2 "
        "is projected to free margin"
    )
    assert math.isclose(order.quantity, 1.0), (
        f"MKT SELL should scale to 1.0 (max_new_margin = baseline + "
        f"available = 50 + -10 = 40 -> max_abs_pos 4 -> max_qty = 4 + "
        f"(-3 projected) = 1); got {order.quantity}"
    )


def test_submit_order_pending_lmt_risk_increasing_still_reserves_capital():
    """
    Excluding LMT from `_projected_position` must NOT exclude it from
    `_reserved_margin`. A pending LMT that ADDS exposure still consumes
    capital, so a new order is sized against the smaller available
    headroom.

    Setup: flat, capital $60, leverage 10x. Pending LMT BUY 5 @ 100
    reserves $50 (risk-increasing from flat). A new MKT BUY 2 (which
    would itself need $20) sees available = $60 - $0 (position margin)
    - $50 (LMT reservation) = $10. Scaler caps the new qty at $10 / $10
    = 1 BTC, not the requested 2.
    """
    pf, q, _ = _new_portfolio(capital=60.0, leverage=10.0, prices={'BTC': 100.0})
    lmt_buy_5 = _order(qty=5.0, direction=Direction.BUY,
                       order_type=OrderType.LMT, price=100.0, order_id='LMT_BUY_5')
    pf.pending_orders = {'LMT_BUY_5': lmt_buy_5}

    # Sanity: LMT BUY 5 reserves 5 * 100 / 10 = 50.
    assert pf._reserved_margin() == pytest.approx(50.0)
    assert pf.calculate_available_balance() == pytest.approx(10.0)

    order = pf.submit_order(
        'BTC', quantity=2.0, direction=Direction.BUY,
        timestamp=DEFAULT_TS, order_type=OrderType.MKT,
    )
    assert order is not None
    assert math.isclose(order.quantity, 1.0), (
        f"MKT BUY 2 should scale to 1.0 because the pending LMT BUY 5 "
        f"still reserves $50 of capital (only projection ignores it, "
        f"not reservation); got {order.quantity}"
    )


# ──────────────────────────────────────────────
# Solvency / liquidation
# ──────────────────────────────────────────────

def test_check_solvency_noop_when_account_balance_nonnegative():
    """
    A solvent account (account_balance >= 0) must not trigger any cancel or
    liquidation side-effects. Pin this so a healthy portfolio is not touched
    by check_solvency.
    """
    pf, q, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    pf.check_solvency(DEFAULT_TS)
    assert q.items == []  # no liquidation orders queued
    assert pf.positions['BTC'] == 1.0  # position untouched


def test_check_solvency_tolerates_negative_available_balance():
    """
    available_balance < 0 alone does NOT trigger liquidation. Only
    account_balance < 0 does. Pin this so a position sitting at unrealized
    loss (but with cash + unrealized still > 0) is not liquidated.
    """
    pf, q, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0  # exactly maxed out
    # Adverse mark to 95: unrealized_pnl = -5; account_balance = 100-5 = 95 (>0).
    pf.update_bar(_bar(close=95.0))
    assert pf.account_balance == pytest.approx(95.0)
    # available_balance = 95 - 95 - 0 = 0; allowed to be 0 or even negative.
    # Critically: NO liquidation order was queued.
    assert all(getattr(item, 'is_liquidation', False) is False for item in q.items)


def test_check_solvency_triggers_on_negative_account_balance_via_update_bar():
    """
    update_bar must call check_solvency, and when the resulting mark drives
    account_balance < 0, a full-close MKT liquidation order (flagged
    is_liquidation=True) must be queued for the underwater position.
    """
    pf, q, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = 1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    # Crash to 50: unrealized = -50; cash=100; account_balance = 50. Still solvent.
    # Crash further to -50? Not possible for a long. Let's open short instead.
    pf2, q2, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf2.positions['BTC'] = -1.0
    pf2.avg_cost['BTC'] = 100.0
    pf2.margin_requirements['BTC'] = 100.0
    # Mark rallies to 250: unrealized = -1*(250-100) = -150. account_balance = 100 - 150 = -50.
    pf2.update_bar(_bar(close=250.0))
    # check_solvency should have fired and queued a liquidation MKT BUY 1.
    liq_orders = [item for item in q2.items
                  if isinstance(item, OrderEvent) and item.is_liquidation]
    assert len(liq_orders) == 1
    liq = liq_orders[0]
    assert liq.symbol == 'BTC'
    assert liq.direction == Direction.BUY  # closing the short
    assert math.isclose(liq.quantity, 1.0)  # full close
    assert liq.order_type == OrderType.MKT


def test_check_solvency_cancels_non_liquidation_pending_fifo():
    """
    When insolvency triggers, every pending non-liquidation order must be
    cancelled before liquidation orders are queued, so stale TP/SL/entry
    LMTs cannot fill alongside the forced close-out.
    """
    pf, q, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    # Pre-load a pending non-liquidation order (e.g. a TP LMT that hasn't filled).
    p = _order(qty=1.0, direction=Direction.BUY, order_type=OrderType.LMT,
               price=90.0, order_id='TP')
    pf.pending_orders = {'TP': p}
    # Drive insolvency.
    pf.update_bar(_bar(close=250.0))
    assert 'TP' not in pf.pending_orders, "non-liquidation pending must be cancelled"


def test_check_solvency_liquidation_orders_exempt_from_fifo_cancel():
    """
    A liquidation order queued on a previous tick (e.g. under
    fill_on='next_open' before it has filled) must not be cancelled by
    the FIFO cancel pass that fires when account_balance is again < 0.
    """
    pf, _, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    # Manually pre-load a pending liquidation order.
    liq = OrderEvent(
        symbol='BTC', order_type=OrderType.MKT, quantity=1.0,
        direction=Direction.BUY, order_id='LIQ-PRE',
        timestamp=DEFAULT_TS, is_liquidation=True,
    )
    pf.pending_orders = {'LIQ-PRE': liq}
    # Crash drives account_balance < 0; FIFO cancel pass runs but must skip LIQ-PRE.
    pf.update_bar(_bar(close=250.0))
    assert 'LIQ-PRE' in pf.pending_orders


def test_check_solvency_liquidates_worst_pnl_first_with_two_positions():
    """
    With multiple open positions, liquidation orders must be queued in
    ascending unrealized_pnl (worst loser first) so the most damaging
    position is closed first if downstream fills only partially complete.
    """
    pf, q, _ = _new_portfolio(symbols=('BTC', 'ETH'), capital=200.0,
                              leverage=1.0,
                              prices={'BTC': 100.0, 'ETH': 200.0})
    # BTC: short 1 @ 100 -> mark to 150 -> unrealized = -50.
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    # ETH: short 1 @ 200 -> mark to 350 -> unrealized = -150 (worse).
    pf.positions['ETH'] = -1.0
    pf.avg_cost['ETH'] = 200.0
    pf.margin_requirements['ETH'] = 200.0
    pf._latest_prices = {'BTC': 150.0, 'ETH': 350.0}
    # Drive update via a bar on either symbol so check_solvency fires.
    # account_balance = 200 - 50 - 150 = 0... need it < 0. Push ETH harder.
    pf._latest_prices['ETH'] = 400.0  # ETH unrealized = -200; account = 200 - 50 - 200 = -50.
    pf.update_bar(_bar(symbol='ETH', close=400.0))
    liq_orders = [item for item in q.items
                  if isinstance(item, OrderEvent) and item.is_liquidation]
    # Both positions should be liquidated, ETH first (worst unrealized).
    assert len(liq_orders) == 2
    assert liq_orders[0].symbol == 'ETH'
    assert liq_orders[1].symbol == 'BTC'


def test_check_solvency_skips_symbol_with_existing_pending_liquidation():
    """Duplicate-submission guard: if a liquidation is already pending
    for a symbol, do not enqueue another one when check_solvency fires."""
    pf, q, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    pre = OrderEvent(
        symbol='BTC', order_type=OrderType.MKT, quantity=1.0,
        direction=Direction.BUY, order_id='LIQ-EXISTING',
        timestamp=DEFAULT_TS, is_liquidation=True,
    )
    pf.pending_orders = {'LIQ-EXISTING': pre}
    pf.update_bar(_bar(close=250.0))
    new_liq = [item for item in q.items
               if isinstance(item, OrderEvent) and item.is_liquidation]
    assert new_liq == []  # no new liquidation order enqueued for BTC


def test_check_solvency_after_update_fill_on_negative_balance():
    """update_fill also runs check_solvency; a fill that lands the account
    below zero (e.g. severe slippage) triggers the liquidation flow."""
    pf, q, _ = _new_portfolio(capital=10.0, leverage=1.0, prices={'BTC': 100.0})
    # Existing short of 1 at avg cost 100, but the mark has rallied so the
    # position is at heavy unrealized loss. Simulate by injecting state.
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    pf._latest_prices['BTC'] = 200.0  # unrealized = -100; account = 10 - 100 = -90.
    # A fill (e.g. partial cover) arrives — update_fill recomputes and runs solvency.
    pf.update_fill(_fill(qty=0.0, direction=Direction.BUY, fill_price=200.0))
    liq = [item for item in q.items
           if isinstance(item, OrderEvent) and item.is_liquidation]
    assert len(liq) >= 1  # at least one liquidation order queued


def test_submit_order_accepts_is_liquidation_flag():
    """The is_liquidation kwarg propagates to OrderEvent."""
    pf, q, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.SELL,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT,
                            is_liquidation=True)
    assert order is not None
    assert order.is_liquidation is True
    # Order log captures the flag too.
    assert pf.order_log[-1]['is_liquidation'] is True


def test_check_solvency_recursion_terminates_after_liquidation_chain():
    """
    End-to-end: an insolvent fill drives `update_fill -> check_solvency ->
    submit liquidation -> (next loop iteration applies the liquidation fill)
    -> update_fill -> check_solvency`. The recursion must terminate when
    there are no more positions to liquidate, and final state must be
    consistent (positions == 0, no infinite loop).

    This guards against future changes to `_liquidate_all_positions` that
    might fire even with zero positions (e.g. a "hedge" or min-notional
    fallback) — the loop's current safety arises implicitly from "no
    positions left, nothing to liquidate."
    """
    pf, q, _ = _new_portfolio(capital=10.0, leverage=1.0, prices={'BTC': 100.0})
    # Existing short of 1 at avg cost 100; mark rallies to 200 -> unrealized = -100.
    # account_balance = cash(10) + unrealized(-100) = -90, well past insolvency.
    pf.positions['BTC'] = -1.0
    pf.avg_cost['BTC'] = 100.0
    pf.margin_requirements['BTC'] = 100.0
    pf._latest_prices['BTC'] = 200.0

    # Drive the chain: a tiny fill (qty=0.0) lands on update_fill which runs
    # check_solvency and queues the liquidation order. We then drain the queue
    # by simulating execution: pop each OrderEvent, build a FillEvent at the
    # cached mark, hand it back to update_fill. Bound the loop so a real
    # infinite loop fails the test instead of hanging the suite.
    pf.update_fill(_fill(qty=0.0, direction=Direction.BUY, fill_price=200.0))

    iterations = 0
    MAX_ITERATIONS = 10
    while iterations < MAX_ITERATIONS:
        iterations += 1
        pending_orders = [o for o in q.items if isinstance(o, OrderEvent)]
        # Take only orders not yet "filled" — track via an attribute on the test queue.
        unfilled = [o for o in pending_orders
                    if o.order_id in pf.pending_orders]
        if not unfilled:
            break
        order = unfilled[0]
        # Apply the fill at the cached mark (no slippage in this test).
        mark = pf._latest_prices[order.symbol]
        fill = _fill(symbol=order.symbol, qty=order.quantity,
                     direction=order.direction, fill_price=mark,
                     order_id=order.order_id)
        pf.update_fill(fill)

    assert iterations < MAX_ITERATIONS, (
        "check_solvency -> liquidation chain failed to terminate"
    )
    # All positions closed and no pending liquidation orders remain.
    assert pf.positions['BTC'] == 0.0
    assert pf.pending_orders == {}


def test_fill_at_worse_price_than_submission_can_drive_insolvency_and_triggers_liquidation():
    """
    Pin the fill-time-validation trade-off: margin is reserved at submission
    price; between submission and fill the price can move arbitrarily, and
    the FIFO reservation walk does NOT re-validate at fill time. A fill at
    a much-worse-than-submission price can therefore drive `account_balance`
    below zero in one bar. The portfolio must catch this AFTER the fill is
    booked (cash and realized P&L have already moved) via check_solvency
    and queue a liquidation order.

    This is by-design behavior for the backtest model — `LiveExecution`
    must implement its own pre-fill margin gating.
    """
    # Capital 100, leverage 1.0. Submit a MKT BUY 1.0 @ ref price 100 — passes
    # margin check (notional 100 = available capital).
    pf, q, _ = _new_portfolio(capital=100.0, leverage=1.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None and order.quantity == 1.0
    # Simulate a gap-up: the actual fill price is 300 (3x submission), simulating
    # a `fill_on='next_open'` gap. No re-validation happens — fill is booked.
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=300.0,
                         order_id=order.order_id))
    # Cash unchanged by opening fill (futures model — no commission here).
    # Position is long 1 @ avg_cost 300. Subsequent bar marks at 100 (mark
    # snaps back, deep unrealized loss).
    pf.update_bar(_bar(close=100.0))
    # unrealized = 1 * (100 - 300) = -200; account = 100 + (-200) = -100. Insolvent.
    assert pf.account_balance < 0
    liq_orders = [item for item in q.items
                  if isinstance(item, OrderEvent) and item.is_liquidation]
    assert len(liq_orders) >= 1, (
        "check_solvency must queue a liquidation order after the bad fill drives insolvency"
    )
    assert liq_orders[0].symbol == 'BTC'
    assert liq_orders[0].direction == Direction.SELL  # close the long


# ──────────────────────────────────────────────
# cancel_order
# ──────────────────────────────────────────────

def test_cancel_order_removes_pending_entry():
    pf, _, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None
    pf.cancel_order(order.order_id)
    assert pf.pending_orders == {}


def test_cancel_order_frees_reserved_margin():
    pf, _, _ = _new_portfolio(capital=10_000.0, leverage=1.0, prices={'BTC': 100.0})
    order = pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                            timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    assert order is not None
    assert math.isclose(pf._reserved_margin(), 100.0)
    pf.cancel_order(order.order_id)
    assert pf._reserved_margin() == 0.0


def test_cancel_order_unknown_id_is_noop():
    pf, _, _ = _new_portfolio()
    pf.cancel_order('does-not-exist')  # must not raise
    assert pf.pending_orders == {}


# ──────────────────────────────────────────────
# Price helpers
# ──────────────────────────────────────────────

def test_get_price_returns_cached_price():
    pf, _, _ = _new_portfolio(prices={'BTC': 123.0})
    assert pf.get_price('BTC') == 123.0


def test_get_price_falls_back_to_data_handler_when_uncached():
    frame = pd.DataFrame(
        {'Open': [10.0], 'High': [10.0], 'Low': [10.0], 'Close': [42.5], 'Volume': [1.0]},
        index=pd.date_range('2026-01-01', periods=1, freq='1h', tz='UTC'),
    )
    pf, _, _ = _new_portfolio(frames={'BTC': frame})
    assert pf.get_price('BTC') == 42.5
    assert pf._latest_prices['BTC'] == 42.5  # cached after lookup


def test_get_price_returns_none_when_no_bars():
    pf, _, _ = _new_portfolio()  # FakeDataHandler with no frames
    assert pf.get_price('BTC') is None


# ──────────────────────────────────────────────
# Record exports
# ──────────────────────────────────────────────

def test_get_equity_curve_returns_dataframe_indexed_by_timestamp():
    pf, _, _ = _new_portfolio()
    ts1 = datetime(2026, 1, 1, 12, 0, 0)
    ts2 = datetime(2026, 1, 1, 13, 0, 0)
    pf.update_bar(_bar(ts=ts1, close=100.0))
    pf.update_bar(_bar(ts=ts2, close=101.0))
    df = pf.get_equity_curve()
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == [ts1, ts2]
    assert {'cash', 'account_balance', 'available_balance'}.issubset(df.columns)


def test_get_equity_curve_empty_when_no_bars():
    pf, _, _ = _new_portfolio()
    df = pf.get_equity_curve()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_trade_log_returns_dataframe_with_expected_columns():
    pf, _, _ = _new_portfolio(capital=10_000.0)
    pf.update_fill(_fill(qty=1.0, direction=Direction.BUY, fill_price=100.0,
                         commission=0.5, order_id='ord-1'))
    df = pf.get_trade_log()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    expected = {'timestamp', 'symbol', 'direction', 'quantity', 'fill_price',
                'fill_notional', 'commission', 'realized_pnl', 'position_after',
                'cash_after', 'order_id'}
    assert expected.issubset(df.columns)


def test_get_trade_log_empty_when_no_fills():
    pf, _, _ = _new_portfolio()
    df = pf.get_trade_log()
    assert df.empty


def test_get_order_log_returns_dataframe_with_expected_columns():
    pf, _, _ = _new_portfolio(capital=10_000.0, prices={'BTC': 100.0})
    pf.submit_order('BTC', quantity=1.0, direction=Direction.BUY,
                    timestamp=DEFAULT_TS, order_type=OrderType.MKT)
    df = pf.get_order_log()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    expected = {'timestamp', 'symbol', 'order_type', 'direction', 'quantity',
                'order_id'}
    assert expected.issubset(df.columns)
    assert 'signal_type' not in df.columns


def test_get_order_log_empty_when_no_orders():
    pf, _, _ = _new_portfolio()
    df = pf.get_order_log()
    assert df.empty
