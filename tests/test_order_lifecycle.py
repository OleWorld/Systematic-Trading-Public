"""
Engine-level order-lifecycle regression tests.

These wire REAL modules (HistoricDataHandler, BacktestPortfolio,
VolTargetingRiskManager, BacktestExecution, Backtester) — unlike the
routing tests in ``test_backtester.py``, which use recording doubles —
because the bugs they pin lived in the *interaction* between modules:

1. Sizing idempotency: with a constant forecast/sigma/price, the system
   must produce exactly ONE fill and then hold (the old
   ``fill_on='next_open'`` mode double-submitted every resize and the
   position oscillated 0→10→20→10→0 forever).
2. Margin-call bar: the risk manager runs AFTER ``check_solvency`` in the
   same bar, so a liquidation order is in flight when it resizes. It must
   size against the projected position (realized + pending MKT) — the old
   realized-only diff left the account with a WRONG-SIGN position (short
   with a long forecast).
3. Cancelled-order fills are voided: the margin-call cancel pass removes
   an order from the portfolio's ledger, but the simulated exchange may
   still deliver its fill — booking it would resurrect a cancelled order.

Run from the repo root:  pytest tests/test_order_lifecycle.py -v
"""

import logging
import queue as thread_queue
from typing import Optional

import pandas as pd
import pytest

from backtester import Backtester
from config import uniform_registry
from data import HistoricDataHandler
from event import BarEvent
from execution import BacktestExecution, CommissionModel, SlippageModel
from portfolio import BacktestPortfolio, PortfolioMarginModel
from riskmanager import VolTargetingRiskManager
from strategy import Strategy
from volatility import VolEstimator


SYM = 'X'


class ConstLongStrategy(Strategy):
    """Forecast +50 (long, half conviction) on every bar."""

    def calculate_forecast(self, event: BarEvent):
        return {'forecast': 50.0}


class StubVol(VolEstimator):
    """sigma = ``base`` until the ``crash_bar``-th completed-bar update,
    then ``spiked`` (a vol spike shrinking the Carver target)."""

    def __init__(self, base: float = 10.0, spiked: Optional[float] = None,
                 crash_bar: Optional[int] = None):
        self.base = base
        self.spiked = spiked
        self.crash_bar = crash_bar
        self.count = 0

    def update(self, event: BarEvent) -> None:
        self.count += 1

    def get_annual_vol(self, symbol: str) -> Optional[float]:
        if (self.crash_bar is not None and self.spiked is not None
                and self.count >= self.crash_bar):
            return self.spiked
        return self.base


def _run_engine(closes, *, capital: float, vol: StubVol,
                annual_target_vol: float = 100.0):
    """Wire the real module graph over a synthetic single-symbol price
    series (10x leverage, 5% maintenance margin — the smoke-runner margin
    setup) and run it. Returns ``(portfolio, risk_manager, index)``."""
    idx = pd.date_range('2024-01-01', periods=len(closes), freq='D', tz='UTC')
    df = pd.DataFrame({'Open': closes, 'High': closes, 'Low': closes,
                       'Close': closes, 'Volume': [1.0] * len(closes)},
                      index=idx)
    events = thread_queue.Queue()
    instruments = uniform_registry(
        [SYM], point_value=1.0, fractional=True,
        slippage=SlippageModel('absolute', 0.0),
        commission=CommissionModel('per_contract', 0.0),
        margin=PortfolioMarginModel.from_leverage(
            10.0, maintenance_margin_rate=0.05,
        ),
    )
    dh = HistoricDataHandler(events, [SYM], '1d', {'1d': 100}, data={SYM: df})
    strategy = ConstLongStrategy(dh, [SYM])
    portfolio = BacktestPortfolio(events, dh, [SYM], instruments,
                                  initial_capital=capital)
    rm = VolTargetingRiskManager(
        portfolio, strategy, vol, data_handler=dh, instruments=instruments,
        annual_target_vol=annual_target_vol,
        vol_target_mode='dollar_volatility',
        position_buffer=0.25, instrument_weight_mode='equal_weight',
        corr_lookback=31, corr_step_size=5, corr_timeframe='1d',
    )
    execution = BacktestExecution(events, instruments)
    Backtester(events, dh, strategy, portfolio, rm, execution).run()
    return portfolio, rm, idx


def test_stable_target_produces_exactly_one_fill():
    """Constant forecast/sigma/price ⇒ constant target of 10 contracts ⇒
    ONE entry fill, then hold. Regression for the resize double-submission
    (the position used to oscillate 0→10→20→10→0 indefinitely)."""
    pf, _, _ = _run_engine([100.0] * 60, capital=1_000_000.0, vol=StubVol())
    trades = pf.get_trade_log()
    assert len(trades) == 1, trades
    assert trades.iloc[0]['direction'] == 'BUY' or \
        trades.iloc[0]['direction'].value == 'BUY'
    assert pf.positions[SYM] == pytest.approx(10.0)  # = τ 100 / sigma 10


def test_margin_call_bar_resizes_against_projected_position():
    """Crash bar: long 10, equity gap-down triggers the maintenance-margin
    call (liquidation SELL 10 in flight) while the vol spike shrinks the
    target to 4. The RM must size against the PROJECTED position (0) and
    BUY 4 — ending the bar at the target, long 4. The old realized-only
    diff submitted SELL 6 and ended the bar SHORT 6 with a +50 LONG
    forecast."""
    closes = [100.0] * 40 + [51.0] * 3
    pf, rm, idx = _run_engine(
        closes, capital=510.0,
        vol=StubVol(base=10.0, spiked=25.0, crash_bar=41),
    )
    crash_ts = idx[40]

    # End state: at the (post-spike) target, never wrong-sign.
    assert pf.positions[SYM] == pytest.approx(4.0)  # = τ 100 / sigma 25
    trades = pf.get_trade_log()
    crash_trades = trades[trades['timestamp'] == crash_ts]
    assert list(crash_trades['position_after']) == [0.0, 4.0], (
        "crash bar must liquidate to flat then re-establish AT TARGET, "
        f"got {list(crash_trades['position_after'])}"
    )
    assert (trades['position_after'] >= 0.0).all(), (
        "a long-forecast book must never go short"
    )

    # Diagnostic decomposition on the crash-bar RM record.
    row = rm.get_records(SYM).loc[crash_ts]
    assert row['current_qty'] == pytest.approx(10.0)
    assert row['pending_mkt_order_quantity'] == pytest.approx(-10.0)
    assert row['trade_qty'] == pytest.approx(4.0)
    assert bool(row['submitted']) is True


def test_deep_insolvency_voids_cancelled_resize_fill(caplog):
    """Deep crash: even after liquidating, the account is under water, so
    the second solvency check cancels the RM's same-bar resize order —
    whose fill is already in the events queue. That fill must be VOIDED
    (position stays flat), not booked."""
    closes = [100.0] * 40 + [5.0] * 3
    with caplog.at_level(logging.WARNING):
        pf, _, idx = _run_engine(
            closes, capital=510.0,
            vol=StubVol(base=10.0, spiked=25.0, crash_bar=41),
        )
    assert pf.positions[SYM] == pytest.approx(0.0)
    trades = pf.get_trade_log()
    # Entry + liquidation only — the cancelled BUY 4 never books.
    assert len(trades) == 2, trades
    assert not (trades['quantity'] == 4.0).any()
    assert any('FILL VOIDED' in r.message for r in caplog.records)
