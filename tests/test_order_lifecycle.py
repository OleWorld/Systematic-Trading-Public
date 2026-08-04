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
from correlation import CorrelationManager
from data import HistoricDataHandler
from event import BarEvent
from execution import BacktestExecution, CommissionModel, SlippageModel
from portfolio import BacktestPortfolio, PortfolioMarginModel
from riskmanager import VolTargetingRiskManager
from strategy import Strategy
from universe import UniverseManager
from volatility import VolEstimator


SYM = 'X'

# Universe/correlation cadence mirroring the old inline RM defaults this
# suite pinned (corr_lookback=32, corr_step_size=5, corr_timeframe='1d'):
# the symbol goes live once 32 bars have streamed, and the single-symbol
# universe earns a weight of 1.0 (the 'singleton' CorrelationEvent reason)
# at the next 5-bar cadence boundary on or after that.
_MIN_HISTORY_BARS = 32
_CORR_STEP_SIZE = 5


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
                annual_target_vol: float = 100.0,
                commission: float = 0.0):
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
        commission=CommissionModel('per_contract', commission),
        margin=PortfolioMarginModel.from_leverage(
            10.0, maintenance_margin_rate=0.05,
        ),
    )
    dh = HistoricDataHandler(events, [SYM], '1d', {'1d': 100}, data={SYM: df})
    strategy = ConstLongStrategy(dh, [SYM])
    portfolio = BacktestPortfolio(events, dh, [SYM], instruments,
                                  initial_capital=capital)
    universe_manager = UniverseManager(
        strategy, dh, min_history_bars=_MIN_HISTORY_BARS,
        history_timeframe='1d',
    )
    correlation_manager = CorrelationManager(
        dh, universe_manager, lookback=_MIN_HISTORY_BARS,
        step_size=_CORR_STEP_SIZE, timeframe='1d',
    )
    rm = VolTargetingRiskManager(
        portfolio, strategy, vol, universe_manager, instruments=instruments,
        annual_target_vol=annual_target_vol,
        vol_target_mode='dollar_volatility',
        position_buffer=0.25, instrument_weight_mode='equal_weight',
    )
    execution = BacktestExecution(events, instruments)
    Backtester(events, dh, strategy, portfolio, rm, execution,
              universe_manager, correlation_manager).run()
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
    call (liquidation SELL 10 in flight, deferred via ``fill_on_next_bar``)
    while the vol spike shrinks the target to 4. The RM must size against
    the PROJECTED position (0) and submit BUY 4 (the old realized-only
    diff submitted SELL 6 and ended the bar SHORT 6 with a +50 LONG
    forecast). The portfolio then scales the re-open to what the
    post-liquidation balance actually backs: balance 20 × leverage 10 /
    price 51 = 200/51 ≈ 3.92 — an account below its maintenance floor
    gets no free full-size re-open netted against the in-flight
    liquidation. Because the liquidation now defers to the symbol's next
    bar event (never retroactively against the already-streamed crash
    bar), the scaled re-open fills FIRST, on the crash bar's close,
    against the still-open pre-liquidation position (10 + 3.92 ≈ 13.92);
    the liquidation catches up one bar later, at that bar's open, landing
    the book at the same balance-backed size."""
    closes = [100.0] * 40 + [51.0] * 3
    pf, rm, idx = _run_engine(
        closes, capital=510.0,
        vol=StubVol(base=10.0, spiked=25.0, crash_bar=41),
    )
    crash_ts = idx[40]
    next_ts = idx[41]
    scaled_reopen = 510.0 - 490.0  # crash-bar balance …
    scaled_reopen = scaled_reopen * 10.0 / 51.0  # × leverage / price ≈ 3.92

    # End state: at the balance-backed size (< the post-spike target 4),
    # never wrong-sign.
    assert pf.positions[SYM] == pytest.approx(scaled_reopen)
    trades = pf.get_trade_log()
    crash_trades = trades[trades['timestamp'] == crash_ts]
    assert crash_trades['position_after'].tolist() == pytest.approx(
        [10.0 + scaled_reopen]
    ), (
        "crash bar: the scaled re-open fills against the still-open "
        "pre-liquidation position (the liquidation itself is now "
        f"deferred), got {list(crash_trades['position_after'])}"
    )
    next_trades = trades[trades['timestamp'] == next_ts]
    assert next_trades['position_after'].tolist() == pytest.approx(
        [scaled_reopen]
    ), (
        "the deferred liquidation must fill on the symbol's next bar "
        "event, at that bar's open, bringing the book down to the "
        f"balance-backed size, got {list(next_trades['position_after'])}"
    )
    assert (trades['position_after'] >= 0.0).all(), (
        "a long-forecast book must never go short"
    )

    # Diagnostic decomposition on the crash-bar RM record: the RM still
    # ASKS for the full diff to target (4); the portfolio scales it.
    row = rm.get_records(SYM).loc[crash_ts]
    assert row['current_qty'] == pytest.approx(10.0)
    assert row['pending_mkt_order_quantity'] == pytest.approx(-10.0)
    assert row['trade_qty'] == pytest.approx(4.0)
    assert bool(row['submitted']) is True


def test_deep_insolvency_rejects_same_bar_resize_reopen(caplog):
    """Deep crash: even after liquidating, the account is under water
    (balance −440), so the RM's same-bar resize re-open is REJECTED at
    submission — the margin budget for new exposure is zero, and an
    insolvent account must not rebuild its position by netting against
    the in-flight liquidation. (Straggler fills for orders cancelled by
    the margin-call pass are still voided — pinned at the portfolio
    level in test_portfolio.py.)"""
    closes = [100.0] * 40 + [5.0] * 3
    with caplog.at_level(logging.WARNING):
        pf, _, idx = _run_engine(
            closes, capital=510.0,
            vol=StubVol(base=10.0, spiked=25.0, crash_bar=41),
        )
    assert pf.positions[SYM] == pytest.approx(0.0)
    trades = pf.get_trade_log()
    # Entry + liquidation only — the BUY 4 re-open never reaches the book.
    assert len(trades) == 2, trades
    assert not (trades['quantity'] == 4.0).any()
    orders = pf.get_order_log()
    assert not (orders['direction'] == 'BUY').iloc[1:].any(), (
        "the rejected re-open must not appear in the order log"
    )
    assert any('ORDER REJECTED' in r.message for r in caplog.records)


def test_finalize_reconciles_final_equity_row_with_end_state():
    """Fills for the FINAL bar's orders book after that bar's equity row
    was appended; ``Backtester.run()`` must finalize the portfolio so the
    curve's last row (and the last timestamp's per-symbol snapshot)
    matches end-of-run state — otherwise the run's last commission and
    realized deltas silently vanish from every curve-derived stat."""
    # Constant price; a vol spike on the very last completed bar shrinks
    # the target 10 → 4, forcing a SELL 6 resize that fills on the final
    # bar's close ($1/contract commission makes the miss money-visible).
    closes = [100.0] * 45
    pf, _, idx = _run_engine(
        closes, capital=1_000_000.0,
        vol=StubVol(base=10.0, spiked=25.0, crash_bar=45),
        commission=1.0,
    )

    assert pf.positions[SYM] == pytest.approx(4.0)
    trades = pf.get_trade_log()
    assert trades['timestamp'].iloc[-1] == idx[-1]  # final-bar fill exists

    eq = pf.get_equity_curve()
    last = eq.iloc[-1]
    assert last['account_balance'] == pytest.approx(pf.calculate_balance())
    assert last['cash'] == pytest.approx(pf.cash)
    assert last['total_commission'] == pytest.approx(pf.total_commission)
    assert last['positions'][SYM] == pytest.approx(4.0)
    assert last['realized_pnl'][SYM] == pytest.approx(pf.realized_pnl[SYM])


def test_margin_call_bar_reopen_scaled_no_phantom_churn():
    """Crash bar with an UNCHANGED target (no vol spike): long 10, crash
    to 51 leaves balance 20 below the maintenance floor 25.5 while the
    Carver target stays 10. Regression for the netting free-pass: the
    same-bar re-open used to fill at the FULL previous size on ~zero
    equity, get margin-called again within the bar, and liquidate again —
    a phantom full-size round trip per margin-call bar. Now the re-open
    is scaled to the balance-backed 200/51 ≈ 3.92; because the liquidation
    defers to the symbol's next bar event (``fill_on_next_bar``), the
    scaled re-open fills FIRST on the crash bar (against the still-open
    pre-liquidation position, 10 + 3.92 ≈ 13.92) and the liquidation
    catches up one bar later, at that bar's open, landing the book at the
    same balance-backed size. The book ends solvent (maintenance 10 <
    balance 20) and no further liquidation or resize fills occur."""
    closes = [100.0] * 40 + [51.0] * 5
    pf, _, idx = _run_engine(closes, capital=510.0, vol=StubVol(base=10.0))
    crash_ts = idx[40]
    next_ts = idx[41]
    scaled_reopen = 20.0 * 10.0 / 51.0  # balance × leverage / price

    trades = pf.get_trade_log()
    # Exactly 3 fills in the whole run: entry, scaled re-open, liquidation.
    assert len(trades) == 3, trades
    crash_trades = trades[trades['timestamp'] == crash_ts]
    assert crash_trades['position_after'].tolist() == pytest.approx(
        [10.0 + scaled_reopen]
    )
    next_trades = trades[trades['timestamp'] == next_ts]
    assert next_trades['position_after'].tolist() == pytest.approx(
        [scaled_reopen]
    )
    # No churn after the liquidation catches up: position holds at the
    # scaled size (subsequent resize attempts are rejected — zero margin
    # headroom).
    assert not (trades['timestamp'] > next_ts).any()
    assert pf.positions[SYM] == pytest.approx(scaled_reopen)
    assert pf.calculate_balance() == pytest.approx(20.0)
