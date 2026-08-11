"""
Context-symbols regression tests: data-only symbols stream through the
engine for strategies to READ but are never traded.

Three layers, all in this file:
- RM guard units: both risk managers skip bars for symbols outside
  ``strategy.symbol_list`` — no records, no sigma update, no orders,
  and crucially no ``universe_manager.status()`` call (which raises
  for unknown symbols — the original crash).
- Backtester wiring check: exact symbol coverage
  (traded ∪ context == data_handler.symbol_list) enforced at
  construction, both directions ValueError.
- Engine-level end-to-end pin: a context symbol streams through the
  REAL 8-component graph, drives a traded symbol's forecast, and stays
  invisible to sizing/universe (this run crashed with ValueError at
  ``status()`` before the guard existed).

Run from the repo root:  pytest tests/test_context_symbols.py -v
"""

import queue as thread_queue
from datetime import datetime
from types import SimpleNamespace
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
from riskmanager import SimpleRiskManager, VolTargetingRiskManager
from strategy import TimeSeriesStrategy
from universe import UniverseManager
from volatility import VolEstimator


DEFAULT_TS = datetime(2026, 1, 1, 12, 0, 0)


def _bar(symbol: str, ts: Optional[datetime] = None,
         close: float = 100.0) -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=ts if ts is not None else DEFAULT_TS,
        open=close, high=close, low=close, close=close, volume=1.0,
        period='1d', is_forming=False,
    )


# ──────────────────────────────────────────────
# RM guard doubles
# ──────────────────────────────────────────────

class _RecordingVol(VolEstimator):
    """Records which symbols' bars reached ``update``; constant sigma."""

    def __init__(self):
        self.updated = []

    def update(self, event: BarEvent) -> None:
        self.updated.append(event.symbol)

    def get_annual_vol(self, symbol: str) -> Optional[float]:
        return 10.0


class _NeverStatusUniverse:
    """status() must never fire for a context bar — that call is the
    pre-guard crash site (ValueError for unknown symbols)."""

    def status(self, symbol):
        raise AssertionError(
            f"universe.status({symbol!r}) called for a context bar"
        )


class _DeadUniverse:
    """Always not-live — lets a traded bar flow through the guard and
    exit via the universal not-live rule (records a row, no order)."""

    def status(self, symbol):
        return SimpleNamespace(live=False, excluded=False,
                               reasons=['warmup_history'])


class _StubPortfolio:
    def __init__(self):
        self.positions = {}
        self.submitted = []

    def projected_position(self, symbol, exclude_order_id=None):
        return 0.0

    def calculate_balance(self):
        return 100_000.0

    def get_price(self, symbol):
        return 100.0

    def submit_order(self, **kwargs):
        self.submitted.append(kwargs)


class _StubStrategy:
    def __init__(self, symbols, context=()):
        self.symbol_list = list(symbols)
        self.context_symbols = list(context)

    def get_forecast(self, symbol):
        return 50.0

    def is_warmed_up(self, symbol):
        return True


# ──────────────────────────────────────────────
# RM guard units
# ──────────────────────────────────────────────

def test_vol_targeting_rm_skips_context_bar():
    """A completed bar for a non-traded symbol exits before the vol
    estimator, before status(), and records nothing."""
    vol = _RecordingVol()
    pf = _StubPortfolio()
    rm = VolTargetingRiskManager(
        pf, _StubStrategy(['X'], context=['CTX']), vol,
        _NeverStatusUniverse(),
        annual_target_vol=100.0, vol_target_mode='dollar_volatility',
        position_buffer=0.25, instrument_weight_mode='equal_weight',
    )
    rm.update_bar(_bar('CTX'))
    assert vol.updated == []
    assert rm.get_records('CTX').empty
    assert pf.submitted == []


def test_vol_targeting_rm_still_processes_traded_bar():
    """Control: the guard must not block traded symbols — a traded bar
    reaches the vol estimator and records a diagnostic row."""
    vol = _RecordingVol()
    rm = VolTargetingRiskManager(
        _StubPortfolio(), _StubStrategy(['X'], context=['CTX']), vol,
        _DeadUniverse(),
        annual_target_vol=100.0, vol_target_mode='dollar_volatility',
        position_buffer=0.25, instrument_weight_mode='equal_weight',
    )
    rm.update_bar(_bar('X'))
    assert vol.updated == ['X']
    assert len(rm.get_records('X')) == 1


def test_simple_rm_skips_context_bar():
    """SimpleRiskManager previously recorded noise rows (and KeyError'd on
    instruments) for foreign symbols — now: nothing."""
    pf = _StubPortfolio()
    rm = SimpleRiskManager(pf, _StubStrategy(['X'], context=['CTX']),
                           size_mode='fixed_quantity', position_size=5.0)
    rm.update_bar(_bar('CTX'))
    assert rm.get_records('CTX').empty
    assert pf.submitted == []


# ──────────────────────────────────────────────
# Backtester wiring check
# ──────────────────────────────────────────────

class _WiringHandler:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)


def _wire(handler_syms, strategy):
    """Backtester with placeholder consumers — only __init__ runs."""
    return Backtester(thread_queue.Queue(), _WiringHandler(handler_syms),
                      strategy, object(), object(), object(),
                      object(), object())


def test_wiring_missing_traded_symbol_raises():
    with pytest.raises(ValueError, match=r"\['Y'\]"):
        _wire(['X'], _StubStrategy(['X', 'Y']))


def test_wiring_missing_context_dependency_raises():
    """The payoff of strategy-declared context: a forgotten flat-price
    series fails at construction, not as a forever-NaN forecast."""
    with pytest.raises(ValueError, match=r"\['C'\]"):
        _wire(['X'], _StubStrategy(['X'], context=['C']))


def test_wiring_stray_data_symbol_raises():
    with pytest.raises(ValueError, match=r"\['Z'\]"):
        _wire(['X', 'Z'], _StubStrategy(['X']))


def test_wiring_exact_coverage_with_context_passes():
    bt = _wire(['C', 'X'], _StubStrategy(['X'], context=['C']))
    assert bt is not None


def test_wiring_duck_typed_strategy_without_context_attr_passes():
    """Pre-context forecast sources (no context_symbols attribute) keep
    working — getattr defaults the declaration to empty."""
    bt = _wire(['X'], SimpleNamespace(symbol_list=['X']))
    assert bt is not None


class _WiringPortfolio:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)


def test_wiring_portfolio_missing_context_symbol_raises():
    """The portfolio must cover every streamed symbol — a context symbol
    absent from its list previously died as a mid-run KeyError."""
    with pytest.raises(ValueError, match=r"portfolio.*\['C'\]"):
        Backtester(thread_queue.Queue(), _WiringHandler(['C', 'X']),
                   _StubStrategy(['X'], context=['C']),
                   _WiringPortfolio(['X']), object(), object(),
                   object(), object())


def test_wiring_portfolio_without_symbol_list_passes():
    """Duck-typed portfolio doubles without symbol_list skip the check."""
    bt = Backtester(thread_queue.Queue(), _WiringHandler(['C', 'X']),
                    _StubStrategy(['X'], context=['C']),
                    object(), object(), object(), object(), object())
    assert bt is not None


def test_wiring_portfolio_covering_all_passes():
    bt = Backtester(thread_queue.Queue(), _WiringHandler(['C', 'X']),
                    _StubStrategy(['X'], context=['C']),
                    _WiringPortfolio(['C', 'X']), object(), object(),
                    object(), object())
    assert bt is not None


# ──────────────────────────────────────────────
# Engine-level end-to-end pin
# ──────────────────────────────────────────────

class ContextSignStrategy(TimeSeriesStrategy):
    """Forecast for the TRADED symbol from the CONTEXT symbol's direction:
    +50 if the context close rose, -50 if it fell (finalized pair — bars
    [-2]/[-3], robust to intra-timestamp symbol ordering), None during
    warmup. With a strictly falling context series the forecast is
    provably context-driven: a strategy ignoring context data could not
    deterministically emit -50."""

    CTX = 'C'

    def calculate_forecast(self, event: BarEvent):
        bars = self.data_handler.get_latest_bars(self.CTX, n=3,
                                                 timeframe='1d')
        if len(bars) < 3:
            return None
        prev, last = bars[-3].close, bars[-2].close
        if last == prev:
            return None
        return {'forecast': 50.0 if last > prev else -50.0}


def _ohlcv(closes, idx):
    return pd.DataFrame({'Open': closes, 'High': closes, 'Low': closes,
                         'Close': closes, 'Volume': [1.0] * len(closes)},
                        index=idx)


def _run_context_engine(n_bars: int = 45):
    """Real 8-component graph: traded 'X' (rising), context 'C' (strictly
    falling). Mirrors tests/test_order_lifecycle.py::_run_engine — 32-bar
    history gate, 5-bar correlation cadence, constant sigma via
    _RecordingVol, zero costs."""
    idx = pd.date_range('2024-01-01', periods=n_bars, freq='D', tz='UTC')
    x_closes = [100.0 + 0.5 * i for i in range(n_bars)]
    c_closes = [200.0 - 1.0 * i for i in range(n_bars)]
    events = thread_queue.Queue()
    instruments = uniform_registry(
        ['C', 'X'], point_value=1.0, fractional=True,
        slippage=SlippageModel('absolute', 0.0),
        commission=CommissionModel('per_contract', 0.0),
        margin=PortfolioMarginModel.from_leverage(
            10.0, maintenance_margin_rate=0.05,
        ),
    )
    dh = HistoricDataHandler(events, ['C', 'X'], '1d', {'1d': 100},
                             data={'X': _ohlcv(x_closes, idx),
                                   'C': _ohlcv(c_closes, idx)})
    strategy = ContextSignStrategy(dh, ['X'], context_symbols=['C'])
    portfolio = BacktestPortfolio(events, dh, ['C', 'X'], instruments,
                                  initial_capital=1_000_000.0)
    universe_manager = UniverseManager(strategy, dh, min_history_bars=32,
                                       history_timeframe='1d')
    correlation_manager = CorrelationManager(dh, universe_manager,
                                             lookback=32, step_size=5,
                                             timeframe='1d')
    rm = VolTargetingRiskManager(
        portfolio, strategy, _RecordingVol(), universe_manager,
        instruments=instruments, annual_target_vol=100.0,
        vol_target_mode='dollar_volatility', position_buffer=0.25,
        instrument_weight_mode='equal_weight',
    )
    execution = BacktestExecution(events, instruments)
    Backtester(events, dh, strategy, portfolio, rm, execution,
               universe_manager, correlation_manager).run()
    return portfolio, rm, strategy, universe_manager


def test_context_symbol_streams_end_to_end():
    """REGRESSION PIN: before the RM guard, this run died on 'C's first
    completed bar with ValueError from UniverseManager.status(). Now the
    context symbol drives the forecast and stays invisible to sizing."""
    portfolio, rm, strategy, um = _run_context_engine()
    # Forecast is context-driven: C falls every bar → -50.
    assert strategy.get_forecast('X') == -50.0
    # ...and sizing acted on it: short position in the traded symbol.
    assert portfolio.positions.get('X', 0.0) < 0
    # Context symbol is invisible to sizing and accounting:
    assert portfolio.positions.get('C', 0.0) == 0.0
    assert rm.get_records('C').empty
    # ...and to the universe:
    log = um.get_transition_log()
    assert 'C' not in set(log['symbol'])
    with pytest.raises(ValueError, match="not in"):
        um.status('C')
