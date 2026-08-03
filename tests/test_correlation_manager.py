"""Unit tests for correlation.CorrelationManager (cadence + refresh)."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from correlation import CorrelationManager
from event import BarEvent
from universe import UniverseManager

T = lambda d: pd.Timestamp(d, tz='UTC')


class StubStrategy:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)
        self._warm = set(symbols)          # strategy gate passes by default

    def is_warmed_up(self, symbol):
        return symbol in self._warm


class StubDataHandler:
    """Serves per-symbol close Series; count_bars mirrors their length."""

    def __init__(self, symbols, maxlen=500, timeframe='1d'):
        self.timeframes: Dict[str, int] = {timeframe: maxlen}
        self.closes: Dict[str, List[float]] = {s: [] for s in symbols}

    def count_bars(self, symbol, timeframe=None):
        return len(self.closes.get(symbol, []))

    def get_latest_bars_df(self, symbol, n=1, timeframe=None):
        vals = self.closes[symbol][-n:]
        idx = pd.date_range('2024-01-01', periods=len(vals), freq='D', tz='UTC')
        return pd.DataFrame({'Close': vals}, index=idx)


def _bar(symbol, ts, close=100.0, forming=False):
    return BarEvent(symbol=symbol, timestamp=ts, open=close, high=close,
                    low=close, close=close, volume=1.0, period='1d',
                    is_forming=forming)


def _build(symbols=('A', 'B'), lookback=32, step_size=2, maxlen=500):
    strat = StubStrategy(symbols)
    dh = StubDataHandler(symbols, maxlen=maxlen)
    um = UniverseManager(strat, dh, min_history_bars=lookback)
    cm = CorrelationManager(dh, um, lookback=lookback, step_size=step_size)
    return cm, um, dh, strat


def _feed_closes(dh, symbol, values):
    dh.closes[symbol] = list(values)


class TestConstruction:
    def test_drift_guard_timeframe_mismatch_raises(self):
        strat = StubStrategy(['A'])
        dh = StubDataHandler(['A'])
        dh.timeframes['4h'] = 500
        um = UniverseManager(strat, dh, min_history_bars=64,
                             history_timeframe='4h')
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=32, timeframe='1d')

    def test_drift_guard_lookback_exceeds_universe_gate_raises(self):
        strat = StubStrategy(['A'])
        dh = StubDataHandler(['A'])
        um = UniverseManager(strat, dh, min_history_bars=32)
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=64)

    def test_param_validation(self):
        strat = StubStrategy(['A'])
        dh = StubDataHandler(['A'])
        um = UniverseManager(strat, dh, min_history_bars=64)
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=31)          # < 32
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=64, step_size=-1)
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=64, mode='bogus')
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=64, floor=1.5)
        with pytest.raises(ValueError):
            CorrelationManager(dh, um, lookback=64, shrinkage='bogus')


class TestCadence:
    def test_refresh_fires_after_step_size_periods(self):
        cm, um, dh, _ = _build(step_size=2)
        assert cm.update_bar(_bar('A', T('2024-01-01'))) is None   # period 1
        evt = cm.update_bar(_bar('A', T('2024-01-02')))            # period 2
        assert evt is not None and evt.reason == 'empty_universe'
        assert cm.last_refresh_timestamp == T('2024-01-02')

    def test_multi_symbol_same_timestamp_ticks_once(self):
        cm, um, dh, _ = _build(step_size=2)
        cm.update_bar(_bar('A', T('2024-01-01')))
        assert cm.update_bar(_bar('B', T('2024-01-01'))) is None   # same period
        assert cm.update_bar(_bar('A', T('2024-01-02'))) is not None

    def test_forming_bars_skipped(self):
        cm, um, dh, _ = _build(step_size=1)
        assert cm.update_bar(_bar('A', T('2024-01-01'), forming=True)) is None

    def test_step_size_zero_never_refreshes(self):
        cm, um, dh, _ = _build(step_size=0)
        for d in pd.date_range('2024-01-01', periods=5, freq='D'):
            assert cm.update_bar(_bar('A', pd.Timestamp(d, tz='UTC'))) is None

    def test_counter_resets_after_refresh(self):
        cm, um, dh, _ = _build(step_size=2)
        cm.update_bar(_bar('A', T('2024-01-01')))
        assert cm.update_bar(_bar('A', T('2024-01-02'))) is not None
        assert cm.update_bar(_bar('A', T('2024-01-03'))) is None    # 1 of 2
        assert cm.update_bar(_bar('A', T('2024-01-04'))) is not None


class TestDegenerateRefresh:
    def test_empty_universe(self):
        cm, um, dh, _ = _build(step_size=1)          # no closes -> no gates pass
        evt = cm.update_bar(_bar('A', T('2024-01-01')))
        assert evt.reason == 'empty_universe'
        assert evt.matrix is None and evt.live_symbols == []

    def test_singleton(self):
        cm, um, dh, strat = _build(symbols=('A', 'B'), lookback=32, step_size=1)
        _feed_closes(dh, 'A', np.linspace(100, 132, 32))   # A passes data gate
        evt = cm.update_bar(_bar('A', T('2024-01-01')))
        assert evt.reason == 'singleton'
        assert evt.matrix is None and evt.live_symbols == ['A']

    def test_refresh_reassesses_universe_first(self):
        cm, um, dh, strat = _build(symbols=('A', 'B'), lookback=32, step_size=1)
        _feed_closes(dh, 'A', np.linspace(100, 132, 32))
        assert um.get_live_symbols() == []            # stale until reassess
        cm.update_bar(_bar('A', T('2024-01-01')))
        assert um.get_live_symbols() == ['A']         # refresh ran reassess_all
