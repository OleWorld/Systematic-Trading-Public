"""Unit tests for universe.UniverseManager (gates, marks, log, events)."""

from typing import Dict, List, Optional

import pandas as pd
import pytest

from event import BarEvent
from universe import UniverseManager, UniverseStatus

T0 = pd.Timestamp('2024-01-01', tz='UTC')


class StubStrategy:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)
        self._warm = set()

    def is_warmed_up(self, symbol):
        return symbol in self._warm


class StubDataHandler:
    def __init__(self, symbols, maxlen=500, timeframe='1d'):
        self.timeframes: Dict[str, int] = {timeframe: maxlen}
        self.counts: Dict[str, int] = {s: 0 for s in symbols}

    def count_bars(self, symbol, timeframe=None):
        return self.counts.get(symbol, 0)


def _bar(symbol, ts=T0, close=100.0):
    return BarEvent(symbol=symbol, timestamp=ts, open=close, high=close,
                    low=close, close=close, volume=1.0, period='1d')


def _build(symbols=('A', 'B'), min_history=3):
    strat = StubStrategy(symbols)
    dh = StubDataHandler(symbols)
    um = UniverseManager(strat, dh, min_history_bars=min_history)
    return um, strat, dh


class TestConstruction:
    def test_all_symbols_start_not_live_with_both_warmup_reasons(self):
        um, _, _ = _build()
        st = um.status('A')
        assert st.live is False and st.excluded is False
        assert st.reasons == ['warmup_forecast', 'warmup_history']

    def test_initial_evaluation_logged_but_no_events(self):
        um, _, _ = _build()
        log = um.get_transition_log()
        assert len(log) == 2 and set(log['trigger']) == {'initial'}
        assert um.drain_events() == []

    def test_unregistered_timeframe_raises(self):
        strat = StubStrategy(['A'])
        with pytest.raises(ValueError):
            UniverseManager(strat, StubDataHandler(['A']),
                            min_history_bars=3, history_timeframe='4h')

    def test_min_history_bounds(self):
        strat = StubStrategy(['A'])
        with pytest.raises(ValueError):
            UniverseManager(strat, StubDataHandler(['A']), min_history_bars=0)
        with pytest.raises(ValueError):
            UniverseManager(strat, StubDataHandler(['A'], maxlen=10),
                            min_history_bars=11)


class TestGates:
    def test_symbol_goes_live_when_both_gates_pass(self):
        um, strat, dh = _build(min_history=3)
        strat._warm.add('A')
        dh.counts['A'] = 3
        um.update_bar(_bar('A'))
        assert um.status('A').live is True
        assert um.status('A').reasons == []
        assert um.get_live_symbols() == ['A']

    def test_one_gate_alone_is_not_enough(self):
        um, strat, dh = _build(min_history=3)
        strat._warm.add('A')                       # strategy gate only
        um.update_bar(_bar('A'))
        assert um.status('A').reasons == ['warmup_history']
        assert um.status('A').live is False

    def test_update_bar_refreshes_event_symbol_only(self):
        um, strat, dh = _build(min_history=1)
        strat._warm.update({'A', 'B'})
        dh.counts['A'] = dh.counts['B'] = 1
        um.update_bar(_bar('A'))
        assert um.status('A').live is True
        # B not yet refreshed by a bar — still shows its initial state:
        assert um.status('B').live is False

    def test_status_returns_defensive_copy(self):
        um, _, _ = _build()
        st = um.status('A')
        st.reasons.append('poison')
        assert 'poison' not in um.status('A').reasons

    def test_status_unknown_symbol_raises(self):
        um, _, _ = _build()
        with pytest.raises(ValueError):
            um.status('ZZZ')


class TestTransitionLog:
    def test_go_live_appends_one_row_with_bar_refresh_trigger(self):
        um, strat, dh = _build(min_history=1)
        strat._warm.add('A')
        dh.counts['A'] = 1
        um.update_bar(_bar('A'))
        log = um.get_transition_log()
        rows = log[(log['symbol'] == 'A') & (log['trigger'] == 'bar_refresh')]
        assert len(rows) == 1
        assert bool(rows.iloc[0]['live']) is True
        assert rows.iloc[0]['reasons'] == ''
        assert rows.iloc[0]['timestamp'] == T0

    def test_no_row_when_nothing_changed(self):
        um, _, _ = _build()
        um.update_bar(_bar('A'))       # still both warmup reasons — no change
        log = um.get_transition_log()
        assert len(log[log['trigger'] == 'bar_refresh']) == 0
