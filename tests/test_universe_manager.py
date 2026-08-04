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


def _go_live(um, strat, dh, symbol):
    """Arrange helper: push a symbol through both gates via one bar,
    then discard the resulting go-live event so callers start clean."""
    strat._warm.add(symbol)
    dh.counts[symbol] = um.min_history_bars
    um.update_bar(_bar(symbol))
    um.drain_events()          # discard the go-live event for arrange phases


class TestExclusionMarks:
    def test_mark_excluded_takes_symbol_not_live(self):
        um, strat, dh = _build(min_history=1)
        _go_live(um, strat, dh, 'A')
        um.mark_excluded('A', 'delisted', T0)
        st = um.status('A')
        assert st.live is False and st.excluded is True
        assert st.reasons == ['delisted']
        assert um.get_live_symbols() == []

    def test_clear_excluded_re_enters_once_gates_pass(self):
        um, strat, dh = _build(min_history=1)
        _go_live(um, strat, dh, 'A')
        um.mark_excluded('A', 'delisted', T0)
        um.clear_excluded('A', 'delisted', T0)
        assert um.status('A').live is True          # dynamic — no permanence

    def test_marks_are_idempotent_no_event_no_log_row(self):
        um, strat, dh = _build(min_history=1)
        _go_live(um, strat, dh, 'A')
        um.mark_excluded('A', 'delisted', T0)
        um.drain_events()
        n_rows = len(um.get_transition_log())
        um.mark_excluded('A', 'delisted', T0)       # re-mark: no-op
        um.clear_excluded('A', 'nonexistent', T0)   # absent mark: no-op
        assert um.drain_events() == []
        assert len(um.get_transition_log()) == n_rows

    def test_mark_unknown_symbol_raises(self):
        um, _, _ = _build()
        with pytest.raises(ValueError):
            um.mark_excluded('ZZZ', 'delisted', T0)

    def test_clear_cannot_force_past_gates(self):
        um, _, _ = _build(min_history=3)            # gates unmet
        um.mark_excluded('A', 'delisted', T0)
        um.clear_excluded('A', 'delisted', T0)
        st = um.status('A')
        assert st.live is False                      # warmup reasons remain
        assert st.reasons == ['warmup_forecast', 'warmup_history']

    def test_reason_required_non_empty(self):
        um, _, _ = _build()
        with pytest.raises(ValueError):
            um.mark_excluded('A', '', T0)


class TestUniverseEvents:
    def test_go_live_emits_event_with_edge_and_trigger(self):
        um, strat, dh = _build(min_history=1)
        strat._warm.add('A')
        dh.counts['A'] = 1
        um.update_bar(_bar('A'))
        events = um.drain_events()
        assert len(events) == 1
        evt = events[0]
        assert evt.symbol == 'A' and evt.live is True and evt.prev_live is False
        assert evt.prev_reasons == ['warmup_forecast', 'warmup_history']
        assert evt.reasons == [] and evt.trigger == 'bar_refresh'
        assert evt.timestamp == T0
        assert um.drain_events() == []               # drained

    def test_mark_and_clear_triggers(self):
        um, strat, dh = _build(min_history=1)
        _go_live(um, strat, dh, 'A')
        um.mark_excluded('A', 'constant_price', T0)
        um.clear_excluded('A', 'constant_price', T0)
        triggers = [e.trigger for e in um.drain_events()]
        assert triggers == ['mark_excluded:constant_price',
                            'clear_excluded:constant_price']

    def test_reassess_all_refreshes_every_symbol(self):
        um, strat, dh = _build(min_history=1)
        strat._warm.update({'A', 'B'})
        dh.counts['A'] = dh.counts['B'] = 1
        um.reassess_all()
        events = um.drain_events()
        assert {e.symbol for e in events} == {'A', 'B'}
        assert all(e.trigger == 'reassess' for e in events)
        assert um.get_live_symbols() == ['A', 'B']

    def test_mark_outside_bar_processing_stamps_last_bar_ts(self):
        um, strat, dh = _build(min_history=1)
        _go_live(um, strat, dh, 'A')                 # last bar ts = T0
        um.mark_excluded('A', 'manual')               # no timestamp passed
        evt = um.drain_events()[0]
        assert evt.timestamp == T0
