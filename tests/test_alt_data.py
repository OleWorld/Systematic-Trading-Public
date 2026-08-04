"""
Unit + integration tests for alternative-data (alt feed) ingestion.

Covers, smallest first:

  Section 1 — `AltRecord` storage type (immutability, copy semantics).
  Section 2 — `DataHandler` base: alt deques, `_append_alt` gates, accessors.
  Section 3 — `HistoricDataHandler` construction gates for `alt_data`.
  Section 4 — stream merge + `update_bar` ingestion (the causality contract).
  Section 5 — end-to-end integration through a real `Backtester`.

Run from the repo root:  pytest tests/test_alt_data.py -v
"""

import dataclasses
import datetime
import queue as thread_queue
from typing import Any, List

import pandas as pd
import pytest

from data import AltRecord
from data._base import DataHandler
from data._historic import HistoricDataHandler

UTC = datetime.timezone.utc
TS = datetime.datetime(2026, 1, 1, tzinfo=UTC)


# ──────────────────────────────────────────────
# Test doubles & helpers
# ──────────────────────────────────────────────

class FakeQueue:
    """Captures every put() for inspection."""

    def __init__(self):
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


# ──────────────────────────────────────────────
# Section 1 — AltRecord storage type
# ──────────────────────────────────────────────

def test_altrecord_fields_are_frozen():
    rec = AltRecord(TS, {'rate': 0.01})
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.timestamp = TS + datetime.timedelta(days=1)


def test_altrecord_values_mapping_is_read_only():
    rec = AltRecord(TS, {'rate': 0.01})
    with pytest.raises(TypeError):
        rec.values['rate'] = 99.0


def test_altrecord_copies_the_caller_dict():
    d = {'rate': 0.01}
    rec = AltRecord(TS, d)
    d['rate'] = 99.0
    assert rec.values['rate'] == 0.01


def test_altrecord_value_access_and_equality():
    a = AltRecord(TS, {'rate': 0.01, 'oi': 5.0})
    b = AltRecord(TS, {'rate': 0.01, 'oi': 5.0})
    assert a.values['oi'] == 5.0
    assert a == b


# ──────────────────────────────────────────────
# Section 2 — DataHandler base: deques, _append_alt gates, accessors
# ──────────────────────────────────────────────

class _StubHandler(DataHandler):
    """Concrete DataHandler with a no-op update_bar so we can drive
    _append_alt directly."""

    def update_bar(self) -> None:
        return None


def _handler(alt_feeds=None, symbols=('BTC', 'ETH')):
    """Build a _StubHandler with a 'funding' feed by default."""
    feeds = {'funding': 500} if alt_feeds is None else alt_feeds
    return _StubHandler(FakeQueue(), list(symbols), '1d', {'1d': 500},
                        alt_feeds=feeds)


def _ts(day, hour=0):
    return datetime.datetime(2026, 1, day, hour, tzinfo=UTC)


def test_append_and_read_roundtrip_oldest_to_newest():
    dh = _handler()
    assert dh._append_alt('funding', 'BTC', _ts(1), {'rate': 0.01}) is True
    assert dh._append_alt('funding', 'BTC', _ts(2), {'rate': -0.02}) is True
    records = dh.get_latest_alt('BTC', 'funding', 5)
    assert [r.timestamp for r in records] == [_ts(1), _ts(2)]
    assert records[-1].values['rate'] == -0.02
    assert dh.count_alt('BTC', 'funding') == 2


def test_get_latest_alt_respects_n_and_nonpositive_n():
    dh = _handler()
    for day in (1, 2, 3):
        dh._append_alt('funding', 'BTC', _ts(day), {'rate': float(day)})
    assert [r.values['rate'] for r in dh.get_latest_alt('BTC', 'funding', 2)] == [2.0, 3.0]
    assert dh.get_latest_alt('BTC', 'funding', 0) == []
    assert dh.get_latest_alt('BTC', 'funding', -1) == []


def test_empty_window_returns_empty_not_error():
    dh = _handler()
    assert dh.get_latest_alt('ETH', 'funding', 5) == []
    assert dh.count_alt('ETH', 'funding') == 0
    assert dh.get_latest_alt_df('ETH', 'funding', 5).empty


def test_nan_field_drops_whole_record():
    dh = _handler()
    ok = dh._append_alt('funding', 'BTC', _ts(1), {'rate': float('nan'), 'oi': 5.0})
    assert ok is False
    assert dh.count_alt('BTC', 'funding') == 0


def test_duplicate_timestamp_first_wins():
    dh = _handler()
    dh._append_alt('funding', 'BTC', _ts(1), {'rate': 1.0})
    ok = dh._append_alt('funding', 'BTC', _ts(1), {'rate': 2.0})
    assert ok is False
    assert dh.get_latest_alt('BTC', 'funding', 1)[-1].values['rate'] == 1.0


def test_backward_timestamp_raises():
    dh = _handler()
    dh._append_alt('funding', 'BTC', _ts(2), {'rate': 1.0})
    with pytest.raises(ValueError, match="[Oo]ut-of-order"):
        dh._append_alt('funding', 'BTC', _ts(1), {'rate': 2.0})


def test_per_symbol_windows_are_independent():
    dh = _handler()
    dh._append_alt('funding', 'BTC', _ts(1), {'rate': 1.0})
    assert dh.count_alt('BTC', 'funding') == 1
    assert dh.count_alt('ETH', 'funding') == 0


def test_maxlen_evicts_oldest():
    dh = _handler(alt_feeds={'funding': 2})
    for day in (1, 2, 3):
        dh._append_alt('funding', 'BTC', _ts(day), {'rate': float(day)})
    records = dh.get_latest_alt('BTC', 'funding', 10)
    assert [r.values['rate'] for r in records] == [2.0, 3.0]


def test_unknown_feed_and_symbol_raise():
    dh = _handler()
    with pytest.raises(ValueError, match="Unknown alt feed"):
        dh.get_latest_alt('BTC', 'open_interest', 1)
    with pytest.raises(ValueError, match="Unknown symbol"):
        dh.get_latest_alt('DOGE', 'funding', 1)
    with pytest.raises(ValueError, match="Unknown"):
        dh._append_alt('open_interest', 'BTC', _ts(1), {'x': 1.0})


def test_get_latest_alt_df_parity_with_records():
    dh = _handler()
    dh._append_alt('funding', 'BTC', _ts(1), {'rate': 0.01, 'oi': 5.0})
    dh._append_alt('funding', 'BTC', _ts(2), {'rate': 0.02, 'oi': 6.0})
    df = dh.get_latest_alt_df('BTC', 'funding', 5)
    assert list(df.columns) == ['rate', 'oi']
    assert list(df.index) == [_ts(1), _ts(2)]
    assert df['rate'].tolist() == [0.01, 0.02]


def test_invalid_alt_feeds_config_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        _handler(alt_feeds={'': 500})
    with pytest.raises(ValueError, match="positive int"):
        _handler(alt_feeds={'funding': 0})
    with pytest.raises(ValueError, match="positive int"):
        _handler(alt_feeds={'funding': 'big'})


def test_no_alt_feeds_is_backward_compatible():
    dh = _StubHandler(FakeQueue(), ['BTC'], '1d', {'1d': 500})
    assert dh.alt_feeds == {}
    with pytest.raises(ValueError, match="Unknown alt feed"):
        dh.count_alt('BTC', 'funding')


# ──────────────────────────────────────────────
# Section 3 — HistoricDataHandler construction gates for alt_data
# ──────────────────────────────────────────────

def _make_ohlcv(closes, *, start='2026-01-01', freq='1D', tz='UTC') -> pd.DataFrame:
    """Small OHLCV frame with O=H=L=C=close[i] and Volume=1.0."""
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz=tz)
    return pd.DataFrame({
        'Open':   list(closes),
        'High':   list(closes),
        'Low':    list(closes),
        'Close':  list(closes),
        'Volume': [1.0] * len(closes),
    }, index=idx)


def _make_alt(values, times, *, column='rate', tz='UTC') -> pd.DataFrame:
    """Single-column alt frame at explicit timestamps."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(times)), tz=tz)
    return pd.DataFrame({column: list(values)}, index=idx)


def _historic(alt_data=None, alt_maxlen=None, symbols=('BTC', 'ETH'),
              n_bars=3):
    """HistoricDataHandler over daily bars for every symbol."""
    bars = {s: _make_ohlcv([100.0 + i for i in range(n_bars)])
            for s in symbols}
    return HistoricDataHandler(
        FakeQueue(), list(symbols), '1d', {'1d': 500},
        data=bars, alt_data=alt_data, alt_maxlen=alt_maxlen,
    )


def test_alt_data_registers_feeds_with_default_maxlen():
    dh = _historic(alt_data={'funding': {'BTC': _make_alt([0.01], ['2026-01-01'])}})
    assert dh.alt_feeds == {'funding': 500}
    assert dh.count_alt('BTC', 'funding') == 0   # nothing streamed yet


def test_alt_maxlen_override_and_typo_guard():
    alt = {'funding': {'BTC': _make_alt([0.01], ['2026-01-01'])}}
    dh = _historic(alt_data=alt, alt_maxlen={'funding': 42})
    assert dh.alt_feeds == {'funding': 42}
    with pytest.raises(ValueError, match="absent from alt_data"):
        _historic(alt_data=alt, alt_maxlen={'fundng': 42})


def test_alt_symbols_must_be_subset_of_symbol_list():
    with pytest.raises(ValueError, match="unregistered symbol"):
        _historic(alt_data={'funding': {'DOGE': _make_alt([0.01], ['2026-01-01'])}})


def test_naive_alt_index_raises():
    with pytest.raises(ValueError):
        _historic(alt_data={'funding': {'BTC': _make_alt([0.01], ['2026-01-01'], tz=None)}})


def test_unsorted_alt_index_raises():
    df = _make_alt([1.0, 2.0], ['2026-01-02', '2026-01-01'])
    with pytest.raises(ValueError, match="not sorted ascending"):
        _historic(alt_data={'funding': {'BTC': df}})


def test_non_numeric_alt_column_raises():
    df = _make_alt(['high', 'low'], ['2026-01-01', '2026-01-02'])
    with pytest.raises(ValueError, match="non-numeric"):
        _historic(alt_data={'funding': {'BTC': df}})


def test_duplicate_alt_columns_raise():
    idx = pd.DatetimeIndex(pd.to_datetime(['2026-01-01']), tz='UTC')
    df = pd.DataFrame([[1.0, 2.0]], columns=['rate', 'rate'], index=idx)
    with pytest.raises(ValueError, match="duplicate column"):
        _historic(alt_data={'funding': {'BTC': df}})


def test_empty_alt_frames_are_skipped():
    dh = _historic(alt_data={'funding': {'BTC': pd.DataFrame()}})
    assert dh.alt_feeds == {'funding': 500}
    assert dh.count_alt('BTC', 'funding') == 0


def test_no_alt_data_is_backward_compatible():
    dh = _historic()
    assert dh.alt_feeds == {}


# ──────────────────────────────────────────────
# Section 4 — stream merge + update_bar ingestion (causality contract)
# ──────────────────────────────────────────────

def _drive_one_bar(dh):
    """Call update_bar once; return the BarEvents it queued (0 or 1)."""
    before = len(dh.events_queue.items)
    dh.update_bar()
    return dh.events_queue.items[before:]


def test_visibility_contract_ts_leq_T():
    """The bar at open-time T sees exactly the alt records with ts <= T."""
    bars = {'BTC': _make_ohlcv([100.0, 101.0, 102.0])}   # 01-01, 01-02, 01-03
    funding = _make_alt(
        [0.001, 0.002, 0.003, 0.004],
        ['2025-12-31 12:00',   # before first bar  -> visible at bar 01-01
         '2026-01-01 00:00',   # == bar-open T     -> visible at bar 01-01 (tie-break)
         '2026-01-01 08:00',   # inside (T, T+1d)  -> visible at bar 01-02
         '2026-01-03 12:00'])  # after last bar    -> ingested at stream end
    dh = HistoricDataHandler(FakeQueue(), ['BTC'], '1d', {'1d': 500},
                             data=bars, alt_data={'funding': {'BTC': funding}})

    events = _drive_one_bar(dh)            # bar 2026-01-01
    assert len(events) == 1
    assert dh.count_alt('BTC', 'funding') == 2
    assert dh.get_latest_alt('BTC', 'funding', 1)[-1].values['rate'] == 0.002

    events = _drive_one_bar(dh)            # bar 2026-01-02
    assert len(events) == 1
    assert dh.count_alt('BTC', 'funding') == 3

    events = _drive_one_bar(dh)            # bar 2026-01-03
    assert len(events) == 1
    assert dh.count_alt('BTC', 'funding') == 3

    events = _drive_one_bar(dh)            # stream end: trailing alt row
    assert events == []
    assert dh.continue_backtest is False
    assert dh.count_alt('BTC', 'funding') == 4


def test_bar_stream_identical_with_and_without_alt_data():
    """Alt data must not change which BarEvents are emitted, or their order."""
    bars = {'BTC': _make_ohlcv([100.0, 101.0]),
            'ETH': _make_ohlcv([200.0, 201.0])}
    funding = _make_alt([0.001, 0.002],
                        ['2026-01-01 08:00', '2026-01-01 16:00'])

    def _all_events(alt_data):
        dh = HistoricDataHandler(FakeQueue(), ['BTC', 'ETH'], '1d',
                                 {'1d': 500},
                                 data={k: v.copy() for k, v in bars.items()},
                                 alt_data=alt_data)
        while dh.continue_backtest:
            dh.update_bar()
        return [(e.symbol, e.timestamp, e.close) for e in dh.events_queue.items]

    assert _all_events(None) == _all_events({'funding': {'BTC': funding}})


def test_multi_symbol_feeds_stay_separate_through_the_stream():
    bars = {'BTC': _make_ohlcv([100.0, 101.0]),
            'ETH': _make_ohlcv([200.0, 201.0])}
    alt = {'funding': {
        'BTC': _make_alt([0.001], ['2026-01-01 00:00']),
        'ETH': _make_alt([0.009], ['2026-01-01 00:00']),
    }}
    dh = HistoricDataHandler(FakeQueue(), ['BTC', 'ETH'], '1d', {'1d': 500},
                             data=bars, alt_data=alt)
    while dh.continue_backtest:
        dh.update_bar()
    assert dh.get_latest_alt('BTC', 'funding', 1)[-1].values['rate'] == 0.001
    assert dh.get_latest_alt('ETH', 'funding', 1)[-1].values['rate'] == 0.009


def test_multiple_feeds_ingest_independently():
    bars = {'BTC': _make_ohlcv([100.0, 101.0])}
    alt = {
        'funding': {'BTC': _make_alt([0.001], ['2026-01-01 00:00'])},
        'open_interest': {'BTC': _make_alt([5e9], ['2026-01-01 12:00'],
                                           column='oi')},
    }
    dh = HistoricDataHandler(FakeQueue(), ['BTC'], '1d', {'1d': 500},
                             data=bars, alt_data=alt)
    _drive_one_bar(dh)                      # bar 01-01: funding print visible
    assert dh.count_alt('BTC', 'funding') == 1
    assert dh.count_alt('BTC', 'open_interest') == 0
    _drive_one_bar(dh)                      # bar 01-02: OI print now visible
    assert dh.count_alt('BTC', 'open_interest') == 1
    assert dh.get_latest_alt('BTC', 'open_interest', 1)[-1].values['oi'] == 5e9


def test_broadcast_same_frame_under_multiple_symbols():
    """One shared frame object wired under several symbols (spec: refinery
    utilization for CL/RB/HO) ingests independently per symbol."""
    bars = {'BTC': _make_ohlcv([100.0, 101.0]),
            'ETH': _make_ohlcv([200.0, 201.0])}
    shared = _make_alt([0.5], ['2026-01-01 00:00'], column='util')
    dh = HistoricDataHandler(FakeQueue(), ['BTC', 'ETH'], '1d', {'1d': 500},
                             data=bars,
                             alt_data={'util': {s: shared for s in ('BTC', 'ETH')}})
    while dh.continue_backtest:
        dh.update_bar()
    assert dh.count_alt('BTC', 'util') == 1
    assert dh.count_alt('ETH', 'util') == 1


def test_nan_alt_row_dropped_in_stream():
    bars = {'BTC': _make_ohlcv([100.0, 101.0])}
    funding = _make_alt([float('nan'), 0.002],
                        ['2025-12-31 12:00', '2026-01-01 12:00'])
    dh = HistoricDataHandler(FakeQueue(), ['BTC'], '1d', {'1d': 500},
                             data=bars, alt_data={'funding': {'BTC': funding}})
    while dh.continue_backtest:
        dh.update_bar()
    records = dh.get_latest_alt('BTC', 'funding', 10)
    assert [r.values['rate'] for r in records] == [0.002]


def test_multi_column_stream_decode_keys_values_to_right_fields():
    """The positional itertuples decode must key each value to the RIGHT
    field name after streaming — including column names that are not
    valid Python identifiers (pandas renames those in itertuples; the
    decode zips positionally against the recorded column order, so the
    renaming must be irrelevant)."""
    bars = {'BTC': _make_ohlcv([100.0, 101.0])}
    idx = pd.DatetimeIndex(pd.to_datetime(['2026-01-01 00:00']), tz='UTC')
    frame = pd.DataFrame(
        {'funding rate': [0.001], '2oi': [5e9], 'basis': [-0.5]}, index=idx)
    dh = HistoricDataHandler(FakeQueue(), ['BTC'], '1d', {'1d': 500},
                             data=bars, alt_data={'mixed': {'BTC': frame}})
    while dh.continue_backtest:
        dh.update_bar()
    rec = dh.get_latest_alt('BTC', 'mixed', 1)[-1]
    assert rec.values['funding rate'] == 0.001
    assert rec.values['2oi'] == 5e9
    assert rec.values['basis'] == -0.5
    df = dh.get_latest_alt_df('BTC', 'mixed', 1)
    assert list(df.columns) == ['funding rate', '2oi', 'basis']


# ──────────────────────────────────────────────
# Section 5 — end-to-end integration through a real Backtester
# ──────────────────────────────────────────────

from backtester import Backtester
from config import uniform_registry
from correlation import CorrelationManager
from execution import BacktestExecution
from portfolio import BacktestPortfolio
from riskmanager import SimpleRiskManager
from strategy import Strategy
from universe import UniverseManager


class _FundingToyStrategy(Strategy):
    """Toy alt consumer: forecast = -sign(latest funding rate) * 50.

    Returns ``None`` (no forecast — warmup) until the symbol has at
    least one funding record, exercising the existing liveness gate for
    feed-coverage gaps.
    """

    def calculate_forecast(self, event):
        if self.data_handler.count_alt(event.symbol, 'funding') == 0:
            return None
        rec = self.data_handler.get_latest_alt(event.symbol, 'funding', 1)[-1]
        rate = rec.values['rate']
        if rate > 0:
            forecast = -50.0
        elif rate < 0:
            forecast = 50.0
        else:
            forecast = 0.0
        return {'forecast': forecast, 'funding_rate': rate}


def _run_end_to_end():
    """Five daily bars for BTC+ETH; funding only for BTC, flipping sign.

    Funding +0.01 known at the 01-01 bar (short from day 1) and -0.01
    known at the 01-03 bar (long from day 3). ETH has no funding data
    and must never warm up or trade.
    """
    q = thread_queue.Queue()
    symbols = ['BTC', 'ETH']
    bars = {s: _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])
            for s in symbols}
    funding = _make_alt([0.01, -0.01],
                        ['2026-01-01 00:00', '2026-01-03 00:00'])
    dh = HistoricDataHandler(q, symbols, '1d', {'1d': 500},
                             data=bars,
                             alt_data={'funding': {'BTC': funding}})
    strat = _FundingToyStrategy(dh, symbols)
    instruments = uniform_registry(symbols)
    portfolio = BacktestPortfolio(q, dh, symbols, instruments,
                                  initial_capital=1_000_000.0)
    rm = SimpleRiskManager(portfolio, strat, size_mode='fixed_quantity',
                           position_size=1.0)
    execution = BacktestExecution(q, instruments)
    # SimpleRiskManager doesn't consume universe/correlation state (it
    # inherits RiskManager's no-op on_universe_event/on_correlation_event),
    # so these two REQUIRED Backtester args are just structurally wired
    # here — lookback=32 is the smallest value CorrelationManager accepts,
    # and min_history_bars must be >= it (the drift guard).
    universe_manager = UniverseManager(strat, dh, min_history_bars=32,
                                       history_timeframe='1d')
    correlation_manager = CorrelationManager(dh, universe_manager,
                                             lookback=32, step_size=0,
                                             timeframe='1d')
    bt = Backtester(events_queue=q, data_handler=dh, strategy=strat,
                    portfolio=portfolio, risk_manager=rm,
                    execution_handler=execution,
                    universe_manager=universe_manager,
                    correlation_manager=correlation_manager)
    bt.run()
    return strat, portfolio


def test_forecast_responds_to_funding_and_position_flips():
    strat, portfolio = _run_end_to_end()
    records = strat.get_records('BTC')
    # Latest-known funding at each bar: +0.01 on 01-01/01-02, -0.01 after.
    assert records['funding_rate'].tolist() == [0.01, 0.01, -0.01, -0.01, -0.01]
    assert records['forecast'].tolist() == [-50.0, -50.0, 50.0, 50.0, 50.0]
    # Sign-of-forecast sizing at 1 contract: short first, long by the end.
    assert portfolio.positions['BTC'] == 1.0


def test_uncovered_symbol_never_warms_and_never_trades():
    strat, portfolio = _run_end_to_end()
    assert strat.is_warmed_up('BTC') is True
    assert strat.is_warmed_up('ETH') is False
    assert strat.get_forecast('ETH') is None
    assert portfolio.positions['ETH'] == 0.0
    trade_log = portfolio.get_trade_log()
    assert not trade_log.empty            # BTC did trade
    assert 'ETH' not in set(trade_log['symbol'])
