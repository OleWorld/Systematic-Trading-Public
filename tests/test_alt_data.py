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
