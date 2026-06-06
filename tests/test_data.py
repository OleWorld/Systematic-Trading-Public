"""
Unit tests for the `data` module.

Covers four layers, smallest first:

  Section 1 — `_timeframe.py` pure helpers (parsing, UTC, period alignment).
  Section 2 — `_ohlcv.py` DataFrame helpers (candle conversion, resampling).
  Section 3 — `DataHandler` deque mechanics + HTF aggregation gating.
  Section 4 — `HistoricDataHandler` in-memory data path (no ArcticDB).

No ArcticDB, no CCXT, no network. A `_StubHandler` subclass exercises the
abstract `DataHandler` directly; `HistoricDataHandler` is driven through its
`data={symbol: df}` constructor path.

Run from the repo root:  pytest tests/test_data.py -v
"""

import datetime
import math
import queue as thread_queue
from typing import Any, List

import pandas as pd
import pytest

from data._base import DataHandler
from data._historic import HistoricDataHandler
from data._ohlcv import _candles_to_dataframe, _resample_ohlcv
from data._timeframe import (
    TIMEFRAME_FALLBACK_ORDER,
    _ensure_utc,
    get_period_start,
    _ms_to_utc,
    _parse_date,
    parse_timeframe_to_seconds,
)
from event import BarEvent

UTC = datetime.timezone.utc


# ──────────────────────────────────────────────
# Test doubles & helpers
# ──────────────────────────────────────────────

class FakeQueue:
    """Captures every put() for inspection."""

    def __init__(self):
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _StubHandler(DataHandler):
    """Concrete DataHandler with a no-op update_bar so we can call _append_bar directly."""

    def update_bar(self) -> None:
        return None


def _make_ohlcv(closes, *, start='2026-01-01', freq='1h', tz='UTC') -> pd.DataFrame:
    """Build a small OHLCV DataFrame with O=H=L=C=close[i] and Volume=1.0."""
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz=tz)
    return pd.DataFrame({
        'Open':   list(closes),
        'High':   list(closes),
        'Low':    list(closes),
        'Close':  list(closes),
        'Volume': [1.0] * len(closes),
    }, index=idx)


# ──────────────────────────────────────────────
# Section 1 — _timeframe.py pure helpers
# ──────────────────────────────────────────────

def test_parse_timeframe_to_seconds_known_units():
    assert parse_timeframe_to_seconds('1m') == 60
    assert parse_timeframe_to_seconds('5m') == 300
    assert parse_timeframe_to_seconds('1h') == 3600
    assert parse_timeframe_to_seconds('4h') == 4 * 3600
    assert parse_timeframe_to_seconds('1d') == 86400
    assert parse_timeframe_to_seconds('1w') == 7 * 86400
    # Monthly/yearly are approximations used for ordering only.
    assert parse_timeframe_to_seconds('1M') == 30 * 86400
    assert parse_timeframe_to_seconds('1Y') == 365 * 86400


def test_parse_timeframe_to_seconds_case_sensitive_units():
    # Lowercase 'm' is minutes; uppercase 'M' is months — must differ.
    assert parse_timeframe_to_seconds('1m') == 60
    assert parse_timeframe_to_seconds('1M') == 30 * 86400
    # Lowercase 'y' has no meaning; only uppercase 'Y' is yearly.
    with pytest.raises(ValueError):
        parse_timeframe_to_seconds('1y')


@pytest.mark.parametrize('bad', ['', 'h', '1', '10mh', '0h', '-1h', 'abc'])
def test_parse_timeframe_to_seconds_rejects_malformed(bad):
    with pytest.raises(ValueError) as exc:
        parse_timeframe_to_seconds(bad)
    # Message should reference the offending input rather than surfacing a
    # raw int() parse error.
    assert bad in str(exc.value) or 'positive' in str(exc.value)


def test_parse_date_date_only_string_is_utc_midnight():
    dt = _parse_date('2026-01-01')
    assert dt == datetime.datetime(2026, 1, 1, tzinfo=UTC)
    assert dt.tzinfo is not None


def test_parse_date_iso_with_z_is_utc():
    dt = _parse_date('2026-01-01T12:00:00Z')
    assert dt == datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_parse_date_iso_with_offset_keeps_tz():
    dt = _parse_date('2026-01-01T12:00:00+02:00')
    # Same instant as 10:00 UTC.
    assert dt.utcoffset() == datetime.timedelta(hours=2)
    assert dt.astimezone(UTC) == datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)


def test_parse_date_passes_through_datetime_unchanged():
    src = datetime.datetime(2026, 3, 15, 9, 30, tzinfo=UTC)
    assert _parse_date(src) is src


def test_ms_to_utc_epoch_zero():
    dt = _ms_to_utc(0)
    assert dt == datetime.datetime(1970, 1, 1, tzinfo=UTC)


def test_ms_to_utc_nonzero_with_fractional():
    # 1_700_000_000_500 ms = 1_700_000_000.5 s
    dt = _ms_to_utc(1_700_000_000_500)
    expected = datetime.datetime.fromtimestamp(1_700_000_000.5, UTC)
    assert dt == expected
    assert dt.tzinfo == UTC


def test_ensure_utc_naive_gets_utc_tag():
    naive = datetime.datetime(2026, 1, 1, 5, 0)
    out = _ensure_utc(naive)
    assert out.tzinfo == UTC
    # Wall-clock value preserved (just labelled UTC).
    assert out.replace(tzinfo=None) == naive


def test_ensure_utc_other_tz_converted_to_utc():
    plus2 = datetime.timezone(datetime.timedelta(hours=2))
    src = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=plus2)
    out = _ensure_utc(src)
    assert out.tzinfo == UTC
    assert out == datetime.datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def testget_period_start_1h_aligns_to_hour():
    ts = datetime.datetime(2026, 4, 18, 12, 34, 56, tzinfo=UTC)
    out = get_period_start(ts, '1h')
    assert out == datetime.datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)


def testget_period_start_4h_aligns_to_4h_block_from_midnight():
    ts = datetime.datetime(2026, 4, 18, 6, 0, 0, tzinfo=UTC)
    out = get_period_start(ts, '4h')
    assert out == datetime.datetime(2026, 4, 18, 4, 0, 0, tzinfo=UTC)


def testget_period_start_daily_aligns_to_midnight():
    ts = datetime.datetime(2026, 4, 18, 12, 34, 56, tzinfo=UTC)
    out = get_period_start(ts, '1d')
    assert out == datetime.datetime(2026, 4, 18, 0, 0, 0, tzinfo=UTC)


def testget_period_start_weekly_aligns_to_monday():
    # 2026-01-08 is a Thursday; ISO Monday of that week is 2026-01-05.
    ts_thu = datetime.datetime(2026, 1, 8, 15, 30, tzinfo=UTC)
    assert get_period_start(ts_thu, '1w') == datetime.datetime(2026, 1, 5, tzinfo=UTC)

    # 2026-01-11 is the Sunday of the same ISO week → still 2026-01-05.
    ts_sun = datetime.datetime(2026, 1, 11, 3, 0, tzinfo=UTC)
    assert get_period_start(ts_sun, '1w') == datetime.datetime(2026, 1, 5, tzinfo=UTC)

    # Monday at exactly 00:00 is idempotent.
    ts_mon = datetime.datetime(2026, 1, 5, tzinfo=UTC)
    assert get_period_start(ts_mon, '1w') == ts_mon

    # Monday later in the day still anchors to that same Monday 00:00.
    ts_mon_late = datetime.datetime(2026, 1, 5, 23, 59, tzinfo=UTC)
    assert get_period_start(ts_mon_late, '1w') == datetime.datetime(2026, 1, 5, tzinfo=UTC)


def testget_period_start_monthly_aligns_to_month_start():
    # Mid-month → first of that same month at 00:00 UTC.
    ts = datetime.datetime(2026, 4, 18, 12, 34, 56, tzinfo=UTC)
    assert get_period_start(ts, '1M') == datetime.datetime(2026, 4, 1, tzinfo=UTC)

    # First-of-month at midnight is idempotent.
    ts_first = datetime.datetime(2026, 4, 1, tzinfo=UTC)
    assert get_period_start(ts_first, '1M') == ts_first


def testget_period_start_quarterly_aligns_to_jan_apr_jul_oct():
    # '3M' buckets to Jan/Apr/Jul/Oct.
    ts_feb = datetime.datetime(2026, 2, 14, tzinfo=UTC)
    assert get_period_start(ts_feb, '3M') == datetime.datetime(2026, 1, 1, tzinfo=UTC)
    ts_may = datetime.datetime(2026, 5, 20, tzinfo=UTC)
    assert get_period_start(ts_may, '3M') == datetime.datetime(2026, 4, 1, tzinfo=UTC)
    ts_nov = datetime.datetime(2026, 11, 1, tzinfo=UTC)
    assert get_period_start(ts_nov, '3M') == datetime.datetime(2026, 10, 1, tzinfo=UTC)


def testget_period_start_yearly_aligns_to_jan_1():
    ts = datetime.datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
    assert get_period_start(ts, '1Y') == datetime.datetime(2026, 1, 1, tzinfo=UTC)
    # Jan 1 at midnight is idempotent.
    ts_jan = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    assert get_period_start(ts_jan, '1Y') == ts_jan


def testget_period_start_rejects_unsupported_multiday():
    # 2-day, 3-day, etc. previously truncated silently to midnight (wrong).
    # They should now raise rather than misalign.
    ts = datetime.datetime(2026, 1, 8, tzinfo=UTC)
    with pytest.raises(ValueError):
        get_period_start(ts, '2d')
    with pytest.raises(ValueError):
        get_period_start(ts, '3d')


def test_timeframe_fallback_order_is_finest_first():
    # Sanity: each entry has strictly greater seconds than the previous.
    secs = [parse_timeframe_to_seconds(tf) for tf in TIMEFRAME_FALLBACK_ORDER]
    assert secs == sorted(secs)
    assert len(set(secs)) == len(secs)


# ──────────────────────────────────────────────
# Section 2 — _ohlcv.py DataFrame helpers
# ──────────────────────────────────────────────

def test_candles_to_dataframe_shape_and_index():
    candles = [
        [0,                  1.0, 2.0, 0.5, 1.5, 10.0],
        [60_000,             1.5, 2.5, 1.0, 2.0, 20.0],
        [120_000,            2.0, 3.0, 1.5, 2.5, 30.0],
    ]
    df = _candles_to_dataframe(candles)
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 3
    assert df.index[0] == datetime.datetime(1970, 1, 1, tzinfo=UTC)
    assert df.index[1] == datetime.datetime(1970, 1, 1, 0, 1, tzinfo=UTC)
    assert df.iloc[0]['Open'] == 1.0
    assert df.iloc[2]['Volume'] == 30.0


def test_candles_to_dataframe_empty():
    df = _candles_to_dataframe([])
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 0


def test_resample_ohlcv_4h_aggregation_rules():
    # 8 hourly bars starting at midnight UTC, closes 1..8.
    df = _make_ohlcv(closes=range(1, 9), start='2026-01-01', freq='1h')
    # Make Open/High/Low distinct so we can pin the agg rules.
    df['Open']  = [1, 2, 3, 4, 5, 6, 7, 8]
    df['High']  = [10, 9, 8, 7, 11, 6, 12, 5]
    df['Low']   = [-1, -2, -3, -4, -5, -6, -7, -8]
    df['Close'] = [1, 2, 3, 4, 5, 6, 7, 8]
    df['Volume'] = [1.0] * 8

    out = _resample_ohlcv(df, '4h')
    assert len(out) == 2
    # First 4h bucket: 00:00 covers hours 0..3.
    assert out.index[0] == pd.Timestamp('2026-01-01 00:00', tz='UTC')
    assert out.iloc[0]['Open']   == 1
    assert out.iloc[0]['High']   == 10
    assert out.iloc[0]['Low']    == -4
    assert out.iloc[0]['Close']  == 4
    assert out.iloc[0]['Volume'] == 4.0
    # Second 4h bucket: 04:00 covers hours 4..7.
    assert out.index[1] == pd.Timestamp('2026-01-01 04:00', tz='UTC')
    assert out.iloc[1]['Open']   == 5
    assert out.iloc[1]['High']   == 12
    assert out.iloc[1]['Low']    == -8
    assert out.iloc[1]['Close']  == 8
    assert out.iloc[1]['Volume'] == 4.0


def test_resample_ohlcv_daily_one_bar_per_day():
    # 48 hourly bars across 2 calendar days.
    df = _make_ohlcv(closes=list(range(48)), start='2026-01-01', freq='1h')
    out = _resample_ohlcv(df, '1d')
    assert len(out) == 2
    assert out.index[0] == pd.Timestamp('2026-01-01', tz='UTC')
    assert out.index[1] == pd.Timestamp('2026-01-02', tz='UTC')
    # Day 1: closes 0..23
    assert out.iloc[0]['Open']   == 0
    assert out.iloc[0]['Close']  == 23
    assert out.iloc[0]['Volume'] == 24.0
    # Day 2: closes 24..47
    assert out.iloc[1]['Open']   == 24
    assert out.iloc[1]['Close']  == 47


def test_resample_ohlcv_uneven_tf_resets_at_midnight():
    # 7h does NOT divide 24h evenly (86400 % 25200 != 0), so the per-day
    # reset path triggers — every day starts a fresh bucket at midnight.
    df = _make_ohlcv(closes=list(range(48)), start='2026-01-01', freq='1h')
    out = _resample_ohlcv(df, '7h')

    # No bucket should span midnight.
    midnights = {pd.Timestamp('2026-01-01', tz='UTC'),
                 pd.Timestamp('2026-01-02', tz='UTC')}
    bucket_starts = set(out.index)
    assert midnights.issubset(bucket_starts)

    # Day 1 buckets: 00:00, 07:00, 14:00, 21:00 → 4 buckets.
    # Day 2 buckets: 00:00, 07:00, 14:00, 21:00 → 4 buckets.
    assert len(out) == 8

    first = out.loc[pd.Timestamp('2026-01-01 00:00', tz='UTC')]
    # Bucket 00:00–06:00 of day 1: closes 0..6, sum volume = 7.
    assert first['Open']   == 0
    assert first['Close']  == 6
    assert first['Volume'] == 7.0


def test_resample_ohlcv_weekly_labels_at_monday():
    # 14 daily bars starting 2026-01-01 (Thursday). Spans 3 ISO weeks:
    #   Mon 2025-12-29 .. Sun 2026-01-04 (4 bars: Thu..Sun)
    #   Mon 2026-01-05 .. Sun 2026-01-11 (7 bars)
    #   Mon 2026-01-12 .. Sun 2026-01-18 (3 bars)
    df = _make_ohlcv(closes=list(range(14)), start='2026-01-01', freq='1D')
    out = _resample_ohlcv(df, '1w')

    expected_mondays = [
        pd.Timestamp('2025-12-29', tz='UTC'),
        pd.Timestamp('2026-01-05', tz='UTC'),
        pd.Timestamp('2026-01-12', tz='UTC'),
    ]
    assert list(out.index) == expected_mondays

    # First bucket: Thu(0), Fri(1), Sat(2), Sun(3) → open=0, close=3, vol=4.
    assert out.iloc[0]['Open']   == 0
    assert out.iloc[0]['Close']  == 3
    assert out.iloc[0]['Volume'] == 4.0
    # Second bucket: 4..10 → open=4, close=10, vol=7.
    assert out.iloc[1]['Open']   == 4
    assert out.iloc[1]['Close']  == 10
    assert out.iloc[1]['Volume'] == 7.0
    # Third bucket: 11, 12, 13 → open=11, close=13, vol=3.
    assert out.iloc[2]['Open']   == 11
    assert out.iloc[2]['Close']  == 13
    assert out.iloc[2]['Volume'] == 3.0


def test_resample_ohlcv_empty_returns_empty():
    empty = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    empty.index = pd.DatetimeIndex([], tz='UTC')
    out = _resample_ohlcv(empty, '4h')
    assert len(out) == 0
    assert list(out.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']


# ──────────────────────────────────────────────
# Section 3 — DataHandler deque + HTF aggregation
# ──────────────────────────────────────────────

def _new_stub(*, base='1h', timeframes=None, symbols=('BTC',)):
    if timeframes is None:
        timeframes = {base: 100}
    return _StubHandler(
        events_queue=FakeQueue(),
        symbol_list=list(symbols),
        base_timeframe=base,
        timeframes=timeframes,
    )


def test_append_bar_distinct_timestamps_extend_deque():
    h = _new_stub()
    t0 = datetime.datetime(2026, 1, 1, 0, tzinfo=UTC)
    t1 = datetime.datetime(2026, 1, 1, 1, tzinfo=UTC)
    h._append_bar('BTC', t0, 1, 2, 0, 1.5, 10)
    h._append_bar('BTC', t1, 2, 3, 1, 2.5, 20)
    deq = h._base_bar_data['BTC']
    assert len(deq) == 2
    assert deq[0].timestamp == t0 and deq[1].timestamp == t1


def test_append_bar_same_timestamp_replaces_last_entry():
    h = _new_stub()
    t0 = datetime.datetime(2026, 1, 1, 0, tzinfo=UTC)
    h._append_bar('BTC', t0, 1, 2, 0, 1.5, 10)
    h._append_bar('BTC', t0, 1, 5, 0, 4.0, 25)
    deq = h._base_bar_data['BTC']
    assert len(deq) == 1
    bar = deq[0]
    assert bar.timestamp == t0
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (1, 5, 0, 4.0, 25)


def test_append_bar_respects_maxlen():
    h = _new_stub(base='1h', timeframes={'1h': 3})
    for i in range(5):
        ts = datetime.datetime(2026, 1, 1, i, tzinfo=UTC)
        h._append_bar('BTC', ts, i, i, i, i, 1)
    deq = h._base_bar_data['BTC']
    assert len(deq) == 3
    # Should retain the last 3 hours (2, 3, 4).
    assert [bar.timestamp.hour for bar in deq] == [2, 3, 4]


def test_htf_aggregation_within_one_period():
    h = _new_stub(timeframes={'1h': 100, '4h': 50})
    # Four 1h bars all inside the 04:00 4h bucket.
    bars = [
        # ts hour, O, H, L, C, V
        (4,  10.0, 12.0,  9.0, 11.0, 100.0),
        (5,  11.0, 15.0, 10.0, 14.0, 150.0),
        (6,  14.0, 14.5, 13.0, 13.5, 120.0),
        (7,  13.5, 16.0, 13.0, 15.5, 130.0),
    ]
    for hour, o, hi, lo, c, v in bars:
        ts = datetime.datetime(2026, 1, 1, hour, tzinfo=UTC)
        h._append_bar('BTC', ts, o, hi, lo, c, v)

    htf = h._htf_bar_data[('BTC', '4h')]
    assert len(htf) == 1
    bar = htf[0]
    assert bar.timestamp == datetime.datetime(2026, 1, 1, 4, tzinfo=UTC)
    assert bar.open == 10.0          # first
    assert bar.high == 16.0          # max
    assert bar.low == 9.0            # min
    assert bar.close == 15.5         # last
    assert math.isclose(bar.volume, 500.0)  # sum


def test_htf_period_rollover_appends_new_entry():
    h = _new_stub(timeframes={'1h': 100, '4h': 50})
    # One 1h bar in the 00:00 4h bucket, then one in the 04:00 bucket.
    h._append_bar('BTC', datetime.datetime(2026, 1, 1, 3, tzinfo=UTC),
                  1, 2, 0, 1.5, 10)
    h._append_bar('BTC', datetime.datetime(2026, 1, 1, 4, tzinfo=UTC),
                  5, 6, 4, 5.5, 20)

    htf = h._htf_bar_data[('BTC', '4h')]
    assert len(htf) == 2
    assert htf[0].timestamp == datetime.datetime(2026, 1, 1, 0, tzinfo=UTC)
    assert htf[1].timestamp == datetime.datetime(2026, 1, 1, 4, tzinfo=UTC)
    # First bucket finalised — only saw the 03:00 bar.
    first = htf[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (1, 2, 0, 1.5, 10)


def test_htf_not_advanced_when_base_is_forming():
    h = _new_stub(timeframes={'1h': 100, '4h': 50})
    ts = datetime.datetime(2026, 1, 1, 4, tzinfo=UTC)
    # Forming emission — base deque updated, but HTF must not be touched.
    h._append_bar('BTC', ts, 10, 11, 9, 10.5, 50, is_forming=True)
    assert len(h._base_bar_data['BTC']) == 1
    assert len(h._htf_bar_data[('BTC', '4h')]) == 0

    # A subsequent forming emission for the same ts replaces base, still no HTF.
    h._append_bar('BTC', ts, 10, 12, 9, 11.5, 75, is_forming=True)
    assert len(h._base_bar_data['BTC']) == 1
    assert h._base_bar_data['BTC'][0].volume == 75
    assert len(h._htf_bar_data[('BTC', '4h')]) == 0

    # Once the bar completes, HTF advances (and uses the completed values, not
    # an accumulation of forming snapshots — so volume is 75, not 125).
    h._append_bar('BTC', ts, 10, 12, 9, 11.5, 75, is_forming=False)
    htf = h._htf_bar_data[('BTC', '4h')]
    assert len(htf) == 1
    assert htf[0].volume == 75


def test_bar_is_immutable():
    from dataclasses import FrozenInstanceError
    from data._bar import Bar
    bar = Bar(datetime.datetime(2026, 1, 1, tzinfo=UTC), 1.0, 2.0, 0.0, 1.5, 10.0)
    with pytest.raises(FrozenInstanceError):
        bar.close = 99.0  # type: ignore[misc]


def test_get_latest_bars_base_default_and_explicit_match():
    h = _new_stub(base='1h', timeframes={'1h': 50})
    for i in range(3):
        ts = datetime.datetime(2026, 1, 1, i, tzinfo=UTC)
        h._append_bar('BTC', ts, i, i, i, i, 1)
    df_default  = h.get_latest_bars('BTC', 5)
    df_explicit = h.get_latest_bars('BTC', 5, '1h')
    assert df_default.equals(df_explicit)


def test_get_latest_bars_n_caps_to_available_and_columns_correct():
    h = _new_stub(base='1h', timeframes={'1h': 50})
    for i in range(3):
        ts = datetime.datetime(2026, 1, 1, i, tzinfo=UTC)
        h._append_bar('BTC', ts, i, i, i, i, 1)
    df = h.get_latest_bars('BTC', 10)
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 3
    assert df.index[0].tzinfo is not None


def test_get_latest_bars_empty_deque_returns_empty_frame():
    h = _new_stub()
    df = h.get_latest_bars('BTC', 5)
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 0


def test_get_latest_bars_unregistered_timeframe_raises():
    h = _new_stub(base='1h', timeframes={'1h': 50, '4h': 50})
    with pytest.raises(ValueError) as exc:
        h.get_latest_bars('BTC', 5, '1d')
    msg = str(exc.value)
    assert '1d' in msg
    assert '1h' in msg and '4h' in msg


def test_htf_weekly_aggregation_one_bucket_per_week():
    # Daily base TF + weekly HTF. 10 daily bars starting 2026-01-01 (Thursday)
    # span two ISO weeks (Mon 2025-12-29 and Mon 2026-01-05). The buggy
    # get_period_start would produce one HTF entry per day.
    h = _new_stub(base='1d', timeframes={'1d': 100, '1w': 50})
    closes = list(range(10))
    for i, c in enumerate(closes):
        ts = datetime.datetime(2026, 1, 1, tzinfo=UTC) + datetime.timedelta(days=i)
        h._append_bar('BTC', ts, c, c, c, c, 1)

    htf = h._htf_bar_data[('BTC', '1w')]
    assert len(htf) == 2

    week1 = htf[0]
    week2 = htf[1]
    assert week1.timestamp == datetime.datetime(2025, 12, 29, tzinfo=UTC)
    assert week2.timestamp == datetime.datetime(2026, 1, 5, tzinfo=UTC)

    # Week 1 covers Thu..Sun (4 daily bars: closes 0..3).
    assert (week1.open, week1.close, week1.volume) == (0, 3, 4)
    # Week 2 covers Mon..Sun (6 of the remaining 7 not yet present? actually 7 bars: 4..9 inclusive? wait)
    # Days 5..10 of January are 6 bars (closes 4..9), all within week2's Mon..Sun range.
    assert (week2.open, week2.close, week2.volume) == (4, 9, 6)


def test_get_latest_bars_unknown_symbol_raises():
    h = _new_stub(base='1h', timeframes={'1h': 50, '4h': 50}, symbols=('BTC',))
    # Base TF: previously silently returned an empty frame (and created a deque).
    with pytest.raises(ValueError) as exc_base:
        h.get_latest_bars('UNKNOWN')
    msg_base = str(exc_base.value)
    assert 'UNKNOWN' in msg_base and 'BTC' in msg_base

    # HTF: previously raised raw KeyError. Should now raise ValueError too.
    with pytest.raises(ValueError) as exc_htf:
        h.get_latest_bars('UNKNOWN', 5, '4h')
    msg_htf = str(exc_htf.value)
    assert 'UNKNOWN' in msg_htf and 'BTC' in msg_htf


def test_unknown_symbol_no_silent_deque_creation():
    h = _new_stub(base='1h', timeframes={'1h': 50}, symbols=('BTC',))
    registered_before = set(h._base_bar_data.keys())
    with pytest.raises(ValueError):
        h.get_latest_bars('UNKNOWN')
    # The failing lookup must not have inserted a new deque (the old
    # defaultdict behaviour silently created one).
    assert set(h._base_bar_data.keys()) == registered_before
    assert 'UNKNOWN' not in h._base_bar_data


# ──────────────────────────────────────────────
# Section 3a — NaN-OHLC gate
# ──────────────────────────────────────────────
#
# DataHandler is the single source of truth for non-NaN OHLC. Bars whose
# Open/High/Low/Close contain NaN are dropped at _append_bar (return False,
# WARNING logged, deque untouched). Volume NaN is intentionally accepted —
# downstream accounting/sizing/execution don't depend on volume.

import logging  # local import keeps the test file's main imports tidy
NAN = float('nan')


def test_append_bar_rejects_nan_close_returns_false(caplog):
    h = _new_stub()
    ts = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    with caplog.at_level(logging.WARNING, logger='data._base'):
        accepted = h._append_bar('BTC', ts, 1.0, 2.0, 0.5, NAN, 10.0)
    assert accepted is False
    assert len(h._base_bar_data['BTC']) == 0
    assert any('NaN OHLC' in rec.message for rec in caplog.records)


def test_append_bar_rejects_partial_nan_high():
    h = _new_stub()
    ts = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    accepted = h._append_bar('BTC', ts, 1.0, NAN, 0.5, 1.5, 10.0)
    assert accepted is False
    assert len(h._base_bar_data['BTC']) == 0


def test_append_bar_accepts_nan_volume():
    """Volume NaN must not drop the bar — only OHLC is gated."""
    h = _new_stub()
    ts = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    accepted = h._append_bar('BTC', ts, 1.0, 2.0, 0.5, 1.5, NAN)
    assert accepted is True
    assert len(h._base_bar_data['BTC']) == 1


def test_append_bar_returns_true_for_valid_bar():
    h = _new_stub()
    ts = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    accepted = h._append_bar('BTC', ts, 1.0, 2.0, 0.5, 1.5, 10.0)
    assert accepted is True


def test_append_bar_rejected_does_not_propagate_to_htf():
    """A NaN base bar must not contaminate the HTF aggregator."""
    h = _new_stub(base='1h', timeframes={'1h': 50, '4h': 50})
    ts = datetime.datetime(2026, 1, 1, 0, tzinfo=UTC)
    h._append_bar('BTC', ts, NAN, NAN, NAN, NAN, 10.0)
    assert len(h._htf_bar_data[('BTC', '4h')]) == 0


def test_historic_handler_drops_nan_row_from_dict_input(caplog):
    """End-to-end: a NaN row in the dict input does not reach the events queue."""
    btc = _make_ohlcv(closes=[100.0, 101.0, 102.0],
                      start='2026-01-01 00:00', freq='1h')
    btc.iloc[1, btc.columns.get_loc('Close')] = NAN  # poison the middle row

    q = FakeQueue()
    h = HistoricDataHandler(
        events_queue=q,
        symbol_list=['BTC_USDT'],
        base_timeframe='1h',
        timeframes={'1h': 50},
        data={'BTC_USDT': btc},
    )

    with caplog.at_level(logging.WARNING, logger='data._base'):
        while h.continue_backtest:
            h.update_bar()

    assert len(q.items) == 2
    assert [b.close for b in q.items] == [100.0, 102.0]
    assert len(h._base_bar_data['BTC_USDT']) == 2
    assert any('NaN OHLC' in rec.message for rec in caplog.records)


# ──────────────────────────────────────────────
# Section 4 — HistoricDataHandler in-memory path
# ──────────────────────────────────────────────

def test_historic_handler_requires_data_or_dates():
    with pytest.raises(ValueError):
        HistoricDataHandler(
            events_queue=thread_queue.Queue(),
            symbol_list=['BTC_USDT'],
            base_timeframe='1h',
            timeframes={'1h': 50},
        )


def test_historic_handler_drains_two_symbols_in_time_order():
    btc = _make_ohlcv(closes=[100, 101, 102, 103, 104],
                      start='2026-01-01 00:00', freq='1h')
    eth = _make_ohlcv(closes=[10, 11, 12, 13, 14],
                     start='2026-01-01 00:30', freq='1h')

    q = FakeQueue()
    h = HistoricDataHandler(
        events_queue=q,
        symbol_list=['BTC_USDT', 'ETH_USDT'],
        base_timeframe='1h',
        timeframes={'1h': 50},
        data={'BTC_USDT': btc, 'ETH_USDT': eth},
    )

    # Drain.
    while h.continue_backtest:
        h.update_bar()

    # All 10 rows emitted.
    assert len(q.items) == 10
    assert all(isinstance(b, BarEvent) for b in q.items)

    # Time-sorted order across both symbols.
    timestamps = [b.timestamp for b in q.items]
    assert timestamps == sorted(timestamps)

    # Interleaving: BTC at :00, ETH at :30, alternating.
    expected_symbols = ['BTC_USDT', 'ETH_USDT'] * 5
    assert [b.symbol for b in q.items] == expected_symbols

    # Period and OHLCV propagated.
    first_btc = q.items[0]
    assert first_btc.period == '1h'
    assert first_btc.close == 100.0
    assert first_btc.volume == 1.0


def test_historic_handler_appends_to_internal_deque():
    btc = _make_ohlcv(closes=[100, 101, 102], start='2026-01-01 00:00', freq='1h')
    h = HistoricDataHandler(
        events_queue=FakeQueue(),
        symbol_list=['BTC_USDT'],
        base_timeframe='1h',
        timeframes={'1h': 50},
        data={'BTC_USDT': btc},
    )
    while h.continue_backtest:
        h.update_bar()

    df = h.get_latest_bars('BTC_USDT', 10)
    assert len(df) == 3
    assert list(df['Close']) == [100.0, 101.0, 102.0]


def test_historic_handler_stops_after_exhausting_stream():
    btc = _make_ohlcv(closes=[100, 101], start='2026-01-01 00:00', freq='1h')
    h = HistoricDataHandler(
        events_queue=FakeQueue(),
        symbol_list=['BTC_USDT'],
        base_timeframe='1h',
        timeframes={'1h': 50},
        data={'BTC_USDT': btc},
    )
    h.update_bar()
    h.update_bar()
    assert h.continue_backtest is True
    h.update_bar()  # drains the StopIteration
    assert h.continue_backtest is False
