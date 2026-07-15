"""Unit tests for the ``data._tz`` UTC-enforcement helpers.

One law, three shapes: timezone-naive input raises ``ValueError``;
tz-aware non-UTC input converts to UTC; non-datetime containers raise
``TypeError``. Scalar carve-out: date-only strings get UTC midnight.

Run from the repo root:  pytest tests/test_data_tz.py -v
"""

import datetime

import pandas as pd
import pytest

from data import ensure_utc_index, ensure_utc_series, ensure_utc_timestamp

UTC = datetime.timezone.utc


# ── ensure_utc_index ─────────────────────────────────────────────────

def test_index_naive_raises():
    idx = pd.to_datetime(['2024-01-01', '2024-01-02'])
    with pytest.raises(ValueError, match=r"data\['BTC'\].*timezone-naive"):
        ensure_utc_index(idx, "data['BTC']")


def test_index_non_datetime_raises_typeerror():
    with pytest.raises(TypeError, match="equity_curve"):
        ensure_utc_index(pd.Index([1, 2, 3]), "equity_curve")


def test_index_utc_passes_through_equal():
    idx = pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True)
    out = ensure_utc_index(idx, "x")
    assert out.equals(idx)
    assert str(out.tz) == 'UTC'


def test_index_other_tz_converted_to_utc():
    idx = pd.date_range('2024-01-01 12:00', periods=2, freq='D',
                        tz='Europe/Berlin')                     # UTC+1 in Jan
    out = ensure_utc_index(idx, "x")
    assert str(out.tz) == 'UTC'
    assert out[0] == pd.Timestamp('2024-01-01 11:00', tz='UTC')


# ── ensure_utc_series ────────────────────────────────────────────────

def test_series_naive_raises():
    s = pd.Series(pd.to_datetime(['2024-01-01', '2024-01-02']))
    with pytest.raises(ValueError, match=r"trade_log\['timestamp'\]"):
        ensure_utc_series(s, "trade_log['timestamp']")


def test_series_non_datetime_dtype_raises_typeerror():
    with pytest.raises(TypeError, match="datetime"):
        ensure_utc_series(pd.Series([1.0, 2.0]), "x")


def test_series_not_a_series_raises_typeerror():
    with pytest.raises(TypeError, match="pd.Series"):
        ensure_utc_series([pd.Timestamp('2024-01-01', tz='UTC')], "x")


def test_series_empty_passes_through():
    s = pd.Series(dtype=float)                 # empty: nothing to validate
    assert ensure_utc_series(s, "x") is s


def test_series_other_tz_converted_to_utc():
    s = pd.Series(pd.to_datetime(['2024-06-01 12:00']).tz_localize(
        'US/Eastern'))                                          # UTC-4 in Jun
    out = ensure_utc_series(s, "x")
    assert str(out.dt.tz) == 'UTC'
    assert out.iloc[0] == pd.Timestamp('2024-06-01 16:00', tz='UTC')


# ── ensure_utc_timestamp ─────────────────────────────────────────────

def test_timestamp_date_only_string_gets_utc_midnight():
    out = ensure_utc_timestamp('2024-01-01', "start")
    assert out == pd.Timestamp('2024-01-01 00:00', tz='UTC')


def test_timestamp_naive_datetime_raises():
    with pytest.raises(ValueError, match="start.*timezone-naive"):
        ensure_utc_timestamp(datetime.datetime(2024, 1, 1, 5, 0), "start")


def test_timestamp_naive_datetime_string_raises():
    with pytest.raises(ValueError, match="start.*timezone-naive"):
        ensure_utc_timestamp('2024-01-01T05:00:00', "start")


def test_timestamp_naive_pd_timestamp_raises():
    with pytest.raises(ValueError, match="timezone-naive"):
        ensure_utc_timestamp(pd.Timestamp('2024-01-01 05:00'), "start")


def test_timestamp_aware_converts_to_utc():
    src = datetime.datetime(2024, 1, 1, 12, 0,
                            tzinfo=datetime.timezone(
                                datetime.timedelta(hours=2)))
    out = ensure_utc_timestamp(src, "start")
    assert out == pd.Timestamp('2024-01-01 10:00', tz='UTC')


def test_timestamp_z_suffix_string_is_utc():
    out = ensure_utc_timestamp('2024-01-01T05:00:00Z', "start")
    assert out == pd.Timestamp('2024-01-01 05:00', tz='UTC')


def test_timestamp_nat_raises():
    with pytest.raises(ValueError, match="NaT"):
        ensure_utc_timestamp(pd.NaT, "start")
