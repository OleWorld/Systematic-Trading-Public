"""
Unit tests for the stateful ``Indicator`` classes and their vectorized
``from_series`` companions.

For each of the 9 indicator classes we cover:

1. ``from_series`` — pinned hand-computed expectations or pandas references
   (preserves the coverage of the previous pure-function ``test_indicator.py``).
2. ``update``-driven stateful execution — output matches ``from_series``
   bar-for-bar (golden test).
3. Upsert semantics on the forming bar — re-ticking the same timestamp does
   NOT corrupt recursive state. The recursive indicators additionally
   verify that re-ticked output equals a fresh instance fed only the final
   tick value.
4. ``is_latest_ready`` lifecycle — false until a finalized non-NaN value exists.
5. ``reset()`` clears all state.
6. ``warmup(history)`` produces the same outputs as bar-by-bar ``update``.

Run from the repo root:  python -m pytest tests/test_indicator_classes.py -v
"""

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from indicator import (
    ATR,
    BBW,
    EMA,
    EWMStdev,
    KAMA,
    PercentRank,
    RSI,
    SMA,
    Stdev,
    TrailingVolatilityStop,
)


# ──────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────

def _ts_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2024-01-01', periods=n, freq='1h')


@pytest.fixture
def random_close() -> pd.Series:
    """Deterministic 200-bar close series for parity tests."""
    rng = np.random.default_rng(0)
    walk = rng.standard_normal(200).cumsum() + 100.0
    return pd.Series(walk, index=_ts_index(200), name='close')


@pytest.fixture
def random_ohlc() -> pd.DataFrame:
    """Deterministic 200-bar OHLC frame for ATR-style tests."""
    rng = np.random.default_rng(0)
    walk = rng.standard_normal(200).cumsum() + 100.0
    high_offsets = np.abs(rng.standard_normal(200)) * 0.5
    low_offsets = np.abs(rng.standard_normal(200)) * 0.5
    return pd.DataFrame({
        'Open': walk,
        'High': walk + high_offsets,
        'Low': walk - low_offsets,
        'Close': walk,
        'Volume': np.zeros(200),
    }, index=_ts_index(200))


# ──────────────────────────────────────────────
# from_series — preserves the pure-function test coverage
# ──────────────────────────────────────────────

def test_sma_from_series_simple_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = SMA.from_series(s, window=3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_from_series_matches_pandas():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = EMA.from_series(s, span=3)
    expected = s.ewm(span=3, min_periods=3, adjust=False).mean()
    np.testing.assert_allclose(out.values, expected.values, equal_nan=True)


def test_stdev_from_series_sample_formula():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = Stdev.from_series(s, length=3)
    assert pd.isna(out.iloc[0])
    assert out.iloc[2] == pytest.approx(1.0)
    assert out.iloc[4] == pytest.approx(1.0)


def test_atr_from_series_matches_ema_of_true_range():
    high = pd.Series([10.0, 12.0, 13.0, 11.5, 14.0, 15.0, 13.5, 12.0, 13.0, 14.5])
    low = pd.Series([9.0, 10.5, 11.0, 10.0, 12.0, 13.5, 11.5, 11.0, 11.5, 13.0])
    close = pd.Series([9.5, 11.5, 12.5, 10.5, 13.5, 14.5, 12.0, 11.5, 12.5, 14.0])

    out = ATR.from_series(high, low, close, length=5)

    # Independent TR computation (skipna max for the first bar).
    tr = [high.iloc[0] - low.iloc[0]]
    for i in range(1, len(high)):
        tr.append(max(
            high.iloc[i] - low.iloc[i],
            abs(high.iloc[i] - close.iloc[i - 1]),
            abs(low.iloc[i] - close.iloc[i - 1]),
        ))
    # ATR uses Wilder/RMA smoothing on TR (alpha = 1/length, i.e. com=length-1).
    expected = pd.Series(tr).ewm(com=5 - 1, min_periods=5, adjust=False).mean()
    np.testing.assert_allclose(out.values, expected.values, equal_nan=True)


def test_bbw_from_series_formula():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = BBW.from_series(s, length=3, mult=2.0)
    # Window [1,2,3]: basis=2, dev=2*1=2, upper=4, lower=0, BBW=4/2=2.
    assert out.iloc[2] == pytest.approx(2.0)
    # Window [2,3,4]: basis=3, dev=2, upper=5, lower=1, BBW=4/3.
    assert out.iloc[3] == pytest.approx(4.0 / 3.0)


def test_percentrank_from_series_monotone_increasing():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = PercentRank.from_series(s, length=5)
    # Window [1..5]: prior [1,2,3,4] all < 5 → 4/4 * 100 = 100.
    assert out.iloc[4] == pytest.approx(100.0)


def test_percentrank_from_series_all_equal_yields_zero():
    s = pd.Series([3.0] * 5)
    out = PercentRank.from_series(s, length=5)
    assert out.iloc[4] == pytest.approx(0.0)


def test_kama_from_series_masks_warmup():
    s = pd.Series([100.0, 101.0, 102.0, 103.0])
    out = KAMA.from_series(s, er_length=3, fast=2, slow=30)
    # First er_length outputs are masked; the recursion is seeded with
    # values[0] internally but never surfaces.
    assert out.iloc[:3].isna().all()
    assert np.isfinite(out.iloc[3])


def test_kama_from_series_constant_input_tracks_constant():
    s = pd.Series([5.0] * 20)
    out = KAMA.from_series(s, er_length=5, fast=2, slow=30)
    assert out.iloc[:5].isna().all()
    np.testing.assert_allclose(out.iloc[5:].to_numpy(), 5.0)


def test_trailing_stop_from_series_uptrend_stays_long():
    price = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    trigger = price.copy()
    atr_s = pd.Series([1.0] * 6)
    stop, direction = TrailingVolatilityStop.from_series(
        price, trigger, atr_s, mult=2.0,
    )
    assert (direction == 1).all()
    assert stop.iloc[0] == pytest.approx(98.0)
    for i in range(1, len(stop)):
        assert stop.iloc[i] >= stop.iloc[i - 1]


def test_trailing_stop_from_series_flips_on_breach():
    price = pd.Series([100.0, 105.0, 110.0, 90.0, 85.0])
    trigger = price.copy()
    atr_s = pd.Series([1.0] * 5)
    stop, direction = TrailingVolatilityStop.from_series(
        price, trigger, atr_s, mult=2.0,
    )
    assert direction.iloc[3] == -1
    assert stop.iloc[3] == pytest.approx(92.0)
    assert direction.iloc[4] == -1
    assert stop.iloc[4] == pytest.approx(87.0)


# ──────────────────────────────────────────────
# Stateful ↔ vectorized parity (golden tests)
# ──────────────────────────────────────────────

def _drive_single(ind, series: pd.Series) -> pd.Series:
    """Push ``series`` into a single-input stateful indicator and return the
    primary output column from ``get_latest_indicators``."""
    for ts, v in series.items():
        ind.update(ts, float(v))
    df = ind.get_latest_indicators(len(series))
    # First (only) public column is the primary output.
    return df.iloc[:, 0]


def test_sma_stateful_matches_vectorized(random_close):
    vec = SMA.from_series(random_close, window=14)
    state = _drive_single(SMA(window=14), random_close)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_ema_stateful_matches_vectorized(random_close):
    vec = EMA.from_series(random_close, span=20)
    state = _drive_single(EMA(span=20), random_close)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_kama_stateful_matches_vectorized(random_close):
    vec = KAMA.from_series(random_close, er_length=10, fast=2, slow=30)
    state = _drive_single(KAMA(er_length=10, fast=2, slow=30), random_close)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_rsi_stateful_matches_vectorized(random_close):
    vec = RSI.from_series(random_close, window=14)
    ind = RSI(window=14)
    for ts, v in random_close.items():
        ind.update(ts, float(v))
    rsi_col = ind.get_latest_indicators(len(random_close))['rsi']
    np.testing.assert_allclose(rsi_col.values, vec.values, equal_nan=True, rtol=1e-12)


def test_percentrank_stateful_matches_vectorized(random_close):
    vec = PercentRank.from_series(random_close, length=30)
    state = _drive_single(PercentRank(length=30), random_close)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_stdev_stateful_matches_vectorized(random_close):
    vec = Stdev.from_series(random_close, length=10)
    state = _drive_single(Stdev(length=10), random_close)
    # Looser tolerance than other indicators: pandas' rolling.std uses a
    # numerically-stable algorithm; np.std(ddof=1) on the same window can
    # differ by ~1e-12 due to floating-point summation order.
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-10)


def test_bbw_stateful_matches_vectorized(random_close):
    vec = BBW.from_series(random_close, length=20, mult=2.0)
    state = _drive_single(BBW(length=20, mult=2.0), random_close)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_atr_stateful_matches_vectorized(random_ohlc):
    h, l, c = random_ohlc['High'], random_ohlc['Low'], random_ohlc['Close']
    vec = ATR.from_series(h, l, c, length=14)
    ind = ATR(length=14)
    for ts in random_ohlc.index:
        ind.update(ts, h.loc[ts], l.loc[ts], c.loc[ts])
    atr_col = ind.get_latest_indicators(len(random_ohlc))['atr']
    np.testing.assert_allclose(atr_col.values, vec.values, equal_nan=True, rtol=1e-12)


def test_trailing_stop_stateful_matches_vectorized(random_ohlc):
    """Compose KAMA → ATR → TrailingVolatilityStop end-to-end and compare
    against the vectorized chain."""
    close = random_ohlc['Close']
    high = random_ohlc['High']
    low = random_ohlc['Low']

    kama_vec = KAMA.from_series(close, er_length=10, fast=2, slow=30)
    atr_vec = ATR.from_series(high, low, close, length=14)
    stop_vec, dir_vec = TrailingVolatilityStop.from_series(
        kama_vec, kama_vec, atr_vec, mult=4.0,
    )

    tvs = TrailingVolatilityStop(mult=4.0)
    for ts in random_ohlc.index:
        tvs.update(ts, price=kama_vec.loc[ts], trigger=kama_vec.loc[ts],
                   atr=atr_vec.loc[ts])
    tvs_df = tvs.get_latest_indicators(len(random_ohlc))
    np.testing.assert_allclose(tvs_df['stop'].values, stop_vec.values,
                               equal_nan=True, rtol=1e-12)
    np.testing.assert_array_equal(tvs_df['direction'].values, dir_vec.values)


# ──────────────────────────────────────────────
# Upsert semantics — the load-bearing invariant
# ──────────────────────────────────────────────

def _three_ticks(ind, ts1, ts2, vals: List[float]) -> None:
    """Push (ts1, vals[0]), then re-tick ts2 with vals[1:], all on a single instance."""
    ind.update(ts1, vals[0])
    for v in vals[1:]:
        ind.update(ts2, v)


def test_kama_upsert_does_not_corrupt_recursion():
    t0, t1, t2 = (dt.datetime(2024, 1, 1, h) for h in (0, 1, 2))

    # Use er_length=2 so the t2 output is past the mask boundary
    # (er_length+1=3 inputs available) and the recursion is exercised.
    # Re-ticked: 3 ticks on t1 with intermediate values ending at 120.
    a = KAMA(er_length=2, fast=2, slow=30)
    for ts, v in [(t0, 100.0), (t1, 110.0), (t1, 115.0), (t1, 120.0), (t2, 125.0)]:
        a.update(ts, v)

    # Clean: only the final tick value at t1.
    b = KAMA(er_length=2, fast=2, slow=30)
    for ts, v in [(t0, 100.0), (t1, 120.0), (t2, 125.0)]:
        b.update(ts, v)

    np.testing.assert_allclose(
        a.get_latest_indicators(10)['kama'].values,
        b.get_latest_indicators(10)['kama'].values,
        equal_nan=True, rtol=1e-12,
    )
    # Re-ticked deque has exactly 3 entries (one per distinct timestamp).
    assert len(a._outputs) == 3
    # And the last output is finite — i.e. the mask actually unmasks.
    assert np.isfinite(a.get_latest_indicators(10)['kama'].iloc[-1])


def test_ema_upsert_does_not_corrupt_recursion():
    t0, t1, t2 = (dt.datetime(2024, 1, 1, h) for h in (0, 1, 2))

    a = EMA(span=3)
    for ts, v in [(t0, 100.0), (t1, 110.0), (t1, 115.0), (t1, 120.0), (t2, 125.0)]:
        a.update(ts, v)

    b = EMA(span=3)
    for ts, v in [(t0, 100.0), (t1, 120.0), (t2, 125.0)]:
        b.update(ts, v)

    np.testing.assert_allclose(
        a.get_latest_indicators(10)['ema'].values,
        b.get_latest_indicators(10)['ema'].values,
        equal_nan=True, rtol=1e-12,
    )


def test_atr_upsert_does_not_corrupt_recursion():
    t0, t1, t2 = (dt.datetime(2024, 1, 1, h) for h in (0, 1, 2))

    # Same h/l/c at the final t1 tick as in the clean run.
    a = ATR(length=3)
    for ts, h, l, c in [
        (t0, 102.0, 99.0, 100.0),
        (t1, 112.0, 108.0, 110.0),
        (t1, 117.0, 113.0, 115.0),
        (t1, 122.0, 118.0, 120.0),
        (t2, 127.0, 123.0, 125.0),
    ]:
        a.update(ts, h, l, c)

    b = ATR(length=3)
    for ts, h, l, c in [
        (t0, 102.0, 99.0, 100.0),
        (t1, 122.0, 118.0, 120.0),
        (t2, 127.0, 123.0, 125.0),
    ]:
        b.update(ts, h, l, c)

    np.testing.assert_allclose(
        a.get_latest_indicators(10)['atr'].values,
        b.get_latest_indicators(10)['atr'].values,
        equal_nan=True, rtol=1e-12,
    )


def test_sma_upsert_replaces_forming_in_place():
    t0, t1 = dt.datetime(2024, 1, 1, 0), dt.datetime(2024, 1, 1, 1)
    sma = SMA(window=2)
    sma.update(t0, 1.0)
    sma.update(t1, 2.0)  # SMA now [nan, 1.5]
    assert sma.forming is not None and sma.forming['sma'] == pytest.approx(1.5)
    sma.update(t1, 4.0)  # re-tick: window becomes [1.0, 4.0], SMA = 2.5
    assert sma.forming is not None and sma.forming['sma'] == pytest.approx(2.5)
    # Deque still has 2 entries — re-tick replaced, did not append.
    assert len(sma._outputs) == 2


# ──────────────────────────────────────────────
# Lifecycle: is_latest_ready, latest, forming, reset, warmup
# ──────────────────────────────────────────────

def test_is_latest_ready_lifecycle_for_sma():
    sma = SMA(window=3)
    ts = _ts_index(5)
    assert sma.is_latest_ready is False
    sma.update(ts[0], 1.0)
    assert sma.is_latest_ready is False  # only one output, no `latest` yet
    sma.update(ts[1], 2.0)
    assert sma.is_latest_ready is False  # latest is iloc[-2] which is NaN (window=3)
    sma.update(ts[2], 3.0)
    assert sma.is_latest_ready is False  # latest still NaN; forming just produced first valid
    sma.update(ts[3], 4.0)
    assert sma.is_latest_ready is True   # latest = SMA at ts[2] = 2.0


def test_latest_and_forming_distinct():
    sma = SMA(window=2)
    ts = _ts_index(3)
    sma.update(ts[0], 1.0)
    sma.update(ts[1], 3.0)
    sma.update(ts[2], 5.0)
    # Outputs: [(ts0, NaN), (ts1, 2.0), (ts2, 4.0)].
    assert sma.latest is not None and sma.latest['sma'] == pytest.approx(2.0)
    assert sma.forming is not None and sma.forming['sma'] == pytest.approx(4.0)


def test_reset_clears_state(random_close):
    a = KAMA(er_length=10, fast=2, slow=30)
    for ts, v in random_close.iloc[:50].items():
        a.update(ts, float(v))
    a.reset()
    assert len(a._outputs) == 0 and len(a._inputs) == 0
    assert a.latest is None and a.forming is None and a.is_latest_ready is False

    # Drive the same 50 bars again; outputs should match a fresh instance.
    for ts, v in random_close.iloc[:50].items():
        a.update(ts, float(v))
    b = KAMA(er_length=10, fast=2, slow=30)
    for ts, v in random_close.iloc[:50].items():
        b.update(ts, float(v))
    pd.testing.assert_frame_equal(
        a.get_latest_indicators(50), b.get_latest_indicators(50),
    )


def test_warmup_series_matches_bar_by_bar(random_close):
    a = SMA(window=10)
    a.warmup(random_close)
    b = SMA(window=10)
    for ts, v in random_close.items():
        b.update(ts, float(v))
    pd.testing.assert_frame_equal(
        a.get_latest_indicators(len(random_close)),
        b.get_latest_indicators(len(random_close)),
    )


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

def test_kama_constant_input_tracks_constant():
    sma = KAMA(er_length=5, fast=2, slow=30)
    ts = _ts_index(20)
    for t in ts:
        sma.update(t, 5.0)
    df = sma.get_latest_indicators(20)['kama']
    # First er_length outputs masked; the rest must equal the constant.
    assert df.iloc[:5].isna().all()
    np.testing.assert_allclose(df.iloc[5:].values, 5.0)


def test_trailing_stop_inherits_state_through_nan_atr():
    tvs = TrailingVolatilityStop(mult=2.0)
    ts = _ts_index(3)
    tvs.update(ts[0], price=100.0, trigger=100.0, atr=float('nan'))
    tvs.update(ts[1], price=101.0, trigger=101.0, atr=float('nan'))
    tvs.update(ts[2], price=102.0, trigger=102.0, atr=1.0)
    df = tvs.get_latest_indicators(3)
    # First two bars: NaN ATR → public stop is NaN, direction defaults to +1.
    assert pd.isna(df['stop'].iloc[0])
    assert pd.isna(df['stop'].iloc[1])
    assert df['direction'].iloc[0] == 1
    assert df['direction'].iloc[1] == 1
    # Third bar: ATR finite, no prior stop → cur_stop = band_long = 102 - 2*1 = 100.
    assert df['stop'].iloc[2] == pytest.approx(100.0)
    assert df['direction'].iloc[2] == 1


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        SMA(window=0)


def test_kama_invalid_fast_slow_raises():
    with pytest.raises(ValueError):
        KAMA(er_length=10, fast=10, slow=5)


# ──────────────────────────────────────────────
# EWMStdev — EWMA of squared values, sqrt'd
# ──────────────────────────────────────────────

def test_ewmstdev_from_series_matches_pandas_ewm_var():
    s = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012, -0.008, 0.02, -0.011])
    out = EWMStdev.from_series(s, span=3)
    expected = (s.pow(2)
                 .ewm(span=3, min_periods=3, adjust=False)
                 .mean()
                 .pow(0.5))
    np.testing.assert_allclose(out.values, expected.values, equal_nan=True)


def test_ewmstdev_from_series_masks_warmup():
    s = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012])
    out = EWMStdev.from_series(s, span=3)
    # First span-1 = 2 outputs are NaN; output at index span-1 = 2 is finite.
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert np.isfinite(out.iloc[2])


def test_ewmstdev_stateful_matches_vectorized(random_close):
    # Use returns rather than prices so the zero-mean assumption is sensible.
    returns = random_close.pct_change().dropna()
    vec = EWMStdev.from_series(returns, span=20)
    state = _drive_single(EWMStdev(span=20), returns)
    np.testing.assert_allclose(state.values, vec.values, equal_nan=True, rtol=1e-12)


def test_ewmstdev_upsert_does_not_corrupt_recursion():
    t0, t1, t2 = (dt.datetime(2024, 1, 1, h) for h in (0, 1, 2))

    a = EWMStdev(span=3)
    for ts, v in [(t0, 0.01), (t1, -0.02), (t1, 0.015), (t1, -0.018), (t2, 0.01)]:
        a.update(ts, v)

    b = EWMStdev(span=3)
    for ts, v in [(t0, 0.01), (t1, -0.018), (t2, 0.01)]:
        b.update(ts, v)

    np.testing.assert_allclose(
        a.get_latest_indicators(10)['stdev'].values,
        b.get_latest_indicators(10)['stdev'].values,
        equal_nan=True, rtol=1e-12,
    )
    # Re-ticked deque has exactly 3 entries (one per distinct timestamp).
    assert len(a._outputs) == 3


def test_ewmstdev_constant_input_tracks_constant():
    """For a constant absolute value of input, EWMA-stdev settles at that value."""
    s = pd.Series([0.02] * 50)
    out = EWMStdev.from_series(s, span=10)
    # First span-1 outputs are NaN. After warmup, var_t = 0.02^2 = 0.0004,
    # stdev = 0.02 from the start (recursion seed = first squared value).
    assert pd.isna(out.iloc[:9]).all()
    np.testing.assert_allclose(out.iloc[9:].values, 0.02, rtol=1e-12)


def test_ewmstdev_zero_input_returns_zero():
    s = pd.Series([0.0] * 20)
    out = EWMStdev.from_series(s, span=5)
    np.testing.assert_allclose(out.iloc[4:].values, 0.0)
    # No NaN propagation — zero is a clean value, not NaN.
    assert not out.iloc[4:].isna().any()


def test_ewmstdev_invalid_span_raises():
    with pytest.raises(ValueError):
        EWMStdev(span=0)
    with pytest.raises(ValueError):
        EWMStdev.from_series(pd.Series([1.0, 2.0]), span=0)
