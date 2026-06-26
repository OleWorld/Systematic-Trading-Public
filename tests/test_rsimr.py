"""
Unit + integration tests for ``RSIMRStrategy``.

Covers:
* The ``_raw_from_rsi`` mapping (anchors, mean-reversion sign, symmetry,
  no-raise at the RSI 0/100 extremes).
* Parameter validation (windows, distinctness, weights, fdm, scalar window).
* End-to-end on a synthetic price series, cross-checked against a vectorized
  recomputation built from ``RSI.from_series`` + arctanh + pandas rolling mean.
* Mean-reversion *direction*: oversold (falling prices → low RSI) → LONG
  (positive forecast); overbought (rising prices → high RSI) → SHORT.
* Forecast-cap clamping, warmup → NaN, and forecast-cache semantics
  (``get_forecast`` None during warmup, ``is_warmed_up`` monotone).

Run from repo root:  pytest tests/test_rsimr.py -v
"""

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from event import BarEvent
from indicator import RSI
from strategy import RSIMRStrategy, Strategy
from strategy.rsimr import _RSI_DEV_CLIP, _raw_from_rsi


# ──────────────────────────────────────────────
# Test doubles  (mirror tests/test_ewmac.py)
# ──────────────────────────────────────────────

class _StepwiseDataHandler:
    """Reveals one bar at a time from a pre-built daily DataFrame.

    ``advance(i)`` exposes ``frame.iloc[: i + 1]``; the strategy reads
    ``get_latest_bars(symbol, n, timeframe='1d')`` and gets a window ending at
    the current step.
    """

    def __init__(self, frame: pd.DataFrame, symbol: str, exec_tf: str = '1d'):
        self._frame = frame
        self._symbol = symbol
        self._exec_tf = exec_tf
        self._end = 0  # exclusive upper bound

    def advance(self, end: int) -> None:
        self._end = end

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> list:
        if symbol != self._symbol:
            return []
        if timeframe is not None and timeframe != self._exec_tf:
            return []
        slice_ = self._frame.iloc[: self._end].tail(n)
        return [_BarRow(timestamp=ts, close=float(row['Close']))
                for ts, row in slice_.iterrows()]


class _BarRow:
    """Lightweight stand-in exposing ``.timestamp`` + ``.close``."""

    def __init__(self, timestamp, close: float):
        self.timestamp = timestamp
        self.close = close


def _bar_event(symbol: str, ts, close: float) -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=ts,
        open=close, high=close, low=close, close=close, volume=1.0,
        period='1d', is_forming=False,
    )


def _build_frame(closes: List[float], start: datetime) -> pd.DataFrame:
    """Build a daily OHLCV frame from a list of closes (1-day spacing)."""
    idx = pd.DatetimeIndex(
        [start + timedelta(days=i) for i in range(len(closes))]
    )
    return pd.DataFrame({
        'Open': closes,
        'High': closes,
        'Low': closes,
        'Close': closes,
        'Volume': [1.0] * len(closes),
    }, index=idx)


def _drive(strat: RSIMRStrategy, dh: _StepwiseDataHandler,
           frame: pd.DataFrame, symbol: str) -> None:
    """Step through every row of ``frame``, advancing the data handler and
    delivering one BarEvent per step. Mirrors what the engine does."""
    for i in range(len(frame)):
        dh.advance(i + 1)
        ts = frame.index[i]
        close = float(frame['Close'].iloc[i])
        strat.update_bar(_bar_event(symbol, ts, close))


# ──────────────────────────────────────────────
# Synthetic price series
# ──────────────────────────────────────────────

def _oscillating_closes(n: int) -> List[float]:
    """Mean-reverting series — a sine wave plus small noise. Keeps RSI in a
    moderate band (away from 0/100), so the arctanh mapping stays
    well-conditioned for the bar-for-bar cross-check."""
    rng = np.random.default_rng(seed=123)
    closes = []
    for i in range(n):
        closes.append(100.0 + 10.0 * np.sin(i * 2 * np.pi / 20.0)
                      + rng.normal(scale=0.5))
    return closes


def _declining_closes(n: int) -> List[float]:
    """Strictly falling series — drives RSI to 0 (deeply oversold)."""
    return [100.0 - 0.5 * i for i in range(n)]


def _rising_closes(n: int) -> List[float]:
    """Strictly rising series — drives RSI to 100 (deeply overbought)."""
    return [100.0 + 0.5 * i for i in range(n)]


# ──────────────────────────────────────────────
# The RSI → forecast mapping
# ──────────────────────────────────────────────

def test_raw_mapping_anchor_points():
    """RSI 50 → 0; the extremes map to large finite values of the
    mean-reversion sign (oversold positive, overbought negative)."""
    assert _raw_from_rsi(50.0) == pytest.approx(0.0, abs=1e-12)
    assert _raw_from_rsi(0.0) > 0.0      # oversold → LONG
    assert _raw_from_rsi(100.0) < 0.0    # overbought → SHORT
    assert np.isfinite(_raw_from_rsi(0.0))
    assert np.isfinite(_raw_from_rsi(100.0))


def test_raw_mapping_is_monotone_decreasing():
    """Higher RSI → lower (more short) raw forecast — the mean-reversion sign."""
    vals = [_raw_from_rsi(r) for r in (0.0, 25.0, 50.0, 75.0, 100.0)]
    assert all(earlier > later for earlier, later in zip(vals, vals[1:]))


def test_raw_mapping_is_antisymmetric_about_fifty():
    """``raw(50 - x) == -raw(50 + x)`` — symmetric fade of equal-distance
    oversold/overbought readings."""
    for x in (5.0, 17.0, 40.0, 49.0):
        assert _raw_from_rsi(50.0 - x) == pytest.approx(-_raw_from_rsi(50.0 + x),
                                                        abs=1e-12)


def test_raw_mapping_clips_deviation_so_extremes_are_bounded():
    """The deviation clip keeps the extreme raw value finite and equal to
    ``atanh(1 - eps)`` — never ``inf`` — even at exactly RSI 0 / 100."""
    import math
    expected = math.atanh(1.0 - _RSI_DEV_CLIP)
    assert _raw_from_rsi(0.0) == pytest.approx(expected, abs=1e-12)
    assert _raw_from_rsi(100.0) == pytest.approx(-expected, abs=1e-12)


# ──────────────────────────────────────────────
# Parameter validation
# ──────────────────────────────────────────────

def _make(symbol='BTC', **kwargs) -> RSIMRStrategy:
    dh = _StepwiseDataHandler(pd.DataFrame(), symbol)
    return RSIMRStrategy(dh, [symbol], **kwargs)


def test_default_rsi_windows_are_fast_medium_slow_trio():
    strat = _make()
    assert strat.rsi_windows == [7, 14, 28]


def test_default_weights_equal_one_third():
    strat = _make()
    assert len(strat.weights) == 3
    assert all(abs(w - 1.0 / 3.0) < 1e-12 for w in strat.weights)


def test_default_fdm_is_one():
    strat = _make()
    assert strat.fdm == 1.0


def test_empty_rsi_windows_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _make(rsi_windows=[])


def test_rsi_window_below_two_rejected():
    with pytest.raises(ValueError, match="RSI window"):
        _make(rsi_windows=[1])


def test_duplicate_rsi_windows_rejected():
    with pytest.raises(ValueError, match="distinct"):
        _make(rsi_windows=[14, 14])


def test_weights_wrong_length_rejected():
    with pytest.raises(ValueError, match="length"):
        _make(rsi_windows=[7, 14], weights=[1.0])


def test_weights_not_summing_to_one_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        _make(rsi_windows=[7, 14], weights=[0.4, 0.4])


@pytest.mark.parametrize("bad_fdm", [0.0, -0.5])
def test_fdm_non_positive_rejected(bad_fdm):
    with pytest.raises(ValueError, match="fdm"):
        _make(fdm=bad_fdm)


def test_forecast_scalar_lookback_below_two_rejected():
    with pytest.raises(ValueError, match="forecast_scalar_lookback"):
        _make(forecast_scalar_lookback=1)


# ──────────────────────────────────────────────
# Vectorized cross-check
# ──────────────────────────────────────────────

def _vectorized_rsimr(
    closes: pd.Series,
    *,
    rsi_windows,
    weights,
    forecast_scalar_lookback,
    fdm: float = 1.0,
) -> pd.Series:
    """Vectorized recomputation matching ``RSIMRStrategy``'s per-bar pipeline.

    The strategy feeds RSI the close on its very first bar (RSI tracks its own
    diffs internally), so the RSI series runs on the full ``closes`` — there is
    no leading offset (unlike EWMAC, which needs a 2-bar price change). The
    per-variation cap mirrors the strategy code (applied before the weighted
    combine); the final ``±FORECAST_CAP`` clamp mirrors ``Strategy.update_bar``.
    """
    lim = 1.0 - _RSI_DEV_CLIP
    per_var = []
    for w, _ in zip(rsi_windows, weights):
        rsi = RSI.from_series(closes, window=w)
        dev = ((rsi - 50.0) / 50.0).clip(-lim, lim)
        raw = -np.arctanh(dev)                              # mean-reversion sign
        abs_mean = raw.abs().rolling(forecast_scalar_lookback).mean()
        scalar = (
            Strategy.TARGET_AVG_ABS_FORECAST / abs_mean.where(abs_mean != 0)
        )
        per_var.append(
            (raw * scalar).clip(-Strategy.FORECAST_CAP, Strategy.FORECAST_CAP)
        )

    combined = fdm * sum(w * f for w, f in zip(weights, per_var))
    return combined.clip(-Strategy.FORECAST_CAP, Strategy.FORECAST_CAP)


def test_recorded_forecast_matches_vectorized_recomputation():
    """End-to-end: drive the stateful strategy through a synthetic series and
    confirm each recorded forecast equals the vectorized recomputation at the
    *previous* bar (the strategy's ``.latest`` lag). The series is mean-reverting
    so RSI stays moderate — keeping arctanh well-conditioned — and the tolerance
    is 1e-7 (looser than EWMAC's 1e-9 because arctanh steeply amplifies tiny RSI
    differences near the extremes)."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    n = 250
    rsi_windows = [7, 14]
    weights = [0.5, 0.5]
    forecast_scalar_lookback = 30

    closes = _oscillating_closes(n)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=rsi_windows,
        weights=weights,
        forecast_scalar_lookback=forecast_scalar_lookback,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    expected = _vectorized_rsimr(
        frame['Close'],
        rsi_windows=rsi_windows,
        weights=weights,
        forecast_scalar_lookback=forecast_scalar_lookback,
    )

    # Strategy's ``.latest`` reads at bar i correspond to vectorized values at
    # bar i-1; align by shifting the expected series forward by one.
    expected_aligned = expected.shift(1)

    valid = recorded.notna() & expected_aligned.notna()
    assert valid.sum() > 0, "no overlapping non-NaN values to compare"
    np.testing.assert_allclose(
        recorded[valid].to_numpy(),
        expected_aligned[valid].to_numpy(),
        atol=1e-7,
    )


# ──────────────────────────────────────────────
# Mean-reversion direction
# ──────────────────────────────────────────────

def test_oversold_market_gets_long_forecast():
    """A persistent decline drives RSI toward 0 (deeply oversold); the
    mean-reverter must produce a POSITIVE (long) forecast — buy the dip."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _declining_closes(80)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[4, 8], weights=[0.5, 0.5],
        forecast_scalar_lookback=10,
    )
    _drive(strat, dh, frame, symbol)

    forecast = strat.get_forecast(symbol)
    assert forecast is not None
    assert forecast > 0.0, f"oversold should be long, got {forecast}"


def test_overbought_market_gets_short_forecast():
    """A persistent rise drives RSI toward 100 (deeply overbought); the
    mean-reverter must produce a NEGATIVE (short) forecast — fade the rally."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _rising_closes(80)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[4, 8], weights=[0.5, 0.5],
        forecast_scalar_lookback=10,
    )
    _drive(strat, dh, frame, symbol)

    forecast = strat.get_forecast(symbol)
    assert forecast is not None
    assert forecast < 0.0, f"overbought should be short, got {forecast}"


# ──────────────────────────────────────────────
# FDM scaling
# ──────────────────────────────────────────────

def test_fdm_scales_combined_forecast():
    """Driving the same series with ``fdm=2.0`` produces combined forecasts
    that are 2× the ``fdm=1.0`` run on every bar where neither run hits the
    ±FORECAST_CAP clamp (the clamp breaks the proportionality)."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(250)
    frame = _build_frame(closes, start)
    rsi_windows = [7, 14]
    weights = [0.5, 0.5]

    def _run(fdm_value: float) -> pd.Series:
        dh = _StepwiseDataHandler(frame, symbol)
        strat = RSIMRStrategy(
            dh, [symbol],
            rsi_windows=rsi_windows, weights=weights, fdm=fdm_value,
            forecast_scalar_lookback=30,
        )
        _drive(strat, dh, frame, symbol)
        return strat.get_records(symbol)['forecast']

    base = _run(1.0)
    doubled = _run(2.0)

    cap = Strategy.FORECAST_CAP
    not_clipped = (
        base.notna() & doubled.notna()
        & (base.abs() < cap - 1e-9)
        & (doubled.abs() < cap - 1e-9)
    )
    assert not_clipped.sum() > 0, "no overlapping non-clipped values to compare"
    np.testing.assert_allclose(
        doubled[not_clipped].to_numpy(),
        2.0 * base[not_clipped].to_numpy(),
        atol=1e-7,
    )


def test_fdm_column_recorded_in_records():
    """Every strategy-populated row carries the configured ``fdm``."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(60)
    frame = _build_frame(closes, start)

    fdm_value = 1.7
    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[4], weights=[1.0], fdm=fdm_value,
        forecast_scalar_lookback=10,
    )
    _drive(strat, dh, frame, symbol)

    records = strat.get_records(symbol)
    assert 'fdm' in records.columns
    populated = records['fdm'].dropna()
    # The first bar has no prior close for a diff → RSI NaN, but the strategy
    # still returns a row carrying ``fdm`` on every processed bar.
    assert len(populated) == len(records)
    assert (populated == fdm_value).all()


# ──────────────────────────────────────────────
# Clamping
# ──────────────────────────────────────────────

def test_forecast_caps_at_plus_minus_one_hundred():
    """A regime shift (quiet mean-reverting phase → sudden strong trend) makes
    the current |raw| jump far above its trailing average, so the dynamic
    scalar pushes the scaled forecast past ±100. Verify it is clamped and
    that the fixture actually drives at least one bar to the cap."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    rng = np.random.default_rng(seed=7)
    # Phase A: quiet oscillation around a flat level → small |raw|, small
    # trailing abs-mean. Phase B: steep sustained decline → RSI slams to 0,
    # |raw| spikes far above the trailing mean → forecast blows past ±100.
    n_a, n_b = 120, 80
    closes = [100.0 + rng.normal(scale=0.1) for _ in range(n_a)]
    base = closes[-1]
    for _ in range(n_b):
        base -= 5.0
        closes.append(base)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[4], weights=[1.0],
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast'].dropna()
    assert (recorded.abs() <= Strategy.FORECAST_CAP + 1e-9).all()
    assert (recorded.abs() >= Strategy.FORECAST_CAP - 1e-9).any(), (
        "fixture did not stress the cap — no bar reached ±100"
    )


def test_per_variation_forecast_columns_present_and_capped():
    """Each ``forecast_<w>`` column exists and individually respects
    ``±Strategy.FORECAST_CAP`` (the cap is applied per variation before the
    weighted combine)."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    rng = np.random.default_rng(seed=11)
    n_a, n_b = 120, 80
    closes = [100.0 + rng.normal(scale=0.1) for _ in range(n_a)]
    base = closes[-1]
    for _ in range(n_b):
        base -= 5.0
        closes.append(base)
    frame = _build_frame(closes, start)

    rsi_windows = [4, 8]
    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=rsi_windows, weights=[0.5, 0.5],
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    records = strat.get_records(symbol)
    per_var_cols = [f'forecast_{w}' for w in rsi_windows]
    for col in per_var_cols:
        assert col in records.columns
        series = records[col].dropna()
        assert (series.abs() <= Strategy.FORECAST_CAP + 1e-9).all(), (
            f"{col} exceeded ±{Strategy.FORECAST_CAP}: max abs = {series.abs().max()}"
        )
    assert any(
        (records[c].dropna().abs() >= Strategy.FORECAST_CAP - 1e-9).any()
        for c in per_var_cols
    )


# ──────────────────────────────────────────────
# Warmup + forecast cache
# ──────────────────────────────────────────────

def test_recorded_forecast_is_nan_during_warmup():
    """No forecast before the dynamic-scalar SMA window has filled."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(60)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[4, 8], weights=[0.5, 0.5],
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    # First non-NaN forecast needs ~ max_window (8) + scalar_lookback (20) bars;
    # the first 25 bars MUST be NaN.
    assert recorded.iloc[:25].isna().all()


def test_strategy_has_no_events_queue_attribute():
    """Strategy takes no events_queue; the attribute is absent."""
    strat = _make(rsi_windows=[4], weights=[1.0], forecast_scalar_lookback=10)
    assert not hasattr(strat, 'events_queue')


def test_cached_forecast_matches_last_recorded_non_nan_forecast():
    """``get_forecast`` reflects the latest non-NaN recorded forecast."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(250)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[7], weights=[1.0],
        forecast_scalar_lookback=30,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    last_valid = recorded.dropna().iloc[-1]
    assert strat.get_forecast(symbol) == pytest.approx(last_valid, abs=1e-9)


def test_cached_forecast_is_none_during_warmup():
    """Before any non-NaN forecast, the cached forecast stays at None."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(20)  # too short to fill the 30-window SMA
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[7], weights=[1.0],
        forecast_scalar_lookback=30,
    )
    _drive(strat, dh, frame, symbol)
    assert strat.get_forecast(symbol) is None


def test_is_warmed_up_flips_and_is_monotone():
    """``is_warmed_up`` is False during warmup, flips True on the first non-NaN
    forecast, and never resets."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _oscillating_closes(250)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = RSIMRStrategy(
        dh, [symbol],
        rsi_windows=[7], weights=[1.0],
        forecast_scalar_lookback=30,
    )
    assert strat.is_warmed_up(symbol) is False

    seen_true = False
    for i in range(len(frame)):
        dh.advance(i + 1)
        ts = frame.index[i]
        close = float(frame['Close'].iloc[i])
        strat.update_bar(_bar_event(symbol, ts, close))
        if strat.is_warmed_up(symbol):
            seen_true = True
        # Monotone: once True, never reverts.
        if seen_true:
            assert strat.is_warmed_up(symbol), "is_warmed_up reverted after True"
    assert seen_true, "strategy never warmed up over the full series"
