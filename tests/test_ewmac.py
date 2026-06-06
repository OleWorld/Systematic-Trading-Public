"""
Unit + integration tests for ``EWMACStrategy``.

Covers:
* Parameter validation (lookback pairs, weights, vol/forecast windows).
* End-to-end on a synthetic price series, cross-checked against a
  vectorized recomputation built from ``EMA.from_series`` +
  pandas rolling stdev/mean.
* Forecast-cache update per bar: ``self.forecasts[symbol]`` matching
  the recorded forecast.

After the post-2026-05 redesign, ``EWMACStrategy`` no longer emits
SignalEvents — it writes the per-bar forecast into ``self.forecasts``,
which the risk manager reads on every completed bar.

Run from repo root:  pytest tests/test_ewmac.py -v
"""

from datetime import datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from event import BarEvent
from indicator import EMA
from strategy import EWMACStrategy, Strategy


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class _StepwiseDataHandler:
    """Reveals one bar at a time from a pre-built daily DataFrame.

    ``advance(i)`` exposes ``frame.iloc[: i + 1]``; the strategy reads
    ``get_latest_bars(symbol, n, timeframe='1d')`` and gets a window
    ending at the current step.
    """

    def __init__(self, frame: pd.DataFrame, symbol: str, exec_tf: str = '1d'):
        self._frame = frame
        self._symbol = symbol
        self._exec_tf = exec_tf
        self._end = 0  # exclusive upper bound

    def advance(self, end: int) -> None:
        self._end = end

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> pd.DataFrame:
        if symbol != self._symbol:
            return pd.DataFrame()
        if timeframe is not None and timeframe != self._exec_tf:
            return pd.DataFrame()
        slice_ = self._frame.iloc[: self._end]
        return slice_.tail(n)

    def get_latest_bar(self, symbol: str,
                       timeframe: Optional[str] = None) -> Optional[Any]:
        df = self.get_latest_bars(symbol, 1, timeframe)
        if df.empty:
            return None
        row = df.iloc[-1]
        return _BarRow(timestamp=df.index[-1], close=float(row['Close']))


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


def _drive(strat: EWMACStrategy, dh: _StepwiseDataHandler,
           frame: pd.DataFrame, symbol: str) -> None:
    """Step through every row of ``frame``, advancing the data handler and
    delivering one BarEvent per step. Mirrors what the engine does."""
    for i in range(len(frame)):
        dh.advance(i + 1)
        ts = frame.index[i]
        close = float(frame['Close'].iloc[i])
        strat.update_bar(_bar_event(symbol, ts, close))


# ──────────────────────────────────────────────
# Parameter validation
# ──────────────────────────────────────────────

def _make(symbol='BTC', **kwargs) -> EWMACStrategy:
    dh = _StepwiseDataHandler(pd.DataFrame(), symbol)
    return EWMACStrategy(dh, [symbol], **kwargs)


def test_default_lookback_pairs_are_carver_slow_trio():
    strat = _make(
        lookback_pairs=[(16, 64), (32, 128), (64, 256)],
        vol_lookback=25,
        forecast_scalar_lookback=500,
    )
    assert strat.lookback_pairs == [(16, 64), (32, 128), (64, 256)]


def test_default_weights_equal_one_third():
    strat = _make(
        lookback_pairs=[(16, 64), (32, 128), (64, 256)],
        vol_lookback=25,
        forecast_scalar_lookback=500,
    )
    assert len(strat.weights) == 3
    assert all(abs(w - 1.0 / 3.0) < 1e-12 for w in strat.weights)


def test_empty_lookback_pairs_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _make(
            lookback_pairs=[],
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_lookback_fast_less_than_two_rejected():
    with pytest.raises(ValueError, match="L_fast"):
        _make(
            lookback_pairs=[(1, 4)],
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_lookback_slow_not_greater_than_fast_rejected():
    with pytest.raises(ValueError, match="L_slow"):
        _make(
            lookback_pairs=[(8, 8)],
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_weights_wrong_length_rejected():
    with pytest.raises(ValueError, match="length"):
        _make(
            lookback_pairs=[(4, 16), (8, 32)], weights=[1.0],
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_weights_not_summing_to_one_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        _make(
            lookback_pairs=[(4, 16), (8, 32)], weights=[0.4, 0.4],
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_vol_lookback_below_two_rejected():
    with pytest.raises(ValueError, match="vol_lookback"):
        _make(
            lookback_pairs=[(16, 64), (32, 128), (64, 256)],
            vol_lookback=1,
            forecast_scalar_lookback=500,
        )


def test_forecast_scalar_lookback_below_two_rejected():
    with pytest.raises(ValueError, match="forecast_scalar_lookback"):
        _make(
            lookback_pairs=[(16, 64), (32, 128), (64, 256)],
            vol_lookback=25,
            forecast_scalar_lookback=1,
        )


def test_default_fdm_is_one():
    strat = _make(
        lookback_pairs=[(16, 64), (32, 128), (64, 256)],
        vol_lookback=25,
        forecast_scalar_lookback=500,
    )
    assert strat.fdm == 1.0


@pytest.mark.parametrize("bad_fdm", [0.0, -0.5])
def test_fdm_non_positive_rejected(bad_fdm):
    with pytest.raises(ValueError, match="fdm"):
        _make(
            lookback_pairs=[(16, 64), (32, 128), (64, 256)],
            fdm=bad_fdm,
            vol_lookback=25,
            forecast_scalar_lookback=500,
        )


def test_fdm_scales_combined_forecast():
    """Driving the same series with ``fdm=2.0`` should produce combined
    forecasts that are 2× the ``fdm=1.0`` run on every bar where neither
    run hits the ±FORECAST_CAP clamp (the clamp is a non-linear ceiling
    that breaks the proportionality)."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(200)
    frame = _build_frame(closes, start)
    lookback_pairs = [(4, 16), (8, 32), (16, 64)]
    weights = [1.0 / 3.0] * 3

    def _run(fdm_value: float) -> pd.Series:
        dh = _StepwiseDataHandler(frame, symbol)
        strat = EWMACStrategy(
            dh, [symbol],
            lookback_pairs=lookback_pairs,
            weights=weights,
            fdm=fdm_value,
            vol_lookback=10,
            forecast_scalar_lookback=20,
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
        atol=1e-9,
    )


def test_fdm_clamps_at_cap_when_combined_exceeds_bound():
    """A large ``fdm`` (well above 1.0) on a steadily trending series can
    push the FDM-multiplied weighted sum past ``±Strategy.FORECAST_CAP``.
    The safety-net clamp in ``Strategy.update_bar`` must keep the recorded
    forecast inside the bound, and the fixture must actually exercise the
    clamp (at least one bar at the cap) so the test is not a trivial pass."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(200)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16), (8, 32), (16, 64)],
        weights=[1.0 / 3.0] * 3,
        fdm=5.0,  # 5x — guaranteed to push at least some bars past ±100.
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast'].dropna()
    cap = Strategy.FORECAST_CAP
    assert (recorded.abs() <= cap + 1e-9).all(), (
        f"combined forecast exceeded ±{cap}: max abs = {recorded.abs().max()}"
    )
    assert (recorded.abs() >= cap - 1e-9).any(), (
        "fixture did not stress the safety-net clamp — no bar reached the cap"
    )


def test_fdm_column_recorded_in_records():
    """The per-bar diagnostic row must carry the strategy's ``fdm`` value
    on every recorded bar so post-hoc analysis can recover the multiplier
    that produced each forecast."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(50)
    frame = _build_frame(closes, start)

    fdm_value = 1.7
    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        fdm=fdm_value,
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    records = strat.get_records(symbol)
    assert 'fdm' in records.columns
    # The very first bar gets a fully-NaN row (no prior close yet → the
    # strategy returns ``None`` and the base class records all-NaN). Every
    # row that the strategy did populate must carry the configured fdm.
    populated = records['fdm'].dropna()
    assert len(populated) == len(records) - 1
    assert (populated == fdm_value).all()


# ──────────────────────────────────────────────
# Vectorized cross-check
# ──────────────────────────────────────────────

def _vectorized_ewmac(
    closes: pd.Series,
    *,
    lookback_pairs,
    weights,
    vol_lookback,
    forecast_scalar_lookback,
    fdm: float = 1.0,
) -> pd.Series:
    """Vectorized recomputation matching ``EWMACStrategy``'s per-bar pipeline.

    The strategy needs *two* finalized closes before it can compute a price
    change (its first processed bar is bar 1, not bar 0), so its EMAs and
    Stdev see inputs starting from bar 1. We mirror that here — fast/slow
    EMAs are seeded on ``closes.iloc[1:]`` and Stdev runs on
    ``closes.diff().iloc[1:]``. The returned Series is reindexed onto
    ``closes`` with bar 0 left as NaN. The final ``±Strategy.FORECAST_CAP``
    clamp mirrors ``Strategy.update_bar``'s base-class clamp; the dynamic
    forecast scalar reads ``Strategy.TARGET_AVG_ABS_FORECAST`` (the
    project-wide target average absolute forecast).
    """
    closes_seen = closes.iloc[1:]
    price_change_seen = closes.diff().iloc[1:]
    stdev = price_change_seen.rolling(vol_lookback).std(ddof=1)

    per_var = []
    for (fast, slow), _ in zip(lookback_pairs, weights):
        fast_ema = EMA.from_series(closes_seen, span=fast)
        slow_ema = EMA.from_series(closes_seen, span=slow)
        raw_xover = fast_ema - slow_ema
        vol_adj = raw_xover / stdev.where(stdev != 0)
        abs_mean = vol_adj.abs().rolling(forecast_scalar_lookback).mean()
        scalar = (
            Strategy.TARGET_AVG_ABS_FORECAST / abs_mean.where(abs_mean != 0)
        )
        # Per-variation cap mirrors the strategy code — applied before the
        # weighted combine, not just on the final sum.
        per_var.append(
            (vol_adj * scalar).clip(-Strategy.FORECAST_CAP, Strategy.FORECAST_CAP)
        )

    combined = fdm * sum(w * f for w, f in zip(weights, per_var))
    final_capped = combined.clip(-Strategy.FORECAST_CAP, Strategy.FORECAST_CAP)
    # Reindex so bar 0 is NaN and the rest aligns with ``closes``.
    return final_capped.reindex(closes.index)


def _trending_closes(n: int) -> List[float]:
    """Synthetic price series — a deterministic noisy uptrend that flips
    direction halfway through, so EWMAC sees both signs."""
    rng = np.random.default_rng(seed=42)
    base = 100.0
    closes = []
    for i in range(n):
        trend = 0.5 if i < n // 2 else -0.5
        base += trend + rng.normal(scale=0.2)
        closes.append(base)
    return closes


def test_recorded_forecast_matches_vectorized_recomputation():
    """End-to-end: drive the stateful strategy through a synthetic series
    and confirm each recorded forecast equals the vectorized recomputation
    at the *previous* bar (the strategy's ``.latest`` lag). Tolerance is
    tight (1e-9) — both paths use identical EMA / rolling-stdev /
    rolling-mean primitives, so this is a true bar-for-bar match."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    n = 200
    lookback_pairs = [(4, 16), (8, 32), (16, 64)]
    weights = [1.0 / 3.0] * 3
    vol_lookback = 10
    forecast_scalar_lookback = 20

    closes = _trending_closes(n)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=lookback_pairs,
        weights=weights,
        vol_lookback=vol_lookback,
        forecast_scalar_lookback=forecast_scalar_lookback,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    expected = _vectorized_ewmac(
        frame['Close'],
        lookback_pairs=lookback_pairs,
        weights=weights,
        vol_lookback=vol_lookback,
        forecast_scalar_lookback=forecast_scalar_lookback,
    )

    # Strategy's ``.latest`` reads at bar i correspond to vectorized values
    # at bar i-1; align by shifting the expected series forward by one.
    expected_aligned = expected.shift(1)

    valid = recorded.notna() & expected_aligned.notna()
    assert valid.sum() > 0, "no overlapping non-NaN values to compare"
    np.testing.assert_allclose(
        recorded[valid].to_numpy(),
        expected_aligned[valid].to_numpy(),
        atol=1e-9,
    )


def test_recorded_forecast_is_nan_during_warmup():
    """No forecast is produced before the rolling SMA window has filled."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(50)  # short series; can't fill 20-window SMA
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    # The first ~36 bars (16 slow EMA + 20 SMA window) MUST be NaN.
    assert recorded.iloc[:30].isna().all()


def test_forecast_caps_at_plus_minus_one_hundred():
    """A regime shift (slow drift → sudden acceleration) produces a vol_adj
    spike before the rolling abs-mean catches up, so the dynamic scalar
    pushes the scaled forecast above the cap. Verify it gets clamped at
    ±100 and never exceeds it."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    rng = np.random.default_rng(seed=7)
    # Phase A: tiny drift (anchors the rolling abs-mean at a small value).
    # Phase B: sudden 100x acceleration — vol_adj jumps far above the
    # rolling abs-mean, so the scaled forecast blows through ±100 and
    # gets clamped.
    n_a, n_b = 120, 80
    base = 100.0
    closes = []
    for i in range(n_a):
        base += 0.05 + rng.normal(scale=0.01)
        closes.append(base)
    for i in range(n_b):
        base += 5.0 + rng.normal(scale=0.01)
        closes.append(base)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast'].dropna()
    # Hard guarantee: nothing above the cap.
    assert (recorded.abs() <= 100.0 + 1e-9).all()
    # The regime shift must have driven at least one bar to the cap.
    assert (recorded.abs() >= 100.0 - 1e-9).any()


def test_per_variation_forecast_caps_at_plus_minus_one_hundred():
    """Each per-variation forecast column (``forecast_<fast>_<slow>``) must
    individually respect ``±Strategy.FORECAST_CAP`` — the cap is applied
    per variation before the weighted combine, so no single variation can
    blow through the bound even on a regime shift that maximally stresses
    the dynamic scalar."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    rng = np.random.default_rng(seed=11)
    # Same regime-shift recipe as test_forecast_caps_at_plus_minus_one_hundred,
    # tuned to push both variations through their dynamic scalars hard.
    n_a, n_b = 120, 80
    base = 100.0
    closes = []
    for i in range(n_a):
        base += 0.05 + rng.normal(scale=0.01)
        closes.append(base)
    for i in range(n_b):
        base += 5.0 + rng.normal(scale=0.01)
        closes.append(base)
    frame = _build_frame(closes, start)

    # Multiple variations so the per-variation columns are distinguishable
    # from the combined ``forecast`` column.
    lookback_pairs = [(4, 16), (8, 32)]
    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=lookback_pairs,
        weights=[0.5, 0.5],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    records = strat.get_records(symbol)
    per_var_cols = [f'forecast_{f}_{s}' for f, s in lookback_pairs]
    for col in per_var_cols:
        series = records[col].dropna()
        assert (series.abs() <= Strategy.FORECAST_CAP + 1e-9).all(), (
            f"{col} exceeded ±{Strategy.FORECAST_CAP}: "
            f"max abs = {series.abs().max()}"
        )
    # And at least one per-variation column actually hit the cap, proving
    # the fixture exercises the bound (not a trivial pass).
    assert any(
        (records[c].dropna().abs() >= Strategy.FORECAST_CAP - 1e-9).any()
        for c in per_var_cols
    )


# ──────────────────────────────────────────────
# Forecast cache (replaces the old signal-emission tests)
# ──────────────────────────────────────────────

def test_strategy_has_no_events_queue_attribute():
    """Strategy no longer accepts an events_queue; the attribute is gone."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(200)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)
    assert not hasattr(strat, 'events_queue')


def test_cached_forecast_matches_last_recorded_non_nan_forecast():
    """``strategy.get_forecast(symbol)`` reflects the latest non-NaN
    recorded forecast — the value the risk manager would read on the
    final bar."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    closes = _trending_closes(200)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)

    recorded = strat.get_records(symbol)['forecast']
    last_valid = recorded.dropna().iloc[-1]
    assert strat.get_forecast(symbol) == pytest.approx(last_valid, abs=1e-9)


def test_cached_forecast_starts_at_zero_during_warmup():
    """Before any non-NaN forecast is computed, the cached forecast stays
    at the initial 0.0 (NaN forecasts don't update the cache)."""
    symbol = 'BTC'
    start = datetime(2026, 1, 1)
    # Short series — can't fill the SMA window, so all forecasts remain NaN.
    closes = _trending_closes(20)
    frame = _build_frame(closes, start)

    dh = _StepwiseDataHandler(frame, symbol)
    strat = EWMACStrategy(
        dh, [symbol],
        lookback_pairs=[(4, 16)],
        weights=[1.0],
        vol_lookback=10,
        forecast_scalar_lookback=20,
    )
    _drive(strat, dh, frame, symbol)
    assert strat.get_forecast(symbol) == 0.0
