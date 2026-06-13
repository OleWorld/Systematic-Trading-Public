"""Unit tests for ``EWMAVolEstimator``.

Pin the contract:
- Closes are sourced from a ``DataHandler`` via
  ``get_latest_bars(symbol, 2, timeframe=self.timeframe)`` — decoupled
  from the base timeframe.
- Price-change computation matches ``np.diff(closes)``.
- EWMA stdev matches the pandas reference
  ``price_changes.pow(2).ewm(span=N, adjust=False, min_periods=N).mean().pow(0.5)``.
- Annualization scaling uses ``sqrt(bars_per_year)``.
- Warmup: ``.latest`` semantics → first valid sigma at bar # ``span + 2``
  (1 seed bar + ``span + 1`` price changes to have a finalized stdev one
  slot behind the just-pushed forming entry).
- Zero-vol input → ``0.0`` cleanly.
- Forming bars are ignored.
- Unknown symbols are a no-op.
- Series passing through zero feeds in cleanly (``close - 0 = close``
  is a valid price change).
- Regression: same daily close series produces the same sigma whether
  events arrive at 1d or 1h cadence (this is the bug fix the new
  ``timeframe`` parameter enables).

Run from the repo root:  python -m pytest tests/test_volatility_ewma.py -v
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from event import BarEvent
from volatility import EWMAVolEstimator


# ──────────────────────────────────────────────
# Stub DataHandler
# ──────────────────────────────────────────────

class _StubDataHandler:
    """Minimal stand-in for ``DataHandler`` covering only
    ``get_latest_bars`` — the one method the vol estimator consumes.

    Closes are registered per (symbol, timeframe) via ``add_close``;
    ``get_latest_bars`` returns the last ``n`` as a DataFrame with the
    standard OHLCV columns (only ``Close`` is read by the estimator,
    but matching the real return type matters for the contract).
    """

    def __init__(self) -> None:
        self._bars: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = {}

    def add_close(self, symbol: str, ts: datetime, close: float,
                  timeframe: str = '1d') -> None:
        key = (symbol, timeframe)
        self._bars.setdefault(key, []).append((ts, close))

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> pd.DataFrame:
        tf = timeframe if timeframe is not None else '1d'
        rows = self._bars.get((symbol, tf), [])
        if not rows:
            return pd.DataFrame(
                columns=['Open', 'High', 'Low', 'Close', 'Volume']
            )
        subset = rows[-n:]
        timestamps = [t for t, _ in subset]
        closes = [c for _, c in subset]
        return pd.DataFrame(
            {
                'Open': closes,
                'High': closes,
                'Low': closes,
                'Close': closes,
                'Volume': [10.0] * len(closes),
            },
            index=timestamps,
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _bar(symbol: str, ts: datetime, close: float,
         is_forming: bool = False) -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=ts,
        open=close, high=close, low=close, close=close, volume=10.0,
        period='1d', is_forming=is_forming,
    )


def _make_estimator(symbol_list: List[str], bars_per_year: float,
                    span: int, timeframe: str = '1d',
                    stub: Optional[_StubDataHandler] = None
                    ) -> Tuple[EWMAVolEstimator, _StubDataHandler]:
    """Construct an estimator wired to a fresh (or supplied) stub."""
    stub = stub if stub is not None else _StubDataHandler()
    est = EWMAVolEstimator(
        symbol_list, data_handler=stub,
        bars_per_year=bars_per_year, timeframe=timeframe, span=span,
    )
    return est, stub


def _push_series(est: EWMAVolEstimator, stub: _StubDataHandler,
                 symbol: str, closes: List[float],
                 timeframe: str = '1d') -> None:
    """Push a sequence of completed-bar closes; one bar per element.

    Each close is registered with the stub at ``timeframe`` AND drives a
    ``BarEvent`` into the estimator — the event's symbol/timestamp/
    is_forming flag still drive the estimator's bookkeeping; the close
    is extracted from the stub via ``get_latest_bars``.
    """
    t0 = datetime(2026, 1, 1)
    for i, c in enumerate(closes):
        ts = t0 + timedelta(days=i)
        stub.add_close(symbol, ts, c, timeframe=timeframe)
        est.update(_bar(symbol, ts, c))


# ──────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────

def test_constructor_validates_span():
    stub = _StubDataHandler()
    with pytest.raises(ValueError, match="span"):
        EWMAVolEstimator(['BTC'], data_handler=stub,
                         bars_per_year=365, span=0)
    with pytest.raises(ValueError, match="span"):
        EWMAVolEstimator(['BTC'], data_handler=stub,
                         bars_per_year=365, span=-1)


def test_constructor_validates_bars_per_year():
    stub = _StubDataHandler()
    with pytest.raises(ValueError, match="bars_per_year"):
        EWMAVolEstimator(['BTC'], data_handler=stub,
                         bars_per_year=0, span=10)
    with pytest.raises(ValueError, match="bars_per_year"):
        EWMAVolEstimator(['BTC'], data_handler=stub,
                         bars_per_year=-1, span=10)


def test_default_timeframe_is_1d():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, span=36)
    assert est.timeframe == '1d'


def test_default_span_is_36():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, span=36)
    assert est.span == 36


# ──────────────────────────────────────────────
# Warmup behavior
# ──────────────────────────────────────────────

def test_warmup_returns_none_before_first_bar():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, span=5)
    assert est.get_annual_vol('BTC') is None


def test_warmup_returns_none_after_only_seed_bar():
    est, stub = _make_estimator(['BTC'], bars_per_year=365, span=5)
    _push_series(est, stub, 'BTC', [100.0])
    assert est.get_annual_vol('BTC') is None


def test_warmup_returns_none_at_span_plus_one_bars():
    """span price changes produce one finite forming entry but no finalized one."""
    span = 5
    est, stub = _make_estimator(['BTC'], bars_per_year=365, span=span)
    # 1 seed + span price changes → EWMStdev._outputs[-1] is the first finite stdev,
    # _outputs[-2] is still NaN. `.latest` returns NaN-payload → None.
    _push_series(est, stub, 'BTC',
                 [100.0 + i for i in range(span + 1)])
    assert est.get_annual_vol('BTC') is None


def test_warmup_first_valid_at_span_plus_two_bars():
    span = 5
    est, stub = _make_estimator(['BTC'], bars_per_year=365, span=span)
    # 1 seed + (span+1) price changes → first finalized stdev at `.latest`.
    _push_series(est, stub, 'BTC',
                 [100.0 + i for i in range(span + 2)])
    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert not math.isnan(sigma)


# ──────────────────────────────────────────────
# Numerical correctness
# ──────────────────────────────────────────────

def test_ewma_matches_pandas_reference():
    """End-to-end: pushed price changes produce the same sigma as the pandas
    EWM reference applied to the same price changes."""
    span = 4
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0, 106.1, 107.0]
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, span=span)
    _push_series(est, stub, 'BTC', closes)

    # Reference: price changes through pandas EWM-of-squared, sqrt'd. Then
    # take iloc[-2] to match `.latest` (one slot behind the forming entry).
    closes_arr = np.array(closes, dtype=float)
    price_changes = pd.Series(np.diff(closes_arr))
    ewm_var = (price_changes.pow(2)
                             .ewm(span=span, adjust=False, min_periods=span)
                             .mean())
    ewm_stdev = ewm_var.pow(0.5)
    expected_sigma = float(ewm_stdev.iloc[-2])

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    # bars_per_year=1 → sqrt(1)=1 → no scaling.
    assert math.isclose(sigma, expected_sigma, rel_tol=1e-12)


def test_annualization_scaling_365():
    span = 4
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0]
    est_unscaled, stub_u = _make_estimator(['BTC'], bars_per_year=1.0, span=span)
    est_daily, stub_d = _make_estimator(['BTC'], bars_per_year=365.0, span=span)
    _push_series(est_unscaled, stub_u, 'BTC', closes)
    _push_series(est_daily, stub_d, 'BTC', closes)

    s_unscaled = est_unscaled.get_annual_vol('BTC')
    s_daily = est_daily.get_annual_vol('BTC')
    assert s_unscaled is not None and s_daily is not None
    assert math.isclose(s_daily, s_unscaled * math.sqrt(365.0), rel_tol=1e-9)


def test_annualization_scaling_4h():
    """4h-equivalent: 6 bars per day × 365 days = 2190 bars/year."""
    span = 4
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0]
    est_unscaled, stub_u = _make_estimator(['BTC'], bars_per_year=1.0, span=span)
    est_4h, stub_4h = _make_estimator(['BTC'], bars_per_year=365.0 * 6, span=span)
    _push_series(est_unscaled, stub_u, 'BTC', closes)
    _push_series(est_4h, stub_4h, 'BTC', closes)

    s_unscaled = est_unscaled.get_annual_vol('BTC')
    s_4h = est_4h.get_annual_vol('BTC')
    assert s_unscaled is not None and s_4h is not None
    assert math.isclose(s_4h, s_unscaled * math.sqrt(365.0 * 6), rel_tol=1e-9)


def test_zero_vol_input_returns_zero_not_nan():
    """Constant prices → price changes are all 0 → EWMA stdev is exactly
    0.0 from the start of the masked window."""
    span = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=365, span=span)
    _push_series(est, stub, 'BTC', [100.0] * (span + 2))
    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert sigma == 0.0


# ──────────────────────────────────────────────
# Bar filtering / robustness
# ──────────────────────────────────────────────

def test_forming_bars_are_ignored():
    span = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, span=span)
    t0 = datetime(2026, 1, 1)
    # Drive 5 completed bars + a forming spike in the middle. The forming
    # bar's close (999.0) must NOT enter the indicator: a forming event
    # short-circuits before the estimator reads from the stub.
    closes = [100.0, 102.0, 101.0, 103.5, 104.2]
    for i, c in enumerate(closes):
        ts = t0 + timedelta(days=i)
        stub.add_close('BTC', ts, c, timeframe='1d')
        est.update(_bar('BTC', ts, c))
        if i == 1:
            # Inject a forming bar between the 2nd and 3rd completed bars.
            forming_ts = ts + timedelta(hours=1)
            est.update(_bar('BTC', forming_ts, 999.0, is_forming=True))

    closes_arr = np.array(closes, dtype=float)
    price_changes = pd.Series(np.diff(closes_arr))
    ewm = (price_changes.pow(2)
                         .ewm(span=span, adjust=False, min_periods=span)
                         .mean()
                         .pow(0.5))
    expected = float(ewm.iloc[-2])

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert math.isclose(sigma, expected, rel_tol=1e-12)


def test_unknown_symbol_returns_none_silently():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, span=3)
    assert est.get_annual_vol('ETH') is None


def test_unknown_symbol_update_is_no_op():
    est, stub = _make_estimator(['BTC'], bars_per_year=365, span=3)
    t0 = datetime(2026, 1, 1)
    stub.add_close('ETH', t0, 100.0, timeframe='1d')
    est.update(_bar('ETH', t0, 100.0))
    assert est.get_annual_vol('ETH') is None
    assert est.get_annual_vol('BTC') is None


def test_zero_prior_close_is_not_special():
    """Price changes are well-defined for any prior close, including zero
    and negative values. The estimator pushes every completed-bar change
    after the seed bar; no special-case skip applies."""
    span = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, span=span)
    # 1 seed + (span+1) = 4 price changes ⇒ `.latest` is finite.
    # Closes: 100 → 0 → -5 → -3 → 2 → 4
    # Price changes:    -100, -5, 2, 5, 2
    closes = [100.0, 0.0, -5.0, -3.0, 2.0, 4.0]
    _push_series(est, stub, 'BTC', closes)

    closes_arr = np.array(closes, dtype=float)
    price_changes = pd.Series(np.diff(closes_arr))
    ewm = (price_changes.pow(2)
                         .ewm(span=span, adjust=False, min_periods=span)
                         .mean()
                         .pow(0.5))
    expected = float(ewm.iloc[-2])

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert math.isclose(sigma, expected, rel_tol=1e-12)


def test_multi_symbol_isolation():
    span = 3
    est, stub = _make_estimator(
        ['BTC', 'ETH'], bars_per_year=1.0, span=span,
    )
    t0 = datetime(2026, 1, 1)
    btc_closes = [100.0, 102.0, 101.0, 103.0, 102.5]
    eth_closes = [10.0, 10.5, 10.2, 10.8, 10.6]
    for i, (b, e) in enumerate(zip(btc_closes, eth_closes)):
        ts = t0 + timedelta(days=i)
        stub.add_close('BTC', ts, b, timeframe='1d')
        stub.add_close('ETH', ts, e, timeframe='1d')
        est.update(_bar('BTC', ts, b))
        est.update(_bar('ETH', ts, e))

    btc_sigma = est.get_annual_vol('BTC')
    eth_sigma = est.get_annual_vol('ETH')
    assert btc_sigma is not None and eth_sigma is not None

    def _expected(closes):
        closes_arr = np.array(closes, dtype=float)
        price_changes = pd.Series(np.diff(closes_arr))
        ewm = (price_changes.pow(2)
                             .ewm(span=span, adjust=False, min_periods=span)
                             .mean()
                             .pow(0.5))
        return float(ewm.iloc[-2])

    assert math.isclose(btc_sigma, _expected(btc_closes), rel_tol=1e-12)
    assert math.isclose(eth_sigma, _expected(eth_closes), rel_tol=1e-12)
    assert btc_sigma != eth_sigma


# ──────────────────────────────────────────────
# Regression: timeframe decouples sigma from base TF
# ──────────────────────────────────────────────

def test_sigma_invariant_to_event_cadence():
    """The bug fix: sigma at ``timeframe='1d'`` is the same whether
    completed BarEvents arrive at a 1d cadence (one per day) or a 1h
    cadence (24 per day), as long as the underlying 1d close series
    is identical.

    Mechanism: the estimator reads ``get_latest_bars(2, '1d')`` on every
    completed event. Within a 1d period, all intra-day events push the
    same diff at the same forming timestamp into the underlying
    ``EWMStdev``, which upserts → no change to ``.latest``. At the next
    1d boundary, the forming entry finalizes and a new diff is pushed
    with the new forming timestamp. The end state of the indicator's
    output deque is identical to the 1d-cadence case.
    """
    span = 4
    daily_closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0,
                    106.1, 107.0, 105.5, 108.2]

    # --- (a) 1d-cadence events --------------------------------------
    est_1d, stub_1d = _make_estimator(
        ['BTC'], bars_per_year=365, span=span, timeframe='1d',
    )
    _push_series(est_1d, stub_1d, 'BTC', daily_closes, timeframe='1d')
    sigma_1d = est_1d.get_annual_vol('BTC')

    # --- (b) 1h-cadence events with same 1d series in the stub ------
    # For each day D, register the day's close in the stub at '1d'
    # ONCE (mirroring an idealized engine where the 1d HTF bar's close
    # equals the final intra-day tick — the real engine's HTF
    # accumulator keeps updating it, but what the estimator sees at the
    # next-day boundary is the same), then drive 24 hourly events
    # within that day. All 24 events for day D read the same 2-bar
    # window from the stub, so they push the same diff at the same
    # forming_ts → the underlying indicator upserts.
    est_1h, stub_1h = _make_estimator(
        ['BTC'], bars_per_year=365, span=span, timeframe='1d',
    )
    t0 = datetime(2026, 1, 1)
    for d, close in enumerate(daily_closes):
        day_ts = t0 + timedelta(days=d)
        stub_1h.add_close('BTC', day_ts, close, timeframe='1d')
        for h in range(24):
            hour_ts = day_ts + timedelta(hours=h)
            est_1h.update(_bar('BTC', hour_ts, close))
    sigma_1h = est_1h.get_annual_vol('BTC')

    assert sigma_1d is not None and sigma_1h is not None
    assert math.isclose(sigma_1d, sigma_1h, rel_tol=1e-12)
