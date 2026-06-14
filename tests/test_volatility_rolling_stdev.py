"""Unit tests for ``RollingStdevVolEstimator``.

Pin the contract:
- Closes are sourced from a ``DataHandler`` via
  ``get_latest_bars(symbol, 2, timeframe=self.timeframe)`` — decoupled
  from the base timeframe.
- Price-change computation matches ``np.diff(prices)``.
- Annualization scaling uses ``sqrt(bars_per_year)`` for both daily-equiv
  (365) and 4h-equiv (365 * 6) factors.
- Warmup: ``get_annual_vol`` returns ``None`` until the first finalized
  non-NaN stdev is available. Because the estimator reads ``stdev.latest``
  (one slot behind the just-pushed forming entry), warmup needs one
  additional change beyond the rolling-window length: 1 seed bar + (lookback
  + 1) completed-bar changes ⇒ first valid at bar # ``lookback + 2``.
- Sigma reflects the previous completed bar's finalized stdev (one-bar
  lag relative to the just-pushed change) — ``.latest`` semantics,
  matching how strategies read indicator outputs.
- Zero-vol input → ``0.0`` cleanly (no NaN, no divide-by-zero).
- Forming bars are ignored.
- Symbols outside the configured list are ignored without error.
- Series passing through zero feeds in cleanly (``close - 0 = close``
  is a valid price change).
- Regression: same daily close series produces the same sigma whether
  events arrive at 1d or 1h cadence (this is the bug fix the new
  ``timeframe`` parameter enables).

Run from the repo root:  python -m pytest tests/test_volatility_rolling_stdev.py -v
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from event import BarEvent
from volatility import RollingStdevVolEstimator


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
                    lookback: int, timeframe: str = '1d',
                    stub: Optional[_StubDataHandler] = None
                    ) -> Tuple[RollingStdevVolEstimator, _StubDataHandler]:
    """Construct an estimator wired to a fresh (or supplied) stub."""
    stub = stub if stub is not None else _StubDataHandler()
    est = RollingStdevVolEstimator(
        symbol_list, data_handler=stub,
        bars_per_year=bars_per_year, timeframe=timeframe, lookback=lookback,
    )
    return est, stub


def _push_series(est: RollingStdevVolEstimator, stub: _StubDataHandler,
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

def test_constructor_validates_lookback():
    stub = _StubDataHandler()
    with pytest.raises(ValueError, match="lookback"):
        RollingStdevVolEstimator(['BTC'], data_handler=stub,
                                 bars_per_year=365, lookback=1)


def test_constructor_validates_bars_per_year():
    stub = _StubDataHandler()
    with pytest.raises(ValueError, match="bars_per_year"):
        RollingStdevVolEstimator(['BTC'], data_handler=stub,
                                 bars_per_year=0, lookback=10)
    with pytest.raises(ValueError, match="bars_per_year"):
        RollingStdevVolEstimator(['BTC'], data_handler=stub,
                                 bars_per_year=-1, lookback=10)


def test_default_timeframe_is_1d():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, lookback=10)
    assert est.timeframe == '1d'


# ──────────────────────────────────────────────
# Warmup behavior
# ──────────────────────────────────────────────

def test_warmup_returns_none_before_first_bar():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, lookback=5)
    assert est.get_annual_vol('BTC') is None


def test_warmup_returns_none_after_only_seed_bar():
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=5)
    _push_series(est, stub, 'BTC', [100.0])
    # First bar just seeds prior_close; no price change computed yet.
    assert est.get_annual_vol('BTC') is None


def test_warmup_returns_none_until_lookback_changes_observed():
    lookback = 5
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=lookback)
    # Bar 1 seeds; bars 2..lookback give lookback-1 price changes → still NaN inside Stdev.
    _push_series(est, stub, 'BTC',
                 [100.0 + i for i in range(lookback)])
    assert est.get_annual_vol('BTC') is None


def test_warmup_returns_none_at_lookback_plus_one_bars():
    """``stdev.forming`` would be finite here, but we read ``.latest`` —
    the second-to-last finalized entry — which is still NaN."""
    lookback = 5
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=lookback)
    # Bar 1 seeds; bars 2..lookback+1 give lookback price changes.
    # Stdev._outputs[-1] is the first finite stdev; _outputs[-2] is still NaN.
    # `.latest` returns _outputs[-2] → None for a NaN payload.
    _push_series(est, stub, 'BTC',
                 [100.0 + i for i in range(lookback + 1)])
    assert est.get_annual_vol('BTC') is None


def test_warmup_first_valid_at_lookback_plus_two_bars():
    lookback = 5
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=lookback)
    # Bar 1 seeds; bars 2..lookback+2 give lookback+1 price changes → Stdev
    # has two finite entries, so `.latest` (= _outputs[-2]) is finite.
    _push_series(est, stub, 'BTC',
                 [100.0 + i for i in range(lookback + 2)])
    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert not math.isnan(sigma)


# ──────────────────────────────────────────────
# Numerical correctness
# ──────────────────────────────────────────────

def test_price_change_matches_numpy():
    """Stdev fed from update matches np.std on the trailing-lookback
    window ending one change ago (``.latest`` semantics — see module
    docstring)."""
    lookback = 4
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0, 106.1]
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, lookback=lookback)
    _push_series(est, stub, 'BTC', closes)

    closes_arr = np.array(closes, dtype=float)
    price_changes = np.diff(closes_arr)
    # `.latest` skips the most recent change: stdev over price_changes[-lookback-1:-1].
    expected_sigma = float(np.std(price_changes[-lookback - 1:-1], ddof=1))

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    # bars_per_year=1 → sqrt(1)=1 → no scaling.
    assert math.isclose(sigma, expected_sigma, rel_tol=1e-9)


def test_annualization_scaling_365():
    lookback = 4
    # lookback+2 closes to clear `.latest` warmup (1 seed + lookback+1 changes).
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0]
    est_unscaled, stub_u = _make_estimator(['BTC'], bars_per_year=1.0,
                                           lookback=lookback)
    est_daily, stub_d = _make_estimator(['BTC'], bars_per_year=365.0,
                                        lookback=lookback)
    _push_series(est_unscaled, stub_u, 'BTC', closes)
    _push_series(est_daily, stub_d, 'BTC', closes)

    s_unscaled = est_unscaled.get_annual_vol('BTC')
    s_daily = est_daily.get_annual_vol('BTC')
    assert s_unscaled is not None and s_daily is not None
    assert math.isclose(s_daily, s_unscaled * math.sqrt(365.0), rel_tol=1e-9)


def test_annualization_scaling_4h():
    """4h-equivalent: 6 bars per day × 365 days = 2190 bars/year."""
    lookback = 4
    closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0]
    est_unscaled, stub_u = _make_estimator(['BTC'], bars_per_year=1.0,
                                           lookback=lookback)
    est_4h, stub_4h = _make_estimator(['BTC'], bars_per_year=365.0 * 6,
                                      lookback=lookback)
    _push_series(est_unscaled, stub_u, 'BTC', closes)
    _push_series(est_4h, stub_4h, 'BTC', closes)

    s_unscaled = est_unscaled.get_annual_vol('BTC')
    s_4h = est_4h.get_annual_vol('BTC')
    assert s_unscaled is not None and s_4h is not None
    assert math.isclose(s_4h, s_unscaled * math.sqrt(365.0 * 6), rel_tol=1e-9)


def test_zero_vol_input_returns_zero_not_nan():
    """Constant prices → price changes are all 0 → stdev is exactly 0.0, not NaN."""
    lookback = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=lookback)
    # lookback+2 bars to clear `.latest` warmup.
    _push_series(est, stub, 'BTC', [100.0] * (lookback + 2))
    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert sigma == 0.0


# ──────────────────────────────────────────────
# Bar filtering / robustness
# ──────────────────────────────────────────────

def test_forming_bars_are_ignored():
    lookback = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, lookback=lookback)
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
            forming_ts = ts + timedelta(hours=1)
            est.update(_bar('BTC', forming_ts, 999.0, is_forming=True))

    closes_arr = np.array(closes, dtype=float)
    price_changes = np.diff(closes_arr)
    expected = float(np.std(price_changes[-lookback - 1:-1], ddof=1))

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert math.isclose(sigma, expected, rel_tol=1e-9)


def test_unknown_symbol_returns_none_silently():
    est, _ = _make_estimator(['BTC'], bars_per_year=365, lookback=3)
    assert est.get_annual_vol('ETH') is None


def test_unknown_symbol_update_is_no_op():
    est, stub = _make_estimator(['BTC'], bars_per_year=365, lookback=3)
    t0 = datetime(2026, 1, 1)
    # No exception, no state change for ETH (not in symbol_list).
    stub.add_close('ETH', t0, 100.0, timeframe='1d')
    est.update(_bar('ETH', t0, 100.0))
    assert est.get_annual_vol('ETH') is None
    assert est.get_annual_vol('BTC') is None


def test_zero_prior_close_is_not_special():
    """Price changes are well-defined for any prior close, including zero
    and negative values. The estimator pushes every completed-bar change
    after the seed bar; no special-case skip applies."""
    lookback = 3
    est, stub = _make_estimator(['BTC'], bars_per_year=1.0, lookback=lookback)
    # 1 seed + (lookback+1) = 5 closes ⇒ `.latest` is finite.
    closes = [100.0, 0.0, -5.0, -3.0, 2.0]
    _push_series(est, stub, 'BTC', closes)

    closes_arr = np.array(closes, dtype=float)
    price_changes = np.diff(closes_arr)
    expected = float(np.std(price_changes[-lookback - 1:-1], ddof=1))

    sigma = est.get_annual_vol('BTC')
    assert sigma is not None
    assert math.isclose(sigma, expected, rel_tol=1e-9)


def test_multi_symbol_isolation():
    lookback = 3
    est, stub = _make_estimator(
        ['BTC', 'ETH'], bars_per_year=1.0, lookback=lookback,
    )
    t0 = datetime(2026, 1, 1)
    # lookback+2 = 5 closes per symbol so `.latest` is finite for each.
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

    btc_arr = np.array(btc_closes, dtype=float)
    eth_arr = np.array(eth_closes, dtype=float)
    btc_pc = np.diff(btc_arr)
    eth_pc = np.diff(eth_arr)
    # `.latest` skips the most recent change.
    btc_expected = float(np.std(btc_pc[-lookback - 1:-1], ddof=1))
    eth_expected = float(np.std(eth_pc[-lookback - 1:-1], ddof=1))

    assert math.isclose(btc_sigma, btc_expected, rel_tol=1e-9)
    assert math.isclose(eth_sigma, eth_expected, rel_tol=1e-9)
    assert btc_sigma != eth_sigma                # state isolated per symbol


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
    ``Stdev``, which upserts → no change to ``.latest``. At the next 1d
    boundary, the forming entry finalizes and a new diff is pushed with
    the new forming timestamp. The end state of the indicator's output
    deque is identical to the 1d-cadence case.
    """
    lookback = 4
    daily_closes = [100.0, 102.0, 101.0, 103.5, 105.2, 104.0,
                    106.1, 107.0, 105.5, 108.2]

    # --- (a) 1d-cadence events --------------------------------------
    est_1d, stub_1d = _make_estimator(
        ['BTC'], bars_per_year=365, lookback=lookback, timeframe='1d',
    )
    _push_series(est_1d, stub_1d, 'BTC', daily_closes, timeframe='1d')
    sigma_1d = est_1d.get_annual_vol('BTC')

    # --- (b) 1h-cadence events with same 1d series in the stub ------
    est_1h, stub_1h = _make_estimator(
        ['BTC'], bars_per_year=365, lookback=lookback, timeframe='1d',
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
