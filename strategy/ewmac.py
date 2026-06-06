"""
Carver's EWMAC trend-following rule (Systematic Trading, appendix B,
p.282) — three look-back variations combined into a single weighted
forecast in ``[-Strategy.FORECAST_CAP, +Strategy.FORECAST_CAP]``.

Per variation ``i`` (look-backs in *daily bars*; default execution
timeframe is ``1d`` so the numbers map 1:1 to Carver's days):

    fast_ema_i  = EMA(close, span=L_fast_i)
    slow_ema_i  = EMA(close, span=L_slow_i)
    raw_xover_i = fast_ema_i - slow_ema_i

    price_stdev = Stdev(close.diff(), length=vol_lookback)   # shared
    vol_adj_i   = raw_xover_i / price_stdev

    # Dynamic forecast scalar (replaces Carver's fixed table 49).
    abs_mean_i  = SMA(|vol_adj_i|, window=forecast_scalar_lookback)
    scalar_i    = Strategy.TARGET_AVG_ABS_FORECAST / abs_mean_i
    scaled_i    = clip(vol_adj_i * scalar_i, ±Strategy.FORECAST_CAP)
                                            # per-variation cap applied
                                            # before combination — keeps
                                            # any single runaway variation
                                            # from dominating the sum.

Combined forecast (default equal weights = 1/N each):

    combined        = fdm * sum(w_i * scaled_i)    # ``fdm`` is Carver's
                                                   # Forecast Diversification
                                                   # Multiplier — scales the
                                                   # combined forecast back up
                                                   # to compensate for
                                                   # diversification-driven
                                                   # cancellation. Default
                                                   # ``1.0`` (no adjustment).
    final_forecast  = combined                     # also clamped to
                                                   # ±Strategy.FORECAST_CAP by
                                                   # ``Strategy.update_bar`` as
                                                   # a safety net.

Output: per-bar dict with the ``'forecast'`` key. The base class clamps
to ``±Strategy.FORECAST_CAP`` and writes to ``self.forecasts[symbol]``;
the risk manager reads it on every completed bar to derive the target
position. Forming bars compute the same forecast (it depends on
finalized values via ``Indicator.latest``), so the cached forecast is
stable across forming ticks within a period and only changes at period
boundaries.

The ``±Strategy.FORECAST_CAP`` scale (vs Carver's ``±20``) is the
project-wide forecast convention — see ``Strategy.update_bar``. The
target-average-absolute-forecast magnitude lives on the same class as
``Strategy.TARGET_AVG_ABS_FORECAST`` (default ``50.0``); both
``CarverVolTargetingRiskManager`` and this strategy read it from there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from event import BarEvent
from indicator import EMA, SMA, Stdev
from strategy._base import Strategy, _DataHandlerLike

__all__ = ['EWMACStrategy']


_DEFAULT_LOOKBACK_PAIRS: List[Tuple[int, int]] = [(16, 64), (32, 128), (64, 256)]


class EWMACStrategy(Strategy):
    """EWMAC trend-follower with three look-back variations and a dynamic
    forecast scalar.

    Per-symbol state:
        ``_stdev``                  — shared price-change Stdev (vol_lookback).
        ``_fast_emas[i]``           — fast EMA for variation ``i``.
        ``_slow_emas[i]``           — slow EMA for variation ``i``.
        ``_abs_vol_adj_smas[i]``    — rolling mean of ``|vol_adj_i|`` over
                                       ``forecast_scalar_lookback`` finalized
                                       bars.
    """

    def __init__(
        self,
        data_handler: _DataHandlerLike,
        symbol_list: List[str],
        execution_timeframe: str = '1d',
        lookback_pairs: Optional[List[Tuple[int, int]]] = None,
        weights: Optional[List[float]] = None,
        fdm: float = 1.0,
        vol_lookback: int = 25,
        forecast_scalar_lookback: int = 500,
    ):
        """Configure parameters and instantiate per-symbol indicators.

        Parameters
        ----------
        execution_timeframe
            Bar timeframe the EWMAC chain runs on (default ``'1d'``).
        lookback_pairs
            List of ``(L_fast, L_slow)`` tuples in bars of
            ``execution_timeframe``. Default ``[(16, 64), (32, 128), (64, 256)]``
            — Carver's "three slowest" set, suitable for most instruments.
            Each pair must have ``L_fast >= 2`` and ``L_slow > L_fast``.
        weights
            Per-variation combination weights. Must have the same length as
            ``lookback_pairs`` and sum to 1.0. ``None`` (default) → equal
            weights.
        fdm
            Forecast Diversification Multiplier (Carver, *Systematic
            Trading* ch. 8 / *Advanced Futures Trading Strategies* ch. 4).
            Multiplied into the combined weighted forecast to scale it
            back up after diversification-driven cancellation: combining
            imperfectly-correlated variations lowers the average absolute
            forecast below ``TARGET_AVG_ABS_FORECAST``, and FDM
            (typically ≥ 1, derived from the variations' correlation
            structure) restores the target. Must be strictly positive.
            Default ``1.0`` — no adjustment, preserves the raw weighted
            sum.
        vol_lookback
            Window for the rolling sample stdev of price changes. Default
            ``25`` — Carver's *Advanced Futures Trading Strategies*
            equal-weight stdev of daily price changes.
        forecast_scalar_lookback
            Window for the rolling mean of ``|vol_adj|`` that feeds the
            dynamic forecast scalar (``target / mean(|vol_adj|)``).
            Default ``500``.
        

        The target average absolute forecast that the per-variation
        scaling drives toward is ``Strategy.TARGET_AVG_ABS_FORECAST``
        (project-wide constant, default ``50.0``).
        """
        super().__init__(data_handler, symbol_list)

        if lookback_pairs is None:
            lookback_pairs = list(_DEFAULT_LOOKBACK_PAIRS)
        if not lookback_pairs:
            raise ValueError("lookback_pairs must be non-empty")
        for fast, slow in lookback_pairs:
            if fast < 2:
                raise ValueError(
                    f"L_fast must be >= 2 (so EMA has more than one input), got {fast}"
                )
            if slow <= fast:
                raise ValueError(
                    f"L_slow must be > L_fast, got fast={fast} slow={slow}"
                )

        n_vars = len(lookback_pairs)
        if weights is None:
            weights = [1.0 / n_vars] * n_vars
        else:
            if len(weights) != n_vars:
                raise ValueError(
                    f"weights length {len(weights)} != lookback_pairs length {n_vars}"
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(
                    f"weights must sum to 1.0, got sum={sum(weights)}"
                )

        if fdm <= 0:
            raise ValueError(f"fdm must be > 0, got {fdm}")

        if vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {vol_lookback}")
        if forecast_scalar_lookback < 2:
            raise ValueError(
                f"forecast_scalar_lookback must be >= 2, got {forecast_scalar_lookback}"
            )

        self.execution_timeframe = execution_timeframe
        self.lookback_pairs: List[Tuple[int, int]] = list(lookback_pairs)
        self.weights: List[float] = list(weights)
        self.fdm = fdm
        self.vol_lookback = vol_lookback
        self.forecast_scalar_lookback = forecast_scalar_lookback

        # Shared price-change stdev per symbol.
        self._stdev: Dict[str, Stdev] = {
            s: Stdev(length=vol_lookback) for s in symbol_list
        }

        # Per-symbol per-variation indicators.
        self._fast_emas: Dict[str, List[EMA]] = {
            s: [EMA(span=fast) for fast, _ in self.lookback_pairs]
            for s in symbol_list
        }
        self._slow_emas: Dict[str, List[EMA]] = {
            s: [EMA(span=slow) for _, slow in self.lookback_pairs]
            for s in symbol_list
        }
        # SMA(window) keeps inputs_maxlen=window. outputs_maxlen needs to
        # be at least 2 so .latest (= _outputs[-2]) is reachable.
        self._abs_vol_adj_smas: Dict[str, List[SMA]] = {
            s: [SMA(window=forecast_scalar_lookback)
                for _ in self.lookback_pairs]
            for s in symbol_list
        }

    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        """Update indicators from the latest forming execution-TF bar and
        compute the combined forecast (NaN during warmup).

        Returns a dict with:

        * ``'forecast'`` — combined capped forecast (possibly NaN).
        * ``'price_stdev'`` — shared price-change Stdev (latest finalized).
        * Per variation ``(fast, slow)``:
          ``'fast_ema_<fast>_<slow>'``, ``'slow_ema_<fast>_<slow>'``,
          ``'vol_adj_<fast>_<slow>'``, ``'abs_mean_<fast>_<slow>'``,
          ``'scalar_<fast>_<slow>'`` (latest finalized intermediates), and
          ``'forecast_<fast>_<slow>'`` — the per-variation forecast
          **capped at ±Strategy.FORECAST_CAP** before combining.

        The base class clamps the top-level forecast to
        ``±Strategy.FORECAST_CAP`` and writes it to ``self.forecasts[symbol]``
        (NaN values are recorded but skipped — the cached forecast stays at
        its prior value).
        """
        symbol = event.symbol

        # ── Latest 1d bars: forming close + previous finalized close. ───
        daily_lookback = self.data_handler.get_latest_bars(
            symbol, 2, timeframe=self.execution_timeframe,
        )
        if len(daily_lookback) < 2:
            return None

        forming_ts = daily_lookback.index[-1]
        forming_close = float(daily_lookback['Close'].iloc[-1])
        prev_close = float(daily_lookback['Close'].iloc[-2])
        price_change = forming_close - prev_close

        # ── Update shared price-change Stdev. ───────────────────────────
        stdev = self._stdev[symbol]
        stdev.update(forming_ts, price_change)

        # ── Per-variation pipeline: update EMAs, push forming vol_adj into
        # the abs-mean SMA, then compute the forecast and intermediates from
        # finalized values. ─────────────────────────────────────────────
        n_vars = len(self.lookback_pairs)
        per_var_forecasts: List[float] = [float('nan')] * n_vars
        per_var_fast: List[float] = [float('nan')] * n_vars
        per_var_slow: List[float] = [float('nan')] * n_vars
        per_var_vol_adj: List[float] = [float('nan')] * n_vars
        per_var_abs_mean: List[float] = [float('nan')] * n_vars
        per_var_scalar: List[float] = [float('nan')] * n_vars

        for i in range(n_vars):
            fast_ema = self._fast_emas[symbol][i]
            slow_ema = self._slow_emas[symbol][i]
            abs_sma = self._abs_vol_adj_smas[symbol][i]

            fast_ema.update(forming_ts, forming_close)
            slow_ema.update(forming_ts, forming_close)

            # Push |vol_adj_forming| into the rolling-mean SMA so it
            # advances on every bar (forming or completed). Skip when any
            # forming input is undefined or the forming stdev is zero —
            # feeding NaN would propagate into the SMA window.
            if (fast_ema.is_forming_ready and slow_ema.is_forming_ready
                    and stdev.is_forming_ready):
                fast_f = fast_ema.forming
                slow_f = slow_ema.forming
                std_f = stdev.forming
                assert (fast_f is not None and slow_f is not None
                        and std_f is not None)  # is_forming_ready ⇒ non-None
                stdev_forming = float(std_f['stdev'])
                if stdev_forming != 0.0:
                    fast_forming = float(fast_f['ema'])
                    slow_forming = float(slow_f['ema'])
                    vol_adj_forming = (
                        (fast_forming - slow_forming) / stdev_forming
                    )
                    abs_sma.update(forming_ts, abs(vol_adj_forming))

            # Compute per-variation forecast (and recorded intermediates)
            # from finalized values.
            if (fast_ema.is_latest_ready and slow_ema.is_latest_ready
                    and stdev.is_latest_ready and abs_sma.is_latest_ready):
                fast_l = fast_ema.latest
                slow_l = slow_ema.latest
                std_l = stdev.latest
                abs_l = abs_sma.latest
                assert (fast_l is not None and slow_l is not None
                        and std_l is not None and abs_l is not None)  # is_latest_ready ⇒ non-None
                fast_latest = float(fast_l['ema'])
                slow_latest = float(slow_l['ema'])
                stdev_latest = float(std_l['stdev'])
                abs_mean_latest = float(abs_l['sma'])
                if stdev_latest != 0.0 and abs_mean_latest != 0.0:
                    vol_adj_latest = (
                        (fast_latest - slow_latest) / stdev_latest
                    )
                    scalar = (
                        Strategy.TARGET_AVG_ABS_FORECAST / abs_mean_latest
                    )
                    per_var_fast[i] = fast_latest
                    per_var_slow[i] = slow_latest
                    per_var_vol_adj[i] = vol_adj_latest
                    per_var_abs_mean[i] = abs_mean_latest
                    per_var_scalar[i] = scalar
                    # Cap each variation individually at ``±FORECAST_CAP``
                    # before combining — keeps one runaway variation from
                    # dominating the weighted sum and keeps the recorded
                    # ``forecast_<fast>_<slow>`` diagnostic on the same
                    # ``[-100, +100]`` scale as the combined ``forecast``.
                    cap = Strategy.FORECAST_CAP
                    per_var_forecasts[i] = max(
                        -cap, min(cap, vol_adj_latest * scalar)
                    )

        # ── Combine per-variation forecasts. ────────────────────────────
        # If ANY variation isn't ready, the combined forecast is undefined.
        # We don't backfill missing variations from finite ones because the
        # weights are calibrated against the full ensemble. ``Strategy.update_bar``
        # clamps the final value to ``±Strategy.FORECAST_CAP`` before storing
        # it in ``self.forecasts[symbol]``, so no clamp is applied here.
        if any(pd.isna(f) for f in per_var_forecasts):
            final_forecast = float('nan')
        else:
            final_forecast = self.fdm * sum(
                w * f for w, f in zip(self.weights, per_var_forecasts)
            )

        # ── Recorded fields. ────────────────────────────────────────────
        std_l = stdev.latest
        price_stdev = (
            float(std_l['stdev'])
            if std_l is not None and not pd.isna(std_l['stdev'])
            else float('nan')
        )

        row: Dict[str, Any] = {
            'forecast': final_forecast,
            'fdm': self.fdm,
            'price_stdev': price_stdev,
        }
        for i, (fast, slow) in enumerate(self.lookback_pairs):
            row[f'fast_ema_{fast}_{slow}'] = per_var_fast[i]
            row[f'slow_ema_{fast}_{slow}'] = per_var_slow[i]
            row[f'vol_adj_{fast}_{slow}'] = per_var_vol_adj[i]
            row[f'abs_mean_{fast}_{slow}'] = per_var_abs_mean[i]
            row[f'scalar_{fast}_{slow}'] = per_var_scalar[i]
            row[f'forecast_{fast}_{slow}'] = per_var_forecasts[i]
        return row
