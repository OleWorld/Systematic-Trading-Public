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
``VolTargetingRiskManager`` and this strategy read it from there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from event import BarEvent
from indicator import EMA, SMA, Stdev
from strategy._base import Strategy, _DataHandlerLike

__all__ = ['EWMACStrategy']


# Default variations as a ``{label: {param_name: value}}`` dict. The labels
# are param-derived so a default-constructed strategy emits diagnostic columns
# reading ``fast_ema_16_64`` etc. (identical to the pre-refactor output).
_DEFAULT_VARIATIONS: Dict[str, Dict[str, int]] = {
    '16_64': {'fast': 16, 'slow': 64},
    '32_128': {'fast': 32, 'slow': 128},
    '64_256': {'fast': 64, 'slow': 256},
}

# Required per-variation param keys — validated by strict set-equality so a
# typo (``'fats'``) fails loudly instead of silently NaN-ing the variation.
_PARAM_KEYS = frozenset({'fast', 'slow'})


class EWMACStrategy(Strategy):
    """EWMAC trend-follower with three look-back variations and a dynamic
    forecast scalar.

    Per-symbol state (per-variation indicators keyed by variation label):
        ``_stdev``                     — shared price-change Stdev (vol_lookback).
        ``_fast_emas[label]``          — fast EMA for variation ``label``.
        ``_slow_emas[label]``          — slow EMA for variation ``label``.
        ``_abs_vol_adj_smas[label]``   — rolling mean of ``|vol_adj_label|`` over
                                          ``forecast_scalar_lookback`` finalized
                                          bars.
    """

    def __init__(
        self,
        data_handler: _DataHandlerLike,
        symbol_list: List[str],
        execution_timeframe: str = '1d',
        variations: Optional[Dict[str, Dict[str, int]]] = None,
        weights: Optional[Dict[str, float]] = None,
        fdm: float = 1.0,
        vol_lookback: int = 25,
        forecast_scalar_lookback: int = 256,
    ):
        """Configure parameters and instantiate per-symbol indicators.

        Parameters
        ----------
        execution_timeframe
            Bar timeframe the EWMAC chain runs on (default ``'1d'``).
        variations
            Mapping ``{label: {'fast': L_fast, 'slow': L_slow}}`` of the
            look-back variations, in bars of ``execution_timeframe``. The
            label is an opaque string keying both the combination weights and
            the per-bar diagnostic columns (``fast_ema_<label>`` …), mirroring
            the orchestrator's ``{label: strategy}`` shape; the nested
            ``{param_name: value}`` params dict generalizes to indicators with
            more parameters. Each params dict must have exactly the keys
            ``{'fast', 'slow'}`` with ``L_fast >= 2`` and ``L_slow > L_fast``.
            Default ``{'16_64': {'fast': 16, 'slow': 64}, '32_128': {'fast':
            32, 'slow': 128}, '64_256': {'fast': 64, 'slow': 256}}`` —
            Carver's "three slowest" set; the param-derived labels keep the
            default diagnostic columns reading ``fast_ema_16_64`` etc.
        weights
            Optional ``{label: weight}`` per-variation combination weights.
            Keys must match ``variations`` exactly; values must be finite,
            non-negative, and sum to ``1.0``. ``None`` (default) → equal
            weight ``1/N`` via ``analytics.equal_weight``.
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
            Default ``256``.
        

        The target average absolute forecast that the per-variation
        scaling drives toward is ``Strategy.TARGET_AVG_ABS_FORECAST``
        (project-wide constant, default ``50.0``).
        """
        super().__init__(data_handler, symbol_list)

        # Shared weight validator (analytics.equal_weight default). Imported
        # lazily so the strategy import path doesn't pull in cvxpy.
        from analytics import validate_named_weights

        if variations is None:
            variations = {
                label: dict(params)
                for label, params in _DEFAULT_VARIATIONS.items()
            }
        if not variations:
            raise ValueError("variations must be a non-empty dict")
        for label, params in variations.items():
            if not isinstance(params, dict) or set(params) != _PARAM_KEYS:
                raise ValueError(
                    f"variation {label!r} params must be a dict with keys "
                    f"{sorted(_PARAM_KEYS)}, got {params!r}"
                )
            fast, slow = params['fast'], params['slow']
            if fast < 2:
                raise ValueError(
                    f"variation {label!r}: fast must be >= 2 (so EMA has more "
                    f"than one input), got {fast}"
                )
            if slow <= fast:
                raise ValueError(
                    f"variation {label!r}: slow must be > fast, got "
                    f"fast={fast} slow={slow}"
                )

        # Keys set-equal to the variation labels, finite, non-negative,
        # sum→1; ``None`` → equal weight (shared with the orchestrator).
        weights = validate_named_weights(weights, list(variations.keys()))

        if fdm <= 0:
            raise ValueError(f"fdm must be > 0, got {fdm}")

        if vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {vol_lookback}")
        if forecast_scalar_lookback < 2:
            raise ValueError(
                f"forecast_scalar_lookback must be >= 2, got {forecast_scalar_lookback}"
            )

        self.execution_timeframe = execution_timeframe
        self.variations: Dict[str, Dict[str, int]] = {
            label: dict(params) for label, params in variations.items()
        }
        self.weights: Dict[str, float] = weights
        self.fdm = fdm
        self.vol_lookback = vol_lookback
        self.forecast_scalar_lookback = forecast_scalar_lookback

        # Shared price-change stdev per symbol.
        self._stdev: Dict[str, Stdev] = {
            s: Stdev(length=vol_lookback) for s in symbol_list
        }

        # Per-symbol per-variation indicators, keyed by variation label.
        self._fast_emas: Dict[str, Dict[str, EMA]] = {
            s: {label: EMA(span=p['fast'])
                for label, p in self.variations.items()}
            for s in symbol_list
        }
        self._slow_emas: Dict[str, Dict[str, EMA]] = {
            s: {label: EMA(span=p['slow'])
                for label, p in self.variations.items()}
            for s in symbol_list
        }
        # SMA(window) keeps inputs_maxlen=window. outputs_maxlen needs to
        # be at least 2 so .latest (= _outputs[-2]) is reachable.
        self._abs_vol_adj_smas: Dict[str, Dict[str, SMA]] = {
            s: {label: SMA(window=forecast_scalar_lookback)
                for label in self.variations}
            for s in symbol_list
        }

    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        """Update indicators from the latest forming execution-TF bar and
        compute the combined forecast (NaN during warmup).

        Returns a dict with:

        * ``'forecast'`` — combined capped forecast (possibly NaN).
        * ``'price_stdev'`` — shared price-change Stdev (latest finalized).
        * Per variation ``label``:
          ``'fast_ema_<label>'``, ``'slow_ema_<label>'``,
          ``'vol_adj_<label>'``, ``'abs_mean_<label>'``,
          ``'scalar_<label>'`` (latest finalized intermediates), and
          ``'forecast_<label>'`` — the per-variation forecast
          **capped at ±Strategy.FORECAST_CAP** before combining.

        The base class clamps the top-level forecast to
        ``±Strategy.FORECAST_CAP`` and writes it to ``self.forecasts[symbol]``
        (NaN values are recorded but skipped — the cached forecast stays at
        its prior value).
        """
        symbol = event.symbol

        # ── Latest 1d bars: forming close + previous finalized close. ───
        # List[Bar] straight from the deque (no DataFrame) — this is the
        # per-bar hot path; ``bar.close`` is already a float.
        daily_lookback = self.data_handler.get_latest_bars(
            symbol, 2, timeframe=self.execution_timeframe,
        )
        if len(daily_lookback) < 2:
            return None

        forming_ts = daily_lookback[-1].timestamp
        forming_close = daily_lookback[-1].close
        prev_close = daily_lookback[-2].close
        price_change = forming_close - prev_close

        # ── Update shared price-change Stdev. ───────────────────────────
        stdev = self._stdev[symbol]
        stdev.update(forming_ts, price_change)

        # ── Per-variation pipeline: update EMAs, push forming vol_adj into
        # the abs-mean SMA, then compute the forecast and intermediates from
        # finalized values. ─────────────────────────────────────────────
        nan = float('nan')
        per_var_forecasts: Dict[str, float] = {l: nan for l in self.variations}
        per_var_fast: Dict[str, float] = {l: nan for l in self.variations}
        per_var_slow: Dict[str, float] = {l: nan for l in self.variations}
        per_var_vol_adj: Dict[str, float] = {l: nan for l in self.variations}
        per_var_abs_mean: Dict[str, float] = {l: nan for l in self.variations}
        per_var_scalar: Dict[str, float] = {l: nan for l in self.variations}

        for label in self.variations:
            fast_ema = self._fast_emas[symbol][label]
            slow_ema = self._slow_emas[symbol][label]
            abs_sma = self._abs_vol_adj_smas[symbol][label]

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
                    per_var_fast[label] = fast_latest
                    per_var_slow[label] = slow_latest
                    per_var_vol_adj[label] = vol_adj_latest
                    per_var_abs_mean[label] = abs_mean_latest
                    per_var_scalar[label] = scalar
                    # Cap each variation individually at ``±FORECAST_CAP``
                    # before combining — keeps one runaway variation from
                    # dominating the weighted sum and keeps the recorded
                    # ``forecast_<label>`` diagnostic on the same
                    # ``[-100, +100]`` scale as the combined ``forecast``.
                    cap = Strategy.FORECAST_CAP
                    per_var_forecasts[label] = max(
                        -cap, min(cap, vol_adj_latest * scalar)
                    )

        # ── Combine per-variation forecasts. ────────────────────────────
        # If ANY variation isn't ready, the combined forecast is undefined.
        # We don't backfill missing variations from finite ones because the
        # weights are calibrated against the full ensemble. ``Strategy.update_bar``
        # clamps the final value to ``±Strategy.FORECAST_CAP`` before storing
        # it in ``self.forecasts[symbol]``, so no clamp is applied here.
        if any(pd.isna(per_var_forecasts[l]) for l in self.variations):
            final_forecast = nan
        else:
            final_forecast = self.fdm * sum(
                self.weights[l] * per_var_forecasts[l] for l in self.variations
            )

        # ── Recorded fields. ────────────────────────────────────────────
        std_l = stdev.latest
        price_stdev = (
            float(std_l['stdev'])
            if std_l is not None and not pd.isna(std_l['stdev'])
            else nan
        )

        row: Dict[str, Any] = {
            'forecast': final_forecast,
            'fdm': self.fdm,
            'price_stdev': price_stdev,
        }
        for label in self.variations:
            row[f'fast_ema_{label}'] = per_var_fast[label]
            row[f'slow_ema_{label}'] = per_var_slow[label]
            row[f'vol_adj_{label}'] = per_var_vol_adj[label]
            row[f'abs_mean_{label}'] = per_var_abs_mean[label]
            row[f'scalar_{label}'] = per_var_scalar[label]
            row[f'forecast_{label}'] = per_var_forecasts[label]
        return row
