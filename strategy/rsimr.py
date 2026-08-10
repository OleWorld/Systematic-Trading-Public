"""
RSI mean-reversion rule — Wilder RSI mapped through ``arctanh`` into a
continuous forecast in ``[-Strategy.FORECAST_CAP, +Strategy.FORECAST_CAP]``,
with one or more RSI windows combined into a single weighted forecast.

**Mean reversion** — an *oversold* market (RSI → 0) is expected to bounce, so
it earns a LONG (positive) forecast; an *overbought* market (RSI → 100) earns
a SHORT (negative) forecast; RSI 50 → 0. This is the sign-flip of the raw RSI
level, hence the leading minus in the mapping below.

Per variation ``i`` (RSI window ``w_i``; default execution timeframe ``'1d'``):

    rsi_i   = RSI(close, window=w_i)                  # 0..100
    dev_i   = clip((rsi_i - 50) / 50, ±(1 - eps))     # ∈ (-1, 1); eps keeps
                                                      #   arctanh finite at the
                                                      #   RSI 0/100 extremes
    raw_i   = -arctanh(dev_i)                         # mean-reversion sign flip;
                                                      #   steepens toward extremes

    # Dynamic forecast scalar (mirrors EWMAC; targets avg |forecast| ≈ 50).
    abs_mean_i = SMA(|raw_i|, window=forecast_scalar_lookback)
    scalar_i   = Strategy.TARGET_AVG_ABS_FORECAST / abs_mean_i
    scaled_i   = clip(raw_i * scalar_i, ±Strategy.FORECAST_CAP)
                                                      # per-variation cap applied
                                                      #   before combination —
                                                      #   keeps any one runaway
                                                      #   variation from dominating

Combined forecast (default equal weights = 1/N each):

    combined = fdm * sum(w_i * scaled_i)              # ``fdm`` is Carver's
                                                      #   Forecast Diversification
                                                      #   Multiplier (default 1.0).

Because ``arctanh`` diverges at ``dev = ±1``, RSI at the 0 / 100 extremes maps
to ±∞ and is clipped to ``±Strategy.FORECAST_CAP`` — so the anchor points
``RSI 0 → +100``, ``RSI 50 → 0``, ``RSI 100 → -100`` hold automatically via the
cap, while the dynamic scalar drives the average absolute forecast toward
``Strategy.TARGET_AVG_ABS_FORECAST``. RSI is already a normalized 0..100
oscillator, so — unlike EWMAC — there is **no** price-volatility normalization
term; the dynamic scalar is the only scaling machinery.

Output: per-bar dict with the ``'forecast'`` key. ``TimeSeriesStrategy.update_bar``
clamps it to ``±Strategy.FORECAST_CAP`` and writes it to
``self.forecasts[symbol]``; the risk manager reads it on every completed bar to
derive the target position. The forecast depends only on finalized values (via
``Indicator.latest``), so it is stable across forming ticks within a period and
only changes at period boundaries.

The sign is fixed to mean reversion. A momentum variant (trade *with* the RSI
level) would simply drop the leading minus in ``_raw_from_rsi``; it is
intentionally **not** a parameter (YAGNI) and can be added trivially if needed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from event import BarEvent
from indicator import RSI, SMA
from strategy._base import Strategy, _DataHandlerLike
from strategy._timeseries import TimeSeriesStrategy

__all__ = ['RSIMRStrategy']


# Default variations as a ``{label: {param_name: value}}`` dict. The labels
# are param-derived so a default-constructed strategy emits diagnostic columns
# reading ``rsi_7`` etc. (identical to the pre-refactor output).
_DEFAULT_VARIATIONS: Dict[str, Dict[str, int]] = {
    '7': {'window': 7},
    '14': {'window': 14},
    '28': {'window': 28},
}

# Required per-variation param key — validated by strict set-equality so a
# typo fails loudly instead of silently NaN-ing the variation.
_PARAM_KEYS = frozenset({'window'})

# Clip the normalized RSI deviation away from ±1 so ``math.atanh`` stays finite
# when RSI hits exactly 0 or 100 (Wilder RSI does reach the bounds when
# ``avg_gain`` or ``avg_loss`` is 0). With this epsilon only RSI within ~5e-5 of
# 0/100 is clipped — the steep near-extreme response is preserved, and the ±100
# anchors are still reached via the forecast cap.
_RSI_DEV_CLIP: float = 1e-6


def _raw_from_rsi(rsi_value: float) -> float:
    """Map a Wilder RSI value (0..100) to the raw mean-reversion forecast.

    ``raw = -arctanh((rsi - 50) / 50)`` with the normalized deviation clipped to
    ``±(1 - _RSI_DEV_CLIP)`` so ``atanh`` never receives ±1. The leading minus is
    the mean-reversion sign flip: oversold (low RSI) → positive (long), overbought
    (high RSI) → negative (short), RSI 50 → 0.
    """
    dev = (rsi_value - 50.0) / 50.0
    lim = 1.0 - _RSI_DEV_CLIP
    if dev > lim:
        dev = lim
    elif dev < -lim:
        dev = -lim
    return -math.atanh(dev)


class RSIMRStrategy(TimeSeriesStrategy):
    """RSI mean-reverter with one or more RSI windows and a dynamic forecast scalar.

    Per-symbol state (per-variation indicators keyed by variation label):
        ``_rsis[label]``          — RSI indicator for variation ``label``.
        ``_abs_raw_smas[label]``  — rolling mean of ``|raw_label|`` over
                                     ``forecast_scalar_lookback`` finalized bars,
                                     feeding the dynamic forecast scalar.
    """

    def __init__(
        self,
        data_handler: _DataHandlerLike,
        symbol_list: List[str],
        execution_timeframe: str = '1d',
        variations: Optional[Dict[str, Dict[str, int]]] = None,
        weights: Optional[Dict[str, float]] = None,
        fdm: float = 1.0,
        forecast_scalar_lookback: int = 256,
    ):
        """Configure parameters and instantiate per-symbol indicators.

        Parameters
        ----------
        execution_timeframe
            Bar timeframe the RSI chain runs on (default ``'1d'``).
        variations
            Mapping ``{label: {'window': w}}`` of the Wilder-RSI variations, in
            bars of ``execution_timeframe``. The label is an opaque string
            keying both the combination weights and the per-bar diagnostic
            columns (``rsi_<label>`` …), mirroring the orchestrator's
            ``{label: strategy}`` shape; the nested ``{param_name: value}``
            params dict generalizes to indicators with more parameters. Each
            params dict must have exactly the key ``{'window'}`` with
            ``window >= 2``. Default ``{'7': {'window': 7}, '14': {'window':
            14}, '28': {'window': 28}}`` — a fast/medium/slow trio; the
            param-derived labels keep the default diagnostic columns reading
            ``rsi_7`` etc. Distinct labels may map to the same window (the
            columns are label-keyed, not window-keyed).
        weights
            Optional ``{label: weight}`` per-variation combination weights.
            Keys must match ``variations`` exactly; values must be finite,
            non-negative, and sum to ``1.0``. ``None`` (default) → equal
            weight ``1/N`` via ``analytics.equal_weight``.
        fdm
            Forecast Diversification Multiplier (Carver). Multiplied into the
            combined weighted forecast to scale it back up after
            diversification-driven cancellation across imperfectly-correlated
            variations. Must be strictly positive. Default ``1.0`` (no
            adjustment).
        forecast_scalar_lookback
            Window for the rolling mean of ``|raw|`` that feeds the dynamic
            forecast scalar (``Strategy.TARGET_AVG_ABS_FORECAST / mean(|raw|)``).
            Default ``256``. Must be ``>= 2``.

        The target average absolute forecast the scaling drives toward is
        ``Strategy.TARGET_AVG_ABS_FORECAST`` (project-wide constant, default
        ``50.0``).
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
            window = params['window']
            if window < 2:
                raise ValueError(
                    f"variation {label!r}: window must be >= 2, got {window}"
                )

        # Keys set-equal to the variation labels, finite, non-negative,
        # sum→1; ``None`` → equal weight (shared with the orchestrator).
        weights = validate_named_weights(weights, list(variations.keys()))

        if fdm <= 0:
            raise ValueError(f"fdm must be > 0, got {fdm}")

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
        self.forecast_scalar_lookback = forecast_scalar_lookback

        # Per-symbol per-variation indicators, keyed by variation label.
        self._rsis: Dict[str, Dict[str, RSI]] = {
            s: {label: RSI(window=p['window'])
                for label, p in self.variations.items()}
            for s in symbol_list
        }
        self._abs_raw_smas: Dict[str, Dict[str, SMA]] = {
            s: {label: SMA(window=forecast_scalar_lookback)
                for label in self.variations}
            for s in symbol_list
        }

    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        """Update RSIs from the latest forming execution-TF bar and compute the
        combined mean-reversion forecast (NaN during warmup).

        Returns a dict with:

        * ``'forecast'`` — combined capped forecast (possibly NaN).
        * ``'fdm'`` — the Forecast Diversification Multiplier.
        * Per variation ``label``: ``'rsi_<label>'`` (finalized RSI),
          ``'raw_<label>'`` (raw arctanh forecast pre-scalar),
          ``'abs_mean_<label>'`` (rolling mean of ``|raw|``),
          ``'scalar_<label>'`` (dynamic forecast scalar),
          ``'forecast_<label>'`` — the per-variation forecast
          **capped at ±Strategy.FORECAST_CAP** before combining — and
          ``'weight_<label>'`` — the variation's combination weight
          (constant per run; recorded for audit/attribution, mirroring
          the orchestrator's ``weight_<label>`` convention).

        The base class clamps the top-level forecast to
        ``±Strategy.FORECAST_CAP`` and writes it to ``self.forecasts[symbol]``
        (NaN values are recorded but skipped — the cached forecast stays at its
        prior value).
        """
        symbol = event.symbol

        # ── Latest forming bar. RSI tracks its own diffs internally, so a
        # single close is all that's needed (no 2-bar price-change term). ───
        bars = self.data_handler.get_latest_bars(
            symbol, 1, timeframe=self.execution_timeframe,
        )
        if not bars:
            return None

        forming_ts = bars[-1].timestamp
        forming_close = bars[-1].close

        # ── Per-variation pipeline: update RSI, push forming |raw| into the
        # abs-mean SMA, then compute the forecast and intermediates from
        # finalized values. ─────────────────────────────────────────────────
        nan = float('nan')
        per_var_forecasts: Dict[str, float] = {l: nan for l in self.variations}
        per_var_rsi: Dict[str, float] = {l: nan for l in self.variations}
        per_var_raw: Dict[str, float] = {l: nan for l in self.variations}
        per_var_abs_mean: Dict[str, float] = {l: nan for l in self.variations}
        per_var_scalar: Dict[str, float] = {l: nan for l in self.variations}

        for label in self.variations:
            rsi = self._rsis[symbol][label]
            abs_sma = self._abs_raw_smas[symbol][label]

            rsi.update(forming_ts, forming_close)

            # Push |raw_forming| into the rolling-mean SMA so it advances on
            # every bar (forming or completed). Skip while the forming RSI is
            # undefined — feeding NaN would propagate into the SMA window.
            if rsi.is_forming_ready:
                rsi_f = rsi.forming
                assert rsi_f is not None  # is_forming_ready ⇒ non-None
                raw_forming = _raw_from_rsi(float(rsi_f['rsi']))
                abs_sma.update(forming_ts, abs(raw_forming))

            # Compute per-variation forecast (and recorded intermediates) from
            # finalized values.
            if rsi.is_latest_ready and abs_sma.is_latest_ready:
                rsi_l = rsi.latest
                abs_l = abs_sma.latest
                assert rsi_l is not None and abs_l is not None  # ready ⇒ non-None
                rsi_latest = float(rsi_l['rsi'])
                abs_mean_latest = float(abs_l['sma'])
                if abs_mean_latest != 0.0:
                    raw_latest = _raw_from_rsi(rsi_latest)
                    scalar = (
                        Strategy.TARGET_AVG_ABS_FORECAST / abs_mean_latest
                    )
                    per_var_rsi[label] = rsi_latest
                    per_var_raw[label] = raw_latest
                    per_var_abs_mean[label] = abs_mean_latest
                    per_var_scalar[label] = scalar
                    # Cap each variation individually before combining — keeps
                    # one runaway variation from dominating the weighted sum and
                    # keeps the recorded ``forecast_<label>`` diagnostic on the
                    # same ``[-100, +100]`` scale as the combined ``forecast``.
                    cap = Strategy.FORECAST_CAP
                    per_var_forecasts[label] = max(
                        -cap, min(cap, raw_latest * scalar)
                    )

        # ── Combine per-variation forecasts. ────────────────────────────────
        # If ANY variation isn't ready, the combined forecast is undefined — we
        # don't backfill from finite variations because the weights are
        # calibrated against the full ensemble. ``TimeSeriesStrategy.update_bar`` clamps
        # the final value to ``±Strategy.FORECAST_CAP`` before storing it, so no
        # clamp is applied here.
        if any(pd.isna(per_var_forecasts[l]) for l in self.variations):
            final_forecast = nan
        else:
            final_forecast = self.fdm * sum(
                self.weights[l] * per_var_forecasts[l] for l in self.variations
            )

        # ── Recorded fields. ────────────────────────────────────────────────
        row: Dict[str, Any] = {
            'forecast': final_forecast,
            'fdm': self.fdm,
        }
        for label in self.variations:
            row[f'rsi_{label}'] = per_var_rsi[label]
            row[f'raw_{label}'] = per_var_raw[label]
            row[f'abs_mean_{label}'] = per_var_abs_mean[label]
            row[f'scalar_{label}'] = per_var_scalar[label]
            row[f'forecast_{label}'] = per_var_forecasts[label]
            row[f'weight_{label}'] = self.weights[label]
        return row
