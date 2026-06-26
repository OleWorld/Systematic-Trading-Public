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

Output: per-bar dict with the ``'forecast'`` key. ``Strategy.update_bar``
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

__all__ = ['RSIMRStrategy']


_DEFAULT_RSI_WINDOWS: List[int] = [7, 14, 28]

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


class RSIMRStrategy(Strategy):
    """RSI mean-reverter with one or more RSI windows and a dynamic forecast scalar.

    Per-symbol state:
        ``_rsis[i]``          — RSI indicator for variation ``i``.
        ``_abs_raw_smas[i]``  — rolling mean of ``|raw_i|`` over
                                 ``forecast_scalar_lookback`` finalized bars,
                                 feeding the dynamic forecast scalar.
    """

    def __init__(
        self,
        data_handler: _DataHandlerLike,
        symbol_list: List[str],
        execution_timeframe: str = '1d',
        rsi_windows: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        fdm: float = 1.0,
        forecast_scalar_lookback: int = 256,
    ):
        """Configure parameters and instantiate per-symbol indicators.

        Parameters
        ----------
        execution_timeframe
            Bar timeframe the RSI chain runs on (default ``'1d'``).
        rsi_windows
            List of Wilder-RSI windows in bars of ``execution_timeframe``.
            Default ``[7, 14, 28]`` — a fast/medium/slow trio. Each window must
            be ``>= 2`` and the windows must be distinct (the per-bar diagnostic
            columns are window-named).
        weights
            Per-variation combination weights. Must have the same length as
            ``rsi_windows`` and sum to 1.0. ``None`` (default) → equal weights.
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

        if rsi_windows is None:
            rsi_windows = list(_DEFAULT_RSI_WINDOWS)
        if not rsi_windows:
            raise ValueError("rsi_windows must be non-empty")
        for w in rsi_windows:
            if w < 2:
                raise ValueError(
                    f"each RSI window must be >= 2, got {w}"
                )
        if len(set(rsi_windows)) != len(rsi_windows):
            raise ValueError(
                f"rsi_windows must be distinct (diagnostic columns are "
                f"window-named), got {rsi_windows}"
            )

        n_vars = len(rsi_windows)
        if weights is None:
            weights = [1.0 / n_vars] * n_vars
        else:
            if len(weights) != n_vars:
                raise ValueError(
                    f"weights length {len(weights)} != rsi_windows length {n_vars}"
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(
                    f"weights must sum to 1.0, got sum={sum(weights)}"
                )

        if fdm <= 0:
            raise ValueError(f"fdm must be > 0, got {fdm}")

        if forecast_scalar_lookback < 2:
            raise ValueError(
                f"forecast_scalar_lookback must be >= 2, got {forecast_scalar_lookback}"
            )

        self.execution_timeframe = execution_timeframe
        self.rsi_windows: List[int] = list(rsi_windows)
        self.weights: List[float] = list(weights)
        self.fdm = fdm
        self.forecast_scalar_lookback = forecast_scalar_lookback

        # Per-symbol per-variation indicators.
        self._rsis: Dict[str, List[RSI]] = {
            s: [RSI(window=w) for w in self.rsi_windows]
            for s in symbol_list
        }
        self._abs_raw_smas: Dict[str, List[SMA]] = {
            s: [SMA(window=forecast_scalar_lookback) for _ in self.rsi_windows]
            for s in symbol_list
        }

    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        """Update RSIs from the latest forming execution-TF bar and compute the
        combined mean-reversion forecast (NaN during warmup).

        Returns a dict with:

        * ``'forecast'`` — combined capped forecast (possibly NaN).
        * ``'fdm'`` — the Forecast Diversification Multiplier.
        * Per variation window ``w``: ``'rsi_<w>'`` (finalized RSI),
          ``'raw_<w>'`` (raw arctanh forecast pre-scalar), ``'abs_mean_<w>'``
          (rolling mean of ``|raw|``), ``'scalar_<w>'`` (dynamic forecast
          scalar), and ``'forecast_<w>'`` — the per-variation forecast
          **capped at ±Strategy.FORECAST_CAP** before combining.

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
        n_vars = len(self.rsi_windows)
        nan = float('nan')
        per_var_forecasts: List[float] = [nan] * n_vars
        per_var_rsi: List[float] = [nan] * n_vars
        per_var_raw: List[float] = [nan] * n_vars
        per_var_abs_mean: List[float] = [nan] * n_vars
        per_var_scalar: List[float] = [nan] * n_vars

        for i in range(n_vars):
            rsi = self._rsis[symbol][i]
            abs_sma = self._abs_raw_smas[symbol][i]

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
                    per_var_rsi[i] = rsi_latest
                    per_var_raw[i] = raw_latest
                    per_var_abs_mean[i] = abs_mean_latest
                    per_var_scalar[i] = scalar
                    # Cap each variation individually before combining — keeps
                    # one runaway variation from dominating the weighted sum and
                    # keeps the recorded ``forecast_<w>`` diagnostic on the same
                    # ``[-100, +100]`` scale as the combined ``forecast``.
                    cap = Strategy.FORECAST_CAP
                    per_var_forecasts[i] = max(
                        -cap, min(cap, raw_latest * scalar)
                    )

        # ── Combine per-variation forecasts. ────────────────────────────────
        # If ANY variation isn't ready, the combined forecast is undefined — we
        # don't backfill from finite variations because the weights are
        # calibrated against the full ensemble. ``Strategy.update_bar`` clamps
        # the final value to ``±Strategy.FORECAST_CAP`` before storing it, so no
        # clamp is applied here.
        if any(pd.isna(f) for f in per_var_forecasts):
            final_forecast = nan
        else:
            final_forecast = self.fdm * sum(
                w * f for w, f in zip(self.weights, per_var_forecasts)
            )

        # ── Recorded fields. ────────────────────────────────────────────────
        row: Dict[str, Any] = {
            'forecast': final_forecast,
            'fdm': self.fdm,
        }
        for i, w in enumerate(self.rsi_windows):
            row[f'rsi_{w}'] = per_var_rsi[i]
            row[f'raw_{w}'] = per_var_raw[i]
            row[f'abs_mean_{w}'] = per_var_abs_mean[i]
            row[f'scalar_{w}'] = per_var_scalar[i]
            row[f'forecast_{w}'] = per_var_forecasts[i]
        return row
