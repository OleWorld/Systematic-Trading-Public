"""Average True Range with Wilder (RMA) smoothing.

True Range = ``max(h - l, |h - prev_close|, |l - prev_close|)``. The
smoothing is Wilder's exponential moving average with ``alpha = 1/length``,
seeded from the first valid TR (the second observed bar — TR is undefined
without a prior close). Emits NaN until ``length`` valid TRs have been
observed (matches the original ``ema(tr, length)`` masking).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['ATR']


class ATR(Indicator):
    """ATR(length). Output dict carries ``atr`` (masked) plus hidden ``_atr_raw``."""

    def __init__(self, length: int = 14, *, outputs_maxlen: int = 500):
        if length < 2:
            raise ValueError(f"length must be >= 2, got {length}")
        # inputs_maxlen=length so len(_inputs) caps at `length`; that's the
        # count threshold for unmasking. We only ever read _inputs[-2] for
        # prev_close, so any maxlen >= 2 suffices for correctness.
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=length)
        self.length = length
        self.alpha = 1.0 / length

    def update(self, ts: datetime, high: float, low: float, close: float) -> None:
        """Push one OHLC bar's H/L/C. ATR ignores Open."""
        self._push(ts, {
            'high': float(high),
            'low': float(low),
            'close': float(close),
        })

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, high: float, low: float, close: float) -> Dict[str, float]:
        n_inputs = len(self._inputs)
        nan = float('nan')

        # First bar: no prev_close → only the (h - l) term of TR is defined.
        # This matches the vectorized path: ``pd.concat([h-l, |h-prev_c|,
        # |l-prev_c|], axis=1).max(axis=1)`` skips NaN, so TR[0] = h[0]-l[0].
        if n_inputs < 2:
            tr = high - low
        else:
            prev_close = self._inputs[-2][1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # Seed Wilder EMA from first valid TR; otherwise fold.
        prev_raw = prev_output.get('_atr_raw', nan) if prev_output is not None else nan
        if not np.isfinite(prev_raw):
            atr_raw = tr
        else:
            atr_raw = prev_raw + self.alpha * (tr - prev_raw)

        # min_periods=length: emit valid once `length` distinct inputs
        # (== `length` valid TRs under the skipna semantics) have been seen.
        atr_out = atr_raw if n_inputs >= self.length else nan
        return {'atr': atr_out, '_atr_raw': atr_raw}

    @staticmethod
    def from_series(high: pd.Series, low: pd.Series, close: pd.Series,
                    length: int = 14) -> pd.Series:
        """Vectorized one-shot: matches the stateful ``atr`` column bar-for-bar.

        Uses Wilder/RMA smoothing (``alpha = 1/length``) on the True Range —
        ``com=length-1`` in pandas terms — to mirror the recursion in
        ``_compute``.
        """
        if length < 2:
            raise ValueError(f"length must be >= 2, got {length}")
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        if len(tr) < length:
            return pd.Series(float('nan'), index=tr.index, dtype=float)
        return tr.ewm(com=length - 1, min_periods=length, adjust=False).mean()
