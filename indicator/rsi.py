"""Relative Strength Index — Wilder smoothing on gain/loss EMAs.

Stateful: maintains running ``avg_gain`` and ``avg_loss`` via
``y[t] = (1 - alpha) * y[t-1] + alpha * x[t]`` with ``alpha = 1 / window``,
seeded from the first valid gain/loss (the second observed input — diff[0]
is undefined). Emits NaN until ``window`` valid gains have been seen
(matches ``ewm(com=window-1, min_periods=window, adjust=False)`` semantics
on a Series whose first diff is NaN).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['RSI']


class RSI(Indicator):
    """Wilder RSI(window). Output dict carries ``rsi``, ``avg_gain``, ``avg_loss``.

    ``avg_gain`` / ``avg_loss`` are the recursive Wilder EMAs of gain/loss; they
    are exposed in the output dict so a re-tick of the forming bar can
    correctly fold from the last finalized values.
    """

    def __init__(self, window: int = 14, *, outputs_maxlen: int = 500):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        # window+1 inputs span window diffs; that's the count threshold
        # for emitting a non-NaN RSI.
        super().__init__(outputs_maxlen=outputs_maxlen,
                         inputs_maxlen=window + 1)
        self.window = window
        self.alpha = 1.0 / window

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        n_inputs = len(self._inputs)
        nan = float('nan')

        # Need a prior input to compute the diff.
        if n_inputs < 2:
            return {'rsi': nan, 'avg_gain': nan, 'avg_loss': nan}

        prev_val = self._inputs[-2][1]['value']
        diff = value - prev_val
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0

        # Seed the recursion with the first valid gain/loss; otherwise fold.
        if prev_output is None or not np.isfinite(prev_output['avg_gain']):
            avg_gain = gain
            avg_loss = loss
        else:
            one_minus_a = 1.0 - self.alpha
            avg_gain = one_minus_a * prev_output['avg_gain'] + self.alpha * gain
            avg_loss = one_minus_a * prev_output['avg_loss'] + self.alpha * loss

        # min_periods=window mask: NaN until `window` valid gains observed.
        # n_inputs distinct inputs → n_inputs - 1 valid gains.
        if n_inputs - 1 < self.window:
            rsi_val = nan
        elif avg_loss == 0.0:
            # 0/0 → NaN; positive_gain / 0 → 100.
            rsi_val = nan if avg_gain == 0.0 else 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - 100.0 / (1.0 + rs)

        return {'rsi': rsi_val, 'avg_gain': avg_gain, 'avg_loss': avg_loss}

    @staticmethod
    def from_series(data: pd.Series, window: int = 14) -> pd.Series:
        """Vectorized one-shot: matches the stateful ``rsi`` column bar-for-bar."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if len(data) < window + 1:
            return pd.Series(float('nan'), index=data.index, dtype=float)

        delta = data.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
