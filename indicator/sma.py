"""Simple Moving Average — stateful, window-based incremental indicator.

The vectorized one-shot is available via ``SMA.from_series`` (used by tests
and ``warmup``); the stateful path keeps a deque of the trailing ``window``
inputs and recomputes the mean each tick. O(window) per tick — same as the
prior pure-function recompute over a rolling window, but tied to a single
input value rather than a full Series rebuild.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['SMA']


class SMA(Indicator):
    """Simple Moving Average over the trailing ``window`` inputs.

    Emits NaN until ``window`` inputs have been observed; then the mean of
    the last ``window`` inputs each tick.
    """

    def __init__(self, window: int, *, outputs_maxlen: int = 500):
        """Configure window length.

        Parameters
        ----------
        window
            Number of inputs averaged. Must be >= 1.
        outputs_maxlen
            Output deque size (default 500).
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=window)
        self.window = window

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation. Upserts on equal ts; advances on new ts."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 **inputs: float) -> Dict[str, float]:
        if len(self._inputs) < self.window:
            return {'sma': float('nan')}
        # _inputs already holds at most `window` entries (deque maxlen=window),
        # so we just average all of them.
        total = 0.0
        for _, vals in self._inputs:
            total += vals['value']
        return {'sma': total / self.window}

    @staticmethod
    def from_series(data: pd.Series, window: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if len(data) < window:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.rolling(window).mean()
