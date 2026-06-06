"""Pine-style rolling percent-rank.

For a window of ``length`` values, returns the percent of the first
``length - 1`` values strictly less than the current (the last) value, on
the 0..100 scale. Emits NaN until ``length`` inputs have been observed.
Matches Pine's ``ta.percentrank`` and the original ``percentrank`` pure
function.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['PercentRank']


class PercentRank(Indicator):
    """Rolling percent-rank over the trailing ``length`` inputs (0..100)."""

    def __init__(self, length: int, *, outputs_maxlen: int = 500):
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=length)
        self.length = length

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        if len(self._inputs) < self.length:
            return {'percentrank': float('nan')}

        current = self._inputs[-1][1]['value']
        prior_size = self.length - 1
        if prior_size == 0:
            return {'percentrank': 0.0}

        # Count strictly-less prior values (last `length - 1` entries before current).
        count = 0
        for i in range(-self.length, -1):
            if self._inputs[i][1]['value'] < current:
                count += 1
        return {'percentrank': count / prior_size * 100.0}

    @staticmethod
    def from_series(data: pd.Series, length: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if len(data) < length:
            return pd.Series(float('nan'), index=data.index, dtype=float)

        def _pr(window: np.ndarray) -> float:
            current = window[-1]
            prior = window[:-1]
            if prior.size == 0:
                return 0.0
            return float((prior < current).sum()) / prior.size * 100.0

        return data.rolling(length).apply(_pr, raw=True)
