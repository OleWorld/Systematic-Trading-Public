"""Rolling sample standard deviation (ddof=1) over the trailing ``length`` inputs.

Matches ``pd.Series.rolling(length).std(ddof=1)``: NaN until ``length``
inputs have been observed; bias-corrected stdev thereafter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['Stdev']


class Stdev(Indicator):
    """Rolling sample standard deviation (ddof=1) over ``length`` inputs."""

    def __init__(self, length: int, *, outputs_maxlen: int = 500):
        if length < 2:
            raise ValueError(f"length must be >= 2 for sample stdev, got {length}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=length)
        self.length = length

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        if len(self._inputs) < self.length:
            return {'stdev': float('nan')}
        arr = np.fromiter((vals['value'] for _, vals in self._inputs),
                          dtype=float, count=self.length)
        return {'stdev': float(np.std(arr, ddof=1))}

    @staticmethod
    def from_series(data: pd.Series, length: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if length < 2:
            raise ValueError(f"length must be >= 2 for sample stdev, got {length}")
        if len(data) < length:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.rolling(length).std(ddof=1)
