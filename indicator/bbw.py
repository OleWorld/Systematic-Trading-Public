"""Bollinger Band Width: ``2 * mult * stdev / abs(basis)``, basis = SMA(length).

Matches Pine's ``ta.bbw`` with an SMA basis at positive prices, but divides
by ``abs(basis)`` so the width stays a non-negative magnitude when prices
(and therefore the basis) can be negative — e.g. crude oil futures during
the 2020-04-20 settlement. Pine's raw ``(upper-lower)/basis`` would invert
its sign there and is undefined at ``basis == 0``; we return NaN at zero
basis. Numbers are bit-for-bit identical to Pine when the basis is positive.

Emits NaN until ``length`` inputs observed; thereafter computes basis and
stdev inline over the trailing window of inputs (no dependency on the
SMA / Stdev classes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator
from indicator.sma import SMA
from indicator.stdev import Stdev

__all__ = ['BBW']


class BBW(Indicator):
    """Bollinger Band Width over the trailing ``length`` inputs."""

    def __init__(self, length: int, mult: float = 1.0, *,
                 outputs_maxlen: int = 500):
        if length < 2:
            raise ValueError(f"length must be >= 2, got {length}")
        if mult <= 0:
            raise ValueError(f"mult must be > 0, got {mult}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=length)
        self.length = length
        self.mult = mult

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        if len(self._inputs) < self.length:
            return {'bbw': float('nan')}
        arr = np.fromiter((vals['value'] for _, vals in self._inputs),
                          dtype=float, count=self.length)
        basis = float(arr.mean())
        std = float(np.std(arr, ddof=1))
        if basis == 0.0:
            return {'bbw': float('nan')}
        return {'bbw': 2.0 * self.mult * std / abs(basis)}

    @staticmethod
    def from_series(data: pd.Series, length: int, mult: float) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        basis = SMA.from_series(data, length)
        dev = mult * Stdev.from_series(data, length)
        upper = basis + dev
        lower = basis - dev
        result = (upper - lower) / basis.abs()
        result[basis == 0] = float('nan')
        return result
