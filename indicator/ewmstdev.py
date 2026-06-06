"""Exponentially-weighted moving standard deviation (RiskMetrics convention).

Recursion (``alpha = 2/(span+1)``, mirrors ``indicator/ema.py``):

    var_t = (1 - alpha) * var_{t-1} + alpha * value_t**2     (zero-mean assumed)
    stdev_t = sqrt(var_t)

Matches ``data.pow(2).ewm(span=span, min_periods=span, adjust=False).mean().pow(0.5)``
bar-for-bar. The first ``span - 1`` emitted ``stdev`` outputs are masked to NaN
(same warmup convention as ``EMA``); the unmasked recursion state is carried
under the hidden ``_var_raw`` field so a re-tick of the forming bar folds
correctly even while the public ``stdev`` is still NaN.

Designed for vol-of-returns. The zero-mean assumption is the standard
RiskMetrics simplification — for noisy short-horizon returns the sample mean
is small relative to the noise floor, and including it adds variance to the
estimator with negligible bias improvement.
"""

from __future__ import annotations

from datetime import datetime
from math import sqrt
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['EWMStdev']


class EWMStdev(Indicator):
    """Exponentially-weighted moving standard deviation (zero-mean, span-parameterized).

    Output dict carries the public ``stdev`` key (NaN for the first
    ``span - 1`` inputs) plus the hidden ``_var_raw`` recursion state
    (always finite once seeded).
    """

    def __init__(self, span: int, *, outputs_maxlen: int = 500):
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=span)
        self.span = span
        self.alpha = 2.0 / (span + 1.0)

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation (typically a return)."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        sq = value * value
        if prev_output is None or not np.isfinite(prev_output.get('_var_raw',
                                                                  float('nan'))):
            var_raw = sq
        else:
            prev_raw = prev_output['_var_raw']
            var_raw = prev_raw + self.alpha * (sq - prev_raw)

        stdev_out = sqrt(var_raw) if len(self._inputs) >= self.span else float('nan')
        return {'stdev': stdev_out, '_var_raw': var_raw}

    @staticmethod
    def from_series(data: pd.Series, span: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful ``stdev`` column bar-for-bar."""
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        if len(data) < span:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        var = data.pow(2).ewm(span=span, min_periods=span, adjust=False).mean()
        return var.pow(0.5)
