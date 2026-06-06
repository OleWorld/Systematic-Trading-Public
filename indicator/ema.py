"""Exponential Moving Average — standard smoothing (alpha = 2/(span+1)).

Matches ``pd.Series.ewm(span=span, min_periods=span, adjust=False).mean()``
(equivalently ``alpha = 2/(span+1)``) — same convention as TradingView's
``ta.ema(close, length)``. The recursion
``y[t] = (1 - alpha) * y[t-1] + alpha * x[t]`` runs from the very first
input (seeded with that input's value), but the first ``span-1`` emitted
outputs are masked to NaN. The unmasked recursion state is carried in the
hidden ``_ema_raw`` field so a re-tick of the forming bar can fold
correctly even while ``ema`` itself is still NaN.

For Wilder/RMA smoothing (``alpha = 1/length``, used by RSI and ATR), use
those indicators directly — they inline the Wilder recursion themselves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['EMA']


class EMA(Indicator):
    """Exponential moving average (standard, ``alpha = 2/(span+1)``).

    Output dict carries the public ``ema`` key (NaN for the first ``span-1``
    inputs) plus the hidden ``_ema_raw`` recursion state (always finite once
    seeded).
    """

    def __init__(self, span: int, *, outputs_maxlen: int = 500):
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        # inputs_maxlen=span so len(_inputs) caps at `span`; "len >= span"
        # is the count threshold for unmasking the output.
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=span)
        self.span = span
        self.alpha = 2.0 / (span + 1.0)

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        # First-ever input (or recovering from a finalized NaN): seed.
        if prev_output is None or not np.isfinite(prev_output.get('_ema_raw',
                                                                  float('nan'))):
            ema_raw = value
        else:
            prev_raw = prev_output['_ema_raw']
            ema_raw = prev_raw + self.alpha * (value - prev_raw)

        # min_periods=span mask: emit NaN until `span` distinct inputs seen.
        ema_out = ema_raw if len(self._inputs) >= self.span else float('nan')
        return {'ema': ema_out, '_ema_raw': ema_raw}

    @staticmethod
    def from_series(data: pd.Series, span: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful ``ema`` column bar-for-bar."""
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        if len(data) < span:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.ewm(span=span, min_periods=span, adjust=False).mean()
