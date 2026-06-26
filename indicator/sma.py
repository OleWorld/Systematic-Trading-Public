"""Simple Moving Average — stateful, window-based incremental indicator.

The vectorized one-shot is available via ``SMA.from_series`` (used by tests
and ``warmup``); the stateful path is **O(1) per tick**. It carries the
trailing-window sum as a hidden ``_sum`` output key (filtered from the public
view by ``Indicator._public``) and slides it each tick — add the new value,
drop the one that rolled out of the window — instead of re-summing the deque.
This is the recursive-indicator idiom already used by ``EMA`` (``_ema_raw``)
and ``EWMStdev`` (``_var_raw``): ``_compute`` folds from ``prev_output``.

To make the rolled-out value readable in ``_compute`` (which runs *after*
``_push`` has already mutated the input deque), the input deque holds one
extra slot (``window + 1``), so the value leaving the window survives as
``_inputs[0]``. The base ``_push`` hands ``_compute`` the correct
``prev_output`` (the last finalized output) on both new bars and same-ts
re-ticks, so the slide formula is uniform across the two. A direct O(window)
sum is used once on the first full window (and on recovery after any NaN
gap, where ``prev_output`` carries no finite ``_sum``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

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
        # One extra input slot beyond `window`: `_compute` runs after `_push`
        # has appended (and evicted), so the value that rolled out of the
        # window must survive one extra tick to be read as `_inputs[0]`.
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=window + 1)
        self.window = window

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation. Upserts on equal ts; advances on new ts."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        """O(1) sliding-window mean.

        Carries the window sum as a hidden ``_sum`` key. Folds from
        ``prev_output['_sum']`` by adding the newest value and dropping the
        one that rolled out of the window (``_inputs[0]``). Falls back to a
        direct O(window) sum on the first full window — and on recovery after
        a NaN gap — where ``prev_output`` carries no finite ``_sum``.
        """
        n = self.window
        if len(self._inputs) < n:
            return {'sma': float('nan')}
        prev_sum = prev_output.get('_sum') if prev_output else None
        if prev_sum is None or pd.isna(prev_sum):
            new_sum = sum(vals['value'] for _, vals in list(self._inputs)[-n:])
        else:
            new_sum = prev_sum + value - self._inputs[0][1]['value']
        return {'sma': new_sum / n, '_sum': new_sum}

    @staticmethod
    def from_series(data: pd.Series, window: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if len(data) < window:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.rolling(window).mean()
