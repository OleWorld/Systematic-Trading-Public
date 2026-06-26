"""Rolling sample standard deviation (ddof=1) over the trailing ``length`` inputs.

Matches ``pd.Series.rolling(length).std(ddof=1)``: NaN until ``length``
inputs have been observed; bias-corrected stdev thereafter.

**O(1) per tick** via a sliding-window Welford update. The trailing-window
``mean`` and ``M2`` (= ``Σ(x − mean)²``) are carried as hidden output keys
(filtered from the public view by ``Indicator._public``); each tick removes
the value rolling out of the window and adds the new one using West's
add/remove formulas, rather than re-summing the deque. This never forms the
``Σx² − (Σx)²/n`` large-minus-large subtraction, so it has no
catastrophic-cancellation regime even for large-mean inputs.

Mechanics mirror the recursive-indicator idiom of ``EMA`` / ``EWMStdev``:
``_compute`` folds from ``prev_output``. The input deque holds one extra slot
(``length + 1``) so the value leaving the window survives as ``_inputs[0]``
(``_compute`` runs after ``_push`` has already mutated the deque). The base
``_push`` supplies the correct ``prev_output`` on both new bars and same-ts
re-ticks, so the slide is uniform across the two. A direct O(length) compute
is used once on the first full window (and on recovery after any NaN gap).

A two-sided noise floor snaps ``M2`` to exactly 0 in the degenerate
constant-window regime (where the West removal step's tiny drift would
otherwise leak a ~1e-15 variance): ``var`` returns exactly 0.0 — matching the
old mean-centered ``np.std`` and keeping strict ``sigma == 0`` guards
downstream valid — and never ``sqrt``s a negative.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from indicator._base import Indicator

__all__ = ['Stdev']

# Relative floor for snapping floating-point variance noise to exactly zero.
# ``M2`` below ``_VAR_NOISE_REL × length × mean²`` is treated as the degenerate
# zero-variance case (constant window). ~80× the worst-case ``n·ε`` rounding
# bound for n ≤ 60, yet far below any real variance-to-mean-square ratio that
# arises here (Stdev is fed near-zero-mean price changes), so it never masks a
# genuine variance.
_VAR_NOISE_REL = 1e-12


class Stdev(Indicator):
    """Rolling sample standard deviation (ddof=1) over ``length`` inputs."""

    def __init__(self, length: int, *, outputs_maxlen: int = 500):
        if length < 2:
            raise ValueError(f"length must be >= 2 for sample stdev, got {length}")
        # One extra input slot beyond `length`: `_compute` runs after `_push`
        # has appended (and evicted), so the value that rolled out of the
        # window must survive one extra tick to be read as `_inputs[0]`.
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=length + 1)
        self.length = length

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        """O(1) sliding-window sample stdev via windowed Welford.

        Carries ``mean`` and ``M2`` (= ``Σ(x − mean)²``) as hidden keys. Folds
        from ``prev_output`` by removing the value rolling out of the window
        (``_inputs[0]``) and adding the newest (West's add/remove formulas).
        Falls back to a direct O(length) compute on the first full window —
        and on recovery after a NaN gap — where ``prev_output`` carries no
        finite ``_mean``. The two-sided noise floor snaps degenerate
        (constant-window) ``M2`` to exactly 0 so ``var`` is exactly 0.0 and
        ``sqrt`` never sees a negative.
        """
        n = self.length
        if len(self._inputs) < n:
            return {'stdev': float('nan')}
        prev_mean = prev_output.get('_mean') if prev_output else None
        if prev_mean is None or pd.isna(prev_mean):
            window = [vals['value'] for _, vals in list(self._inputs)[-n:]]
            mean = sum(window) / n
            m2 = 0.0
            for x in window:
                d = x - mean
                m2 += d * d
        else:
            x_old = self._inputs[0][1]['value']        # value rolling out of the window
            prev_m2 = prev_output['_m2']
            m_rem = (n * prev_mean - x_old) / (n - 1)   # remove x_old (n → n−1)
            m2_rem = prev_m2 - (x_old - prev_mean) * (x_old - m_rem)
            delta = value - m_rem                       # add value (n−1 → n)
            mean = m_rem + delta / n
            m2 = m2_rem + delta * (value - mean)
        if m2 < _VAR_NOISE_REL * n * mean * mean:       # two-sided ε-noise floor
            m2 = 0.0
        return {'stdev': math.sqrt(m2 / (n - 1)), '_mean': mean, '_m2': m2}

    @staticmethod
    def from_series(data: pd.Series, length: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if length < 2:
            raise ValueError(f"length must be >= 2 for sample stdev, got {length}")
        if len(data) < length:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.rolling(length).std(ddof=1)
