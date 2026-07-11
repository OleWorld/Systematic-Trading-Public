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

The incremental remove/add step introduces ``O(ε·n·mean²)`` cancellation
drift, which at large means can both fabricate a tiny negative ``M2`` (which
would ``sqrt`` to NaN) and mask genuine low-dispersion variance. When ``M2``
falls into that dust band the slide **recomputes ``M2`` directly from the
window** (the same exact ``O(length)`` mean-centered sum used on the first
full window): a genuinely constant window yields exactly 0.0 when the
window has been constant from the start — matching the old mean-centered
``np.std``; the risk manager's zero-vol guard is RELATIVE (``sigma == 0 or
sigma < 1e-6·|close|``), so dust-level leakage is absorbed there. A window
that becomes constant MID-STREAM can retain float dust from the incremental
slide when the window mean is ~0 (the dust band is mean-relative) — a known
open limitation. A real variance is recovered faithfully, so the output
stays translation-invariant and matches ``from_series`` for every input
scale (never snapping genuine variance to zero, never ``sqrt``ing a
negative).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from indicator._base import Indicator

__all__ = ['Stdev']

# Relative trigger for a direct (exact) window recompute. ``M2`` below
# ``_VAR_NOISE_REL × length × mean²`` is "dust-suspect": the incremental
# remove/add step's cancellation error is ``O(ε·n·mean²)``, so at this
# magnitude an incremental ``M2`` can no longer be trusted to distinguish a
# constant window from a genuine low-dispersion one. Crossing it triggers an
# exact O(length) recompute (NOT a snap-to-zero), so genuine variance is
# preserved at every input scale while true constants still resolve to exactly
# 0.0. Generous on purpose — false triggers only cost an O(length) recompute.
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

    def _direct_window_moments(self, n: int) -> tuple[float, float]:
        """Exact ``(mean, M2)`` over the trailing ``n`` inputs via a mean-centered
        sum of squares — ``O(n)``, translation-invariant, and ``M2 >= 0`` by
        construction (a constant window yields exactly ``0.0``)."""
        window = [vals['value'] for _, vals in list(self._inputs)[-n:]]
        mean = sum(window) / n
        m2 = 0.0
        for x in window:
            d = x - mean
            m2 += d * d
        return mean, m2

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        """O(1) sliding-window sample stdev via windowed Welford.

        Carries ``mean`` and ``M2`` (= ``Σ(x − mean)²``) as hidden keys. Folds
        from ``prev_output`` by removing the value rolling out of the window
        (``_inputs[0]``) and adding the newest (West's add/remove formulas).
        Falls back to a direct O(length) compute on the first full window —
        and on recovery after a NaN gap — where ``prev_output`` carries no
        finite ``_mean``. When the incremental ``M2`` lands in the
        ``_VAR_NOISE_REL`` dust band (where remove/add cancellation drift can
        both fabricate a negative ``M2`` and mask genuine low-dispersion
        variance), it recomputes ``M2`` directly from the window — exact for
        every input scale — instead of snapping to 0, so ``var`` is exactly
        0.0 only for genuinely constant windows and ``sqrt`` never sees a
        negative.
        """
        n = self.length
        if len(self._inputs) < n:
            return {'stdev': float('nan')}
        prev_mean = prev_output.get('_mean') if prev_output else None
        if prev_mean is None or pd.isna(prev_mean):
            mean, m2 = self._direct_window_moments(n)
        else:
            x_old = self._inputs[0][1]['value']        # value rolling out of the window
            prev_m2 = prev_output['_m2']
            m_rem = (n * prev_mean - x_old) / (n - 1)   # remove x_old (n → n−1)
            m2_rem = prev_m2 - (x_old - prev_mean) * (x_old - m_rem)
            delta = value - m_rem                       # add value (n−1 → n)
            mean = m_rem + delta / n
            m2 = m2_rem + delta * (value - mean)
            if m2 < _VAR_NOISE_REL * n * mean * mean:   # dust-suspect → exact recompute
                mean, m2 = self._direct_window_moments(n)
        return {'stdev': math.sqrt(m2 / (n - 1)), '_mean': mean, '_m2': m2}

    @staticmethod
    def from_series(data: pd.Series, length: int) -> pd.Series:
        """Vectorized one-shot: matches the stateful output bar-for-bar."""
        if length < 2:
            raise ValueError(f"length must be >= 2 for sample stdev, got {length}")
        if len(data) < length:
            return pd.Series(float('nan'), index=data.index, dtype=float)
        return data.rolling(length).std(ddof=1)
