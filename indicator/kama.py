"""Kaufman Adaptive Moving Average — stateful, recursive after seed.

Computes an efficiency ratio (ER) over the trailing ``er_length`` window,
then a smoothing constant ``sc = (ER*(2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2``,
then a recursive update ``k[t] = k[t-1] + sc[t] * (src[t] - k[t-1])``.

The recursion is seeded from the first input (matches Pine's ``nz()``
seeding) and runs from there in a hidden ``_kama_raw`` field. The public
``kama`` value is masked to NaN until ``er_length + 1`` inputs have been
observed (the first valid ER window) — same pattern as EMA's ``_ema_raw``
and ATR's ``_atr_raw``. Before that boundary ER is treated as 0 (sc
collapses to the slow constant) so the recursion advances cleanly even
while the public output is still masked.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['KAMA']


class KAMA(Indicator):
    """Kaufman Adaptive Moving Average.

    Output dict carries the public ``kama`` key (NaN for the first
    ``er_length`` inputs) plus the hidden ``_kama_raw`` recursion state
    (always finite once seeded).
    """

    def __init__(self, er_length: int = 10, fast: int = 2, slow: int = 30,
                 *, outputs_maxlen: int = 500):
        """Configure adaptive parameters.

        Parameters
        ----------
        er_length
            Lookback for the efficiency ratio. Must be >= 1.
        fast, slow
            Smoothing-constant span endpoints. ``fast`` must be < ``slow``.
        outputs_maxlen
            Output deque size (default 500).
        """
        if er_length < 1:
            raise ValueError(f"er_length must be >= 1, got {er_length}")
        if fast < 1 or slow < 1 or fast >= slow:
            raise ValueError(
                f"need 1 <= fast < slow, got fast={fast} slow={slow}"
            )
        # Need er_length + 1 values to span the ER window's diffs.
        super().__init__(outputs_maxlen=outputs_maxlen,
                         inputs_maxlen=er_length + 1)
        self.er_length = er_length
        self.fast = fast
        self.slow = slow
        self._sc_fast = 2.0 / (fast + 1)
        self._sc_slow = 2.0 / (slow + 1)

    def update(self, ts: datetime, value: float) -> None:
        """Push one observation."""
        self._push(ts, {'value': float(value)})

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, value: float) -> Dict[str, float]:
        n_inputs = len(self._inputs)
        nan = float('nan')

        # Recursion seed lives in hidden ``_kama_raw``; first-ever input
        # (or recovery from a wiped state) seeds it with ``value``.
        prev_raw = (prev_output.get('_kama_raw', nan)
                    if prev_output is not None else nan)
        if not np.isfinite(prev_raw):
            kama_raw = value
        else:
            # Efficiency ratio: ER = |chg| / Σ|diffs| over the trailing window.
            # Need er_length + 1 inputs for er_length differences.
            if n_inputs >= self.er_length + 1:
                cur = self._inputs[-1][1]['value']
                old = self._inputs[-(self.er_length + 1)][1]['value']
                chg = abs(cur - old)
                vol = 0.0
                for i in range(-self.er_length, 0):
                    vol += abs(self._inputs[i][1]['value']
                               - self._inputs[i - 1][1]['value'])
                er = (chg / vol) if vol != 0 else 0.0
            else:
                er = 0.0
            sc = (er * (self._sc_fast - self._sc_slow) + self._sc_slow) ** 2
            kama_raw = prev_raw + sc * (value - prev_raw)

        # min_periods=er_length+1 mask: NaN until first valid ER window.
        kama_out = kama_raw if n_inputs >= self.er_length + 1 else nan
        return {'kama': kama_out, '_kama_raw': kama_raw}

    @staticmethod
    def from_series(src: pd.Series, er_length: int = 10,
                    fast: int = 2, slow: int = 30) -> pd.Series:
        """Vectorized one-shot: matches the stateful ``kama`` column bar-for-bar.

        First ``er_length`` outputs are masked to NaN; the underlying
        recursion runs from input #0 in a private array and is unmasked
        from index ``er_length`` onward.
        """
        n = len(src)
        if n == 0:
            return pd.Series(dtype=float, index=src.index)

        values = src.to_numpy(dtype=float)
        sc_fast = 2.0 / (fast + 1)
        sc_slow = 2.0 / (slow + 1)

        raw = np.empty(n, dtype=float)
        raw[0] = values[0]

        for i in range(1, n):
            if i >= er_length:
                chg = abs(values[i] - values[i - er_length])
                vol = np.abs(np.diff(values[i - er_length:i + 1])).sum()
                er = chg / vol if vol != 0 else 0.0
            else:
                er = 0.0
            sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
            raw[i] = raw[i - 1] + sc * (values[i] - raw[i - 1])

        out = raw.copy()
        if er_length <= n:
            out[:er_length] = np.nan
        else:
            out[:] = np.nan
        return pd.Series(out, index=src.index, dtype=float)
