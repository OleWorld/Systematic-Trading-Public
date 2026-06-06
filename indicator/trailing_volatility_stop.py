"""ATR-anchored trailing volatility stop with direction flip detection.

Stateful translation of the original ``trailing_volatility_stop`` pure
function. The recursion state is ``(stop, direction)``; each step uses the
last finalized ``(stop, direction)`` as ``prev`` (selected by the base
class on re-ticks of the forming bar) and emits a new pair from the
current ``price``, ``trigger``, and ``atr`` scalars.

This indicator does NOT consume OHLCV — its inputs are typically the
outputs of other indicators (e.g. KAMA's value and ATR's value). Feed it
via the typed scalar ``update(ts, price, trigger, atr)``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from indicator._base import Indicator

__all__ = ['TrailingVolatilityStop']


class TrailingVolatilityStop(Indicator):
    """Trailing stop seeded with ``(stop=NaN, direction=+1)`` until ATR finite.

    Output dict carries ``stop`` (float, NaN during warmup) and ``direction``
    (int, ±1; defaults to +1 until the first finite band sets a real stop).
    """

    def __init__(self, mult: float, *, outputs_maxlen: int = 500):
        if mult <= 0:
            raise ValueError(f"mult must be > 0, got {mult}")
        super().__init__(outputs_maxlen=outputs_maxlen, inputs_maxlen=2)
        self.mult = mult

    def update(self, ts: datetime, price: float, trigger: float,
               atr: float) -> None:
        """Push one tick of state-machine inputs.

        ``price`` anchors the band (long band = price - mult*ATR; short band =
        price + mult*ATR). ``trigger`` is the series whose crossing of the
        current stop flips direction. ``atr`` is the band width source; NaN
        means inherit the previous state unchanged.
        """
        self._push(ts, {
            'price': float(price),
            'trigger': float(trigger),
            'atr': float(atr),
        })

    def _compute(self, prev_output: Optional[Dict[str, float]],
                 *, price: float, trigger: float, atr: float) -> Dict[str, float]:
        if prev_output is None:
            prev_dir = 1
            prev_stop = float('nan')
        else:
            prev_dir = int(prev_output['direction'])
            prev_stop = float(prev_output['stop'])

        # NaN ATR → state unchanged (warmup or input dropout).
        if not np.isfinite(atr):
            return {'stop': prev_stop, 'direction': prev_dir}

        band_long = price - self.mult * atr
        band_short = price + self.mult * atr

        if prev_dir == 1:
            # No prior stop (warmup just ended) → take the band directly.
            cur_stop = (band_long if not np.isfinite(prev_stop)
                        else max(band_long, prev_stop))
            if trigger < cur_stop:
                cur_dir = -1
                cur_stop = band_short
            else:
                cur_dir = 1
        elif prev_dir == -1:
            cur_stop = (band_short if not np.isfinite(prev_stop)
                        else min(band_short, prev_stop))
            if trigger > cur_stop:
                cur_dir = 1
                cur_stop = band_long
            else:
                cur_dir = -1
        else:
            raise ValueError(f"Unexpected prev_dir: {prev_dir!r}")

        return {'stop': cur_stop, 'direction': cur_dir}

    @staticmethod
    def from_series(price_src: pd.Series, trigger_src: pd.Series,
                    atr_series: pd.Series,
                    mult: float) -> Tuple[pd.Series, pd.Series]:
        """Vectorized one-shot: returns ``(stop, direction)`` series.

        Matches the stateful indicator's ``stop``/``direction`` columns
        bar-for-bar when fed the same scalar inputs in order.
        """
        n = len(price_src)
        if n == 0:
            empty = pd.Series(dtype=float, index=price_src.index)
            return empty, empty.astype(int)

        price = price_src.to_numpy(dtype=float)
        trig = trigger_src.to_numpy(dtype=float)
        atr_arr = atr_series.to_numpy(dtype=float)

        stop = np.full(n, np.nan, dtype=float)
        direction = np.ones(n, dtype=int)

        prev_stop = float('nan')
        prev_dir = 1

        for i in range(n):
            if np.isnan(atr_arr[i]):
                stop[i] = prev_stop
                direction[i] = prev_dir
                continue

            band_long = price[i] - mult * atr_arr[i]
            band_short = price[i] + mult * atr_arr[i]

            if prev_dir == 1:
                cur_stop = (band_long if not np.isfinite(prev_stop)
                            else max(band_long, prev_stop))
                if trig[i] < cur_stop:
                    cur_dir = -1
                    cur_stop = band_short
                else:
                    cur_dir = 1
            else:
                cur_stop = (band_short if not np.isfinite(prev_stop)
                            else min(band_short, prev_stop))
                if trig[i] > cur_stop:
                    cur_dir = 1
                    cur_stop = band_long
                else:
                    cur_dir = -1

            stop[i] = cur_stop
            direction[i] = cur_dir
            prev_stop = cur_stop
            prev_dir = cur_dir

        return (
            pd.Series(stop, index=price_src.index, dtype=float),
            pd.Series(direction, index=price_src.index, dtype=int),
        )
