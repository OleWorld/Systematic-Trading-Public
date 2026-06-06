"""Indicator ABC: stateful, upsert-by-timestamp incremental indicators.

Concrete indicators (SMA, EMA, KAMA, RSI, …) live as sibling modules and
implement ``_compute`` plus a typed ``update`` thin wrapper. Strategies hold
per-symbol indicator instances and feed them one tick at a time via
``update(ts, ...)`` — a typed scalar entry point with a subclass-specific
signature. Callers extract whichever fields they need from the bar in hand.

``_push`` holds the load-bearing invariant:

    _outputs[-1] == current forming entry (mutates with same-ts re-ticks)
    _outputs[-2] == last fully finalized entry (immutable from this point)

Recursive math (KAMA / EMA / RSI / Wilder ATR / trailing stop) folds from the
last finalized output. ``_push`` picks ``prev_output`` accordingly: on a new
bar the previous bar's forming entry is now finalized (read at index -1); on
a re-tick the soon-to-be-overwritten entry is at index -1 and the actual
finalized prior is one slot back at -2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Tuple

import pandas as pd


_DequeEntry = Tuple[datetime, Dict[str, float]]


class Indicator(ABC):
    """Stateful indicator with upsert-by-timestamp semantics on the forming bar.

    State (managed by base class):

    - ``_inputs``: deque of ``(ts, input_dict)``. Window-based subclasses
      (SMA, Stdev, PercentRank, BBW) read the trailing window from here.
      Recursive-only subclasses leave it at the default maxlen=2.
    - ``_outputs``: deque of ``(ts, output_dict)``. Last entry is the
      forming output; the entry one slot back is the last finalized output.
      ``get_latest_indicators(n)`` materializes a DataFrame from this deque.

    Subclasses implement ``_compute(prev_output, **inputs) -> dict``. The base
    class handles upsert and ``prev_output`` selection. Subclasses also
    implement a typed ``update(ts, …)`` wrapper that calls ``self._push``.
    """

    def __init__(self, *, outputs_maxlen: int = 500, inputs_maxlen: int = 2):
        if outputs_maxlen < 2:
            raise ValueError(f"outputs_maxlen must be >= 2, got {outputs_maxlen}")
        if inputs_maxlen < 1:
            raise ValueError(f"inputs_maxlen must be >= 1, got {inputs_maxlen}")
        self._outputs: Deque[_DequeEntry] = deque(maxlen=outputs_maxlen)
        self._inputs: Deque[_DequeEntry] = deque(maxlen=inputs_maxlen)

    # ── core engine ──────────────────────────────

    def _push(self, ts: datetime, inputs: Dict[str, float]) -> None:
        """Upsert input, compute output, upsert output. Picks ``prev_output``
        so recursive math always folds from the last finalized entry.

        New bar (``ts != _outputs[-1].ts``): the previous forming entry is now
        finalized — read it at index -1, then append a new forming entry.

        Re-tick (``ts == _outputs[-1].ts``): the entry at -1 is stale and about
        to be overwritten. The genuinely-finalized prior is at -2. Replace -1.
        """
        is_new_bar = (not self._outputs) or (self._outputs[-1][0] != ts)

        if is_new_bar:
            self._inputs.append((ts, inputs))
            prev = self._outputs[-1][1] if self._outputs else None
        else:
            self._inputs[-1] = (ts, inputs)
            prev = self._outputs[-2][1] if len(self._outputs) >= 2 else None

        value = self._compute(prev, **inputs)

        if is_new_bar:
            self._outputs.append((ts, value))
        else:
            self._outputs[-1] = (ts, value)

    @abstractmethod
    def _compute(self, prev_output: Optional[Dict[str, float]],
                 **inputs: float) -> Dict[str, float]:
        """Compute the output dict for this tick.

        ``prev_output`` is the **last finalized** output dict, or ``None`` if
        no prior finalized entry exists. Recursive subclasses fold from
        ``prev_output``; window-based subclasses read the trailing window
        from ``self._inputs``.
        """
        raise NotImplementedError

    # ── queries ──────────────────────────────────

    @staticmethod
    def _public(vals: Dict[str, float]) -> Dict[str, float]:
        """Filter out ``_``-prefixed keys (internal recursion state).

        Indicators that need to carry hidden recursion state (e.g. EMA's
        unmasked ``_ema_raw`` while emitting masked NaN early outputs) put
        it under a leading-underscore key. Public queries filter it out.
        """
        return {k: v for k, v in vals.items() if not k.startswith('_')}

    def get_latest_indicators(self, n: int) -> pd.DataFrame:
        """Return the last ``n`` output rows as a DataFrame indexed by timestamp.

        Columns are the public keys of the per-tick output dict (leading
        ``_`` keys are filtered). Empty DataFrame if no outputs yet.
        """
        if not self._outputs:
            return pd.DataFrame()
        n_avail = len(self._outputs)
        start = max(0, n_avail - n)
        subset = list(self._outputs)[start:]
        timestamps = [ts for ts, _ in subset]
        rows = [self._public(vals) for _, vals in subset]
        return pd.DataFrame(rows, index=pd.Index(timestamps))

    @property
    def latest(self) -> Optional[pd.Series]:
        """Most-recently finalized output values (``iloc[-2]`` of outputs).

        Returns ``None`` until at least 2 outputs exist (one finalized + one
        forming). Strategies use this for signal logic — by convention, the
        last entry is always treated as forming and only the prior is read.
        ``_``-prefixed keys are filtered from the returned Series.
        """
        if len(self._outputs) < 2:
            return None
        ts, vals = self._outputs[-2]
        return pd.Series(self._public(vals), name=ts)

    @property
    def forming(self) -> Optional[pd.Series]:
        """Current forming output values (``iloc[-1]`` of outputs).

        Returns ``None`` until at least 1 output exists. Used when one
        indicator's forming output is fed into another (e.g. KAMA's forming
        value into ``TrailingVolatilityStop``). ``_``-prefixed keys filtered.
        """
        if not self._outputs:
            return None
        ts, vals = self._outputs[-1]
        return pd.Series(self._public(vals), name=ts)

    @property
    def is_latest_ready(self) -> bool:
        """True iff ``latest`` is non-None and contains no NaNs (public keys only)."""
        latest = self.latest
        if latest is None:
            return False
        return not latest.isna().any()

    @property
    def is_forming_ready(self) -> bool:
        """True iff ``forming`` is non-None and contains no NaNs (public keys only)."""
        forming = self.forming
        if forming is None:
            return False
        return not forming.isna().any()

    # ── lifecycle ────────────────────────────────

    def reset(self) -> None:
        """Clear all state. The indicator returns to its initial post-construction state."""
        self._outputs.clear()
        self._inputs.clear()

    def warmup(self, history: Any) -> None:
        """Bulk-seed by iterating ``update`` over historical data.

        Accepts:

        - ``pd.Series`` — each row is a scalar input. Calls ``update(ts, value)``
          (single-positional). Suitable for single-input indicators.
        - ``pd.DataFrame`` — passes each row as keyword arguments to ``update``.
          Column names must match the subclass's ``update`` parameter names
          (e.g. ``high, low, close`` for ATR; ``price, trigger, atr`` for
          ``TrailingVolatilityStop``). Callers feeding indicators from raw
          OHLCV frames should extract the relevant column(s) themselves.

        NaN values in a Series are skipped. The default is iterative; cheap
        but O(n_rows) Python calls. Subclasses with vectorized seeding may
        override.
        """
        if isinstance(history, pd.Series):
            for ts, v in history.items():
                if pd.isna(v):
                    continue
                self.update(ts, float(v))
            return

        if isinstance(history, pd.DataFrame):
            for ts, row in history.iterrows():
                kwargs = {k: float(v) for k, v in row.items() if not pd.isna(v)}
                if not kwargs:
                    continue
                self.update(ts, **kwargs)
            return

        raise TypeError(
            f"warmup expects pd.Series or pd.DataFrame, got {type(history).__name__}"
        )

    # ── subclass contract ────────────────────────

    def update(self, ts: datetime, *args: float, **kwargs: float) -> None:
        """Typed scalar entry point — subclasses override with their own
        positional/keyword arguments and forward to ``self._push``.

        The base implementation is a generic fallback that forwards positional
        args under generic names (``v0``, ``v1``, …) merged with keyword args.
        Subclasses should always override for clarity and type safety.
        """
        inputs: Dict[str, float] = {f"v{i}": float(v) for i, v in enumerate(args)}
        inputs.update({k: float(v) for k, v in kwargs.items()})
        self._push(ts, inputs)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} outputs={len(self._outputs)} "
            f"inputs={len(self._inputs)}>"
        )
