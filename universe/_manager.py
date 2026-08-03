"""UniverseManager — dynamic tradable-universe state, history, and events.

Owns per-symbol liveness: two MEASURED gates (strategy warmup ->
'warmup_forecast'; min_history_bars at history_timeframe ->
'warmup_history') plus externally-pushed EXCLUSION MARKS (reasoned
strings via mark_excluded/clear_excluded — Task 4). live <=> no reason
present; nothing is permanent — symbols exit and re-enter as gates and
marks change. Knows nothing about weights, correlation, or sizing.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from universe._status import UniverseStatus

logger = logging.getLogger(__name__)


class _StrategyLike(Protocol):
    """Subset of the Strategy surface that UniverseManager reads from.

    ``symbol_list`` is the declared universe the manager evaluates and
    exposes back via its own ``symbol_list`` property.
    ``is_warmed_up(symbol)`` is the strategy's measured end-of-warmup
    signal (True once the first non-NaN forecast has been cached) —
    consumed by ``_derive_reasons`` as the strategy gate: unmet, it
    contributes the ``'warmup_forecast'`` reason.
    """

    symbol_list: List[str]

    def is_warmed_up(self, symbol: str) -> bool: ...


class _DataHandlerLike(Protocol):
    """Subset of the DataHandler surface that UniverseManager reads from.

    ``timeframes`` is read at construction time to validate that the
    configured ``history_timeframe`` is registered (and to read its
    deque maxlen for the ``min_history_bars`` bound). ``count_bars`` is
    the O(1) data-availability gate ``_derive_reasons`` uses as the data
    gate: fewer than ``min_history_bars`` at ``history_timeframe``
    contributes the ``'warmup_history'`` reason.
    """

    timeframes: Dict[str, int]

    def count_bars(self, symbol: str,
                   timeframe: Optional[str] = None) -> int: ...


class UniverseManager:
    """Single source of truth for symbol liveness (see module docstring)."""

    def __init__(self, strategy: _StrategyLike, data_handler: _DataHandlerLike,
                 min_history_bars: int, history_timeframe: str = '1d'):
        """Validate the data gate and run the initial evaluation.

        min_history_bars: bars required at history_timeframe (O(1)
        count_bars) before the data gate passes; reason 'warmup_history'
        while unmet. Must be >= 1 and <= the timeframe's deque maxlen.
        Raises ValueError on invalid params.
        """
        if history_timeframe not in data_handler.timeframes:
            raise ValueError(
                f"history_timeframe '{history_timeframe}' not registered in "
                f"data_handler.timeframes; available: "
                f"{list(data_handler.timeframes.keys())}"
            )
        if min_history_bars < 1:
            raise ValueError(
                f"min_history_bars must be >= 1, got {min_history_bars}"
            )
        maxlen = data_handler.timeframes[history_timeframe]
        if min_history_bars > maxlen:
            raise ValueError(
                f"min_history_bars ({min_history_bars}) exceeds the "
                f"'{history_timeframe}' deque maxlen ({maxlen}); no symbol "
                f"could ever go live."
            )
        self.strategy = strategy
        self.data_handler = data_handler
        self.min_history_bars = min_history_bars
        self.history_timeframe = history_timeframe

        self._universe: Dict[str, UniverseStatus] = {}
        self._marks: Dict[str, List[str]] = {}      # exclusion marks, in mark order
        self._pending_events: List[Any] = []        # UniverseEvent (Task 4)
        self._log_rows: List[Dict[str, Any]] = []
        self._last_bar_ts: Optional[Any] = None
        for s in self.strategy.symbol_list:
            self._refresh(s, timestamp=None, trigger='initial')

    @property
    def symbol_list(self) -> List[str]:
        """The declared universe (delegates to the strategy)."""
        return list(self.strategy.symbol_list)

    # ── Measurement & derivation ─────────────────────────────────────

    def _derive_reasons(self, symbol: str) -> List[str]:
        """Measure gates + append marks, in canonical order."""
        reasons: List[str] = []
        if not self.strategy.is_warmed_up(symbol):
            reasons.append('warmup_forecast')
        if self.data_handler.count_bars(
                symbol, timeframe=self.history_timeframe) < self.min_history_bars:
            reasons.append('warmup_history')
        reasons.extend(self._marks.get(symbol, []))
        return reasons

    def _refresh(self, symbol: str, timestamp: Any, trigger: str) -> None:
        """Re-derive one symbol's status; log + emit on a real transition.

        Initial evaluations append a log row but emit no event (no edge).
        """
        reasons = self._derive_reasons(symbol)
        live = not reasons
        excluded = bool(self._marks.get(symbol))
        old = self._universe.get(symbol)
        if old is not None and old.live == live and old.reasons == reasons:
            return
        status = UniverseStatus(live=live, excluded=excluded,
                                reasons=list(reasons))
        self._universe[symbol] = status
        self._log_rows.append({
            'timestamp': timestamp, 'symbol': symbol, 'live': live,
            'excluded': excluded, 'reasons': ','.join(reasons),
            'trigger': trigger,
        })
        if old is not None:
            self._emit(symbol, old, status, timestamp, trigger)
            logger.info(
                "%s universe: %s -> %s %s (trigger=%s)", symbol,
                'live' if old.live else 'not live',
                'live' if live else 'not live', reasons, trigger,
            )

    def _emit(self, symbol: str, old: UniverseStatus, new: UniverseStatus,
              timestamp: Any, trigger: str) -> None:
        """Queue a UniverseEvent for the engine to drain (Task 4)."""
        try:
            from event import UniverseEvent    # local import; Task 4 adds the type
        except ImportError:
            return
        self._pending_events.append(UniverseEvent(
            timestamp=timestamp, symbol=symbol, live=new.live,
            excluded=new.excluded, reasons=list(new.reasons),
            prev_live=old.live, prev_reasons=list(old.reasons),
            trigger=trigger,
        ))

    # ── Engine-driven per-bar hook ───────────────────────────────────

    def update_bar(self, event: Any) -> None:
        """Refresh the event symbol's status (O(1)); gates only change on
        a symbol's own bars, so per-symbol refresh keeps all symbols
        fresh. Runs on forming and completed bars alike."""
        self._last_bar_ts = event.timestamp
        if event.symbol in self._universe:
            self._refresh(event.symbol, event.timestamp, 'bar_refresh')

    # ── Read surface ─────────────────────────────────────────────────

    def status(self, symbol: str) -> UniverseStatus:
        """Defensive copy of the symbol's current status; lazily evaluates
        symbols first seen after construction. Raises ValueError for a
        symbol outside strategy.symbol_list."""
        st = self._universe.get(symbol)
        if st is None:
            if symbol not in self.strategy.symbol_list:
                raise ValueError(f"Unknown symbol {symbol!r}: not in "
                                 f"strategy.symbol_list")
            self._refresh(symbol, self._last_bar_ts, 'initial')
            st = self._universe[symbol]
        return UniverseStatus(live=st.live, excluded=st.excluded,
                              reasons=list(st.reasons))

    def get_live_symbols(self) -> List[str]:
        """Live symbols, in strategy.symbol_list order."""
        return [s for s in self.strategy.symbol_list
                if s in self._universe and self._universe[s].live]

    def get_transition_log(self) -> pd.DataFrame:
        """One row per initial evaluation + per transition:
        timestamp, symbol, live, excluded, reasons (comma-joined), trigger."""
        if not self._log_rows:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'live',
                                         'excluded', 'reasons', 'trigger'])
        return pd.DataFrame(self._log_rows)

    def drain_events(self) -> List[Any]:
        """Return and clear pending UniverseEvents (emission order)."""
        out, self._pending_events = self._pending_events, []
        return out
