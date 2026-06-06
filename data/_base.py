import abc
import datetime
import logging
import queue as thread_queue
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data._bar import Bar
from data._timeframe import get_period_start

logger = logging.getLogger(__name__)


class DataHandler(abc.ABC):
    """Abstract Base Class for DataHandlers.

    Responsible for:
    - Maintaining rolling windows of market data (bars, and future: liquidations, funding, etc.).
    - Emitting events to the system event queue for strategy consumption.
    """

    def __init__(self, events_queue: thread_queue.Queue[Any], symbol_list: List[str],
                 base_timeframe: str, timeframes: Dict[str, int]):
        """Initialize the data handler with rolling windows and HTF accumulators.

        Parameters
        ----------
        events_queue : Queue
            Thread-safe queue for emitting BarEvent objects.
        symbol_list : list of str
            Trading symbols to track (e.g. ``['BTC_USDT', 'ETH_USDT']``).
        base_timeframe : str
            Base bar timeframe (e.g. ``'1m'``, ``'1h'``).
        timeframes : dict
            Mapping of ``{timeframe_string: maxlen}`` for rolling windows.
            Must include ``base_timeframe`` as a key.
        """
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        self.base_timeframe = base_timeframe
        self.timeframes = timeframes  # {tf_string: maxlen}
        self.continue_backtest = True

        # Base TF rolling window: Symbol -> Deque of immutable Bar instances.
        # Pre-populated for every symbol in ``symbol_list`` so unknown-symbol
        # lookups raise rather than silently creating an empty deque.
        base_maxlen = self.timeframes[self.base_timeframe]
        self._base_bar_data: Dict[str, deque[Bar]] = {
            symbol: deque(maxlen=base_maxlen) for symbol in symbol_list
        }

        # HTF deques — one per (symbol, tf), only for non-base timeframes.
        # The LAST entry is the current forming HTF bar; earlier entries are
        # final. The forming bar is rebuilt (replaced) each time a new base
        # bar lands in the same period — the stored Bar itself is immutable.
        self._htf_bar_data: Dict[Tuple[str, str], deque[Bar]] = {}
        for tf, tf_maxlen in self.timeframes.items():
            if tf == self.base_timeframe:
                continue
            for symbol in symbol_list:
                self._htf_bar_data[(symbol, tf)] = deque(maxlen=tf_maxlen)

    @abc.abstractmethod
    def update_bar(self) -> None:
        """Push the latest bar(s) to the event queue."""
        raise NotImplementedError("Implement update_bar()")

    # ── bar storage ──────────────────────────────

    def _append_bar(self, symbol: str, ts: datetime.datetime,
                    o: float, h: float, l: float, c: float, v: float,
                    *, is_forming: bool = False) -> bool:
        """Upsert the latest base bar; propagate to HTF only when complete.

        The ``DataHandler`` is the single source of truth for non-NaN OHLC.
        Bars whose ``open``/``high``/``low``/``close`` contain a NaN are
        rejected here: the deque is left untouched, a WARNING is logged,
        and the method returns ``False``. Volume is intentionally not
        validated — a NaN volume from a flaky CCXT tick should not drop
        the bar, since downstream accounting/sizing/execution don't depend
        on volume. Callers that also queue a ``BarEvent`` should gate the
        ``events_queue.put`` on the return value so a rejected bar never
        reaches downstream modules.

        Upsert semantics: if the last entry of the base deque shares this
        timestamp, replace it with a new ``Bar`` (tracking a forming bar
        as it ticks in live — the stored ``Bar`` is immutable, but the
        deque slot is overwritten). Otherwise append a new forming entry.

        HTF accumulation is gated on ``is_forming`` because live forming
        emissions carry cumulative OHLCV for the in-progress base bar —
        feeding them into the HTF volume accumulator would double-count.
        Backtest always passes ``is_forming=False`` (one emission per bar).

        Returns
        -------
        bool
            ``True`` if the bar was accepted and stored; ``False`` if it was
            rejected for NaN OHLC.
        """
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            logger.warning(
                "Dropping bar with NaN OHLC: symbol=%s ts=%s O=%s H=%s L=%s C=%s",
                symbol, ts, o, h, l, c,
            )
            return False

        bar = Bar(ts, o, h, l, c, v)
        deq = self._base_bar_data[symbol]
        if deq and deq[-1].timestamp == ts:
            deq[-1] = bar
        else:
            deq.append(bar)

        if not is_forming:
            for tf in self.timeframes:
                if tf == self.base_timeframe:
                    continue
                self._update_htf_bar(symbol, tf, ts, o, h, l, c, v)

        return True

    def _update_htf_bar(self, symbol: str, timeframe: str,
                        ts: datetime.datetime,
                        o: float, h: float, l: float, c: float, v: float) -> None:
        """Aggregate a completed base bar into the forming HTF bar.

        The forming HTF bar is the last entry of ``_htf_bar_data[(sym, tf)]``.
        A base bar that opens a new HTF period appends a fresh forming entry
        (the previous last entry is already final and is not re-touched).
        A base bar in the same period replaces the last entry with a new
        ``Bar`` carrying updated H/L/C/V — the stored ``Bar`` itself is
        immutable, but the deque slot is overwritten so the forming bar
        is visible to ``get_latest_bars``. Aggregation is built only from
        completed base bars, which keeps volume correct.
        """
        key = (symbol, timeframe)
        period_start = get_period_start(ts, timeframe)
        deq = self._htf_bar_data[key]

        if not deq or deq[-1].timestamp != period_start:
            deq.append(Bar(period_start, o, h, l, c, v))
        else:
            last = deq[-1]
            deq[-1] = Bar(
                timestamp=last.timestamp,
                open=last.open,
                high=max(last.high, h),
                low=min(last.low, l),
                close=c,
                volume=last.volume + v,
            )

    # ── queries ──────────────────────────────────

    def _deque_to_df(self, data_deque: deque, n: int) -> pd.DataFrame:
        """Convert the last *n* entries of a bar deque to a DataFrame."""
        if len(data_deque) == 0:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        n_available = len(data_deque)
        start_idx = max(0, n_available - n)
        subset = list(data_deque)[start_idx:]
        timestamps = [bar.timestamp for bar in subset]
        ohlcv = [(bar.open, bar.high, bar.low, bar.close, bar.volume)
                 for bar in subset]
        return pd.DataFrame(ohlcv, index=timestamps,
                            columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def get_latest_bars(self, symbol: str, n: int = 1,
                        timeframe: Optional[str] = None) -> pd.DataFrame:
        """Returns the last *n* bars as a DataFrame with datetime index.

        Columns: Open, High, Low, Close, Volume (float64).

        The last row (``iloc[-1]``) is the current **forming** bar — it can
        still mutate (in live, as new ticks arrive for the same timestamp;
        at HTF, as more base bars close within the current period). Use
        ``iloc[-2]`` and earlier rows for signal logic that must only see
        completed bars.

        Parameters
        ----------
        timeframe : str, optional
            If None or equal to base_timeframe, returns base-TF bars.
            Otherwise returns bars from the requested HTF deque.
            The timeframe must have been registered in the ``timeframes`` dict.

        Raises
        ------
        ValueError
            If ``symbol`` was not registered via ``symbol_list``, or if
            ``timeframe`` was not registered via ``timeframes``.
        """
        if symbol not in self._base_bar_data:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Registered: {self.symbol_list}"
            )

        if timeframe is None or timeframe == self.base_timeframe:
            return self._deque_to_df(self._base_bar_data[symbol], n)

        if timeframe not in self.timeframes:
            available = list(self.timeframes.keys())
            raise ValueError(
                f"Timeframe '{timeframe}' not registered. "
                f"Available: {available}"
            )

        return self._deque_to_df(self._htf_bar_data[(symbol, timeframe)], n)

    def get_latest_bar(self, symbol: str,
                       timeframe: Optional[str] = None) -> Optional[Bar]:
        """Return the last (forming) ``Bar`` directly from the deque, or ``None``.

        Thin convenience that skips the ``DataFrame`` round-trip in
        ``get_latest_bars(symbol, 1, timeframe)``. Useful when picking off
        scalar fields (e.g. ``bar.close``, ``bar.high``) to feed a stateful
        indicator's ``update(ts, ...)``.

        Parameters
        ----------
        timeframe : str, optional
            If None or equal to ``base_timeframe``, returns the latest base
            bar. Otherwise returns the latest bar from the requested HTF
            deque. Must have been registered in ``timeframes``.

        Raises
        ------
        ValueError
            If ``symbol`` was not registered via ``symbol_list``, or if
            ``timeframe`` was not registered via ``timeframes``.
        """
        if symbol not in self._base_bar_data:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Registered: {self.symbol_list}"
            )

        if timeframe is None or timeframe == self.base_timeframe:
            deq = self._base_bar_data[symbol]
            return deq[-1] if deq else None

        if timeframe not in self.timeframes:
            available = list(self.timeframes.keys())
            raise ValueError(
                f"Timeframe '{timeframe}' not registered. "
                f"Available: {available}"
            )

        deq = self._htf_bar_data[(symbol, timeframe)]
        return deq[-1] if deq else None
