import abc
import datetime
import logging
import queue as thread_queue
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from data._alt import AltRecord
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
                 base_timeframe: str, timeframes: Dict[str, int],
                 alt_feeds: Optional[Dict[str, int]] = None):
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
        alt_feeds : dict, optional
            Mapping of ``{feed_name: maxlen}`` for alternative-data
            rolling windows (funding rates, open interest, ...). One
            deque per ``symbol_list × feed`` pair is pre-populated so
            unknown-key lookups raise rather than silently creating
            state. Default ``None`` — no alt feeds.
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

        # Timestamp of the last accepted COMPLETED bar per symbol — the
        # duplicate gate in ``_append_bar``. Forming re-ticks never set it,
        # so the live forming→completed finalization at the same timestamp
        # is still accepted; only a second completed bar at the same
        # timestamp (dirty input) is rejected.
        self._last_completed_ts: Dict[str, Optional[datetime.datetime]] = {
            symbol: None for symbol in symbol_list
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

        # Alt-feed rolling windows — named per-symbol multi-field series
        # (funding rates, open interest, ...). One deque per
        # (symbol, feed) pair, pre-populated for every symbol × feed
        # combination so unknown-key lookups raise rather than silently
        # creating state. An empty deque means "no data yet": a feed
        # does not have to cover every symbol (mirrors how a
        # late-listing symbol looks on the bar side).
        self.alt_feeds: Dict[str, int] = dict(alt_feeds) if alt_feeds else {}
        for feed, maxlen in self.alt_feeds.items():
            if not isinstance(feed, str) or not feed:
                raise ValueError(
                    f"Alt feed names must be non-empty strings, got {feed!r}"
                )
            if not isinstance(maxlen, int) or isinstance(maxlen, bool) \
                    or maxlen <= 0:
                raise ValueError(
                    f"Alt feed {feed!r} maxlen must be a positive int, "
                    f"got {maxlen!r}"
                )
        self._alt_data: Dict[Tuple[str, str], deque[AltRecord]] = {
            (symbol, feed): deque(maxlen=maxlen)
            for feed, maxlen in self.alt_feeds.items()
            for symbol in symbol_list
        }
        # Timestamp of the last accepted record per (symbol, feed) — the
        # duplicate and out-of-order gates in ``_append_alt``.
        self._last_alt_ts: Dict[Tuple[str, str], Optional[datetime.datetime]] = {
            key: None for key in self._alt_data
        }

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

        Upsert semantics apply to FORMING re-ticks only: if the last entry
        of the base deque shares this timestamp, replace it with a new
        ``Bar`` (tracking a forming bar as it ticks in live — the stored
        ``Bar`` is immutable, but the deque slot is overwritten; the
        completed emission that finalizes those forming ticks lands on the
        same timestamp and takes the same replace path). Otherwise append
        a new forming entry.

        A SECOND completed bar at a timestamp that already accepted a
        completed bar is dirty input (e.g. a duplicated row in a
        caller-supplied frame) and is rejected like the NaN case: the
        deque is left untouched (first bar wins), a WARNING is logged,
        and the method returns ``False`` — so the duplicate never
        re-enters the HTF volume accumulator and never emits a second
        ``BarEvent``. A bar whose timestamp is EARLIER than the last
        accepted completed bar raises ``ValueError`` (forming or
        completed): out-of-order input violates the sorted-stream
        contract and everything computed since the true position of that
        bar is suspect — the run dies loudly instead of continuing on
        corrupt state.

        HTF accumulation is gated on ``is_forming`` because live forming
        emissions carry cumulative OHLCV for the in-progress base bar —
        feeding them into the HTF volume accumulator would double-count.
        Backtest always passes ``is_forming=False`` (one emission per bar).

        Returns
        -------
        bool
            ``True`` if the bar was accepted and stored; ``False`` if it was
            rejected for NaN OHLC or as a duplicate completed bar. A bar
            earlier than the last accepted completed bar raises
            ``ValueError`` instead of returning.
        """
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            logger.warning(
                "Dropping bar with NaN OHLC: symbol=%s ts=%s O=%s H=%s L=%s C=%s",
                symbol, ts, o, h, l, c,
            )
            return False
        last_completed = self._last_completed_ts[symbol]
        if last_completed is not None and ts < last_completed:
            raise ValueError(
                f"Out-of-order bar for {symbol!r}: ts={ts} is earlier than "
                f"the last accepted completed bar at {last_completed}. "
                f"Bars must arrive in ascending time order — a backward "
                f"bar would corrupt HTF aggregates, indicator recursion, "
                f"and the latest-price cache, so the run is aborted "
                f"rather than continued on corrupt state."
            )
        if not is_forming and ts == last_completed:
            logger.warning(
                "Dropping duplicate completed bar: symbol=%s ts=%s "
                "(a completed bar at this timestamp was already accepted; "
                "first bar wins)",
                symbol, ts,
            )
            return False

        bar = Bar(ts, o, h, l, c, v)
        deq = self._base_bar_data[symbol]
        if deq and deq[-1].timestamp == ts:
            deq[-1] = bar
        else:
            deq.append(bar)

        if not is_forming:
            self._last_completed_ts[symbol] = ts
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

        NaN base volumes (allowed through by design — see ``_append_bar``)
        are treated as ``0.0`` in the HTF volume sum: one flaky tick must
        not poison the whole period's aggregate with NaN. The HTF volume
        is therefore the sum of the period's *non-NaN* base volumes.
        """
        key = (symbol, timeframe)
        period_start = get_period_start(ts, timeframe)
        deq = self._htf_bar_data[key]
        v_safe = 0.0 if pd.isna(v) else v

        if not deq or deq[-1].timestamp != period_start:
            deq.append(Bar(period_start, o, h, l, c, v_safe))
        else:
            last = deq[-1]
            deq[-1] = Bar(
                timestamp=last.timestamp,
                open=last.open,
                high=max(last.high, h),
                low=min(last.low, l),
                close=c,
                volume=last.volume + v_safe,
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

    def _select_deque(self, symbol: str,
                      timeframe: Optional[str]) -> "deque[Bar]":
        """Resolve and return the bar deque for ``symbol`` at ``timeframe``.

        ``timeframe`` ``None`` or equal to ``base_timeframe`` selects the
        base-TF deque; otherwise the registered HTF deque. Shared by
        ``get_latest_bars`` / ``get_latest_bars_df`` / ``count_bars``.

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
            return self._base_bar_data[symbol]
        if timeframe not in self.timeframes:
            available = list(self.timeframes.keys())
            raise ValueError(
                f"Timeframe '{timeframe}' not registered. "
                f"Available: {available}"
            )
        return self._htf_bar_data[(symbol, timeframe)]

    def get_latest_bars(self, symbol: str, n: int = 1,
                        timeframe: Optional[str] = None) -> List[Bar]:
        """Return the last *n* bars as a list of ``Bar`` objects (oldest→newest).

        Reads straight from the deque — **no DataFrame construction** — so it
        is the cheap per-bar accessor for hot paths (strategies, vol
        estimators picking off ``bar.close`` / ``bar.timestamp``). The last
        element (``[-1]``) is the current **forming** bar — it can still
        mutate (in live, as new ticks arrive for the same timestamp; at HTF,
        as more base bars close within the current period). Use ``[-2]`` and
        earlier for signal logic that must only see completed bars. Returns up
        to *n* bars (fewer if the deque is shorter), empty list if none yet.
        For a DataFrame view, use ``get_latest_bars_df``.

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
        deq = self._select_deque(symbol, timeframe)
        if n <= 0 or not deq:
            return []
        start = max(0, len(deq) - n)
        return list(deq)[start:]

    def get_latest_bars_df(self, symbol: str, n: int = 1,
                           timeframe: Optional[str] = None) -> pd.DataFrame:
        """Returns the last *n* bars as a DataFrame with datetime index.

        Columns: Open, High, Low, Close, Volume (float64). The
        DataFrame-materializing counterpart to ``get_latest_bars`` — for
        research, plotting, correlation windows, and other non-hot callers
        that want a frame. The last row (``iloc[-1]``) is the current
        **forming** bar (see ``get_latest_bars``).

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
        return self._deque_to_df(self._select_deque(symbol, timeframe), n)

    def count_bars(self, symbol: str,
                   timeframe: Optional[str] = None) -> int:
        """Number of bars currently stored for ``symbol`` at ``timeframe``.

        O(1) deque length — the cheap way to test data-availability gates
        (e.g. "are there at least ``corr_lookback`` bars yet?") without
        materializing a DataFrame. The count includes the current forming bar.

        Raises
        ------
        ValueError
            If ``symbol`` was not registered via ``symbol_list``, or if
            ``timeframe`` was not registered via ``timeframes``.
        """
        return len(self._select_deque(symbol, timeframe))

    # ── alt-feed storage & queries ───────────────

    def _append_alt(self, feed: str, symbol: str, ts: datetime.datetime,
                    values: Mapping[str, float]) -> bool:
        """Append one alt record to the ``(symbol, feed)`` rolling window.

        The ingestion seam for alternative data, shaped like
        ``_append_bar``: validates, stores, and returns
        accepted/rejected so a future event-emitting variant (or a
        portfolio-side consumer such as funding-PnL accounting) can gate
        on the return value exactly like the bar path gates
        ``events_queue.put`` on ``_append_bar``.

        Gates (mirroring ``_append_bar``):
        - Any NaN field value drops the WHOLE record (WARNING, returns
          ``False``) — keeps the "everything in a window is non-NaN"
          invariant uniform across bars and alt data.
        - A second record at an already-accepted timestamp for this
          ``(symbol, feed)`` is dropped (WARNING, first record wins).
        - A record EARLIER than the last accepted one raises
          ``ValueError`` — unreachable via the sorted+heap-merged
          historic path, kept as the backstop for a future live handler.

        Unlike bars there is no forming/completed distinction: every
        accepted record is final on arrival.

        Returns
        -------
        bool
            ``True`` if the record was accepted and stored; ``False``
            if it was rejected for a NaN field or as a duplicate.

        Raises
        ------
        ValueError
            If ``(symbol, feed)`` was not registered, or on an
            out-of-order timestamp.
        """
        key = (symbol, feed)
        deq = self._alt_data.get(key)
        if deq is None:
            raise ValueError(
                f"Unknown alt target: feed={feed!r} symbol={symbol!r}. "
                f"Registered feeds: {sorted(self.alt_feeds)}; registered "
                f"symbols: {self.symbol_list}"
            )
        if any(pd.isna(v) for v in values.values()):
            logger.warning(
                "Dropping alt record with NaN field: feed=%s symbol=%s "
                "ts=%s values=%s", feed, symbol, ts, dict(values),
            )
            return False
        last_ts = self._last_alt_ts[key]
        if last_ts is not None and ts < last_ts:
            raise ValueError(
                f"Out-of-order alt record for feed={feed!r} "
                f"symbol={symbol!r}: ts={ts} is earlier than the last "
                f"accepted record at {last_ts}. Records must arrive in "
                f"ascending time order."
            )
        if ts == last_ts:
            logger.warning(
                "Dropping duplicate alt record: feed=%s symbol=%s ts=%s "
                "(a record at this timestamp was already accepted; first "
                "record wins)", feed, symbol, ts,
            )
            return False
        deq.append(AltRecord(ts, values))
        self._last_alt_ts[key] = ts
        return True

    def _select_alt_deque(self, symbol: str, feed: str) -> "deque[AltRecord]":
        """Resolve the alt deque for ``(symbol, feed)``.

        Raises
        ------
        ValueError
            If ``feed`` was not registered via ``alt_feeds``, or
            ``symbol`` was not registered via ``symbol_list``.
        """
        deq = self._alt_data.get((symbol, feed))
        if deq is None:
            if feed not in self.alt_feeds:
                raise ValueError(
                    f"Unknown alt feed '{feed}'. Registered: "
                    f"{sorted(self.alt_feeds)}"
                )
            raise ValueError(
                f"Unknown symbol '{symbol}'. Registered: {self.symbol_list}"
            )
        return deq

    def get_latest_alt(self, symbol: str, feed: str,
                       n: int = 1) -> List[AltRecord]:
        """Return the last *n* alt records (oldest→newest) for a feed.

        Reads straight from the deque — no DataFrame construction — the
        cheap per-bar accessor for strategy hot paths. UNLIKE bar
        windows, ``[-1]`` is a FINALIZED record: alt data has no
        forming/completed distinction. Returns up to *n* records (fewer
        if the window is shorter), empty list if none yet.

        Raises
        ------
        ValueError
            If ``symbol`` or ``feed`` was not registered.
        """
        deq = self._select_alt_deque(symbol, feed)
        if n <= 0 or not deq:
            return []
        start = max(0, len(deq) - n)
        return list(deq)[start:]

    def get_latest_alt_df(self, symbol: str, feed: str,
                          n: int = 1) -> pd.DataFrame:
        """Return the last *n* alt records as a DataFrame.

        Datetime index, one column per field — the research/plotting
        counterpart to ``get_latest_alt``. An empty window returns an
        empty DataFrame with no columns (field names are unknown until
        the first record arrives).

        Raises
        ------
        ValueError
            If ``symbol`` or ``feed`` was not registered.
        """
        records = self.get_latest_alt(symbol, feed, n)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(
            [dict(r.values) for r in records],
            index=[r.timestamp for r in records],
        )

    def count_alt(self, symbol: str, feed: str) -> int:
        """Number of records stored for ``(symbol, feed)`` — O(1).

        The cheap availability gate (e.g. "any funding data yet?")
        mirroring ``count_bars``.

        Raises
        ------
        ValueError
            If ``symbol`` or ``feed`` was not registered.
        """
        return len(self._select_alt_deque(symbol, feed))
