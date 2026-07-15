import heapq
import logging
import queue as thread_queue
from typing import Any, Dict, Generator, Iterator, List, Tuple

import pandas as pd

from data._base import DataHandler
from data._tz import ensure_utc_index
from event import BarEvent

logger = logging.getLogger(__name__)


class HistoricDataHandler(DataHandler):
    """HistoricDataHandler is designed for backtesting.

    It is fed pre-built per-symbol OHLCV DataFrames (``data={symbol: df}``) and
    yields bars one by one in time-sorted order across all symbols.

    Each DataFrame must be indexed by a timezone-aware ``DatetimeIndex`` and
    expose ``Open``/``High``/``Low``/``Close``/``Volume`` columns. Sourcing,
    cleaning, and windowing the data is the caller's responsibility. Naive-indexed
    frames raise ``ValueError`` at construction; tz-aware non-UTC frames are converted
    to UTC. Each frame's index must be sorted ascending — unsorted input raises
    ``ValueError`` at construction (same-timestamp adjacent duplicates are
    tolerated here and handled by the stream gate: first bar wins).
    """

    def __init__(self, events_queue: thread_queue.Queue[Any], symbol_list: List[str],
                 base_timeframe: str,
                 timeframes: Dict[str, int],
                 data: Dict[str, pd.DataFrame]):
        """Initialize for backtesting from in-memory DataFrames.

        ``data`` maps each symbol to a time-indexed OHLCV DataFrame and is the
        sole data source. Raises ``ValueError`` if it is missing or empty.
        """
        super().__init__(events_queue, symbol_list, base_timeframe, timeframes)

        if not data:
            raise ValueError(
                "data is required: pass data={symbol: DataFrame} with a "
                "tz-aware DatetimeIndex and Open/High/Low/Close/Volume columns."
            )

        normalized: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            idx = ensure_utc_index(df.index, f"data[{sym!r}]")
            if not idx.is_monotonic_increasing:
                raise ValueError(
                    f"data[{sym!r}] index is not sorted ascending: bars "
                    f"must be supplied in time order. An out-of-order bar "
                    f"would silently corrupt HTF aggregates, indicator "
                    f"recursion, and the latest-price cache, so the run "
                    f"is refused up front."
                )
            # set_axis returns a new frame sharing the data — the caller's
            # frame is never mutated and nothing is copied.
            normalized[sym] = df.set_axis(idx)

        self._bar_generators = self._build_stream(normalized)

    # ── stream construction ─────────────────────

    def _build_stream(self, dataframes: Dict[str, pd.DataFrame]) -> Generator[Tuple[str, Any], None, None]:
        """Convert per-symbol DataFrames into a single time-sorted bar generator."""
        generators = {sym: df.itertuples() for sym, df in dataframes.items()
                      if df is not None and not df.empty}
        return self._merge_generators(generators)

    def _merge_generators(self, generators: Dict[str, Iterator[Any]]) -> Generator[Tuple[str, Any], None, None]:
        """Merge multiple symbol generators into a single time-sorted stream.

        A heap of ``(timestamp, seq, symbol, row)`` heads yields the
        earliest-timestamp row in O(log S) per bar (S = symbols with data
        remaining). ``seq`` is the symbol's insertion index in
        ``generators``, so equal timestamps break to the FIRST-inserted
        symbol — exactly the tie-break the previous linear ``min()`` scan
        over the heads dict produced. ``(timestamp, seq)`` is unique per
        heap entry (one head per symbol), so the row itself is never
        compared.
        """
        heap: List[Tuple[Any, int, str, Any]] = []
        for seq, (sym, gen) in enumerate(generators.items()):
            try:
                row = next(gen)
            except StopIteration:
                continue
            heap.append((row.Index, seq, sym, row))
        heapq.heapify(heap)

        while heap:
            _, seq, sym, row = heapq.heappop(heap)

            yield (sym, row)

            try:
                nxt = next(generators[sym])
            except StopIteration:
                continue
            heapq.heappush(heap, (nxt.Index, seq, sym, nxt))

    def update_bar(self) -> None:
        """Pushes the next bar to the queue."""
        try:
            symbol, row = next(self._bar_generators)

            bar = BarEvent(
                symbol=symbol,
                timestamp=row.Index,
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=float(row.Volume),
                period=self.base_timeframe
            )

            if self._append_bar(symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume):
                self.events_queue.put(bar)

        except StopIteration:
            self.continue_backtest = False
