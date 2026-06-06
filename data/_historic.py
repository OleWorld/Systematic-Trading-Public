import logging
import queue as thread_queue
from typing import Any, Dict, Generator, Iterator, List, Tuple

import pandas as pd

from data._base import DataHandler
from event import BarEvent

logger = logging.getLogger(__name__)


class HistoricDataHandler(DataHandler):
    """HistoricDataHandler is designed for backtesting.

    It is fed pre-built per-symbol OHLCV DataFrames (``data={symbol: df}``) and
    yields bars one by one in time-sorted order across all symbols.

    Each DataFrame must be indexed by a timezone-aware ``DatetimeIndex`` and
    expose ``Open``/``High``/``Low``/``Close``/``Volume`` columns. Sourcing,
    cleaning, and windowing the data is the caller's responsibility.
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

        self._bar_generators = self._build_stream(data)

    # ── stream construction ─────────────────────

    def _build_stream(self, dataframes: Dict[str, pd.DataFrame]) -> Generator[Tuple[str, Any], None, None]:
        """Convert per-symbol DataFrames into a single time-sorted bar generator."""
        generators = {sym: df.itertuples() for sym, df in dataframes.items()
                      if df is not None and not df.empty}
        return self._merge_generators(generators)

    def _merge_generators(self, generators: Dict[str, Iterator[Any]]) -> Generator[Tuple[str, Any], None, None]:
        """Merge multiple symbol generators into a single time-sorted stream.

        Uses a current-heads dict to track the next row from each symbol,
        always yielding the row with the earliest timestamp.
        """
        current_heads: Dict[str, Any] = {}

        for sym, gen in generators.items():
            try:
                current_heads[sym] = next(gen)
            except StopIteration:
                pass

        while current_heads:
            earliest_sym = min(current_heads, key=lambda s: current_heads[s].Index)
            row = current_heads[earliest_sym]

            yield (earliest_sym, row)

            try:
                current_heads[earliest_sym] = next(generators[earliest_sym])
            except StopIteration:
                del current_heads[earliest_sym]

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
