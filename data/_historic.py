import datetime
import logging
import queue as thread_queue
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import pandas as pd

from data._base import DataHandler
from data._timeframe import (
    _parse_date, parse_timeframe_to_seconds,
    TIMEFRAME_FALLBACK_ORDER,
)
from data._arctic import _make_symbol_key, _init_arctic_lib
from data._ohlcv import _resample_ohlcv
from event import BarEvent

logger = logging.getLogger(__name__)


class HistoricDataHandler(DataHandler):
    """HistoricDataHandler is designed for backtesting.

    It reads from ArcticDB (or accepts pre-built DataFrames) and yields
    bars one by one in time-sorted order across all symbols.
    """

    def __init__(self, events_queue: thread_queue.Queue[Any], symbol_list: List[str],
                 base_timeframe: str,
                 timeframes: Dict[str, int],
                 start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None,
                 db_path: str = "arctic_data",
                 data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize for backtesting.

        Two data paths: pass ``data={symbol: df}`` to feed DataFrames directly,
        or pass ``start_date``/``end_date``/``db_path`` to load from ArcticDB
        (with automatic fallback resampling from finer timeframes).
        """
        super().__init__(events_queue, symbol_list, base_timeframe, timeframes)

        if data is not None:
            self._bar_generators = self._build_stream(data)
        else:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date are required when data is not provided.")
            self.start_date = _parse_date(start_date)
            self.end_date = _parse_date(end_date)

            try:
                self.store, self.lib = _init_arctic_lib(db_path)
            except Exception as e:
                logger.error("Error initializing ArcticDB: %s", e)
                raise

            self._bar_generators = self._build_stream(self._load_from_db())

    # ── DB loading (decomposed) ─────────────────

    def _load_from_db(self) -> Dict[str, pd.DataFrame]:
        """Read per-symbol DataFrames from ArcticDB, with fallback resampling."""
        dataframes: Dict[str, pd.DataFrame] = {}
        target_seconds = parse_timeframe_to_seconds(self.base_timeframe)

        for symbol in self.symbol_list:
            symbol_key = _make_symbol_key(symbol, self.base_timeframe)
            df = self._try_read_exact(symbol_key)

            if df is None or df.empty:
                df = self._try_read_with_fallback(symbol, target_seconds)

            if df is None or df.empty:
                logger.warning("No data for %s in range.", symbol)
                continue

            dataframes[symbol] = df

        return dataframes

    def _try_read_exact(self, symbol_key: str) -> Optional[pd.DataFrame]:
        """Try reading the exact timeframe key from ArcticDB."""
        if not self.lib.has_symbol(symbol_key):
            return None
        try:
            return self.lib.read(
                symbol_key,
                date_range=(self.start_date, self.end_date)
            ).data
        except Exception as e:
            logger.error("Error reading %s: %s", symbol_key, e)
            return None

    def _try_read_with_fallback(self, symbol: str,
                                target_seconds: int) -> Optional[pd.DataFrame]:
        """Find a more granular timeframe in DB and resample up."""
        for fallback_tf in TIMEFRAME_FALLBACK_ORDER:
            if parse_timeframe_to_seconds(fallback_tf) >= target_seconds:
                continue

            fallback_key = _make_symbol_key(symbol, fallback_tf)
            if not self.lib.has_symbol(fallback_key):
                continue

            try:
                raw_df = self.lib.read(
                    fallback_key,
                    date_range=(self.start_date, self.end_date)
                ).data
                if raw_df.empty:
                    continue

                logger.info("Resampling %s -> %s for %s",
                            fallback_key, self.base_timeframe, symbol)
                return _resample_ohlcv(raw_df, self.base_timeframe)
            except Exception as e:
                logger.error("Error reading fallback %s: %s", fallback_key, e)
                continue

        return None

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
