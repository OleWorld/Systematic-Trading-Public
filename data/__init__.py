"""data — Market data handlers for backtesting.

The engine is fed pre-built per-symbol OHLCV DataFrames: the caller supplies
``data={symbol: df}`` (tz-aware ``DatetimeIndex`` + ``Open``/``High``/``Low``/
``Close``/``Volume`` columns) and is responsible for sourcing, cleaning, and
windowing that data. No database or exchange client is involved.

Submodules (internal):
    _timeframe   Date/time parsing, timeframe conversion, period alignment
    _ohlcv       DataFrame construction and OHLCV resampling
    _base        DataHandler ABC (rolling windows, HTF aggregation)
    _historic    HistoricDataHandler (backtesting)
"""

from data._base import DataHandler
from data._historic import HistoricDataHandler
from data._ohlcv import resample
from data._timeframe import (
    get_period_start,
    parse_timeframe_to_seconds,
    TIMEFRAME_FALLBACK_ORDER,
)

__all__ = [
    "DataHandler",
    "HistoricDataHandler",
    "resample",
    "get_period_start",
    "parse_timeframe_to_seconds",
    "TIMEFRAME_FALLBACK_ORDER",
]
