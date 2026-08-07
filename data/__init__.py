"""data — Market data handlers for backtesting.

The engine is fed pre-built per-symbol OHLCV DataFrames: the caller supplies
``data={symbol: df}`` (tz-aware ``DatetimeIndex`` + ``Open``/``High``/``Low``/
``Close``/``Volume`` columns) and is responsible for sourcing, cleaning, and
windowing that data. No database or exchange client is involved. Optional
alternative data (funding rates, open interest, ...) rides along as named
per-symbol "alt feeds" via ``alt_data={feed: {symbol: df}}``.

Submodules (internal):
    _timeframe   Date/time parsing, timeframe conversion, period alignment
    _ohlcv       DataFrame construction and OHLCV resampling
    _bar         Bar storage dataclass for the rolling deques
    _alt         AltRecord storage dataclass for alt-feed rolling deques
    _base        DataHandler ABC (rolling windows, HTF aggregation, alt feeds)
    _historic    HistoricDataHandler (backtesting)
    _live        LiveDataHandler (stub)
    _tz          UTC-enforcement helpers (naive raises, aware converts)
"""

from data._alt import AltRecord
from data._base import DataHandler
from data._historic import HistoricDataHandler
from data._live import LiveDataHandler
from data._ohlcv import resample
from data._timeframe import (
    get_period_start,
    parse_timeframe_to_seconds,
)
from data._tz import ensure_utc_index, ensure_utc_series, ensure_utc_timestamp

__all__ = [
    "AltRecord",
    "DataHandler",
    "HistoricDataHandler",
    "LiveDataHandler",
    "resample",
    "get_period_start",
    "parse_timeframe_to_seconds",
    "ensure_utc_index",
    "ensure_utc_series",
    "ensure_utc_timestamp",
]
