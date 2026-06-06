"""data — Market data handlers for backtesting and live trading.

Naming convention: this package splits its mode-specific handlers by data
*shape* (`_historic` for bounded/replayable, `_live` for streaming) rather
than by run mode (`_backtest`/`_live`) as in `portfolio/`, `riskmanager/`,
and `execution/`. The name describes what the data is, not when it is
used — a `HistoricDataHandler` could feed a backtest, a paper replay, or
a research notebook.

Submodules (internal):
    _timeframe   Date/time parsing, timeframe conversion, period alignment
    _arctic      ArcticDB initialization helpers
    _exchange    CCXT exchange config and paginated candle fetching
    _ohlcv       DataFrame construction and OHLCV resampling
    _base        DataHandler ABC (rolling windows, HTF aggregation)
    _historic    HistoricDataHandler (backtesting)
    _live        LiveDataHandler (live WebSocket streaming)
    _db_sync     update_historical_db utility
"""

from data._base import DataHandler
from data._historic import HistoricDataHandler
from data._live import LiveDataHandler
from data._db_sync import update_historical_db
from data._ohlcv import resample
from data._timeframe import (
    get_period_start,
    parse_timeframe_to_seconds,
    TIMEFRAME_FALLBACK_ORDER,
)

__all__ = [
    "DataHandler",
    "HistoricDataHandler",
    "LiveDataHandler",
    "update_historical_db",
    "resample",
    "get_period_start",
    "parse_timeframe_to_seconds",
    "TIMEFRAME_FALLBACK_ORDER",
]
