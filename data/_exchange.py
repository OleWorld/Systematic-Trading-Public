import logging
import time
from typing import Any, Dict, List, Optional

from data._timeframe import _ms_to_utc

logger = logging.getLogger(__name__)


def _build_exchange_config(market_type: str) -> Dict[str, Any]:
    """Build CCXT exchange config dict from market type."""
    if market_type not in ('spot', 'swap', 'future'):
        raise ValueError(f"Unknown market_type: '{market_type}'. Must be 'spot', 'swap', or 'future'.")
    config: Dict[str, Any] = {'enableRateLimit': True}
    if market_type in ('swap', 'future'):
        config['options'] = {'defaultType': market_type}
    return config


def _fetch_all_candles(exchange: Any, symbol: str, timeframe: str,
                       since_ms: Optional[int] = None, limit: int = 1000) -> List[List[Any]]:
    """Paginated fetch of OHLCV candles from exchange.

    If since_ms is None, fetches only the last ``limit`` candles.
    """
    if since_ms is None:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv if ohlcv else []

    all_candles: List[List[Any]] = []
    current_since = since_ms
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
        if not ohlcv:
            break
        all_candles.extend(ohlcv)
        current_since = ohlcv[-1][0] + 1  # Next ms after last candle

        logger.debug("  Fetched %d candles, total: %d, last: %s",
                     len(ohlcv), len(all_candles), _ms_to_utc(ohlcv[-1][0]))

        if len(ohlcv) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000.0)

    return all_candles
