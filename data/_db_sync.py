import datetime
import logging
from typing import Any, List, Union

import ccxt

from data._timeframe import (
    _parse_date, _ensure_utc, parse_timeframe_to_seconds,
)
from data._arctic import _make_symbol_key, _init_arctic_lib
from data._exchange import _build_exchange_config, _fetch_all_candles
from data._ohlcv import _candles_to_dataframe

logger = logging.getLogger(__name__)


def update_historical_db(symbol_list: List[str], start_date: Union[str, datetime.datetime],
                         exchange_id: str, timeframe: str, market_type: str,
                         db_path: str = "arctic_data",
                         overwrite: bool = False) -> None:
    """Sync ArcticDB with Exchange data from start_date to now.

    If overwrite=True, fetches all data from start_date and replaces existing data.
    If overwrite=False (default), only fetches data after the last stored timestamp.
    """
    dt = _parse_date(start_date)
    start_str = start_date if isinstance(start_date, str) else start_date.isoformat()

    logger.info("Syncing DB for %s from %s (market: %s, overwrite=%s)...", symbol_list, start_str, market_type, overwrite)

    try:
        _, lib = _init_arctic_lib(db_path)
    except Exception as e:
        logger.error("DB Error: %s", e)
        return

    exchange: Any = getattr(ccxt, exchange_id)(_build_exchange_config(market_type))

    for symbol in symbol_list:
        symbol_key = _make_symbol_key(symbol, timeframe)

        # Determine fetch start point
        if not overwrite and lib.has_symbol(symbol_key):
            existing_df = lib.read(symbol_key).data
            if not existing_df.empty:
                last_ts = _ensure_utc(existing_df.index[-1])
                tf_seconds = parse_timeframe_to_seconds(timeframe)
                fetch_from = last_ts + datetime.timedelta(seconds=tf_seconds)
                since_ms = int(fetch_from.timestamp() * 1000)
                logger.info("Fetching %s from %s (incremental)...", symbol, fetch_from)
            else:
                since_ms = int(dt.timestamp() * 1000)
                logger.info("Fetching %s from %s (empty DB)...", symbol, start_str)
        else:
            since_ms = int(dt.timestamp() * 1000)
            logger.info("Fetching %s from %s...", symbol, start_str)

        try:
            all_candles = _fetch_all_candles(exchange, symbol, timeframe, since_ms=since_ms)
        except Exception as e:
            logger.error("Error fetching: %s", e)
            continue

        if all_candles:
            df = _candles_to_dataframe(all_candles)
            df = df[~df.index.duplicated(keep='last')]

            if overwrite and lib.has_symbol(symbol_key):
                lib.delete(symbol_key)
                logger.info("  Deleted existing data for %s", symbol_key)
                lib.write(symbol_key, df)
            elif lib.has_symbol(symbol_key):
                lib.append(symbol_key, df)
            else:
                lib.write(symbol_key, df)

            logger.info("  Saved %d rows to %s", len(df), symbol_key)
        else:
            logger.info("  No new data for %s", symbol)
