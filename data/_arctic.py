import logging
from typing import Any, Tuple

import arcticdb

logger = logging.getLogger(__name__)


def _make_symbol_key(symbol: str, timeframe: str) -> str:
    """Generate ArcticDB symbol key, e.g. 'BTC_USDT_1m'."""
    return f"{symbol.replace('/', '_')}_{timeframe}"


def _init_arctic_lib(db_path: str) -> Tuple[Any, Any]:
    """Initialize ArcticDB store and market_data library.

    WINDOWS NOTE: Explicit map_size=1TB is required for LMDB on Windows.
    """
    store = arcticdb.Arctic(f"lmdb://{db_path}?map_size=1TB")
    lib_name = "market_data"
    if lib_name not in store.list_libraries():
        store.create_library(lib_name)
    return store, store[lib_name]
