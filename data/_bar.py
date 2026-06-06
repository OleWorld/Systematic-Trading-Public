"""Internal storage type for DataHandler rolling windows.

The ``Bar`` dataclass is the in-memory format for bars held inside the
deques maintained by ``DataHandler``. It is frozen/immutable; only
``DataHandler`` is expected to construct instances. External consumers
read bars via ``DataHandler.get_latest_bars()``, which returns a
``pd.DataFrame``.
"""
import datetime
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bar:
    """Immutable OHLCV bar stored in DataHandler rolling deques.

    Instances are constructed only by ``DataHandler`` (base-TF upserts
    and HTF aggregation). Frozen so a caller that obtains a reference
    cannot silently mutate shared state; slotted for lower per-bar
    memory overhead at deque sizes of several hundred to thousands.
    """
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
