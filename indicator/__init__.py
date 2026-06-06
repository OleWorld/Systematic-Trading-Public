"""indicator — Stateful, upsert-aware technical indicators.

Each indicator is a class with a single entry point ``update(ts, ...)`` —
a typed scalar form whose signature is subclass-specific. Indicators upsert
by timestamp on the forming bar so re-ticks within a single period don't
corrupt recursive state. Strategies hold per-symbol instances in
``__init__``, push one tick's worth of input per ``calculate_signal``, and
read ``.latest`` (last finalized) / ``.forming`` (current). See
``indicator/_base.py`` for the ABC and the upsert invariant.

Each class also exposes a ``from_series(...)`` static method that returns
the same series the stateful indicator would emit if fed the inputs in
order — used for warmup, research, and golden tests.
"""

from indicator._base import Indicator
from indicator.atr import ATR
from indicator.bbw import BBW
from indicator.ema import EMA
from indicator.ewmstdev import EWMStdev
from indicator.kama import KAMA
from indicator.percentrank import PercentRank
from indicator.rsi import RSI
from indicator.sma import SMA
from indicator.stdev import Stdev
from indicator.trailing_volatility_stop import TrailingVolatilityStop

__all__ = [
    'Indicator',
    'ATR',
    'BBW',
    'EMA',
    'EWMStdev',
    'KAMA',
    'PercentRank',
    'RSI',
    'SMA',
    'Stdev',
    'TrailingVolatilityStop',
]
