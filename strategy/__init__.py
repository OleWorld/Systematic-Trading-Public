"""strategy — Trading strategy framework.

Submodules:
    _base        Strategy ABC — shared, mode-agnostic machinery every
                 template builds on (forecast cache + ±100 clamp, measured
                 warmup flags, per-bar records, forecast constants, the
                 protected ``_commit_forecast_row`` seam); ``update_bar``
                 is abstract
    _timeseries  TimeSeriesStrategy — per-event template for strategies
                 that read one symbol's bars and forecast that symbol
                 alone; subclasses implement ``calculate_forecast``
    ewmac        Carver's EWMAC trend-following rule — three EWMA-crossover
                 variations combined into a single weighted forecast in
                 [-100, +100] with a dynamic forecast scalar driving
                 avg |f| toward 50
    rsimr        RSI mean-reversion rule — Wilder RSI mapped through arctanh
                 (oversold → long, overbought → short) into a weighted forecast
                 in [-100, +100], same dynamic forecast scalar driving avg |f|
                 toward 50

Timeseries strategies subclass ``TimeSeriesStrategy``; a future
cross-sectional template (cross-symbol bar alignment, batch signal
generation) will subclass ``Strategy`` directly, honoring the same
forecast-oracle contract. Traders add new concrete strategies as sibling
modules (e.g. `strategy/momentum.py`) and re-export them from this
package's `__init__` if they should be part of the public import surface.
"""

from strategy._base import Strategy
from strategy._timeseries import TimeSeriesStrategy
from strategy.ewmac import EWMACStrategy
from strategy.rsimr import RSIMRStrategy

__all__ = [
    "Strategy",
    "TimeSeriesStrategy",
    "EWMACStrategy",
    "RSIMRStrategy",
]
