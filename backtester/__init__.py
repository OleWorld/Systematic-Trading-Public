"""backtester — Event-loop engine that drives the bar/signal/order/fill pipeline.

Submodules (internal):
    _engine  Backtester class (event loop)
"""

from backtester._engine import Backtester

__all__ = ["Backtester"]
