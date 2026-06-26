"""execution — Order execution handlers for backtesting and live trading.

Submodules (internal):
    _base        ExecutionHandler ABC
    _slippage    SlippageModel (price-space fill slippage)
    _commission  CommissionModel (per-contract / rate, point-value aware)
    _backtest    BacktestExecution
    _live        LiveExecution (stub)
"""

from execution._base import ExecutionHandler
from execution._slippage import SlippageModel
from execution._commission import CommissionModel
from execution._backtest import BacktestExecution
from execution._live import LiveExecution

__all__ = [
    "ExecutionHandler",
    "BacktestExecution",
    "SlippageModel",
    "CommissionModel",
    "LiveExecution",
]
