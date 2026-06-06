"""config — Infrastructure configuration dataclasses for run modes.

Submodules (internal):
    _backtest  BacktestConfig (data, portfolio, risk, execution parameters)
"""

from config._backtest import BacktestConfig

__all__ = ["BacktestConfig"]
