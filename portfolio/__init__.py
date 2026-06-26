"""portfolio — Account state and order submission for backtesting and live trading.

Submodules (internal):
    _base      Portfolio ABC + structural-typing Protocols
    _backtest  BacktestPortfolio (cross-margin futures accounting + simulated margin checks)
    _margin    MarginModel ABC + PortfolioMarginModel (default, point-value aware)
    _live      LivePortfolio (stub)
"""

from portfolio._base import Portfolio
from portfolio._margin import MarginModel, PortfolioMarginModel
from portfolio._backtest import BacktestPortfolio
from portfolio._live import LivePortfolio

__all__ = [
    "Portfolio",
    "BacktestPortfolio",
    "MarginModel",
    "PortfolioMarginModel",
    "LivePortfolio",
]
