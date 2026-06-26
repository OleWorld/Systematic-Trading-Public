"""config — Infrastructure configuration dataclasses for run modes.

Submodules (internal):
    _backtest    BacktestConfig (run-level: data, capital, risk, fill timing)
    _instrument  InstrumentConfig (per-symbol: point_value, fractional,
                 slippage, commission, margin) + uniform_registry factory
"""

from config._backtest import BacktestConfig
from config._instrument import InstrumentConfig, uniform_registry

__all__ = ["BacktestConfig", "InstrumentConfig", "uniform_registry"]
