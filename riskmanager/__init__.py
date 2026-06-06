"""riskmanager — Forecast-aware position sizing for backtesting and live trading.

Submodules (internal):
    _base                  RiskManager ABC + structural-typing Protocols
    _simple_riskmanager    SimpleRiskManager (simple forecast follower —
                           direction from forecast sign, magnitude from
                           configured sizing mode)
    _carver_vol_targeting  CarverVolTargetingRiskManager (vol-targeting
                           Carver framework — target notional scales with
                           annualized_target_vol / sigma and forecast
                           magnitude; execution-mode agnostic — same
                           class for backtest and live)
"""

from riskmanager._base import RiskManager
from riskmanager._simple_riskmanager import SimpleRiskManager
from riskmanager._carver_vol_targeting import CarverVolTargetingRiskManager

__all__ = [
    "RiskManager",
    "SimpleRiskManager",
    "CarverVolTargetingRiskManager",
]
