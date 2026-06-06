"""volatility — Annualized price-unit volatility estimators for forecast-aware position sizing.

Submodules (internal):
    _base            VolEstimator ABC
    _ewma            EWMAVolEstimator (default — exponentially-weighted stdev
                     of price changes × sqrt(bars_per_year), span=36 by default)
    _rolling_stdev   RollingStdevVolEstimator (equal-weight sample stdev of
                     price changes × sqrt(bars_per_year))

Used by ``CarverVolTargetingRiskManager`` to derive
``annualized_target_vol / sigma`` per completed bar. Future estimators
(Yang-Zhang, GARCH) slot in by implementing the same ``VolEstimator``
interface.
"""

from volatility._base import VolEstimator, bars_per_year
from volatility._ewma import EWMAVolEstimator
from volatility._rolling_stdev import RollingStdevVolEstimator

__all__ = [
    "VolEstimator",
    "EWMAVolEstimator",
    "RollingStdevVolEstimator",
    "bars_per_year",
]
