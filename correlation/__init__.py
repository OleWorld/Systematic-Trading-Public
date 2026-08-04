"""correlation — walk-forward correlation estimation on a cadence.

Submodules (internal):
    _manager   CorrelationManager (cadence, refresh pipeline,
               constant-price exclusion lifecycle)
"""

from correlation._manager import CorrelationManager

__all__ = ["CorrelationManager"]
