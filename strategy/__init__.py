"""strategy — Trading strategy framework.

Submodules:
    _base   Strategy ABC + dependency Protocols (template module)
    ewmac   Carver's EWMAC trend-following rule — three EWMA-crossover
            variations combined into a single weighted forecast in
            [-100, +100] with a dynamic forecast scalar driving
            avg |f| toward 50

Traders add new concrete strategies as sibling modules (e.g.
`strategy/momentum.py`) and re-export them from this package's `__init__`
if they should be part of the public import surface.
"""

from strategy._base import Strategy
from strategy.ewmac import EWMACStrategy

__all__ = [
    "Strategy",
    "EWMACStrategy",
]
