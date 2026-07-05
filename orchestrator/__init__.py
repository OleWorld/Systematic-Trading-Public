"""orchestrator — multi-strategy forecast aggregation.

A "PM over traders" layer: the ``Orchestrator`` wraps N strategies,
drives each one on every bar, and exposes a single combined forecast per
symbol (Carver weighted-sum × FDM, capped, with per-symbol weight
renormalization over the contributing strategies). It is an optional,
additive layer that occupies the strategy slot of the engine and the
risk manager — both treat it interchangeably with a single ``Strategy``
because it satisfies the same forecast-source read surface
(``symbol_list`` / ``get_forecast`` / ``is_warmed_up``).

Single concrete class for now; callers do ``from orchestrator import
Orchestrator``.
"""

from orchestrator._orchestrator import Orchestrator

__all__ = ['Orchestrator']
