"""Portfolio abstract base class and structural-typing Protocols for dependencies."""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Protocol

from event import BarEvent, FillEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Structural types for dependencies
# ──────────────────────────────────────────────

class _DataHandlerLike(Protocol):
    """Minimal interface `BacktestPortfolio` needs from the data handler.

    Consumes ``get_latest_bars`` — the last *n* ``Bar`` objects (a plain
    list, no DataFrame) — for the cold-start price fallback in ``get_price``.
    """
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Any]: ...


class _EventsQueueLike(Protocol):
    """Minimal interface `BacktestPortfolio` needs from the events queue."""
    def put(self, item: Any) -> None: ...


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class Portfolio(ABC):
    """Abstract base class for Portfolio management."""

    @abstractmethod
    def update_bar(self, event: BarEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_fill(self, event: FillEvent) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        """Post-run reconciliation hook, called once by ``Backtester.run()``
        after the event loop drains.

        Fills for the final bar's orders book *after* that bar's equity
        row is recorded, so implementations that keep an equity curve
        should re-snapshot their last row here — otherwise the curve
        permanently misses the run's last realized PnL, slippage, and
        commission. Default is a no-op (concrete, not abstract, so
        stubs and live placeholders need not override it).
        """
        return None
