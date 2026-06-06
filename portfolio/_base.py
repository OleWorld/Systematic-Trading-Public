"""Portfolio abstract base class and structural-typing Protocols for dependencies."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

import pandas as pd

from event import BarEvent, FillEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Structural types for dependencies
# ──────────────────────────────────────────────

class _DataHandlerLike(Protocol):
    """Minimal interface `BacktestPortfolio` needs from the data handler."""
    def get_latest_bars(self, symbol: str, n: int) -> pd.DataFrame: ...


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
