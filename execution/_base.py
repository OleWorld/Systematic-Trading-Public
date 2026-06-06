"""ExecutionHandler abstract base class."""

import logging
from abc import ABC, abstractmethod

from event import BarEvent, OrderEvent

logger = logging.getLogger(__name__)


class ExecutionHandler(ABC):
    """Abstract base class for Execution/Order Routing."""

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_bar(self, event: BarEvent) -> None:
        raise NotImplementedError
