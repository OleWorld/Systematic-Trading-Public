"""LiveExecution — placeholder for live order routing (not implemented)."""

import logging

from event import BarEvent, OrderEvent
from execution._base import ExecutionHandler

logger = logging.getLogger(__name__)


class LiveExecution(ExecutionHandler):
    def execute_order(self, event: OrderEvent) -> None:
        pass

    def update_bar(self, event: BarEvent) -> None:
        pass
