"""LivePortfolio — placeholder for live account state (not implemented)."""

import logging

from event import BarEvent, FillEvent
from portfolio._base import Portfolio

logger = logging.getLogger(__name__)


class LivePortfolio(Portfolio):
    def update_bar(self, event: BarEvent) -> None:
        pass

    def update_fill(self, event: FillEvent) -> None:
        pass
