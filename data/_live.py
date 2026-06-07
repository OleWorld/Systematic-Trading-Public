"""LiveDataHandler — placeholder for live market-data streaming (not implemented)."""

import logging

from data._base import DataHandler

logger = logging.getLogger(__name__)


class LiveDataHandler(DataHandler):
    def update_bar(self) -> None:
        pass
