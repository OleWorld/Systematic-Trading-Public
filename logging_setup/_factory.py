"""Idempotent installer for the bar-timestamp ``LogRecord`` factory."""

import logging

from logging_setup._context import _current_bar_ts


_factory_installed: bool = False


def inject_bar_timestamp_factory() -> None:
    """
    Install a ``LogRecord`` factory that stamps every record with the
    current bar timestamp. Idempotent — calling twice does not stack.
    """
    global _factory_installed
    if _factory_installed:
        return
    base_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = base_factory(*args, **kwargs)
        record.bar_ts = _current_bar_ts.get()
        return record

    logging.setLogRecordFactory(_factory)
    _factory_installed = True
