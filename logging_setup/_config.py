"""Caller-facing ``configure_logging`` entrypoint and default format strings."""

import logging
from typing import Optional

from logging_setup._factory import inject_bar_timestamp_factory


_DEFAULT_FMT = (
    "%(asctime)s | %(bar_ts)s | %(levelname)-5s | %(name)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """
    Configure root logging with the bar-timestamp column and install
    the bar-timestamp record factory.

    Replaces ``logging.basicConfig`` for callers of this project. Safe
    to call multiple times: the factory is only installed once. Re-runs
    update the level on the root logger and on any handlers it owns.
    """
    fmt = fmt or _DEFAULT_FMT
    datefmt = datefmt or _DEFAULT_DATEFMT

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    inject_bar_timestamp_factory()
