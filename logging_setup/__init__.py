"""logging_setup — Bar-timestamp column for logging across the event loop.

Every log record emitted while the event loop is processing a bar is stamped
with that bar's timestamp via a ``contextvars.ContextVar``-backed
``LogRecord`` factory. The format string includes both wall-clock time and
bar time so operators can scan a fixed-position ``bar_ts`` column when
visually debugging:

    2026-04-26 10:30:01 | 2026-04-26 10:00:00 | INFO | strategy.foo | message

Pre-loop infra logs (data-handler setup, module wiring, etc.) emit
with ``bar_ts == "-"`` because no bar has entered the loop yet.

Why a record factory and not a ``logging.Filter``?  A filter on the root
logger only runs for records emitted directly on the root; child-logger
records bypass it (only filters on the root's *handlers* would run for child
records). The factory installs once at the ``logging`` module level and
applies to every record from every logger uniformly.

Submodules (internal):
    _context  ContextVar plumbing (set/clear bar timestamp, format helper)
    _factory  Idempotent LogRecord-factory installer
    _config   Caller-facing ``configure_logging`` entrypoint + defaults
"""

from logging_setup._context import (
    set_current_bar_timestamp,
    clear_current_bar_timestamp,
)
from logging_setup._factory import inject_bar_timestamp_factory
from logging_setup._config import configure_logging

__all__ = [
    "configure_logging",
    "set_current_bar_timestamp",
    "clear_current_bar_timestamp",
    "inject_bar_timestamp_factory",
]
