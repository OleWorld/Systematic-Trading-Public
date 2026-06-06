"""ContextVar plumbing for the per-bar timestamp column.

The current bar timestamp lives in a module-level ``ContextVar`` so that
async / threaded code paths see independent values without explicit
plumbing. ``_format_ts`` accepts the loose set of timestamp types the rest
of the codebase deals in (``pd.Timestamp``, ``datetime``, ISO strings,
``NaT``, ``None``) and renders them as ``YYYY-MM-DD HH:MM:SS`` or ``"-"``.
"""

from contextvars import ContextVar
from datetime import datetime
from typing import Any

import pandas as pd


_DASH = "-"
_FMT = "%Y-%m-%d %H:%M:%S"

_current_bar_ts: ContextVar[str] = ContextVar("current_bar_ts", default=_DASH)


def _format_ts(ts: Any) -> str:
    """
    Render ``ts`` as ``YYYY-MM-DD HH:MM:SS`` or ``"-"`` if absent.

    Accepts ``pd.Timestamp``, ``datetime``, ``np.datetime64``, ISO
    strings, and any value ``pd.Timestamp(ts)`` can parse. ``None`` and
    ``NaT`` both render as ``"-"``.
    """
    if ts is None:
        return _DASH
    try:
        if pd.isna(ts):
            return _DASH
    except (TypeError, ValueError):
        pass
    if isinstance(ts, str):
        return ts if len(ts) >= 10 else _DASH
    if isinstance(ts, datetime) and not isinstance(ts, pd.Timestamp):
        return ts.strftime(_FMT)
    return pd.Timestamp(ts).strftime(_FMT)


def set_current_bar_timestamp(ts: Any) -> None:
    """
    Set the bar timestamp seen by all subsequent log records on this
    context. Pass ``None`` or ``NaT`` to revert to ``"-"``.
    """
    _current_bar_ts.set(_format_ts(ts))


def clear_current_bar_timestamp() -> None:
    """Reset the bar timestamp to ``"-"`` for the current context."""
    _current_bar_ts.set(_DASH)
