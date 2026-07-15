"""
UTC-enforcement helpers — the single boundary law for external timestamps.

Timezone-naive input raises ``ValueError`` (a naive timestamp is ambiguous;
silently localizing it risks index misalignment — and therefore lookahead —
once intraday data enters the system). Tz-aware non-UTC input is
unambiguous and converts to UTC silently. Non-datetime containers raise
``TypeError``. The one scalar carve-out: a date-only string
(``'2024-01-01'``) has no time component to mis-read and gets UTC midnight.

Applied once per call at the caller-facing entry points only — data-handler
construction, ``data.resample``, and the ``analytics``/``validation``
research entry points. Never per bar: every ``Bar``/``BarEvent``/equity-row
timestamp downstream inherits UTC-awareness from these gates.
"""

import re
from typing import Any

import pandas as pd

_DATE_ONLY_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def ensure_utc_index(index: Any, name: str) -> pd.DatetimeIndex:
    """
    Validate ``index`` as a tz-aware ``DatetimeIndex`` and return it in UTC.

    ``name`` labels the offending input in error messages (e.g.
    ``"data['BTCUSDT']"``). Raises ``TypeError`` when ``index`` is not a
    ``DatetimeIndex``; ``ValueError`` when it is timezone-naive. Tz-aware
    non-UTC input is converted (unambiguous instant).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} index must be a pd.DatetimeIndex, "
            f"got {type(index).__name__}"
        )
    if index.tz is None:
        raise ValueError(
            f"{name} index is timezone-naive; supply a tz-aware UTC "
            f"DatetimeIndex (e.g. pd.to_datetime(..., utc=True))"
        )
    return index.tz_convert('UTC')


def ensure_utc_series(series: Any, name: str) -> pd.Series:
    """
    Validate a datetime ``Series`` (e.g. a trade log's ``timestamp``
    column) as tz-aware and return it in UTC.

    An empty Series passes through unchanged (nothing to validate — empty
    trade logs are legal). Raises ``TypeError`` on a non-Series or a
    non-datetime dtype; ``ValueError`` on a timezone-naive datetime dtype.
    Tz-aware non-UTC converts.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pd.Series, got {type(series).__name__}"
        )
    if series.empty:
        return series
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError(
            f"{name} must hold datetime values, got dtype {series.dtype}"
        )
    if series.dt.tz is None:
        raise ValueError(
            f"{name} is timezone-naive; supply tz-aware UTC timestamps "
            f"(e.g. pd.to_datetime(..., utc=True))"
        )
    return series.dt.tz_convert('UTC')


def ensure_utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    """
    Validate a scalar timestamp (``str`` / ``datetime`` / ``pd.Timestamp``)
    and return a tz-aware UTC ``pd.Timestamp``.

    Carve-out: a date-only string (``'2024-01-01'``) is unambiguous and
    gets UTC midnight. Any other naive input — a naive ``datetime``/
    ``Timestamp`` or a tz-less datetime string — raises ``ValueError``, as
    does ``NaT``. Aware input converts to UTC.
    """
    if isinstance(value, str) and _DATE_ONLY_RE.match(value):
        return pd.Timestamp(value, tz='UTC')
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"{name} is NaT — not a usable timestamp")
    if ts.tzinfo is None:
        raise ValueError(
            f"{name} ({value!r}) is timezone-naive; pass a tz-aware value "
            f"(e.g. pd.Timestamp(..., tz='UTC') or '2024-01-01T00:00:00Z') "
            f"or a date-only string ('2024-01-01' = UTC midnight)"
        )
    return ts.tz_convert('UTC')
