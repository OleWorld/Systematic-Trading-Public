import datetime
from typing import Tuple, Union


# Preferred order for finding source data to resample from (most granular first)
TIMEFRAME_FALLBACK_ORDER = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


def _parse_date(date_input: Union[str, datetime.datetime]) -> datetime.datetime:
    """Parse string or datetime to timezone-aware UTC datetime."""
    if isinstance(date_input, datetime.datetime):
        return date_input
    if 'T' in date_input:
        return datetime.datetime.fromisoformat(date_input.replace('Z', '+00:00'))
    dt = datetime.datetime.strptime(date_input, '%Y-%m-%d')
    return dt.replace(tzinfo=datetime.timezone.utc)


def _ms_to_utc(ms: float) -> datetime.datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.datetime.fromtimestamp(ms / 1000.0, datetime.timezone.utc)


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    """Ensure datetime has UTC timezone info, converting if necessary."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


# Unit suffixes are case-sensitive: lowercase 'm' is minutes while
# uppercase 'M' is months; 'Y' (yearly) has no lowercase counterpart.
# The second-values for 'M' and 'Y' are approximations (30d, 365d) —
# months and years have no fixed duration, so these are used ONLY for
# ordering/comparison (fallback lookup, HTF-vs-base checks). Period
# boundaries for monthly/yearly are computed in ``get_period_start``
# from the timeframe string, not from seconds.
_UNITS_SECONDS = {
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
    'M': 30 * 86400,
    'Y': 365 * 86400,
}
_UNITS_PANDAS = {
    'm': 'min',
    'h': 'h',
    'd': 'D',
    'w': 'W',
    'M': 'MS',   # month-start
    'Y': 'YS',   # year-start
}


def _split_timeframe(timeframe: str) -> Tuple[int, str]:
    """Validate ``timeframe`` and return (multiplier:int, unit:str).

    Unit matching is case-sensitive: ``'m'`` is minutes, ``'M'`` is months;
    ``'Y'`` is years (no lowercase form). Raises ``ValueError`` with a
    descriptive message for empty input, missing multiplier, unknown unit,
    non-numeric multiplier, or a non-positive multiplier.
    """
    if not isinstance(timeframe, str) or len(timeframe) < 2:
        raise ValueError(
            f"Invalid timeframe '{timeframe}': expected format like '1m', '4h', '1d'."
        )
    unit = timeframe[-1]
    if unit not in _UNITS_SECONDS:
        raise ValueError(
            f"Unknown timeframe unit '{unit}' in '{timeframe}'. "
            f"Must be one of {list(_UNITS_SECONDS.keys())}."
        )
    raw_multiplier = timeframe[:-1]
    try:
        multiplier = int(raw_multiplier)
    except ValueError:
        raise ValueError(
            f"Invalid timeframe multiplier in '{timeframe}': "
            f"'{raw_multiplier}' is not an integer."
        )
    if multiplier <= 0:
        raise ValueError(
            f"Timeframe multiplier must be a positive integer, got '{timeframe}'."
        )
    return multiplier, unit


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string (e.g., '1m', '1h', '1d') to seconds.

    For monthly (``'M'``) and yearly (``'Y'``) the returned value is an
    approximation (30 days and 365 days respectively) — sufficient for
    ordering-style comparisons but NOT for bucket arithmetic. Use
    ``get_period_start`` for calendar-correct period alignment.
    """
    multiplier, unit = _split_timeframe(timeframe)
    return multiplier * _UNITS_SECONDS[unit]


def _timeframe_to_pandas_offset(timeframe: str) -> str:
    """Convert timeframe string to pandas offset alias.

    Examples: ``'1m'`` -> ``'1min'``, ``'4h'`` -> ``'4h'``,
    ``'1M'`` -> ``'1MS'`` (month-start), ``'1Y'`` -> ``'1YS'`` (year-start).
    """
    multiplier, unit = _split_timeframe(timeframe)
    return f"{multiplier}{_UNITS_PANDAS[unit]}"


def get_period_start(ts: datetime.datetime, timeframe: str) -> datetime.datetime:
    """Compute the period-start timestamp for a bar at ``ts``.

    Supported timeframes:
      * Sub-daily (e.g. ``'1m'``, ``'4h'``): aligned to midnight UTC each day
        (matching ``_resample_ohlcv`` behaviour).
      * Daily (``'1d'``): truncated to midnight UTC.
      * Weekly (``'1w'``): aligned to Monday 00:00 UTC of the ISO week
        containing ``ts`` (open-time convention).
      * Monthly (``'nM'``): aligned to the 1st day (00:00 UTC) of the
        ``n``-month block starting from January within the same calendar
        year. ``'1M'`` gives the first of the month; ``'3M'`` gives
        quarterly boundaries (Jan/Apr/Jul/Oct); ``'6M'`` gives semi-annual
        boundaries (Jan/Jul).
      * Yearly (``'nY'``): aligned to Jan 1 (00:00 UTC) of the ``n``-year
        block.

    Any other multi-day timeframe (e.g. 2-day, 3-day) raises ``ValueError``
    rather than silently misaligning.
    """
    multiplier, unit = _split_timeframe(timeframe)

    if unit == 'Y':
        year_start = (ts.year // multiplier) * multiplier
        return datetime.datetime(year_start, 1, 1, tzinfo=ts.tzinfo)

    if unit == 'M':
        month_start = ((ts.month - 1) // multiplier) * multiplier + 1
        return datetime.datetime(ts.year, month_start, 1, tzinfo=ts.tzinfo)

    tf_seconds = multiplier * _UNITS_SECONDS[unit]

    if tf_seconds < 86400:
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        secs_since_midnight = int((ts - midnight).total_seconds())
        period_offset = (secs_since_midnight // tf_seconds) * tf_seconds
        return midnight + datetime.timedelta(seconds=period_offset)
    if tf_seconds == 86400:
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    if tf_seconds == 604800:
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - datetime.timedelta(days=ts.weekday())
    raise ValueError(
        f"Unsupported timeframe '{timeframe}'; "
        "only sub-daily, daily, weekly, monthly, and yearly are supported."
    )
