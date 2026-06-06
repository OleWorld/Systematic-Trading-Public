from typing import Any, Callable, Dict, List, Union

import pandas as pd

from data._timeframe import (
    _ms_to_utc,
    parse_timeframe_to_seconds,
    _timeframe_to_pandas_offset,
)


_AggSpec = Dict[str, Union[str, Callable[[pd.Series], Any]]]


def _candles_to_dataframe(candles: List[List[Any]]) -> pd.DataFrame:
    """Convert raw OHLCV candle lists [[ms, O, H, L, C, V], ...] to a pandas DataFrame."""
    indices = [_ms_to_utc(c[0]) for c in candles]
    data = [[c[1], c[2], c[3], c[4], c[5]] for c in candles]
    return pd.DataFrame(data, index=indices, columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def resample(df: pd.DataFrame, timeframe: str, agg: _AggSpec) -> pd.DataFrame:
    """Resample a time-indexed DataFrame to ``timeframe`` using ``agg``.

    Bucket alignment: sub-daily bars are aligned to midnight UTC each day,
    weekly to Monday 00:00 UTC of the ISO week (open-time convention),
    matching ``get_period_start`` so historic resampling and live HTF
    accumulation produce identical bucket boundaries.

    ``agg`` maps column name to a pandas aggregation (string op or callable).
    The caller decides the agg dict and is responsible for dropping empty
    buckets afterwards (e.g. ``df.dropna(subset=[<sentinel_col>])``).
    """
    if df.empty:
        return df.iloc[0:0].copy()

    tf_seconds = parse_timeframe_to_seconds(timeframe)

    if tf_seconds == 604800:
        # Weekly: bucket each row by Monday-of-week, label at Monday 00:00.
        monday_idx = (df.index - pd.to_timedelta(df.index.weekday, unit='D')).normalize()
        resampled = df.groupby(monday_idx).agg(agg)
        resampled.index.name = df.index.name
        return resampled
    if tf_seconds >= 86400:
        # Daily and above (monthly, yearly).
        offset = _timeframe_to_pandas_offset(timeframe)
        return df.resample(offset).agg(agg)
    if 86400 % tf_seconds == 0:
        # Sub-daily that divides evenly into 24h (e.g., 4h, 15m).
        offset = _timeframe_to_pandas_offset(timeframe)
        return df.resample(offset, origin='start_day').agg(agg)
    # Sub-daily that doesn't divide evenly (e.g. 33m) — reset at midnight each day.
    offset = _timeframe_to_pandas_offset(timeframe)
    parts = []
    for date, group in df.groupby(df.index.date):
        day_origin = pd.Timestamp(date, tz=group.index.tz)
        parts.append(group.resample(offset, origin=day_origin).agg(agg))
    return pd.concat(parts)


_OHLCV_AGG: _AggSpec = {
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
}


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a higher timeframe.

    Thin caller around ``resample`` with the standard OHLCV agg dict.
    Empty buckets (no underlying base bars) are dropped via the ``Open``
    column.
    """
    resampled = resample(df, timeframe, _OHLCV_AGG)
    if resampled.empty:
        return resampled
    resampled.dropna(subset=['Open'], inplace=True)
    return resampled
