from typing import Any, Callable, Dict, List, Union

import pandas as pd

from data._tz import ensure_utc_index
from data._timeframe import (
    _ms_to_utc,
    get_period_start,
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

    Bucket alignment: all buckets are defined by ``get_period_start`` —
    sub-daily aligned to midnight UTC each day, weekly to Monday 00:00 UTC,
    monthly/yearly to calendar blocks — so historic resampling and live HTF
    accumulation produce identical bucket boundaries for every supported
    timeframe. Timeframes ``get_period_start`` rejects (e.g. '2d', '2w')
    raise ``ValueError`` here too. Only non-empty buckets are returned.

    The input index must be a tz-aware ``DatetimeIndex`` (naive raises ``ValueError``); non-UTC input is converted to UTC before bucketing.

    ``agg`` maps column name to a pandas aggregation (string op or callable).
    The caller decides the agg dict and is responsible for dropping empty
    buckets afterwards (e.g. ``df.dropna(subset=[<sentinel_col>])``).
    """
    if df.empty:
        return df.iloc[0:0].copy()

    df = df.set_axis(ensure_utc_index(df.index, 'df'))

    tf_seconds = parse_timeframe_to_seconds(timeframe)

    if tf_seconds >= 86400:
        # Daily and above (daily, weekly, monthly, yearly): bucket every
        # row by ``get_period_start`` — the same calendar-block authority
        # the live HTF accumulation uses — so historic resampling and live
        # aggregation produce identical bucket boundaries by construction.
        # Unsupported multi-unit aliases ('2d', '2w', ...) raise inside
        # get_period_start, exactly like the live path.
        bucket_idx = pd.Index(
            [get_period_start(ts, timeframe) for ts in df.index]
        )
        resampled = df.groupby(bucket_idx).agg(agg)
        resampled.index.name = df.index.name
        return resampled
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
