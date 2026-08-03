"""
Pure serialization helpers for the run archive.

Everything here is a side-effect-free function shared by ``runlog._save``
and ``runlog._load``: Parquet-safe DataFrame sanitization, the type-tagged
JSON codec for the ``backtest_stats`` Series, config / instrument / model
flattening for the manifest, and small utilities (label slugging, JSON
cleaning). No project imports besides pandas/numpy — the archive layer is
fully duck-typed.
"""

import dataclasses
import datetime
import enum
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Version of the on-disk archive layout. Bump when the folder layout or a
#: table schema changes incompatibly; ``load_run`` records it in the manifest
#: so future readers can dispatch. v2 (2026-07): manifest gains ``data_range``
#: (derived actual backtest range); config no longer carries start/end dates.
#: v3 (2026-08): universe.parquet transition log; RM state no longer
#: snapshots live_symbols (universe manager owns liveness).
SCHEMA_VERSION = 3


def replace_with_retry(src: Any, dst: Any, *, attempts: int = 6,
                       base_delay: float = 0.05) -> None:
    """``os.replace(src, dst)`` with bounded retry on ``PermissionError``.

    On Windows a just-written file or directory can be transiently locked
    by an antivirus/indexer scan, so an immediate rename fails with
    ``PermissionError`` (WinError 5) even though a retry moments later
    succeeds. Retries with exponential backoff (``base_delay * 2**k``,
    ~1.55 s total at the defaults) and re-raises the final
    ``PermissionError`` once ``attempts`` renames have failed — fail-loud
    is preserved; only the transient lock is absorbed. Any other
    exception propagates immediately.
    """
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            logger.warning(
                "replace_with_retry: rename %s -> %s hit a transient lock "
                "(attempt %d/%d); retrying in %.2fs",
                src, dst, attempt, attempts, delay,
            )
            time.sleep(delay)
            delay *= 2


_SLUG_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')


def slugify(label: str) -> str:
    """
    Sanitize ``label`` into a filesystem-safe slug.

    Any run of characters outside ``[A-Za-z0-9._-]`` is replaced with a
    single ``-`` (symbols like ``BTC/USDT:USDT`` would otherwise break
    Windows paths). Raises ``ValueError`` if the result is empty.
    """
    if not isinstance(label, str):
        raise TypeError(f"label must be a str, got {type(label).__name__}")
    slug = _SLUG_PATTERN.sub('-', label).strip('-')
    if not slug:
        raise ValueError(f"label {label!r} contains no filesystem-safe characters")
    return slug


# ── Parquet-safe DataFrame sanitization ──────────────────────────────────

def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a Parquet-safe copy of ``df``.

    * A named or datetime index is reset into a regular column (tables are
      written column-only; the loader knows each table's index contract).
    * Object-dtype columns are coerced: ``Enum`` values -> ``.value``; then
      all-numeric -> ``float64`` (``None`` -> NaN), all-bool -> nullable
      ``boolean``, all-str -> left as-is; anything else falls back to
      ``repr`` with a WARNING (the raw-archive path never raises on odd
      values).
    * Non-object dtypes (numerics, tz-aware datetimes, categoricals, bools)
      pass through untouched — pyarrow handles them natively.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    out = df.copy()
    if out.index.name is not None or isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = _sanitize_object_column(out[col], col)
    return out


def _sanitize_object_column(series: pd.Series, name: Any) -> pd.Series:
    """Coerce one object-dtype column to a Parquet-friendly dtype."""
    series = series.map(
        lambda v: v.value if isinstance(v, enum.Enum) else v
    )
    non_null = [v for v in series if not pd.isna(v)]
    if not non_null:
        # All-missing column: dtype is unknowable — store as float64 NaN.
        return series.astype('float64')
    if all(isinstance(v, (bool, np.bool_)) for v in non_null):
        return series.astype('boolean')
    if all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, (bool, np.bool_))
        for v in non_null
    ):
        return series.astype('float64')
    if all(isinstance(v, str) for v in non_null):
        return series
    if all(isinstance(v, (pd.Timestamp, datetime.datetime)) for v in non_null):
        try:
            return pd.to_datetime(series)
        except (ValueError, TypeError):
            pass  # mixed tz-aware/naive — fall through to repr
    logger.warning(
        "Column %r holds mixed/unsupported object values; storing repr() strings",
        name,
    )
    return series.map(lambda v: None if pd.isna(v) else repr(v))


# ── backtest_stats Series <-> type-tagged JSON ───────────────────────────

def encode_stats(stats: pd.Series) -> Dict[str, Any]:
    """
    Encode the mixed-dtype ``backtest_stats`` Series as a type-tagged,
    order-preserving JSON payload: ``{"labels": [...], "entries":
    {label: {"t": tag, "v": value}}}``. Tags: ``timestamp`` (ISO-8601 with
    tz), ``timedelta`` (ISO duration), ``nat``, ``float`` (``None`` = NaN),
    ``bool``, ``str``, ``null``.
    """
    if not isinstance(stats, pd.Series):
        raise TypeError(f"expected a Series, got {type(stats).__name__}")
    entries: Dict[str, Dict[str, Any]] = {}
    for label, value in stats.items():
        entries[str(label)] = _encode_stat_value(value)
    return {'labels': [str(l) for l in stats.index], 'entries': entries}


def _encode_stat_value(value: Any) -> Dict[str, Any]:
    """Tag one stats value for JSON; see ``encode_stats`` for the tag set."""
    if value is None:
        return {'t': 'null', 'v': None}
    if value is pd.NaT:
        return {'t': 'nat', 'v': None}
    if isinstance(value, pd.Timestamp):
        return {'t': 'timestamp', 'v': value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {'t': 'timedelta', 'v': value.isoformat()}
    if isinstance(value, (bool, np.bool_)):
        return {'t': 'bool', 'v': bool(value)}
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return {'t': 'float', 'v': None if math.isnan(v) or math.isinf(v) else v}
    if isinstance(value, str):
        return {'t': 'str', 'v': value}
    logger.warning("Stats value %r has unsupported type; storing repr()", value)
    return {'t': 'str', 'v': repr(value)}


def decode_stats(payload: Dict[str, Any]) -> pd.Series:
    """Rebuild the ``backtest_stats`` Series from ``encode_stats`` output."""
    if not isinstance(payload, dict) or 'labels' not in payload or 'entries' not in payload:
        raise ValueError("stats payload missing 'labels'/'entries' keys")
    values = [
        _decode_stat_value(payload['entries'][label]) for label in payload['labels']
    ]
    return pd.Series(values, index=payload['labels'], dtype=object)


def _decode_stat_value(entry: Dict[str, Any]) -> Any:
    """Inverse of ``_encode_stat_value``; raises ``ValueError`` on unknown tags."""
    tag, value = entry['t'], entry['v']
    if tag == 'null':
        return None
    elif tag == 'nat':
        return pd.NaT
    elif tag == 'timestamp':
        return pd.Timestamp(value)
    elif tag == 'timedelta':
        return pd.Timedelta(value)
    elif tag == 'bool':
        return bool(value)
    elif tag == 'float':
        return float('nan') if value is None else float(value)
    elif tag == 'str':
        return value
    else:
        raise ValueError(f"Unexpected stats value tag: {tag!r}")


# ── Manifest JSON cleaning ───────────────────────────────────────────────

def json_safe(obj: Any) -> Any:
    """
    Recursively convert ``obj`` into plain JSON-serializable values:
    numpy scalars -> Python, NaN/inf -> ``None``, Timestamp/Timedelta ->
    ISO strings, Enum -> ``.value``, dict keys -> ``str``. Unknown objects
    degrade to ``repr`` (the manifest is documentation-grade — it must
    always serialize).
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, enum.Enum):
        return json_safe(obj.value)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return repr(obj)


# ── Config / instrument / model flattening ───────────────────────────────

def serialize_model(model: Any) -> Dict[str, Any]:
    """
    Flatten a slippage/commission/margin model for the manifest:
    dataclass -> ``{'class': name, 'params': asdict}``, anything else ->
    ``{'class': name, 'repr': repr}``. Documentation-grade only — not
    intended for object reconstruction.
    """
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        return {
            'class': type(model).__name__,
            'params': json_safe(dataclasses.asdict(model)),
        }
    return {'class': type(model).__name__, 'repr': repr(model)}


def serialize_config(config: Any) -> Optional[Dict[str, Any]]:
    """
    Flatten a ``BacktestConfig`` (any dataclass) into a JSON-safe dict.
    ``None`` passes through; a non-dataclass raises ``TypeError``.
    """
    if config is None:
        return None
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return json_safe(dataclasses.asdict(config))
    raise TypeError(
        f"config must be a dataclass instance, got {type(config).__name__}"
    )


def serialize_instruments(instruments: Optional[Dict[str, Any]]) -> Optional[List[Dict]]:
    """
    Flatten the ``{symbol: InstrumentConfig}`` registry, deduped by object
    identity (``uniform_registry`` shares ONE config across all symbols ->
    one entry, not one per symbol). Each entry: ``{'symbols': [...],
    'instrument': {point_value, fractional, slippage, commission, margin}}``
    with the three models flattened via ``serialize_model``.
    """
    if instruments is None:
        return None
    if not isinstance(instruments, dict):
        raise TypeError(
            f"instruments must be a dict, got {type(instruments).__name__}"
        )
    groups: Dict[int, Dict[str, Any]] = {}
    for symbol, cfg in instruments.items():
        group = groups.get(id(cfg))
        if group is None:
            group = {
                'symbols': [],
                'instrument': {
                    'point_value': json_safe(getattr(cfg, 'point_value', None)),
                    'fractional': json_safe(getattr(cfg, 'fractional', None)),
                    'slippage': serialize_model(getattr(cfg, 'slippage', None)),
                    'commission': serialize_model(getattr(cfg, 'commission', None)),
                    'margin': serialize_model(getattr(cfg, 'margin', None)),
                },
            }
            groups[id(cfg)] = group
        group['symbols'].append(symbol)
    return list(groups.values())
