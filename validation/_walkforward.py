"""
Slice-once walk-forward over a parameter sweep.

Folds are analysis-time slices of each cell's full-history per-bar PnL —
valid because the engine is causal/truncation-invariant, so any window of a
full run is that config's honest out-of-sample record (no re-runs). Two
caveats are contracts, not runtime checks: stitching OOS slices across cells
is exact under dollar-vol targeting and approximate under percent-vol
targeting; strategies with offline-trained components (future ML) void the
causality precondition and need per-fold retraining instead. The private
core consumes a plain ``cell-key -> PnL Series`` mapping, so a future
``walk_forward_from_runs`` adapter can reuse it unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from volatility import bars_per_year as _bars_per_year

from ._common import is_lower_better, window_stats

logger = logging.getLogger(__name__)

_NAN = float('nan')
_SELECTION_METRICS = ('Sharpe Ratio', 'Sortino Ratio', 'Net PnL [$]',
                      'Max Drawdown [$]')
_OOS_METRICS = ('Sharpe Ratio', 'Net PnL [$]', 'Max Drawdown [$]')
_STITCHED_LABELS = ['Sharpe Ratio', 'Sortino Ratio', 'Net PnL [$]',
                    'Volatility (Ann.) [$]', 'Max Drawdown [$]',
                    'Return [%]', 'CAGR [%]', 'Max Drawdown [%]']


@dataclass(frozen=True)
class WalkForwardResult:
    """
    ``folds``: one row per fold — IS/OOS bounds, the winner's params, its IS
    selection score, and its OOS Sharpe / Net PnL / Max Drawdown.
    ``stitched_pnl``: the concatenated winners' OOS PnL (the honest OOS
    series). ``stitched_stats``: reduced stats over the stitched path
    (percent metrics NaN unless every cell shares one initial capital).
    ``oos_matrix(metric)``: every cell's OOS window stat per fold — one
    column per cell (fixed-config consistency = read one column).
    """
    folds: pd.DataFrame
    stitched_pnl: pd.Series
    stitched_stats: pd.Series
    _oos_stats: Dict[str, pd.DataFrame] = field(repr=False, default=None)

    def oos_matrix(self, metric: str = 'Sharpe Ratio') -> pd.DataFrame:
        """Folds x cells frame of ``metric`` over each OOS window."""
        if metric not in self._oos_stats:
            raise ValueError(f"unknown metric {metric!r}; available: "
                             f"{sorted(self._oos_stats)}")
        return self._oos_stats[metric].copy()


def _fold_edges(index: pd.DatetimeIndex, oos, start: pd.Timestamp,
                is_window: Optional[pd.Timedelta]) -> List[Tuple]:
    """
    Build ``(is_start, oos_start, oos_end_exclusive)`` triples. Offset-alias
    ``oos`` -> calendar-aligned period edges (each period's first bar >=
    start); Timedelta ``oos`` -> fixed windows from ``start``. Anchored
    (``is_window is None``): IS = [start, oos_start), first fold requires IS
    span >= that fold's own OOS length. Rolling: IS = [oos_start-is_window,
    oos_start), first fold requires is_start >= start. The final partial
    period is kept (real OOS data).
    """
    idx = index[index >= start]
    if len(idx) == 0:
        return []
    sentinel = idx[-1] + pd.Timedelta(1, 'ns')
    if isinstance(oos, str):
        try:
            groups = pd.Series(0, index=idx).resample(oos)
        except ValueError:
            raise ValueError(f"oos must be a pandas offset alias or "
                             f"Timedelta, got {oos!r}")
        starts = [grp.index[0] for _, grp in groups if len(grp)]
    elif isinstance(oos, pd.Timedelta):
        if oos <= pd.Timedelta(0):
            raise ValueError(f"oos Timedelta must be positive, got {oos}")
        starts, edge = [], start
        while edge <= idx[-1]:
            starts.append(edge)
            edge = edge + oos
    else:
        raise ValueError(f"oos must be a pandas offset alias or Timedelta, "
                         f"got {type(oos).__name__}")
    edges = list(starts) + [sentinel]
    folds = []
    for j in range(1, len(edges) - 1):
        oos_start, oos_end = edges[j], edges[j + 1]
        if is_window is None:
            if oos_start - start >= oos_end - oos_start:
                folds.append((start, oos_start, oos_end))
        else:
            is_start = oos_start - is_window
            if is_start >= start:
                folds.append((is_start, oos_start, oos_end))
    return folds


def _walk_forward_core(
    pnl_by_cell: Dict[Tuple, pd.Series],
    *,
    param_names: Tuple[str, ...],
    folds: List[Tuple],
    selection_metric: str,
    bars_per_year: float,
    baseline: Optional[float],
) -> WalkForwardResult:
    """The fold engine (ML seam): selection + stitching over a plain ordered
    ``cell-key -> full-history PnL`` mapping. Cells must share one index."""
    keys = list(pnl_by_cell)
    lower = is_lower_better(selection_metric)
    fold_rows = []
    oos_pieces: List[pd.Series] = []
    oos_stats: Dict[str, pd.DataFrame] = {
        m: pd.DataFrame(index=pd.Index([], name='fold'),
                        columns=pd.MultiIndex.from_tuples(
                            keys, names=list(param_names)), dtype=float)
        for m in _SELECTION_METRICS}
    fold_id = 0
    for is_start, oos_start, oos_end in folds:
        scores: Dict[Tuple, float] = {}
        for key, pnl in pnl_by_cell.items():
            is_slice = pnl[(pnl.index >= is_start) & (pnl.index < oos_start)]
            scores[key] = window_stats(
                is_slice, bars_per_year=bars_per_year)[selection_metric]
            oos_slice = pnl[(pnl.index >= oos_start) & (pnl.index < oos_end)]
            ws = window_stats(oos_slice, bars_per_year=bars_per_year)
            for m in _SELECTION_METRICS:
                oos_stats[m].loc[fold_id, key] = ws[m]
        valid = {k: v for k, v in scores.items() if not pd.isna(v)}
        if not valid:
            logger.warning("walk_forward: fold %d (%s → %s) has no scorable "
                           "cell — skipped", fold_id, is_start, oos_start)
            for m in _SELECTION_METRICS:
                oos_stats[m] = oos_stats[m].drop(index=fold_id,
                                                 errors='ignore')
            continue
        winner = (min if lower else max)(valid, key=valid.get)
        wpnl = pnl_by_cell[winner]
        oos_slice = wpnl[(wpnl.index >= oos_start) & (wpnl.index < oos_end)]
        oos_pieces.append(oos_slice)
        wstats = window_stats(oos_slice, bars_per_year=bars_per_year)
        row = {'fold': fold_id, 'is_start': is_start,
               'oos_start': oos_start,
               'oos_end': oos_slice.index[-1] if len(oos_slice) else oos_end}
        row.update(dict(zip(param_names, winner)))
        row[f'is_{selection_metric}'] = valid[winner]
        for m in _OOS_METRICS:
            row[f'oos {m}'] = wstats[m]
        fold_rows.append(row)
        fold_id += 1

    stitched = (pd.concat(oos_pieces) if oos_pieces
                else pd.Series(dtype=float))
    ws = window_stats(stitched, bars_per_year=bars_per_year,
                      baseline=baseline)
    stitched_stats = pd.Series(
        {label: ws.get(label, _NAN) for label in _STITCHED_LABELS},
        dtype=float)
    folds_df = pd.DataFrame(fold_rows)
    return WalkForwardResult(folds=folds_df, stitched_pnl=stitched,
                             stitched_stats=stitched_stats,
                             _oos_stats=oos_stats)


def walk_forward(
    sweep,
    *,
    selection_metric: str = 'Sharpe Ratio',
    oos='YE',
    is_window=None,
    start=None,
) -> WalkForwardResult:
    """
    Slice-once walk-forward over a ``SweepResult``: per fold, score every
    cell's in-sample PnL slice with ``selection_metric`` (dollar-based
    window stats only), pick the direction-aware winner, and stitch the
    winners' out-of-sample slices into the honest OOS series. ``oos`` is a
    pandas offset alias (calendar-aligned folds, ``'YE'`` default) or a
    ``pd.Timedelta``; ``is_window=None`` anchors the IS window at ``start``
    (default: the latest first-fill across cells), a Timedelta-parseable
    value makes it rolling. Percent stitched stats require a uniform
    initial capital across cells (else NaN + WARNING).
    """
    if selection_metric not in _SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of "
                         f"{_SELECTION_METRICS}, got {selection_metric!r}")
    keys = sweep.keys()
    if not keys:
        raise ValueError("sweep holds no cells")
    param_names = sweep.param_names
    pnl_by_cell = {key: sweep.pnl(**dict(zip(param_names, key)))
                   for key in keys}
    index = pnl_by_cell[keys[0]].index
    for key, pnl in pnl_by_cell.items():
        if not pnl.index.equals(index):
            raise ValueError(
                f"cell {key!r} PnL index differs — all cells must share an "
                f"identical index (same data feed)")
    if len(index) == 0:
        raise ValueError("cells have empty equity curves")
    if start is None:
        start = sweep.common_first_fill()
        if start is None:
            start = index[0]
    start = pd.Timestamp(start)
    if start.tzinfo is None:           # naive explicit start = UTC
        start = start.tz_localize('UTC')
    if is_window is not None:
        is_window = pd.Timedelta(is_window)
        if is_window <= pd.Timedelta(0):
            raise ValueError(f"is_window must be positive, got {is_window}")
    folds = _fold_edges(index, oos, start, is_window)

    capitals = {sweep.initial_capital(**dict(zip(param_names, key)))
                for key in keys}
    if len(capitals) == 1:
        baseline = capitals.pop()
    else:
        baseline = None
        logger.warning("walk_forward: cells differ in initial_capital %s — "
                       "percent stitched stats are NaN", sorted(capitals))
    bpy = _bars_per_year(sweep.timeframe, sweep.days_convention)
    return _walk_forward_core(pnl_by_cell, param_names=param_names,
                              folds=folds, selection_metric=selection_metric,
                              bars_per_year=bpy, baseline=baseline)
