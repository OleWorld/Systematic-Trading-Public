"""
Parameter-sweep harness: run a caller-supplied backtest factory over a grid.

``param_sweep`` calls ``run_fn(**params)`` once per grid cell (Cartesian
product, optional ``where`` filter) and collects each returned
portfolio-like's collapsed equity curve, trade log, and initial capital into
a ``SweepResult`` — per-cell ``backtest_stats`` table, PnL accessors, best
cell, and heatmaps. Stats honor the ``stats_start`` warmup-trim policy
(default ``'common_first_fill'``: the latest first-fill across cells, so
slow-warmup cells aren't penalized by dilution asymmetry). The optional
per-cell disk cache with resume lands via ``cache_dir``.
"""

import itertools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from analytics import backtest_stats

from ._common import collapse_equity, first_fill, is_lower_better, \
    pnl_from_equity

logger = logging.getLogger(__name__)

_GRID_VALUE_TYPES = (str, bool, int, float)
_STATS_START_POLICIES = ('common_first_fill',)
_EQUITY_COLS = ['account_balance', 'total_commission']


class _Cell:
    """One completed grid cell: params + collapsed equity + trades + capital."""

    def __init__(self, params: Dict[str, Any], equity: pd.DataFrame,
                 trades: pd.DataFrame, initial_capital: float,
                 runtime_s: float):
        self.params = params
        self.equity = equity
        self.trades = trades
        self.initial_capital = initial_capital
        self.runtime_s = runtime_s


def _validate_grid(grid) -> None:
    """Grid law: non-empty dict of non-empty, duplicate-free scalar lists."""
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty dict of {param: [values]}")
    for name, values in grid.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"grid param names must be non-empty str, "
                             f"got {name!r}")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"grid[{name!r}] must be a non-empty list")
        for v in values:
            if not isinstance(v, _GRID_VALUE_TYPES):
                raise TypeError(
                    f"grid[{name!r}] values must be str/int/float/bool, "
                    f"got {type(v).__name__}: {v!r}")
        if len(set(values)) != len(values):
            raise ValueError(f"grid[{name!r}] contains duplicate values")


def _validate_stats_start(stats_start) -> None:
    """``None`` | ``'common_first_fill'`` | anything ``pd.Timestamp`` takes."""
    if stats_start is None or stats_start in _STATS_START_POLICIES:
        return
    try:
        pd.Timestamp(stats_start)
    except (ValueError, TypeError):
        raise ValueError(
            f"stats_start must be None, 'common_first_fill', or a "
            f"timestamp; got {stats_start!r}")


def _extract_cell(params: Dict[str, Any], portfolio_like,
                  runtime_s: float) -> _Cell:
    """Pull the factory contract's surface off a portfolio-like object."""
    for attr in ('get_equity_curve', 'get_trade_log', 'initial_capital'):
        if not hasattr(portfolio_like, attr):
            raise TypeError(
                f"run_fn must return a portfolio-like exposing "
                f"get_equity_curve/get_trade_log/initial_capital; the object "
                f"for {params!r} lacks {attr!r}")
    equity = collapse_equity(portfolio_like.get_equity_curve())
    if equity.empty:
        equity = pd.DataFrame(columns=_EQUITY_COLS)
    else:
        missing = [c for c in _EQUITY_COLS if c not in equity.columns]
        if missing:
            raise ValueError(f"equity curve for {params!r} is missing "
                             f"columns: {missing}")
        equity = equity[_EQUITY_COLS].copy()
    initial_capital = float(portfolio_like.initial_capital)
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got "
                         f"{initial_capital} for {params!r}")
    return _Cell(params=params, equity=equity,
                 trades=portfolio_like.get_trade_log().copy(),
                 initial_capital=initial_capital, runtime_s=runtime_s)


class SweepResult:
    """
    Result of a parameter sweep: one ``_Cell`` per completed grid cell, in
    deterministic grid order. ``table`` (lazy, cached) carries the param
    columns first, then every ``analytics.backtest_stats`` label, computed
    with ``start=stats_start_resolved``. Cell accessors take the params as
    keyword arguments (``sweep.pnl(fast=16, slow=64)``).
    """

    def __init__(self, cells: Dict[Tuple, _Cell], *, grid: Dict[str, list],
                 timeframe: str, days_convention: str, stats_start,
                 cache_dir=None):
        self._cells = cells
        self.grid = {k: list(v) for k, v in grid.items()}
        self.timeframe = timeframe
        self.days_convention = days_convention
        self.stats_start = stats_start
        self.cache_dir = cache_dir
        self._table: Optional[pd.DataFrame] = None
        self._resolved_start = 'UNRESOLVED'

    @property
    def param_names(self) -> Tuple[str, ...]:
        """Grid parameter names, in grid order (= cell-key tuple order)."""
        return tuple(self.grid)

    def keys(self) -> List[Tuple]:
        """Cell key tuples (param values in ``param_names`` order)."""
        return list(self._cells)

    def _key(self, params: Dict[str, Any]) -> Tuple:
        """Validate a params kwargs dict and return its cell key; raises
        ``ValueError`` on wrong names, ``KeyError`` on an unknown cell."""
        if set(params) != set(self.param_names):
            raise ValueError(f"expected exactly the params "
                             f"{sorted(self.param_names)}, got "
                             f"{sorted(params)}")
        key = tuple(params[n] for n in self.param_names)
        if key not in self._cells:
            raise KeyError(f"no cell for params {params!r} (dropped by "
                           f"'where', not in the grid, or not yet cached)")
        return key

    def equity(self, **params) -> pd.DataFrame:
        """The cell's collapsed equity curve (copy)."""
        return self._cells[self._key(params)].equity.copy()

    def trades(self, **params) -> pd.DataFrame:
        """The cell's trade log (copy)."""
        return self._cells[self._key(params)].trades.copy()

    def initial_capital(self, **params) -> float:
        """The cell's initial capital."""
        return self._cells[self._key(params)].initial_capital

    def pnl(self, **params) -> pd.Series:
        """The cell's full-history per-bar dollar PnL (untrimmed — consumers
        apply their own windows)."""
        cell = self._cells[self._key(params)]
        return pnl_from_equity(cell.equity,
                               initial_capital=cell.initial_capital)

    def common_first_fill(self) -> Optional[pd.Timestamp]:
        """Latest first-fill timestamp across cells that have fills; ``None``
        when no cell has any fill."""
        firsts = [first_fill(c.trades) for c in self._cells.values()]
        firsts = [f for f in firsts if f is not None]
        return max(firsts) if firsts else None

    @property
    def stats_start_resolved(self) -> Optional[pd.Timestamp]:
        """The ``stats_start`` policy resolved to a concrete timestamp."""
        if self._resolved_start == 'UNRESOLVED':
            if self.stats_start is None:
                self._resolved_start = None
            elif self.stats_start == 'common_first_fill':
                self._resolved_start = self.common_first_fill()
            else:
                ts = pd.Timestamp(self.stats_start)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                self._resolved_start = ts
        return self._resolved_start

    @property
    def table(self) -> pd.DataFrame:
        """One row per cell: param columns, then all ``backtest_stats``
        labels (lazy; recomputed never — cached on first access)."""
        if self._table is None:
            start = self.stats_start_resolved
            rows = []
            for key, cell in self._cells.items():
                stats = backtest_stats(
                    cell.equity, cell.trades,
                    initial_capital=cell.initial_capital,
                    timeframe=self.timeframe,
                    days_convention=self.days_convention, start=start)
                rows.append({**cell.params, **stats.to_dict()})
            self._table = pd.DataFrame(rows)
        return self._table

    def best(self, metric: str = 'Sharpe Ratio') -> Dict[str, Any]:
        """Params dict of the best cell under ``metric`` (drawdown labels
        minimize, everything else maximizes); all-NaN columns raise."""
        if metric not in self.table.columns or metric in self.param_names:
            raise ValueError(f"unknown metric {metric!r}")
        col = pd.to_numeric(self.table[metric], errors='coerce')
        if col.isna().all():
            raise ValueError(f"metric {metric!r} is NaN for every cell")
        pos = int(col.idxmin() if is_lower_better(metric) else col.idxmax())
        return dict(zip(self.param_names, self.keys()[pos]))

    def heatmap(self, metric: str = 'Sharpe Ratio', x: Optional[str] = None,
                y: Optional[str] = None,
                fixed: Optional[Dict[str, Any]] = None):
        """Pivot ``metric`` over two swept params -> ``ParamHeatmap``."""
        from ._heatmap import build_param_heatmap
        return build_param_heatmap(self, metric=metric, x=x, y=y, fixed=fixed)


def param_sweep(
    run_fn: Callable[..., Any],
    grid: Dict[str, list],
    *,
    timeframe: str,
    days_convention: str,
    where: Optional[Callable[..., bool]] = None,
    cache_dir=None,
    stats_start='common_first_fill',
) -> SweepResult:
    """
    Run ``run_fn(**params)`` for every grid cell and collect a ``SweepResult``.

    ``grid`` maps param names to duplicate-free scalar lists; cells are the
    Cartesian product in dict-insertion x value order, filtered by ``where``.
    A factory exception propagates immediately (fail-loud; with a cache the
    restart resumes at the failed cell). ``timeframe``/``days_convention``
    set stats annualization; ``stats_start`` is the warmup-trim policy
    (``'common_first_fill'`` default / ``None`` / explicit timestamp).
    """
    if not callable(run_fn):
        raise TypeError(f"run_fn must be callable, got "
                        f"{type(run_fn).__name__}")
    _validate_grid(grid)
    if where is not None and not callable(where):
        raise TypeError(f"where must be callable or None, got "
                        f"{type(where).__name__}")
    _validate_stats_start(stats_start)
    from volatility import bars_per_year
    bars_per_year(timeframe, days_convention)   # validate early, raises

    names = tuple(grid)
    combos = [dict(zip(names, combo))
              for combo in itertools.product(*grid.values())]
    if where is not None:
        combos = [p for p in combos if where(**p)]

    cache = _CellCache(cache_dir, grid=grid, timeframe=timeframe,
                       days_convention=days_convention,
                       stats_start=stats_start) if cache_dir else None

    cells: Dict[Tuple, _Cell] = {}
    for i, params in enumerate(combos, start=1):
        key = tuple(params[n] for n in names)
        if cache is not None and cache.has(params):
            cell = cache.load(params)
            source = 'cached'
        else:
            t0 = time.perf_counter()
            cell = _extract_cell(params, run_fn(**params),
                                 runtime_s=time.perf_counter() - t0)
            if cache is not None:
                cache.store(cell)
            source = 'ran'
        cells[key] = cell
        logger.info("cell %d/%d %s: %s (%.1fs)", i, len(combos), params,
                    source, cell.runtime_s)
    return SweepResult(cells, grid=grid, timeframe=timeframe,
                       days_convention=days_convention,
                       stats_start=stats_start, cache_dir=cache_dir)


class _CellCache:
    """Placeholder — implemented in the cache task; constructing with a
    non-None cache_dir before then is an explicit error."""

    def __init__(self, root, **kwargs):
        raise NotImplementedError("cache_dir support lands in the cache task")
