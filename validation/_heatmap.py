"""
Parameter-sweep heatmaps: pivot one sweep metric over two grid axes and
render a pandas Styler (background gradient + best-cell highlight).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from ._common import is_lower_better

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

logger = logging.getLogger(__name__)

#: Best-cell highlight: bold red on yellow, so it survives the gradient.
_HIGHLIGHT_CSS = 'background-color: #ffd54f; color: red; font-weight: bold;'


@dataclass(frozen=True)
class ParamHeatmap:
    """
    One metric pivoted over two swept params: ``heatmap`` rows are the ``y``
    values, columns the ``x`` values (``where``-dropped cells are NaN);
    ``best_cell`` is ``(y_val, x_val)`` under the metric's direction
    (drawdowns minimize), ``None`` when every cell is NaN. ``fixed`` holds
    the pinned params; ``stats_start`` echoes the sweep's resolved
    warmup-trim point for the caption. 1-D sweeps have ``y=None`` and a
    single row labeled by the metric name.
    """
    heatmap: pd.DataFrame
    metric: str
    x: str
    y: Optional[str]
    fixed: Dict[str, Any] = field(default_factory=dict)
    best_cell: Optional[Tuple[Any, Any]] = None
    stats_start: Optional[pd.Timestamp] = None

    def styled(self) -> "Styler":
        """Notebook display: ``background_gradient`` (reversed colormap for
        lower-is-better metrics so good is always green), ``—`` for NaN,
        best cell highlighted, caption naming metric/best/pins/start. The
        underlying frame is unchanged and reachable via ``.data``."""
        cmap = 'RdYlGn_r' if is_lower_better(self.metric) else 'RdYlGn'

        def _highlight(frame: pd.DataFrame) -> pd.DataFrame:
            css = pd.DataFrame('', index=frame.index, columns=frame.columns)
            if self.best_cell is not None:
                css.loc[self.best_cell[0], self.best_cell[1]] = _HIGHLIGHT_CSS
            return css

        bits = [self.metric]
        if self.best_cell is not None:
            y_val, x_val = self.best_cell
            best_val = self.heatmap.loc[y_val, x_val]
            at = (f"{self.x}={x_val}" if self.y is None
                  else f"{self.y}={y_val}, {self.x}={x_val}")
            bits.append(f"best at {at} → {best_val:,.2f}")
        if self.fixed:
            bits.append("fixed: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.fixed.items())))
        bits.append(f"stats from {self.stats_start}"
                    if self.stats_start is not None else "stats: full history")
        styler = (self.heatmap.style
                  .format('{:,.2f}', na_rep='—')
                  .apply(_highlight, axis=None)
                  .set_caption(" — ".join(bits)))
        if self.best_cell is not None:      # all-NaN: gradient would warn
            styler = styler.background_gradient(cmap=cmap, axis=None)
        return styler


def build_param_heatmap(sweep, *, metric: str, x: Optional[str],
                        y: Optional[str],
                        fixed: Optional[Dict[str, Any]]) -> ParamHeatmap:
    """
    Pivot ``metric`` from ``sweep.table``. Axis defaults: with exactly two
    multi-valued params they become ``x`` (first) and ``y``; with one it
    becomes ``x``; otherwise both must be passed. Every other multi-valued
    param must be pinned via ``fixed`` (single-valued params pin
    implicitly); a clear ``ValueError`` names unpinned params.
    """
    table = sweep.table
    if metric not in table.columns or metric in sweep.param_names:
        raise ValueError(f"unknown metric {metric!r}")
    swept = [p for p in sweep.param_names if len(sweep.grid[p]) > 1]
    if x is None and y is None:
        if len(swept) == 2:
            x, y = swept[0], swept[1]
        elif len(swept) == 1:
            x = swept[0]
        else:
            raise ValueError(
                f"{len(swept)} params are swept ({swept}); pass x= and y= "
                f"explicitly and pin the rest via fixed=")
    for axis in (x, y):
        if axis is not None and axis not in sweep.param_names:
            raise ValueError(f"unknown axis param {axis!r}; grid params: "
                             f"{list(sweep.param_names)}")
    if x is None or (y is not None and x == y):
        raise ValueError(f"x and y must be distinct grid params, got "
                         f"x={x!r}, y={y!r}")
    fixed = dict(fixed or {})
    for name, value in fixed.items():
        if name not in sweep.param_names or name in (x, y):
            raise ValueError(f"fixed param {name!r} is not a pinnable "
                             f"grid param")
        if value not in sweep.grid[name]:
            raise ValueError(f"fixed value {name}={value!r} is not in the "
                             f"grid")
    unpinned = [p for p in swept
                if p not in (x, y) and p not in fixed]
    if unpinned:
        raise ValueError(f"multi-valued params not pinned: {unpinned}; "
                         f"pass them via fixed=")
    pins = dict(fixed)
    for p in sweep.param_names:
        if p not in (x, y) and p not in pins:
            pins[p] = sweep.grid[p][0]        # single-valued: implicit pin

    sub = table
    for name, value in pins.items():
        sub = sub[sub[name] == value]
    if y is not None:
        frame = (sub.pivot(index=y, columns=x, values=metric)
                 .reindex(index=sweep.grid[y], columns=sweep.grid[x]))
    else:
        frame = (sub.set_index(x)[metric]
                 .reindex(sweep.grid[x]).to_frame().T)
        frame.index = [metric]
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"metric {metric!r} is not numeric — heatmaps "
                         f"require a numeric stats column")

    best_cell = None
    values = frame.to_numpy(dtype=float)
    if np.isfinite(values).any():
        flat = (np.nanargmin(values) if is_lower_better(metric)
                else np.nanargmax(values))
        r, c = np.unravel_index(int(flat), values.shape)
        best_cell = (frame.index[r], frame.columns[c])
    else:
        logger.debug("build_param_heatmap: every cell is NaN for %r — "
                     "no best cell", metric)
    return ParamHeatmap(heatmap=frame, metric=metric, x=x, y=y, fixed=fixed,
                        best_cell=best_cell,
                        stats_start=sweep.stats_start_resolved)
