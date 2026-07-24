"""Post-run per-instrument turnover statistics (Carver convention).

``turnover_stats`` condenses a ``BacktestPortfolio`` equity curve and
trade log into a per-instrument table of position/turnover metrics.
One-shot — call it after ``Backtester.run()`` completes, not per bar.

Turnover convention
-------------------
A continuously-resized vol-target book rarely goes flat, so literal
flat-to-flat episodes degenerate (one "trade" can span the whole
backtest). Carver's convention (Systematic Trading ch. 12) is used
instead: one round trip = trading twice the average absolute position,
so ``round trips / year = (Σ|fill qty| / 2) / avg |position| / years``
and ``avg holding [days] = days-per-year / round trips per year`` —
well-defined even when the position never touches zero.

Active window
-------------
Each symbol's statistics run from its **first fill** to the end of the
backtest (zeros after listing count; pre-listing/warmup bars do not).
A whole-backtest window would dilute the average position and inflate
the annualization span for late-listing symbols under the staggered-
listing liveness gate. Symbols with no fills are excluded entirely.

The final ``'Average'`` row carries only the cross-instrument mean of
``Avg Holding [days]``: contract counts are not comparable across
instruments, and a cross-instrument mean of turnover ratios was
deliberately omitted as well.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analytics._stats import _collapse_equity
from data import ensure_utc_index, ensure_utc_series
from volatility import bars_per_year

logger = logging.getLogger(__name__)

__all__ = ['turnover_stats']

_NAN = float('nan')

_COLUMNS = (
    'Avg Position [contracts]',
    'Round Trips / Year',
    'Avg Holding [days]',
)

# Required columns (subsets of the BacktestPortfolio schemas actually
# consumed here).
_REQUIRED_EQUITY_COLS = ('positions',)
_REQUIRED_TRADE_COLS = ('timestamp', 'symbol', 'quantity')


def turnover_stats(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    timeframe: str,
    days_convention: str,
) -> pd.DataFrame:
    """Per-instrument turnover table (plus a cross-instrument average row).

    Parameters
    ----------
    equity_curve
        Output of ``portfolio.get_equity_curve()`` — timestamp-indexed,
        one row per ``BarEvent``; collapsed internally to one row per
        timestamp (last wins). Must carry the per-symbol ``positions``
        dict column when non-empty.
    trade_log
        Output of ``portfolio.get_trade_log()`` — one row per
        ``FillEvent``, with ``timestamp``, ``symbol`` and ``quantity``
        (positive magnitude). A timezone-naive equity-curve index (or
        trade-log timestamp column) raises ``ValueError``; tz-aware
        non-UTC input is converted to UTC.
    timeframe
        Bar timeframe of the equity curve (the engine's
        ``base_timeframe``, e.g. ``'1d'``); with ``days_convention``,
        sets the bars-per-year annualization via
        ``volatility.bars_per_year``.
    days_convention
        ``'calendar'`` (365 days/year) or ``'business'`` (252 trading
        days/year); also fixes the days-per-year behind
        ``Avg Holding [days]``.

    Returns
    -------
    pd.DataFrame
        One row per traded symbol (alphabetical) with the fixed columns
        ``'Avg Position [contracts]'``, ``'Round Trips / Year'``,
        ``'Avg Holding [days]'``, plus a final ``'Average'`` row where
        only the holding-period mean is populated (see the module
        docstring). Data edge cases (either input empty, zero average
        position) yield an empty fixed-schema frame or NaN cells —
        never a raise.

    Raises
    ------
    TypeError
        If ``equity_curve`` or ``trade_log`` is not a DataFrame.
    ValueError
        If ``timeframe``/``days_convention`` is invalid, a non-empty
        equity-curve index or trade-log timestamp column is
        timezone-naive, or a non-empty input lacks its required
        columns.
    """
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError(
            f"equity_curve must be a DataFrame, got {type(equity_curve).__name__}"
        )
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError(
            f"trade_log must be a DataFrame, got {type(trade_log).__name__}"
        )
    if not equity_curve.empty:
        equity_curve = equity_curve.set_axis(
            ensure_utc_index(equity_curve.index, 'equity_curve'))
    if not trade_log.empty and 'timestamp' in trade_log.columns:
        trade_log = trade_log.assign(
            timestamp=ensure_utc_series(trade_log['timestamp'],
                                        "trade_log['timestamp']"))
    bpy = bars_per_year(timeframe, days_convention)  # raises on bad inputs
    dpy = bars_per_year('1d', days_convention)
    if not equity_curve.empty:
        missing = [c for c in _REQUIRED_EQUITY_COLS
                   if c not in equity_curve.columns]
        if missing:
            raise ValueError(
                f"equity_curve is missing required columns: {missing}"
            )
    if not trade_log.empty:
        missing = [c for c in _REQUIRED_TRADE_COLS
                   if c not in trade_log.columns]
        if missing:
            raise ValueError(
                f"trade_log is missing required columns: {missing}"
            )

    eq = _collapse_equity(equity_curve)
    if eq.empty or trade_log.empty:
        logger.debug("turnover_stats: empty input — empty table")
        return pd.DataFrame(columns=list(_COLUMNS))

    # Per-symbol |position| series (ffill/fillna guards synthetic inputs
    # whose dicts don't cover every symbol on every row — the real
    # portfolio always writes full dicts).
    pos = (
        pd.DataFrame(list(eq['positions']), index=eq.index)
        .astype(float)
        .ffill()
        .fillna(0.0)
        .abs()
    )

    grouped = trade_log.groupby('symbol')
    first_fill = grouped['timestamp'].min()
    gross_qty = grouped['quantity'].apply(lambda q: q.abs().sum())

    rows = {}
    for sym in sorted(str(s) for s in first_fill.index):
        if sym not in pos.columns:
            rows[sym] = (_NAN, _NAN, _NAN)
            continue
        window = pos.loc[pos.index >= first_fill[sym], sym]
        avg_pos = float(window.mean()) if len(window) else _NAN
        years = len(window) / bpy
        if avg_pos > 0 and years > 0:  # NaN avg_pos fails the > 0 check
            rt_per_year = float(gross_qty[sym]) / 2.0 / avg_pos / years
        else:
            rt_per_year = _NAN
        avg_hold = dpy / rt_per_year if rt_per_year > 0 else _NAN
        rows[sym] = (avg_pos, rt_per_year, avg_hold)

    table = pd.DataFrame.from_dict(rows, orient='index',
                                   columns=list(_COLUMNS))
    holds = table['Avg Holding [days]']
    mean_hold = float(holds.mean()) if holds.notna().any() else _NAN
    table.loc['Average'] = (_NAN, _NAN, mean_hold)
    return table
