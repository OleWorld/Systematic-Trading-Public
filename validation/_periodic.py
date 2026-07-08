"""
Per-period (yearly/quarterly/monthly) backtest statistics — the regime view.

One row per calendar period: the global per-bar dollar PnL series is
bucketed (a period's first bar diffs against the prior period's closing
balance — no gaps, no double-counts), ``Return [%]`` / ``Max Drawdown [%]``
run against the period-START balance, and drawdown is within-period (the
running peak reseeds each period, so a multi-year underwater episode shows
per-period depth). Trades bucket by fill timestamp with the
per-closing-fill convention.
"""

import logging

import pandas as pd

from volatility import bars_per_year as _bars_per_year

from ._common import pnl_from_equity, window_stats

logger = logging.getLogger(__name__)

_NAN = float('nan')
_COLUMNS = ['Start', 'End', 'Net PnL [$]', 'Return [%]', 'Sharpe Ratio',
            'Sortino Ratio', 'Volatility (Ann.) [$]', 'Max Drawdown [$]',
            'Max Drawdown [%]', '# Closing Trades', 'Win Rate [%]']


def periodic_stats(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    initial_capital: float,
    timeframe: str,
    days_convention: str,
    freq: str = 'YE',
    start=None,
) -> pd.DataFrame:
    """
    Per-period stats table (index = the pandas resample bin label, e.g. the
    period-end timestamp for ``'YE'``). ``freq`` is any pandas offset alias
    (``'YE'`` default, ``'QE'``, ``'ME'``, ...). ``start`` (naive = UTC)
    trims to bars at/after it AND reseeds the baseline to the true balance
    entering the window — pre-start PnL never folds into the first kept
    period (deliberately different from the ``backtest_stats`` trim, whose
    baseline stays ``initial_capital``): a periodic table measures each
    period from its actual starting equity. Bad params raise; empty
    inputs yield an empty fixed-schema frame; periods with < 2 bars carry
    NaN ratio stats (data law).
    """
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError(
            f"trade_log must be a DataFrame, got {type(trade_log).__name__}")
    bpy = _bars_per_year(timeframe, days_convention)     # raises on bad input
    try:
        pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        raise ValueError(f"freq must be a pandas offset alias, got {freq!r}")
    pnl = pnl_from_equity(equity_curve, initial_capital=initial_capital)
    baseline = float(initial_capital)
    if start is not None and not pnl.empty:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:        # naive start = UTC
            start_ts = start_ts.tz_localize('UTC')
        # reseed to the true balance entering the window (see docstring)
        baseline += float(pnl[pnl.index < start_ts].sum())
        pnl = pnl[pnl.index >= start_ts]
    if pnl.empty:
        logger.debug("periodic_stats: no bars in window — empty table")
        return pd.DataFrame(columns=_COLUMNS)

    closing = pd.DataFrame(columns=['timestamp', 'realized_pnl'])
    if not trade_log.empty and 'realized_pnl' in trade_log.columns:
        closing = trade_log[trade_log['realized_pnl'].astype(float) != 0.0]
        if start is not None and 'timestamp' in closing.columns:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:    # naive start = UTC
                start_ts = start_ts.tz_localize('UTC')
            closing = closing[closing['timestamp'] >= start_ts]
    trades_by_period = {}
    if not closing.empty and 'timestamp' in closing.columns:
        for label, grp in closing.groupby(
                pd.Grouper(key='timestamp', freq=freq)):
            if len(grp):
                trades_by_period[label] = grp['realized_pnl'].astype(float)

    rows = {}
    for label, grp in pnl.resample(freq):
        if len(grp) == 0:
            continue
        ws = window_stats(grp, bars_per_year=bpy, baseline=baseline)
        tr = trades_by_period.get(label, pd.Series(dtype=float))
        rows[label] = {
            'Start': grp.index[0], 'End': grp.index[-1],
            'Net PnL [$]': ws['Net PnL [$]'],
            'Return [%]': ws['Return [%]'],     # baseline always given
            'Sharpe Ratio': ws['Sharpe Ratio'],
            'Sortino Ratio': ws['Sortino Ratio'],
            'Volatility (Ann.) [$]': ws['Volatility (Ann.) [$]'],
            'Max Drawdown [$]': ws['Max Drawdown [$]'],
            'Max Drawdown [%]': ws['Max Drawdown [%]'],
            '# Closing Trades': int(len(tr)),
            'Win Rate [%]': (100.0 * float((tr > 0).sum()) / len(tr)
                             if len(tr) else _NAN),
        }
        baseline += float(grp.sum())     # next period starts where this ended
    out = pd.DataFrame.from_dict(rows, orient='index', columns=_COLUMNS)
    out.index.name = 'period'
    return out
