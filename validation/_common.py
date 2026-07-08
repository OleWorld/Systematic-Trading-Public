"""
Shared internals for the ``validation`` package.

Single home for the ``analytics.backtest_stats`` conventions the whole suite
relies on: equity-curve collapse (one row per timestamp, last wins), per-bar
dollar-PnL derivation (first bar measured against ``initial_capital``),
reduced window statistics over a PnL slice, the metric direction map, and the
first-fill probe used by warmup-trim policies.
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NAN = float('nan')

#: Stats labels where smaller is better (drawdowns); everything else maximizes.
_LOWER_IS_BETTER_PREFIXES = ('Max Drawdown', 'Avg Drawdown')


def is_lower_better(metric: str) -> bool:
    """True when a smaller value of ``metric`` is better (drawdown labels)."""
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a str, got {type(metric).__name__}")
    return metric.startswith(_LOWER_IS_BETTER_PREFIXES)


def collapse_equity(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp (last row wins) — multi-symbol bars write N rows
    per timestamp; the last reflects every symbol's price plug-in. Mirrors
    ``analytics.backtest_stats``. Raises ``TypeError`` on non-DataFrame input."""
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError(
            f"equity_curve must be a DataFrame, got {type(equity_curve).__name__}"
        )
    if equity_curve.empty:
        return equity_curve
    return equity_curve.groupby(level=0, sort=False).last()


def pnl_from_equity(
    equity_curve: pd.DataFrame,
    *,
    initial_capital: float,
    start=None,
) -> pd.Series:
    """
    Per-bar dollar PnL, derived exactly as ``analytics.backtest_stats``:
    collapse -> optional ``start`` trim -> ``balance.diff()`` with the first
    kept bar measured against ``initial_capital`` (pre-``start`` PnL folds
    into the first kept bar — the documented ``backtest_stats`` semantics).
    Empty input yields an empty float Series; a non-empty curve without
    ``account_balance`` raises ``ValueError``; ``initial_capital <= 0`` raises.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    eq = collapse_equity(equity_curve)
    if start is not None and not eq.empty:
        start_ts = pd.Timestamp(start, tz='UTC')
        eq = eq.loc[eq.index >= start_ts]
    if eq.empty:
        return pd.Series(dtype=float)
    if 'account_balance' not in eq.columns:
        raise ValueError("equity_curve is missing required column "
                         "'account_balance'")
    bal = eq['account_balance'].astype(float)
    pnl = bal.diff()
    pnl.iloc[0] = bal.iloc[0] - initial_capital
    return pnl


def window_stats(
    pnl: pd.Series,
    *,
    bars_per_year: float,
    baseline: Optional[float] = None,
) -> Dict[str, float]:
    """
    Reduced ``backtest_stats`` over one PnL window; labels match
    ``analytics._stats`` byte-for-byte. Dollar metrics always; ``Return [%]``
    / ``CAGR [%]`` / ``Max Drawdown [%]`` only when ``baseline`` (the equity
    level at the window start) is given and positive. Drawdowns run on the
    within-window cumulative path with the running peak seeded at the window
    start. Degenerate data (empty, zero variance) yields NaN — never raises.
    """
    out: Dict[str, float] = {
        'Sharpe Ratio': _NAN, 'Sortino Ratio': _NAN, 'Net PnL [$]': _NAN,
        'Volatility (Ann.) [$]': _NAN, 'Max Drawdown [$]': _NAN,
    }
    if baseline is not None:
        out.update({'Return [%]': _NAN, 'CAGR [%]': _NAN,
                    'Max Drawdown [%]': _NAN})
    vals = pnl.astype(float).to_numpy()
    n = len(vals)
    if n == 0:
        return out
    out['Net PnL [$]'] = float(vals.sum())
    if n >= 2:
        std = float(np.std(vals, ddof=1))
        out['Volatility (Ann.) [$]'] = std * math.sqrt(bars_per_year)
        if std > 0:
            out['Sharpe Ratio'] = float(
                vals.mean() / std * math.sqrt(bars_per_year))
    downside = math.sqrt(float((np.minimum(vals, 0.0) ** 2).mean()))
    if downside > 0:
        out['Sortino Ratio'] = float(
            vals.mean() / downside * math.sqrt(bars_per_year))
    cum = np.cumsum(vals)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    out['Max Drawdown [$]'] = float((peak - cum).max())
    if baseline is not None and baseline > 0:
        balance = baseline + cum
        bpeak = np.maximum.accumulate(
            np.concatenate(([baseline], balance)))[1:]
        out['Max Drawdown [%]'] = float(
            (100.0 * (bpeak - balance) / bpeak).max())
        out['Return [%]'] = 100.0 * float(cum[-1]) / baseline
        years = n / bars_per_year
        final = float(balance[-1])
        if years > 0 and final > 0:
            out['CAGR [%]'] = 100.0 * ((final / baseline) ** (1.0 / years)
                                       - 1.0)
    return out


def first_fill(trade_log: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Earliest fill timestamp in a trade log; ``None`` for an empty log or
    one without a ``timestamp`` column (never raises on data shape)."""
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError(
            f"trade_log must be a DataFrame, got {type(trade_log).__name__}")
    if trade_log.empty or 'timestamp' not in trade_log.columns:
        return None
    return pd.Timestamp(trade_log['timestamp'].min())
