"""Post-run backtest summary statistics (dollar-first, futures-friendly).

``backtest_stats`` condenses a ``BacktestPortfolio`` equity curve and
trade log into a backtesting.py-style ``pd.Series`` of summary metrics.
One-shot — call it after ``Backtester.run()`` completes, not per bar.

Unit policy
-----------
Dollar and percentage variants are paired wherever both are meaningful
(drawdowns, volatility, total return); percentage-only where dollars
make no sense (CAGR, win rate); dollar-only where percent makes no
sense for futures (trade PnL, commission, equity levels). Every
percentage metric is computed on the **account equity curve** (dollar-
denominated and positive), never on instrument prices — futures and
spread prices can be negative or zero, where price-relative returns are
meaningless, but account-level percentages stay well-defined.

Net-vs-gross caveat
-------------------
Trade-log ``realized_pnl`` is **gross of commission** (the portfolio
books ``cash += realized - commission``, tracking commission
separately), so the trade-level stats (Win Rate, Profit Factor,
Expectancy, Best/Worst/Avg Trade, SQN) are gross-of-commission, while
the equity-based stats (Net PnL, Sharpe, drawdowns) are net. The two
families can therefore look mutually inconsistent on high-commission
runs; ``Total Commission [$]`` quantifies the gap.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from volatility import bars_per_year

logger = logging.getLogger(__name__)

_NAN = float('nan')

# Required equity-curve columns (subset of the BacktestPortfolio row schema
# actually consumed here).
_REQUIRED_EQUITY_COLS = ('account_balance', 'total_commission')


def backtest_stats(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    initial_capital: float,
    timeframe: str,
    days_convention: str,
) -> pd.Series:
    """Summarize a backtest into an ordered ``pd.Series`` of statistics.

    Parameters
    ----------
    equity_curve
        Output of ``portfolio.get_equity_curve()`` — timestamp-indexed,
        one row per ``BarEvent`` (N rows per timestamp with N symbols).
        Collapsed internally to one row per timestamp (last row wins —
        it reflects every symbol's price plug-in at that timestamp).
        Must contain ``account_balance`` and ``total_commission`` when
        non-empty. Empty frames are a clean edge case (NaN/NaT output).
    trade_log
        Output of ``portfolio.get_trade_log()`` — one row per
        ``FillEvent``. A "closing trade" is any fill with
        ``realized_pnl != 0`` (per-closing-fill convention; no
        round-trip reconstruction). Empty frames yield zero counts and
        NaN trade stats.
    initial_capital
        Starting capital — the t0 baseline for PnL, returns, and the
        running drawdown peak. Must be ``> 0``.
    timeframe
        Bar timeframe of the equity curve (the engine's
        ``base_timeframe``, e.g. ``'1d'``). Together with
        ``days_convention``, sets the annualization factor via
        ``volatility.bars_per_year``. A mismatched value silently
        mis-annualizes — pass the timeframe the curve was actually
        recorded at.
    days_convention
        ``'calendar'`` (365 days/year) or ``'business'`` (252 trading
        days/year); also fixes the days-per-year used to rescale
        per-bar volatility to the daily lines.

    Returns
    -------
    pd.Series
        Object-dtype Series whose index is the fixed, ordered label list
        (``print(stats)`` is the intended display). Timestamps for
        ``Start``/``End``, Timedeltas for durations, floats elsewhere.
        Data edge cases (empty inputs, zero volatility, no closing
        trades, zero gross loss) yield NaN/NaT — never a raise. Note
        ``Profit Factor`` is NaN (not inf) when there are no losing
        trades, and ``Avg Trade [$]`` numerically equals
        ``Expectancy [$]`` under the per-closing-fill trade definition
        (both kept for backtesting.py familiarity).

    Raises
    ------
    TypeError
        If ``equity_curve`` or ``trade_log`` is not a DataFrame.
    ValueError
        If ``initial_capital <= 0``, ``days_convention``/``timeframe`` is
        invalid, or a non-empty ``equity_curve`` lacks required columns.

    Notes
    -----
    Ratio definitions (rf = 0 throughout):

    - Sharpe / Sortino are computed on per-bar **dollar** PnL (no
      compounding assumption — appropriate for futures accounts where
      capital is margin, not invested notional).
    - Calmar = ``CAGR [%] / Max Drawdown [%]`` (both legs percentage —
      the standard pairing).
    - Drawdown episodes are contiguous runs below the running peak
      (initial capital included in the peak). The unrecovered final
      episode is **included** in the max/avg depth and duration as a
      lower bound on its true value. An episode underwater from the
      first bar measures its duration from the first equity timestamp.
    """
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError(
            f"equity_curve must be a DataFrame, got {type(equity_curve).__name__}"
        )
    if not isinstance(trade_log, pd.DataFrame):
        raise TypeError(
            f"trade_log must be a DataFrame, got {type(trade_log).__name__}"
        )
    if initial_capital <= 0:
        raise ValueError(
            f"initial_capital must be > 0, got {initial_capital}"
        )
    bpy = bars_per_year(timeframe, days_convention)  # raises on bad inputs
    dpy = bars_per_year('1d', days_convention)
    if not equity_curve.empty:
        missing = [c for c in _REQUIRED_EQUITY_COLS
                   if c not in equity_curve.columns]
        if missing:
            raise ValueError(
                f"equity_curve is missing required columns: {missing}"
            )

    eq = _collapse_equity(equity_curve)
    empty = eq.empty
    if empty:
        logger.debug("backtest_stats: empty equity curve — NaN output")

    bal = eq['account_balance'].astype(float) if not empty else pd.Series(dtype=float)

    # --- Time bounds ---
    start = eq.index[0] if not empty else pd.NaT
    end = eq.index[-1] if not empty else pd.NaT
    duration = end - start if not empty else pd.NaT

    # --- Equity / PnL levels ---
    final = bal.iloc[-1] if not empty else _NAN
    peak = max(bal.max(), initial_capital) if not empty else _NAN
    net_pnl = final - initial_capital if not empty else _NAN
    commission = eq['total_commission'].iloc[-1] if not empty else _NAN
    return_pct = 100.0 * net_pnl / initial_capital if not empty else _NAN

    # --- Per-bar dollar PnL and equity returns (vs the t0 baseline) ---
    # ``pnl``/``ret`` have one entry per equity row: the first measures
    # the first bar against initial_capital, the rest are row-to-row.
    if not empty:
        pnl = bal.diff()
        pnl.iloc[0] = bal.iloc[0] - initial_capital
        ret = bal.pct_change()
        ret.iloc[0] = bal.iloc[0] / initial_capital - 1.0
        years = len(pnl) / bpy
    else:
        pnl = pd.Series(dtype=float)
        ret = pd.Series(dtype=float)
        years = 0.0

    cagr = _NAN
    if years > 0 and final > 0:
        cagr = 100.0 * ((final / initial_capital) ** (1.0 / years) - 1.0)

    # --- Volatility (daily + annualized, $ and %) ---
    pnl_std = pnl.std(ddof=1) if len(pnl) >= 2 else _NAN
    ret_std = ret.std(ddof=1) if len(ret) >= 2 else _NAN
    vol_ann_usd = pnl_std * math.sqrt(bpy)
    vol_daily_usd = vol_ann_usd / math.sqrt(dpy)
    vol_ann_pct = 100.0 * ret_std * math.sqrt(bpy)
    vol_daily_pct = vol_ann_pct / math.sqrt(dpy)

    # --- Ratios on dollar PnL ---
    sharpe = _NAN
    if len(pnl) >= 2 and pnl_std > 0:
        sharpe = pnl.mean() / pnl_std * math.sqrt(bpy)
    sortino = _NAN
    if len(pnl) >= 1:
        downside = math.sqrt(float((np.minimum(pnl, 0.0) ** 2).mean()))
        if downside > 0:
            sortino = pnl.mean() / downside * math.sqrt(bpy)

    # --- Drawdowns (running peak includes the initial-capital baseline) ---
    if not empty:
        running_peak = np.maximum(bal.cummax(), initial_capital)
        dd_usd = running_peak - bal
        dd_pct = 100.0 * dd_usd / running_peak
        max_dd_usd = float(dd_usd.max())
        max_dd_pct = float(dd_pct.max())
        episodes = _drawdown_episodes(dd_usd, dd_pct)
    else:
        max_dd_usd = _NAN
        max_dd_pct = _NAN
        episodes = []

    if episodes:
        avg_dd_usd = float(np.mean([e[0] for e in episodes]))
        avg_dd_pct = float(np.mean([e[1] for e in episodes]))
        durations = [e[2] for e in episodes]
        max_dd_dur = max(durations)
        avg_dd_dur = sum(durations, pd.Timedelta(0)) / len(durations)
    else:
        avg_dd_usd = _NAN
        avg_dd_pct = _NAN
        max_dd_dur = pd.NaT
        avg_dd_dur = pd.NaT

    calmar = _NAN
    if not pd.isna(cagr) and max_dd_pct and max_dd_pct > 0:
        calmar = cagr / max_dd_pct

    # --- Trade-level stats (per-closing-fill) ---
    trades = _closing_trades(trade_log)
    n_fills = len(trade_log)
    n_trades = len(trades)
    if n_trades > 0:
        win_rate = 100.0 * float((trades > 0).sum()) / n_trades
        best = float(trades.max())
        worst = float(trades.min())
        avg = float(trades.mean())
        gross_win = float(trades[trades > 0].sum())
        gross_loss = float(-trades[trades < 0].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else _NAN
        trade_std = trades.std(ddof=1) if n_trades >= 2 else _NAN
        sqn = (math.sqrt(n_trades) * avg / trade_std
               if n_trades >= 2 and trade_std > 0 else _NAN)
    else:
        win_rate = best = worst = avg = profit_factor = sqn = _NAN

    out: Dict[str, Any] = {
        'Start': start,
        'End': end,
        'Duration': duration,
        'Equity Final [$]': final,
        'Equity Peak [$]': peak,
        'Net PnL [$]': net_pnl,
        'Total Commission [$]': commission,
        'Return [%]': return_pct,
        'CAGR [%]': cagr,
        'Volatility (Daily) [$]': vol_daily_usd,
        'Volatility (Ann.) [$]': vol_ann_usd,
        'Volatility (Daily) [%]': vol_daily_pct,
        'Volatility (Ann.) [%]': vol_ann_pct,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown [$]': max_dd_usd,
        'Max Drawdown [%]': max_dd_pct,
        'Avg Drawdown [$]': avg_dd_usd,
        'Avg Drawdown [%]': avg_dd_pct,
        'Max Drawdown Duration': max_dd_dur,
        'Avg Drawdown Duration': avg_dd_dur,
        'Profit Factor': profit_factor,
        'Expectancy [$]': avg,
        'SQN': sqn,
        '# Fills': n_fills,
        '# Closing Trades': n_trades,
        'Win Rate [%]': win_rate,
        'Best Trade [$]': best,
        'Worst Trade [$]': worst,
        'Avg Trade [$]': avg,
    }
    return pd.Series(out, dtype=object)


def _collapse_equity(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: the last row per group (multi-symbol bars
    write N rows per timestamp; the last reflects every symbol's price
    plug-in). Relies on the engine's time-sorted bar stream — not
    re-sorted here, so an unsorted curve surfaces as a loud test failure
    rather than being silently masked."""
    if equity_curve.empty:
        return equity_curve
    return equity_curve.groupby(level=0, sort=False).last()


def _drawdown_episodes(
    dd_usd: pd.Series,
    dd_pct: pd.Series,
) -> List[Tuple[float, float, pd.Timedelta]]:
    """Split a drawdown series into contiguous underwater episodes.

    Returns one ``(max_depth_usd, max_depth_pct, duration)`` tuple per
    episode. Duration runs from the prior peak (the last at-peak row
    before the run; the first row if underwater from the start) to the
    recovery row (the first at-peak row after the run; the final row if
    never recovered — included as a lower bound).
    """
    is_dd = dd_usd > 0
    if not is_dd.any():
        return []
    groups = (is_dd != is_dd.shift()).cumsum()
    episodes: List[Tuple[float, float, pd.Timedelta]] = []
    positions = np.arange(len(dd_usd))
    for _, idx in dd_usd.groupby(groups).groups.items():
        if not is_dd.loc[idx[0]]:
            continue
        pos = positions[dd_usd.index.get_indexer(idx)]
        peak_ts = dd_usd.index[pos[0] - 1] if pos[0] > 0 else dd_usd.index[0]
        recovery_ts = (dd_usd.index[pos[-1] + 1]
                       if pos[-1] + 1 < len(dd_usd) else dd_usd.index[-1])
        episodes.append((
            float(dd_usd.iloc[pos].max()),
            float(dd_pct.iloc[pos].max()),
            recovery_ts - peak_ts,
        ))
    return episodes


def _closing_trades(trade_log: pd.DataFrame) -> pd.Series:
    """Per-closing-fill trade PnLs: ``realized_pnl`` of every fill where
    it is nonzero (gross of commission — see module docstring)."""
    if trade_log.empty or 'realized_pnl' not in trade_log.columns:
        return pd.Series(dtype=float)
    pnl = trade_log['realized_pnl'].astype(float)
    return pnl[pnl != 0.0]
