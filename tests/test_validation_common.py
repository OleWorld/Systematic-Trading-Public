"""Tests for validation._common — shared PnL/stats internals."""

import numpy as np
import pandas as pd
import pytest

from analytics import backtest_stats
from validation._common import (collapse_equity, first_fill, is_lower_better,
                                pnl_from_equity, window_stats)


def _equity(balances, start='2024-01-01', commission=0.0):
    """Equity-curve double: one row per daily timestamp (UTC)."""
    idx = pd.date_range(start, periods=len(balances), freq='D', tz='UTC')
    return pd.DataFrame({'account_balance': balances,
                         'total_commission': commission}, index=idx)


def test_collapse_keeps_last_row_per_timestamp():
    idx = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'], utc=True)
    eq = pd.DataFrame({'account_balance': [1.0, 2.0, 3.0]}, index=idx)
    out = collapse_equity(eq)
    assert list(out['account_balance']) == [2.0, 3.0]


def test_pnl_first_bar_measured_against_initial_capital():
    eq = _equity([1010.0, 1005.0, 1030.0])
    pnl = pnl_from_equity(eq, initial_capital=1000.0)
    assert list(pnl) == [10.0, -5.0, 25.0]


def test_pnl_start_trim_folds_prior_pnl_into_first_kept_bar():
    eq = _equity([1010.0, 1005.0, 1030.0])
    pnl = pnl_from_equity(eq, initial_capital=1000.0, start='2024-01-02')
    assert list(pnl) == [5.0, 25.0]   # 1005 - 1000, then 1030 - 1005


def test_pnl_param_validation():
    with pytest.raises(TypeError):
        pnl_from_equity([1, 2], initial_capital=1000.0)
    with pytest.raises(ValueError):
        pnl_from_equity(_equity([1.0]), initial_capital=0.0)
    with pytest.raises(ValueError):
        pnl_from_equity(pd.DataFrame({'x': [1.0]},
                                     index=pd.to_datetime(['2024-01-01'], utc=True)),
                        initial_capital=1000.0)


def test_pnl_empty_curve_gives_empty_series():
    assert pnl_from_equity(pd.DataFrame(), initial_capital=1000.0).empty


def test_window_stats_matches_backtest_stats_conventions():
    rng = np.random.default_rng(7)
    balances = 1_000_000 + np.cumsum(rng.normal(50, 1_000, size=300))
    eq = _equity(balances.tolist())
    trades = pd.DataFrame()   # no trades — equity-based stats only
    bs = backtest_stats(eq, trades, initial_capital=1_000_000,
                        timeframe='1d', days_convention='calendar')
    pnl = pnl_from_equity(eq, initial_capital=1_000_000)
    ws = window_stats(pnl, bars_per_year=365.0, baseline=1_000_000)
    for label in ('Sharpe Ratio', 'Sortino Ratio', 'Net PnL [$]',
                  'Volatility (Ann.) [$]', 'Max Drawdown [$]',
                  'Max Drawdown [%]', 'Return [%]', 'CAGR [%]'):
        assert ws[label] == pytest.approx(float(bs[label]), rel=1e-12), label


def test_window_stats_without_baseline_has_no_percent_keys():
    ws = window_stats(pd.Series([1.0, -1.0, 2.0]), bars_per_year=365.0)
    assert 'Return [%]' not in ws and 'CAGR [%]' not in ws
    assert 'Max Drawdown [%]' not in ws


def test_window_stats_degenerate_data_is_nan_never_raises():
    empty = window_stats(pd.Series(dtype=float), bars_per_year=365.0)
    assert all(pd.isna(v) for v in empty.values())
    flat = window_stats(pd.Series([0.0, 0.0, 0.0]), bars_per_year=365.0)
    assert pd.isna(flat['Sharpe Ratio'])          # zero variance
    assert flat['Net PnL [$]'] == 0.0


def test_is_lower_better():
    assert is_lower_better('Max Drawdown [$]')
    assert is_lower_better('Avg Drawdown [%]')
    assert not is_lower_better('Sharpe Ratio')


def test_first_fill():
    ts = pd.to_datetime(['2024-02-01', '2024-01-15'], utc=True)
    log = pd.DataFrame({'timestamp': ts, 'realized_pnl': [1.0, 0.0]})
    assert first_fill(log) == pd.Timestamp('2024-01-15', tz='UTC')
    assert first_fill(pd.DataFrame()) is None
