"""Tests for validation._periodic — per-period regime stats."""

import numpy as np
import pandas as pd
import pytest

from analytics import backtest_stats
from validation import periodic_stats


def _two_year_curve(ic=1_000_000.0):
    """2 years of daily bars: +100/day 2023, -50/day 2024 (deterministic)."""
    idx = pd.date_range('2023-01-01', '2024-12-31', freq='D', tz='UTC')
    pnl = np.where(idx < pd.Timestamp('2024-01-01', tz='UTC'), 100.0, -50.0)
    eq = pd.DataFrame({'account_balance': ic + np.cumsum(pnl),
                       'total_commission': 0.0}, index=idx)
    trades = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-03-01', '2023-06-01',
                                     '2024-02-01'], utc=True),
        'realized_pnl': [500.0, -200.0, 300.0]})
    return eq, trades


def test_two_period_hand_case():
    eq, trades = _two_year_curve()
    out = periodic_stats(eq, trades, initial_capital=1_000_000.0,
                         timeframe='1d', days_convention='calendar')
    assert len(out) == 2
    y23, y24 = out.iloc[0], out.iloc[1]
    assert y23['Net PnL [$]'] == pytest.approx(100.0 * 365)
    assert y24['Net PnL [$]'] == pytest.approx(-50.0 * 366)   # 2024 leap
    # Return [%] is against the PERIOD-START balance, not initial capital
    assert y24['Return [%]'] == pytest.approx(
        100.0 * (-50.0 * 366) / (1_000_000.0 + 100.0 * 365))
    # boundary bar: 2024-01-01 PnL diffs against 2023-12-31 close (-50, not
    # a gap) — total across periods reconciles exactly
    assert (y23['Net PnL [$]'] + y24['Net PnL [$]']
            == pytest.approx(100.0 * 365 - 50.0 * 366))
    # trades bucket by fill timestamp
    assert y23['# Closing Trades'] == 2 and y24['# Closing Trades'] == 1
    assert y23['Win Rate [%]'] == pytest.approx(50.0)
    # drawdown is within-period: 2023 rises monotonically -> ~0
    assert y23['Max Drawdown [$]'] == pytest.approx(0.0)
    assert y24['Max Drawdown [$]'] == pytest.approx(50.0 * 366)


def test_consistency_with_backtest_stats_on_period_slice():
    eq, trades = _two_year_curve()
    out = periodic_stats(eq, trades, initial_capital=1_000_000.0,
                         timeframe='1d', days_convention='calendar')
    # the FIRST period's baseline is initial_capital, so its row must match
    # backtest_stats computed on that period's slice
    eq23 = eq.loc[:'2023-12-31']
    bs = backtest_stats(eq23, trades[trades['timestamp'].dt.year == 2023],
                        initial_capital=1_000_000.0, timeframe='1d',
                        days_convention='calendar')
    for label in ('Net PnL [$]', 'Sharpe Ratio', 'Max Drawdown [$]',
                  'Win Rate [%]'):
        a, b = out.iloc[0][label], float(bs[label])
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), label


def test_quarterly_freq_and_start_trim():
    eq, trades = _two_year_curve()
    q = periodic_stats(eq, trades, initial_capital=1_000_000.0,
                       timeframe='1d', days_convention='calendar', freq='QE')
    assert len(q) == 8
    trimmed = periodic_stats(eq, trades, initial_capital=1_000_000.0,
                             timeframe='1d', days_convention='calendar',
                             start='2024-01-01')
    assert len(trimmed) == 1
    # start= reseeds the baseline to the true balance entering the window —
    # pre-start PnL must NOT fold into the first kept period
    assert trimmed.iloc[0]['Net PnL [$]'] == pytest.approx(-50.0 * 366)
    assert trimmed.iloc[0]['Return [%]'] == pytest.approx(
        100.0 * (-50.0 * 366) / (1_000_000.0 + 100.0 * 365))


def test_edge_and_param_law():
    with pytest.raises(ValueError):
        periodic_stats(pd.DataFrame(), pd.DataFrame(), initial_capital=1.0,
                       timeframe='1d', days_convention='calendar',
                       freq='not-a-freq')
    with pytest.raises(ValueError):
        periodic_stats(pd.DataFrame(), pd.DataFrame(), initial_capital=1.0,
                       timeframe='1d', days_convention='calendar', freq=None)
    with pytest.raises(TypeError):
        periodic_stats('nope', pd.DataFrame(), initial_capital=1.0,
                       timeframe='1d', days_convention='calendar')
    out = periodic_stats(pd.DataFrame(), pd.DataFrame(),
                         initial_capital=1_000_000.0, timeframe='1d',
                         days_convention='calendar')
    assert out.empty and 'Sharpe Ratio' in out.columns


def test_periodic_stats_naive_trade_log_raises():
    """UTC law: a naive trade-log timestamp column raises (it previously
    slipped past the gate and silently produced zero-trade periods)."""
    idx = pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True)
    eq = pd.DataFrame({'account_balance': [100.0, 110.0],
                       'total_commission': [0.0, 0.0]}, index=idx)
    log = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-02']),  # naive
                        'realized_pnl': [5.0]})
    with pytest.raises(ValueError, match=r"trade_log\['timestamp'\]"):
        periodic_stats(eq, log, initial_capital=100.0, timeframe='1d',
                       days_convention='calendar')
