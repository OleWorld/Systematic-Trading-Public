"""Tests for validation._walkforward — slice-once walk-forward."""

import numpy as np
import pandas as pd
import pytest

from validation import WalkForwardResult, param_sweep, walk_forward


class _FakePortfolio:
    """Duck-typed portfolio double (duplicated per test file — repo
    convention; tests/ is not a package)."""

    def __init__(self, pnl, initial_capital=1_000_000.0, fills=()):
        idx = pd.date_range('2023-01-01', periods=len(pnl), freq='D', tz='UTC')
        self._eq = pd.DataFrame(
            {'account_balance': initial_capital + np.cumsum(pnl),
             'total_commission': 0.0}, index=idx)
        self._trades = pd.DataFrame(
            {'timestamp': pd.to_datetime(list(fills), utc=True),
             'realized_pnl': [1.0] * len(fills)})
        self.initial_capital = initial_capital

    def get_equity_curve(self):
        return self._eq

    def get_trade_log(self):
        return self._trades


def _regime_sweep(initial_capital_b=1_000_000.0):
    """Two cells over 4 years of daily data. Cell a=1 earns +100/day in
    2021-2022 and -50/day after; cell a=2 is the mirror image — so the
    per-fold winner is known by construction (selection lags one fold)."""
    idx = pd.date_range('2021-01-01', '2024-12-31', freq='D', tz='UTC')
    flip = idx < pd.Timestamp('2023-01-01', tz='UTC')

    def run_fn(a):
        base = np.where(flip, 100.0, -50.0)
        pnl = base if a == 1 else -base
        rng = np.random.default_rng(a)
        pnl = pnl + rng.normal(0.0, 1.0, size=len(idx))   # break exact ties
        fp = _FakePortfolio(pnl, initial_capital=(1_000_000.0 if a == 1
                                                  else initial_capital_b))
        fp._eq.index = idx                                # align calendars
        fp._trades = pd.DataFrame(
            {'timestamp': [idx[0]], 'realized_pnl': [1.0]})
        return fp

    return param_sweep(run_fn, grid={'a': [1, 2]}, timeframe='1d',
                       days_convention='calendar')


def test_yearly_anchored_selection_and_stitching():
    sweep = _regime_sweep()
    wf = walk_forward(sweep, oos='YE')
    assert isinstance(wf, WalkForwardResult)
    folds = wf.folds
    # first OOS fold needs >= one full prior year in IS -> starts 2022
    assert folds['oos_start'].iloc[0].year == 2022
    assert list(folds['fold']) == [0, 1, 2]               # 2022, 2023, 2024
    # ANCHORED IS always starts 2021. a=1's cumulative record stays ahead
    # even after the 2023 regime flip (two +100 years vs one -50 year), so
    # a=1 wins EVERY fold — the classic anchored-selection lag.
    assert list(folds['a']) == [1, 1, 1]
    # stitched = concat of the winners' OOS slices (all a=1 here)
    pnl1 = sweep.pnl(a=1)
    manual = pd.concat([pnl1.loc['2022-01-01':'2022-12-31'],
                        pnl1.loc['2023-01-01':'2023-12-31'],
                        pnl1.loc['2024-01-01':'2024-12-31']])
    pd.testing.assert_series_equal(wf.stitched_pnl, manual,
                                   check_names=False)
    # folds table carries IS score + OOS stats columns
    for col in ('is_Sharpe Ratio', 'oos Sharpe Ratio', 'oos Net PnL [$]',
                'oos Max Drawdown [$]'):
        assert col in folds.columns


def test_oos_matrix_shape_and_fixed_config_column():
    sweep = _regime_sweep()
    wf = walk_forward(sweep, oos='YE')
    m = wf.oos_matrix('Sharpe Ratio')
    assert m.shape == (3, 2)
    assert list(m.columns.names) == ['a']
    # fixed-config consistency: a=1 positive in 2022, negative after
    col = m[(1,)]
    assert col.iloc[0] > 0 and col.iloc[2] < 0


def test_rolling_is_window():
    wf = walk_forward(_regime_sweep(), oos='YE', is_window='365D')
    # ROLLING trailing-year IS reacts one fold after the flip: 2022's IS is
    # 2021 (a=1), 2023's IS is 2022 (still a=1), 2024's IS is 2023 (a=2).
    assert list(wf.folds['a']) == [1, 1, 2]


def test_stitched_stats_labels_and_uniform_capital():
    wf = walk_forward(_regime_sweep(), oos='YE')
    labels = list(wf.stitched_stats.index)
    assert labels == ['Sharpe Ratio', 'Sortino Ratio', 'Net PnL [$]',
                      'Volatility (Ann.) [$]', 'Max Drawdown [$]',
                      'Return [%]', 'CAGR [%]', 'Max Drawdown [%]']
    assert not pd.isna(wf.stitched_stats['Return [%]'])


def test_non_uniform_capital_nans_percent_stats():
    wf = walk_forward(_regime_sweep(initial_capital_b=2_000_000.0), oos='YE')
    assert pd.isna(wf.stitched_stats['Return [%]'])
    assert pd.isna(wf.stitched_stats['CAGR [%]'])
    assert not pd.isna(wf.stitched_stats['Sharpe Ratio'])


def test_timedelta_oos_and_explicit_start():
    wf = walk_forward(_regime_sweep(), oos=pd.Timedelta('180D'),
                      start='2021-01-01')
    assert len(wf.folds) >= 6
    spans = (wf.folds['oos_end'] - wf.folds['oos_start']).iloc[:-1]
    assert (spans <= pd.Timedelta('180D')).all()


def test_validation_raises():
    sweep = _regime_sweep()
    with pytest.raises(ValueError):
        walk_forward(sweep, selection_metric='Win Rate [%]')   # unsupported
    with pytest.raises(ValueError):
        walk_forward(sweep, oos=12345)                          # bad oos type


def test_mismatched_indexes_raise():
    def run_fn(a):
        t = 100 if a == 1 else 90
        return _FakePortfolio(np.full(t, 10.0), fills=['2023-02-01'])
    sweep = param_sweep(run_fn, grid={'a': [1, 2]}, timeframe='1d',
                        days_convention='calendar')
    with pytest.raises(ValueError, match='identical'):
        walk_forward(sweep, oos='ME')
