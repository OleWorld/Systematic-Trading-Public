"""Tests for validation._heatmap — pivot + Styler rendering."""

import numpy as np
import pandas as pd
import pytest

from validation import ParamHeatmap, param_sweep


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


def _seeded_pnl(fast, slow, t=300, warmup=31):
    # flat before the fixture's 2023-02-01 first fill, drift after (keeps
    # the common-first-fill trim from folding synthetic PnL into bar 1)
    rng = np.random.default_rng(fast * 1000 + slow)
    pnl = rng.normal(50.0 * fast / slow, 50.0, size=t)
    pnl[:warmup] = 0.0
    return pnl


def _factory():
    def run_fn(fast, slow):
        return _FakePortfolio(_seeded_pnl(fast, slow),
                              fills=['2023-02-01', '2023-06-01'])
    return run_fn


def _sweep(**kw):
    return param_sweep(_factory(), grid={'fast': [4, 8], 'slow': [16, 32]},
                       timeframe='1d', days_convention='calendar', **kw)


def test_pivot_orientation_and_values():
    sweep = _sweep()
    hm = sweep.heatmap('Sharpe Ratio')          # x=fast (first), y=slow
    assert isinstance(hm, ParamHeatmap)
    assert list(hm.heatmap.columns) == [4, 8]   # x values
    assert list(hm.heatmap.index) == [16, 32]   # y values
    table = sweep.table.set_index(['fast', 'slow'])['Sharpe Ratio']
    assert hm.heatmap.loc[16, 8] == pytest.approx(float(table.loc[(8, 16)]))
    assert hm.best_cell == (16, 8)              # fast=8/slow=16 ranks highest


def test_where_dropped_cells_render_nan():
    sweep = param_sweep(_factory(), grid={'fast': [4, 8], 'slow': [16, 32]},
                        where=lambda fast, slow: not (fast == 8 and slow == 16),
                        timeframe='1d', days_convention='calendar')
    hm = sweep.heatmap('Sharpe Ratio')
    assert pd.isna(hm.heatmap.loc[16, 8])


def test_three_params_require_x_y_and_fixed():
    def run_fn(fast, slow, span):
        return _FakePortfolio(_seeded_pnl(fast, slow),
                              fills=['2023-02-01'])
    sweep = param_sweep(run_fn,
                        grid={'fast': [4, 8], 'slow': [16, 32],
                              'span': [10, 20]},
                        timeframe='1d', days_convention='calendar')
    with pytest.raises(ValueError, match='span'):
        sweep.heatmap('Sharpe Ratio', x='fast', y='slow')   # span unpinned
    hm = sweep.heatmap('Sharpe Ratio', x='fast', y='slow', fixed={'span': 10})
    assert hm.fixed == {'span': 10}
    with pytest.raises(ValueError):
        sweep.heatmap('Sharpe Ratio')                       # >2 swept, no x/y


def test_single_param_gives_one_row_frame():
    def run_fn(fast):
        return _FakePortfolio(_seeded_pnl(fast, 16), fills=['2023-02-01'])
    sweep = param_sweep(run_fn, grid={'fast': [4, 8]}, timeframe='1d',
                        days_convention='calendar')
    hm = sweep.heatmap('Sharpe Ratio')
    assert hm.y is None
    assert list(hm.heatmap.index) == ['Sharpe Ratio']
    assert list(hm.heatmap.columns) == [4, 8]


def test_bad_metric_or_axis_raises():
    sweep = _sweep()
    with pytest.raises(ValueError):
        sweep.heatmap('No Such Metric')
    with pytest.raises(ValueError):
        sweep.heatmap('Sharpe Ratio', x='nope', y='slow')


def test_styled_smoke_and_caption():
    hm = _sweep().heatmap('Sharpe Ratio')
    styler = hm.styled()
    html = styler.to_html()
    assert 'Sharpe Ratio' in html and 'best' in html


def test_drawdown_best_cell_minimizes():
    sweep = _sweep()
    hm = sweep.heatmap('Max Drawdown [$]')
    table = sweep.table.set_index(['fast', 'slow'])['Max Drawdown [$]']
    fast, slow = table.idxmin()
    assert hm.best_cell == (slow, fast)   # (y_val, x_val)


def test_all_nan_metric_has_no_best_cell():
    # the fakes' fills are all winners -> zero gross loss -> Profit Factor is
    # NaN for every cell (backtest_stats convention), so no best cell exists
    hm = _sweep().heatmap('Profit Factor')
    assert hm.best_cell is None
    assert hm.heatmap.isna().all().all()
