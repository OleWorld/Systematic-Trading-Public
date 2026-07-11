"""Tests for validation._sweep — grid execution, SweepResult, cell cache."""

import logging
import os

import numpy as np
import pandas as pd
import pytest

from validation import load_sweep, param_sweep


class _FakePortfolio:
    """Duck-typed portfolio-like: the factory contract's full surface."""

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
    """Deterministic per-cell PnL: flat (zero) for the first ``warmup`` bars
    — honoring the flat-before-first-fill premise the fills fixture claims
    (2023-01-01 start + 31 flat days = first fill 2023-02-01) — then drift
    scaling with fast/slow so cells rank predictably. Noise sits an order of
    magnitude below the drift gaps so rankings are robust, not seed-lottery;
    the flat head keeps the common-first-fill trim from folding synthetic
    pre-start PnL into the first kept bar (backtest_stats folds pre-start
    PnL there by design)."""
    rng = np.random.default_rng(fast * 1000 + slow)
    pnl = rng.normal(50.0 * fast / slow, 50.0, size=t)
    pnl[:warmup] = 0.0
    return pnl


def _factory(calls=None):
    def run_fn(fast, slow):
        if calls is not None:
            calls.append((fast, slow))
        return _FakePortfolio(_seeded_pnl(fast, slow),
                              fills=['2023-02-01', '2023-06-01'])
    return run_fn


def _make_sweep(**kw):
    return param_sweep(_factory(), grid={'fast': [4, 8], 'slow': [16, 32]},
                       timeframe='1d', days_convention='calendar', **kw)


def test_grid_enumeration_order_and_where():
    calls = []
    param_sweep(_factory(calls), grid={'fast': [4, 8], 'slow': [16, 32]},
                where=lambda fast, slow: not (fast == 8 and slow == 16),
                timeframe='1d', days_convention='calendar')
    assert calls == [(4, 16), (4, 32), (8, 32)]   # product order, cell dropped


def test_table_has_param_columns_then_stats():
    sweep = _make_sweep()
    assert list(sweep.table.columns[:2]) == ['fast', 'slow']
    assert 'Sharpe Ratio' in sweep.table.columns
    assert len(sweep.table) == 4
    assert sweep.keys() == [(4, 16), (4, 32), (8, 16), (8, 32)]


def test_stats_start_common_first_fill_and_none():
    sweep = _make_sweep()   # every fake fills first at 2023-02-01
    assert sweep.stats_start_resolved == pd.Timestamp('2023-02-01', tz='UTC')
    full = _make_sweep(stats_start=None)
    assert full.stats_start_resolved is None
    # the trim actually reaches backtest_stats: trimmed table starts at the
    # first fill, full table at the first bar. (Net PnL is trim-invariant by
    # design — final minus initial_capital — so it is NOT the probe.)
    assert sweep.table['Start'].iloc[0] == pd.Timestamp('2023-02-01', tz='UTC')
    assert full.table['Start'].iloc[0] == pd.Timestamp('2023-01-01', tz='UTC')


def test_stats_start_no_fills_resolves_none():
    def run_fn(x):
        return _FakePortfolio(_seeded_pnl(4, 16), fills=[])
    sweep = param_sweep(run_fn, grid={'x': [1]}, timeframe='1d',
                        days_convention='calendar')
    assert sweep.stats_start_resolved is None


def test_cell_accessors_and_unknown_params_raise():
    sweep = _make_sweep()
    pnl = sweep.pnl(fast=4, slow=16)
    assert len(pnl) == 300
    assert sweep.initial_capital(fast=4, slow=16) == 1_000_000.0
    with pytest.raises(KeyError):
        sweep.pnl(fast=5, slow=16)
    with pytest.raises(ValueError):
        sweep.pnl(fast=4)          # missing param name


def test_best_is_direction_aware():
    sweep = _make_sweep()
    # highest drift/vol is fast=8, slow=16 (ratio 0.5)
    assert sweep.best('Sharpe Ratio') == {'fast': 8, 'slow': 16}
    best_dd = sweep.best('Max Drawdown [$]')
    col = sweep.table.set_index(['fast', 'slow'])['Max Drawdown [$]']
    assert tuple(best_dd.values()) == col.idxmin()


def test_grid_validation_raises():
    kw = dict(timeframe='1d', days_convention='calendar')
    with pytest.raises(ValueError):
        param_sweep(_factory(), grid={}, **kw)
    with pytest.raises(ValueError):
        param_sweep(_factory(), grid={'fast': []}, **kw)
    with pytest.raises(ValueError):
        param_sweep(_factory(), grid={'fast': [4, 4]}, **kw)
    with pytest.raises(TypeError):
        param_sweep(_factory(), grid={'fast': [[4]]}, **kw)
    with pytest.raises(TypeError):
        param_sweep('not callable', grid={'fast': [4]}, **kw)
    with pytest.raises(ValueError):
        param_sweep(_factory(), grid={'fast': [4]}, stats_start='bogus policy',
                    **kw)


def test_factory_exception_propagates():
    def boom(fast):
        raise RuntimeError('factory blew up')
    with pytest.raises(RuntimeError, match='factory blew up'):
        param_sweep(boom, grid={'fast': [4]}, timeframe='1d',
                    days_convention='calendar')


from validation import load_sweep


def test_cache_roundtrip_and_resume(tmp_path):
    cache = tmp_path / 'sweep'
    calls = []
    kw = dict(grid={'fast': [4, 8], 'slow': [16, 32]}, timeframe='1d',
              days_convention='calendar', cache_dir=str(cache))
    first = param_sweep(_factory(calls), **kw)
    assert len(calls) == 4
    # resume: identical call runs nothing new, table identical
    second = param_sweep(_factory(calls), **kw)
    assert len(calls) == 4
    pd.testing.assert_frame_equal(first.table, second.table)
    # offline load: same table again
    loaded = load_sweep(str(cache))
    pd.testing.assert_frame_equal(first.table, loaded.table)
    assert loaded.param_names == ('fast', 'slow')
    assert loaded.keys() == first.keys()


def test_cache_preserves_param_types(tmp_path):
    def run_fn(fast, label, frac):
        return _FakePortfolio(_seeded_pnl(fast, 16), fills=['2023-02-01'])
    kw = dict(grid={'fast': [4], 'label': ['a/b'], 'frac': [0.25]},
              timeframe='1d', days_convention='calendar',
              cache_dir=str(tmp_path / 's'))
    param_sweep(run_fn, **kw)
    loaded = load_sweep(str(tmp_path / 's'))
    assert loaded.keys() == [(4, 'a/b', 0.25)]   # int/str/float round-trip


def test_manifest_mismatch_raises(tmp_path):
    cache = tmp_path / 'sweep'
    param_sweep(_factory(), grid={'fast': [4], 'slow': [16]},
                timeframe='1d', days_convention='calendar',
                cache_dir=str(cache))
    with pytest.raises(ValueError, match='manifest'):
        param_sweep(_factory(), grid={'fast': [4, 8], 'slow': [16]},
                    timeframe='1d', days_convention='calendar',
                    cache_dir=str(cache))
    with pytest.raises(ValueError, match='manifest'):
        param_sweep(_factory(), grid={'fast': [4], 'slow': [16]},
                    timeframe='1d', days_convention='business',
                    cache_dir=str(cache))


def test_partial_cache_resumes_after_crash(tmp_path):
    cache = tmp_path / 'sweep'

    class _Boom(Exception):
        pass

    calls = []

    def flaky(fast, slow):
        calls.append((fast, slow))
        if (fast, slow) == (8, 16):
            raise _Boom()
        return _FakePortfolio(_seeded_pnl(fast, slow), fills=['2023-02-01'])

    kw = dict(grid={'fast': [4, 8], 'slow': [16, 32]}, timeframe='1d',
              days_convention='calendar', cache_dir=str(cache))
    with pytest.raises(_Boom):
        param_sweep(flaky, **kw)
    assert calls == [(4, 16), (4, 32), (8, 16)]
    # partial cache loads with the cells it has
    assert load_sweep(str(cache)).keys() == [(4, 16), (4, 32)]
    # fixed factory resumes: only the 2 missing cells run
    calls.clear()
    full = param_sweep(_factory(calls), **kw)
    assert calls == [(8, 16), (8, 32)]
    assert len(full.keys()) == 4


def test_load_sweep_missing_manifest_raises(tmp_path):
    with pytest.raises(ValueError, match='manifest'):
        load_sweep(str(tmp_path / 'nope'))


def test_cell_store_survives_transient_rename_lock(tmp_path, monkeypatch):
    calls = {'n': 0}
    real_replace = os.replace
    def flaky(src, dst):
        calls['n'] += 1
        if calls['n'] == 1:
            raise PermissionError(5, 'Access is denied')
        return real_replace(src, dst)
    monkeypatch.setattr('runlog._serialize.os.replace', flaky)
    monkeypatch.setattr('runlog._serialize.time.sleep', lambda s: None)
    sweep = _make_sweep(cache_dir=tmp_path / 'cache')
    assert len(sweep.keys()) == 4          # all cells stored despite the lock
    reloaded = load_sweep(tmp_path / 'cache')
    assert len(reloaded.keys()) == 4


# ──────────────────────────────────────────────
# Table reseeds the entering balance (F4)
# ──────────────────────────────────────────────

def test_table_reseeds_entering_balance_no_fold_in_spike():
    """Cell A earns +100/bar for 30 bars BEFORE the common first fill;
    cell B is flat there. Under the old fold-in trim, A's first kept bar
    carried a synthetic +3000 spike. Reseeded stats must equal a manual
    backtest_stats call with initial_capital = capital + 3000."""
    def run_fn(cell):
        pnl = np.zeros(100)
        if cell == 1:
            pnl[:30] = 100.0                  # head PnL before common start
        pnl[30:] = 10.0
        pnl[31::2] = 20.0                     # in-window variance: a constant
        #                                       window would make Sharpe NaN
        #                                       on BOTH sides (NaN != NaN)
        # cell 1's first fill is early; cell 2's is at bar 30 -> common
        fills = ['2023-01-05', '2023-01-31'] if cell == 1 else ['2023-01-31']
        return _FakePortfolio(pnl, fills=fills)

    sweep = param_sweep(run_fn, grid={'cell': [1, 2]}, timeframe='1d',
                        days_convention='calendar',
                        stats_start='common_first_fill')
    start = sweep.stats_start_resolved
    assert start == pd.Timestamp('2023-01-31', tz='UTC')

    from analytics import backtest_stats
    eq1, tr1 = sweep.equity(cell=1), sweep.trades(cell=1)
    cap1 = sweep.initial_capital(cell=1)
    expected = backtest_stats(
        eq1, tr1,
        initial_capital=cap1 + 3000.0,                    # entering balance
        timeframe='1d', days_convention='calendar', start=start)
    row = sweep.table[sweep.table['cell'] == 1].iloc[0]
    assert row['Sharpe Ratio'] == expected['Sharpe Ratio']
    assert row['Net PnL [$]'] == expected['Net PnL [$]']
    # and it must DIFFER from the old fold-in numbers
    folded = backtest_stats(
        eq1, tr1, initial_capital=cap1,
        timeframe='1d', days_convention='calendar', start=start)
    assert row['Sharpe Ratio'] != folded['Sharpe Ratio']


def test_table_stats_start_none_unchanged():
    sweep = _make_sweep(stats_start=None)
    from analytics import backtest_stats
    params = dict(zip(sweep.param_names, sweep.keys()[0]))
    expected = backtest_stats(sweep.equity(**params), sweep.trades(**params),
                              initial_capital=sweep.initial_capital(**params),
                              timeframe='1d', days_convention='calendar')
    row = sweep.table.iloc[0]
    assert row['Sharpe Ratio'] == expected['Sharpe Ratio']
    assert row['Net PnL [$]'] == expected['Net PnL [$]']


def test_table_nonpositive_entering_balance_falls_back_full_history(caplog):
    def run_fn(cell):
        pnl = np.zeros(100)
        pnl[:30] = -60_000.0                  # wipes out 1M capital pre-start
        return _FakePortfolio(pnl, fills=['2023-01-31'])
    sweep = param_sweep(run_fn, grid={'cell': [1]}, timeframe='1d',
                        days_convention='calendar',
                        stats_start='common_first_fill')
    with caplog.at_level(logging.WARNING, logger='validation._sweep'):
        table = sweep.table
    assert any('non-positive' in rec.message for rec in caplog.records)
    from analytics import backtest_stats
    expected = backtest_stats(sweep.equity(cell=1), sweep.trades(cell=1),
                              initial_capital=sweep.initial_capital(cell=1),
                              timeframe='1d', days_convention='calendar')
    assert table.iloc[0]['Net PnL [$]'] == expected['Net PnL [$]']
