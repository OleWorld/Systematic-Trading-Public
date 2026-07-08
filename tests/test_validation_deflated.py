"""Tests for validation._deflated — PSR + Deflated Sharpe."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from validation import (DeflatedSharpeResult, deflated_sharpe, param_sweep,
                        probabilistic_sharpe)


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


def _equity_from_pnl(pnl, ic=1_000_000.0):
    idx = pd.date_range('2023-01-01', periods=len(pnl), freq='D', tz='UTC')
    return pd.DataFrame({'account_balance': ic + np.cumsum(pnl),
                         'total_commission': 0.0}, index=idx)


def test_psr_hand_case_symmetric_two_point():
    # pnl alternating 2, 0: mean=1, std(ddof=1)=~1.00251..., skew=0,
    # kurtosis (non-excess, biased) = 1.0 — every moment hand-checkable.
    pnl = np.array([2.0, 0.0] * 100)
    eq = _equity_from_pnl(pnl)
    got = probabilistic_sharpe(eq, initial_capital=1_000_000.0,
                               timeframe='1d', days_convention='calendar')
    t = len(pnl)
    sr = pnl.mean() / pnl.std(ddof=1)
    expected = float(norm.cdf(sr * math.sqrt(t - 1)
                              / math.sqrt(1.0 - 0.0 + ((1.0 - 1.0) / 4.0)
                                          * sr ** 2)))
    assert got == pytest.approx(expected, rel=1e-9)


def test_psr_decreases_with_higher_benchmark():
    rng = np.random.default_rng(0)
    eq = _equity_from_pnl(rng.normal(100.0, 1_000.0, size=500))
    kw = dict(initial_capital=1_000_000.0, timeframe='1d',
              days_convention='calendar')
    assert probabilistic_sharpe(eq, **kw, benchmark_sharpe=0.0) \
        > probabilistic_sharpe(eq, **kw, benchmark_sharpe=1.0)


def test_psr_degenerate_is_nan():
    flat = _equity_from_pnl(np.zeros(10))
    assert pd.isna(probabilistic_sharpe(flat, initial_capital=1_000_000.0,
                                        timeframe='1d',
                                        days_convention='calendar'))
    short = _equity_from_pnl(np.array([1.0, 2.0]))
    assert pd.isna(probabilistic_sharpe(short, initial_capital=1_000_000.0,
                                        timeframe='1d',
                                        days_convention='calendar'))


def _sweep(n_cells=6):
    def run_fn(k):
        return _FakePortfolio(_seeded_pnl(k, 16), fills=['2023-02-01'])
    return param_sweep(run_fn, grid={'k': list(range(1, n_cells + 1))},
                       timeframe='1d', days_convention='calendar')


def test_dsr_single_trial_reduces_to_psr_vs_zero():
    sweep = _sweep(n_cells=1)
    res = deflated_sharpe(sweep)
    assert isinstance(res, DeflatedSharpeResult)
    assert res.sr0_annualized == 0.0 and res.n_trials == 1
    psr = probabilistic_sharpe(
        sweep.equity(k=1), initial_capital=1_000_000.0, timeframe='1d',
        days_convention='calendar', start=sweep.stats_start_resolved)
    assert res.dsr == pytest.approx(psr, rel=1e-12)


def test_dsr_monotonically_decreases_in_n_trials():
    sweep = _sweep()
    dsrs = [deflated_sharpe(sweep, n_trials=n).dsr for n in (2, 10, 100)]
    assert dsrs[0] > dsrs[1] > dsrs[2]
    assert deflated_sharpe(sweep).n_trials == 6      # default = cell count


def test_dsr_winner_and_audit_fields():
    sweep = _sweep()
    res = deflated_sharpe(sweep)
    assert res.winner_params == sweep.best('Sharpe Ratio')
    assert res.n_bars > 0 and res.var_sr >= 0.0
    assert 0.0 <= res.dsr <= 1.0


def test_dsr_validation():
    sweep = _sweep()
    with pytest.raises(ValueError):
        deflated_sharpe(sweep, n_trials=0)
