"""Tests for validation._bootstrap — resampling machinery + bootstrap_stats."""

import math

import numpy as np
import pandas as pd
import pytest

from validation._bootstrap import (_flat_top_window, _resample_indices,
                                   politis_white_block_length)


def _ar1(t, phi, seed=0, scale=1.0):
    """AR(1) series x_t = phi*x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, scale, size=t)
    x = np.empty(t)
    x[0] = eps[0]
    for i in range(1, t):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _lag1_autocorr(m):
    """Mean lag-1 autocorrelation across rows of a 2-D array."""
    a, b = m[:, :-1], m[:, 1:]
    ac = ((a - a.mean(axis=1, keepdims=True))
          * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
    return float((ac / (a.std(axis=1) * b.std(axis=1))).mean())


def test_flat_top_window_hand_values():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    assert list(_flat_top_window(x)) == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]


def test_block_length_larger_for_persistent_series():
    white = politis_white_block_length(_ar1(1000, phi=0.0, seed=1))
    persistent = politis_white_block_length(_ar1(1000, phi=0.9, seed=1))
    assert persistent > white
    assert persistent > 10.0
    assert white < 5.0


def test_block_length_bounds_and_degenerate():
    t = 500
    b = politis_white_block_length(_ar1(t, phi=0.95, seed=2))
    assert 1.0 <= b <= min(3.0 * np.sqrt(t), t / 3.0)
    assert politis_white_block_length(np.zeros(100)) == 1.0   # constant
    assert politis_white_block_length(np.array([1.0, 2.0])) == 1.0  # too short


def test_block_length_deterministic():
    x = _ar1(800, phi=0.5, seed=3)
    assert politis_white_block_length(x) == politis_white_block_length(x)


def test_indices_shapes_and_ranges():
    rng = np.random.default_rng(0)
    for method in ('stationary', 'circular', 'iid'):
        idx = _resample_indices(rng, t=50, b=7, method=method, block_length=5.0)
        assert idx.shape == (7, 50)
        assert idx.min() >= 0 and idx.max() < 50


def test_unknown_method_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _resample_indices(rng, t=50, b=2, method='jackknife', block_length=5.0)


def test_block_methods_preserve_autocorrelation_iid_destroys_it():
    x = _ar1(1000, phi=0.8, seed=4)
    rng = np.random.default_rng(5)
    stat = x[_resample_indices(rng, 1000, 200, 'stationary', 20.0)]
    circ = x[_resample_indices(rng, 1000, 200, 'circular', 20.0)]
    iid = x[_resample_indices(rng, 1000, 200, 'iid', 20.0)]
    assert _lag1_autocorr(stat) > 0.5
    assert _lag1_autocorr(circ) > 0.5
    assert abs(_lag1_autocorr(iid)) < 0.1


def test_stationary_blocks_are_contiguous_runs():
    """Within a resample row, consecutive indices mostly step by +1 (mod T)."""
    rng = np.random.default_rng(6)
    idx = _resample_indices(rng, t=200, b=50, method='stationary',
                            block_length=10.0)
    steps = (idx[:, 1:] - idx[:, :-1]) % 200
    frac_contiguous = float((steps == 1).mean())
    assert 0.8 < frac_contiguous < 0.95   # ~1 - 1/block_length


def _naive_politis_white(x):
    """Scalar-loop transcription of the documented Politis-White algorithm,
    independent of the vectorized production code — guards against
    indexing/coefficient slips between the two."""
    t = len(x)
    if t < 3:
        return 1.0
    mean = sum(x) / t
    xc = [v - mean for v in x]
    denom = sum(v * v for v in xc)
    if denom == 0.0:
        return 1.0

    def autocov(k):
        return sum(xc[i + k] * xc[i] for i in range(t - k)) / t

    gamma0 = denom / t
    k_n = max(5, int(math.sqrt(math.log10(t))))
    n_lags = min(int(math.ceil(math.sqrt(t))) + k_n, t - 1)
    threshold = 2.0 * math.sqrt(math.log10(t) / t)
    rho = [autocov(k) / gamma0 for k in range(1, n_lags + 1)]
    m_hat = None
    for m in range(0, n_lags - k_n + 1):
        if all(abs(r) < threshold for r in rho[m:m + k_n]):
            m_hat = m
            break
    if m_hat is None:
        m_hat = max(1, n_lags - k_n)
    big_m = min(2 * m_hat, t - 1)
    if big_m < 1:
        return 1.0

    def lam(u):
        au = abs(u)
        if au <= 0.5:
            return 1.0
        if au <= 1.0:
            return 2.0 * (1.0 - au)
        return 0.0

    g = 2.0 * sum(lam(k / big_m) * k * autocov(k)
                  for k in range(1, big_m + 1))
    d = 2.0 * (gamma0 + 2.0 * sum(lam(k / big_m) * autocov(k)
                                  for k in range(1, big_m + 1))) ** 2
    if d <= 0.0:
        return 1.0
    b = ((2.0 * g * g) / d) ** (1.0 / 3.0) * t ** (1.0 / 3.0)
    return float(min(max(b, 1.0), min(3.0 * math.sqrt(t), t / 3.0)))


def test_block_length_matches_naive_reference():
    """Vectorized production estimator == independent loop transcription
    (golden cross-check, same pattern as the strategies' vectorized
    recomputation tests)."""
    for phi, seed, t in ((0.0, 1, 200), (0.5, 3, 500), (0.9, 2, 800)):
        x = _ar1(t, phi=phi, seed=seed)
        assert politis_white_block_length(x) == pytest.approx(
            _naive_politis_white(list(x)), rel=1e-9), (phi, seed, t)


from analytics import backtest_stats
from validation import BootstrapResult, bootstrap_stats


def _drift_equity(t=400, drift=200.0, noise=1_000.0, seed=11, ic=1_000_000):
    rng = np.random.default_rng(seed)
    pnl = rng.normal(drift, noise, size=t)
    idx = pd.date_range('2023-01-01', periods=t, freq='D', tz='UTC')
    return pd.DataFrame({'account_balance': ic + np.cumsum(pnl),
                         'total_commission': 0.0}, index=idx)


def test_estimates_equal_backtest_stats():
    eq = _drift_equity()
    res = bootstrap_stats(eq, initial_capital=1_000_000, timeframe='1d',
                          days_convention='calendar', n_resamples=50, seed=0)
    bs = backtest_stats(eq, pd.DataFrame(), initial_capital=1_000_000,
                        timeframe='1d', days_convention='calendar')
    for label in ('Sharpe Ratio', 'Net PnL [$]', 'CAGR [%]',
                  'Max Drawdown [$]', 'Max Drawdown [%]'):
        assert res.table.loc[label, 'estimate'] == pytest.approx(
            float(bs[label]), rel=1e-12), label


def test_result_shape_and_determinism():
    eq = _drift_equity()
    kw = dict(initial_capital=1_000_000, timeframe='1d',
              days_convention='calendar', n_resamples=200, seed=42)
    a = bootstrap_stats(eq, **kw)
    b = bootstrap_stats(eq, **kw)
    assert isinstance(a, BootstrapResult)
    assert list(a.table.columns) == ['estimate', 'ci_low', 'ci_high', 'p_value']
    assert list(a.table.index) == ['Sharpe Ratio', 'Net PnL [$]', 'CAGR [%]',
                                   'Max Drawdown [$]', 'Max Drawdown [%]']
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.block_length == b.block_length and a.n_bars == 400


def test_pvalue_small_for_strong_drift_large_for_noise():
    strong = bootstrap_stats(_drift_equity(drift=500.0, noise=1_000.0),
                             initial_capital=1_000_000, timeframe='1d',
                             days_convention='calendar',
                             n_resamples=500, seed=1)
    # negative drift => the null must NOT be rejected, robustly (p > 0.5-ish)
    noise = bootstrap_stats(_drift_equity(drift=-100.0, noise=1_000.0, seed=13),
                            initial_capital=1_000_000, timeframe='1d',
                            days_convention='calendar',
                            n_resamples=500, seed=1)
    assert strong.table.loc['Sharpe Ratio', 'p_value'] < 0.01
    assert noise.table.loc['Sharpe Ratio', 'p_value'] > 0.3
    assert pd.isna(strong.table.loc['Max Drawdown [$]', 'p_value'])


def test_ci_brackets_estimate_and_respects_level():
    res = bootstrap_stats(_drift_equity(), initial_capital=1_000_000,
                          timeframe='1d', days_convention='calendar',
                          n_resamples=500, ci=0.90, seed=2)
    row = res.table.loc['Sharpe Ratio']
    assert row['ci_low'] < row['estimate'] < row['ci_high']


def test_param_validation_raises():
    eq = _drift_equity(t=10)
    kw = dict(initial_capital=1_000_000, timeframe='1d',
              days_convention='calendar')
    with pytest.raises(ValueError):
        bootstrap_stats(eq, **kw, method='parametric')
    with pytest.raises(ValueError):
        bootstrap_stats(eq, **kw, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_stats(eq, **kw, ci=1.0)
    with pytest.raises(ValueError):
        bootstrap_stats(eq, **kw, block_length=0.0)
    with pytest.raises(ValueError):
        bootstrap_stats(eq, initial_capital=-1.0, timeframe='1d',
                        days_convention='calendar')


def test_degenerate_data_gives_nan_never_raises():
    res = bootstrap_stats(pd.DataFrame(), initial_capital=1_000_000,
                          timeframe='1d', days_convention='calendar')
    assert res.table['ci_low'].isna().all() and res.n_bars == 0
    idx = pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC')
    flat = pd.DataFrame({'account_balance': 1_000_000.0,
                         'total_commission': 0.0}, index=idx)
    res = bootstrap_stats(flat, initial_capital=1_000_000, timeframe='1d',
                          days_convention='calendar', n_resamples=50, seed=0)
    assert pd.isna(res.table.loc['Sharpe Ratio', 'ci_low'])   # zero variance


def test_explicit_block_length_is_used():
    res = bootstrap_stats(_drift_equity(), initial_capital=1_000_000,
                          timeframe='1d', days_convention='calendar',
                          n_resamples=50, block_length=7.5, seed=0)
    assert res.block_length == 7.5
