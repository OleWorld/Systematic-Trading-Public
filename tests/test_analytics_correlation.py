"""Unit tests for ``analytics.correlation_matrix``.

Run from the repo root:  python -m pytest tests/test_analytics_correlation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from analytics import correlation_matrix


def _frame(**cols) -> pd.DataFrame:
    """Build a DataFrame from keyword series with a default RangeIndex."""
    return pd.DataFrame(cols)


# ──────────────────────────────────────────────
# Core correlation cases
# ──────────────────────────────────────────────

def test_identity_two_columns_correlates_to_one():
    """Two identical columns → off-diagonal == 1.0."""
    x = np.arange(50, dtype=float)
    m = correlation_matrix(_frame(a=x, b=x))
    assert m.shape == (2, 2)
    assert np.isclose(m.loc['a', 'b'], 1.0)
    assert np.isclose(m.loc['b', 'a'], 1.0)


def test_anti_correlated_two_columns_correlates_to_minus_one():
    """y = -x → off-diagonal == -1.0."""
    x = np.linspace(-1.0, 1.0, 80)
    m = correlation_matrix(_frame(a=x, b=-x))
    assert np.isclose(m.loc['a', 'b'], -1.0)


def test_independent_columns_correlation_near_zero():
    """Independent normals over a large sample → off-diagonal near 0."""
    rng = np.random.default_rng(seed=42)
    a = rng.normal(size=5_000)
    b = rng.normal(size=5_000)
    m = correlation_matrix(_frame(a=a, b=b))
    assert abs(m.loc['a', 'b']) < 0.1


def test_diagonal_is_one_and_matrix_is_symmetric():
    """For any frame, diagonal == 1 and M == M.T."""
    rng = np.random.default_rng(seed=7)
    df = _frame(
        a=rng.normal(size=200),
        b=rng.normal(size=200),
        c=rng.normal(size=200),
    )
    m = correlation_matrix(df)
    np.testing.assert_allclose(np.diag(m.to_numpy()), 1.0)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T, atol=0.0)
    assert list(m.index) == ['a', 'b', 'c']
    assert list(m.columns) == ['a', 'b', 'c']


# ──────────────────────────────────────────────
# Method dispatch
# ──────────────────────────────────────────────

def test_spearman_method_uses_rank_correlation():
    """Spearman on a monotone-but-nonlinear pair should yield 1.0 even though
    Pearson does not — proves the method parameter is honored."""
    x = np.arange(1, 21, dtype=float)
    y = x ** 3  # strictly monotone but nonlinear
    df = _frame(a=x, b=y)
    spearman = correlation_matrix(df, method='spearman')
    pearson = correlation_matrix(df, method='pearson')
    assert np.isclose(spearman.loc['a', 'b'], 1.0)
    assert pearson.loc['a', 'b'] < 1.0  # nonlinearity hurts Pearson


# ──────────────────────────────────────────────
# NaN handling (pairwise pandas default)
# ──────────────────────────────────────────────

def test_pairwise_nan_handling_matches_pandas_default():
    """Inject NaNs in different rows of two columns. The helper should
    behave like pandas' pairwise corr — no exception, and the (a, b) entry
    should match a manual pairwise computation."""
    rng = np.random.default_rng(seed=3)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    a[5:10] = np.nan
    b[20:25] = np.nan
    df = _frame(a=a, b=b)
    m = correlation_matrix(df)
    # Pandas pairwise: drop rows where EITHER a or b is NaN, then corr.
    expected = df.dropna().corr().loc['a', 'b']
    assert np.isclose(m.loc['a', 'b'], expected)


# ──────────────────────────────────────────────
# Lookback
# ──────────────────────────────────────────────

def test_lookback_uses_only_recent_rows():
    """Build a regime-shift frame: first half correlated, second half
    anti-correlated. With lookback=100 (second half only) we should see
    correlation near -1; with lookback=None (full frame) we should see
    a value in between."""
    x_first = np.linspace(-1.0, 1.0, 100)
    x_second = np.linspace(-1.0, 1.0, 100)
    a = np.concatenate([x_first, x_second])
    b = np.concatenate([x_first, -x_second])  # second half: b = -a
    df = _frame(a=a, b=b)

    full = correlation_matrix(df).loc['a', 'b']
    recent = correlation_matrix(df, lookback=100).loc['a', 'b']

    assert np.isclose(recent, -1.0)
    # Full-window correlation is somewhere between -1 and +1, definitely not
    # at either extreme — and clearly distinct from the recent-window value.
    assert -0.5 < full < 0.5
    assert abs(full - recent) > 0.4


def test_lookback_larger_than_frame_degrades_to_full():
    """Oversized lookback should silently use the full frame."""
    rng = np.random.default_rng(seed=11)
    df = _frame(a=rng.normal(size=20), b=rng.normal(size=20))
    full = correlation_matrix(df, lookback=None)
    oversized = correlation_matrix(df, lookback=10_000)
    np.testing.assert_allclose(full.to_numpy(), oversized.to_numpy())


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="values must be a pd.DataFrame"):
        correlation_matrix([1, 2, 3])  # type: ignore[arg-type]


def test_rejects_single_column():
    df = _frame(a=np.arange(10, dtype=float))
    with pytest.raises(ValueError, match=">= 2 columns"):
        correlation_matrix(df)


def test_rejects_single_row():
    df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
    with pytest.raises(ValueError, match=">= 2 rows"):
        correlation_matrix(df)


def test_rejects_lookback_below_two():
    df = _frame(a=np.arange(10, dtype=float), b=np.arange(10, dtype=float))
    for bad in (1, 0, -5):
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            correlation_matrix(df, lookback=bad)


def test_rejects_unknown_method():
    df = _frame(a=np.arange(10, dtype=float), b=np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="method must be one of"):
        correlation_matrix(df, method='bogus')


def test_accepts_all_documented_methods():
    """pearson, spearman, kendall must all be accepted without error."""
    rng = np.random.default_rng(seed=0)
    df = _frame(a=rng.normal(size=30), b=rng.normal(size=30))
    for m in ('pearson', 'spearman', 'kendall'):
        result = correlation_matrix(df, method=m)
        assert result.shape == (2, 2)


# ──────────────────────────────────────────────
# Ledoit-Wolf shrinkage
# ──────────────────────────────────────────────

def _noisy_frame(n_rows: int = 60, n_cols: int = 4, seed: int = 5) -> pd.DataFrame:
    """Correlated noisy returns — small sample so LW intensity is > 0."""
    rng = np.random.default_rng(seed=seed)
    common = rng.normal(size=n_rows)
    cols = {
        f'c{i}': common + rng.normal(scale=1.5, size=n_rows)
        for i in range(n_cols)
    }
    return pd.DataFrame(cols)


def _sklearn_lw_corr(values: pd.DataFrame) -> np.ndarray:
    """Reference: direct sklearn LedoitWolf fit + cov→corr conversion."""
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(values.to_numpy(dtype=float)).covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def test_ledoit_wolf_matches_direct_sklearn_fit():
    """Golden: shrinkage='ledoit_wolf' equals cov2corr of a direct sklearn fit."""
    df = _noisy_frame()
    m = correlation_matrix(df, shrinkage='ledoit_wolf')
    np.testing.assert_allclose(m.to_numpy(), _sklearn_lw_corr(df), atol=1e-12)
    assert list(m.index) == list(df.columns)
    assert list(m.columns) == list(df.columns)


def test_ledoit_wolf_output_is_valid_correlation_matrix():
    """Unit diagonal, symmetric, strictly PD, entries in [-1, 1]."""
    m = correlation_matrix(_noisy_frame(), shrinkage='ledoit_wolf')
    arr = m.to_numpy()
    np.testing.assert_allclose(np.diag(arr), 1.0)
    np.testing.assert_allclose(arr, arr.T, atol=0.0)
    assert np.all(arr >= -1.0) and np.all(arr <= 1.0)
    assert np.linalg.eigvalsh(arr).min() > 0.0  # PD, not just PSD


def test_ledoit_wolf_pulls_off_diagonals_toward_zero():
    """Identity-target shrinkage shrinks every off-diagonal toward 0
    (|shrunk| <= |sample| holds elementwise; strict on noisy data)."""
    df = _noisy_frame()
    sample = correlation_matrix(df).to_numpy()
    shrunk = correlation_matrix(df, shrinkage='ledoit_wolf').to_numpy()
    off = ~np.eye(len(df.columns), dtype=bool)
    assert np.all(np.abs(shrunk[off]) < np.abs(sample[off]))


def test_ledoit_wolf_attrs_expose_intensity():
    """Fitted shrinkage coefficient is attached as attrs['lw_shrinkage']."""
    df = _noisy_frame()
    m = correlation_matrix(df, shrinkage='ledoit_wolf')
    delta = m.attrs['lw_shrinkage']
    assert 0.0 < delta <= 1.0
    from sklearn.covariance import LedoitWolf
    expected = LedoitWolf().fit(df.to_numpy(dtype=float)).shrinkage_
    assert np.isclose(delta, expected)


def test_ledoit_wolf_respects_lookback():
    """Shrinkage on lookback=N equals a direct fit on the last N rows."""
    df = _noisy_frame(n_rows=200)
    m = correlation_matrix(df, lookback=60, shrinkage='ledoit_wolf')
    np.testing.assert_allclose(
        m.to_numpy(), _sklearn_lw_corr(df.iloc[-60:]), atol=1e-12,
    )


def test_ledoit_wolf_uses_listwise_deletion_for_nans():
    """LW needs complete rows: NaN rows are dropped listwise (unlike the
    pairwise unshrunk path), so the result equals a fit on df.dropna()."""
    df = _noisy_frame()
    df.iloc[3, 0] = np.nan
    df.iloc[10, 2] = np.nan
    m = correlation_matrix(df, shrinkage='ledoit_wolf')
    np.testing.assert_allclose(
        m.to_numpy(), _sklearn_lw_corr(df.dropna()), atol=1e-12,
    )


def test_ledoit_wolf_too_few_complete_rows_raises():
    """Fewer than 2 complete rows after listwise deletion → ValueError."""
    df = _frame(
        a=np.array([1.0, np.nan, 2.0, np.nan]),
        b=np.array([np.nan, 1.0, 3.0, np.nan]),
    )
    with pytest.raises(ValueError, match="complete rows"):
        correlation_matrix(df, shrinkage='ledoit_wolf')


def test_ledoit_wolf_requires_pearson():
    """Shrinkage is covariance-based — only valid with method='pearson'."""
    df = _noisy_frame()
    for m in ('spearman', 'kendall'):
        with pytest.raises(ValueError, match="method='pearson'"):
            correlation_matrix(df, method=m, shrinkage='ledoit_wolf')


def test_rejects_unknown_shrinkage():
    df = _noisy_frame()
    with pytest.raises(ValueError, match="shrinkage must be"):
        correlation_matrix(df, shrinkage='bogus')
