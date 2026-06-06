"""Unit tests for ``analytics.diversification_multiplier``.

Run from the repo root:  python -m pytest tests/test_analytics_diversification_multiplier.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from analytics import diversification_multiplier


def _corr(labels, off_diag=0.0):
    """Build an N x N correlation matrix labelled by ``labels`` with a uniform
    off-diagonal value (diagonal is always 1.0)."""
    n = len(labels)
    m = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


# ──────────────────────────────────────────────
# Core formula cases
# ──────────────────────────────────────────────

def test_two_asset_uncorrelated_equal_weights_returns_sqrt_two():
    """ρ=0, equal weights → wᵀρw = 0.5, DM = sqrt(2)."""
    corr = _corr(['a', 'b'], off_diag=0.0)
    weights = {'a': 0.5, 'b': 0.5}
    assert math.isclose(
        diversification_multiplier(weights, corr), math.sqrt(2.0), rel_tol=1e-12
    )


def test_two_asset_perfectly_correlated_equal_weights_returns_one():
    """ρ=1, any positive weights → wᵀρw = 1.0, DM = 1.0 (no diversification)."""
    corr = _corr(['a', 'b'], off_diag=1.0)
    weights = {'a': 0.5, 'b': 0.5}
    assert math.isclose(
        diversification_multiplier(weights, corr), 1.0, rel_tol=1e-12
    )


def test_three_asset_uncorrelated_equal_weights_returns_sqrt_three():
    """ρ=0 across the board, equal 1/3 weights → DM = sqrt(3)."""
    corr = _corr(['a', 'b', 'c'], off_diag=0.0)
    weights = {'a': 1 / 3, 'b': 1 / 3, 'c': 1 / 3}
    assert math.isclose(
        diversification_multiplier(weights, corr), math.sqrt(3.0), rel_tol=1e-12
    )


def test_single_asset_returns_one():
    """N=1 → DM = 1 (a single instrument cannot diversify itself)."""
    corr = pd.DataFrame([[1.0]], index=['a'], columns=['a'])
    assert math.isclose(
        diversification_multiplier({'a': 1.0}, corr), 1.0, rel_tol=1e-12
    )


def test_dict_and_series_inputs_produce_identical_result():
    """Same weights as dict vs pd.Series → same DM."""
    corr = _corr(['a', 'b', 'c'], off_diag=0.2)
    as_dict = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    as_series = pd.Series(as_dict)
    assert math.isclose(
        diversification_multiplier(as_dict, corr),
        diversification_multiplier(as_series, corr),
        rel_tol=1e-15,
    )


def test_label_alignment_independent_of_input_order():
    """Scrambled weight order vs corr_matrix.index order → same DM (proves
    label-based, not positional, alignment)."""
    corr = _corr(['a', 'b', 'c'], off_diag=0.3)
    ordered = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    scrambled = {'c': 0.2, 'a': 0.5, 'b': 0.3}
    assert math.isclose(
        diversification_multiplier(ordered, corr),
        diversification_multiplier(scrambled, corr),
        rel_tol=1e-15,
    )


def test_partial_correlation_lies_between_one_and_sqrt_n():
    """For a non-zero, non-one correlation, DM ∈ (1, sqrt(N))."""
    corr = _corr(['a', 'b', 'c'], off_diag=0.5)
    dm = diversification_multiplier({'a': 1 / 3, 'b': 1 / 3, 'c': 1 / 3}, corr)
    assert 1.0 < dm < math.sqrt(3.0)


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def test_rejects_non_dataframe_corr_matrix():
    with pytest.raises(TypeError, match="corr_matrix must be a pd.DataFrame"):
        diversification_multiplier({'a': 0.5, 'b': 0.5}, [[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]


def test_rejects_non_square_corr_matrix():
    bad = pd.DataFrame(np.zeros((2, 3)), index=['a', 'b'], columns=['x', 'y', 'z'])
    with pytest.raises(ValueError, match="square"):
        diversification_multiplier({'a': 0.5, 'b': 0.5}, bad)


def test_rejects_asymmetric_corr_matrix():
    bad = pd.DataFrame(
        [[1.0, 0.2], [0.5, 1.0]], index=['a', 'b'], columns=['a', 'b']
    )
    with pytest.raises(ValueError, match="symmetric"):
        diversification_multiplier({'a': 0.5, 'b': 0.5}, bad)


def test_rejects_weights_missing_a_symbol():
    corr = _corr(['a', 'b', 'c'], off_diag=0.0)
    with pytest.raises(ValueError, match="weights"):
        diversification_multiplier({'a': 0.6, 'b': 0.4}, corr)


def test_rejects_weights_with_extra_symbol():
    corr = _corr(['a', 'b'], off_diag=0.0)
    with pytest.raises(ValueError, match="weights"):
        diversification_multiplier({'a': 0.3, 'b': 0.3, 'c': 0.4}, corr)


def test_rejects_negative_weight():
    corr = _corr(['a', 'b'], off_diag=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        diversification_multiplier({'a': 1.2, 'b': -0.2}, corr)


def test_rejects_weights_summing_below_one():
    corr = _corr(['a', 'b'], off_diag=0.0)
    with pytest.raises(ValueError, match="sum"):
        diversification_multiplier({'a': 0.4, 'b': 0.5}, corr)


def test_rejects_weights_summing_above_one():
    corr = _corr(['a', 'b'], off_diag=0.0)
    with pytest.raises(ValueError, match="sum"):
        diversification_multiplier({'a': 0.6, 'b': 0.5}, corr)
