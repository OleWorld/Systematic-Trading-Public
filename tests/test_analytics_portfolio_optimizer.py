"""Unit tests for ``analytics`` portfolio-weight optimizers.

Covers ``equal_weight``, ``min_variance`` (long-only QP), and
``risk_parity`` (ERC). Solver outputs are exact only to tolerance, so
golden comparisons use ``abs_tol`` ≈ 1e-8 rather than machine precision.

Run from the repo root:  python -m pytest tests/test_analytics_portfolio_optimizer.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from analytics import (
    diversification_multiplier, equal_weight, min_variance, risk_parity,
)


def _corr(labels, off_diag=0.0):
    """Build an N x N correlation matrix labelled by ``labels`` with a uniform
    off-diagonal value (diagonal is always 1.0)."""
    n = len(labels)
    m = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


def _random_pd_corr(labels, seed):
    """Sample correlation matrix of one-factor data — heterogeneous
    off-diagonals, positive-definite almost surely."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    rows = 50 * n
    loadings = rng.uniform(0.2, 1.0, size=n)
    common = rng.normal(size=(rows, 1))
    data = common * loadings + rng.normal(size=(rows, n))
    m = np.corrcoef(data, rowvar=False)
    return pd.DataFrame(m, index=labels, columns=labels)


def _risk_contributions(weights, corr_matrix, vols=None):
    """RC_i = w_i (Σw)_i, label-aligned to ``corr_matrix.index``."""
    labels = list(corr_matrix.index)
    sigma = corr_matrix.to_numpy(dtype=float)
    if vols is not None:
        sig = np.array([vols[label] for label in labels], dtype=float)
        sigma = sigma * np.outer(sig, sig)
    w = np.array([weights[label] for label in labels], dtype=float)
    return w * (sigma @ w)


# ──────────────────────────────────────────────
# equal_weight
# ──────────────────────────────────────────────

def test_equal_weight_two_labels():
    assert equal_weight(['a', 'b']) == {'a': 0.5, 'b': 0.5}


def test_equal_weight_three_labels():
    w = equal_weight(['a', 'b', 'c'])
    for label in ('a', 'b', 'c'):
        assert math.isclose(w[label], 1.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_equal_weight_single_label():
    assert equal_weight(['solo']) == {'solo': 1.0}


def test_equal_weight_empty_labels_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        equal_weight([])


def test_equal_weight_duplicate_labels_raises():
    with pytest.raises(ValueError, match="duplicates"):
        equal_weight(['a', 'b', 'a'])


# ──────────────────────────────────────────────
# min_variance — solutions
# ──────────────────────────────────────────────

def test_min_variance_two_uncorrelated_yields_equal_weights():
    w = min_variance(_corr(['a', 'b']))
    assert math.isclose(w['a'], 0.5, abs_tol=1e-8)
    assert math.isclose(w['b'], 0.5, abs_tol=1e-8)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_min_variance_downweights_correlated_pair_against_uncorrelated_solo():
    """A and B correlated 0.6, C uncorrelated with both → C gets the
    largest weight; A and B split the rest symmetrically."""
    rho = pd.DataFrame(
        [[1.0, 0.6, 0.0],
         [0.6, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    w = min_variance(rho)
    assert w['C'] > w['A']
    assert w['C'] > w['B']
    assert math.isclose(w['A'], w['B'], abs_tol=1e-8)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_min_variance_interior_solution_matches_closed_form():
    """When no bound is active (closed-form weights all positive), the QP
    optimum equals the equality-constrained closed form ρ⁻¹1 / 1ᵀρ⁻¹1."""
    rho = pd.DataFrame(
        [[1.0, 0.5, 0.2],
         [0.5, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    inv = np.linalg.inv(rho.to_numpy())
    closed_form = inv.sum(axis=1) / inv.sum()
    assert (closed_form > 0).all()      # interior — closed form is the optimum
    w = min_variance(rho)
    for label, expected in zip(rho.index, closed_form):
        assert math.isclose(w[label], expected, abs_tol=1e-6)


def test_min_variance_bound_activation_zeroes_diversification_drag():
    """A is 0.7-correlated with both B and C (which are only mildly
    correlated with each other): A's unconstrained weight is negative
    (raw closed form = (-0.2, 0.6, 0.6)), so the long-only optimum pins
    w_A to the bound and re-optimizes B/C → 0.5 each (KKT-verified)."""
    rho = pd.DataFrame(
        [[1.0, 0.7, 0.7],
         [0.7, 1.0, 0.3],
         [0.7, 0.3, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    inv = np.linalg.inv(rho.to_numpy())
    raw = inv.sum(axis=1) / inv.sum()
    assert raw[0] < 0                   # confirms the bound must activate
    w = min_variance(rho)
    assert abs(w['A']) < 1e-6
    assert math.isclose(w['B'], 0.5, abs_tol=1e-6)
    assert math.isclose(w['C'], 0.5, abs_tol=1e-6)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_min_variance_bound_weights_snap_to_exact_zero():
    """Interior-point solvers approach an active bound asymptotically
    (~1e-12 dust, never exact 0.0). Sub-threshold weights must come back
    as EXACT zeros so downstream zero-weight semantics fire (the RM's
    skip_reason='zero_weight' short-circuit keys on == 0, and dust
    weights otherwise produce absurd 1e-12-contract orders)."""
    rho = pd.DataFrame(
        [[1.0, 0.7, 0.7],
         [0.7, 1.0, 0.3],
         [0.7, 0.3, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    w = min_variance(rho)
    assert w['A'] == 0.0    # exact — not just close
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_min_variance_with_vols_matches_two_asset_closed_form():
    """Two-asset covariance min-var closed form:
    w1 = (σ2² − ρσ1σ2) / (σ1² + σ2² − 2ρσ1σ2)."""
    rho_val, s1, s2 = 0.3, 1.0, 2.0
    rho = _corr(['a', 'b'], off_diag=rho_val)
    vols = {'a': s1, 'b': s2}
    expected_a = (s2**2 - rho_val * s1 * s2) / (
        s1**2 + s2**2 - 2 * rho_val * s1 * s2
    )
    w = min_variance(rho, vols)
    assert math.isclose(w['a'], expected_a, abs_tol=1e-8)
    assert math.isclose(w['b'], 1.0 - expected_a, abs_tol=1e-8)


def test_min_variance_vols_as_dict_and_series_identical():
    rho = _corr(['a', 'b', 'c'], off_diag=0.4)
    vols = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    w_dict = min_variance(rho, vols)
    w_series = min_variance(rho, pd.Series(vols))
    for label in vols:
        assert math.isclose(w_dict[label], w_series[label], rel_tol=1e-12)


def test_min_variance_label_alignment_independent_of_matrix_order():
    """The same matrix presented in scrambled label order produces the
    same label→weight mapping (proves label-based, not positional)."""
    rho = pd.DataFrame(
        [[1.0, 0.5, 0.2],
         [0.5, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    scrambled = rho.loc[['C', 'A', 'B'], ['C', 'A', 'B']]
    w_ordered = min_variance(rho)
    w_scrambled = min_variance(scrambled)
    for label in ('A', 'B', 'C'):
        assert math.isclose(w_ordered[label], w_scrambled[label], abs_tol=1e-8)


def test_min_variance_output_feeds_diversification_multiplier():
    """Non-negative, sum-to-1 output passes the DM validators directly."""
    rho = _random_pd_corr(['A', 'B', 'C', 'D'], seed=7)
    w = min_variance(rho)
    assert diversification_multiplier(w, rho) >= 1.0


# ──────────────────────────────────────────────
# min_variance — validation
# ──────────────────────────────────────────────

def test_min_variance_non_dataframe_raises_typeerror():
    with pytest.raises(TypeError, match="pd.DataFrame"):
        min_variance(np.eye(2))


def test_min_variance_non_square_raises():
    bad = pd.DataFrame([[1.0, 0.0]], index=['a'], columns=['a', 'b'])
    with pytest.raises(ValueError, match="square"):
        min_variance(bad)


def test_min_variance_mismatched_index_columns_raises():
    bad = pd.DataFrame(np.eye(2), index=['a', 'b'], columns=['b', 'a'])
    with pytest.raises(ValueError, match="index and columns"):
        min_variance(bad)


def test_min_variance_nan_entry_raises():
    bad = _corr(['a', 'b'])
    bad.iloc[0, 1] = np.nan
    bad.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        min_variance(bad)


def test_min_variance_asymmetric_raises():
    bad = pd.DataFrame(
        [[1.0, 0.5], [0.1, 1.0]], index=['a', 'b'], columns=['a', 'b'],
    )
    with pytest.raises(ValueError, match="symmetric"):
        min_variance(bad)


def test_min_variance_vols_key_mismatch_raises():
    rho = _corr(['a', 'b'])
    with pytest.raises(ValueError, match="missing=\\['b'\\], extra=\\['z'\\]"):
        min_variance(rho, {'a': 1.0, 'z': 2.0})


def test_min_variance_non_positive_vol_raises():
    rho = _corr(['a', 'b'])
    with pytest.raises(ValueError, match="finite and > 0"):
        min_variance(rho, {'a': 1.0, 'b': 0.0})


def test_min_variance_vols_wrong_type_raises():
    rho = _corr(['a', 'b'])
    with pytest.raises(TypeError, match="mapping or pd.Series"):
        min_variance(rho, [1.0, 2.0])


def test_min_variance_bad_solver_params_raise():
    rho = _corr(['a', 'b'])
    with pytest.raises(ValueError, match="tol must be > 0"):
        min_variance(rho, tol=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        min_variance(rho, max_iter=0)


@pytest.mark.filterwarnings("ignore::UserWarning")  # cvxpy warns pre-raise
def test_min_variance_solver_failure_raises():
    """An iteration budget too small to converge surfaces as ValueError.

    (The pre-CVXPY fixture here was a materially non-PSD matrix that
    merely stalled SLSQP; that input is now rejected up front by the PSD
    validation, so failure is injected on a *valid* matrix instead.)
    """
    rho = _random_pd_corr(['A', 'B', 'C', 'D', 'E'], seed=5)
    with pytest.raises(ValueError, match="solver failed"):
        min_variance(rho, max_iter=1)


# ──────────────────────────────────────────────
# risk_parity — solutions
# ──────────────────────────────────────────────

def test_risk_parity_identity_corr_yields_equal_weights():
    for labels in (['a', 'b'], ['a', 'b', 'c']):
        w = risk_parity(_corr(labels))
        for label in labels:
            assert math.isclose(w[label], 1.0 / len(labels), abs_tol=1e-8)
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_risk_parity_uniform_corr_with_vols_matches_inverse_vol():
    """Uniform pairwise correlation is ERC's closed-form special case:
    w_i ∝ 1/σ_i."""
    rho = _corr(['a', 'b', 'c'], off_diag=0.5)
    vols = {'a': 1.0, 'b': 2.0, 'c': 4.0}
    inv_vol_sum = sum(1.0 / s for s in vols.values())
    w = risk_parity(rho, vols)
    for label, s in vols.items():
        assert math.isclose(w[label], (1.0 / s) / inv_vol_sum, abs_tol=1e-6)


def test_risk_parity_equalizes_risk_contributions_corr_only():
    """The defining ERC property: RC_i = w_i (Σw)_i equal across labels,
    on heterogeneous positive-definite correlation matrices."""
    for seed in (1, 2, 3):
        rho = _random_pd_corr(['A', 'B', 'C', 'D', 'E'], seed=seed)
        w = risk_parity(rho)
        rc = _risk_contributions(w, rho)
        assert rc.max() - rc.min() < 1e-6 * rc.mean()
        assert all(wi > 0 for wi in w.values())
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_risk_parity_equalizes_risk_contributions_with_vols():
    rho = _random_pd_corr(['A', 'B', 'C', 'D'], seed=11)
    vols = {'A': 0.5, 'B': 1.0, 'C': 2.0, 'D': 5.0}
    w = risk_parity(rho, vols)
    rc = _risk_contributions(w, rho, vols)
    assert rc.max() - rc.min() < 1e-6 * rc.mean()


def test_risk_parity_label_alignment_independent_of_matrix_order():
    rho = _random_pd_corr(['A', 'B', 'C'], seed=23)
    scrambled = rho.loc[['C', 'A', 'B'], ['C', 'A', 'B']]
    w_ordered = risk_parity(rho)
    w_scrambled = risk_parity(scrambled)
    for label in ('A', 'B', 'C'):
        assert math.isclose(w_ordered[label], w_scrambled[label], abs_tol=1e-8)


def test_risk_parity_output_feeds_diversification_multiplier():
    rho = _random_pd_corr(['A', 'B', 'C', 'D'], seed=7)
    w = risk_parity(rho)
    assert diversification_multiplier(w, rho) >= 1.0


# ──────────────────────────────────────────────
# risk_parity — validation
# ──────────────────────────────────────────────

def test_risk_parity_shares_matrix_and_vols_validation():
    """Spot-check that risk_parity runs the same validators."""
    with pytest.raises(TypeError, match="pd.DataFrame"):
        risk_parity(np.eye(2))
    bad = pd.DataFrame(
        [[1.0, 0.5], [0.1, 1.0]], index=['a', 'b'], columns=['a', 'b'],
    )
    with pytest.raises(ValueError, match="symmetric"):
        risk_parity(bad)
    with pytest.raises(ValueError, match="missing="):
        risk_parity(_corr(['a', 'b']), {'a': 1.0})


def test_risk_parity_bad_solver_params_raise():
    rho = _corr(['a', 'b'])
    with pytest.raises(ValueError, match="tol must be > 0"):
        risk_parity(rho, tol=-1.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        risk_parity(rho, max_iter=0)


@pytest.mark.filterwarnings("ignore::UserWarning")  # cvxpy warns pre-raise
def test_risk_parity_solver_failure_raises():
    rho = _random_pd_corr(['A', 'B', 'C', 'D', 'E'], seed=3)
    with pytest.raises(ValueError, match="solver failed"):
        risk_parity(rho, max_iter=1)


# ──────────────────────────────────────────────
# PSD validation / repair (shared via _build_sigma)
# ──────────────────────────────────────────────

def test_min_variance_repairs_eps_non_psd_matrix():
    """λ_min barely below zero (numerical dust, e.g. a floored matrix) is
    repaired by eigenvalue clipping, not rejected. Uniform off-diag just
    under -1/2 on N=3 puts λ_min ≈ -4e-11."""
    rho = _corr(['a', 'b', 'c'], off_diag=-0.5 - 2e-11)
    assert np.linalg.eigvalsh(rho.to_numpy()).min() < 0  # genuinely non-PSD
    w = min_variance(rho)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    for label in ('a', 'b', 'c'):
        assert w[label] >= 0
        assert math.isclose(w[label], 1.0 / 3.0, abs_tol=1e-6)  # symmetry


def test_min_variance_materially_non_psd_raises():
    """λ_min far below tolerance (-0.2 here) is a broken input, not dust —
    it would silently define a non-convex QP. Must raise."""
    rho = _corr(['a', 'b', 'c'], off_diag=-0.6)  # λ_min = 1 - 1.2 = -0.2
    with pytest.raises(ValueError, match="non-PSD"):
        min_variance(rho)


def test_risk_parity_materially_non_psd_raises():
    """risk_parity shares the PSD validation via _build_sigma."""
    rho = _corr(['a', 'b', 'c'], off_diag=-0.6)
    with pytest.raises(ValueError, match="non-PSD"):
        risk_parity(rho)


# ──────────────────────────────────────────────
# Near-singular robustness (high-dimensional regime)
# ──────────────────────────────────────────────

def _near_singular_sample_corr(n_assets=100, n_obs=61, seed=42):
    """Sample correlation of more assets than observations — rank-deficient
    (rank <= n_obs - 1), the regime that motivated the CVXPY migration."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_obs, 1))
    data = common * rng.uniform(0.2, 1.0, size=n_assets) \
        + rng.normal(size=(n_obs, n_assets))
    labels = [f's{i}' for i in range(n_assets)]
    m = np.corrcoef(data, rowvar=False)
    return pd.DataFrame(m, index=labels, columns=labels)


def test_min_variance_solves_near_singular_100x100():
    rho = _near_singular_sample_corr()
    w = min_variance(rho)
    arr = np.array(list(w.values()))
    assert math.isclose(arr.sum(), 1.0, abs_tol=1e-9)
    assert np.all(arr >= 0)


def test_risk_parity_solves_near_singular_100x100():
    rho = _near_singular_sample_corr()
    w = risk_parity(rho)
    arr = np.array(list(w.values()))
    assert math.isclose(arr.sum(), 1.0, abs_tol=1e-9)
    assert np.all(arr > 0)
