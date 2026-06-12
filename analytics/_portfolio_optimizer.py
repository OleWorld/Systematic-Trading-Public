"""Portfolio-weight optimizers — equal weight, min-variance, ERC risk parity.

Label-agnostic, one-shot weight calculators. Each function returns
``Dict[str, float]`` keyed by the input labels (taken from
``corr_matrix.index``, or from ``labels`` for ``equal_weight``) — never a
bare positional array — so a weight can never be attached to the wrong
component by ordering mistakes. Labels are opaque strings: instrument
symbols, strategy names, and forecast-variation names all work — the
optimizer does not care what level of the allocation hierarchy it is
weighting, only that the scheme's constraint is met.

Shared contract
---------------
* **Named weights in, named weights out.** Vector inputs (``vols``) are
  accepted as ``dict`` / ``pd.Series`` and aligned to
  ``corr_matrix.index`` by label; input order is irrelevant.
* **Per-scheme constraint declaration.** Every scheme documents the
  constraint set its weights satisfy. All schemes here are
  *full-investment long-only*: ``w_i >= 0`` and ``sum(w) == 1``. Future
  schemes (e.g. a market-neutral weighting for cross-sectional
  strategies) will declare different constraints (``sum(w) == 0``,
  ``sum(|w|) == 1``) as new functions without touching existing ones.
* **Correlation-as-covariance equivalence.** The optimizers consume
  ``Σ = diag(σ) · ρ · diag(σ)``. Min-variance and ERC weights are
  invariant to positive scaling of ``Σ``, so passing ``vols=None``
  (``Σ = ρ``) is *identical* to optimizing the true covariance under
  equal volatilities. That equal-vol convention is the correct default
  inside the Carver vol-targeting stack, where position sizing already
  normalizes every instrument by its own σ.

This is a research/setup-time helper package (see ``analytics``
package docstring): call at portfolio build time or on a coarse
walk-forward cadence, not per bar.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ['equal_weight', 'min_variance', 'risk_parity']

_VolsLike = Union[Mapping[str, float], pd.Series]
_SYMMETRY_TOL = 1e-9
# Strictly-positive lower bound for the ERC solver: keeps the log
# barrier finite while being far below any economically meaningful
# weight.
_ERC_WEIGHT_FLOOR = 1e-12


def equal_weight(labels: Sequence[str]) -> Dict[str, float]:
    """Return ``{label: 1/N}`` over ``labels``.

    Constraint: full-investment long-only (``w_i >= 0``, ``sum(w) == 1``).

    Parameters
    ----------
    labels
        Component names — instrument symbols, strategy names, variation
        names. Must be non-empty and free of duplicates (a duplicate
        would silently collapse into one dict key, breaking the
        sum-to-1 constraint).

    Returns
    -------
    Dict[str, float]
        ``1/N`` per label.

    Raises
    ------
    ValueError
        If ``labels`` is empty or contains duplicates.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("labels must not be empty")
    duplicates = sorted(l for l, c in Counter(labels).items() if c > 1)
    if duplicates:
        raise ValueError(f"labels contain duplicates: {duplicates}")
    n = len(labels)
    return {label: 1.0 / n for label in labels}


def min_variance(
    corr_matrix: pd.DataFrame,
    vols: Optional[_VolsLike] = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 1_000,
) -> Dict[str, float]:
    """Long-only minimum-variance weights.

    Solves the constrained quadratic program

        min  wᵀ Σ w    s.t.    sum(w) = 1,  w >= 0

    numerically (``scipy.optimize.minimize(method='SLSQP')`` with the
    analytic jacobian ``2Σw``). This is the *exact* long-only optimum —
    unlike the closed form ``ρ⁻¹1 / 1ᵀρ⁻¹1``, which only solves the
    equality-constrained problem and must be patched with a
    clip-and-renormalize heuristic when it goes negative.

    Constraint: full-investment long-only (``w_i >= 0``, ``sum(w) == 1``).

    Parameters
    ----------
    corr_matrix
        N×N correlation matrix as a ``pd.DataFrame`` (square, symmetric
        within ``1e-9``, ``index == columns``, no NaN/inf). Output labels
        are taken from its index. Typically produced by
        ``analytics.correlation_matrix``.
    vols
        Optional per-label volatilities as ``dict`` or ``pd.Series``,
        keyed by the same labels as ``corr_matrix.index`` (label-aligned;
        order irrelevant). All values must be finite and ``> 0``. When
        given, the optimizer runs on the covariance
        ``Σ = ρ ∘ σσᵀ``; when ``None`` (default), ``Σ = ρ`` — the
        equal-volatility convention (see module docstring).
    tol
        Solver termination tolerance (forwarded to scipy). Must be > 0.
    max_iter
        Solver iteration cap. Must be >= 1.

    Returns
    -------
    Dict[str, float]
        Weights keyed by ``corr_matrix.index`` labels; non-negative,
        summing to 1.

    Raises
    ------
    TypeError
        If ``corr_matrix`` is not a ``pd.DataFrame`` or ``vols`` is not
        a mapping/Series.
    ValueError
        On invalid ``corr_matrix`` / ``vols`` / ``tol`` / ``max_iter``,
        or if the solver fails to converge.
    """
    _validate_solver_params(tol, max_iter)
    sigma = _build_sigma(corr_matrix, vols)
    n = sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: float(w @ sigma @ w),
        w0,
        jac=lambda w: 2.0 * (sigma @ w),
        method='SLSQP',
        bounds=[(0.0, 1.0)] * n,
        constraints=[{
            'type': 'eq',
            'fun': lambda w: w.sum() - 1.0,
            'jac': lambda w: np.ones(n),
        }],
        tol=tol,
        options={'maxiter': max_iter},
    )
    if not result.success:
        raise ValueError(f"min_variance solver failed: {result.message}")
    # Numerical hygiene only (magnitudes ~ machine eps): SLSQP satisfies
    # bounds/constraints to tolerance, so clip stray -1e-17s and rescale
    # to an exact sum of 1 for downstream sum-to-1 validators.
    w = np.clip(result.x, 0.0, None)
    w = w / w.sum()
    return {label: float(wi) for label, wi in zip(corr_matrix.index, w)}


def risk_parity(
    corr_matrix: pd.DataFrame,
    vols: Optional[_VolsLike] = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 1_000,
) -> Dict[str, float]:
    """Equal-risk-contribution (ERC) weights.

    Finds the weights at which every component contributes equally to
    portfolio volatility: ``RC_i = w_i (Σw)_i`` equal across ``i``. The
    ERC conditions are a coupled quadratic system with no closed form
    for heterogeneous correlations, so this solves Spinu's strictly
    convex reformulation

        min  ½ wᵀ Σ w − Σ_i ln(w_i)

    (``scipy.optimize.minimize(method='L-BFGS-B')`` with the analytic
    gradient ``Σw − 1/w``); the unnormalized optimum satisfies
    ``w_i (Σw)_i = 1`` for all ``i``, and rescaling to sum 1 preserves
    the ERC property. The log barrier makes the solution unique and
    strictly positive — no clipping needed by construction.

    Constraint: full-investment long-only (``w_i > 0``, ``sum(w) == 1``).

    Parameters
    ----------
    corr_matrix
        N×N correlation matrix as a ``pd.DataFrame`` (square, symmetric
        within ``1e-9``, ``index == columns``, no NaN/inf). Output labels
        are taken from its index.
    vols
        Optional per-label volatilities, semantics identical to
        ``min_variance``: when given, ``Σ = ρ ∘ σσᵀ``; when ``None``
        (default), ``Σ = ρ`` — the equal-vol convention, under which
        ERC answers "what weights equalize risk contributions *given
        correlations alone*". With uniform correlations this collapses
        to the inverse-volatility closed form ``w_i ∝ 1/σ_i``.
    tol
        Solver termination tolerance (forwarded to scipy). Must be > 0.
    max_iter
        Solver iteration cap. Must be >= 1.

    Returns
    -------
    Dict[str, float]
        Weights keyed by ``corr_matrix.index`` labels; strictly
        positive, summing to 1.

    Raises
    ------
    TypeError
        If ``corr_matrix`` is not a ``pd.DataFrame`` or ``vols`` is not
        a mapping/Series.
    ValueError
        On invalid ``corr_matrix`` / ``vols`` / ``tol`` / ``max_iter``,
        or if the solver fails to converge.
    """
    _validate_solver_params(tol, max_iter)
    sigma = _build_sigma(corr_matrix, vols)
    n = sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: float(0.5 * (w @ sigma @ w) - np.log(w).sum()),
        w0,
        jac=lambda w: sigma @ w - 1.0 / w,
        method='L-BFGS-B',
        bounds=[(_ERC_WEIGHT_FLOOR, None)] * n,
        tol=tol,
        options={'maxiter': max_iter},
    )
    if not result.success:
        raise ValueError(f"risk_parity solver failed: {result.message}")
    w = result.x / result.x.sum()
    return {label: float(wi) for label, wi in zip(corr_matrix.index, w)}


def _validate_solver_params(tol: float, max_iter: int) -> None:
    """Raise ``ValueError`` on non-positive ``tol`` or ``max_iter < 1``."""
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")


def _validate_corr_matrix(corr_matrix: pd.DataFrame) -> np.ndarray:
    """Validate ``corr_matrix`` and return it as a float ndarray.

    Checks: is a ``pd.DataFrame``; square; ``index == columns`` (labels
    and order); all entries finite (NaN/inf would otherwise flow
    silently into the solver); symmetric within ``1e-9``. Raises
    ``TypeError`` / ``ValueError`` accordingly.
    """
    if not isinstance(corr_matrix, pd.DataFrame):
        raise TypeError(
            f"corr_matrix must be a pd.DataFrame, got {type(corr_matrix).__name__}"
        )
    n_rows, n_cols = corr_matrix.shape
    if n_rows != n_cols:
        raise ValueError(
            f"corr_matrix must be square, got shape {corr_matrix.shape}"
        )
    if not corr_matrix.index.equals(corr_matrix.columns):
        raise ValueError(
            "corr_matrix index and columns must match (labels and order)"
        )
    rho = corr_matrix.to_numpy(dtype=float)
    if not np.isfinite(rho).all():
        raise ValueError("corr_matrix must not contain NaN or inf entries")
    if not np.allclose(rho, rho.T, atol=_SYMMETRY_TOL):
        raise ValueError("corr_matrix must be symmetric within 1e-9")
    return rho


def _build_sigma(
    corr_matrix: pd.DataFrame,
    vols: Optional[_VolsLike],
) -> np.ndarray:
    """Build the matrix the optimizers consume: ``ρ`` or ``diag(σ)·ρ·diag(σ)``.

    Validates ``corr_matrix`` (see ``_validate_corr_matrix``). When
    ``vols`` is given, aligns it to ``corr_matrix.index`` by label
    (keys must be set-equal to the index; order irrelevant) and requires
    every value finite and ``> 0``. Raises ``TypeError`` /
    ``ValueError`` accordingly.
    """
    rho = _validate_corr_matrix(corr_matrix)
    if vols is None:
        return rho

    if isinstance(vols, pd.Series):
        vols_dict = vols.to_dict()
    elif isinstance(vols, Mapping):
        vols_dict = dict(vols)
    else:
        raise TypeError(
            f"vols must be a mapping or pd.Series, got {type(vols).__name__}"
        )
    expected_keys = set(corr_matrix.index)
    got_keys = set(vols_dict.keys())
    if got_keys != expected_keys:
        missing = expected_keys - got_keys
        extra = got_keys - expected_keys
        raise ValueError(
            f"vols keys must equal corr_matrix.index as a set; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    sig = np.array(
        [float(vols_dict[label]) for label in corr_matrix.index], dtype=float,
    )
    if not np.isfinite(sig).all() or np.any(sig <= 0):
        raise ValueError(
            f"vols must be finite and > 0, got {vols_dict}"
        )
    return rho * np.outer(sig, sig)
