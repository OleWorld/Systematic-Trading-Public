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
* **CVXPY/CLARABEL solver stack.** Both corr-based schemes are convex
  programs solved through CVXPY with the CLARABEL interior-point solver
  (QP for min-variance; exponential cone for the ERC log barrier).
  Chosen for robustness on near-singular matrices — the high-dimensional
  regime (N approaching or exceeding the estimation window) where
  generic NLP solvers degrade — and for expressiveness (future
  constraint sets are declarative one-liners), not for speed: at this
  problem size canonicalization overhead dominates, and solves run at a
  coarse walk-forward cadence where per-solve cost is irrelevant.
* **PSD contract.** ``Σ`` must be positive semidefinite — the convexity
  certificate is global, not feasible-region-local. Numerical dust
  (``λ_min >= -1e-8``, e.g. from element-wise correlation flooring) is
  repaired by eigenvalue clipping; materially non-PSD input raises
  ``ValueError`` instead of silently defining a non-convex program.

This is a research/setup-time helper package (see ``analytics``
package docstring): call at portfolio build time or on a coarse
walk-forward cadence, not per bar.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Optional, Sequence, Union

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = ['equal_weight', 'min_variance', 'risk_parity']

_VolsLike = Union[Mapping[str, float], pd.Series]
_SYMMETRY_TOL = 1e-9
# PSD tolerance: eigenvalues in [-_PSD_TOL, 0) are numerical dust
# (clipped to zero); anything below is a materially broken input.
_PSD_TOL = 1e-8
# Weights below this are interior-point dust at an active w >= 0 bound
# (the solver approaches the bound asymptotically, never reaching exact
# 0.0) — far below any economic weight. Snapped to exact zero so
# downstream zero-weight semantics (e.g. the risk manager's
# skip_reason='zero_weight' short-circuit) fire instead of producing
# 1e-12-contract orders.
_ZERO_SNAP = 1e-10


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

    via CVXPY/CLARABEL. This is the *exact* long-only optimum — unlike
    the closed form ``ρ⁻¹1 / 1ᵀρ⁻¹1``, which only solves the
    equality-constrained problem and must be patched with a
    clip-and-renormalize heuristic when it goes negative.

    Constraint: full-investment long-only (``w_i >= 0``, ``sum(w) == 1``).

    Parameters
    ----------
    corr_matrix
        N×N correlation matrix as a ``pd.DataFrame`` (square, symmetric
        within ``1e-9``, ``index == columns``, no NaN/inf, PSD within
        the ``1e-8`` dust tolerance — see module docstring). Output
        labels are taken from its index. Typically produced by
        ``analytics.correlation_matrix``.
    vols
        Optional per-label volatilities as ``dict`` or ``pd.Series``,
        keyed by the same labels as ``corr_matrix.index`` (label-aligned;
        order irrelevant). All values must be finite and ``> 0``. When
        given, the optimizer runs on the covariance
        ``Σ = ρ ∘ σσᵀ``; when ``None`` (default), ``Σ = ρ`` — the
        equal-volatility convention (see module docstring).
    tol
        Solver convergence tolerance, forwarded to CLARABEL as
        ``tol_gap_abs`` / ``tol_gap_rel`` / ``tol_feas``. Must be > 0.
        Default ``1e-12`` (CLARABEL reaches it cleanly at these problem
        sizes, giving weight precision well inside the documented
        ≈1e-6/1e-9 test tolerances).
    max_iter
        Solver iteration cap, forwarded to CLARABEL. Must be >= 1.

    Returns
    -------
    Dict[str, float]
        Weights keyed by ``corr_matrix.index`` labels; non-negative,
        summing to 1. Weights at the long-only bound are **exact**
        zeros: sub-``1e-10`` interior-point dust is snapped to ``0.0``
        before renormalization.

    Raises
    ------
    TypeError
        If ``corr_matrix`` is not a ``pd.DataFrame`` or ``vols`` is not
        a mapping/Series.
    ValueError
        On invalid ``corr_matrix`` / ``vols`` / ``tol`` / ``max_iter``,
        a materially non-PSD matrix, or if the solver fails to reach an
        OPTIMAL status.
    """
    _validate_solver_params(tol, max_iter)
    sigma = _build_sigma(corr_matrix, vols)
    n = sigma.shape[0]
    w = cp.Variable(n)
    problem = cp.Problem(
        # psd_wrap skips CVXPY's own eigen-check; _build_sigma has
        # already validated/repaired PSD-ness.
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))),
        [cp.sum(w) == 1, w >= 0],  # type: ignore[list-item]  # cvxpy __eq__ stub
    )
    _solve(problem, 'min_variance', tol, max_iter)
    # Numerical hygiene (magnitudes ~ solver tolerance): clip stray
    # -1e-12s, snap bound-dust to exact zeros (see _ZERO_SNAP), and
    # rescale to an exact sum of 1 for downstream sum-to-1 validators.
    w_opt = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
    w_opt[w_opt < _ZERO_SNAP] = 0.0
    w_opt = w_opt / w_opt.sum()
    return {label: float(wi) for label, wi in zip(corr_matrix.index, w_opt)}


def risk_parity(
    corr_matrix: pd.DataFrame,
    vols: Optional[_VolsLike] = None,
    *,
    tol: float = 1e-9,
    max_iter: int = 1_000,
) -> Dict[str, float]:
    """Equal-risk-contribution (ERC) weights.

    Finds the weights at which every component contributes equally to
    portfolio volatility: ``RC_i = w_i (Σw)_i`` equal across ``i``. The
    ERC conditions are a coupled quadratic system with no closed form
    for heterogeneous correlations, so this solves Spinu's strictly
    convex reformulation

        min  ½ wᵀ Σ w − Σ_i ln(w_i)

    via CVXPY/CLARABEL (the log barrier enters as an exponential cone);
    the unnormalized optimum satisfies ``w_i (Σw)_i = 1`` for all ``i``,
    and rescaling to sum 1 preserves the ERC property. The log barrier
    makes the solution unique and strictly positive — no explicit bounds
    needed, the log domain enforces ``w > 0``. The interior-point
    solution is then refined to ~machine precision by a short Newton
    polish on the first-order system ``Σw − 1/w = 0`` (exp-cone solves
    deliver ~tolerance-level weight accuracy; the polish closes the gap
    so outputs meet the documented ≈1e-6/1e-9 precision).

    Constraint: full-investment long-only (``w_i > 0``, ``sum(w) == 1``).

    Parameters
    ----------
    corr_matrix
        N×N correlation matrix as a ``pd.DataFrame`` (square, symmetric
        within ``1e-9``, ``index == columns``, no NaN/inf, PSD within
        the ``1e-8`` dust tolerance — see module docstring). Output
        labels are taken from its index.
    vols
        Optional per-label volatilities, semantics identical to
        ``min_variance``: when given, ``Σ = ρ ∘ σσᵀ``; when ``None``
        (default), ``Σ = ρ`` — the equal-vol convention, under which
        ERC answers "what weights equalize risk contributions *given
        correlations alone*". With uniform correlations this collapses
        to the inverse-volatility closed form ``w_i ∝ 1/σ_i``.
    tol
        Solver convergence tolerance, forwarded to CLARABEL as
        ``tol_gap_abs`` / ``tol_gap_rel`` / ``tol_feas``. Must be > 0.
        Default ``1e-9`` — the CVXPY stage only needs to land inside the
        Newton-polish basin; pushing exp-cone solves to 1e-12 trips
        CLARABEL's reduced-accuracy fallback on some instances.
    max_iter
        Solver iteration cap, forwarded to CLARABEL. Must be >= 1.

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
        a materially non-PSD matrix, or if the solver fails to reach an
        OPTIMAL status.
    """
    _validate_solver_params(tol, max_iter)
    sigma = _build_sigma(corr_matrix, vols)
    n = sigma.shape[0]
    w = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(
            0.5 * cp.quad_form(w, cp.psd_wrap(sigma)) - cp.sum(cp.log(w))
        ),
    )
    _solve(problem, 'risk_parity', tol, max_iter)
    w_opt = np.asarray(w.value, dtype=float)
    w_opt = _newton_polish_erc(sigma, w_opt)
    w_opt = w_opt / w_opt.sum()
    return {label: float(wi) for label, wi in zip(corr_matrix.index, w_opt)}


def _newton_polish_erc(sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Refine a near-optimal ERC solution to ~machine precision.

    Newton iterations on Spinu's first-order system ``F(w) = Σw − 1/w =
    0``. The optimum is unique and strictly positive, and the Hessian
    ``Σ + diag(1/w²)`` is SPD for any ``w > 0`` even when ``Σ`` itself
    is singular, so every step is well-defined. From an interior-point
    start (accurate to ~solver tolerance) convergence is quadratic —
    two or three steps reach the ``1e-12`` residual exit. Step-halving
    keeps iterates strictly positive; the iteration cap (10) only binds
    on inputs where Newton cannot improve further, in which case the
    best iterate so far is returned (never worse than the solver's).
    """
    # Solver output is strictly positive (log domain); the floor only
    # guards against representation-level dust on the way in.
    w = np.maximum(w, 1e-12)
    for _ in range(10):
        grad = sigma @ w - 1.0 / w
        if float(np.max(np.abs(grad))) < 1e-12:
            break
        hess = sigma + np.diag(1.0 / np.square(w))
        step = np.linalg.solve(hess, -grad)
        t = 1.0
        while np.any(w + t * step <= 0.0):
            t *= 0.5
        w = w + t * step
    return w


def _solve(
    problem: cp.Problem, name: str, tol: float, max_iter: int,
) -> None:
    """Solve ``problem`` with CLARABEL; ``ValueError`` on anything non-OPTIMAL.

    ``cp.SolverError`` (canonicalization or solver-level failure,
    including iteration-limit exhaustion) and any terminal status other
    than ``OPTIMAL`` — including ``OPTIMAL_INACCURATE`` — surface as
    ``ValueError(f"{name} solver failed: ...")``, preserving the strict
    failure contract callers rely on.
    """
    try:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=int(max_iter),
            tol_gap_abs=tol,
            tol_gap_rel=tol,
            tol_feas=tol,
        )
    except cp.SolverError as exc:
        raise ValueError(f"{name} solver failed: {exc}") from exc
    if problem.status != cp.OPTIMAL:
        raise ValueError(
            f"{name} solver failed: status {problem.status!r}"
        )


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
    ``TypeError`` / ``ValueError`` accordingly. PSD-ness is checked
    separately in ``_ensure_psd`` (on the final Σ, after the optional
    vols scaling).
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


def _ensure_psd(sigma: np.ndarray) -> np.ndarray:
    """Return a PSD version of ``sigma`` or raise on material violation.

    ``λ_min >= 0`` → returned unchanged. ``-1e-8 <= λ_min < 0`` →
    numerical dust (e.g. an element-wise-floored correlation matrix):
    negative eigenvalues are clipped to zero, the matrix reconstructed
    and re-symmetrized. ``λ_min < -1e-8`` → ``ValueError`` — the input
    is materially non-PSD and would define a non-convex program (which
    the previous scipy stack accepted silently).
    """
    lam_min = float(np.linalg.eigvalsh(sigma)[0])
    if lam_min >= 0.0:
        return sigma
    if lam_min < -_PSD_TOL:
        raise ValueError(
            f"matrix is materially non-PSD (min eigenvalue {lam_min:.3e} "
            f"< -{_PSD_TOL:.0e}); a valid correlation/covariance matrix "
            f"is required"
        )
    vals, vecs = np.linalg.eigh(sigma)
    repaired = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    return 0.5 * (repaired + repaired.T)


def _build_sigma(
    corr_matrix: pd.DataFrame,
    vols: Optional[_VolsLike],
) -> np.ndarray:
    """Build the matrix the optimizers consume: ``ρ`` or ``diag(σ)·ρ·diag(σ)``.

    Validates ``corr_matrix`` (see ``_validate_corr_matrix``). When
    ``vols`` is given, aligns it to ``corr_matrix.index`` by label
    (keys must be set-equal to the index; order irrelevant) and requires
    every value finite and ``> 0``. The result passes through
    ``_ensure_psd`` (dust-repair or reject — see its docstring). Raises
    ``TypeError`` / ``ValueError`` accordingly.
    """
    rho = _validate_corr_matrix(corr_matrix)
    if vols is None:
        return _ensure_psd(rho)

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
    return _ensure_psd(rho * np.outer(sig, sig))
