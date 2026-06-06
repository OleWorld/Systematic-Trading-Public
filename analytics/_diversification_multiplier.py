"""Carver's Diversification Multiplier for a weighted, correlated bundle.

Computes

    DM = 1 / sqrt(wᵀ ρ w)

where ``w`` is the weight vector (non-negative, summing to 1) and ``ρ``
is the correlation matrix of the underlying series (instruments,
strategy variations, or sub-trading-systems — anything whose risk a
weighted basket pools).

DM quantifies the volatility-cancellation credit from holding multiple
imperfectly-correlated components: perfectly correlated → ``1.0`` (no
credit); perfectly uncorrelated equal-weighted N components →
``sqrt(N)``; partial correlations land in between. Multiplying a
risk-target by DM scales the bundle's expected vol *back up* to the
single-component target, compensating for diversification-driven
cancellation. Reference: Carver, *Leveraged Trading*.

This is a one-shot research/setup helper — call once at portfolio
build time with the correlation matrix and the chosen capital weights.
Not a per-bar primitive.
"""

from __future__ import annotations

import math
from typing import Mapping, Union

import numpy as np
import pandas as pd

__all__ = ['diversification_multiplier']

_WeightsLike = Union[Mapping[str, float], pd.Series]
_SYMMETRY_TOL = 1e-9
_SUM_TOL = 1e-9


def diversification_multiplier(
    weights: _WeightsLike,
    corr_matrix: pd.DataFrame,
) -> float:
    """Return ``1 / sqrt(wᵀ ρ w)`` for the given weights and correlation matrix.

    Parameters
    ----------
    weights
        Capital weights, keyed by the same labels as ``corr_matrix.index``.
        Accepted as ``dict`` or ``pd.Series``. Aligned to
        ``corr_matrix.index`` by label (no positional alignment), so
        the input order is irrelevant. All values must be ``>= 0`` and
        must sum to ``1.0`` within ``1e-9`` — DM is a capital-share
        diversification quantity, only meaningful for a fully-allocated
        weight vector.
    corr_matrix
        N×N correlation matrix as a ``pd.DataFrame``. Must be square,
        symmetric (within ``1e-9``), and have matching index/columns.
        Typically produced by ``analytics.correlation_matrix``.

    Returns
    -------
    float
        The diversification multiplier. ``1.0`` for ``N=1`` or
        perfectly-correlated inputs; ``sqrt(N)`` for the
        equal-weighted, fully-uncorrelated case; in between for
        partial correlations.

    Raises
    ------
    TypeError
        If ``corr_matrix`` is not a ``pd.DataFrame``.
    ValueError
        If ``corr_matrix`` is not square, not symmetric, has mismatched
        index/columns; if ``weights`` keys don't equal
        ``corr_matrix.index`` as a set; if any weight is negative; if
        weights don't sum to ``1.0``; or if the resulting
        ``wᵀ ρ w`` is non-positive (numerical edge — non-PSD matrix).

    Notes
    -----
    The formula is reusable beyond instrument-level allocation:
    forecast-variation weights and sub-trading-system weights can both
    feed this helper, as long as the weight vector represents a
    fully-allocated capital share at that level.
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
    rho = corr_matrix.to_numpy()
    if not np.allclose(rho, rho.T, atol=_SYMMETRY_TOL):
        raise ValueError("corr_matrix must be symmetric within 1e-9")

    if isinstance(weights, pd.Series):
        weights_dict = weights.to_dict()
    else:
        weights_dict = dict(weights)

    expected_keys = set(corr_matrix.index)
    got_keys = set(weights_dict.keys())
    if got_keys != expected_keys:
        missing = expected_keys - got_keys
        extra = got_keys - expected_keys
        raise ValueError(
            f"weights keys must equal corr_matrix.index as a set; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    w = np.array([weights_dict[label] for label in corr_matrix.index], dtype=float)
    if np.any(w < 0):
        raise ValueError(
            f"weights must be non-negative, got {weights_dict}"
        )
    total = float(w.sum())
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOL):
        raise ValueError(
            f"weights must sum to 1.0 within {_SUM_TOL}, got sum={total}"
        )

    portfolio_variance = float(w @ rho @ w)
    if portfolio_variance <= 0:
        raise ValueError(
            f"wᵀ ρ w must be > 0 (got {portfolio_variance}); "
            "check that corr_matrix is positive semi-definite"
        )
    return 1.0 / math.sqrt(portfolio_variance)
