"""Correlation-matrix helper for portfolio/forecast diversification analysis.

One-shot Pearson correlation matrix of a DataFrame's columns. Two intended
use cases:

* **Variation correlations** — input columns = per-variation forecast time
  series (e.g. EWMAC look-back variations). Output feeds Carver-style
  weight derivation and the IDM (Instrument Diversification Multiplier).
* **Sub-trading-system correlations** — input columns = per-instrument
  return series for one trading rule. Output feeds top-level allocation.

Optionally applies Ledoit-Wolf shrinkage (``shrinkage='ledoit_wolf'``):
columns are standardized to unit variance and the resulting correlation
matrix is shrunk toward the identity with the closed-form optimal
intensity (Ledoit & Wolf 2004). Standardizing first makes the shrunk
estimate **scale-invariant** — like the unshrunk path — so feeding
absolute price changes of differently-priced instruments does not
collapse cross-correlations toward 0. Shrinkage keeps the estimate
well-conditioned and positive-definite even when the column count
approaches or exceeds the row count — the high-dimensional regime where
the raw sample estimator degrades into noise.

This is a research/setup-time helper, not a per-bar indicator: compute
once from a historical or backtest-derived DataFrame, then hold the
result as a fixed input to a weighting routine.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ['correlation_matrix']

_ALLOWED_METHODS = {'pearson', 'spearman', 'kendall'}
_ALLOWED_SHRINKAGE = {None, 'ledoit_wolf'}


def correlation_matrix(
    values: pd.DataFrame,
    *,
    lookback: Optional[int] = None,
    method: str = 'pearson',
    shrinkage: Optional[str] = None,
) -> pd.DataFrame:
    """Return the N×N column-wise correlation matrix of ``values``.

    Parameters
    ----------
    values
        Wide-format DataFrame, one column per series. Without shrinkage,
        NaNs are handled pairwise (pandas default): each (i, j) entry
        uses observations where both columns are finite. With
        ``shrinkage='ledoit_wolf'``, deletion is **listwise** instead —
        the estimator needs complete rows, so any row containing a NaN
        is dropped for every column.
    lookback
        Number of most-recent rows to use. ``None`` (default) uses the
        entire frame. Must be ``>= 2`` when provided. If ``lookback``
        exceeds ``len(values)``, the full frame is used (no padding,
        no error) — the helper degrades gracefully at the start of a
        backtest before enough history has accumulated.
    method
        ``'pearson'`` (default), ``'spearman'``, or ``'kendall'`` — forwarded
        to ``pandas.DataFrame.corr``. Pearson is the right choice for
        variation forecasts and instrument returns.
    shrinkage
        ``None`` (default — raw sample estimator) or ``'ledoit_wolf'``:
        standardize each column to unit variance, fit
        ``sklearn.covariance.LedoitWolf`` on the (lookback-sliced,
        listwise-complete) window, and convert the shrunk covariance to a
        correlation matrix via ``D⁻¹ Σ D⁻¹`` with ``D = sqrt(diag(Σ))``.
        Standardizing first shrinks the *correlation* structure toward the
        identity, making the result scale-invariant (matching the unshrunk
        path). Requires ``method='pearson'``. The fitted shrinkage intensity
        (in correlation space) is attached to the result as
        ``result.attrs['lw_shrinkage']`` (a float in ``(0, 1]``) for
        diagnostics. Requires scikit-learn (imported lazily — only this
        code path needs it).

    Returns
    -------
    pd.DataFrame
        N×N matrix indexed and columned by ``values.columns``. Diagonal
        is ``1.0``; off-diagonal entries are in ``[-1, 1]``. Symmetric.
        Strictly positive-definite when shrinkage is applied with a
        nonzero fitted intensity.

    Raises
    ------
    TypeError
        If ``values`` is not a ``pd.DataFrame``.
    ValueError
        If ``values`` has fewer than two columns, fewer than two rows,
        ``lookback`` is not ``None`` and ``< 2``, ``method`` is not
        one of ``{'pearson', 'spearman', 'kendall'}``, ``shrinkage`` is
        not ``None`` or ``'ledoit_wolf'``, ``shrinkage`` is combined
        with a non-Pearson ``method``, or fewer than two complete rows
        survive listwise deletion on the shrinkage path.
    ImportError
        If ``shrinkage='ledoit_wolf'`` is requested and scikit-learn is
        not installed.
    """
    if not isinstance(values, pd.DataFrame):
        raise TypeError(
            f"values must be a pd.DataFrame, got {type(values).__name__}"
        )
    if values.shape[1] < 2:
        raise ValueError(
            f"values must have >= 2 columns, got {values.shape[1]}"
        )
    if values.shape[0] < 2:
        raise ValueError(
            f"values must have >= 2 rows, got {values.shape[0]}"
        )
    if lookback is not None and lookback < 2:
        raise ValueError(f"lookback must be >= 2 when provided, got {lookback}")
    if method not in _ALLOWED_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_ALLOWED_METHODS)}, got {method!r}"
        )
    if shrinkage not in _ALLOWED_SHRINKAGE:
        raise ValueError(
            f"shrinkage must be None or 'ledoit_wolf', got {shrinkage!r}"
        )
    if shrinkage == 'ledoit_wolf' and method != 'pearson':
        raise ValueError(
            f"shrinkage='ledoit_wolf' requires method='pearson' "
            f"(shrinkage is covariance-based), got method={method!r}"
        )

    window = values if lookback is None else values.iloc[-lookback:]
    if shrinkage is None:
        return window.corr(method=method)
    elif shrinkage == 'ledoit_wolf':
        return _ledoit_wolf_corr(window)
    else:
        raise ValueError(f"Unexpected shrinkage: {shrinkage!r}")


def _ledoit_wolf_corr(window: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk correlation matrix of ``window``'s columns.

    Standardizes each column to unit variance, then fits
    ``sklearn.covariance.LedoitWolf`` (2004 scaled-identity-target
    estimator) on the listwise-complete rows — so shrinkage acts on the
    *correlation* structure (scale-invariant), not the raw-scale
    covariance. Converts the shrunk covariance to a correlation matrix,
    clips entries to ``[-1, 1]`` (numerical hygiene), and restores an
    exact ``1.0`` diagonal. Constant (zero-variance) columns get NaN
    off-diagonals (correlation undefined), mirroring the unshrunk path.
    The fitted intensity goes into ``result.attrs['lw_shrinkage']``.

    Raises ``ValueError`` when fewer than 2 complete rows survive NaN
    removal, ``ImportError`` when scikit-learn is missing.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError(
            "shrinkage='ledoit_wolf' requires scikit-learn "
            "(pip install scikit-learn)"
        ) from exc

    complete = window.dropna()
    if len(complete) < 2:
        raise ValueError(
            f"shrinkage='ledoit_wolf' needs >= 2 complete rows after "
            f"listwise NaN removal, got {len(complete)}"
        )
    arr = complete.to_numpy(dtype=float)
    # Standardize columns to unit variance before the fit, so Ledoit-Wolf
    # shrinks the *correlation* structure toward the identity rather than the
    # raw-scale covariance toward a scaled identity. Without this the target
    # μI is dominated by the largest-variance column; the shrunk diagonal of
    # every small-scale column is inflated by orders of magnitude and the
    # cov→corr step collapses their cross-correlations toward 0. z-scores are
    # invariant to per-column scaling, so the result is scale-invariant — the
    # same contract the unshrunk .corr() path already honors.
    std = arr.std(axis=0, ddof=0)
    nonzero = std > 0.0
    z = np.zeros_like(arr)
    z[:, nonzero] = (arr[:, nonzero] - arr[:, nonzero].mean(axis=0)) / std[nonzero]
    lw = LedoitWolf().fit(z)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    # Constant (zero-variance) columns have undefined correlation; mirror the
    # unshrunk .corr() path and report NaN off-diagonals rather than a spurious
    # 0, while preserving the documented unit diagonal.
    if not nonzero.all():
        const = ~nonzero
        corr[const, :] = np.nan
        corr[:, const] = np.nan
    np.fill_diagonal(corr, 1.0)
    result = pd.DataFrame(corr, index=window.columns, columns=window.columns)
    result.attrs['lw_shrinkage'] = float(lw.shrinkage_)
    return result
