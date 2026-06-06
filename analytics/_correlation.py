"""Correlation-matrix helper for portfolio/forecast diversification analysis.

One-shot Pearson correlation matrix of a DataFrame's columns. Two intended
use cases:

* **Variation correlations** — input columns = per-variation forecast time
  series (e.g. EWMAC look-back variations). Output feeds Carver-style
  weight derivation and the IDM (Instrument Diversification Multiplier).
* **Sub-trading-system correlations** — input columns = per-instrument
  return series for one trading rule. Output feeds top-level allocation.

This is a research/setup-time helper, not a per-bar indicator: compute
once from a historical or backtest-derived DataFrame, then hold the
result as a fixed input to a weighting routine.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = ['correlation_matrix']

_ALLOWED_METHODS = {'pearson', 'spearman', 'kendall'}


def correlation_matrix(
    values: pd.DataFrame,
    *,
    lookback: Optional[int] = None,
    method: str = 'pearson',
) -> pd.DataFrame:
    """Return the N×N column-wise correlation matrix of ``values``.

    Parameters
    ----------
    values
        Wide-format DataFrame, one column per series. NaNs are handled
        pairwise (pandas default): each (i, j) entry uses observations
        where both columns are finite.
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

    Returns
    -------
    pd.DataFrame
        N×N matrix indexed and columned by ``values.columns``. Diagonal
        is ``1.0``; off-diagonal entries are in ``[-1, 1]``. Symmetric.

    Raises
    ------
    TypeError
        If ``values`` is not a ``pd.DataFrame``.
    ValueError
        If ``values`` has fewer than two columns, fewer than two rows,
        ``lookback`` is not ``None`` and ``< 2``, or ``method`` is not
        one of ``{'pearson', 'spearman', 'kendall'}``.
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

    window = values if lookback is None else values.iloc[-lookback:]
    return window.corr(method=method)
