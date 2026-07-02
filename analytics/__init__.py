"""analytics — Research/setup helpers for portfolio diversification analysis.

Exposes one-shot helpers used to derive Carver-style variation /
instrument weights and the matching Diversification Multiplier (IDM at
the instrument level; reusable at variation and sub-trading-system
levels). Helpers in this package are intended to be called once from
research notebooks or strategy-setup code, not on every bar — see
``indicator/`` for the stateful per-bar primitives.

Public surface:

* ``correlation_matrix(values, *, lookback=None, method='pearson')`` —
  N×N correlation matrix of a wide DataFrame's columns.
* ``diversification_multiplier(weights, corr_matrix)`` — Carver's
  ``1 / sqrt(wᵀ ρ w)``. Quantifies the vol-cancellation credit of a
  weighted bundle.
* ``equal_weight(labels)`` / ``min_variance(corr_matrix, vols=None)`` /
  ``risk_parity(corr_matrix, vols=None)`` — label-agnostic portfolio
  weight optimizers returning named weights (``Dict[str, float]``).
  Min-variance solves the long-only QP exactly; risk_parity solves
  full ERC (equal risk contribution). See
  ``analytics._portfolio_optimizer`` for the shared contract
  (label-aligned inputs, per-scheme constraints, corr-vs-covariance
  equivalence).
* ``validate_named_weights(weights, labels)`` — validate / default a
  label→weight mapping (``None`` → equal weight; else keys set-equal to
  ``labels``, finite, non-negative, sum→1). The one shared entry point
  for weight validation across the allocation hierarchy (instrument,
  strategy, and forecast-variation weights).
* ``backtest_stats(equity_curve, trade_log, *, initial_capital,
  timeframe, days_convention)`` — post-run summary statistics (drawdowns,
  Sharpe/Sortino/Calmar, volatility, trade stats) as an ordered
  ``pd.Series``; dollar-first with percentage twins where meaningful.
* ``pnl_attribution(equity_curve, system, *, eps=1e-6)`` — post-run
  decomposition of the actual per-bar PnL (gross of commission) across
  an ``Orchestrator``'s strategies and their variations, by signed
  forecast-contribution shares (one-bar lag). Returns a
  ``PnLAttributionResult`` whose ``by('strategy')`` /
  ``by('strategy', 'variation')`` / ``by('symbol', ...)`` views all sum
  per bar to ``result.total``.
* ``turnover_stats(equity_curve, trade_log, *, timeframe,
  days_convention)`` — per-instrument Carver turnover table (avg
  |position| in contracts, round trips/year, avg holding days over the
  first-fill→end active window) plus an ``'Average'`` row carrying only
  the holding-period mean.
"""

from analytics._attribution import PnLAttributionResult, pnl_attribution
from analytics._correlation import correlation_matrix
from analytics._diversification_multiplier import diversification_multiplier
from analytics._portfolio_optimizer import (
    equal_weight,
    min_variance,
    risk_parity,
    validate_named_weights,
)
from analytics._stats import backtest_stats
from analytics._turnover import turnover_stats

__all__ = [
    'PnLAttributionResult',
    'backtest_stats',
    'correlation_matrix',
    'diversification_multiplier',
    'equal_weight',
    'min_variance',
    'pnl_attribution',
    'risk_parity',
    'turnover_stats',
    'validate_named_weights',
]
