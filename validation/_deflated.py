"""
Probabilistic and Deflated Sharpe Ratios (Bailey & Lopez de Prado).

PSR: probability the true Sharpe exceeds a benchmark, adjusting the observed
per-bar Sharpe for track length, skewness, and kurtosis. DSR: a PSR whose
benchmark is the expected MAXIMUM Sharpe under the null across N trials — the
selection-bias haircut for the best cell of a parameter sweep.

Reference: Bailey, D. and M. Lopez de Prado (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality", Journal of Portfolio Management 40(5); Lopez de Prado,
"Advances in Financial Machine Learning" (2018), sections 14.7.2-14.7.3.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from volatility import bars_per_year as _bars_per_year

from ._common import window_pnl

logger = logging.getLogger(__name__)

_NAN = float('nan')
_MIN_PNL_OBS = 3
_EULER_GAMMA = 0.5772156649015329


def _per_bar_sharpe(vals: np.ndarray) -> float:
    """Per-bar (non-annualized) Sharpe; NaN for T<2 or zero variance."""
    if len(vals) < 2:
        return _NAN
    std = float(np.std(vals, ddof=1))
    return float(vals.mean() / std) if std > 0 else _NAN


def _psr(vals: np.ndarray, sr_benchmark: float) -> float:
    """PSR on per-bar PnL vs a per-bar benchmark Sharpe. Skewness is the
    biased ML estimate; kurtosis is NON-excess (Gaussian = 3), matching the
    paper's gamma_4 convention. NaN when T<3, variance is zero, or the
    variance term is non-positive."""
    t = len(vals)
    if t < _MIN_PNL_OBS:
        return _NAN
    sr = _per_bar_sharpe(vals)
    if pd.isna(sr):
        return _NAN
    g3 = float(_skew(vals))
    g4 = float(_kurtosis(vals, fisher=False))
    var_term = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    if var_term <= 0:
        logger.debug("_psr: non-positive variance term %.3g — NaN", var_term)
        return _NAN
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def probabilistic_sharpe(
    equity_curve: pd.DataFrame,
    *,
    initial_capital: float,
    timeframe: str,
    days_convention: str,
    benchmark_sharpe: float = 0.0,
    start=None,
) -> float:
    """
    PSR of a backtest: probability its true Sharpe exceeds
    ``benchmark_sharpe`` (given in ANNUALIZED units; de-annualized
    internally), adjusting for track length, skew, and kurtosis of the
    per-bar dollar PnL. ``start`` windows the series without folding
    pre-start PnL into the first kept bar (see ``window_pnl``). Degenerate
    data (T<3, zero variance) yields NaN.
    """
    bpy = _bars_per_year(timeframe, days_convention)
    pnl, _ = window_pnl(equity_curve, initial_capital=initial_capital,
                        start=start)
    return _psr(pnl.to_numpy(dtype=float),
                float(benchmark_sharpe) / math.sqrt(bpy))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """DSR verdict plus every input to the haircut (auditable):
    ``dsr`` (probability the winner's true Sharpe > 0 after N trials),
    ``winner_params``, annualized observed/benchmark Sharpes, trial count,
    per-bar SR variance across cells, winner PnL skew/kurtosis, T."""
    dsr: float
    winner_params: Dict[str, Any]
    sharpe_annualized: float
    sr0_annualized: float
    n_trials: int
    var_sr: float
    skew: float
    kurt: float
    n_bars: int


def deflated_sharpe(sweep, *, n_trials: Optional[int] = None
                    ) -> DeflatedSharpeResult:
    """
    Deflated Sharpe over a ``SweepResult``: the best cell's PSR evaluated
    against ``SR0 = sqrt(V[{SR_n}]) * ((1-gamma)*ppf(1-1/N) +
    gamma*ppf(1-1/(N*e)))`` — the expected maximum Sharpe of N zero-skill
    trials. Per-cell per-bar Sharpes use the sweep's ``stats_start`` trim.
    ``start`` windows the series without folding pre-start PnL into the
    first kept bar (see ``window_pnl``). ``n_trials`` defaults to the cell
    count (conservative — correlated cells overstate the trial count);
    N<=1 or V=0 degrades SR0 to 0 (plain PSR). Raises on an empty sweep or
    ``n_trials < 1``; an all-degenerate sweep (no cell has a finite Sharpe)
    yields ``dsr=NaN`` with ``winner_params`` set arbitrarily to the first
    cell (grid order) and ``n_bars=0`` (data law).
    """
    keys = sweep.keys()
    if not keys:
        raise ValueError("sweep holds no cells")
    if n_trials is not None and (not isinstance(n_trials, int)
                                 or n_trials < 1):
        raise ValueError(f"n_trials must be a positive int, got {n_trials}")
    n = n_trials if n_trials is not None else len(keys)
    bpy = _bars_per_year(sweep.timeframe, sweep.days_convention)
    start = sweep.stats_start_resolved

    srs: Dict[tuple, float] = {}
    pnls: Dict[tuple, np.ndarray] = {}
    for key in keys:
        params = dict(zip(sweep.param_names, key))
        pnl, _ = window_pnl(sweep.equity(**params),
                            initial_capital=sweep.initial_capital(**params),
                            start=start)
        vals = pnl.to_numpy(dtype=float)
        pnls[key] = vals
        srs[key] = _per_bar_sharpe(vals)
    valid = {k: v for k, v in srs.items() if not pd.isna(v)}
    if not valid:
        logger.debug("deflated_sharpe: no cell has a finite Sharpe — NaN")
        return DeflatedSharpeResult(
            dsr=_NAN, winner_params=dict(zip(sweep.param_names, keys[0])),
            sharpe_annualized=_NAN, sr0_annualized=_NAN, n_trials=n,
            var_sr=_NAN, skew=_NAN, kurt=_NAN, n_bars=0)

    winner = max(valid, key=valid.get)
    wvals = pnls[winner]
    var_sr = (float(np.var(list(valid.values()), ddof=1))
              if len(valid) >= 2 else 0.0)
    if n <= 1 or var_sr <= 0.0:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(var_sr) * (
            (1.0 - _EULER_GAMMA) * float(norm.ppf(1.0 - 1.0 / n))
            + _EULER_GAMMA * float(norm.ppf(1.0 - 1.0 / (n * math.e))))
    dsr = _psr(wvals, sr0)
    return DeflatedSharpeResult(
        dsr=dsr,
        winner_params=dict(zip(sweep.param_names, winner)),
        sharpe_annualized=valid[winner] * math.sqrt(bpy),
        sr0_annualized=sr0 * math.sqrt(bpy),
        n_trials=n, var_sr=var_sr,
        skew=float(_skew(wvals)) if len(wvals) >= _MIN_PNL_OBS else _NAN,
        kurt=(float(_kurtosis(wvals, fisher=False))
              if len(wvals) >= _MIN_PNL_OBS else _NAN),
        n_bars=len(wvals))
