"""
Block-bootstrap confidence intervals and p-values on backtest results.

``bootstrap_stats`` resamples the per-bar dollar PnL (derived per the
``analytics.backtest_stats`` conventions) with the stationary bootstrap
(Politis & Romano 1994; default), the circular block bootstrap, or naive IID
resampling, and reports percentile CIs plus one-sided p-values under the
zero-mean-PnL null for the location metrics. The automatic expected block
length follows Politis & White (2004) with the Patton, Politis & White (2009)
correction.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from volatility import bars_per_year as _bars_per_year

from ._common import window_pnl

logger = logging.getLogger(__name__)

_NAN = float('nan')
_MIN_PNL_OBS = 3

_METHODS = ('stationary', 'circular', 'iid')
_METRIC_LABELS = ('Sharpe Ratio', 'Net PnL [$]', 'CAGR [%]',
                  'Max Drawdown [$]', 'Max Drawdown [%]')
#: Metrics with a natural H0 (zero-mean per-bar PnL) — the only ones that
#: get a p-value; drawdown has no natural null.
_P_VALUE_METRICS = ('Sharpe Ratio', 'Net PnL [$]', 'CAGR [%]')


def _flat_top_window(x: np.ndarray) -> np.ndarray:
    """Trapezoidal flat-top lag window: 1 on |x|<=1/2, 2(1-|x|) on
    1/2<|x|<=1, 0 beyond (Politis & Romano flat-top kernel)."""
    ax = np.abs(np.asarray(x, dtype=float))
    return np.where(ax <= 0.5, 1.0,
                    np.where(ax <= 1.0, 2.0 * (1.0 - ax), 0.0))


def politis_white_block_length(values: np.ndarray) -> float:
    """
    Automatic expected block length for the stationary bootstrap.

    Politis & White (2004) with the Patton–Politis–White (2009) correction:
    pick bandwidth ``m_hat`` as the smallest lag after which ``K_n``
    consecutive sample autocorrelations sit inside the
    ``2*sqrt(log10(T)/T)`` band (falling back to the scan bound when none
    does), set ``M = 2*m_hat``, then
    ``b = ((2*G^2)/D)^(1/3) * T^(1/3)`` with
    ``G = 2*sum(lam_k * k * gamma_k)`` and
    ``D = 2*(gamma_0 + 2*sum(lam_k * gamma_k))^2`` over flat-top weights
    ``lam_k``. Clamped to ``[1, min(3*sqrt(T), T/3)]``. Series shorter than
    3 observations or constant return 1.0. This is the stationary-bootstrap
    optimum; ``bootstrap_stats`` also applies it as the expected block
    length for ``method='circular'``, whose own theoretical constant differs
    by a factor of ~1.14x (accepted approximation, not re-derived here).
    """
    x = np.asarray(values, dtype=float)
    t = len(x)
    if t < _MIN_PNL_OBS or float(np.std(x)) == 0.0:
        return 1.0
    x_c = x - x.mean()
    denom = float(np.dot(x_c, x_c))          # T * biased variance
    # sqrt(log10 T) vs some references' log10 T: both floor at 5 for any
    # realistic T, so the choice is deliberate and functionally inert.
    k_n = max(5, int(math.sqrt(math.log10(t))))
    n_lags = min(int(math.ceil(math.sqrt(t))) + k_n, t - 1)
    rho = np.array([float(np.dot(x_c[k:], x_c[:-k])) / denom
                    for k in range(1, n_lags + 1)])
    threshold = 2.0 * math.sqrt(math.log10(t) / t)
    m_hat = None
    for m in range(0, n_lags - k_n + 1):
        if np.all(np.abs(rho[m:m + k_n]) < threshold):
            m_hat = m
            break
    if m_hat is None:
        m_hat = max(1, n_lags - k_n)
    big_m = min(2 * m_hat, t - 1)
    if big_m < 1:
        return 1.0
    ks = np.arange(1, big_m + 1)
    gamma = np.array([float(np.dot(x_c[k:], x_c[:-k])) / t for k in ks])
    gamma0 = denom / t
    lam = _flat_top_window(ks / big_m)
    g_hat = 2.0 * float(np.sum(lam * ks * gamma))
    d_sb = 2.0 * (gamma0 + 2.0 * float(np.sum(lam * gamma))) ** 2
    if d_sb <= 0.0:
        return 1.0
    b = ((2.0 * g_hat ** 2) / d_sb) ** (1.0 / 3.0) * t ** (1.0 / 3.0)
    cap = min(3.0 * math.sqrt(t), t / 3.0)
    result = float(min(max(b, 1.0), cap))
    logger.debug("politis_white_block_length: T=%d m_hat=%d M=%d b=%.2f",
                 t, m_hat, big_m, result)
    return result


def _iid_indices(rng, t: int, b: int) -> np.ndarray:
    """IID resampling with replacement: (b, t) index matrix."""
    return rng.integers(0, t, size=(b, t)).astype(np.int32, copy=False)


def _circular_indices(rng, t: int, b: int, block_length: float) -> np.ndarray:
    """Circular block bootstrap: fixed-length blocks, wrap-around starts."""
    length = max(1, int(round(block_length)))
    n_blocks = -(-t // length)                       # ceil division
    starts = rng.integers(0, t, size=(b, n_blocks))
    offsets = np.arange(length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % t
    return idx.reshape(b, -1)[:, :t].astype(np.int32, copy=False)


def _stationary_indices(rng, t: int, b: int, mean_block: float) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano): geometric block lengths with
    mean ``mean_block``, wrap-around continuation. Fully vectorized: each
    position either continues the previous index (+1 mod T, prob 1-p) or
    restarts at a fresh uniform index (prob p = 1/mean_block)."""
    p = min(1.0, 1.0 / max(float(mean_block), 1.0))
    starts = rng.integers(0, t, size=(b, t))
    new_block = rng.random(size=(b, t)) < p
    new_block[:, 0] = True
    pos = np.arange(t)
    block_start_pos = np.maximum.accumulate(
        np.where(new_block, pos, -1), axis=1)
    start_vals = np.take_along_axis(starts, block_start_pos, axis=1)
    return ((start_vals + (pos - block_start_pos)) % t).astype(np.int32,
                                                                copy=False)


def _resample_indices(rng, t: int, b: int, method: str,
                      block_length: float) -> np.ndarray:
    """Dispatch to the method's index generator; unknown methods raise."""
    if method == 'stationary':
        return _stationary_indices(rng, t, b, block_length)
    elif method == 'circular':
        return _circular_indices(rng, t, b, block_length)
    elif method == 'iid':
        return _iid_indices(rng, t, b)
    else:
        raise ValueError(f"Unexpected method: {method!r}")


@dataclass(frozen=True)
class BootstrapResult:
    """
    Result of ``bootstrap_stats``: ``table`` is indexed by metric label with
    columns ``estimate`` (from the real series — identical to
    ``analytics.backtest_stats``), ``ci_low``/``ci_high`` (percentile CI),
    and ``p_value`` (one-sided vs the zero-mean-PnL null; NaN for drawdown
    metrics). ``block_length`` is the resolved expected block length (1.0
    for ``'iid'``). ``print(result.table)`` is the display.
    """
    table: pd.DataFrame
    method: str
    block_length: float
    n_resamples: int
    ci: float
    seed: Optional[int]
    n_bars: int


def _metrics_on_paths(pnl_matrix: np.ndarray, *, initial_capital: float,
                      bpy: float) -> Dict[str, np.ndarray]:
    """Vectorized metric set over (B, T) PnL paths; conventions match
    ``analytics.backtest_stats`` (peak seeded at ``initial_capital``)."""
    b, t = pnl_matrix.shape
    net = pnl_matrix.sum(axis=1)
    mean = pnl_matrix.mean(axis=1)
    std = pnl_matrix.std(axis=1, ddof=1) if t >= 2 else np.full(b, _NAN)
    with np.errstate(invalid='ignore', divide='ignore'):
        sharpe = np.where(std > 0, mean / std * math.sqrt(bpy), _NAN)
    balance = initial_capital + np.cumsum(pnl_matrix, axis=1)
    peak = np.maximum(np.maximum.accumulate(balance, axis=1), initial_capital)
    dd_usd = (peak - balance).max(axis=1)
    dd_pct = (100.0 * (peak - balance) / peak).max(axis=1)
    years = t / bpy
    final = balance[:, -1]
    with np.errstate(invalid='ignore'):
        cagr = np.where(
            (years > 0) & (final > 0),
            100.0 * ((final / initial_capital) ** (1.0 / years) - 1.0), _NAN)
    return {'Sharpe Ratio': sharpe, 'Net PnL [$]': net, 'CAGR [%]': cagr,
            'Max Drawdown [$]': dd_usd, 'Max Drawdown [%]': dd_pct}


def bootstrap_stats(
    equity_curve: pd.DataFrame,
    *,
    initial_capital: float,
    timeframe: str,
    days_convention: str,
    start=None,
    n_resamples: int = 5000,
    method: str = 'stationary',
    block_length: Optional[float] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Bootstrap CIs and p-values on backtest metrics from an equity curve.

    Derives per-bar dollar PnL from the full history, then windows it:
    ``start`` slices to ``index >= start`` and reseeds the baseline to the
    true balance entering the window (no fold-in — see ``window_pnl``);
    with ``start=None`` the estimates are identical to
    ``analytics.backtest_stats`` on the same curve. Resamples the windowed
    PnL ``n_resamples`` times with ``method`` (``'stationary'`` default /
    ``'circular'`` / ``'iid'``), and reports, per metric: the real-series
    estimate, the ``ci`` percentile interval, and — for Sharpe / Net PnL /
    CAGR — a one-sided p-value under H0: zero-mean PnL (the observed PnL is
    mean-centered and re-resampled; ``p = (1+#{stat* >= estimate})/(B+1)``).
    ``block_length=None`` resolves via ``politis_white_block_length``.
    Bad params raise; degenerate data (empty, T<3, zero variance) yields
    NaN cells, never a raise.
    """
    bpy = _bars_per_year(timeframe, days_convention)   # raises on bad inputs
    if method not in _METHODS:
        raise ValueError(f"Unexpected method: {method!r}")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive int, got {n_resamples}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if block_length is not None and block_length <= 0:
        raise ValueError(f"block_length must be > 0, got {block_length}")
    pnl, baseline = window_pnl(equity_curve,
                               initial_capital=initial_capital, start=start)
    vals = pnl.to_numpy(dtype=float)
    t = len(vals)

    table = pd.DataFrame(_NAN, index=list(_METRIC_LABELS),
                         columns=['estimate', 'ci_low', 'ci_high', 'p_value'])
    if t > 0:
        est = _metrics_on_paths(vals[None, :], initial_capital=baseline,
                                bpy=bpy)
        for label in _METRIC_LABELS:
            table.loc[label, 'estimate'] = float(est[label][0])

    if t < _MIN_PNL_OBS:
        degenerate = True
    else:
        degenerate = float(np.std(vals, ddof=1)) == 0.0
    if degenerate:
        logger.debug("bootstrap_stats: degenerate PnL (T=%d) — NaN CIs", t)
        return BootstrapResult(table=table, method=method,
                               block_length=_NAN, n_resamples=n_resamples,
                               ci=ci, seed=seed, n_bars=t)

    if method == 'iid':
        resolved = 1.0
    elif block_length is not None:
        resolved = float(block_length)
    else:
        resolved = politis_white_block_length(vals)

    rng = np.random.default_rng(seed)
    idx = _resample_indices(rng, t, n_resamples, method, resolved)
    dist = _metrics_on_paths(vals[idx], initial_capital=baseline,
                             bpy=bpy)
    centered = vals - vals.mean()
    idx_null = _resample_indices(rng, t, n_resamples, method, resolved)
    null = _metrics_on_paths(centered[idx_null],
                             initial_capital=baseline, bpy=bpy)

    lo_q, hi_q = (1.0 - ci) / 2.0, (1.0 + ci) / 2.0
    for label in _METRIC_LABELS:
        d = dist[label]
        d = d[~np.isnan(d)]
        if len(d):
            table.loc[label, 'ci_low'] = float(np.quantile(d, lo_q))
            table.loc[label, 'ci_high'] = float(np.quantile(d, hi_q))
        estimate = table.loc[label, 'estimate']
        if label in _P_VALUE_METRICS and not pd.isna(estimate):
            nd = null[label]
            nd = nd[~np.isnan(nd)]
            if len(nd):
                table.loc[label, 'p_value'] = float(
                    (1 + np.sum(nd >= estimate)) / (len(nd) + 1))
    return BootstrapResult(table=table, method=method, block_length=resolved,
                           n_resamples=n_resamples, ci=ci, seed=seed,
                           n_bars=t)
