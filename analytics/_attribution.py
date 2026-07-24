"""Post-run PnL attribution by strategy and variation (pure proportional).

``pnl_attribution`` decomposes the portfolio's **actual** per-bar PnL —
the single source of truth — across the strategies of an
``orchestrator.Orchestrator`` (or the variations of a single
``Strategy``) using the forecast contributions recorded per bar. One-shot
— call it after ``Backtester.run()`` completes, not per bar.

Attribution scheme
------------------
Position sizing is linear in the combined forecast, and the combined
forecast is a weighted sum of contributions, so each symbol's per-bar
PnL is split by the **signed forecast-contribution shares**
``share_i = w_i·f_i / Σ w_j·f_j`` recorded on the *previous* bar (the
forecast at bar *t* fills at close(*t*) and earns the *t*→*t+1* move).
The forecast cap and the (effective) FDM scale every contributor
equally, so they cancel exactly in the ratio. Components therefore sum
to the actual total by construction; sizing frictions (position-buffer
dead-band, whole-lot rounding, clamps) are smeared pro-rata. Shares are
signed and unbounded in general — opposing strategies show as
offsetting PnL. Two guard rails keep the decomposition well-defined:

- near-cancellation rows (``|Σc| < eps·Σ|c|``) fall back to magnitude
  shares ``|c|/Σ|c|`` (bounded, still sums to 1);
- bars with no prior forecast at all (t0, warmup head) route their PnL
  to the reserved ``'unattributed'`` strategy — ≈0 outside t0 in
  practice, which doubles as a health check.

Net-vs-gross caveat
-------------------
The per-symbol PnL substrate (``realized_pnl + unrealized_pnl``
snapshots) is **gross of commission** — commission is account-level
only. ``result.total`` therefore reconciles to
``Net PnL + Total Commission`` from ``backtest_stats``; commission is
deliberately not attributed.

Timing assumptions
------------------
The one-bar share lag matches the engine's fill timing (orders fill on
the signal bar's close). Slippage on a fill lands at *t* or *t+1*
depending on same-timestamp symbol processing order (a one-bar smear;
totals are unaffected because a cumulative series is diffed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from analytics._stats import _collapse_equity
from data import ensure_utc_index

logger = logging.getLogger(__name__)

__all__ = ['PnLAttributionResult', 'pnl_attribution']

# Reserved labels (collisions with real labels raise at validation).
_UNATTRIBUTED = 'unattributed'   # strategy label for unattributable PnL
_PASSTHROUGH = 'all'             # variation label for children w/o variations
_UNATTRIBUTED_VARIATION = ''     # variation label paired with 'unattributed'

_ALLOWED_DIMS = ('strategy', 'variation', 'symbol')

# Required equity-curve columns (subset of the BacktestPortfolio row
# schema actually consumed here).
_REQUIRED_EQUITY_COLS = ('realized_pnl', 'unrealized_pnl')


@dataclass(frozen=True)
class PnLAttributionResult:
    """Attributed per-bar PnL in tidy long form, plus pivot-on-demand views.

    Attributes
    ----------
    long
        Tidy frame with columns ``timestamp`` (bar timestamp),
        ``symbol`` / ``strategy`` / ``variation`` (categoricals), and
        ``pnl`` (attributed per-bar dollars). Zero-PnL rows are dropped
        (views re-densify with 0.0). Unattributable PnL carries
        ``strategy='unattributed'``, ``variation=''`` and the true
        symbol.
    total
        Per-bar system gross PnL (dollars) on the full collapsed
        timestamp axis — the reconciliation target: every ``by(...)``
        view sums per bar to this series (to float precision).
    """

    long: pd.DataFrame
    total: pd.Series

    def by(self, *dims: str) -> pd.DataFrame:
        """Pivot the long frame into a timestamp-indexed wide view.

        Parameters
        ----------
        *dims
            One or more of ``'strategy'``, ``'variation'``,
            ``'symbol'`` — unique, in the desired column-nesting order
            (one dim → flat columns; several → a MultiIndex in the
            given order, e.g. ``by('symbol', 'strategy')``).

        Returns
        -------
        pd.DataFrame
            Per-bar attributed PnL summed over the omitted dimensions,
            reindexed to the full ``total`` axis with missing cells
            0.0. Note variation labels are unique only *within* a
            strategy — ``by('variation')`` alone sums same-named
            variations across strategies; use
            ``by('strategy', 'variation')`` for the unambiguous view.

        Raises
        ------
        ValueError
            If ``dims`` is empty, contains duplicates, or contains a
            name outside the allowed set.
        """
        if not dims:
            raise ValueError(
                f"by() needs at least one dimension out of {_ALLOWED_DIMS}"
            )
        bad = [d for d in dims if d not in _ALLOWED_DIMS]
        if bad:
            raise ValueError(
                f"Unknown attribution dimension(s) {bad}; "
                f"allowed: {_ALLOWED_DIMS}"
            )
        if len(set(dims)) != len(dims):
            raise ValueError(f"Duplicate dimensions in {dims}")
        if self.long.empty:
            return pd.DataFrame(index=self.total.index.copy())
        out = self.long.pivot_table(
            index='timestamp', columns=list(dims), values='pnl',
            aggfunc='sum', fill_value=0.0, observed=True,
        )
        return out.reindex(self.total.index, fill_value=0.0)

    @property
    def by_strategy(self) -> pd.DataFrame:
        """Convenience for ``by('strategy')``."""
        return self.by('strategy')

    @property
    def by_variation(self) -> pd.DataFrame:
        """Convenience for ``by('strategy', 'variation')``."""
        return self.by('strategy', 'variation')


def pnl_attribution(
    equity_curve: pd.DataFrame,
    system: Any,
    *,
    eps: float = 1e-6,
) -> PnLAttributionResult:
    """Attribute the actual per-bar PnL across strategies and variations.

    Parameters
    ----------
    equity_curve
        Output of ``portfolio.get_equity_curve()`` — timestamp-indexed,
        one row per ``BarEvent``; collapsed internally to one row per
        timestamp (last wins). Must carry the per-symbol
        ``realized_pnl`` and ``unrealized_pnl`` dict columns when
        non-empty. Empty frames are a clean edge case (empty result). A
        timezone-naive equity-curve index raises ``ValueError``;
        tz-aware non-UTC input is converted to UTC.
    system
        The forecast source the backtest traded — an
        ``orchestrator.Orchestrator`` (detected by its ``strategies``
        dict) or a single ``Strategy`` (strategy label =
        ``type(system).__name__``). Duck-typed: needs ``symbol_list``,
        ``get_records(symbol)`` and, per child strategy,
        ``get_records`` + the ``weights``/``forecast_<label>`` variation
        convention (children without it degrade to one ``'all'``
        passthrough variation).
    eps
        Relative near-cancellation threshold: rows where
        ``|Σ contributions| < eps · Σ|contributions|`` fall back to
        magnitude shares. Must be finite and > 0.

    Returns
    -------
    PnLAttributionResult
        Tidy long frame + reconciliation ``total`` (see the dataclass
        docstring). ``result.by(dims).sum(axis=1)`` equals ``total``
        per bar (float precision) for every dims choice, and
        ``total.cumsum().iloc[-1]`` equals the final
        ``Σ realized + Σ unrealized`` (gross of commission).

    Raises
    ------
    TypeError
        If ``equity_curve`` is not a DataFrame or ``system`` lacks the
        forecast-source surface (``get_records`` / ``symbol_list``).
    ValueError
        If ``eps`` is invalid, a non-empty curve has a timezone-naive
        index, a non-empty curve lacks the required PnL columns, a
        strategy label collides with ``'unattributed'``, a variation
        label collides with ``'all'``/``''``, or a managed symbol's
        records lack the expected forecast/weight columns.
    """
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError(
            f"equity_curve must be a DataFrame, got {type(equity_curve).__name__}"
        )
    if not callable(getattr(system, 'get_records', None)):
        raise TypeError(
            f"system must expose get_records(symbol), got {type(system).__name__}"
        )
    if getattr(system, 'symbol_list', None) is None:
        raise TypeError(
            f"system must expose symbol_list, got {type(system).__name__}"
        )
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if not equity_curve.empty:
        equity_curve = equity_curve.set_axis(
            ensure_utc_index(equity_curve.index, 'equity_curve'))
        missing = [c for c in _REQUIRED_EQUITY_COLS
                   if c not in equity_curve.columns]
        if missing:
            raise ValueError(
                f"equity_curve is missing required columns: {missing}"
            )

    children, single = _resolve_children(system)
    labels = list(children)

    eq = _collapse_equity(equity_curve)
    if eq.empty:
        logger.debug("pnl_attribution: empty equity curve — empty result")
        return PnLAttributionResult(_empty_long(), pd.Series(dtype=float))

    # Per-symbol cumulative gross PnL and its per-bar diff (first row vs
    # the 0 baseline). ffill/fillna guard synthetic inputs whose dicts
    # don't cover every symbol on every row (the real portfolio always
    # writes full dicts).
    idx = eq.index
    pnl_wide = (
        pd.DataFrame(list(eq['realized_pnl']), index=idx)
        .add(pd.DataFrame(list(eq['unrealized_pnl']), index=idx), fill_value=0.0)
        .astype(float)
        .ffill()
        .fillna(0.0)
    )
    pnl_diff = pnl_wide.diff()
    pnl_diff.iloc[0] = pnl_wide.iloc[0]
    total = pnl_diff.sum(axis=1)

    managed = {str(s) for s in system.symbol_list}

    # Long-frame accumulators: positional row indices into ``idx`` plus
    # parallel label/value arrays (concatenated once at the end).
    pos_parts: List[np.ndarray] = []
    sym_parts: List[np.ndarray] = []
    strat_parts: List[np.ndarray] = []
    var_parts: List[np.ndarray] = []
    pnl_parts: List[np.ndarray] = []

    def _append(sym: str, strat: str, var: str, values: np.ndarray) -> None:
        nz = np.nonzero(values)[0]
        if nz.size == 0:
            return
        pos_parts.append(nz)
        sym_parts.append(np.full(nz.size, sym, dtype=object))
        strat_parts.append(np.full(nz.size, strat, dtype=object))
        var_parts.append(np.full(nz.size, var, dtype=object))
        pnl_parts.append(values[nz])

    for sym in pnl_diff.columns:
        pnl_s = pnl_diff[sym].to_numpy(dtype=float)
        if not np.any(pnl_s):
            continue

        rec = system.get_records(str(sym)) if str(sym) in managed else pd.DataFrame()
        if rec.empty:
            # Never forecast (unmanaged, or no bars ever reached it).
            _append(str(sym), _UNATTRIBUTED, _UNATTRIBUTED_VARIATION, pnl_s)
            continue
        rec = rec[~rec.index.duplicated(keep='last')]

        shares = _system_shares(rec, labels, single, eps, str(sym))
        # THE SHIFT: shares from the symbol's bar t apply to the PnL at
        # its next bar (forecast t → fill close(t) → PnL at t+1); then
        # align to the global axis — ffill across other symbols'
        # timestamps is safe because this symbol's diff is 0 there.
        aligned = shares.shift(1).reindex(idx).ffill()
        covered = aligned.notna().all(axis=1).to_numpy()
        attr = aligned.to_numpy(dtype=float) * pnl_s[:, None]
        attr[~covered] = 0.0
        unatt = np.where(covered, 0.0, pnl_s)

        for j, lab in enumerate(labels):
            attr_l = attr[:, j]
            if not np.any(attr_l):
                continue
            split, leftover = _variation_split(
                children[lab], str(sym), attr_l, idx, eps,
            )
            if leftover is not None:
                unatt = unatt + leftover
            for vlab, values in split.items():
                _append(str(sym), lab, vlab, values)
        _append(str(sym), _UNATTRIBUTED, _UNATTRIBUTED_VARIATION, unatt)

    if not pos_parts:
        return PnLAttributionResult(_empty_long(), total)

    positions = np.concatenate(pos_parts)
    long = pd.DataFrame({
        'timestamp': idx.take(positions),
        'symbol': pd.Categorical(np.concatenate(sym_parts)),
        'strategy': pd.Categorical(np.concatenate(strat_parts)),
        'variation': pd.Categorical(np.concatenate(var_parts)),
        'pnl': np.concatenate(pnl_parts),
    })
    long.sort_values('timestamp', kind='stable', ignore_index=True,
                     inplace=True)
    n_unatt = int((long['strategy'] == _UNATTRIBUTED).sum())
    if n_unatt:
        logger.debug("pnl_attribution: %d unattributed long rows", n_unatt)
    return PnLAttributionResult(long, total)


def _resolve_children(system: Any) -> tuple:
    """Split ``system`` into ``({label: child}, single_mode)``.

    Orchestrator mode when ``system.strategies`` is a non-empty dict;
    otherwise the system itself is the sole child, labeled by its class
    name. Validates the reserved-label collisions here (the one place
    every label passes through).
    """
    strategies = getattr(system, 'strategies', None)
    if isinstance(strategies, dict) and strategies:
        children: Dict[str, Any] = dict(strategies)
        single = False
    else:
        children = {type(system).__name__: system}
        single = True

    if _UNATTRIBUTED in children:
        raise ValueError(
            f"strategy label {_UNATTRIBUTED!r} is reserved for "
            f"unattributable PnL"
        )
    for label, child in children.items():
        weights = getattr(child, 'weights', None)
        if isinstance(weights, dict):
            reserved = [v for v in weights
                        if v in (_PASSTHROUGH, _UNATTRIBUTED_VARIATION)]
            if reserved:
                raise ValueError(
                    f"variation label(s) {reserved} of strategy "
                    f"{label!r} are reserved"
                )
    return children, single


def _system_shares(
    rec: pd.DataFrame,
    labels: List[str],
    single: bool,
    eps: float,
    symbol: str,
) -> pd.DataFrame:
    """Per-bar strategy-level shares on the symbol's own record index.

    Orchestrator mode: proportional shares of the recorded
    ``weight_<label> × forecast_<label>`` contributions (the recorded
    weight is the renormalized weight actually used; it is 0.0 exactly
    when the forecast is NaN, so NaN→0 is consistent). Single-strategy
    mode: share 1.0 from the first non-NaN forecast onward (mirroring
    the forecast cache, which persists across later NaN bars); all-NaN
    before.
    """
    if single:
        if 'forecast' not in rec.columns:
            raise ValueError(
                f"records for {symbol!r} lack the 'forecast' column"
            )
        active = rec['forecast'].notna().cummax().to_numpy()
        return pd.DataFrame(
            {labels[0]: np.where(active, 1.0, np.nan)}, index=rec.index,
        )
    missing = [c for lab in labels
               for c in (f'weight_{lab}', f'forecast_{lab}')
               if c not in rec.columns]
    if missing:
        raise ValueError(
            f"records for {symbol!r} are missing orchestrator "
            f"columns: {missing}"
        )
    contrib = pd.DataFrame(
        {lab: rec[f'weight_{lab}'].to_numpy(dtype=float)
              * rec[f'forecast_{lab}'].to_numpy(dtype=float)
         for lab in labels},
        index=rec.index,
    )
    return _proportional_shares(contrib, eps)


def _variation_split(
    child: Any,
    symbol: str,
    attr_l: np.ndarray,
    idx: pd.Index,
    eps: float,
) -> tuple:
    """Split one strategy's attributed PnL across its variations.

    Conditional variation shares are computed only on bars where the
    child's own combined ``forecast`` is non-NaN, then forward-filled —
    mirroring the forecast cache the orchestrator actually consumed
    (per-column ffill would blend values from different bars). The
    already-attributed strategy PnL is multiplied by those shares, so
    variations sum to the strategy amount by construction. Returns
    ``({variation: values}, leftover_or_None)`` where ``leftover``
    carries any PnL on bars with no defined variation share (invariant
    breach with a well-formed system; routed to 'unattributed' so
    reconciliation survives duck-typed children regardless).
    """
    crec = (child.get_records(symbol)
            if callable(getattr(child, 'get_records', None))
            else pd.DataFrame())
    weights = getattr(child, 'weights', None)
    vlabels = list(weights) if isinstance(weights, dict) and weights else []
    usable = (
        bool(vlabels)
        and not crec.empty
        and 'forecast' in crec.columns
        and all(f'forecast_{v}' in crec.columns for v in vlabels)
    )
    if not usable:
        # No variation convention — one passthrough bucket.
        return {_PASSTHROUGH: attr_l}, None

    crec = crec[~crec.index.duplicated(keep='last')]
    vcon = pd.DataFrame(
        {v: float(weights[v]) * crec[f'forecast_{v}'].astype(float)
         for v in vlabels},
        index=crec.index,
    )
    valid = crec['forecast'].notna()
    vsh = _proportional_shares(vcon.where(valid), eps).ffill()
    vsh_np = vsh.shift(1).reindex(idx).ffill().to_numpy(dtype=float)

    leftover = None
    undefined = np.isnan(vsh_np).any(axis=1) & (attr_l != 0.0)
    if undefined.any():
        leftover = np.where(undefined, attr_l, 0.0)
        attr_l = np.where(undefined, 0.0, attr_l)

    split = vsh_np * attr_l[:, None]
    split[np.isnan(split)] = 0.0
    return {v: split[:, k] for k, v in enumerate(vlabels)}, leftover


def _proportional_shares(contrib: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Row-wise signed proportional shares of a contribution frame.

    Normal rows: ``c / Σc`` (signed, sums to 1). Near-cancellation rows
    (``|Σc| < eps·Σ|c|``): magnitude fallback ``|c| / Σ|c|`` (bounded,
    sums to 1). All-zero/NaN rows: all-NaN (the caller routes the
    following bar's PnL to 'unattributed'). NaN/inf contributions count
    as 0.
    """
    c = np.nan_to_num(contrib.to_numpy(dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    d = c.sum(axis=1)
    a = np.abs(c).sum(axis=1)
    out = np.full_like(c, np.nan)
    normal = (a > 0.0) & (np.abs(d) >= eps * a)
    fallback = (a > 0.0) & ~normal
    out[normal] = c[normal] / d[normal, None]
    out[fallback] = np.abs(c[fallback]) / a[fallback, None]
    return pd.DataFrame(out, index=contrib.index, columns=contrib.columns)


def _empty_long() -> pd.DataFrame:
    """Empty long frame with the fixed column schema."""
    return pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
        'symbol': pd.Categorical([]),
        'strategy': pd.Categorical([]),
        'variation': pd.Categorical([]),
        'pnl': pd.Series(dtype=float),
    })
