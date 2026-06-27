"""Reverse-DCF implied-CAGR heatmap for single-name equity research.

A *forward* DCF assumes a growth rate and produces a fair value. This
helper does the inverse: it takes the **current market price as given**
and solves for the revenue CAGR the price implies, then sweeps two
chosen model inputs across user-supplied grids to show how that implied
CAGR responds to the modelling assumptions. The output is a 2-D heatmap
(``pd.DataFrame``): index = sensitivity variable A, columns =
sensitivity variable B, each cell = the solved CAGR **in percentage
points** (e.g. ``38.9`` means 38.9 %). The cell at the caller's central
("consensus") assumptions is the CAGR the market is pricing in today —
read off ``ReverseDCFResult.consensus_cagr`` to judge whether that
growth is realistic.

Input format (units)
--------------------
* **Rate inputs are fractions, not whole-number percents**:
  ``wacc=0.12`` (12 %), ``terminal_growth=0.02`` (2 %), ``fcf_margin=0.08``
  (8 %), ``cagr_bounds=(-0.5, 1.0)`` (−50 %…+100 %). Passing ``12`` for a
  12 % WACC is read as 1200 % and the cell will not solve (→ ``NaN``); a
  guardrail warns when rate values look like percents.
* **Dollar inputs are absolute amounts** in one consistent currency:
  ``market_price`` (per share), ``base_revenue``, ``base_fcf``,
  ``net_debt`` (company totals; ``net_debt`` negative = net cash).
* ``shares_outstanding`` is a raw share count; ``horizon`` is an integer
  number of years.
* When an input is swept, its grid values carry that input's unit (so a
  ``'wacc'`` axis takes fractions, a ``'horizon'`` axis takes integers).
* **Output**: the heatmap's cell values and ``consensus_cagr`` are CAGR in
  percentage points; a rate-like **axis** (``wacc`` / ``terminal_growth``
  / ``fcf_margin`` index or column) is *labelled* in percentage points
  too (``0.09`` → ``9.0``), so the whole table reads in percent, while
  dollar / count / ``horizon`` axes keep their native units. Everything
  *fed in* stays a fraction.

Model (two-stage FCFF → Enterprise Value → equity)
--------------------------------------------------
For a candidate revenue CAGR ``g`` over an ``N``-year explicit horizon::

    revenue_t = base_revenue * (1+g)**t          (t = 1..N)
    fcf_t     = revenue_t * fcf_margin           (FCF = revenue × margin)
    pv_explicit = Σ fcf_t / (1+wacc)**t

    fcf_N       = base_revenue*(1+g)**N * fcf_margin
    tv_N        = fcf_N * (1+terminal_growth) / (wacc - terminal_growth)   # Gordon
    pv_terminal = tv_N / (1+wacc)**N

    enterprise_value = pv_explicit + pv_terminal
    equity_value     = enterprise_value - net_debt
    implied_price    = equity_value / shares_outstanding

``implied_price(g)`` is monotonically increasing in ``g`` (the cash
flows grow), so the reverse solve ``implied_price(g) − market_price = 0``
has a unique root, found with ``scipy.optimize.brentq`` over
``cagr_bounds``. Holding the market price fixed, a *higher* WACC needs a
*higher* implied CAGR to justify it, while a *higher* FCF margin needs a
*lower* one.

This is a one-shot research helper (call from a notebook, not per bar).
It is a pure calculator — every input is supplied by the caller; nothing
is fetched (the repo has no fundamental-data source).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

__all__ = ['reverse_dcf_cagr', 'ReverseDCFResult']

logger = logging.getLogger(__name__)

# The scalar model inputs that may be chosen as a sensitivity axis. ``cagr``
# is the solved output and can never be an axis.
_ALLOWED_AXES: Tuple[str, ...] = (
    'market_price',
    'shares_outstanding',
    'base_revenue',
    'fcf_margin',
    'wacc',
    'terminal_growth',
    'horizon',
    'net_debt',
)

_NAN = float('nan')
_HIGHLIGHT_CSS = 'background-color: #ffd54f; color: red; font-weight: bold;'

# Rate-like inputs expected as fractions (e.g. 0.08 = 8 %). A value with
# absolute magnitude > 1 is almost certainly a whole-number percent typo
# (e.g. 8 for 8 %) — the guardrail warns rather than silently NaN-ing.
# A rate-like axis is also *labelled* in percentage points on the output
# heatmap (matching the cell values), so the whole table reads in percent.
_RATE_LIKE_PARAMS: Tuple[str, ...] = ('wacc', 'terminal_growth', 'fcf_margin')

# Decimal places when converting a rate fraction to a percent axis label —
# purely cosmetic (kills float noise like 9.000000000000002 → 9.0). The
# same transform feeds consensus_cell, so heatmap.loc stays exact.
_PCT_DISPLAY_DECIMALS = 10


@dataclass(frozen=True)
class ReverseDCFResult:
    """Result of :func:`reverse_dcf_cagr` — the heatmap plus the consensus cell.

    Attributes
    ----------
    heatmap
        2-D ``pd.DataFrame`` of solved revenue CAGRs **in percentage
        points** (``38.9`` means 38.9 %). The index holds the ``axis_a``
        grid values (``index.name == axis_a``); the columns hold the
        ``axis_b`` grid values (``columns.name == axis_b``) — a rate-like
        axis (``wacc`` / ``terminal_growth`` / ``fcf_margin``) is labelled
        in percentage points (``0.09`` → ``9.0``), other axes in their
        native unit. A cell is ``NaN`` where no solution exists (see
        :func:`reverse_dcf_cagr`).
    axis_a, axis_b
        The names of the swept inputs (the index / column dimensions).
    consensus_cell
        ``(axis_a_value, axis_b_value)`` — the caller's central
        assumptions for the two swept inputs, in the **same (display)
        unit as the heatmap labels** (percentage points for a rate-like
        axis, native otherwise), so ``heatmap.loc[consensus_cell]`` is
        exact. Always present in the grids.
    consensus_cagr
        ``heatmap.loc[consensus_cell]`` — the CAGR the market is pricing
        in under the consensus assumptions, in percentage points
        (``NaN`` if that cell has no solution).
    name
        Optional company label passed to :func:`reverse_dcf_cagr` (e.g.
        a ticker or company name), echoed in the ``styled()`` caption so
        results are distinguishable when several companies are analyzed.
        ``None`` when not supplied.
    """

    heatmap: pd.DataFrame
    axis_a: str
    axis_b: str
    consensus_cell: Tuple[float, float]
    consensus_cagr: float
    name: Optional[str] = None

    def styled(self) -> "Styler":
        """Return a ``pandas`` ``Styler`` for notebook display.

        Renders every cell as a percentage with a ``%`` suffix (the
        heatmap already holds percentage points; ``NaN`` shown as ``—``),
        appends ``%`` to a rate-like axis's labels (``wacc`` /
        ``terminal_growth`` / ``fcf_margin``), and highlights the
        consensus cell in bold red. The caption is prefixed with ``name``
        when one was supplied. The underlying frame is unchanged and
        reachable via ``.data``.
        """
        a_val, b_val = self.consensus_cell

        def _highlight(frame: pd.DataFrame) -> pd.DataFrame:
            css = pd.DataFrame('', index=frame.index, columns=frame.columns)
            css.loc[a_val, b_val] = _HIGHLIGHT_CSS
            return css

        def _fmt_axis(name: str, val) -> str:
            return f"{val:.1f}%" if name in _RATE_LIKE_PARAMS else f"{val}"

        prefix = f"{self.name} — " if self.name else ""
        consensus = (
            "no solution"
            if math.isnan(self.consensus_cagr)
            else f"{self.consensus_cagr:.1f}%"
        )
        caption = (
            f"{prefix}Implied revenue CAGR — consensus at "
            f"{self.axis_a}={_fmt_axis(self.axis_a, a_val)}, "
            f"{self.axis_b}={_fmt_axis(self.axis_b, b_val)} → {consensus}"
        )

        styler = (
            self.heatmap.style
            .format('{:.1f}%', na_rep='—')
            .apply(_highlight, axis=None)
            .set_caption(caption)
        )
        # Rate-like axes already carry percent-point labels — append '%'.
        if self.axis_a in _RATE_LIKE_PARAMS:
            styler = styler.format_index('{:.1f}%', axis=0)
        if self.axis_b in _RATE_LIKE_PARAMS:
            styler = styler.format_index('{:.1f}%', axis=1)
        return styler


def reverse_dcf_cagr(
    *,
    name: Optional[str] = None,
    market_price: float,
    shares_outstanding: float,
    base_revenue: float,
    fcf_margin: Optional[float] = None,
    base_fcf: Optional[float] = None,
    wacc: float,
    terminal_growth: float,
    horizon: int,
    net_debt: float = 0.0,
    axis_a: str = 'wacc',
    axis_a_values: Sequence[float],
    axis_b: str = 'fcf_margin',
    axis_b_values: Sequence[float],
    cagr_bounds: Tuple[float, float] = (-0.50, 1.00),
) -> ReverseDCFResult:
    """Solve the revenue CAGR implied by the market price, over a 2-D grid.

    Runs the reverse two-stage FCFF DCF (see the module docstring) at
    every combination of the two swept inputs. The scalars passed for
    ``axis_a`` / ``axis_b`` are the consensus assumptions; each is
    inserted into its grid if absent, so the consensus cell always
    exists.

    Parameters
    ----------
    name
        Optional company label (ticker or name) for display. Stored on
        the result and prefixed to the ``styled()`` caption so several
        companies' heatmaps stay distinguishable. ``None`` (default)
        omits it. Must be a string when provided.
    market_price
        Current price per share (the target the reverse solve matches).
        Must be ``> 0``.
    shares_outstanding
        Diluted share count for the equity-value-to-price bridge. ``> 0``.
    base_revenue
        Latest-period (year-0) revenue, the base the projection grows
        from. ``> 0``.
    fcf_margin
        Free-cash-flow margin applied to projected revenue
        (``FCF = revenue × fcf_margin``). Provide **exactly one** of
        ``fcf_margin`` or ``base_fcf``. Must be non-zero.
    base_fcf
        Latest-period (year-0) **dollar** free cash flow — a convenient
        alternative to ``fcf_margin`` when the raw revenue and FCF
        figures are easier to source than a pre-computed margin. The
        margin is derived once as ``base_fcf / base_revenue`` (the
        consensus margin) and then held as the structural margin
        assumption, so sweeping ``base_revenue`` scales FCF while keeping
        the margin constant. Same currency units as ``base_revenue``;
        must be non-zero. Provide **exactly one** of ``base_fcf`` or
        ``fcf_margin``. Not itself a sweep axis — sweep ``fcf_margin``
        (or ``base_revenue``) instead.
    wacc
        Weighted average cost of capital (the discount rate), as a
        fraction. Must satisfy ``0 < wacc < 1`` and ``wacc >
        terminal_growth`` (Gordon-growth requirement).
    terminal_growth
        Perpetuity growth rate of FCF beyond the horizon, as a fraction.
        Must be ``< wacc``.
    horizon
        Number of explicit forecast years ``N``. Integer ``>= 1``.
    net_debt
        Total debt minus cash, in the same currency units as
        ``base_revenue`` × ``fcf_margin``. Negative means net cash.
        Default ``0.0``.
    axis_a, axis_b
        Names of the two inputs to sweep (the index and column
        dimensions). Each must be one of ``market_price``,
        ``shares_outstanding``, ``base_revenue``, ``fcf_margin``,
        ``wacc``, ``terminal_growth``, ``horizon``, ``net_debt``, and
        the two must differ. Default ``'wacc'`` × ``'fcf_margin'``.
    axis_a_values, axis_b_values
        The grid values for each axis (non-empty), **in the unit of the
        swept input** — fractions for a rate axis (``wacc`` /
        ``terminal_growth`` / ``fcf_margin``), integers ``>= 1`` for a
        ``'horizon'`` axis, currency for a dollar axis. The corresponding
        consensus scalar is inserted if missing, and each grid is sorted
        ascending and de-duplicated. A ``UserWarning`` is emitted if a
        rate axis (or a rate scalar) carries a value with magnitude
        ``> 1`` — almost always a whole-number-percent typo (e.g. ``8``
        for 8 %), which would silently produce ``NaN`` cells.
    cagr_bounds
        ``(low, high)`` bracket for the root search, as CAGR **fractions**
        (not percents). Default ``(-0.50, 1.00)`` (−50 % to +100 %). A
        cell whose implied CAGR falls outside this bracket is ``NaN`` —
        widen the bounds to resolve it. Must have ``low < high``.

    Returns
    -------
    ReverseDCFResult
        The heatmap DataFrame (cell values = implied CAGR in **percentage
        points**, e.g. ``38.9``; a rate-like axis's index/column labels —
        ``wacc`` / ``terminal_growth`` / ``fcf_margin`` — are likewise in
        percentage points, so the table reads in percent end-to-end), the
        axis names, and the consensus cell / CAGR (``consensus_cell`` is
        in those same display units). A cell is ``NaN`` when the per-cell
        configuration
        has no Gordon-growth solution (``wacc <= terminal_growth``), the
        implied CAGR lies outside ``cagr_bounds``, or the per-cell math
        fails numerically. Up-front (consensus) parameters are validated
        and raise; per-cell invalid configurations degrade to ``NaN`` so
        the rest of the heatmap still renders.

    Warns
    -----
    UserWarning
        If a rate input (``wacc``, ``terminal_growth``, ``fcf_margin`` —
        whether a scalar or a swept grid value) has magnitude ``> 1``,
        which almost always means whole-number percents were passed where
        fractions are expected.

    Raises
    ------
    TypeError
        If any axis grid contains a non-numeric value, or ``name`` is
        not a string.
    ValueError
        If ``market_price``/``shares_outstanding``/``base_revenue`` are
        not ``> 0``; neither or both of ``fcf_margin`` / ``base_fcf``
        are supplied; ``fcf_margin`` (or ``base_fcf``) is zero; ``wacc``
        is
        outside ``(0, 1)``; ``terminal_growth >= wacc``; ``horizon`` is
        not an integer ``>= 1``; ``axis_a``/``axis_b`` are not allowed
        names or are equal; an axis grid is empty; a ``'horizon'`` grid
        holds a non-integer or value ``< 1``; or ``cagr_bounds`` is not
        ``low < high``.
    """
    # ── Validate consensus (base) parameters ──────────────────────────
    if name is not None and not isinstance(name, str):
        raise TypeError(f"name must be a string or None, got {type(name).__name__}")
    if not _is_real(market_price) or market_price <= 0:
        raise ValueError(f"market_price must be > 0, got {market_price!r}")
    if not _is_real(shares_outstanding) or shares_outstanding <= 0:
        raise ValueError(
            f"shares_outstanding must be > 0, got {shares_outstanding!r}"
        )
    if not _is_real(base_revenue) or base_revenue <= 0:
        raise ValueError(f"base_revenue must be > 0, got {base_revenue!r}")
    # Accept either an explicit margin or a dollar FCF figure
    # (margin = base_fcf / base_revenue) — whichever is easier to source.
    fcf_margin = _resolve_fcf_margin(fcf_margin, base_fcf, base_revenue)
    if not _is_real(wacc) or not (0.0 < wacc < 1.0):
        raise ValueError(f"wacc must be in (0, 1), got {wacc!r}")
    if not _is_real(terminal_growth):
        raise ValueError(
            f"terminal_growth must be a real number, got {terminal_growth!r}"
        )
    if terminal_growth >= wacc:
        raise ValueError(
            f"terminal_growth must be < wacc (Gordon growth), got "
            f"terminal_growth={terminal_growth!r}, wacc={wacc!r}"
        )
    if not _is_integer(horizon) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    if not _is_real(net_debt):
        raise ValueError(f"net_debt must be a real number, got {net_debt!r}")

    if axis_a not in _ALLOWED_AXES:
        raise ValueError(
            f"axis_a must be one of {list(_ALLOWED_AXES)}, got {axis_a!r}"
        )
    if axis_b not in _ALLOWED_AXES:
        raise ValueError(
            f"axis_b must be one of {list(_ALLOWED_AXES)}, got {axis_b!r}"
        )
    if axis_a == axis_b:
        raise ValueError(f"axis_a and axis_b must differ, both are {axis_a!r}")

    lo, hi = cagr_bounds
    if not (_is_real(lo) and _is_real(hi)) or lo >= hi:
        raise ValueError(
            f"cagr_bounds must be (low, high) with low < high, got {cagr_bounds!r}"
        )

    base = {
        'market_price': float(market_price),
        'shares_outstanding': float(shares_outstanding),
        'base_revenue': float(base_revenue),
        'fcf_margin': float(fcf_margin),
        'wacc': float(wacc),
        'terminal_growth': float(terminal_growth),
        'horizon': int(horizon),
        'net_debt': float(net_debt),
    }

    a_vals = _prepare_axis(axis_a, axis_a_values, base[axis_a])
    b_vals = _prepare_axis(axis_b, axis_b_values, base[axis_b])

    _warn_if_percent_like(axis_a, a_vals, axis_b, b_vals, base)

    # ── Solve every cell ──────────────────────────────────────────────
    data = np.full((len(a_vals), len(b_vals)), _NAN, dtype=float)
    nan_count = 0
    bracket = (float(lo), float(hi))
    for (i, a_val), (j, b_val) in product(enumerate(a_vals), enumerate(b_vals)):
        params = dict(base)
        params[axis_a] = a_val
        params[axis_b] = b_val
        cagr = _solve_cell(params, bracket)
        if math.isnan(cagr):
            nan_count += 1
        data[i, j] = cagr

    # Present the implied CAGR in percentage points (38.9 rather than
    # 0.389) — far easier to read; the solve itself works in fractions.
    # A rate-like axis is relabelled to percent too (so the whole table
    # reads in percent); consensus_cell uses the identical transform so
    # heatmap.loc lookups stay exact.
    heatmap = pd.DataFrame(
        data * 100.0,
        index=pd.Index([_to_display(axis_a, v) for v in a_vals], name=axis_a),
        columns=pd.Index([_to_display(axis_b, v) for v in b_vals], name=axis_b),
    )

    consensus_cell = (
        _to_display(axis_a, base[axis_a]),
        _to_display(axis_b, base[axis_b]),
    )
    consensus_cagr = float(heatmap.loc[consensus_cell[0], consensus_cell[1]])

    if nan_count:
        logger.debug(
            "reverse_dcf_cagr: %d/%d cells had no solution (NaN) for "
            "axis_a=%s × axis_b=%s",
            nan_count,
            data.size,
            axis_a,
            axis_b,
        )

    return ReverseDCFResult(
        heatmap=heatmap,
        axis_a=axis_a,
        axis_b=axis_b,
        consensus_cell=consensus_cell,
        consensus_cagr=consensus_cagr,
        name=name,
    )


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────

def _to_display(axis_name: str, value: float):
    """Map an axis value to its display unit for heatmap labelling.

    A rate-like axis (``wacc`` / ``terminal_growth`` / ``fcf_margin``) is
    converted to percentage points (``×100``, rounded to kill float noise)
    so its index/column labels read in percent, matching the cell values.
    Every other axis (``horizon``, dollar amounts, share counts) passes
    through unchanged. Applied identically to grid labels and
    ``consensus_cell``, so ``heatmap.loc`` lookups remain exact.
    """
    if axis_name in _RATE_LIKE_PARAMS:
        return round(value * 100.0, _PCT_DISPLAY_DECIMALS)
    return value


def _warn_if_percent_like(
    axis_a: str,
    a_vals: Sequence[float],
    axis_b: str,
    b_vals: Sequence[float],
    base: dict,
) -> None:
    """Emit a ``UserWarning`` if a rate-like input looks like a percent typo.

    For each of ``wacc`` / ``terminal_growth`` / ``fcf_margin``, gathers
    the values it will actually take — the prepared grid when it is a
    swept axis, otherwise its single consensus scalar — and flags any
    with magnitude ``> 1`` (e.g. ``8`` meant as 8 %). These would size
    the DCF with 800 %-style rates and silently yield ``NaN`` cells, so
    warning loudly turns a confusing empty heatmap into an actionable
    message. Does not raise — an extreme-but-deliberate rate stays legal.
    """
    swept = {axis_a: a_vals, axis_b: b_vals}
    offenders = {}
    for param in _RATE_LIKE_PARAMS:
        values = swept.get(param, [base[param]])
        bad = [v for v in values if abs(v) > 1.0]
        if bad:
            offenders[param] = bad
    if offenders:
        detail = '; '.join(f"{p}={vals}" for p, vals in offenders.items())
        warnings.warn(
            f"reverse_dcf_cagr: {detail} — rate inputs (wacc, "
            f"terminal_growth, fcf_margin) must be fractions, not "
            f"whole-number percents (e.g. 0.08 for 8%, not 8). Values with "
            f"magnitude > 1 will not solve and appear as NaN.",
            UserWarning,
            stacklevel=3,
        )


def _is_real(x: object) -> bool:
    """True for a real (non-bool) Python/NumPy number."""
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(
        x, bool
    )


def _is_integer(x: object) -> bool:
    """True for a non-bool Python/NumPy integer."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _resolve_fcf_margin(
    fcf_margin: Optional[float],
    base_fcf: Optional[float],
    base_revenue: float,
) -> float:
    """Resolve the consensus FCF margin from ``fcf_margin`` xor ``base_fcf``.

    Exactly one of the two must be supplied. When ``base_fcf`` (a dollar
    figure) is given, the margin is derived once as
    ``base_fcf / base_revenue`` — convenient because raw revenue and FCF
    are easier to source than a pre-computed ratio. ``base_revenue`` has
    already been validated ``> 0`` by the caller, so the division is
    safe. Raises ``ValueError`` if neither or both are given, or if the
    resulting margin would be zero / non-real.
    """
    if (fcf_margin is None) == (base_fcf is None):
        raise ValueError(
            "provide exactly one of fcf_margin or base_fcf, got "
            f"fcf_margin={fcf_margin!r}, base_fcf={base_fcf!r}"
        )
    if base_fcf is not None:
        if not _is_real(base_fcf):
            raise ValueError(f"base_fcf must be a real number, got {base_fcf!r}")
        if base_fcf == 0:
            raise ValueError(
                f"base_fcf must be non-zero (derived margin would be 0), "
                f"got {base_fcf!r}"
            )
        return float(base_fcf) / float(base_revenue)
    if not _is_real(fcf_margin) or fcf_margin == 0:
        raise ValueError(f"fcf_margin must be non-zero, got {fcf_margin!r}")
    return float(fcf_margin)


def _prepare_axis(name: str, values: Sequence[float], base_value: float) -> list:
    """Validate, coerce, sort and de-duplicate one axis grid.

    Ensures ``base_value`` (the consensus scalar) is present so the
    consensus cell always exists. ``'horizon'`` grids are integer-typed
    (values ``>= 1``); every other axis is float-typed. Raises
    ``TypeError`` on a non-numeric value, ``ValueError`` on an empty grid
    or an out-of-domain ``'horizon'`` value.
    """
    seq = list(values)
    if not seq:
        raise ValueError(f"{name} grid (axis values) must be non-empty")

    if name == 'horizon':
        out = []
        for v in seq:
            if not _is_integer(v):
                raise ValueError(
                    f"horizon grid values must be integers >= 1, got {v!r}"
                )
            iv = int(v)
            if iv < 1:
                raise ValueError(
                    f"horizon grid values must be integers >= 1, got {iv!r}"
                )
            out.append(iv)
        out.append(int(base_value))
        return sorted(set(out))

    out = []
    for v in seq:
        if not _is_real(v):
            raise TypeError(
                f"{name} grid values must be real numbers, got {type(v).__name__}"
            )
        out.append(float(v))
    out.append(float(base_value))
    return sorted(set(out))


def _implied_price(g: float, params: dict) -> float:
    """Forward two-stage FCFF implied price per share for revenue CAGR ``g``.

    Reads the model inputs from ``params``. Assumes the Gordon-growth
    guard (``wacc > terminal_growth``) has already been checked by the
    caller.
    """
    base_revenue = params['base_revenue']
    fcf_margin = params['fcf_margin']
    wacc = params['wacc']
    terminal_growth = params['terminal_growth']
    n = int(params['horizon'])
    net_debt = params['net_debt']
    shares = params['shares_outstanding']

    one_plus_w = 1.0 + wacc
    one_plus_g = 1.0 + g

    pv_explicit = 0.0
    for t in range(1, n + 1):
        fcf_t = base_revenue * one_plus_g ** t * fcf_margin
        pv_explicit += fcf_t / one_plus_w ** t

    fcf_n = base_revenue * one_plus_g ** n * fcf_margin
    tv_n = fcf_n * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = tv_n / one_plus_w ** n

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    return equity_value / shares


def _solve_cell(params: dict, bracket: Tuple[float, float]) -> float:
    """Root-find the implied revenue CAGR for one cell, or ``NaN``.

    Returns ``NaN`` when the Gordon-growth guard fails
    (``wacc <= terminal_growth``), the root is not bracketed within
    ``bracket`` (implied CAGR outside the search window), or the per-cell
    math fails numerically.
    """
    if params['wacc'] <= params['terminal_growth']:
        return _NAN

    target = params['market_price']
    lo, hi = bracket

    def f(g: float) -> float:
        return _implied_price(g, params) - target

    try:
        f_lo = f(lo)
        f_hi = f(hi)
    except (ZeroDivisionError, OverflowError, FloatingPointError, ValueError):
        return _NAN
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return _NAN
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0.0) == (f_hi > 0.0):  # same sign → no root in the bracket
        return _NAN

    try:
        return float(brentq(f, lo, hi))
    except (ValueError, RuntimeError):  # pragma: no cover - bracket re-checked above
        return _NAN
