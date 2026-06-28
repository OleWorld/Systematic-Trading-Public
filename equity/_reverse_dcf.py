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

Business segments (one "swing" segment)
---------------------------------------
A real company is a mix of segments with different growth and margins.
Pass ``segments=[Segment(...), ...]`` to model them: ``base_revenue`` is
then the company **total**, each ``Segment`` carries a ``revenue_share``
(fractions summing to 1), its own ``fcf_margin``, and a pinned annualized
``growth`` — **except exactly one** segment whose ``growth=None`` marks it
the **swing**. A single market price is one equation, so the reverse-solve
recovers exactly one unknown: the swing segment's growth, holding every
other segment fixed. So for a company that is 10 % fast-growing AI + 90 %
mature mobile (growing ~2.5 %), you reverse-solve **the AI segment's
implied growth**. Omitting ``segments`` is the single-segment default —
the one implicit segment is the swing and the math is identical to the
homogeneous-company case (one ``base_revenue`` × one ``fcf_margin``).

Growth is a single annualized rate per segment (a per-year trajectory
would just annualize back to this rate).

Input format (units)
--------------------
* **Rate inputs are fractions, not whole-number percents**:
  ``wacc=0.12`` (12 %), ``terminal_growth=0.02`` (2 %), ``fcf_margin=0.08``
  (8 %), a segment ``growth=0.30`` (30 %), ``revenue_share=0.10`` (10 %),
  ``cagr_bounds=(-0.5, 1.0)`` (−50 %…+100 %). Passing ``12`` for a 12 %
  WACC is read as 1200 % and the cell will not solve (→ ``NaN``); a
  guardrail warns when rate values look like percents.
* **Dollar inputs are absolute amounts** in one consistent currency:
  ``market_price`` (per share), ``base_revenue`` (company total),
  ``base_fcf``, ``net_debt`` (company totals; ``net_debt`` negative = net
  cash).
* ``shares_outstanding`` is a raw share count; ``horizon`` is an integer
  number of years.
* When an input is swept, its grid values carry that input's unit (so a
  ``'wacc'`` axis takes fractions, a ``'horizon'`` axis takes integers, an
  ``'AI.fcf_margin'`` axis takes margin fractions).
* **Output**: the heatmap's cell values and ``consensus_cagr`` are CAGR in
  percentage points; a rate-like **axis** (``wacc`` / ``terminal_growth``
  or any segment ``fcf_margin`` / ``growth`` / ``revenue_share``) is
  *labelled* in percentage points too (``0.09`` → ``9.0``), so the whole
  table reads in percent, while dollar / count / ``horizon`` axes keep
  their native units. Everything *fed in* stays a fraction.

Sweep axes
----------
``axis_a`` / ``axis_b`` name the two swept inputs. Global inputs are bare
strings (``'market_price'``, ``'shares_outstanding'``, ``'base_revenue'``,
``'wacc'``, ``'terminal_growth'``, ``'horizon'``, ``'net_debt'``);
segment-level inputs use dotted ``'<segment>.<field>'`` strings with
``field`` ∈ ``{fcf_margin, growth, revenue_share}`` (e.g.
``'AI.fcf_margin'``, ``'mobile.growth'``, ``'AI.revenue_share'``). Bare
``'fcf_margin'`` is valid only in single-segment mode; the swing segment's
``growth`` cannot be an axis (it is the solved output). Sweeping a
``revenue_share`` shifts the **mix**: the swept segment takes the value and
the other segments rescale proportionally so the total stays
``base_revenue`` (see ``_apply_revenue_shares``).

Model (two-stage FCFF → Enterprise Value → equity)
--------------------------------------------------
For a candidate swing-segment revenue CAGR ``g`` over an ``N``-year
explicit horizon, summing across segments (the swing uses ``g``, the rest
their pinned growth)::

    revenue_{s,t} = base_revenue * share_s * (1+growth_s)**t      (t = 1..N)
    fcf_t         = Σ_s revenue_{s,t} * fcf_margin_s
    pv_explicit   = Σ_t fcf_t / (1+wacc)**t

    fcf_N       = Σ_s base_revenue*share_s*(1+growth_s)**N * fcf_margin_s
    tv_N        = fcf_N * (1+terminal_growth) / (wacc - terminal_growth)  # Gordon
    pv_terminal = tv_N / (1+wacc)**N

    enterprise_value = pv_explicit + pv_terminal
    equity_value     = enterprise_value - net_debt
    implied_price    = equity_value / shares_outstanding

All segments revert to ``terminal_growth`` after the horizon (one blended
Gordon terminal value — the standard two-stage assumption).
``implied_price(g)`` is monotone in ``g`` (only the swing's cash flows move
with ``g``; the fixed segments are constant), so the reverse solve
``implied_price(g) − market_price = 0`` has a unique root, found with
``scipy.optimize.brentq`` over ``cagr_bounds``. Holding the market price
fixed, a *higher* WACC needs a *higher* implied CAGR to justify it, while a
*higher* swing FCF margin needs a *lower* one.

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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

__all__ = ['reverse_dcf_cagr', 'ReverseDCFResult', 'Segment']

logger = logging.getLogger(__name__)

# Global (company-level) inputs that may be chosen as a sensitivity axis by
# their bare name. ``cagr`` is the solved output and can never be an axis;
# ``fcf_margin`` is a *segment* field — valid bare only in single-segment
# mode, otherwise addressed as ``'<segment>.fcf_margin'``.
_GLOBAL_AXES: Tuple[str, ...] = (
    'market_price',
    'shares_outstanding',
    'base_revenue',
    'wacc',
    'terminal_growth',
    'horizon',
    'net_debt',
)

# Per-segment fields addressable as a dotted ``'<segment>.<field>'`` axis.
_SEGMENT_FIELDS: Tuple[str, ...] = ('fcf_margin', 'growth', 'revenue_share')

# Names a segment may not take (would make a bare axis ambiguous).
_RESERVED_SEGMENT_NAMES = frozenset(_GLOBAL_AXES) | {'fcf_margin', 'cagr'}

_NAN = float('nan')
_HIGHLIGHT_CSS = 'background-color: #ffd54f; color: red; font-weight: bold;'

# Rate-like fields expected as fractions (e.g. 0.08 = 8 %). A value with
# absolute magnitude > 1 is almost certainly a whole-number percent typo
# (e.g. 8 for 8 %) — the guardrail warns rather than silently NaN-ing. A
# rate-like axis is also *labelled* in percentage points on the output
# heatmap (matching the cell values), so the whole table reads in percent.
# Keyed by the axis's field suffix so dotted segment axes inherit it.
_RATE_LIKE_FIELDS: Tuple[str, ...] = (
    'wacc',
    'terminal_growth',
    'fcf_margin',
    'growth',
    'revenue_share',
)

# Decimal places when converting a rate fraction to a percent axis label —
# purely cosmetic (kills float noise like 9.000000000000002 → 9.0). The
# same transform feeds consensus_cell, so heatmap.loc stays exact.
_PCT_DISPLAY_DECIMALS = 10

# Shares must sum to 1.0 within this tolerance; also the floor below which a
# segment's remaining share is treated as having no room (see renormalize).
_SHARE_TOL = 1e-9


@dataclass(frozen=True)
class Segment:
    """One business segment of a multi-segment reverse DCF.

    Pass a list of these as ``reverse_dcf_cagr(segments=...)`` to model a
    company as a mix of streams with different growth and margins. Exactly
    one segment must be the **swing** (``growth=None``) — its annualized
    growth is the reverse-solved output; every other segment's ``growth``
    is a pinned caller assumption.

    Attributes
    ----------
    name
        Unique segment label, used in dotted sweep axes
        (``'<name>.fcf_margin'`` etc.) and the result caption. Must be a
        non-empty string, may not contain ``'.'``, and may not collide with
        a reserved global axis name (``wacc``, ``net_debt``, …, or
        ``fcf_margin`` / ``cagr``).
    revenue_share
        This segment's fraction of company ``base_revenue``, in ``(0, 1]``.
        Shares across all segments must sum to ``1.0``.
    fcf_margin
        Free-cash-flow margin applied to this segment's projected revenue
        (``FCF = revenue × fcf_margin``), as a fraction. Any real value for
        a fixed segment (a loss-making segment may be negative); the swing
        segment's margin must be **non-zero** (otherwise its growth would
        not move the price and the solve is undefined).
    growth
        Pinned annualized revenue growth for a fixed segment, as a fraction
        (``0.025`` = 2.5 %). ``None`` marks this segment the **swing** (its
        growth is solved). Exactly one segment in the list may be ``None``.
    """

    name: str
    revenue_share: float
    fcf_margin: float
    growth: Optional[float] = None


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
        axis (``wacc`` / ``terminal_growth`` or any segment ``fcf_margin`` /
        ``growth`` / ``revenue_share``) is labelled in percentage points
        (``0.09`` → ``9.0``), other axes in their native unit. A cell is
        ``NaN`` where no solution exists (see :func:`reverse_dcf_cagr`).
    axis_a, axis_b
        The names of the swept inputs (the index / column dimensions).
    consensus_cell
        ``(axis_a_value, axis_b_value)`` — the caller's central
        assumptions for the two swept inputs, in the **same (display)
        unit as the heatmap labels** (percentage points for a rate-like
        axis, native otherwise), so ``heatmap.loc[consensus_cell]`` is
        exact. Always present in the grids.
    consensus_cagr
        ``heatmap.loc[consensus_cell]`` — the implied growth the market is
        pricing in under the consensus assumptions, in percentage points
        (``NaN`` if that cell has no solution). In multi-segment mode this
        is the **swing segment's** implied growth.
    swing_segment
        Name of the swing segment whose growth was reverse-solved, or
        ``None`` in single-segment mode. Echoed in the ``styled()`` caption
        ("Implied <name> growth — …").
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
    swing_segment: Optional[str] = None
    name: Optional[str] = None

    def styled(self) -> "Styler":
        """Return a ``pandas`` ``Styler`` for notebook display.

        Renders every cell as a percentage with a ``%`` suffix (the
        heatmap already holds percentage points; ``NaN`` shown as ``—``),
        appends ``%`` to a rate-like axis's labels (``wacc`` /
        ``terminal_growth`` / any segment ``fcf_margin`` / ``growth`` /
        ``revenue_share``), and highlights the consensus cell in bold red.
        The caption names the swing segment ("Implied <swing> growth …")
        and is prefixed with ``name`` when one was supplied. The underlying
        frame is unchanged and reachable via ``.data``.
        """
        a_val, b_val = self.consensus_cell

        def _highlight(frame: pd.DataFrame) -> pd.DataFrame:
            css = pd.DataFrame('', index=frame.index, columns=frame.columns)
            css.loc[a_val, b_val] = _HIGHLIGHT_CSS
            return css

        def _fmt_axis(name: str, val) -> str:
            return f"{val:.1f}%" if _is_rate_like_axis(name) else f"{val}"

        prefix = f"{self.name} — " if self.name else ""
        metric = (
            f"{self.swing_segment} growth" if self.swing_segment else "revenue CAGR"
        )
        consensus = (
            "no solution"
            if math.isnan(self.consensus_cagr)
            else f"{self.consensus_cagr:.1f}%"
        )
        caption = (
            f"{prefix}Implied {metric} — consensus at "
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
        if _is_rate_like_axis(self.axis_a):
            styler = styler.format_index('{:.1f}%', axis=0)
        if _is_rate_like_axis(self.axis_b):
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
    segments: Optional[Sequence[Segment]] = None,
    axis_a: str = 'wacc',
    axis_a_values: Sequence[float],
    axis_b: str = 'fcf_margin',
    axis_b_values: Sequence[float],
    cagr_bounds: Tuple[float, float] = (-0.50, 1.00),
) -> ReverseDCFResult:
    """Solve the implied revenue CAGR from the market price, over a 2-D grid.

    Runs the reverse two-stage FCFF DCF (see the module docstring) at
    every combination of the two swept inputs. The scalars passed for
    ``axis_a`` / ``axis_b`` are the consensus assumptions; each is
    inserted into its grid if absent, so the consensus cell always
    exists. In multi-segment mode the solved value is the **swing**
    segment's growth (the segment with ``growth=None``), holding every
    other segment fixed.

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
        from. The **company total** (split across segments by their
        ``revenue_share``). ``> 0``.
    fcf_margin
        Free-cash-flow margin applied to projected revenue
        (``FCF = revenue × fcf_margin``). **Single-segment mode only** —
        provide **exactly one** of ``fcf_margin`` or ``base_fcf`` and leave
        ``segments`` unset. Must be non-zero. In multi-segment mode each
        ``Segment`` carries its own margin and this must be omitted.
    base_fcf
        Latest-period (year-0) **dollar** free cash flow — a single-segment
        convenience alternative to ``fcf_margin`` (margin derived as
        ``base_fcf / base_revenue``). Same currency units as
        ``base_revenue``; must be non-zero. Provide **exactly one** of
        ``base_fcf`` or ``fcf_margin``; both must be omitted when
        ``segments`` is given. Not itself a sweep axis.
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
        ``base_revenue``. Negative means net cash. Default ``0.0``.
    segments
        Optional list of :class:`Segment` modelling the company as a mix
        of streams. When given, ``base_revenue`` is the company total,
        each segment supplies its own ``revenue_share`` / ``fcf_margin`` /
        ``growth``, and **exactly one** segment has ``growth=None`` (the
        swing — its growth is solved). ``None`` (default) is single-segment
        mode (one implicit swing segment using ``fcf_margin`` / ``base_fcf``
        — the math is identical to the homogeneous-company case).
    axis_a, axis_b
        Names of the two inputs to sweep (the index and column
        dimensions), and must differ. A global input is a bare string
        (``'market_price'``, ``'shares_outstanding'``, ``'base_revenue'``,
        ``'wacc'``, ``'terminal_growth'``, ``'horizon'``, ``'net_debt'``);
        a segment-level input is dotted ``'<segment>.<field>'`` with
        ``field`` ∈ ``{fcf_margin, growth, revenue_share}``. Bare
        ``'fcf_margin'`` is valid only single-segment; the swing segment's
        ``growth`` may not be swept (it is the output). Default
        ``'wacc'`` × ``'fcf_margin'`` (the single-segment default).
    axis_a_values, axis_b_values
        The grid values for each axis (non-empty), **in the unit of the
        swept input** — fractions for a rate axis (``wacc`` /
        ``terminal_growth`` / a segment ``fcf_margin`` / ``growth`` /
        ``revenue_share``), integers ``>= 1`` for a ``'horizon'`` axis,
        currency for a dollar axis. The corresponding consensus scalar is
        inserted if missing, and each grid is sorted ascending and
        de-duplicated. A ``UserWarning`` is emitted if a rate axis (or a
        rate scalar) carries a value with magnitude ``> 1`` — almost always
        a whole-number-percent typo (e.g. ``8`` for 8 %).
    cagr_bounds
        ``(low, high)`` bracket for the root search, as CAGR **fractions**
        (not percents). Default ``(-0.50, 1.00)`` (−50 % to +100 %). A
        cell whose implied CAGR falls outside this bracket is ``NaN`` —
        widen the bounds to resolve it (a fast-growing swing segment may
        imply > 100 %). Must have ``low < high``.

    Returns
    -------
    ReverseDCFResult
        The heatmap DataFrame (cell values = implied CAGR in **percentage
        points**; a rate-like axis's labels likewise in percentage points),
        the axis names, the consensus cell / CAGR, and the ``swing_segment``
        name (``None`` single-segment). A cell is ``NaN`` when the per-cell
        configuration has no Gordon-growth solution (``wacc <=
        terminal_growth``), the implied CAGR lies outside ``cagr_bounds``,
        a swept ``revenue_share`` leaves no room for the other segments, or
        the per-cell math fails numerically. Up-front (consensus) parameters
        are validated and raise; per-cell invalid configurations degrade to
        ``NaN`` so the rest of the heatmap still renders.

    Warns
    -----
    UserWarning
        If a rate input (``wacc``, ``terminal_growth``, or a segment
        ``fcf_margin`` / ``growth`` / ``revenue_share`` — scalar or swept
        grid value) has magnitude ``> 1``, which almost always means
        whole-number percents were passed where fractions are expected.

    Raises
    ------
    TypeError
        If any axis grid contains a non-numeric value, ``name`` is not a
        string, or ``segments`` contains a non-:class:`Segment`.
    ValueError
        If ``market_price``/``shares_outstanding``/``base_revenue`` are
        not ``> 0``; the single-segment ``fcf_margin`` / ``base_fcf`` rule
        is violated (neither, both, zero, or supplied alongside
        ``segments``); ``wacc`` is outside ``(0, 1)``; ``terminal_growth >=
        wacc``; ``horizon`` is not an integer ``>= 1``; the ``segments``
        list is malformed (empty, duplicate/reserved/dotted name, a share
        outside ``(0, 1]`` or not summing to 1, not exactly one swing, or a
        zero swing margin); ``axis_a``/``axis_b`` are not valid axis names
        (unknown global, bad segment/field, the swing's ``growth``, or bare
        ``fcf_margin`` in segment mode) or are equal; an axis grid is empty;
        a ``'horizon'`` grid holds a non-integer or value ``< 1``; or
        ``cagr_bounds`` is not ``low < high``.
    """
    # ── Validate consensus (base) global parameters ───────────────────
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

    # ── Build the segment structure (single-segment or multi-segment) ──
    if segments is None:
        # Single-segment: resolve the one margin from fcf_margin xor base_fcf.
        resolved = _resolve_fcf_margin(fcf_margin, base_fcf, base_revenue)
        seg_list: List[Dict] = [
            {'name': None, 'revenue_share': 1.0, 'fcf_margin': resolved, 'growth': None}
        ]
        swing_index = 0
        segments_provided = False
        segment_names: frozenset = frozenset()
        swing_name: Optional[str] = None
    else:
        if fcf_margin is not None or base_fcf is not None:
            raise ValueError(
                "with segments=, supply each segment's fcf_margin on the "
                "Segment; top-level fcf_margin/base_fcf must be omitted"
            )
        seg_list, swing_index = _build_segments(segments)
        segments_provided = True
        segment_names = frozenset(s['name'] for s in seg_list)
        swing_name = seg_list[swing_index]['name']

    # ── Validate axes ─────────────────────────────────────────────────
    _validate_axis('axis_a', axis_a, segments_provided, segment_names, swing_name)
    _validate_axis('axis_b', axis_b, segments_provided, segment_names, swing_name)
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
        'wacc': float(wacc),
        'terminal_growth': float(terminal_growth),
        'horizon': int(horizon),
        'net_debt': float(net_debt),
        'segments': seg_list,
        'swing_index': swing_index,
    }

    a_vals = _prepare_axis(axis_a, axis_a_values, _get_axis_value(base, axis_a))
    b_vals = _prepare_axis(axis_b, axis_b_values, _get_axis_value(base, axis_b))

    _warn_if_percent_like({axis_a: a_vals, axis_b: b_vals}, base, segments_provided)

    # ── Solve every cell ──────────────────────────────────────────────
    data = np.full((len(a_vals), len(b_vals)), _NAN, dtype=float)
    nan_count = 0
    bracket = (float(lo), float(hi))
    for (i, a_val), (j, b_val) in product(enumerate(a_vals), enumerate(b_vals)):
        cell, feasible = _build_cell(base, axis_a, a_val, axis_b, b_val)
        cagr = _solve_cell(cell, bracket) if feasible else _NAN
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
        _to_display(axis_a, _get_axis_value(base, axis_a)),
        _to_display(axis_b, _get_axis_value(base, axis_b)),
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
        swing_segment=swing_name,
        name=name,
    )


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────

def _axis_field(axis: str) -> str:
    """The field a sweep axis targets — the suffix after ``'.'`` if dotted,
    else the axis name itself (so a global axis is its own field)."""
    return axis.partition('.')[2] if '.' in axis else axis


def _is_rate_like_axis(axis: str) -> bool:
    """True if the axis's field is a fraction-valued rate (so it is labelled
    in percentage points and policed by the percent-typo guardrail)."""
    return _axis_field(axis) in _RATE_LIKE_FIELDS


def _to_display(axis_name: str, value: float):
    """Map an axis value to its display unit for heatmap labelling.

    A rate-like axis (``wacc`` / ``terminal_growth`` or any segment
    ``fcf_margin`` / ``growth`` / ``revenue_share``) is converted to
    percentage points (``×100``, rounded to kill float noise) so its
    index/column labels read in percent, matching the cell values. Every
    other axis (``horizon``, dollar amounts, share counts) passes through
    unchanged. Applied identically to grid labels and ``consensus_cell``,
    so ``heatmap.loc`` lookups remain exact.
    """
    if _is_rate_like_axis(axis_name):
        return round(value * 100.0, _PCT_DISPLAY_DECIMALS)
    return value


def _warn_if_percent_like(
    swept: Dict[str, Sequence[float]],
    params: dict,
    segments_provided: bool,
) -> None:
    """Emit a ``UserWarning`` if a rate-like input looks like a percent typo.

    For each rate-like input — global ``wacc`` / ``terminal_growth`` and,
    per segment, ``fcf_margin`` / ``revenue_share`` (plus ``growth`` for
    fixed segments; the swing's growth is solved, not an input) — gathers
    the values it will actually take (the prepared grid when swept, else its
    single consensus scalar) and flags any with magnitude ``> 1`` (e.g.
    ``8`` meant as 8 %, or a ``revenue_share`` of ``10`` meant as 10 %).
    These would size the DCF with 800 %-style rates and silently yield
    ``NaN`` cells, so warning loudly turns a confusing empty heatmap into an
    actionable message. Does not raise — an extreme-but-deliberate rate
    stays legal.
    """
    offenders: Dict[str, list] = {}

    def check(axis_name: str, consensus_val: float) -> None:
        values = swept.get(axis_name, [consensus_val])
        bad = [v for v in values if abs(v) > 1.0]
        if bad:
            offenders[axis_name] = bad

    check('wacc', params['wacc'])
    check('terminal_growth', params['terminal_growth'])
    if segments_provided:
        for idx, seg in enumerate(params['segments']):
            check(f"{seg['name']}.fcf_margin", seg['fcf_margin'])
            check(f"{seg['name']}.revenue_share", seg['revenue_share'])
            if idx != params['swing_index']:
                check(f"{seg['name']}.growth", seg['growth'])
    else:
        check('fcf_margin', params['segments'][0]['fcf_margin'])

    if offenders:
        detail = '; '.join(f"{p}={vals}" for p, vals in offenders.items())
        warnings.warn(
            f"reverse_dcf_cagr: {detail} — rate inputs (wacc, "
            f"terminal_growth, and segment fcf_margin / growth / "
            f"revenue_share) must be fractions, not whole-number percents "
            f"(e.g. 0.08 for 8%, not 8). Values with magnitude > 1 will not "
            f"solve and appear as NaN.",
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
    """Resolve the single-segment FCF margin from ``fcf_margin`` xor ``base_fcf``.

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


def _build_segments(segments: Sequence[Segment]) -> Tuple[List[Dict], int]:
    """Validate a ``segments`` list and convert to internal dicts.

    Returns ``(seg_list, swing_index)`` where ``seg_list`` is a list of
    mutable ``{name, revenue_share, fcf_margin, growth}`` dicts (the swing's
    ``growth`` is ``None``) and ``swing_index`` locates the swing. Enforces:
    non-empty list; each a :class:`Segment`; unique, non-empty, dot-free,
    non-reserved names; each ``revenue_share ∈ (0, 1]`` with the set summing
    to ``1.0``; real margins; exactly one ``growth=None`` (the swing) with
    the rest real; and a non-zero swing margin. Raises ``TypeError`` /
    ``ValueError`` on any violation.
    """
    seq = list(segments)
    if not seq:
        raise ValueError("segments must be a non-empty list of Segment")

    seg_list: List[Dict] = []
    names: set = set()
    swing_idxs: List[int] = []
    share_sum = 0.0
    for idx, s in enumerate(seq):
        if not isinstance(s, Segment):
            raise TypeError(
                f"segments[{idx}] must be a Segment, got {type(s).__name__}"
            )
        nm = s.name
        if not isinstance(nm, str) or not nm:
            raise ValueError(
                f"segments[{idx}].name must be a non-empty string, got {nm!r}"
            )
        if '.' in nm:
            raise ValueError(f"segment name {nm!r} must not contain '.'")
        if nm in _RESERVED_SEGMENT_NAMES:
            raise ValueError(
                f"segment name {nm!r} collides with a reserved axis name"
            )
        if nm in names:
            raise ValueError(f"duplicate segment name {nm!r}")
        names.add(nm)

        if not _is_real(s.revenue_share) or not (0.0 < s.revenue_share <= 1.0):
            raise ValueError(
                f"segment {nm!r} revenue_share must be in (0, 1], got "
                f"{s.revenue_share!r}"
            )
        share_sum += float(s.revenue_share)

        if not _is_real(s.fcf_margin):
            raise ValueError(
                f"segment {nm!r} fcf_margin must be a real number, got "
                f"{s.fcf_margin!r}"
            )

        growth = s.growth
        if growth is None:
            swing_idxs.append(idx)
        elif not _is_real(growth):
            raise ValueError(
                f"segment {nm!r} growth must be a real number or None "
                f"(None marks the swing), got {growth!r}"
            )

        seg_list.append(
            {
                'name': nm,
                'revenue_share': float(s.revenue_share),
                'fcf_margin': float(s.fcf_margin),
                'growth': None if growth is None else float(growth),
            }
        )

    if abs(share_sum - 1.0) > _SHARE_TOL:
        raise ValueError(
            f"segment revenue_share values must sum to 1.0, got {share_sum!r}"
        )
    if len(swing_idxs) != 1:
        raise ValueError(
            f"exactly one segment must have growth=None (the swing), found "
            f"{len(swing_idxs)}"
        )
    swing_index = swing_idxs[0]
    if seg_list[swing_index]['fcf_margin'] == 0.0:
        raise ValueError(
            f"the swing segment {seg_list[swing_index]['name']!r} must have a "
            f"non-zero fcf_margin (its growth could not otherwise move the price)"
        )
    return seg_list, swing_index


def _validate_axis(
    label: str,
    axis: str,
    segments_provided: bool,
    segment_names: frozenset,
    swing_name: Optional[str],
) -> None:
    """Validate one sweep-axis name (``label`` is ``'axis_a'``/``'axis_b'``).

    A dotted ``'<segment>.<field>'`` axis needs ``segments=`` supplied, a
    known segment, a valid field, and (for ``growth``) a non-swing segment.
    A bare axis must be a known global; bare ``'fcf_margin'`` is accepted
    only in single-segment mode. Raises ``TypeError`` / ``ValueError``.
    """
    if not isinstance(axis, str):
        raise TypeError(f"{label} must be a string, got {type(axis).__name__}")
    if '.' in axis:
        if not segments_provided:
            raise ValueError(
                f"{label}={axis!r}: dotted segment axes require segments= "
                f"(single-segment mode has no named segments)"
            )
        seg_name, _, field = axis.partition('.')
        if seg_name not in segment_names:
            raise ValueError(f"{label}={axis!r}: unknown segment {seg_name!r}")
        if field not in _SEGMENT_FIELDS:
            raise ValueError(
                f"{label}={axis!r}: field must be one of {list(_SEGMENT_FIELDS)}"
            )
        if field == 'growth' and seg_name == swing_name:
            raise ValueError(
                f"{label}={axis!r}: cannot sweep the swing segment's growth "
                f"(it is the reverse-solved output)"
            )
        return
    if axis in _GLOBAL_AXES:
        return
    if axis == 'fcf_margin':
        if segments_provided:
            raise ValueError(
                f"{label}='fcf_margin': in segment mode address a segment's "
                f"margin as '<segment>.fcf_margin'"
            )
        return
    raise ValueError(
        f"{label} must be one of {list(_GLOBAL_AXES)} (or 'fcf_margin' in "
        f"single-segment mode, or a '<segment>.<field>' axis), got {axis!r}"
    )


def _prepare_axis(name: str, values: Sequence[float], base_value: float) -> list:
    """Validate, coerce, sort and de-duplicate one axis grid.

    Ensures ``base_value`` (the consensus scalar) is present so the
    consensus cell always exists. A ``'horizon'`` grid is integer-typed
    (values ``>= 1``); every other axis — globals and dotted segment fields
    alike — is float-typed. Raises ``TypeError`` on a non-numeric value,
    ``ValueError`` on an empty grid or an out-of-domain ``'horizon'`` value.
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


def _find_segment(params: dict, name: str) -> Dict:
    """Return the segment dict named ``name`` (validated to exist upstream)."""
    for seg in params['segments']:
        if seg['name'] == name:
            return seg
    raise KeyError(name)  # pragma: no cover - axis validation guarantees presence


def _get_axis_value(params: dict, axis: str):
    """Read the consensus value an axis currently holds in ``params``.

    Handles globals (top-level keys), bare ``'fcf_margin'`` (the single
    segment's margin in single-segment mode), and dotted
    ``'<segment>.<field>'`` (that segment's field).
    """
    if '.' in axis:
        seg_name, _, field = axis.partition('.')
        return _find_segment(params, seg_name)[field]
    if axis == 'fcf_margin':
        return params['segments'][0]['fcf_margin']
    return params[axis]


def _set_axis_value(params: dict, axis: str, value: float) -> bool:
    """Write a non-``revenue_share`` axis value into ``params`` in place.

    Handles globals, bare ``'fcf_margin'`` (single-segment), and dotted
    ``fcf_margin`` / ``growth`` fields. ``revenue_share`` is **not** handled
    here — it is applied jointly by :func:`_apply_revenue_shares` so that
    two share axes renormalize coherently. Returns ``True`` (always
    feasible).
    """
    if '.' in axis:
        seg_name, _, field = axis.partition('.')
        _find_segment(params, seg_name)[field] = float(value)
        return True
    if axis == 'fcf_margin':
        params['segments'][0]['fcf_margin'] = float(value)
        return True
    if axis == 'horizon':
        params['horizon'] = int(value)
        return True
    params[axis] = float(value)
    return True


def _apply_revenue_shares(params: dict, assignments: Dict[str, float]) -> bool:
    """Set one or more segments' ``revenue_share`` and renormalize the rest.

    The named segments take their assigned shares; every **unset** segment
    rescales proportionally to refill the remaining ``1 − Σ assigned``, so
    the mix shifts while total revenue stays ``base_revenue`` and shares sum
    to 1.0. For a single assignment this is the documented
    ``other_i = other_i_old · (1 − new_share)/(1 − old_share)`` (since the
    unset segments' old shares sum to ``1 − old_share``).

    Returns ``False`` (→ the cell becomes ``NaN``) when the assignment is
    infeasible: an assigned share outside ``(0, 1]``; assigned shares
    summing past 1 (no room for unset segments); unset segments that
    currently hold zero share (nothing to scale); or — when there are no
    unset segments — assigned shares not already summing to 1.0.
    """
    for v in assignments.values():
        if not (0.0 < v <= 1.0):
            return False
    set_sum = sum(assignments.values())

    segs = params['segments']
    unset = [s for s in segs if s['name'] not in assignments]
    if unset:
        remaining = 1.0 - set_sum
        if remaining < -_SHARE_TOL:
            return False
        unset_old_sum = sum(s['revenue_share'] for s in unset)
        if unset_old_sum <= 0.0:
            return False
        factor = max(remaining, 0.0) / unset_old_sum
        for s in unset:
            s['revenue_share'] = s['revenue_share'] * factor
    elif abs(set_sum - 1.0) > _SHARE_TOL:
        # Every segment is pinned by an axis but they don't sum to 1.
        return False

    for name, v in assignments.items():
        _find_segment(params, name)['revenue_share'] = v
    return True


def _copy_params(base: dict) -> dict:
    """Shallow-copy ``base`` but deep-copy the mutable ``segments`` dicts so a
    per-cell axis write never bleeds into another cell."""
    out = dict(base)
    out['segments'] = [dict(s) for s in base['segments']]
    return out


def _build_cell(
    base: dict, axis_a: str, a_val: float, axis_b: str, b_val: float
) -> Tuple[dict, bool]:
    """Materialize the per-cell ``params`` with both axis values applied.

    Non-``revenue_share`` axes write directly; ``revenue_share`` axes are
    collected and applied jointly (so two share axes renormalize coherently
    against the unset segments). Returns ``(cell, feasible)`` — ``feasible``
    is ``False`` when a share assignment leaves no room (the cell is then
    ``NaN``).
    """
    cell = _copy_params(base)
    shares: Dict[str, float] = {}
    feasible = True
    for axis, val in ((axis_a, a_val), (axis_b, b_val)):
        if '.' in axis and _axis_field(axis) == 'revenue_share':
            shares[axis.partition('.')[0]] = float(val)
        else:
            feasible = _set_axis_value(cell, axis, val) and feasible
    if feasible and shares:
        feasible = _apply_revenue_shares(cell, shares)
    return cell, feasible


def _implied_price(g: float, params: dict) -> float:
    """Forward two-stage FCFF implied price per share for swing growth ``g``.

    Sums FCF across all segments (the swing segment uses ``g``; the rest
    use their pinned ``growth``), discounts the explicit horizon, adds the
    blended Gordon terminal value, bridges EV → equity → per share. Reads
    the model inputs from ``params``; assumes the Gordon-growth guard
    (``wacc > terminal_growth``) has already been checked by the caller.
    """
    base_revenue = params['base_revenue']
    wacc = params['wacc']
    terminal_growth = params['terminal_growth']
    n = int(params['horizon'])
    net_debt = params['net_debt']
    shares = params['shares_outstanding']
    segs = params['segments']
    swing_index = params['swing_index']

    one_plus_w = 1.0 + wacc

    pv_explicit = 0.0
    fcf_n = 0.0
    for idx, seg in enumerate(segs):
        seg_growth = g if idx == swing_index else seg['growth']
        rev0_margin = base_revenue * seg['revenue_share'] * seg['fcf_margin']
        one_plus_g = 1.0 + seg_growth
        for t in range(1, n + 1):
            fcf_t = rev0_margin * one_plus_g ** t
            pv_explicit += fcf_t / one_plus_w ** t
        fcf_n += rev0_margin * one_plus_g ** n

    tv_n = fcf_n * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = tv_n / one_plus_w ** n

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    return equity_value / shares


def _solve_cell(params: dict, bracket: Tuple[float, float]) -> float:
    """Root-find the implied swing-segment growth for one cell, or ``NaN``.

    Returns ``NaN`` when the Gordon-growth guard fails
    (``wacc <= terminal_growth``), the root is not bracketed within
    ``bracket`` (implied growth outside the search window), or the per-cell
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
