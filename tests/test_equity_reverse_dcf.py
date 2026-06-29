"""Unit tests for ``equity.reverse_dcf_cagr``.

Run from the repo root:  python -m pytest tests/test_equity_reverse_dcf.py -v

The core correctness check is a *round-trip*: an independent forward
two-stage FCFF valuation produces a price at a known CAGR ``g*``; feeding
that price back through ``reverse_dcf_cagr`` must recover ``g*``. The
forward helper below is deliberately a separate, plain implementation so
it acts as a golden reference rather than re-using the module internals.
"""

import math

import numpy as np
import pandas as pd
import pytest

from equity import ReverseDCFResult, Segment, reverse_dcf_cagr


# ──────────────────────────────────────────────
# Independent golden forward model
# ──────────────────────────────────────────────

def _forward_price(
    g,
    *,
    base_revenue,
    fcf_margin,
    wacc,
    terminal_growth,
    horizon,
    net_debt,
    shares_outstanding,
):
    """Plain two-stage FCFF implied price per share at revenue CAGR ``g``."""
    pv = 0.0
    for t in range(1, horizon + 1):
        fcf_t = base_revenue * (1 + g) ** t * fcf_margin
        pv += fcf_t / (1 + wacc) ** t
    fcf_n = base_revenue * (1 + g) ** horizon * fcf_margin
    tv_n = fcf_n * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += tv_n / (1 + wacc) ** horizon
    equity = pv - net_debt
    return equity / shares_outstanding


# Shared base ("consensus") assumptions for a Nokia-ish name.
_BASE = dict(
    shares_outstanding=500.0,
    base_revenue=1000.0,
    fcf_margin=0.15,
    wacc=0.09,
    terminal_growth=0.025,
    horizon=10,
    net_debt=200.0,
)


# ──────────────────────────────────────────────
# Round-trip correctness
# ──────────────────────────────────────────────

def test_round_trip_recovers_known_cagr_at_consensus():
    """Price the model at g*=0.08, reverse it, recover g* (in %) at consensus."""
    g_star = 0.08
    price = _forward_price(g_star, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.09, 0.10],
        axis_b='fcf_margin',
        axis_b_values=[0.12, 0.15, 0.18],
        **_BASE,
    )
    # Rate axes are labelled in percent: wacc 0.09 → 9.0, margin 0.15 → 15.0.
    assert res.consensus_cell == (9.0, 15.0)
    # Output is in percentage points: g*=0.08 → 8.0.
    assert res.consensus_cagr == pytest.approx(g_star * 100, abs=1e-6)


def test_every_solved_cell_reprices_to_market_price():
    """Plugging each solved CAGR (÷100 back to a fraction) reprices to market."""
    price = _forward_price(0.07, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.07, 0.09, 0.11, 0.13],
        axis_b='fcf_margin',
        axis_b_values=[0.10, 0.15, 0.20],
        **_BASE,
    )
    others = {k: v for k, v in _BASE.items() if k not in ('wacc', 'fcf_margin')}
    # index/columns are now percentage points (wacc/fcf_margin are rate axes),
    # so divide by 100 to recover the fractions the forward model expects.
    for w_pct in res.heatmap.index:
        for m_pct in res.heatmap.columns:
            cagr_pct = res.heatmap.loc[w_pct, m_pct]
            if math.isnan(cagr_pct):
                continue
            recomputed = _forward_price(
                cagr_pct / 100, wacc=w_pct / 100, fcf_margin=m_pct / 100, **others
            )
            assert recomputed == pytest.approx(price, rel=1e-6)


# ──────────────────────────────────────────────
# Monotonicity (the economic intuition)
# ──────────────────────────────────────────────

def test_implied_cagr_increases_with_wacc_and_decreases_with_margin():
    """Higher WACC ⇒ need more growth; higher FCF margin ⇒ need less growth."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.07, 0.08, 0.09, 0.10, 0.11],
        axis_b='fcf_margin',
        axis_b_values=[0.10, 0.12, 0.15, 0.18, 0.20],
        **_BASE,
    )
    # Down each column (WACC ascending): implied CAGR rises.
    for m in res.heatmap.columns:
        col = res.heatmap[m].dropna()
        assert col.is_monotonic_increasing
    # Across each row (margin ascending): implied CAGR falls.
    for w in res.heatmap.index:
        row = res.heatmap.loc[w].dropna()
        assert row.is_monotonic_decreasing


# ──────────────────────────────────────────────
# EV → equity bridge
# ──────────────────────────────────────────────

def test_net_cash_implies_lower_cagr_than_net_debt():
    """Net cash lifts equity value, so the same price needs less growth."""
    price = _forward_price(0.08, **_BASE)
    common = dict(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
    )
    net_debt_args = {k: v for k, v in _BASE.items() if k != 'net_debt'}
    res_debt = reverse_dcf_cagr(net_debt=200.0, **net_debt_args, **common)
    res_cash = reverse_dcf_cagr(net_debt=-200.0, **net_debt_args, **common)
    assert res_cash.consensus_cagr < res_debt.consensus_cagr


# ──────────────────────────────────────────────
# Guards → NaN cells (do not raise)
# ──────────────────────────────────────────────

def test_cell_with_wacc_below_terminal_growth_is_nan():
    """A swept WACC at/under terminal growth violates Gordon growth → NaN."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.02, 0.05, 0.09],  # 0.02 <= terminal_growth (0.025)
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    # Labels are in percent (wacc 0.02 → 2.0, 0.09 → 9.0; margin 0.15 → 15.0).
    assert math.isnan(res.heatmap.loc[2.0, 15.0])
    assert not math.isnan(res.heatmap.loc[9.0, 15.0])


def test_cagr_outside_bounds_is_nan():
    """A market price implying growth past the search bracket yields NaN."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price * 1000.0,  # absurd price → implied CAGR > 100%
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        cagr_bounds=(-0.50, 1.00),
        **_BASE,
    )
    assert math.isnan(res.consensus_cagr)


# ──────────────────────────────────────────────
# Consensus cell handling
# ──────────────────────────────────────────────

def test_consensus_scalar_inserted_when_absent_from_grid():
    """The consensus WACC/margin are inserted into their grids if missing."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.07, 0.11],  # base 0.09 absent
        axis_b='fcf_margin',
        axis_b_values=[0.10, 0.20],  # base 0.15 absent
        **_BASE,
    )
    # Inserted consensus scalars appear as percent labels (0.09→9.0, 0.15→15.0).
    assert 9.0 in res.heatmap.index
    assert 15.0 in res.heatmap.columns
    assert res.consensus_cell == (9.0, 15.0)
    assert res.consensus_cagr == pytest.approx(
        res.heatmap.loc[9.0, 15.0], abs=1e-12, nan_ok=True
    )


def test_axis_grids_sorted_and_deduplicated():
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.11, 0.09, 0.09, 0.07],  # unsorted + duplicate
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    assert list(res.heatmap.index) == [7.0, 9.0, 11.0]  # percent labels


# ──────────────────────────────────────────────
# base_fcf dollar input (margin derived from base_fcf / base_revenue)
# ──────────────────────────────────────────────

def _base_without_margin():
    return {k: v for k, v in _BASE.items() if k != 'fcf_margin'}


def test_base_fcf_dollar_input_matches_equivalent_margin():
    """Passing base_fcf (=margin×revenue) reproduces the explicit-margin heatmap."""
    price = _forward_price(0.08, **_BASE)  # _BASE carries fcf_margin=0.15
    grid = dict(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.09, 0.10],
        axis_b='fcf_margin',
        axis_b_values=[0.12, 0.15, 0.18],
    )
    res_margin = reverse_dcf_cagr(**grid, **_BASE)
    res_fcf = reverse_dcf_cagr(
        **grid,
        base_fcf=0.15 * _BASE['base_revenue'],  # 150.0
        **_base_without_margin(),
    )
    pd.testing.assert_frame_equal(res_fcf.heatmap, res_margin.heatmap)
    assert res_fcf.consensus_cagr == pytest.approx(
        res_margin.consensus_cagr, abs=1e-12
    )


def test_base_fcf_dollar_input_round_trips_to_known_cagr():
    """Reverse-solving with a base_fcf figure recovers the priced-in CAGR."""
    g_star = 0.06
    price = _forward_price(g_star, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        base_fcf=0.15 * _BASE['base_revenue'],
        **_base_without_margin(),
    )
    assert res.consensus_cagr == pytest.approx(g_star * 100, abs=1e-6)  # 6.0%


# ──────────────────────────────────────────────
# General axes (not just WACC × margin)
# ──────────────────────────────────────────────

def test_general_axes_terminal_growth_by_horizon():
    """Any two inputs can be axes; labels and dtypes are preserved."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='terminal_growth',
        axis_a_values=[0.015, 0.025, 0.035],
        axis_b='horizon',
        axis_b_values=[5, 10, 15],
        **_BASE,
    )
    assert res.heatmap.index.name == 'terminal_growth'
    assert res.heatmap.columns.name == 'horizon'
    # horizon is not rate-like → integer labels stay native; terminal_growth
    # is rate-like → percent labels (0.025 → 2.5).
    assert list(res.heatmap.columns) == [5, 10, 15]
    assert list(res.heatmap.index) == [1.5, 2.5, 3.5]
    assert res.consensus_cell == (2.5, 10)
    assert res.consensus_cagr == pytest.approx(8.0, abs=1e-6)  # 0.08 → 8.0%


# ──────────────────────────────────────────────
# Company name label
# ──────────────────────────────────────────────

def test_name_is_stored_and_shown_in_caption():
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        name='Nokia',
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    assert res.name == 'Nokia'
    assert 'Nokia' in res.styled().to_html()


def test_name_defaults_to_none_and_caption_has_no_prefix():
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    assert res.name is None
    # No stray "None — " prefix leaks into the rendered caption.
    assert 'None — ' not in res.styled().to_html()


def test_rejects_non_string_name():
    with pytest.raises(TypeError, match="name must be a string"):
        reverse_dcf_cagr(**_valid_kwargs(name=123))


# ──────────────────────────────────────────────
# Result type / styling
# ──────────────────────────────────────────────

def test_result_is_reverse_dcf_result_with_named_axes():
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    assert isinstance(res, ReverseDCFResult)
    assert res.axis_a == 'wacc' and res.axis_b == 'fcf_margin'
    assert isinstance(res.heatmap, pd.DataFrame)


def test_styled_returns_styler_over_the_heatmap():
    from pandas.io.formats.style import Styler

    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.09, 0.10],
        axis_b='fcf_margin',
        axis_b_values=[0.12, 0.15, 0.18],
        **_BASE,
    )
    styler = res.styled()
    assert isinstance(styler, Styler)
    pd.testing.assert_frame_equal(styler.data, res.heatmap)
    # Rendering must not raise (exercises the highlight + format callbacks).
    html = styler.to_html()
    assert isinstance(html, str) and len(html) > 0


def test_styled_highlights_consensus_in_red_with_percent_suffix():
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.09, 0.10],
        axis_b='fcf_margin',
        axis_b_values=[0.12, 0.15, 0.18],
        **_BASE,
    )
    html = res.styled().to_html()
    assert 'color: red' in html              # consensus cell font is red
    assert '8.0%' in html                    # cells rendered with % suffix


# ──────────────────────────────────────────────
# Output units (percentage points)
# ──────────────────────────────────────────────

def test_heatmap_values_are_percentage_points():
    """Cells are 100× the fractional CAGR (38.9 not 0.389)."""
    price = _forward_price(0.08, **_BASE)
    res = reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    # 0.08 fraction → 8.0 percentage points (well above 1, so unambiguous).
    # Axis labels are percent too: wacc 0.09 → 9.0, margin 0.15 → 15.0.
    assert res.heatmap.loc[9.0, 15.0] == pytest.approx(8.0, abs=1e-6)


def test_rate_axis_labelled_percent_but_dollar_axis_stays_native():
    """Rate axes get percent labels; a dollar axis (net_debt) keeps its units."""
    price = _forward_price(0.08, **_BASE)
    others = {k: v for k, v in _BASE.items() if k != 'net_debt'}
    res = reverse_dcf_cagr(
        market_price=price,
        net_debt=200.0,
        axis_a='wacc',          # rate-like → percent labels
        axis_a_values=[0.08, 0.10],
        axis_b='net_debt',      # dollar → native labels
        axis_b_values=[100.0, 300.0],
        **others,
    )
    assert list(res.heatmap.index) == [8.0, 9.0, 10.0]          # wacc → percent
    assert list(res.heatmap.columns) == [100.0, 200.0, 300.0]   # net_debt native
    assert res.consensus_cell == (9.0, 200.0)
    # styled(): rate axis labels carry a '%' suffix, dollar axis does not.
    html = res.styled().to_html()
    assert '9.0%' in html
    assert '200.0%' not in html


# ──────────────────────────────────────────────
# Percent-typo guardrail (rate inputs must be fractions)
# ──────────────────────────────────────────────

def test_warns_when_wacc_axis_values_look_like_percents():
    price = _forward_price(0.08, **_BASE)
    with pytest.warns(UserWarning, match="must be fractions"):
        reverse_dcf_cagr(
            market_price=price,
            axis_a='wacc',
            axis_a_values=[8, 9, 10, 11, 12],  # percents, not fractions
            axis_b='fcf_margin',
            axis_b_values=[0.15],
            **_BASE,
        )


def test_warns_when_fcf_margin_axis_values_look_like_percents():
    price = _forward_price(0.08, **_BASE)
    with pytest.warns(UserWarning, match="must be fractions"):
        reverse_dcf_cagr(
            market_price=price,
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='fcf_margin',
            axis_b_values=[3, 4, 5, 6],  # percents, not fractions
            **_BASE,
        )


def test_no_warning_for_fractional_rates(recwarn):
    price = _forward_price(0.08, **_BASE)
    reverse_dcf_cagr(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.10, 0.12],
        axis_b='fcf_margin',
        axis_b_values=[0.05, 0.15, 0.25],
        **_BASE,
    )
    msgs = [str(w.message) for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert not any('must be fractions' in m for m in msgs)


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def _valid_kwargs(**overrides):
    """A fully-valid call's kwargs, with selective overrides for error tests."""
    kwargs = dict(
        market_price=10.0,
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='fcf_margin',
        axis_b_values=[0.15],
        **_BASE,
    )
    kwargs.update(overrides)
    return kwargs


def test_rejects_unknown_axis_name():
    with pytest.raises(ValueError, match="axis_a must be one of"):
        reverse_dcf_cagr(**_valid_kwargs(axis_a='cagr'))


def test_rejects_equal_axes():
    with pytest.raises(ValueError, match="must differ"):
        reverse_dcf_cagr(**_valid_kwargs(axis_b='wacc'))


def test_rejects_non_positive_market_price():
    with pytest.raises(ValueError, match="market_price must be > 0"):
        reverse_dcf_cagr(**_valid_kwargs(market_price=0.0))


def test_rejects_non_positive_shares():
    with pytest.raises(ValueError, match="shares_outstanding must be > 0"):
        reverse_dcf_cagr(**_valid_kwargs(shares_outstanding=0.0))


def test_rejects_non_positive_base_revenue():
    with pytest.raises(ValueError, match="base_revenue must be > 0"):
        reverse_dcf_cagr(**_valid_kwargs(base_revenue=-5.0))


def test_rejects_zero_fcf_margin():
    with pytest.raises(ValueError, match="fcf_margin must be non-zero"):
        reverse_dcf_cagr(**_valid_kwargs(fcf_margin=0.0))


def test_rejects_neither_fcf_margin_nor_base_fcf():
    with pytest.raises(ValueError, match="exactly one of fcf_margin or base_fcf"):
        reverse_dcf_cagr(**_valid_kwargs(fcf_margin=None))


def test_rejects_both_fcf_margin_and_base_fcf():
    # _valid_kwargs already supplies fcf_margin=0.15; adding base_fcf makes two.
    with pytest.raises(ValueError, match="exactly one of fcf_margin or base_fcf"):
        reverse_dcf_cagr(**_valid_kwargs(base_fcf=150.0))


def test_rejects_zero_base_fcf():
    kwargs = _valid_kwargs(base_fcf=0.0)
    kwargs.pop('fcf_margin')  # leave only base_fcf so it's the one that's checked
    with pytest.raises(ValueError, match="base_fcf must be non-zero"):
        reverse_dcf_cagr(**kwargs)


def test_rejects_wacc_out_of_unit_interval():
    with pytest.raises(ValueError, match="wacc must be in"):
        reverse_dcf_cagr(**_valid_kwargs(wacc=1.5))


def test_rejects_terminal_growth_at_or_above_wacc():
    with pytest.raises(ValueError, match="terminal_growth must be < wacc"):
        reverse_dcf_cagr(**_valid_kwargs(terminal_growth=0.09))


def test_rejects_non_integer_horizon():
    with pytest.raises(ValueError, match="horizon must be an integer"):
        reverse_dcf_cagr(**_valid_kwargs(horizon=10.5))


def test_rejects_zero_horizon():
    with pytest.raises(ValueError, match="horizon must be an integer"):
        reverse_dcf_cagr(**_valid_kwargs(horizon=0))


def test_rejects_empty_axis_grid():
    with pytest.raises(ValueError, match="non-empty"):
        reverse_dcf_cagr(**_valid_kwargs(axis_a_values=[]))


def test_rejects_non_integer_horizon_axis_value():
    with pytest.raises(ValueError, match="horizon grid values must be integers"):
        reverse_dcf_cagr(
            **_valid_kwargs(axis_b='horizon', axis_b_values=[5, 5.5])
        )


def test_rejects_non_numeric_axis_value():
    with pytest.raises(TypeError, match="must be real numbers"):
        reverse_dcf_cagr(**_valid_kwargs(axis_a_values=[0.09, 'oops']))


def test_rejects_inverted_cagr_bounds():
    with pytest.raises(ValueError, match="cagr_bounds"):
        reverse_dcf_cagr(**_valid_kwargs(cagr_bounds=(1.0, -1.0)))


# ══════════════════════════════════════════════════════════════════════
# Multi-segment ("swing segment") reverse DCF
# ══════════════════════════════════════════════════════════════════════

# Independent golden forward model summed over segments. Each segment is a
# (revenue_share, fcf_margin, growth) tuple; deliberately a separate plain
# implementation so it stays a golden reference, not a re-use of internals.
def _forward_price_segments(
    *,
    base_revenue,
    segments,
    wacc,
    terminal_growth,
    horizon,
    net_debt,
    shares_outstanding,
):
    """Plain multi-segment two-stage FCFF implied price per share."""
    pv = 0.0
    fcf_n = 0.0
    for share, margin, growth in segments:
        rev0_margin = base_revenue * share * margin
        for t in range(1, horizon + 1):
            pv += rev0_margin * (1 + growth) ** t / (1 + wacc) ** t
        fcf_n += rev0_margin * (1 + growth) ** horizon
    tv_n = fcf_n * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += tv_n / (1 + wacc) ** horizon
    return (pv - net_debt) / shares_outstanding


# Company-level consensus assumptions (no fcf_margin — that is per segment).
_SEG_GLOBALS = dict(
    shares_outstanding=500.0,
    base_revenue=1000.0,
    wacc=0.09,
    terminal_growth=0.025,
    horizon=10,
    net_debt=200.0,
)


def _ai_mobile():
    """A valid 2-segment book: AI (swing) 10 % + mobile (fixed @2.5 %) 90 %."""
    return [
        Segment('AI', 0.10, 0.20, None),       # swing: growth solved
        Segment('mobile', 0.90, 0.12, 0.025),  # fixed
    ]


# ── Swing round-trip correctness ──────────────────────────────────────

def test_swing_round_trip_recovers_known_ai_growth():
    """Price the book at a known AI growth g*, reverse it, recover g* (in %)."""
    g_star = 0.30
    price = _forward_price_segments(
        segments=[(0.10, 0.20, g_star), (0.90, 0.12, 0.025)], **_SEG_GLOBALS
    )
    res = reverse_dcf_cagr(
        market_price=price,
        segments=_ai_mobile(),
        axis_a='wacc',
        axis_a_values=[0.09],
        axis_b='AI.fcf_margin',
        axis_b_values=[0.18, 0.20, 0.22],
        **_SEG_GLOBALS,
    )
    assert res.swing_segment == 'AI'
    # wacc 0.09 → 9.0; AI margin 0.20 → 20.0 (rate-like dotted axis in %).
    assert res.consensus_cell == (9.0, 20.0)
    assert res.consensus_cagr == pytest.approx(g_star * 100, abs=1e-4)


def test_single_segment_matches_flat_call_exactly():
    """A one-segment book (share 1.0) reproduces the flat fcf_margin heatmap."""
    price = _forward_price(0.08, **_BASE)  # _BASE carries fcf_margin=0.15
    grid = dict(
        market_price=price,
        axis_a='wacc',
        axis_a_values=[0.08, 0.09, 0.10],
        axis_b='terminal_growth',
        axis_b_values=[0.02, 0.025, 0.03],
    )
    res_flat = reverse_dcf_cagr(**grid, **_BASE)
    res_seg = reverse_dcf_cagr(
        **grid,
        segments=[Segment('co', 1.0, 0.15, None)],
        **_base_without_margin(),
    )
    pd.testing.assert_frame_equal(res_seg.heatmap, res_flat.heatmap)
    assert res_seg.swing_segment == 'co'
    assert res_flat.swing_segment is None


# ── Fixed-segment growth as a sweep axis ──────────────────────────────

def test_pinned_mobile_growth_axis_moves_implied_ai_growth():
    """Higher pinned mobile growth ⇒ AI needs less growth for the same price."""
    g_star = 0.30
    price = _forward_price_segments(
        segments=[(0.10, 0.20, g_star), (0.90, 0.12, 0.025)], **_SEG_GLOBALS
    )
    res = reverse_dcf_cagr(
        market_price=price,
        segments=_ai_mobile(),
        axis_a='mobile.growth',
        axis_a_values=[0.02, 0.025, 0.03],
        axis_b='wacc',
        axis_b_values=[0.09],
        **_SEG_GLOBALS,
    )
    # Down the index (mobile growth ascending): implied AI growth falls.
    col = res.heatmap[9.0].dropna()
    assert col.is_monotonic_decreasing
    # At consensus mobile growth (2.5 %) the AI implied growth is g*.
    assert res.heatmap.loc[2.5, 9.0] == pytest.approx(g_star * 100, abs=1e-4)


# ── Revenue-mix sweep + N-general renormalization ─────────────────────

def test_revenue_share_sweep_renormalizes_other_segments():
    """Each solved cell reprices when the other segments renormalize (3-seg)."""
    g_star = 0.30
    fwd = [(0.10, 0.20, g_star), (0.60, 0.12, 0.025), (0.30, 0.15, 0.05)]
    price = _forward_price_segments(segments=fwd, **_SEG_GLOBALS)
    res = reverse_dcf_cagr(
        market_price=price,
        segments=[
            Segment('AI', 0.10, 0.20, None),
            Segment('mobile', 0.60, 0.12, 0.025),
            Segment('networks', 0.30, 0.15, 0.05),
        ],
        axis_a='AI.revenue_share',
        axis_a_values=[0.08, 0.10, 0.15],
        axis_b='wacc',
        axis_b_values=[0.09],
        cagr_bounds=(-0.5, 2.0),
        **_SEG_GLOBALS,
    )
    old_ai = 0.10
    for s_pct in res.heatmap.index:                 # 8.0, 10.0, 15.0
        g_pct = res.heatmap.loc[s_pct, 9.0]
        assert not math.isnan(g_pct)
        s = s_pct / 100
        factor = (1 - s) / (1 - old_ai)             # proportional renorm
        recomputed = _forward_price_segments(
            segments=[
                (s, 0.20, g_pct / 100),
                (0.60 * factor, 0.12, 0.025),
                (0.30 * factor, 0.15, 0.05),
            ],
            **_SEG_GLOBALS,
        )
        assert recomputed == pytest.approx(price, rel=1e-6)
    # Consensus mix (AI 10 %) recovers g*.
    assert res.heatmap.loc[10.0, 9.0] == pytest.approx(g_star * 100, abs=1e-4)


def test_out_of_range_revenue_share_cell_is_nan():
    """A swept share of 0 leaves no room for the swing → NaN (does not raise)."""
    g_star = 0.30
    price = _forward_price_segments(
        segments=[(0.10, 0.20, g_star), (0.90, 0.12, 0.025)], **_SEG_GLOBALS
    )
    res = reverse_dcf_cagr(
        market_price=price,
        segments=_ai_mobile(),
        axis_a='AI.revenue_share',
        axis_a_values=[0.0, 0.10, 0.20],   # 0.0 is infeasible
        axis_b='wacc',
        axis_b_values=[0.09],
        cagr_bounds=(-0.5, 2.0),
        **_SEG_GLOBALS,
    )
    assert math.isnan(res.heatmap.loc[0.0, 9.0])
    assert not math.isnan(res.heatmap.loc[10.0, 9.0])


# ── Dotted-axis display + labelling ───────────────────────────────────

def test_segment_margin_axis_labelled_in_percent():
    """A '<seg>.fcf_margin' axis labels in percentage points and styled() adds %."""
    g_star = 0.30
    price = _forward_price_segments(
        segments=[(0.10, 0.20, g_star), (0.90, 0.12, 0.025)], **_SEG_GLOBALS
    )
    res = reverse_dcf_cagr(
        market_price=price,
        segments=_ai_mobile(),
        axis_a='AI.fcf_margin',
        axis_a_values=[0.18, 0.20, 0.22],
        axis_b='wacc',
        axis_b_values=[0.09],
        **_SEG_GLOBALS,
    )
    assert list(res.heatmap.index) == [18.0, 20.0, 22.0]   # margin → percent
    assert res.heatmap.index.name == 'AI.fcf_margin'
    html = res.styled().to_html()
    assert '20.0%' in html
    assert 'AI growth' in html                              # caption names swing


# ── Axis rejection in segment mode ────────────────────────────────────

def test_rejects_sweeping_swing_segment_growth():
    with pytest.raises(ValueError, match="cannot sweep the swing"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='AI.growth',
            axis_a_values=[0.2, 0.3],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_SEG_GLOBALS,
        )


def test_rejects_bare_fcf_margin_axis_in_segment_mode():
    with pytest.raises(ValueError, match="segment mode"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_unknown_segment_in_axis():
    with pytest.raises(ValueError, match="unknown segment"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='cloud.fcf_margin',
            axis_a_values=[0.20],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_SEG_GLOBALS,
        )


def test_rejects_bad_segment_field_in_axis():
    with pytest.raises(ValueError, match="field must be one of"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='AI.foo',
            axis_a_values=[0.20],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_SEG_GLOBALS,
        )


def test_rejects_dotted_axis_without_segments():
    with pytest.raises(ValueError, match="require segments"):
        reverse_dcf_cagr(
            market_price=10.0,
            axis_a='AI.fcf_margin',
            axis_a_values=[0.20],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_BASE,  # single-segment (carries fcf_margin)
        )


# ── Segment-list validation ───────────────────────────────────────────

def test_rejects_shares_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 0.10, 0.20, None), Segment('mobile', 0.80, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_no_swing_segment():
    with pytest.raises(ValueError, match="exactly one segment must have growth=None"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 0.10, 0.20, 0.30), Segment('mobile', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_two_swing_segments():
    with pytest.raises(ValueError, match="exactly one segment must have growth=None"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 0.10, 0.20, None), Segment('mobile', 0.90, 0.12, None)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_segment_name_with_dot():
    with pytest.raises(ValueError, match="must not contain"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('A.I', 0.10, 0.20, None), Segment('mobile', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_SEG_GLOBALS,
        )


def test_rejects_duplicate_segment_name():
    with pytest.raises(ValueError, match="duplicate segment name"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 0.10, 0.20, None), Segment('AI', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_reserved_segment_name():
    with pytest.raises(ValueError, match="reserved axis name"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('wacc', 0.10, 0.20, None), Segment('mobile', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='mobile.fcf_margin',
            axis_b_values=[0.12],
            **_SEG_GLOBALS,
        )


def test_rejects_share_out_of_range():
    with pytest.raises(ValueError, match="revenue_share must be in"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 1.5, 0.20, None), Segment('mobile', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


def test_rejects_zero_swing_margin():
    with pytest.raises(ValueError, match="non-zero fcf_margin"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[Segment('AI', 0.10, 0.0, None), Segment('mobile', 0.90, 0.12, 0.025)],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='mobile.fcf_margin',
            axis_b_values=[0.12],
            **_SEG_GLOBALS,
        )


def test_rejects_empty_segments_list():
    with pytest.raises(ValueError, match="non-empty"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=[],
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='wacc',
            axis_b_values=[0.09],
            **_SEG_GLOBALS,
        )


def test_rejects_top_level_margin_with_segments():
    with pytest.raises(ValueError, match="top-level fcf_margin"):
        reverse_dcf_cagr(
            market_price=10.0,
            fcf_margin=0.15,           # not allowed alongside segments
            segments=_ai_mobile(),
            axis_a='wacc',
            axis_a_values=[0.09],
            axis_b='AI.fcf_margin',
            axis_b_values=[0.20],
            **_SEG_GLOBALS,
        )


# ── Percent-typo guardrail extends to segment inputs ──────────────────

def test_warns_when_segment_growth_axis_looks_like_percents():
    with pytest.warns(UserWarning, match="must be fractions"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='mobile.growth',
            axis_a_values=[2, 3],      # percents, not fractions
            axis_b='wacc',
            axis_b_values=[0.09],
            cagr_bounds=(-0.5, 2.0),
            **_SEG_GLOBALS,
        )


def test_warns_when_revenue_share_axis_looks_like_percents():
    with pytest.warns(UserWarning, match="must be fractions"):
        reverse_dcf_cagr(
            market_price=10.0,
            segments=_ai_mobile(),
            axis_a='AI.revenue_share',
            axis_a_values=[5, 10],     # percents, not fractions
            axis_b='wacc',
            axis_b_values=[0.09],
            cagr_bounds=(-0.5, 2.0),
            **_SEG_GLOBALS,
        )
