"""Unit tests for ``analytics.turnover_stats``.

Pin:
- Golden hand-computed Carver turnover (round trips/year, holding days)
  under the calendar convention.
- The active window starts at the first fill (pre-listing zeros dilute
  the average position otherwise).
- Zero-fill symbols are excluded; the 'Average' row carries the
  cross-instrument means of the unitless columns (round trips/year,
  holding days) and NaN avg position (contract counts are not
  comparable); rows sort alphabetically with 'Average' last.
- Multi-row-per-timestamp equity curves collapse to the last row.
- Data edge cases yield NaN / empty frames — never raise.
- Parameter validation raises (non-DataFrame, bad convention, missing
  columns).

Run from the repo root:  python -m pytest tests/test_analytics_turnover.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from analytics import turnover_stats

_COLUMNS = [
    'Avg Position [contracts]',
    'Round Trips / Year',
    'Avg Holding [days]',
]


# ──────────────────────────────────────────────
# Builders (mirror the real portfolio schemas)
# ──────────────────────────────────────────────

def _equity_curve(positions_seq, *, start='2024-01-01') -> pd.DataFrame:
    """Equity-curve frame with per-timestamp ``positions`` dicts.

    Two rows per timestamp, garbage dict first (collapse trap — the last
    row per timestamp must win).
    """
    keys = sorted({s for row in positions_seq for s in row})
    idx = pd.date_range(start, periods=len(positions_seq), freq='D',
                        tz='UTC')
    rows = []
    for ts, vals in zip(idx, positions_seq):
        true_p = {s: float(vals.get(s, 0.0)) for s in keys}
        garbage = {s: v + 999.0 for s, v in true_p.items()}
        for positions in (garbage, true_p):
            rows.append({
                'timestamp': ts,
                'symbol': keys[0],
                'positions': dict(positions),
                'account_balance': 0.0,
            })
    return pd.DataFrame(rows).set_index('timestamp')


def _trade_log(fills, *, start='2024-01-01') -> pd.DataFrame:
    """Trade-log frame from ``(day_offset, symbol, quantity)`` tuples."""
    base = pd.Timestamp(start, tz='UTC')
    rows = [{
        'timestamp': base + pd.Timedelta(days=day),
        'symbol': sym,
        'direction': 'BUY',
        'quantity': float(qty),
        'fill_price': 100.0,
        'fill_notional': 100.0 * qty,
        'commission': 0.0,
        'realized_pnl': 0.0,
        'position_after': float(qty),
        'cash_after': 0.0,
        'order_id': str(i),
    } for i, (day, sym, qty) in enumerate(fills)]
    return pd.DataFrame(rows)


def _stats(positions_seq, fills, **kwargs):
    defaults = dict(timeframe='1d', days_convention='calendar')
    defaults.update(kwargs)
    return turnover_stats(_equity_curve(positions_seq), _trade_log(fills),
                          **defaults)


# ──────────────────────────────────────────────
# Golden values
# ──────────────────────────────────────────────

def test_golden_single_symbol_calendar():
    # First fill at day 2; active window = 4 bars of |position| = 2.
    # rt/yr = (2/2) / 2 / (4/365) = 45.625; holding = 365/45.625 = 8 days.
    positions = [{'AAA': 0}, {'AAA': 0}, {'AAA': 2}, {'AAA': 2},
                 {'AAA': 2}, {'AAA': 2}]
    table = _stats(positions, [(2, 'AAA', 2)])

    assert list(table.columns) == _COLUMNS
    assert list(table.index) == ['AAA', 'Average']
    row = table.loc['AAA']
    assert row['Avg Position [contracts]'] == pytest.approx(2.0)
    assert row['Round Trips / Year'] == pytest.approx(45.625)
    assert row['Avg Holding [days]'] == pytest.approx(8.0)


def test_active_window_excludes_pre_fill_bars():
    # A whole-backtest window would average |position| over 6 bars
    # (8/6 ≈ 1.333); the first-fill window pins it at exactly 2.0.
    positions = [{'AAA': 0}, {'AAA': 0}, {'AAA': 2}, {'AAA': 2},
                 {'AAA': 2}, {'AAA': 2}]
    table = _stats(positions, [(2, 'AAA', 2)])
    assert table.loc['AAA', 'Avg Position [contracts]'] == pytest.approx(2.0)


def test_business_convention_changes_annualization():
    positions = [{'AAA': 0}, {'AAA': 0}, {'AAA': 2}, {'AAA': 2},
                 {'AAA': 2}, {'AAA': 2}]
    table = _stats(positions, [(2, 'AAA', 2)],
                   days_convention='business')
    # Same math with 252 days/year.
    assert table.loc['AAA', 'Round Trips / Year'] == pytest.approx(
        (2 / 2) / 2.0 / (4 / 252))
    assert table.loc['AAA', 'Avg Holding [days]'] == pytest.approx(8.0)


# ──────────────────────────────────────────────
# Table shape: ordering, Average row, exclusions
# ──────────────────────────────────────────────

def test_multi_symbol_average_row_unitless_means():
    positions = [
        {'AAA': 0, 'BBB': 0},
        {'AAA': 0, 'BBB': 1},
        {'AAA': 2, 'BBB': 1},
        {'AAA': 2, 'BBB': 1},
        {'AAA': 2, 'BBB': 1},
        {'AAA': 2, 'BBB': 1},
    ]
    table = _stats(positions, [(2, 'AAA', 2), (1, 'BBB', 1)])

    assert list(table.index) == ['AAA', 'BBB', 'Average']
    # AAA: rt 45.625, holds 8 days (golden above); BBB: window 5 bars,
    # avg 1, rt = 0.5 * 365/5 = 36.5 → holding 10 days.
    assert table.loc['BBB', 'Avg Holding [days]'] == pytest.approx(10.0)
    avg = table.loc['Average']
    assert math.isnan(avg['Avg Position [contracts]'])
    assert avg['Round Trips / Year'] == pytest.approx((45.625 + 36.5) / 2)
    assert avg['Avg Holding [days]'] == pytest.approx(9.0)


def test_symbol_without_fills_is_excluded():
    positions = [{'AAA': 2, 'CCC': 5}, {'AAA': 2, 'CCC': 5}]
    table = _stats(positions, [(0, 'AAA', 2)])
    assert 'CCC' not in table.index
    assert list(table.index) == ['AAA', 'Average']


def test_collapse_uses_last_row_per_timestamp():
    # The builder writes a garbage (+999) positions dict in the first of
    # the two rows per timestamp — a wrong collapse shifts avg position.
    positions = [{'AAA': 2}, {'AAA': 2}]
    table = _stats(positions, [(0, 'AAA', 2)])
    assert table.loc['AAA', 'Avg Position [contracts]'] == pytest.approx(2.0)


# ──────────────────────────────────────────────
# Data edge cases (never raise)
# ──────────────────────────────────────────────

def test_zero_average_position_yields_nan_not_raise():
    positions = [{'AAA': 0}, {'AAA': 0}]
    table = _stats(positions, [(0, 'AAA', 1)])
    row = table.loc['AAA']
    assert row['Avg Position [contracts]'] == pytest.approx(0.0)
    assert math.isnan(row['Round Trips / Year'])
    assert math.isnan(row['Avg Holding [days]'])
    assert math.isnan(table.loc['Average', 'Round Trips / Year'])
    assert math.isnan(table.loc['Average', 'Avg Holding [days]'])


def test_filled_symbol_missing_from_positions_yields_nan_row():
    positions = [{'AAA': 1}, {'AAA': 1}]
    table = _stats(positions, [(0, 'AAA', 1), (0, 'XXX', 1)])
    assert list(table.index) == ['AAA', 'XXX', 'Average']
    assert table.loc['XXX'].isna().all()


def test_empty_inputs_return_empty_fixed_schema_frame():
    empty_eq = pd.DataFrame()
    empty_tl = pd.DataFrame()
    filled_eq = _equity_curve([{'AAA': 1}])
    filled_tl = _trade_log([(0, 'AAA', 1)])
    for eq, tl in [(empty_eq, empty_tl), (empty_eq, filled_tl),
                   (filled_eq, empty_tl)]:
        table = turnover_stats(eq, tl, timeframe='1d',
                               days_convention='calendar')
        assert list(table.columns) == _COLUMNS
        assert table.empty


# ──────────────────────────────────────────────
# Parameter validation
# ──────────────────────────────────────────────

def test_non_dataframe_inputs_raise():
    with pytest.raises(TypeError, match="equity_curve must be a DataFrame"):
        turnover_stats([1], _trade_log([]), timeframe='1d',
                       days_convention='calendar')
    with pytest.raises(TypeError, match="trade_log must be a DataFrame"):
        turnover_stats(pd.DataFrame(), 'nope', timeframe='1d',
                       days_convention='calendar')


def test_bad_annualization_params_raise():
    with pytest.raises(ValueError):
        _stats([{'AAA': 1}], [(0, 'AAA', 1)], days_convention='lunar')
    with pytest.raises(ValueError):
        _stats([{'AAA': 1}], [(0, 'AAA', 1)], timeframe='bogus')


def test_missing_required_columns_raise():
    eq = _equity_curve([{'AAA': 1}]).drop(columns=['positions'])
    with pytest.raises(ValueError, match="equity_curve is missing"):
        turnover_stats(eq, _trade_log([(0, 'AAA', 1)]), timeframe='1d',
                       days_convention='calendar')
    tl = _trade_log([(0, 'AAA', 1)]).drop(columns=['quantity'])
    with pytest.raises(ValueError, match="trade_log is missing"):
        turnover_stats(_equity_curve([{'AAA': 1}]), tl, timeframe='1d',
                       days_convention='calendar')


def test_turnover_stats_naive_curve_raises():
    """UTC law (tz audit 2026-07): a naive equity-curve index raises."""
    idx = pd.to_datetime(['2024-01-01', '2024-01-02'])      # tz-naive
    eq = pd.DataFrame({'account_balance': [100.0, 110.0]}, index=idx)
    with pytest.raises(ValueError, match="equity_curve.*timezone-naive"):
        turnover_stats(eq, pd.DataFrame(), timeframe='1d',
                       days_convention='calendar')
