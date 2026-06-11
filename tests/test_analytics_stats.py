"""Unit tests for ``analytics.backtest_stats``.

Pin:
- Exact ordered output labels (the display contract — ``print(stats)``).
- Hand-computed dollar/percentage drawdowns, episode durations, Sharpe,
  volatility (daily/annualized × $/%), CAGR.
- Multi-symbol equity curves (N rows per timestamp) collapse to the last
  row per timestamp.
- Per-closing-fill trade stats (fills with ``realized_pnl != 0``).
- Edge cases return NaN/NaT/0 cleanly (empty inputs, constant balance,
  no closing trades, all-winning trade log) — never raise.
- Parameter validation raises (non-DataFrame, non-positive capital,
  unknown convention).

Run from the repo root:  python -m pytest tests/test_analytics_stats.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from analytics import backtest_stats


# ──────────────────────────────────────────────
# Builders (mirror the real portfolio schemas)
# ──────────────────────────────────────────────

def _equity_curve(balances, *, symbols=('BTC',), commission=0.0,
                  start='2024-01-01') -> pd.DataFrame:
    """Equity-curve frame shaped like ``BacktestPortfolio.get_equity_curve()``.

    One row per (timestamp, symbol) — N rows per timestamp when multiple
    symbols are given, with only the LAST row per timestamp carrying the
    'true' end-of-event balance (earlier rows get a garbage offset so a
    wrong collapse is caught).
    """
    idx = pd.date_range(start, periods=len(balances), freq='D', tz='UTC')
    rows = []
    for ts, bal in zip(idx, balances):
        for i, sym in enumerate(symbols):
            is_last = i == len(symbols) - 1
            rows.append({
                'timestamp': ts,
                'symbol': sym,
                'cash': float(bal),
                'unrealized_pnl': {s: 0.0 for s in symbols},
                'realized_pnl': {s: 0.0 for s in symbols},
                'account_balance': float(bal) if is_last else float(bal) + 999.0,
                'simple_return': float('nan'),
                'log_return': float('nan'),
                'position_margin': 0.0,
                'margin_requirements': {s: 0.0 for s in symbols},
                'available_balance': float(bal),
                'positions': {s: 0.0 for s in symbols},
                'total_commission': float(commission),
            })
    df = pd.DataFrame(rows)
    df.set_index('timestamp', inplace=True)
    return df


def _trade_log(pnls, *, start='2024-01-01') -> pd.DataFrame:
    """Trade-log frame shaped like ``BacktestPortfolio.get_trade_log()``.

    ``pnls`` is the per-fill ``realized_pnl`` sequence; zero entries mimic
    opening fills, nonzero entries are closing fills.
    """
    idx = pd.date_range(start, periods=len(pnls), freq='D', tz='UTC')
    return pd.DataFrame({
        'timestamp': idx,
        'symbol': 'BTC',
        'direction': 'BUY',
        'quantity': 1.0,
        'fill_price': 100.0,
        'fill_notional': 100.0,
        'commission': 0.1,
        'realized_pnl': [float(p) for p in pnls],
        'position_after': 1.0,
        'cash_after': 100_000.0,
        'order_id': list(range(len(pnls))),
    })


def _stats(balances, pnls=(), **kwargs):
    defaults = dict(initial_capital=100_000.0, timeframe='1d',
                    convention='crypto')
    defaults.update(kwargs)
    return backtest_stats(
        _equity_curve(balances), _trade_log(list(pnls)), **defaults,
    )


_EXPECTED_LABELS = [
    'Start', 'End', 'Duration',
    'Equity Final [$]', 'Equity Peak [$]',
    'Net PnL [$]', 'Total Commission [$]',
    'Return [%]', 'CAGR [%]',
    'Volatility (Daily) [$]', 'Volatility (Ann.) [$]',
    'Volatility (Daily) [%]', 'Volatility (Ann.) [%]',
    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
    'Max Drawdown [$]', 'Max Drawdown [%]',
    'Avg Drawdown [$]', 'Avg Drawdown [%]',
    'Max Drawdown Duration', 'Avg Drawdown Duration',
    'Profit Factor', 'Expectancy [$]', 'SQN',
    '# Fills', '# Closing Trades', 'Win Rate [%]',
    'Best Trade [$]', 'Worst Trade [$]', 'Avg Trade [$]',
]


# ──────────────────────────────────────────────
# Display contract
# ──────────────────────────────────────────────

def test_output_is_series_with_exact_ordered_labels():
    stats = _stats([100_000, 101_000])
    assert isinstance(stats, pd.Series)
    assert list(stats.index) == _EXPECTED_LABELS


# ──────────────────────────────────────────────
# Drawdowns (hand-computed, both units, durations)
# ──────────────────────────────────────────────

def test_known_drawdown_dollar_and_pct():
    """balances [100k, 110k, 90k, 95k, 112k, 100k] (daily, initial 100k):

    running peak (incl. baseline): [100k, 110k, 110k, 110k, 112k, 112k]
    dd$  = [0, 0, 20k, 15k, 0, 12k]
    episode 1: t2-t3, depth $20k / 18.1818 % (20k/110k), peak t1 → recovery t4 (3 days)
    episode 2: t5, unrecovered, depth $12k / 10.7142857 % (12k/112k), peak t4 → End (1 day)
    """
    stats = _stats([100_000, 110_000, 90_000, 95_000, 112_000, 100_000])
    assert math.isclose(stats['Max Drawdown [$]'], 20_000.0)
    assert math.isclose(stats['Max Drawdown [%]'], 100 * 20_000 / 110_000)
    assert math.isclose(stats['Avg Drawdown [$]'], 16_000.0)
    assert math.isclose(
        stats['Avg Drawdown [%]'],
        100 * (20_000 / 110_000 + 12_000 / 112_000) / 2,
    )
    assert stats['Max Drawdown Duration'] == pd.Timedelta(days=3)
    assert stats['Avg Drawdown Duration'] == pd.Timedelta(days=2)


def test_drawdown_against_initial_capital_baseline():
    """A losing streak from bar 1 is measured against initial capital even
    though no equity row ever printed the peak."""
    stats = _stats([95_000, 90_000])
    assert math.isclose(stats['Max Drawdown [$]'], 10_000.0)
    assert math.isclose(stats['Max Drawdown [%]'], 10.0)
    assert math.isclose(stats['Equity Peak [$]'], 100_000.0)


# ──────────────────────────────────────────────
# Returns / volatility / ratios (hand-computed)
# ──────────────────────────────────────────────

def test_pnl_equity_and_return_lines():
    stats = _stats([101_000, 103_000, 102_000, 106_000])
    assert math.isclose(stats['Equity Final [$]'], 106_000.0)
    assert math.isclose(stats['Equity Peak [$]'], 106_000.0)
    assert math.isclose(stats['Net PnL [$]'], 6_000.0)
    assert math.isclose(stats['Return [%]'], 6.0)
    # CAGR over 4 daily bars (crypto: 365 bars/year)
    expected_cagr = ((106_000 / 100_000) ** (365 / 4) - 1) * 100
    assert math.isclose(stats['CAGR [%]'], expected_cagr, rel_tol=1e-12)


def test_sharpe_and_volatility_all_four_units():
    """With timeframe='1d' / convention='crypto', bpy == dpy == 365 so the
    daily volatility lines equal the plain per-bar stds."""
    balances = [101_000, 103_000, 102_000, 106_000]
    stats = _stats(balances)

    bal = pd.Series([100_000.0] + [float(b) for b in balances])
    pnl = bal.diff().dropna()                       # [1k, 2k, -1k, 4k]
    ret = bal.pct_change().dropna()

    assert math.isclose(
        stats['Sharpe Ratio'],
        pnl.mean() / pnl.std(ddof=1) * math.sqrt(365), rel_tol=1e-12,
    )
    downside = math.sqrt((np.minimum(pnl, 0.0) ** 2).mean())
    assert math.isclose(
        stats['Sortino Ratio'],
        pnl.mean() / downside * math.sqrt(365), rel_tol=1e-12,
    )
    assert math.isclose(
        stats['Volatility (Daily) [$]'], pnl.std(ddof=1), rel_tol=1e-12,
    )
    assert math.isclose(
        stats['Volatility (Ann.) [$]'],
        pnl.std(ddof=1) * math.sqrt(365), rel_tol=1e-12,
    )
    assert math.isclose(
        stats['Volatility (Daily) [%]'], ret.std(ddof=1) * 100, rel_tol=1e-12,
    )
    assert math.isclose(
        stats['Volatility (Ann.) [%]'],
        ret.std(ddof=1) * math.sqrt(365) * 100, rel_tol=1e-12,
    )


def test_calmar_is_cagr_over_max_drawdown_pct():
    stats = _stats([110_000, 99_000, 104_500])
    assert math.isclose(
        stats['Calmar Ratio'],
        stats['CAGR [%]'] / stats['Max Drawdown [%]'], rel_tol=1e-12,
    )


def test_daily_vol_rescaled_from_hourly_bars():
    """timeframe='1h' / crypto: bpy = 365*24, dpy = 365 → daily $ vol =
    per-bar std × sqrt(24)."""
    balances = [101_000, 103_000, 102_000, 106_000]
    eq = _equity_curve(balances)
    eq.index = pd.date_range('2024-01-01', periods=len(balances),
                             freq='h', tz='UTC')
    stats = backtest_stats(eq, _trade_log([]), initial_capital=100_000.0,
                           timeframe='1h', convention='crypto')
    bal = pd.Series([100_000.0] + [float(b) for b in balances])
    pnl = bal.diff().dropna()
    assert math.isclose(
        stats['Volatility (Daily) [$]'],
        pnl.std(ddof=1) * math.sqrt(24), rel_tol=1e-12,
    )
    assert math.isclose(
        stats['Volatility (Ann.) [$]'],
        pnl.std(ddof=1) * math.sqrt(365 * 24), rel_tol=1e-12,
    )


# ──────────────────────────────────────────────
# Multi-symbol collapse
# ──────────────────────────────────────────────

def test_multi_symbol_curve_collapses_to_last_row_per_timestamp():
    balances = [100_000, 110_000, 90_000, 95_000, 112_000, 100_000]
    multi = backtest_stats(
        _equity_curve(balances, symbols=('BTC', 'ETH')), _trade_log([]),
        initial_capital=100_000.0, timeframe='1d', convention='crypto',
    )
    single = _stats(balances)
    for label in _EXPECTED_LABELS:
        m, s = multi[label], single[label]
        if isinstance(m, float) and pd.isna(m):
            assert pd.isna(s)
        else:
            assert m == s, label


# ──────────────────────────────────────────────
# Trade-level stats (per-closing-fill definition)
# ──────────────────────────────────────────────

def test_trade_stats_from_synthetic_fill_log():
    stats = _stats([100_000, 100_100], pnls=[0.0, 100.0, 0.0, -50.0, 30.0])
    assert stats['# Fills'] == 5
    assert stats['# Closing Trades'] == 3
    assert math.isclose(stats['Win Rate [%]'], 100 * 2 / 3)
    assert math.isclose(stats['Best Trade [$]'], 100.0)
    assert math.isclose(stats['Worst Trade [$]'], -50.0)
    assert math.isclose(stats['Avg Trade [$]'], (100 - 50 + 30) / 3)
    assert math.isclose(stats['Expectancy [$]'], stats['Avg Trade [$]'])
    assert math.isclose(stats['Profit Factor'], 130.0 / 50.0)
    trades = pd.Series([100.0, -50.0, 30.0])
    assert math.isclose(
        stats['SQN'],
        math.sqrt(3) * trades.mean() / trades.std(ddof=1), rel_tol=1e-12,
    )


def test_all_winning_trades_profit_factor_is_nan():
    stats = _stats([100_000, 100_100], pnls=[50.0, 30.0])
    assert pd.isna(stats['Profit Factor'])
    assert math.isclose(stats['Win Rate [%]'], 100.0)


def test_no_closing_trades_trade_stats_are_nan():
    stats = _stats([100_000, 100_100], pnls=[0.0, 0.0])
    assert stats['# Fills'] == 2
    assert stats['# Closing Trades'] == 0
    for label in ('Win Rate [%]', 'Profit Factor', 'Expectancy [$]', 'SQN',
                  'Best Trade [$]', 'Worst Trade [$]', 'Avg Trade [$]'):
        assert pd.isna(stats[label]), label


# ──────────────────────────────────────────────
# Edge cases — never raise
# ──────────────────────────────────────────────

def test_empty_inputs_return_full_series_of_nans():
    stats = backtest_stats(pd.DataFrame(), pd.DataFrame(),
                           initial_capital=100_000.0, timeframe='1d',
                           convention='crypto')
    assert list(stats.index) == _EXPECTED_LABELS
    assert pd.isna(stats['Start']) and pd.isna(stats['End'])
    assert pd.isna(stats['Equity Final [$]'])
    assert pd.isna(stats['Max Drawdown [$]'])
    assert stats['# Fills'] == 0
    assert stats['# Closing Trades'] == 0
    assert pd.isna(stats['Win Rate [%]'])


def test_constant_balance_zero_vol_is_clean():
    stats = _stats([100_000, 100_000, 100_000, 100_000])
    assert math.isclose(stats['Max Drawdown [$]'], 0.0)
    assert math.isclose(stats['Max Drawdown [%]'], 0.0)
    assert pd.isna(stats['Avg Drawdown [$]'])           # no episodes
    assert pd.isna(stats['Max Drawdown Duration'])
    assert pd.isna(stats['Sharpe Ratio'])               # zero stdev
    assert pd.isna(stats['Sortino Ratio'])
    assert pd.isna(stats['Calmar Ratio'])
    assert math.isclose(stats['Volatility (Ann.) [$]'], 0.0)


# ──────────────────────────────────────────────
# Parameter validation — caller bugs raise
# ──────────────────────────────────────────────

def test_non_dataframe_inputs_rejected():
    with pytest.raises(TypeError):
        backtest_stats('not a frame', _trade_log([]),
                       initial_capital=100_000.0, timeframe='1d',
                       convention='crypto')
    with pytest.raises(TypeError):
        backtest_stats(_equity_curve([100_000]), [1, 2, 3],
                       initial_capital=100_000.0, timeframe='1d',
                       convention='crypto')


def test_non_positive_initial_capital_rejected():
    with pytest.raises(ValueError, match="initial_capital"):
        _stats([100_000], initial_capital=0.0)
    with pytest.raises(ValueError, match="initial_capital"):
        _stats([100_000], initial_capital=-1.0)


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        _stats([100_000], convention='lunar')


def test_missing_required_columns_rejected():
    bad = pd.DataFrame(
        {'cash': [1.0]},
        index=pd.date_range('2024-01-01', periods=1, freq='D', tz='UTC'),
    )
    with pytest.raises(ValueError, match="account_balance"):
        backtest_stats(bad, _trade_log([]), initial_capital=100_000.0,
                       timeframe='1d', convention='crypto')
