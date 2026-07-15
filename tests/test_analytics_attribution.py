"""Unit tests for ``analytics.pnl_attribution``.

Pin:
- Golden hand-computed proportional shares with the one-bar shift
  (forecast at *t* earns the *t*→*t+1* PnL).
- Per-symbol shift alignment under staggered listings.
- Clamp/FDM invariance (shares are scale-free in the contributions).
- Near-cancellation magnitude fallback; all-zero → 'unattributed'.
- Variation carry-forward: shares ffilled from the last valid bar,
  mirroring the forecast cache; variations sum exactly to the strategy.
- ``by()`` pivot contract: dims validation, MultiIndex nesting order,
  every view sums per bar to ``total``.
- Single-``Strategy`` mode and the real-``Orchestrator`` schema guard.
- Parameter validation raises; empty inputs never raise.

Run from the repo root:  python -m pytest tests/test_analytics_attribution.py -v
"""

import numpy as np
import pandas as pd
import pytest

from analytics import PnLAttributionResult, pnl_attribution


# ──────────────────────────────────────────────
# Builders (mirror the real portfolio/record schemas)
# ──────────────────────────────────────────────

def _ts(n, start='2024-01-01'):
    return pd.date_range(start, periods=n, freq='D', tz='UTC')


def _equity_curve(cum_pnl, *, start='2024-01-01') -> pd.DataFrame:
    """Equity-curve frame shaped like ``BacktestPortfolio.get_equity_curve()``.

    ``cum_pnl`` is a list of ``{symbol: cumulative total pnl}`` dicts, one
    per timestamp — written into ``realized_pnl`` with all-zero
    ``unrealized_pnl``. Two rows per timestamp, with garbage dicts in the
    first row so a wrong collapse is caught (last row wins).
    """
    keys = sorted({s for row in cum_pnl for s in row})
    idx = _ts(len(cum_pnl), start)
    rows = []
    for ts, vals in zip(idx, cum_pnl):
        true_r = {s: float(vals.get(s, 0.0)) for s in keys}
        garbage = {s: v + 999.0 for s, v in true_r.items()}
        for realized in (garbage, true_r):
            rows.append({
                'timestamp': ts,
                'symbol': keys[0],
                'realized_pnl': dict(realized),
                'unrealized_pnl': {s: 0.0 for s in keys},
                'account_balance': 0.0,
            })
    return pd.DataFrame(rows).set_index('timestamp')


def _orch_records(idx, forecasts, weights) -> pd.DataFrame:
    """Orchestrator-record frame: ``forecast_<label>`` / ``weight_<label>``."""
    data = {}
    for lab, vals in forecasts.items():
        data[f'forecast_{lab}'] = [float(v) for v in vals]
    for lab, vals in weights.items():
        data[f'weight_{lab}'] = [float(v) for v in vals]
    return pd.DataFrame(data, index=idx)


def _child_records(idx, combined, variations) -> pd.DataFrame:
    """Child-strategy record frame: ``forecast`` + ``forecast_<label>``."""
    data = {'forecast': [float(v) for v in combined]}
    for lab, vals in variations.items():
        data[f'forecast_{lab}'] = [float(v) for v in vals]
    return pd.DataFrame(data, index=idx)


class _FakeChild:
    """Duck-typed strategy: ``symbol_list`` + ``get_records`` (+ optional
    ``weights`` for the variation convention)."""

    def __init__(self, symbol_list, records=None, weights=None):
        self.symbol_list = list(symbol_list)
        self._records = dict(records or {})
        if weights is not None:
            self.weights = dict(weights)

    def get_records(self, symbol):
        return self._records.get(symbol, pd.DataFrame())


class _FakeOrchestrator:
    """Duck-typed orchestrator: ``strategies`` dict + ``symbol_list`` +
    ``get_records``."""

    def __init__(self, strategies, symbol_list, records):
        self.strategies = dict(strategies)
        self.symbol_list = list(symbol_list)
        self._records = dict(records)

    def get_records(self, symbol):
        return self._records.get(symbol, pd.DataFrame())


def _two_strategy_system(idx, forecasts, weights, symbols=('AAA',)):
    """Orchestrator with two variation-less children sharing one record
    frame across ``symbols``."""
    rec = _orch_records(idx, forecasts, weights)
    return _FakeOrchestrator(
        strategies={lab: _FakeChild(symbols) for lab in forecasts},
        symbol_list=list(symbols),
        records={s: rec for s in symbols},
    )


NAN = float('nan')


# ──────────────────────────────────────────────
# Golden attribution (strategy level)
# ──────────────────────────────────────────────

def test_golden_two_strategies_constant_shares():
    # Contributions 0.5*20 = 10 and 0.5*10 = 5 → shares 2/3, 1/3.
    idx = _ts(4)
    system = _two_strategy_system(
        idx,
        forecasts={'a': [20, 20, 20, 20], 'b': [10, 10, 10, 10]},
        weights={'a': [0.5] * 4, 'b': [0.5] * 4},
    )
    eq = _equity_curve([{'AAA': v} for v in (0, 30, 15, 45)])
    res = pnl_attribution(eq, system)

    by = res.by('strategy')
    np.testing.assert_allclose(by['a'].to_numpy(), [0, 20, -10, 20],
                               atol=1e-12)
    np.testing.assert_allclose(by['b'].to_numpy(), [0, 10, -5, 10],
                               atol=1e-12)
    assert 'unattributed' not in by.columns  # pnl was 0 before coverage
    # Variation-less children collapse to the 'all' passthrough bucket.
    assert list(res.by('strategy', 'variation').columns) == [
        ('a', 'all'), ('b', 'all'),
    ]


def test_one_bar_shift_uses_prior_bar_shares():
    # Shares flip 100% a → 100% b between t0 and t1; PnL at t+1 must use
    # the share from t.
    idx = _ts(3)
    system = _two_strategy_system(
        idx,
        forecasts={'a': [20, 0, 0], 'b': [0, 20, 20]},
        weights={'a': [0.5] * 3, 'b': [0.5] * 3},
    )
    eq = _equity_curve([{'AAA': v} for v in (0, 10, 30)])
    res = pnl_attribution(eq, system)

    by = res.by('strategy')
    np.testing.assert_allclose(by['a'].to_numpy(), [0, 10, 0], atol=1e-12)
    np.testing.assert_allclose(by['b'].to_numpy(), [0, 0, 20], atol=1e-12)


def test_staggered_listing_shifts_per_symbol():
    # Symbol B's records start at t2 — its t3 PnL must use B's own t2
    # shares (a global shift on the union axis would misalign them).
    idx = _ts(4)
    rec_a = _orch_records(
        idx,
        forecasts={'a': [10] * 4, 'b': [10] * 4},
        weights={'a': [0.5] * 4, 'b': [0.5] * 4},
    )
    rec_b = _orch_records(
        idx[2:],
        forecasts={'a': [10, 0], 'b': [0, 10]},
        weights={'a': [0.5] * 2, 'b': [0.5] * 2},
    )
    system = _FakeOrchestrator(
        strategies={'a': _FakeChild(['AAA', 'BBB']),
                    'b': _FakeChild(['AAA', 'BBB'])},
        symbol_list=['AAA', 'BBB'],
        records={'AAA': rec_a, 'BBB': rec_b},
    )
    eq = _equity_curve([
        {'AAA': 0, 'BBB': 0},
        {'AAA': 10, 'BBB': 0},
        {'AAA': 10, 'BBB': 0},
        {'AAA': 10, 'BBB': 50},
    ])
    res = pnl_attribution(eq, system)

    by_sym = res.by('symbol', 'strategy')
    # BBB's t3 PnL uses BBB's t2 shares (100% strategy 'a').
    np.testing.assert_allclose(by_sym[('BBB', 'a')].to_numpy(),
                               [0, 0, 0, 50], atol=1e-12)
    assert ('BBB', 'b') not in by_sym.columns


def test_shares_invariant_to_contribution_scaling():
    # Scaling every contribution equally (what the cap and the FDM do to
    # the combined forecast) leaves the shares untouched.
    idx = _ts(3)
    eq = _equity_curve([{'AAA': v} for v in (0, 30, 60)])
    base = _two_strategy_system(
        idx,
        forecasts={'a': [20] * 3, 'b': [10] * 3},
        weights={'a': [0.5] * 3, 'b': [0.5] * 3},
    )
    scaled = _two_strategy_system(
        idx,
        forecasts={'a': [200] * 3, 'b': [100] * 3},
        weights={'a': [0.5] * 3, 'b': [0.5] * 3},
    )
    pd.testing.assert_frame_equal(
        pnl_attribution(eq, base).by('strategy'),
        pnl_attribution(eq, scaled).by('strategy'),
    )


# ──────────────────────────────────────────────
# Edge rules: near-cancellation, unattributed
# ──────────────────────────────────────────────

def test_near_cancellation_falls_back_to_magnitude_shares():
    idx = _ts(2)
    system = _two_strategy_system(
        idx,
        forecasts={'a': [10, 10], 'b': [-10, -10]},
        weights={'a': [0.5] * 2, 'b': [0.5] * 2},
    )
    eq = _equity_curve([{'AAA': 0}, {'AAA': 100}])
    res = pnl_attribution(eq, system)

    by = res.by('strategy')
    np.testing.assert_allclose(by['a'].to_numpy(), [0, 50], atol=1e-12)
    np.testing.assert_allclose(by['b'].to_numpy(), [0, 50], atol=1e-12)


def test_pnl_before_first_forecast_goes_to_unattributed():
    # No contribution at t0/t1 (NaN forecasts, zero weights): the t1 PnL
    # has no prior forecast to attribute to.
    idx = _ts(4)
    system = _two_strategy_system(
        idx,
        forecasts={'a': [NAN, NAN, 20, 20], 'b': [NAN, NAN, 10, 10]},
        weights={'a': [0, 0, 0.5, 0.5], 'b': [0, 0, 0.5, 0.5]},
    )
    eq = _equity_curve([{'AAA': v} for v in (0, 40, 40, 70)])
    res = pnl_attribution(eq, system)

    by = res.by('strategy')
    np.testing.assert_allclose(by['unattributed'].to_numpy(),
                               [0, 40, 0, 0], atol=1e-12)
    np.testing.assert_allclose(by['a'].to_numpy(), [0, 0, 0, 20],
                               atol=1e-12)
    np.testing.assert_allclose(by['b'].to_numpy(), [0, 0, 0, 10],
                               atol=1e-12)


def test_unmanaged_symbol_pnl_goes_to_unattributed():
    idx = _ts(2)
    system = _two_strategy_system(
        idx,
        forecasts={'a': [20] * 2, 'b': [10] * 2},
        weights={'a': [0.5] * 2, 'b': [0.5] * 2},
        symbols=('AAA',),
    )
    eq = _equity_curve([{'AAA': 0, 'ZZZ': 0}, {'AAA': 30, 'ZZZ': 7}])
    res = pnl_attribution(eq, system)

    by_sym = res.by('symbol', 'strategy')
    np.testing.assert_allclose(
        by_sym[('ZZZ', 'unattributed')].to_numpy(), [0, 7], atol=1e-12)
    np.testing.assert_allclose(res.total.to_numpy(), [0, 37], atol=1e-12)


# ──────────────────────────────────────────────
# Variation level
# ──────────────────────────────────────────────

def _single_child_system(idx, crec, symbols=('AAA',), weights=None):
    """Orchestrator wrapping ONE child (weight 1.0 once cached), whose
    variation records are ``crec``."""
    cached = pd.Series(crec['forecast']).ffill()
    rec = _orch_records(
        idx,
        forecasts={'a': cached.tolist()},
        weights={'a': [0.0 if pd.isna(v) else 1.0 for v in cached]},
    )
    child = _FakeChild(symbols, records={s: crec for s in symbols},
                       weights=weights or {'x': 0.5, 'y': 0.5})
    return _FakeOrchestrator(strategies={'a': child},
                             symbol_list=list(symbols),
                             records={s: rec for s in symbols})


def test_golden_variation_split():
    # Variation contributions 0.5*30 = 15 and 0.5*10 = 5 → 0.75 / 0.25.
    idx = _ts(3)
    crec = _child_records(idx, combined=[20, 20, 20],
                          variations={'x': [30] * 3, 'y': [10] * 3})
    system = _single_child_system(idx, crec)
    eq = _equity_curve([{'AAA': v} for v in (0, 40, 40)])
    res = pnl_attribution(eq, system)

    by_var = res.by('strategy', 'variation')
    np.testing.assert_allclose(by_var[('a', 'x')].to_numpy(), [0, 30, 0],
                               atol=1e-12)
    np.testing.assert_allclose(by_var[('a', 'y')].to_numpy(), [0, 10, 0],
                               atol=1e-12)


def test_variation_shares_carry_forward_across_nan_bars():
    # t1 valid (mix 0.75/0.25), t2 NaN (cache carries forward), t3 valid
    # (mix 0.5/0.5). The t3 PnL uses t2's shares = ffilled t1 mix.
    idx = _ts(4)
    crec = _child_records(
        idx,
        combined=[NAN, 20, NAN, 10],
        variations={'x': [NAN, 30, NAN, 10], 'y': [NAN, 10, 5, 10]},
    )
    system = _single_child_system(idx, crec)
    eq = _equity_curve([{'AAA': v} for v in (0, 0, 10, 30)])
    res = pnl_attribution(eq, system)

    by_var = res.by('strategy', 'variation')
    # t2 PnL 10 ← t1 shares (0.75/0.25); t3 PnL 20 ← t2 shares, ffilled
    # from t1 because t2's own row is invalid.
    np.testing.assert_allclose(by_var[('a', 'x')].to_numpy(),
                               [0, 0, 7.5, 15], atol=1e-12)
    np.testing.assert_allclose(by_var[('a', 'y')].to_numpy(),
                               [0, 0, 2.5, 5], atol=1e-12)
    # Variations sum exactly to the strategy column.
    strat = res.by('strategy')['a']
    np.testing.assert_allclose(
        by_var[('a', 'x')] + by_var[('a', 'y')], strat, atol=1e-12)


# ──────────────────────────────────────────────
# Single-Strategy mode
# ──────────────────────────────────────────────

def test_single_strategy_mode_uses_class_name_and_variations():
    idx = _ts(3)
    crec = _child_records(idx, combined=[NAN, 20, 20],
                          variations={'x': [NAN, 30, 30], 'y': [NAN, 10, 10]})
    system = _FakeChild(['AAA'], records={'AAA': crec},
                        weights={'x': 0.5, 'y': 0.5})
    eq = _equity_curve([{'AAA': v} for v in (0, 5, 45)])
    res = pnl_attribution(eq, system)

    by = res.by('strategy')
    assert '_FakeChild' in by.columns
    # t1 PnL 5 predates the first forecast → unattributed; t2 PnL 40 is
    # the strategy's, split 0.75/0.25 across variations.
    np.testing.assert_allclose(by['unattributed'].to_numpy(), [0, 5, 0],
                               atol=1e-12)
    np.testing.assert_allclose(by['_FakeChild'].to_numpy(), [0, 0, 40],
                               atol=1e-12)
    by_var = res.by('strategy', 'variation')
    np.testing.assert_allclose(by_var[('_FakeChild', 'x')].to_numpy(),
                               [0, 0, 30], atol=1e-12)


# ──────────────────────────────────────────────
# Real Orchestrator schema guard
# ──────────────────────────────────────────────

def test_real_orchestrator_record_schema():
    from event import BarEvent
    from orchestrator import Orchestrator
    from strategy import Strategy

    class _ConstLong(Strategy):
        def calculate_forecast(self, event):
            return {'forecast': 20.0}

    class _ConstShort(Strategy):
        def calculate_forecast(self, event):
            return {'forecast': -10.0}

    symbols = ['AAA']
    orch = Orchestrator({'long': _ConstLong(None, symbols),
                         'short': _ConstShort(None, symbols)})
    idx = _ts(3)
    for ts in idx:
        orch.update_bar(BarEvent(symbol='AAA', timestamp=ts, open=1.0,
                                 high=1.0, low=1.0, close=1.0, volume=0.0))

    eq = _equity_curve([{'AAA': v} for v in (0, 10, 25)])
    res = pnl_attribution(eq, orch)

    # Equal weights, contributions +10 / −5 → signed shares +2 / −1.
    by = res.by('strategy')
    np.testing.assert_allclose(by['long'].to_numpy(), [0, 20, 30],
                               atol=1e-9)
    np.testing.assert_allclose(by['short'].to_numpy(), [0, -10, -15],
                               atol=1e-9)
    np.testing.assert_allclose(by.sum(axis=1), res.total, atol=1e-9)


# ──────────────────────────────────────────────
# by() pivot contract & reconciliation
# ──────────────────────────────────────────────

def _random_result(seed=7):
    rng = np.random.default_rng(seed)
    n = 25
    idx = _ts(n)
    symbols = ['AAA', 'BBB', 'CCC']
    labels = ['s1', 's2']
    records, child_recs = {}, {lab: {} for lab in labels}
    for sym in symbols:
        warm = int(rng.integers(0, 5))
        forecasts, weights = {}, {}
        for lab, w in zip(labels, (0.6, 0.4)):
            f = rng.normal(0, 30, n)
            f[:warm] = np.nan
            forecasts[lab] = f
            weights[lab] = np.where(np.isnan(f), 0.0, w)
        records[sym] = _orch_records(idx, forecasts, weights)
        for lab in labels:
            child_recs[lab][sym] = _child_records(
                idx, combined=forecasts[lab],
                variations={'x': rng.normal(0, 30, n),
                            'y': rng.normal(0, 30, n)},
            )
    system = _FakeOrchestrator(
        strategies={lab: _FakeChild(symbols, records=child_recs[lab],
                                    weights={'x': 0.5, 'y': 0.5})
                    for lab in labels},
        symbol_list=symbols,
        records=records,
    )
    cum = np.cumsum(rng.normal(0, 100, (n, len(symbols))), axis=0)
    eq = _equity_curve([dict(zip(symbols, row)) for row in cum])
    return pnl_attribution(eq, system), eq


def test_every_view_sums_to_total():
    res, _ = _random_result()
    for dims in [('strategy',), ('strategy', 'variation'), ('symbol',),
                 ('symbol', 'strategy'), ('variation', 'symbol')]:
        view = res.by(*dims)
        np.testing.assert_allclose(view.sum(axis=1), res.total,
                                   atol=1e-9, err_msg=str(dims))


def test_total_cumsum_matches_final_cumulative_pnl():
    res, eq = _random_result(seed=11)
    final = sum(eq['realized_pnl'].iloc[-1].values())
    assert res.total.cumsum().iloc[-1] == pytest.approx(final, abs=1e-9)


def test_by_multiindex_nesting_order():
    res, _ = _random_result()
    fwd = res.by('symbol', 'strategy')
    rev = res.by('strategy', 'symbol')
    assert fwd.columns.names == ['symbol', 'strategy']
    assert rev.columns.names == ['strategy', 'symbol']
    pair = fwd.columns[0]
    assert (pair[1], pair[0]) in rev.columns
    single = res.by('strategy')
    assert not isinstance(single.columns, pd.MultiIndex)


def test_by_rejects_bad_dims():
    res, _ = _random_result()
    with pytest.raises(ValueError, match="at least one"):
        res.by()
    with pytest.raises(ValueError, match="Unknown attribution dimension"):
        res.by('strategy', 'bogus')
    with pytest.raises(ValueError, match="Duplicate"):
        res.by('strategy', 'strategy')


def test_by_variation_convenience_matches_by():
    res, _ = _random_result()
    pd.testing.assert_frame_equal(res.by_strategy, res.by('strategy'))
    pd.testing.assert_frame_equal(res.by_variation,
                                  res.by('strategy', 'variation'))


# ──────────────────────────────────────────────
# Edge cases & parameter validation
# ──────────────────────────────────────────────

def test_empty_equity_curve_returns_empty_result():
    system = _two_strategy_system(
        _ts(2), forecasts={'a': [1, 1], 'b': [1, 1]},
        weights={'a': [0.5] * 2, 'b': [0.5] * 2},
    )
    res = pnl_attribution(pd.DataFrame(), system)
    assert isinstance(res, PnLAttributionResult)
    assert res.long.empty
    assert res.total.empty
    assert res.by('strategy').empty


def test_non_dataframe_equity_curve_raises():
    system = _FakeChild(['AAA'])
    with pytest.raises(TypeError, match="equity_curve must be a DataFrame"):
        pnl_attribution([1, 2, 3], system)


def test_system_without_records_surface_raises():
    with pytest.raises(TypeError, match="get_records"):
        pnl_attribution(pd.DataFrame(), object())

    class _NoSymbols:
        def get_records(self, symbol):
            return pd.DataFrame()

    with pytest.raises(TypeError, match="symbol_list"):
        pnl_attribution(pd.DataFrame(), _NoSymbols())


def test_bad_eps_raises():
    system = _FakeChild(['AAA'])
    with pytest.raises(ValueError, match="eps"):
        pnl_attribution(pd.DataFrame(), system, eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        pnl_attribution(pd.DataFrame(), system, eps=float('nan'))


def test_missing_pnl_columns_raises():
    system = _FakeChild(['AAA'])
    bad = pd.DataFrame({'account_balance': [1.0]}, index=_ts(1))
    with pytest.raises(ValueError, match="missing required columns"):
        pnl_attribution(bad, system)


def test_reserved_labels_raise():
    idx = _ts(2)
    eq = _equity_curve([{'AAA': 0}, {'AAA': 1}])
    bad_strategy = _FakeOrchestrator(
        strategies={'unattributed': _FakeChild(['AAA'])},
        symbol_list=['AAA'],
        records={'AAA': _orch_records(idx, {'unattributed': [1, 1]},
                                      {'unattributed': [1.0, 1.0]})},
    )
    with pytest.raises(ValueError, match="reserved"):
        pnl_attribution(eq, bad_strategy)

    bad_variation = _FakeChild(['AAA'], weights={'all': 1.0})
    with pytest.raises(ValueError, match="reserved"):
        pnl_attribution(eq, bad_variation)


def test_missing_orchestrator_columns_raises():
    idx = _ts(2)
    rec = pd.DataFrame({'forecast_a': [1.0, 1.0]}, index=idx)  # no weight_a
    system = _FakeOrchestrator(
        strategies={'a': _FakeChild(['AAA'])},
        symbol_list=['AAA'],
        records={'AAA': rec},
    )
    eq = _equity_curve([{'AAA': 0}, {'AAA': 5}])
    with pytest.raises(ValueError, match="missing orchestrator"):
        pnl_attribution(eq, system)


def test_pnl_attribution_naive_curve_raises():
    """UTC law (tz audit 2026-07): a naive equity-curve index raises."""
    class _Sys:
        symbol_list = ['BTC']

        def get_records(self, symbol):
            return pd.DataFrame()

    idx = pd.to_datetime(['2024-01-01', '2024-01-02'])      # tz-naive
    eq = pd.DataFrame({'account_balance': [100.0, 110.0],
                       'total_commission': [0.0, 0.0]}, index=idx)
    with pytest.raises(ValueError, match="equity_curve.*timezone-naive"):
        pnl_attribution(eq, _Sys())
