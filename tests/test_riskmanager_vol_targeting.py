"""Unit tests for the slimmed VolTargetingRiskManager.

The RM no longer derives rho or owns universe state: weights/IDM arrive
via on_correlation_event payloads, universe state is read from a real
UniverseManager, and sizing enforces the universal not-live rule.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from event import BarEvent, CorrelationEvent, Direction, OrderType, UniverseEvent
from riskmanager import VolTargetingRiskManager
from universe import UniverseManager

T0 = pd.Timestamp('2024-01-01', tz='UTC')


class StubStrategy:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)
        self.forecasts: Dict[str, Optional[float]] = {}
        self._warm = set()

    def get_forecast(self, symbol):
        return self.forecasts.get(symbol)

    def is_warmed_up(self, symbol):
        return symbol in self._warm


class StubDataHandler:
    def __init__(self, symbols, maxlen=500, timeframe='1d'):
        self.timeframes = {timeframe: maxlen}
        self.counts = {s: 0 for s in symbols}

    def count_bars(self, symbol, timeframe=None):
        return self.counts.get(symbol, 0)


class StubVolEstimator:
    def __init__(self):
        self.sigma: Dict[str, Optional[float]] = {}

    def update(self, event):
        return None

    def get_annual_vol(self, symbol):
        return self.sigma.get(symbol)


class StubPortfolio:
    def __init__(self):
        self.positions: Dict[str, float] = {}
        self._pending: Dict[str, float] = {}
        self.balance = 1_000_000.0
        self.submitted: List[dict] = []

    def get_price(self, symbol):
        return 100.0

    def calculate_balance(self):
        return self.balance

    def projected_position(self, symbol):
        return self.positions.get(symbol, 0.0) + self._pending.get(symbol, 0.0)

    def submit_order(self, symbol, quantity, direction, timestamp, order_type,
                     price=None, is_liquidation=False, fill_on_next_bar=False):
        signed = quantity if direction == Direction.BUY else -quantity
        self._pending[symbol] = self._pending.get(symbol, 0.0) + signed
        self.submitted.append(dict(symbol=symbol, quantity=quantity,
                                   direction=direction, timestamp=timestamp,
                                   order_type=order_type,
                                   fill_on_next_bar=fill_on_next_bar))
        return object()


def _bar(symbol, ts=T0, close=100.0, forming=False):
    return BarEvent(symbol=symbol, timestamp=ts, open=close, high=close,
                    low=close, close=close, volume=1.0, period='1d',
                    is_forming=forming)


def _corr_event(reason, live, matrix=None, ts=T0):
    return CorrelationEvent(timestamp=ts, matrix=matrix,
                            live_symbols=list(live), reason=reason)


def _identityish(labels, off_diag=0.0):
    n = len(labels)
    m = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


def _build(symbols=('A', 'B'), min_history=1, **rm_kwargs):
    strat = StubStrategy(symbols)
    dh = StubDataHandler(symbols)
    um = UniverseManager(strat, dh, min_history_bars=min_history)
    pf = StubPortfolio()
    ve = StubVolEstimator()
    rm = VolTargetingRiskManager(
        pf, strat, ve, um,
        annual_target_vol=rm_kwargs.pop('annual_target_vol', 250_000.0),
        **rm_kwargs,
    )
    return rm, pf, strat, dh, um, ve


def _make_live(um, strat, dh, *symbols):
    for s in symbols:
        strat._warm.add(s)
        dh.counts[s] = um.min_history_bars
        um.update_bar(_bar(s))
    um.drain_events()


class TestConstructor:
    def test_weights_start_empty_no_construction_recalc(self):
        rm, *_ = _build()
        assert rm.instrument_weight == {}
        assert rm.idm == 1.0

    def test_removed_params_are_gone(self):
        rm, pf, strat, dh, um, ve = _build()
        with pytest.raises(TypeError):
            VolTargetingRiskManager(pf, strat, ve, um,
                                    annual_target_vol=1.0, corr_lookback=60)
        with pytest.raises(TypeError):
            VolTargetingRiskManager(pf, strat, ve, um,
                                    annual_target_vol=1.0, data_handler=dh)

    def test_deleted_public_surface(self):
        rm, *_ = _build()
        assert not hasattr(rm, 'calculate_instrument_weight')
        assert not hasattr(rm, 'get_live_symbols')
        assert not hasattr(rm, 'remove_symbol')

    def test_validation_carried_over(self):
        rm, pf, strat, dh, um, ve = _build()
        with pytest.raises(ValueError):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=None)
        with pytest.raises(ValueError):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    vol_target_mode='bogus')
        with pytest.raises(ValueError):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    instrument_weight_mode='bogus')
        with pytest.raises(ValueError):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    idm=3.0, idm_cap=2.5)


class TestOnCorrelationEvent:
    def test_ok_equal_weight_sets_weights_and_idm(self):
        rm, pf, strat, dh, um, ve = _build()
        m = _identityish(['A', 'B'])
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'], m))
        assert rm.instrument_weight == pytest.approx({'A': 0.5, 'B': 0.5})
        assert rm.idm == pytest.approx(np.sqrt(2))        # uncorrelated pair
        assert rm._corr_matrix_cache is m

    def test_idm_cap_binds(self):
        rm, *_ = _build(symbols=tuple('ABCDEFGH'), idm_cap=2.5)
        labels = list('ABCDEFGH')
        rm.on_correlation_event(_corr_event('ok', labels,
                                            _identityish(labels)))
        assert rm.idm == pytest.approx(2.5)               # sqrt(8) capped

    def test_fallback_reasons_equal_weight_over_live_idm_untouched(self):
        rm, *_ = _build()
        rm.idm = 1.7
        for reason in ('insufficient_observations', 'too_few_symbols',
                       'nan_fallback'):
            rm.on_correlation_event(_corr_event(reason, ['A', 'B']))
            assert rm.instrument_weight == pytest.approx({'A': 0.5, 'B': 0.5})
            assert rm.idm == 1.7

    def test_empty_universe_clears_weights(self):
        rm, *_ = _build()
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'],
                                            _identityish(['A', 'B'])))
        rm.on_correlation_event(_corr_event('empty_universe', []))
        assert rm.instrument_weight == {}

    def test_singleton(self):
        rm, *_ = _build()
        rm.on_correlation_event(_corr_event('singleton', ['A']))
        assert rm.instrument_weight == {'A': 1.0}
        assert rm.idm == 1.0

    def test_unknown_reason_raises(self):
        rm, *_ = _build()
        with pytest.raises(ValueError):
            rm.on_correlation_event(_corr_event('bogus', ['A']))

    def test_risk_parity_mode_uses_optimizer(self):
        rm, *_ = _build(instrument_weight_mode='risk_parity')
        m = _identityish(['A', 'B'], off_diag=0.5)
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'], m))
        # Symmetric 2-asset ERC == 50/50 regardless of correlation:
        assert rm.instrument_weight == pytest.approx({'A': 0.5, 'B': 0.5},
                                                     abs=1e-6)


class TestBudgetGroups:
    class BudgetStrategy(StubStrategy):
        def __init__(self, symbols, groups):
            super().__init__(symbols)
            self._groups = groups

        def get_budget_groups(self):
            return self._groups

    def test_disjoint_groups_scale_by_budget(self):
        strat = self.BudgetStrategy(
            ['A', 'B', 'C', 'D'],
            {'g1': (0.75, ['A', 'B']), 'g2': (0.25, ['C', 'D'])},
        )
        dh = StubDataHandler(strat.symbol_list)
        um = UniverseManager(strat, dh, min_history_bars=1)
        rm = VolTargetingRiskManager(StubPortfolio(), strat,
                                     StubVolEstimator(), um,
                                     annual_target_vol=250_000.0)
        labels = ['A', 'B', 'C', 'D']
        rm.on_correlation_event(_corr_event('ok', labels,
                                            _identityish(labels)))
        assert rm.instrument_weight == pytest.approx(
            {'A': 0.375, 'B': 0.375, 'C': 0.125, 'D': 0.125})


class TestSizingSkipLadder:
    def _sized(self, rm, pf, strat, dh, um, ve, symbol='A',
               forecast=50.0, sigma=10.0, weight=None):
        _make_live(um, strat, dh, symbol)
        strat.forecasts[symbol] = forecast
        ve.sigma[symbol] = sigma
        if weight is not None:
            rm.instrument_weight = dict(weight)
        rm.update_bar(_bar(symbol))
        return rm.get_records(symbol).iloc[-1]

    def test_full_pipeline_submits_carver_target(self):
        rm, pf, strat, dh, um, ve = _build()
        row = self._sized(rm, pf, strat, dh, um, ve,
                          weight={'A': 1.0})
        # dollar_volatility: target = idm*iw*tau*(f/50) / (pv*sigma)
        assert row['target_qty'] == pytest.approx(
            1.0 * 1.0 * 250_000.0 * 1.0 / (1.0 * 10.0))
        assert row['submitted'] is True or row['submitted'] == True
        assert pf.submitted and pf.submitted[-1]['symbol'] == 'A'

    def test_not_live_flat_skips_with_primary_reason(self):
        rm, pf, strat, dh, um, ve = _build()
        rm.update_bar(_bar('A'))                     # never made live
        row = rm.get_records('A').iloc[-1]
        assert row['skip_reason'] == 'warmup_forecast'
        assert row['submitted'] == False

    def test_not_live_mark_flattens_held_position_ahead_of_sigma(self):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        pf.positions['A'] = 5.0
        um.mark_excluded('A', 'delisted', T0)
        um.drain_events()                            # RM not notified (unit test)
        ve.sigma['A'] = None                         # dead sigma cannot block
        rm.update_bar(_bar('A'))
        assert pf.submitted[-1]['symbol'] == 'A'
        assert pf.submitted[-1]['quantity'] == pytest.approx(5.0)
        assert pf.submitted[-1]['direction'] == Direction.SELL

    def test_live_unweighted_skips_waiting_weight_recalc(self):
        rm, pf, strat, dh, um, ve = _build()
        row = self._sized(rm, pf, strat, dh, um, ve, weight={})
        assert row['skip_reason'] == 'waiting_weight_recalc'

    def test_warmup_volatility_and_zero_vol(self):
        rm, pf, strat, dh, um, ve = _build()
        row = self._sized(rm, pf, strat, dh, um, ve, sigma=None,
                          weight={'A': 1.0})
        assert row['skip_reason'] == 'warmup_volatility'
        row = self._sized(rm, pf, strat, dh, um, ve, sigma=1e-9,
                          weight={'A': 1.0})
        assert row['skip_reason'] == 'zero_vol'

    def test_none_forecast_backstop(self):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        ve.sigma['A'] = 10.0
        rm.instrument_weight = {'A': 1.0}
        strat.forecasts.pop('A', None)
        rm.update_bar(_bar('A'))
        assert rm.get_records('A').iloc[-1]['skip_reason'] == 'warmup_forecast'

    def test_records_drop_universe_columns(self):
        rm, pf, strat, dh, um, ve = _build()
        self._sized(rm, pf, strat, dh, um, ve, weight={'A': 1.0})
        cols = rm.get_records('A').columns
        assert 'universe_live' not in cols and 'universe_reasons' not in cols

    def test_dead_band_and_at_target(self):
        rm, pf, strat, dh, um, ve = _build(position_buffer=0.25)
        row = self._sized(rm, pf, strat, dh, um, ve, weight={'A': 1.0})
        target = row['target_qty']
        pf.positions['A'] = target                   # exactly at target
        pf._pending.clear()
        rm.update_bar(_bar('A'))
        assert rm.get_records('A').iloc[-1]['skip_reason'] == 'at_target'
        pf.positions['A'] = target * 0.9             # inside 25% buffer
        rm.update_bar(_bar('A'))
        assert rm.get_records('A').iloc[-1]['skip_reason'] == 'dead_band'


def _not_live_edge(symbol, reasons=('delisted',), ts=T0):
    """Build a not-live-EDGE ``UniverseEvent`` (``prev_live=True`` ->
    ``live=False``) — the only transition ``on_universe_event`` acts on."""
    return UniverseEvent(timestamp=ts, symbol=symbol, live=False,
                         excluded=True, reasons=list(reasons),
                         prev_live=True, prev_reasons=[],
                         trigger=f'mark_excluded:{reasons[0]}')


class TestOnUniverseEvent:
    def test_not_live_edge_flattens_same_bar_with_fill_on_next_bar(self):
        rm, pf, strat, dh, um, ve = _build()
        pf.positions['A'] = 4.0
        rm.on_universe_event(_not_live_edge('A'))
        sub = pf.submitted[-1]
        assert sub['symbol'] == 'A' and sub['quantity'] == pytest.approx(4.0)
        assert sub['direction'] == Direction.SELL
        assert sub['fill_on_next_bar'] is True
        assert sub['order_type'] == OrderType.MKT

    def test_flat_symbol_submits_nothing(self):
        rm, pf, strat, dh, um, ve = _build()
        rm.on_universe_event(_not_live_edge('A'))
        assert pf.submitted == []

    def test_projected_position_prevents_double_flatten(self):
        rm, pf, strat, dh, um, ve = _build()
        pf.positions['A'] = 4.0
        rm.on_universe_event(_not_live_edge('A'))
        assert len(pf.submitted) == 1
        rm.on_universe_event(_not_live_edge('A'))     # projected already 0
        assert len(pf.submitted) == 1

    def test_pop_and_same_bar_rescale(self):
        rm, pf, strat, dh, um, ve = _build(symbols=('A', 'B', 'C'))
        rm.instrument_weight = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        rm.on_universe_event(_not_live_edge('A'))
        assert rm.instrument_weight == pytest.approx({'B': 0.6, 'C': 0.4})

    def test_idm_recomputed_from_cached_submatrix(self):
        rm, pf, strat, dh, um, ve = _build(symbols=('A', 'B', 'C'))
        labels = ['A', 'B', 'C']
        rm.on_correlation_event(_corr_event('ok', labels,
                                            _identityish(labels)))
        assert rm.idm == pytest.approx(np.sqrt(3))
        rm.on_universe_event(_not_live_edge('A'))
        assert rm.idm == pytest.approx(np.sqrt(2))    # 2 uncorrelated survivors

    def test_single_survivor_sets_idm_one(self):
        rm, pf, strat, dh, um, ve = _build()
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'],
                                            _identityish(['A', 'B'])))
        rm.on_universe_event(_not_live_edge('A'))
        assert rm.instrument_weight == {'B': 1.0}
        assert rm.idm == 1.0

    def test_no_cache_leaves_idm(self):
        rm, pf, strat, dh, um, ve = _build(symbols=('A', 'B', 'C'))
        rm.idm = 1.7
        rm.instrument_weight = {'A': 0.5, 'B': 0.25, 'C': 0.25}
        rm.on_universe_event(_not_live_edge('A'))
        assert rm.idm == 1.7

    def test_stale_cache_missing_survivor_leaves_idm_but_still_rescales(self):
        """CRITICAL guard (Task 7 review carry-over): the cached matrix
        must cover EVERY survivor, not just some — a survivor absent from
        it (e.g. it went live only after the last correlation refresh)
        makes the cache stale/incoherent for the current book, so the IDM
        recompute must be skipped even though the cache is non-None and
        even though it *does* cover some of the survivors. Weight rescale
        is unconditional and must still happen."""
        rm, pf, strat, dh, um, ve = _build(symbols=('A', 'B', 'C'))
        # Cache covers only A, B — C is absent (never appeared in a
        # correlation refresh, e.g. it went live afterward).
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'],
                                            _identityish(['A', 'B'])))
        assert rm.idm == pytest.approx(np.sqrt(2))
        rm.idm = 1.7                                  # sentinel
        rm.instrument_weight = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        rm.on_universe_event(_not_live_edge('A'))      # survivors: B, C
        assert rm.instrument_weight == pytest.approx({'B': 0.6, 'C': 0.4})
        assert rm.idm == 1.7                           # untouched, not recomputed

    def test_live_edge_is_a_no_op(self):
        rm, pf, strat, dh, um, ve = _build()
        rm.instrument_weight = {'B': 1.0}
        rm.on_universe_event(UniverseEvent(
            timestamp=T0, symbol='A', live=True, excluded=False, reasons=[],
            prev_live=False, prev_reasons=['warmup_history'],
            trigger='bar_refresh'))
        assert rm.instrument_weight == {'B': 1.0} and pf.submitted == []
