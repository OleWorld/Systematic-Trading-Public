"""Unit tests for the slimmed VolTargetingRiskManager.

The RM no longer derives rho or owns universe state: weights/IDM arrive
via on_correlation_event payloads, universe state is read from a real
UniverseManager, and sizing enforces the universal not-live rule.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from config import uniform_registry
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

    def test_not_live_edge_warns_on_held_position(self, caplog):
        """Ports ``test_held_position_absent_from_universe_warns``: the
        flatten WARNING (``on_universe_event``, riskmanager/_vol_targeting.py
        ~517-522) fires at WARNING level on the riskmanager logger, naming
        the symbol and its universe reasons — anchored to the (already
        separately covered) flatten submission so the log line is pinned
        to the exact branch that emits it, not just "some warning fired
        somewhere"."""
        rm, pf, strat, dh, um, ve = _build()
        pf.positions['A'] = 4.0
        with caplog.at_level(logging.WARNING, logger='riskmanager._vol_targeting'):
            rm.on_universe_event(_not_live_edge('A', reasons=('delisted',)))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any('A went not-live' in r.message and 'delisted' in r.message
                  for r in warnings)
        sub = pf.submitted[-1]
        assert sub['symbol'] == 'A' and sub['quantity'] == pytest.approx(4.0)

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


# ──────────────────────────────────────────────────────────────────────
# Deferred coverage (Task 7 review checklist, closed in Task 12) +
# old-suite audit ports (pre-split tests/test_riskmanager_vol_targeting.py
# @ 6bb381d) rewired through on_correlation_event per the task-12 brief.
# ──────────────────────────────────────────────────────────────────────

class TestConstructorValidationExtra:
    """idm <= 0 and position_buffer-out-of-range still raise (the 6 params
    the slim RM kept from the pre-split constructor); ports
    ``test_constructor_rejects_non_positive_idm`` /
    ``test_constructor_rejects_position_buffer_outside_unit_interval`` /
    the idm_cap default+bounds tests, none of which had a post-split
    equivalent."""

    def test_idm_must_be_positive(self):
        rm, pf, strat, dh, um, ve = _build()
        with pytest.raises(ValueError, match='idm'):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    idm=0.0)
        with pytest.raises(ValueError, match='idm'):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    idm=float('nan'))

    def test_position_buffer_must_be_in_unit_interval(self):
        rm, pf, strat, dh, um, ve = _build()
        with pytest.raises(ValueError, match='position_buffer'):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    position_buffer=-0.1)
        with pytest.raises(ValueError, match='position_buffer'):
            VolTargetingRiskManager(pf, strat, ve, um, annual_target_vol=1.0,
                                    position_buffer=1.0)

    def test_idm_cap_default_and_bounds(self):
        rm, *_ = _build()
        assert rm.idm_cap == 2.5
        with pytest.raises(ValueError, match='idm_cap'):
            _build(idm_cap=0.5)
        rm_one, *_ = _build(idm_cap=1.0)
        assert rm_one.idm_cap == 1.0
        rm_none, *_ = _build(idm_cap=None)
        assert rm_none.idm_cap is None

    def test_annual_target_vol_range_per_mode(self):
        """Ports ``test_constructor_rejects_annual_target_vol_outside_open_unit_interval``:
        (0, 1) under 'percent_volatility', ``> 0`` under 'dollar_volatility'."""
        rm, pf, strat, dh, um, ve = _build()
        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError, match='annual_target_vol'):
                VolTargetingRiskManager(
                    pf, strat, ve, um, annual_target_vol=bad,
                    vol_target_mode='percent_volatility')
        for bad in (0.0, -250_000.0):
            with pytest.raises(ValueError, match='annual_target_vol'):
                VolTargetingRiskManager(
                    pf, strat, ve, um, annual_target_vol=bad,
                    vol_target_mode='dollar_volatility')

    def test_position_buffer_and_instrument_weight_mode_defaults(self):
        """Ports ``test_constructor_position_buffer_default_is_quarter`` /
        ``test_constructor_default_instrument_weight_mode_is_equal_weight``."""
        rm, *_ = _build()
        assert rm.position_buffer == 0.25
        assert rm.instrument_weight_mode == 'equal_weight'


class TestZeroWeight:
    """A present-but-zero instrument weight is a target of 0, not a skip:
    flat -> 'zero_weight' label; held -> flattens via the normal submit
    path. Ports ``test_skip_when_instrument_weight_is_zero`` /
    ``test_flat_position_zero_weight_still_labelled_zero_weight`` /
    ``test_held_position_zero_instrument_weight_is_flattened``."""

    def test_flat_symbol_at_zero_weight_labelled_zero_weight(self):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        strat.forecasts['A'] = 50.0
        ve.sigma['A'] = 10.0
        rm.instrument_weight = {'A': 0.0}
        rm.update_bar(_bar('A'))
        row = rm.get_records('A').iloc[-1]
        assert row['skip_reason'] == 'zero_weight'
        assert pf.submitted == []

    def test_held_position_at_zero_weight_flattens_via_submit_path(self):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        strat.forecasts['A'] = 50.0
        ve.sigma['A'] = 10.0
        pf.positions['A'] = 3.0
        rm.instrument_weight = {'A': 0.0}
        rm.update_bar(_bar('A'))
        row = rm.get_records('A').iloc[-1]
        assert bool(row['submitted']) is True
        assert row['skip_reason'] is None
        assert row['target_qty'] == 0.0
        assert pf.submitted[-1]['direction'] == Direction.SELL
        assert pf.submitted[-1]['quantity'] == pytest.approx(3.0)


class TestWholeLotRounding:
    """Ports ``test_non_fractional_rounds_target_to_whole_lot``: a
    ``fractional=False`` instrument's continuous target rounds to the
    nearest contract BEFORE the diff/dead-band (Carver Sec 10.7)."""

    def test_non_fractional_target_rounds_before_diff(self):
        strat = StubStrategy(['A'])
        dh = StubDataHandler(['A'])
        um = UniverseManager(strat, dh, min_history_bars=1)
        pf = StubPortfolio()
        ve = StubVolEstimator()
        instruments = uniform_registry(['A'], fractional=False)
        rm = VolTargetingRiskManager(pf, strat, ve, um, instruments=instruments,
                                     annual_target_vol=250_000.0)
        _make_live(um, strat, dh, 'A')
        strat.forecasts['A'] = 50.0
        ve.sigma['A'] = 8_000.0
        rm.instrument_weight = {'A': 1.0}
        # target = 1*1*250_000*1 / (1*8000) = 31.25 -> rounds to 31.
        rm.update_bar(_bar('A'))
        row = rm.get_records('A').iloc[-1]
        assert row['target_qty'] == pytest.approx(31.0)
        assert pf.submitted[-1]['quantity'] == pytest.approx(31.0)


class TestVolTargetModeFormula:
    """Ports ``test_percent_mode_target_scales_with_capital``: under
    ``vol_target_mode='percent_volatility'`` the cash target is
    ``capital * idm * iw * tau * (forecast/50)`` — it compounds with
    account equity (the contrast pin against 'dollar_volatility', which
    does not)."""

    def test_percent_mode_scales_with_capital(self):
        rm1, pf1, strat1, dh1, um1, ve1 = _build(
            vol_target_mode='percent_volatility', annual_target_vol=0.25)
        rm2, pf2, strat2, dh2, um2, ve2 = _build(
            vol_target_mode='percent_volatility', annual_target_vol=0.25)
        pf2.balance = 2.0 * pf1.balance
        for rm, pf, strat, dh, um, ve in (
            (rm1, pf1, strat1, dh1, um1, ve1),
            (rm2, pf2, strat2, dh2, um2, ve2),
        ):
            _make_live(um, strat, dh, 'A')
            strat.forecasts['A'] = 50.0
            ve.sigma['A'] = 10.0
            rm.instrument_weight = {'A': 1.0}
            rm.update_bar(_bar('A'))
        assert pf2.submitted[0]['quantity'] == pytest.approx(
            2.0 * pf1.submitted[0]['quantity'])


class TestStrandedPositionWarning:
    """Ports the ``test_held_position_no_target_skip_warns`` /
    ``test_flat_position_no_target_skip_does_not_warn`` pair: a
    target-derivation skip (sigma not ready) that strands a HELD position
    emits a WARNING naming it 'unmanaged'; the same skip while flat stays
    quiet."""

    def test_held_position_on_skip_warns(self, caplog):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        pf.positions['A'] = 4.0
        ve.sigma['A'] = None                      # warmup_volatility skip
        with caplog.at_level(logging.WARNING):
            rm.update_bar(_bar('A'))
        row = rm.get_records('A').iloc[-1]
        assert row['skip_reason'] == 'warmup_volatility'
        assert pf.submitted == []
        assert any('unmanaged' in r.message for r in caplog.records)

    def test_flat_position_on_skip_does_not_warn(self, caplog):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        ve.sigma['A'] = None                      # warmup_volatility skip
        with caplog.at_level(logging.WARNING):
            rm.update_bar(_bar('A'))
        assert pf.submitted == []
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestFormingBarGate:
    """Ports ``test_forming_bar_is_skipped``: ``update_bar`` returns
    before touching the vol estimator or the diagnostic buffer on a
    forming bar (one resize per COMPLETED bar)."""

    def test_forming_bar_returns_without_recording(self):
        rm, pf, strat, dh, um, ve = _build()
        _make_live(um, strat, dh, 'A')
        strat.forecasts['A'] = 50.0
        ve.sigma['A'] = 10.0
        rm.instrument_weight = {'A': 1.0}
        rm.update_bar(_bar('A', forming=True))
        assert rm.get_records('A').empty
        assert pf.submitted == []


class TestCorrModeRouting:
    """Ports the ``calculate_instrument_weight(mode='min_variance', ...)``
    tests: 'min_variance' dispatches through ``on_correlation_event``
    exactly like 'risk_parity' (``test_risk_parity_mode_uses_optimizer``);
    the old direct entry point is gone but the optimizer math survives."""

    def test_min_variance_mode_routed_through_correlation_event(self):
        rm, *_ = _build(instrument_weight_mode='min_variance')
        m = _identityish(['A', 'B'])
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'], m))
        assert rm.instrument_weight == pytest.approx({'A': 0.5, 'B': 0.5})

    def test_min_variance_downweights_correlated_pair(self):
        """A strongly-correlated pair is downweighted against an
        uncorrelated third asset (ports
        ``test_min_variance_downweights_correlated_pair_against_uncorrelated_solo``)."""
        rm, *_ = _build(symbols=('A', 'B', 'C'),
                        instrument_weight_mode='min_variance')
        labels = ['A', 'B', 'C']
        m = pd.DataFrame(
            [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=labels, columns=labels,
        )
        rm.on_correlation_event(_corr_event('ok', labels, m))
        assert rm.instrument_weight['C'] > rm.instrument_weight['A']
        assert rm.instrument_weight['A'] == pytest.approx(
            rm.instrument_weight['B'])


class TestCorrMatrixCacheUntouched:
    """Ports the implicit cache-coherence contract exercised throughout
    the pop/rescale tests: ``_corr_matrix_cache`` is set ONLY by an 'ok'
    event and is left alone by every other reason (degenerate fallbacks,
    'empty_universe', 'singleton')."""

    def test_cache_untouched_on_non_ok_events(self):
        rm, *_ = _build()
        m = _identityish(['A', 'B'])
        rm.on_correlation_event(_corr_event('ok', ['A', 'B'], m))
        assert rm._corr_matrix_cache is m
        for reason, live in (
            ('insufficient_observations', ['A', 'B']),
            ('too_few_symbols', ['A', 'B']),
            ('nan_fallback', ['A', 'B']),
            ('empty_universe', []),
            ('singleton', ['A']),
        ):
            rm.on_correlation_event(_corr_event(reason, live))
            assert rm._corr_matrix_cache is m


class TestBudgetGroupsExtra:
    """Old-suite ports for ``_grouped_weights`` / ``_budget_groups``
    (pre-split "Strategy-budgeted instrument weights" section) rewired
    through ``on_correlation_event`` per the task-12 brief. Reuses
    ``TestBudgetGroups.BudgetStrategy``."""

    BudgetStrategy = TestBudgetGroups.BudgetStrategy

    def _grouped_rm(self, symbols, groups, **rm_kwargs):
        """Build a portfolio/data-handler/universe-manager/RM quartet for
        a ``BudgetStrategy`` over ``symbols`` with the given
        ``{label: (budget, universe)}`` groups; ``min_history_bars=1`` so
        ``_make_live`` (or a direct correlation event) is enough to make
        any subset live."""
        strat = self.BudgetStrategy(symbols, groups)
        dh = StubDataHandler(strat.symbol_list)
        um = UniverseManager(strat, dh, min_history_bars=1)
        pf = StubPortfolio()
        ve = StubVolEstimator()
        rm = VolTargetingRiskManager(
            pf, strat, ve, um,
            annual_target_vol=rm_kwargs.pop('annual_target_vol', 250_000.0),
            **rm_kwargs,
        )
        return rm, pf, strat, dh, um, ve

    def test_bare_strategy_is_single_implicit_group(self):
        rm, *_ = _build(symbols=('A', 'B'))
        assert rm._budget_groups() == {'__all__': (1.0, ['A', 'B'])}

    def test_overlap_sums_books(self, caplog):
        """Overlap example: a(0.7)={X,Y}, b(0.3)={Y,Z} -> X=.35, Y=.35+.15
        (sum of both books) =.50, Z=.15; also carries the multi-group
        recalc INFO summary line (ports
        ``test_multi_group_recalc_logs_budget_shares``)."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Y', 'Z'], {'a': (0.7, ['X', 'Y']), 'b': (0.3, ['Y', 'Z'])})
        labels = ['X', 'Y', 'Z']
        with caplog.at_level(logging.INFO):
            rm.on_correlation_event(_corr_event('ok', labels,
                                                _identityish(labels)))
        assert rm.instrument_weight == pytest.approx(
            {'X': 0.35, 'Y': 0.50, 'Z': 0.15})
        assert any('budget groups (configured' in r.message
                  for r in caplog.records)

    def test_grouped_min_variance_uses_group_submatrix(self):
        """pair(0.5)={A,B} + solo(0.5)={C}: the pair's min-var solve runs
        on its own 2x2 submatrix (always 50/50 for two equal-vol assets),
        so A=B=0.25 and the solo group takes its full 0.5 budget."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['A', 'B', 'C'], {'pair': (0.5, ['A', 'B']), 'solo': (0.5, ['C'])},
            instrument_weight_mode='min_variance')
        labels = ['A', 'B', 'C']
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        assert rm.instrument_weight == pytest.approx(
            {'A': 0.25, 'B': 0.25, 'C': 0.5}, abs=1e-9)

    def test_grouped_risk_parity_book_sums_equal_budgets(self):
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['A', 'B', 'C', 'D'],
            {'g1': (0.7, ['A', 'B']), 'g2': (0.3, ['C', 'D'])},
            instrument_weight_mode='risk_parity')
        labels = ['A', 'B', 'C', 'D']
        rm.on_correlation_event(_corr_event(
            'ok', labels, _identityish(labels, off_diag=0.3)))
        book1 = rm.instrument_weight['A'] + rm.instrument_weight['B']
        book2 = rm.instrument_weight['C'] + rm.instrument_weight['D']
        assert book1 == pytest.approx(0.7, abs=1e-6)
        assert book2 == pytest.approx(0.3, abs=1e-6)
        assert all(w > 0 for w in rm.instrument_weight.values())

    def test_zero_budget_group_symbols_carry_zero_weight(self):
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Z'], {'a': (1.0, ['X']), 'b': (0.0, ['Z'])})
        labels = ['X', 'Z']
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        assert rm.instrument_weight == pytest.approx({'X': 1.0, 'Z': 0.0})

    def test_zero_budget_group_flattens_held_position(self):
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Z'], {'a': (1.0, ['X']), 'b': (0.0, ['Z'])})
        pf.positions['Z'] = 5.0
        _make_live(um, strat, dh, 'Z')
        strat.forecasts['Z'] = 50.0
        ve.sigma['Z'] = 8000.0
        labels = ['X', 'Z']
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        rm.update_bar(_bar('Z'))
        assert pf.submitted[-1]['symbol'] == 'Z'
        assert pf.submitted[-1]['direction'] == Direction.SELL
        assert pf.submitted[-1]['quantity'] == pytest.approx(5.0)
        row = rm.get_records('Z').iloc[-1]
        assert bool(row['submitted']) is True
        assert row['skip_reason'] is None

    def test_dead_group_redistributes_then_reenters_with_info_log(self, caplog):
        """Group b has no live members at the first refresh -> group a
        takes the full (INFO-logged) budget; once Z appears in a later
        refresh's live set, the configured 0.7/0.3 split is restored —
        the walk-forward self-heal (ports
        ``test_dead_group_budget_redistributes_then_reenters`` +
        ``test_dead_group_logs_info_and_redistributes``)."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Y', 'Z'], {'a': (0.7, ['X', 'Y']), 'b': (0.3, ['Z'])})
        with caplog.at_level(logging.INFO):
            rm.on_correlation_event(_corr_event('ok', ['X', 'Y'],
                                                _identityish(['X', 'Y'])))
        assert rm.instrument_weight == pytest.approx({'X': 0.5, 'Y': 0.5})
        assert any('budget group' in r.message and r.levelno == logging.INFO
                  for r in caplog.records)
        labels2 = ['X', 'Y', 'Z']
        rm.on_correlation_event(_corr_event('ok', labels2,
                                            _identityish(labels2)))
        assert rm.instrument_weight == pytest.approx(
            {'X': 0.35, 'Y': 0.35, 'Z': 0.3})

    def test_all_zero_live_budgets_fall_back_to_equal(self, caplog):
        """All non-zero budget sits on a dead group ('DEAD' never goes
        live) -> the renormalization denominator is 0 -> WARNING + equal
        budgets over the live groups."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['DEAD', 'X', 'Z'],
            {'a': (0.0, ['X']), 'b': (0.0, ['Z']), 'c': (1.0, ['DEAD'])})
        labels = ['X', 'Z']
        with caplog.at_level(logging.WARNING):
            rm.on_correlation_event(_corr_event('ok', labels,
                                                _identityish(labels)))
        assert rm.instrument_weight == pytest.approx({'X': 0.5, 'Z': 0.5})
        assert any('zero budget' in r.message for r in caplog.records)

    def test_bad_budget_falls_back_to_equal_with_warning(self, caplog):
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Z'], {'a': (-0.5, ['X']), 'b': (1.5, ['Z'])})
        with caplog.at_level(logging.WARNING):
            groups = rm._budget_groups()
        assert groups == {'a': (0.5, ['X']), 'b': (0.5, ['Z'])}
        assert any('non-finite/negative budget' in r.message
                  for r in caplog.records)

    def test_disjoint_idm_from_summed_weights(self):
        """Diverging universes are the regime where the summed weights
        (and hence wTrhow) actually move the IDM: identity rho over
        a=(0.7,{A,B}) / b=(0.3,{C,D}) with 2-asset min-var 50/50 within
        each book gives w=(.35,.35,.15,.15) and IDM=1/sqrt(sum(wi^2))
        = 1/sqrt(0.29), below the 2.5 cap."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['A', 'B', 'C', 'D'],
            {'a': (0.7, ['A', 'B']), 'b': (0.3, ['C', 'D'])},
            instrument_weight_mode='min_variance')
        labels = ['A', 'B', 'C', 'D']
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        assert rm.instrument_weight['A'] == pytest.approx(0.35, abs=1e-9)
        assert rm.instrument_weight['C'] == pytest.approx(0.15, abs=1e-9)
        assert rm.idm == pytest.approx(1.0 / math.sqrt(0.29), rel=1e-9)

    def test_uncovered_kept_symbol_warns_and_keeps_zero_weight(self, caplog):
        """A malformed custom source whose groups miss a live symbol: it
        keeps its seeded 0.0 weight and the gap is WARNING-logged instead
        of silent (the real Orchestrator's union universe cannot produce
        this)."""
        class _PartialSource(StubStrategy):
            def get_budget_groups(self):
                return {'a': (1.0, ['X'])}        # omits Y

        strat = _PartialSource(['X', 'Y'])
        dh = StubDataHandler(strat.symbol_list)
        um = UniverseManager(strat, dh, min_history_bars=1)
        rm = VolTargetingRiskManager(StubPortfolio(), strat, StubVolEstimator(),
                                     um, annual_target_vol=250_000.0)
        labels = ['X', 'Y']
        with caplog.at_level(logging.WARNING):
            rm.on_correlation_event(_corr_event('ok', labels,
                                                _identityish(labels)))
        assert rm.instrument_weight['X'] == pytest.approx(1.0)
        assert rm.instrument_weight['Y'] == 0.0
        assert any('covered by no budget group' in r.message
                  for r in caplog.records)

    def test_grouped_fallback_on_data_gap_honors_budgets_idm_untouched(self):
        """The data-gap equal-weight fallback (a degenerate CorrelationEvent
        reason) is budget-aware too: the 0.7/0.3 split survives even
        though no matrix was estimated, and idm is left as-is (ports
        ``test_grouped_fallback_honors_budgets_on_data_gap``)."""
        rm, pf, strat, dh, um, ve = self._grouped_rm(
            ['X', 'Z'], {'a': (0.7, ['X']), 'b': (0.3, ['Z'])})
        rm.idm = 1.7
        rm.on_correlation_event(_corr_event('insufficient_observations',
                                            ['X', 'Z']))
        assert rm.instrument_weight == pytest.approx({'X': 0.7, 'Z': 0.3})
        assert rm.idm == 1.7


class TestIdmCapExtra:
    """Ports the idm_cap behavioral tests that used the deleted explicit-
    corr_matrix hook, rewired through ``on_correlation_event``."""

    def test_idm_cap_none_disables_capping(self):
        labels = [f'S{i}' for i in range(7)]
        rm, *_ = _build(symbols=tuple(labels), idm_cap=None)
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        assert rm.idm == pytest.approx(math.sqrt(7.0), rel=1e-6)

    def test_idm_cap_applies_under_risk_parity_too(self):
        labels = [f'S{i}' for i in range(7)]
        rm, *_ = _build(symbols=tuple(labels),
                        instrument_weight_mode='risk_parity')
        rm.on_correlation_event(_corr_event('ok', labels, _identityish(labels)))
        assert rm.idm == pytest.approx(2.5)
