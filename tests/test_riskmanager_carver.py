"""
Unit tests for ``CarverVolTargetingRiskManager``.

Pin:
- Golden cash-vol formula:
    target_qty = (capital * idm * sw * iw * annualized_target_vol * (forecast/50)) / sigma
  where ``sigma`` is the annualized stdev of price changes ($-vol),
  ``sw`` = strategy_weight, ``iw`` = instrument_weight.
- Direction sign comes from ``trade_qty``.
- Skip-on-warmup: vol estimator returns None → no order.
- Skip-on-zero-vol.
- Forming bars are skipped.
- Idempotency: a stable forecast on consecutive bars submits at most one
  order (the second call sees the realized position at target).
- Position buffer (Carver §10.7) dead-band.
- Constructor validation (idm, annualized_target_vol, position_buffer ranges).
- Built-in instrument / strategy weight defaults + recalc.

Run from the repo root:  pytest tests/test_riskmanager_carver.py -v
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from event import BarEvent, OrderType, Direction
from riskmanager import CarverVolTargetingRiskManager


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakePortfolio:
    """Captures submit_order calls; configurable balance/positions.

    ``get_price`` is left on the surface for compatibility with other
    fixtures, but the cash-vol risk manager no longer calls it.
    """

    def __init__(self, balance: float = 1_000_000.0,
                 positions: Optional[Dict[str, float]] = None):
        self._balance = balance
        self.positions: Dict[str, float] = positions if positions is not None else {}
        self.submitted: List[Dict[str, Any]] = []

    def get_price(self, symbol: str) -> Optional[float]:
        return None

    def calculate_balance(self) -> float:
        return self._balance

    def submit_order(self, symbol, quantity, direction, timestamp,
                     order_type, price=None):
        self.submitted.append({
            'symbol': symbol, 'quantity': quantity, 'direction': direction,
            'timestamp': timestamp, 'order_type': order_type, 'price': price,
        })
        return None


class FakeStrategy:
    """Forecast oracle + symbol_list source for the risk manager.

    ``symbol_list`` is read at construction time by
    ``CarverVolTargetingRiskManager.calculate_instrument_weight`` to
    populate the equal-weight default.
    """

    def __init__(self, forecasts: Optional[Dict[str, float]] = None,
                 symbol_list: Optional[List[str]] = None):
        self._forecasts: Dict[str, float] = (
            dict(forecasts) if forecasts is not None else {}
        )
        self.symbol_list: List[str] = (
            list(symbol_list) if symbol_list is not None else ['BTC']
        )

    def get_forecast(self, symbol: str) -> float:
        return self._forecasts.get(symbol, 0.0)


class FakeVolEstimator:
    """Returns a configurable per-symbol annualized $-vol (or None for warmup)."""

    def __init__(self, vols: Optional[Dict[str, Optional[float]]] = None):
        self._vols: Dict[str, Optional[float]] = (
            dict(vols) if vols is not None else {}
        )
        self.update_calls: List[BarEvent] = []

    def update(self, event: BarEvent) -> None:
        self.update_calls.append(event)

    def get_annualized_vol(self, symbol: str) -> Optional[float]:
        return self._vols.get(symbol)


class FakeDataHandler:
    """Returns a configurable per-symbol close series via ``get_latest_bars``.

    ``closes`` maps symbol → pd.Series of Close prices (datetime-indexed
    is recommended for realism but not enforced; the risk manager only
    uses the values). When a symbol has no entry, returns an empty
    DataFrame (matching the cold-deque path). ``timeframes`` exposes the
    set of registered timeframes so the constructor validation can pass.
    """

    def __init__(
        self,
        closes: Optional[Dict[str, pd.Series]] = None,
        timeframes: Optional[Dict[str, int]] = None,
    ):
        self._closes: Dict[str, pd.Series] = (
            dict(closes) if closes is not None else {}
        )
        self.timeframes: Dict[str, int] = (
            dict(timeframes) if timeframes is not None else {'1d': 500}
        )
        self.calls: List[Dict[str, Any]] = []

    def get_latest_bars(self, symbol: str, n: int = 1,
                        timeframe: Optional[str] = None) -> pd.DataFrame:
        self.calls.append({'symbol': symbol, 'n': n, 'timeframe': timeframe})
        s = self._closes.get(symbol)
        if s is None or len(s) == 0:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        tail = s.iloc[-n:]
        return pd.DataFrame({
            'Open': tail.values, 'High': tail.values, 'Low': tail.values,
            'Close': tail.values, 'Volume': 1.0,
        }, index=tail.index)


def _bar(symbol: str = 'BTC', is_forming: bool = False,
         ts: Optional[datetime] = None) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else datetime(2026, 1, 1, 12, 0, 0),
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        period='1d', is_forming=is_forming,
    )


def _make(
    *,
    forecast: float = 50.0,
    # Default sigma is $-vol = old %-vol (0.40) × reference price (20_000) = 8000.
    # This keeps the golden target_qty (1.5625) numerically unchanged across
    # the convention switch (it's the algebraic equivalent of the old formula
    # for positive-price single instruments).
    sigma: Optional[float] = 8000.0,
    weight: float = 0.5,
    balance: float = 100_000.0,
    positions: Optional[Dict[str, float]] = None,
    idm: float = 1.0,
    annualized_target_vol: float = 0.25,
    position_buffer: float = 0.0,
):
    pf = FakePortfolio(balance=balance, positions=positions or {'BTC': 0.0})
    strat = FakeStrategy({'BTC': forecast}, symbol_list=['BTC'])
    vol = FakeVolEstimator({'BTC': sigma})
    dh = FakeDataHandler()
    rm = CarverVolTargetingRiskManager(
        pf, strat, vol,
        data_handler=dh,
        idm=idm,
        annualized_target_vol=annualized_target_vol,
        position_buffer=position_buffer,
    )
    # Override the equal-weight default (1.0 for the single 'BTC' symbol)
    # so tests can pin a specific instrument weight independent of N.
    rm.instrument_weight = {'BTC': weight}
    return pf, strat, vol, rm


# ──────────────────────────────────────────────
# Construction validation
# ──────────────────────────────────────────────

def test_constructor_rejects_non_positive_idm():
    with pytest.raises(ValueError, match="idm"):
        _make(idm=0)


def test_constructor_rejects_annualized_target_vol_outside_open_unit_interval():
    with pytest.raises(ValueError, match="annualized_target_vol"):
        _make(annualized_target_vol=0.0)
    with pytest.raises(ValueError, match="annualized_target_vol"):
        _make(annualized_target_vol=1.0)
    with pytest.raises(ValueError, match="annualized_target_vol"):
        _make(annualized_target_vol=-0.1)


def test_constructor_rejects_position_buffer_outside_unit_interval():
    with pytest.raises(ValueError, match="position_buffer"):
        _make(position_buffer=-0.1)
    with pytest.raises(ValueError, match="position_buffer"):
        _make(position_buffer=1.0)


def test_constructor_position_buffer_default_is_quarter():
    """Production default is 0.25 (Carver §10.7 dead-band, cost-aware sizing).

    Constructed directly (no _make override) so this test pins the public
    default on the class itself, not the test-helper override.
    """
    pf = FakePortfolio(balance=100_000.0, positions={'BTC': 0.0})
    rm = CarverVolTargetingRiskManager(
        pf, FakeStrategy({'BTC': 0.0}, symbol_list=['BTC']),
        FakeVolEstimator({'BTC': 8000.0}),
        data_handler=FakeDataHandler(),
    )
    assert rm.position_buffer == 0.25


def test_constructor_default_instrument_weight_mode_is_equal_weight():
    """Field default = 'equal_weight'; constructor stores the kwarg verbatim."""
    pf = FakePortfolio(balance=100_000.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 0.0}, symbol_list=['BTC'])
    vol = FakeVolEstimator({'BTC': 8000.0})
    dh = FakeDataHandler()
    rm_default = CarverVolTargetingRiskManager(pf, strat, vol, data_handler=dh)
    assert rm_default.instrument_weight_mode == 'equal_weight'
    rm_explicit = CarverVolTargetingRiskManager(
        pf, strat, vol, data_handler=dh, instrument_weight_mode='equal_weight',
    )
    assert rm_explicit.instrument_weight_mode == 'equal_weight'


def test_constructor_with_min_variance_mode_and_empty_deques_falls_back(caplog):
    """min_variance + empty data handler → WARNING + equal-weight fallback,
    no exception. ``self.idm`` is left at its constructor default."""
    pf = FakePortfolio(balance=100_000.0, positions={'BTC': 0.0, 'ETH': 0.0})
    strat = FakeStrategy({'BTC': 0.0, 'ETH': 0.0}, symbol_list=['BTC', 'ETH'])
    vol = FakeVolEstimator({'BTC': 8000.0, 'ETH': 8000.0})
    dh = FakeDataHandler()                                      # empty closes
    with caplog.at_level('WARNING'):
        rm = CarverVolTargetingRiskManager(
            pf, strat, vol, data_handler=dh,
            instrument_weight_mode='min_variance',
        )
    assert math.isclose(rm.instrument_weight['BTC'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5, rel_tol=1e-12)
    assert rm.idm == 1.0                                        # constructor default
    assert any('min_variance' in r.message and 'falling back' in r.message
               for r in caplog.records)


def test_constructor_with_unknown_instrument_weight_mode_raises():
    """A bogus mode passed to the constructor surfaces immediately via __init__."""
    pf = FakePortfolio(balance=100_000.0, positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 0.0}, symbol_list=['BTC'])
    vol = FakeVolEstimator({'BTC': 8000.0})
    with pytest.raises(ValueError, match="mode"):
        CarverVolTargetingRiskManager(
            pf, strat, vol, data_handler=FakeDataHandler(),
            instrument_weight_mode='bogus',
        )


def test_calculate_instrument_weight_with_no_mode_arg_reads_stored_field():
    """When ``mode`` is omitted, the method reads ``self.instrument_weight_mode``.

    Overwrite the field post-construction and call with no args to confirm
    the fallback (rather than the historical 'equal_weight' literal default).
    """
    pf = FakePortfolio(balance=100_000.0, positions={'BTC': 0.0, 'ETH': 0.0})
    strat = FakeStrategy({'BTC': 0.0, 'ETH': 0.0}, symbol_list=['BTC', 'ETH'])
    vol = FakeVolEstimator({'BTC': 8000.0, 'ETH': 8000.0})
    rm = CarverVolTargetingRiskManager(
        pf, strat, vol, data_handler=FakeDataHandler(),
    )                                                           # default mode
    # Flip the stored mode + provide a corr matrix; the next no-arg call
    # must dispatch on the stored field, not the old literal default.
    rm.instrument_weight_mode = 'min_variance'
    corr = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=['BTC', 'ETH'], columns=['BTC', 'ETH'],
    )
    rm.calculate_instrument_weight(corr_matrix=corr)
    # min_variance on uncorrelated equal-vol pair → equal weights
    assert math.isclose(rm.instrument_weight['BTC'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5, rel_tol=1e-12)


def test_constructor_rejects_corr_lookback_below_two():
    with pytest.raises(ValueError, match="corr_lookback"):
        CarverVolTargetingRiskManager(
            FakePortfolio(), FakeStrategy(symbol_list=['BTC']),
            FakeVolEstimator(), data_handler=FakeDataHandler(),
            corr_lookback=1,
        )


def test_constructor_rejects_negative_corr_step_size():
    with pytest.raises(ValueError, match="corr_step_size"):
        CarverVolTargetingRiskManager(
            FakePortfolio(), FakeStrategy(symbol_list=['BTC']),
            FakeVolEstimator(), data_handler=FakeDataHandler(),
            corr_step_size=-1,
        )


def test_constructor_rejects_unregistered_corr_timeframe():
    """``corr_timeframe`` must appear in ``data_handler.timeframes``."""
    dh = FakeDataHandler(timeframes={'1d': 500})                # no '4h'
    with pytest.raises(ValueError, match="corr_timeframe"):
        CarverVolTargetingRiskManager(
            FakePortfolio(), FakeStrategy(symbol_list=['BTC']),
            FakeVolEstimator(), data_handler=dh, corr_timeframe='4h',
        )


# ──────────────────────────────────────────────
# Built-in weight defaults + recalc
# ──────────────────────────────────────────────

def test_instrument_weight_default_is_equal_weight_two_symbols():
    """``calculate_instrument_weight`` runs at __init__ and produces 1/N."""
    pf = FakePortfolio()
    strat = FakeStrategy({'BTC': 0.0, 'ETH': 0.0}, symbol_list=['BTC', 'ETH'])
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=FakeDataHandler(),
    )
    assert math.isclose(rm.instrument_weight['BTC'], 0.5)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5)
    assert math.isclose(sum(rm.instrument_weight.values()), 1.0, abs_tol=1e-9)


def test_instrument_weight_default_is_equal_weight_three_symbols():
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=['BTC', 'ETH', 'SOL'])
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=FakeDataHandler(),
    )
    for s in ['BTC', 'ETH', 'SOL']:
        assert math.isclose(rm.instrument_weight[s], 1.0 / 3.0)
    assert math.isclose(sum(rm.instrument_weight.values()), 1.0, abs_tol=1e-9)


def test_strategy_weight_default_is_one_for_single_strategy():
    """Placeholder: ``{strategy_class_name: 1.0}`` for the one bound strategy."""
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=['BTC'])
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=FakeDataHandler(),
    )
    assert rm.strategy_weight == {'FakeStrategy': 1.0}


def test_calculate_instrument_weight_is_recallable_after_symbol_list_change():
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=['BTC'])
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=FakeDataHandler(),
    )
    assert rm.instrument_weight == {'BTC': 1.0}
    strat.symbol_list = ['BTC', 'ETH']
    rm.calculate_instrument_weight()
    assert math.isclose(rm.instrument_weight['BTC'], 0.5)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5)


def test_calculate_strategy_weight_is_recallable():
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=['BTC'])
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=FakeDataHandler(),
    )
    rm.strategy_weight = {'OverriddenName': 0.7}
    rm.calculate_strategy_weight()
    assert rm.strategy_weight == {'FakeStrategy': 1.0}


# ──────────────────────────────────────────────
# Golden Carver cash-vol formula
# ──────────────────────────────────────────────

def test_golden_carver_formula():
    """Capital=100k, idm=1, sw=1.0, iw=0.5, annualized_target_vol=0.25,
    sigma=$8000 ($-vol), forecast=+50 →
       annual_cash_target = 100k * 1 * 1 * 0.5 * 0.25 * (50/50) = 12_500
       target_qty         = 12_500 / 8_000 = 1.5625.
    From 0 → BUY 1.5625."""
    pf, _, _, rm = _make()
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['symbol'] == 'BTC'
    assert call['direction'] == Direction.BUY
    assert math.isclose(call['quantity'], 1.5625, rel_tol=1e-12)
    assert call['order_type'] == OrderType.MKT


def test_negative_forecast_submits_sell_with_correct_magnitude():
    """forecast=-50 inverts the sign of annual_cash_target → target_qty = -1.5625
    → SELL 1.5625 from a flat position."""
    pf, _, _, rm = _make(forecast=-50.0)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    call = pf.submitted[0]
    assert call['direction'] == Direction.SELL
    assert math.isclose(call['quantity'], 1.5625, rel_tol=1e-12)


def test_forecast_one_hundred_doubles_target():
    """forecast=+100 → factor = 100/50 = 2 → annual_cash_target = 25_000
    → target_qty = 3.125."""
    pf, _, _, rm = _make(forecast=100.0)
    rm.update_bar(_bar())
    assert math.isclose(pf.submitted[0]['quantity'], 3.125, rel_tol=1e-12)


def test_forecast_zero_with_long_position_flattens():
    """annual_cash_target = 0 → target_qty = 0. From positions = +1.0, trade =
    -1.0 → SELL 1.0."""
    pf, _, _, rm = _make(forecast=0.0, positions={'BTC': 1.0})
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    assert pf.submitted[0]['direction'] == Direction.SELL
    assert math.isclose(pf.submitted[0]['quantity'], 1.0)


def test_idm_scales_target_linearly():
    """Doubling idm doubles the target quantity."""
    pf1, _, _, rm1 = _make(idm=1.0)
    rm1.update_bar(_bar())
    pf2, _, _, rm2 = _make(idm=2.0)
    rm2.update_bar(_bar())
    assert math.isclose(pf2.submitted[0]['quantity'],
                        2.0 * pf1.submitted[0]['quantity'], rel_tol=1e-12)


def test_instrument_weight_scales_target_linearly():
    pf1, _, _, rm1 = _make(weight=0.25)
    rm1.update_bar(_bar())
    pf2, _, _, rm2 = _make(weight=0.50)
    rm2.update_bar(_bar())
    assert math.isclose(pf2.submitted[0]['quantity'],
                        2.0 * pf1.submitted[0]['quantity'], rel_tol=1e-12)


def test_strategy_weight_scales_target_linearly():
    """Halving strategy_weight halves the target quantity."""
    pf1, _, _, rm1 = _make()
    rm1.update_bar(_bar())
    pf2, _, _, rm2 = _make()
    rm2.strategy_weight = {'FakeStrategy': 0.5}
    rm2.update_bar(_bar())
    assert math.isclose(pf2.submitted[0]['quantity'],
                        0.5 * pf1.submitted[0]['quantity'], rel_tol=1e-12)


# ──────────────────────────────────────────────
# Skip conditions
# ──────────────────────────────────────────────

def test_forming_bar_is_skipped():
    pf, _, vol, rm = _make()
    rm.update_bar(_bar(is_forming=True))
    assert pf.submitted == []
    # Vol estimator should also not be updated.
    assert vol.update_calls == []


def test_warmup_skip_when_vol_is_none():
    pf, _, _, rm = _make(sigma=None)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_skip_when_vol_is_zero():
    pf, _, _, rm = _make(sigma=0.0)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_skip_when_instrument_weight_is_zero():
    pf, _, _, rm = _make(weight=0.0)
    rm.update_bar(_bar())
    assert pf.submitted == []
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'zero_weight'


def test_skip_when_strategy_weight_is_zero():
    pf, _, _, rm = _make()
    rm.strategy_weight = {'FakeStrategy': 0.0}
    rm.update_bar(_bar())
    assert pf.submitted == []
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'zero_weight'


# ──────────────────────────────────────────────
# Idempotency on stable forecast
# ──────────────────────────────────────────────

def test_second_call_with_position_at_target_submits_no_order():
    """Realized position already matches target → trade_qty ≈ 0 → no order."""
    # target_qty = 1.5625 (golden case). Set positions to that exact value.
    pf, _, _, rm = _make(positions={'BTC': 1.5625})
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_second_call_after_first_resize_with_no_state_change_submits_no_order():
    """Simulating the engine: first bar submits the open order; before the
    second bar runs, the portfolio's realized position is updated to match
    the target. The second call should add NO new order."""
    pf, _, _, rm = _make()
    rm.update_bar(_bar(ts=datetime(2026, 1, 1)))
    assert len(pf.submitted) == 1
    # Simulate the fill: realized position now matches the target.
    pf.positions['BTC'] = 1.5625
    pf.submitted.clear()
    rm.update_bar(_bar(ts=datetime(2026, 1, 2)))
    assert pf.submitted == []


# ──────────────────────────────────────────────
# Position buffer (Carver §10.7)
# ──────────────────────────────────────────────

def test_position_buffer_blocks_small_diff():
    """target = 1.5625, current = 1.5 → trade = 0.0625, |trade|/|target| ≈ 0.04.
    With position_buffer=0.10, the trade is below the dead-band → no order."""
    pf, _, _, rm = _make(positions={'BTC': 1.5}, position_buffer=0.10)
    rm.update_bar(_bar())
    assert pf.submitted == []


def test_position_buffer_allows_large_diff():
    """Same target but current = 1.0 → trade = 0.5625, |trade|/|target| ≈ 0.36.
    Above 0.10 buffer → order fires."""
    pf, _, _, rm = _make(positions={'BTC': 1.0}, position_buffer=0.10)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    assert math.isclose(pf.submitted[0]['quantity'], 0.5625, rel_tol=1e-12)


def test_position_buffer_does_not_block_flatten():
    """When forecast=0 the target is 0; the dead-band collapses to zero
    so any open position is flattened regardless of position_buffer."""
    pf, _, _, rm = _make(forecast=0.0, positions={'BTC': 1.5},
                         position_buffer=0.50)
    rm.update_bar(_bar())
    assert len(pf.submitted) == 1
    assert pf.submitted[0]['direction'] == Direction.SELL
    assert math.isclose(pf.submitted[0]['quantity'], 1.5)


# ──────────────────────────────────────────────
# Vol estimator update cadence
# ──────────────────────────────────────────────

def test_vol_estimator_updated_on_completed_bar():
    pf, _, vol, rm = _make()
    bar = _bar()
    rm.update_bar(bar)
    assert vol.update_calls == [bar]


def test_vol_estimator_not_updated_on_forming_bar():
    pf, _, vol, rm = _make()
    rm.update_bar(_bar(is_forming=True))
    assert vol.update_calls == []


# ──────────────────────────────────────────────
# Per-bar diagnostics (get_records / skip_reason)
# ──────────────────────────────────────────────

def test_get_records_empty_for_unknown_symbol():
    _, _, _, rm = _make()
    df = rm.get_records('UNKNOWN')
    assert df.empty


def test_one_row_per_completed_bar_none_for_forming():
    """Forming bars must not produce a record; each completed bar adds one."""
    _, _, _, rm = _make()
    rm.update_bar(_bar(ts=datetime(2026, 1, 1), is_forming=True))
    rm.update_bar(_bar(ts=datetime(2026, 1, 2)))
    rm.update_bar(_bar(ts=datetime(2026, 1, 3)))
    df = rm.get_records('BTC')
    assert len(df) == 2
    assert list(df.index) == [datetime(2026, 1, 2), datetime(2026, 1, 3)]


def test_record_skip_reason_warmup():
    pf, _, _, rm = _make(sigma=None)
    rm.update_bar(_bar())
    df = rm.get_records('BTC')
    assert len(df) == 1
    row = df.iloc[0]
    assert row['skip_reason'] == 'warmup'
    assert not row['submitted']
    assert row['sigma'] is None
    assert row['instrument_weight'] is None       # not read on warmup
    assert row['strategy_weight'] is None         # not read on warmup
    assert row['target_qty'] is None
    assert row['trade_qty'] is None
    assert pf.submitted == []


def test_record_skip_reason_zero_vol():
    pf, _, _, rm = _make(sigma=0.0)
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'zero_vol'
    assert not row['submitted']
    assert row['sigma'] == 0.0
    assert row['instrument_weight'] is None       # not read after zero-vol skip
    assert row['strategy_weight'] is None
    assert row['target_qty'] is None
    assert pf.submitted == []


def test_record_skip_reason_zero_weight_instrument():
    pf, _, _, rm = _make(weight=0.0)
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'zero_weight'
    assert not row['submitted']
    assert row['instrument_weight'] == 0.0
    assert math.isclose(row['strategy_weight'], 1.0)
    assert row['sigma'] == 8000.0                 # was read before the skip
    assert row['target_qty'] is None
    assert pf.submitted == []


def test_record_skip_reason_zero_weight_strategy():
    pf, _, _, rm = _make()
    rm.strategy_weight = {'FakeStrategy': 0.0}
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'zero_weight'
    assert not row['submitted']
    assert row['strategy_weight'] == 0.0
    assert math.isclose(row['instrument_weight'], 0.5)
    assert pf.submitted == []


def test_record_skip_reason_dead_band():
    """target = 1.5625, current = 1.5, |trade|/|target| ≈ 0.04 < 0.10 buffer."""
    pf, _, _, rm = _make(positions={'BTC': 1.5}, position_buffer=0.10)
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'dead_band'
    assert not row['submitted']
    assert math.isclose(row['target_qty'], 1.5625, rel_tol=1e-12)
    assert math.isclose(row['trade_qty'], 0.0625, rel_tol=1e-12)
    assert math.isclose(row['buffer_threshold'], 0.15625, rel_tol=1e-12)
    assert pf.submitted == []


def test_record_skip_reason_at_target():
    """Realized position equals target → trade_qty < 1e-12 → 'at_target'."""
    pf, _, _, rm = _make(positions={'BTC': 1.5625})
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['skip_reason'] == 'at_target'
    assert not row['submitted']
    assert math.isclose(row['target_qty'], 1.5625, rel_tol=1e-12)
    assert abs(row['trade_qty']) < 1e-12
    assert pf.submitted == []


def test_record_submitted_row_matches_golden_formula():
    """Happy path: every numeric field matches the analytical formula."""
    pf, _, _, rm = _make(
        balance=1_000_000.0, idm=1.5, annualized_target_vol=0.2, weight=0.5,
        forecast=50.0, sigma=10_000.0,
    )
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    # 1_000_000 * 1.5 * 1.0 * 0.5 * 0.2 * (50/50) = 150_000
    # target_qty = 150_000 / 10_000 = 15.0
    assert row['submitted']
    assert row['skip_reason'] is None
    assert row['forecast'] == 50.0
    assert row['sigma'] == 10_000.0
    assert math.isclose(row['instrument_weight'], 0.5)
    assert math.isclose(row['strategy_weight'], 1.0)
    assert row['capital'] == 1_000_000.0
    assert row['idm'] == 1.5
    assert row['annualized_target_vol'] == 0.2
    assert math.isclose(row['annual_cash_target'], 150_000.0, rel_tol=1e-12)
    assert math.isclose(row['target_qty'], 15.0, rel_tol=1e-12)
    assert row['current_qty'] == 0.0
    assert math.isclose(row['trade_qty'], 15.0, rel_tol=1e-12)
    assert row['buffer_threshold'] == 0.0          # position_buffer default 0.0 in _make
    assert len(pf.submitted) == 1
    assert math.isclose(pf.submitted[0]['quantity'], 15.0, rel_tol=1e-12)


def test_record_skip_reason_none_when_order_submitted():
    """Default _make (sigma=8000, capital=100k) submits the golden order."""
    _, _, _, rm = _make()
    rm.update_bar(_bar())
    row = rm.get_records('BTC').iloc[0]
    assert row['submitted']
    assert row['skip_reason'] is None


# ──────────────────────────────────────────────
# calculate_instrument_weight — mode='min_variance'
# ──────────────────────────────────────────────

def _build_rm(symbols, *, data_handler: Optional[FakeDataHandler] = None,
              **kwargs):
    """Construct a risk manager with the default equal-weight init for
    ``symbols``. Used as the starting point for min_variance recalc tests.

    ``data_handler`` defaults to an empty ``FakeDataHandler`` (no closes
    registered) for callers that exercise paths where the data-handler
    surface is unused (caller supplies an explicit ``corr_matrix``).
    """
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=list(symbols))
    dh = data_handler if data_handler is not None else FakeDataHandler()
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=dh, **kwargs,
    )
    return rm


def _corr_df(labels, off_diag):
    """N×N correlation matrix with a uniform off-diagonal value."""
    n = len(labels)
    m = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


def test_equal_weight_remains_default_mode():
    """Calling ``calculate_instrument_weight()`` with no args keeps the
    historical equal-weight behavior (regression guard)."""
    rm = _build_rm(['BTC', 'ETH'])
    rm.calculate_instrument_weight()
    assert math.isclose(rm.instrument_weight['BTC'], 0.5)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5)


def test_equal_weight_silently_ignores_corr_matrix():
    """Passing a corr_matrix to mode='equal_weight' must not error and must
    not change the equal-weight result."""
    rm = _build_rm(['BTC', 'ETH'])
    corr = _corr_df(['BTC', 'ETH'], off_diag=0.9)
    rm.calculate_instrument_weight(mode='equal_weight', corr_matrix=corr)
    assert math.isclose(rm.instrument_weight['BTC'], 0.5)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5)


def test_min_variance_two_uncorrelated_assets_yields_equal_weights():
    """With ρ=0 and equal vols, min-var weights are 1/N (closed form)."""
    rm = _build_rm(['BTC', 'ETH'])
    corr = _corr_df(['BTC', 'ETH'], off_diag=0.0)
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    assert math.isclose(rm.instrument_weight['BTC'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5, rel_tol=1e-12)
    assert math.isclose(sum(rm.instrument_weight.values()), 1.0, abs_tol=1e-12)


def test_min_variance_downweights_correlated_pair_against_uncorrelated_solo():
    """ρ = [[1, 0.7, 0],
            [0.7, 1, 0],
            [0, 0, 1]]
    →   the uncorrelated asset gets a larger weight than each of the
    correlated pair, since the correlated pair carries redundant risk."""
    rm = _build_rm(['A', 'B', 'C'])
    rho = pd.DataFrame(
        [[1.0, 0.7, 0.0],
         [0.7, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=rho)
    w_a = rm.instrument_weight['A']
    w_b = rm.instrument_weight['B']
    w_c = rm.instrument_weight['C']
    assert math.isclose(w_a, w_b, rel_tol=1e-12)            # symmetry
    assert w_c > w_a                                        # uncorrelated dominates
    assert math.isclose(w_a + w_b + w_c, 1.0, abs_tol=1e-12)


def test_min_variance_clips_negative_weight_and_renormalizes():
    """Construct a corr matrix where the raw inverse-corr formula produces a
    negative weight for one symbol: A is highly correlated with both B and
    C, while B and C are only mildly correlated with each other. The
    risk-manager must clip A's weight to 0 and renormalize the survivors
    to sum to 1."""
    rm = _build_rm(['A', 'B', 'C'])
    rho = pd.DataFrame(
        [[1.0, 0.7, 0.7],
         [0.7, 1.0, 0.3],
         [0.7, 0.3, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=rho)
    assert rm.instrument_weight['A'] == 0.0
    assert math.isclose(rm.instrument_weight['B'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['C'], 0.5, rel_tol=1e-12)
    assert math.isclose(sum(rm.instrument_weight.values()), 1.0, abs_tol=1e-12)


def test_min_variance_accepts_corr_matrix_with_scrambled_symbol_order():
    """The corr matrix's index drives ordering, so its row order can differ
    from ``strategy.symbol_list`` order. The dict result must still be
    keyed by the correct labels."""
    rm = _build_rm(['BTC', 'ETH'])
    corr = _corr_df(['ETH', 'BTC'], off_diag=0.0)            # scrambled order
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    assert math.isclose(rm.instrument_weight['BTC'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5, rel_tol=1e-12)


# ──────────────────────────────────────────────
# calculate_instrument_weight — validation
# ──────────────────────────────────────────────

def test_rejects_unknown_mode():
    rm = _build_rm(['BTC', 'ETH'])
    with pytest.raises(ValueError, match="mode"):
        rm.calculate_instrument_weight(mode='bogus')


def test_min_variance_without_corr_matrix_and_empty_deques_falls_back(caplog):
    """No corr_matrix + empty data handler → WARNING + equal-weight, no raise."""
    rm = _build_rm(['BTC', 'ETH'])
    with caplog.at_level('WARNING'):
        rm.calculate_instrument_weight(mode='min_variance')
    assert math.isclose(rm.instrument_weight['BTC'], 0.5, rel_tol=1e-12)
    assert math.isclose(rm.instrument_weight['ETH'], 0.5, rel_tol=1e-12)
    assert any('min_variance' in r.message and 'falling back' in r.message
               for r in caplog.records)


def test_min_variance_with_label_mismatch_raises():
    """corr_matrix index must equal symbol_list as a set."""
    rm = _build_rm(['BTC', 'ETH'])
    bad = _corr_df(['BTC', 'SOL'], off_diag=0.0)
    with pytest.raises(ValueError, match="symbol_list"):
        rm.calculate_instrument_weight(mode='min_variance', corr_matrix=bad)


def test_min_variance_with_asymmetric_index_columns_raises():
    """corr_matrix.index must equal corr_matrix.columns."""
    rm = _build_rm(['BTC', 'ETH'])
    bad = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=['BTC', 'ETH'], columns=['ETH', 'BTC'],
    )
    with pytest.raises(ValueError, match="index"):
        rm.calculate_instrument_weight(mode='min_variance', corr_matrix=bad)


# ──────────────────────────────────────────────
# Inline corr derivation from data handler
# ──────────────────────────────────────────────

def _price_series(n: int, *, seed: int = 0, drift: float = 0.0) -> pd.Series:
    """Synthetic log-normal price series of length ``n`` with a small drift."""
    rng = np.random.default_rng(seed=seed)
    returns = rng.normal(loc=drift, scale=0.02, size=n)
    return pd.Series(
        100.0 * np.exp(np.cumsum(returns)),
        index=pd.date_range('2024-01-01', periods=n, freq='D'),
    )


def test_min_variance_derives_corr_from_filled_deques_and_auto_updates_idm():
    """With ``corr_lookback + 5`` bars per symbol pre-loaded, the derived
    matrix matches ``prices.pct_change().corr()`` from the SAME trailing
    window the risk manager pulled, weights are finite and sum to 1,
    and ``self.idm`` is auto-updated from the same matrix via
    ``analytics.diversification_multiplier``."""
    from analytics import diversification_multiplier
    symbols = ['BTC', 'ETH']
    lookback = 100
    closes = {s: _price_series(lookback + 5, seed=i) for i, s in enumerate(symbols)}
    dh = FakeDataHandler(closes=closes)
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=symbols)
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=dh,
        instrument_weight_mode='min_variance', corr_lookback=lookback,
    )
    # Weights sum to 1, non-negative.
    assert math.isclose(sum(rm.instrument_weight.values()), 1.0, abs_tol=1e-9)
    assert all(w >= 0 for w in rm.instrument_weight.values())
    # Diagnostics: the IDM matches what you'd get by computing the
    # corr matrix externally from the SAME ``corr_lookback`` trailing
    # window (the risk manager calls get_latest_bars(s, lookback, ...)).
    trailing = pd.DataFrame({s: closes[s].iloc[-lookback:] for s in symbols})
    expected_corr = trailing.pct_change(fill_method=None).dropna().corr()
    expected_idm = diversification_multiplier(rm.instrument_weight, expected_corr)
    assert math.isclose(rm.idm, expected_idm, rel_tol=1e-9)


def test_min_variance_passed_corr_matrix_does_not_query_data_handler():
    """When the caller supplies ``corr_matrix`` explicitly, the inline
    derivation branch is skipped — the data handler is never queried."""
    rm = _build_rm(['BTC', 'ETH'])
    dh: FakeDataHandler = rm.data_handler                       # type: ignore[assignment]
    dh.calls.clear()                                            # discard the ctor-time fallback call (which had no closes)
    corr = _corr_df(['BTC', 'ETH'], off_diag=0.0)
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    assert dh.calls == []                                       # no data-handler reads


def test_min_variance_passed_corr_matrix_also_updates_idm():
    """IDM auto-update fires regardless of corr-matrix source."""
    from analytics import diversification_multiplier
    rm = _build_rm(['BTC', 'ETH'])
    corr = _corr_df(['BTC', 'ETH'], off_diag=0.0)
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    expected = diversification_multiplier(rm.instrument_weight, corr)
    assert math.isclose(rm.idm, expected, rel_tol=1e-12)


def test_min_variance_fallback_leaves_idm_untouched():
    """Equal-weight-fallback path must not touch ``self.idm``."""
    rm = _build_rm(['BTC', 'ETH'])                              # empty data handler
    rm.idm = 1.7                                                # sentinel
    rm.calculate_instrument_weight(mode='min_variance')         # falls back
    assert rm.idm == 1.7


def test_returns_used_are_simple_pct_change():
    """The inline derivation uses ``pct_change()`` on Close prices.

    Build two series whose log-return correlation differs measurably from
    their level correlation; the derived corr must match the
    ``pct_change`` value, not the level correlation.
    """
    symbols = ['A', 'B']
    closes = {s: _price_series(100, seed=i) for i, s in enumerate(symbols)}
    dh = FakeDataHandler(closes=closes)
    pf = FakePortfolio()
    strat = FakeStrategy(symbol_list=symbols)
    # Use equal-weight construction (no auto-recalc fires from __init__),
    # then capture the derived weights via the explicit min-variance call.
    rm = CarverVolTargetingRiskManager(
        pf, strat, FakeVolEstimator(), data_handler=dh,
        corr_lookback=100,
    )
    rm.calculate_instrument_weight(mode='min_variance')

    # The min-variance solution for a 2-asset matrix with the data-handler
    # corr should be reproducible end-to-end from the pct_change call.
    expected_corr = pd.DataFrame(closes).pct_change(fill_method=None).dropna().corr()
    rho = expected_corr.loc['A', 'B']
    # Closed form for the 2-asset min-variance under equal-vol: still 1/N
    # regardless of ρ, so we use that as a sanity check on the solver…
    assert math.isclose(rm.instrument_weight['A'], 0.5, rel_tol=1e-9)
    assert math.isclose(rm.instrument_weight['B'], 0.5, rel_tol=1e-9)
    # …and we verify that the matrix the solver consumed matches the
    # pct_change matrix by replaying the IDM equation:
    from analytics import diversification_multiplier
    expected_idm = diversification_multiplier(rm.instrument_weight, expected_corr)
    assert math.isclose(rm.idm, expected_idm, rel_tol=1e-9)
    # Sanity: ρ alone determines IDM for 2-asset equal-weight: 1/sqrt((1+ρ)/2).
    assert math.isclose(
        expected_idm, 1.0 / math.sqrt((1.0 + rho) / 2.0), rel_tol=1e-9,
    )


# ──────────────────────────────────────────────
# Auto-recalc cadence in update_bar
# ──────────────────────────────────────────────

def _make_min_variance_rm_with_closes(symbols, n_bars, step_size, *, lookback=100):
    """Build a min-variance risk manager with pre-loaded closes for ``symbols``.

    Returns ``(rm, dh, vol, pf)`` — the data handler is non-empty so the
    initial __init__ recompute succeeds (not a fallback).
    """
    closes = {s: _price_series(n_bars, seed=i) for i, s in enumerate(symbols)}
    dh = FakeDataHandler(closes=closes)
    pf = FakePortfolio(positions={s: 0.0 for s in symbols})
    strat = FakeStrategy({s: 0.0 for s in symbols}, symbol_list=list(symbols))
    vol = FakeVolEstimator({s: 8000.0 for s in symbols})
    rm = CarverVolTargetingRiskManager(
        pf, strat, vol, data_handler=dh,
        instrument_weight_mode='min_variance',
        corr_lookback=lookback,
        corr_step_size=step_size,
    )
    return rm, dh, vol, pf


def test_auto_recalc_fires_every_step_size_completed_bars():
    """With step_size=3, ``calculate_instrument_weight`` should be auto-recalled
    on completed bars 3, 6, 9, ... — and never on forming bars."""
    rm, dh, _, _ = _make_min_variance_rm_with_closes(
        ['BTC', 'ETH'], n_bars=120, step_size=3,
    )

    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)

    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    # Drive 10 completed bars (alternating BTC/ETH so each symbol gets
    # exposure). step_size=3 → recalcs at bar 3, 6, 9.
    symbols_cycle = ['BTC', 'ETH']
    for i in range(10):
        sym = symbols_cycle[i % 2]
        rm.update_bar(_bar(symbol=sym, ts=datetime(2026, 1, 1 + i)))
    assert len(calls) == 3                                      # bars 3, 6, 9


def test_auto_recalc_ignores_forming_bars():
    """Forming bars must not increment the counter — only completed bars do."""
    rm, _, _, _ = _make_min_variance_rm_with_closes(
        ['BTC', 'ETH'], n_bars=120, step_size=2,
    )
    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)
    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    # 5 forming bars → 0 recalcs.
    for i in range(5):
        rm.update_bar(_bar(ts=datetime(2026, 1, 1 + i), is_forming=True))
    assert calls == []

    # Then 4 completed bars at step_size=2 → recalcs at bars 2 and 4.
    for i in range(4):
        rm.update_bar(_bar(ts=datetime(2026, 2, 1 + i)))
    assert len(calls) == 2


def test_corr_step_size_zero_disables_auto_recalc():
    """With step_size=0, no auto-recalc fires regardless of bar count."""
    rm, _, _, _ = _make_min_variance_rm_with_closes(
        ['BTC', 'ETH'], n_bars=120, step_size=0,
    )
    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)
    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    for i in range(20):
        rm.update_bar(_bar(ts=datetime(2026, 1, 1 + i)))
    assert calls == []


def test_auto_recalc_only_active_under_min_variance_mode():
    """``equal_weight`` mode must NOT auto-recall (the counter is short-circuited
    by the mode check in update_bar)."""
    pf = FakePortfolio(positions={'BTC': 0.0})
    strat = FakeStrategy({'BTC': 0.0}, symbol_list=['BTC'])
    vol = FakeVolEstimator({'BTC': 8000.0})
    rm = CarverVolTargetingRiskManager(
        pf, strat, vol, data_handler=FakeDataHandler(),
        instrument_weight_mode='equal_weight',
        corr_step_size=2,                                       # non-zero, but mode short-circuits
    )
    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)
    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    for i in range(10):
        rm.update_bar(_bar(ts=datetime(2026, 1, 1 + i)))
    assert calls == []


def test_auto_recalc_does_not_multi_increment_within_same_period():
    """N symbols at the same timestamp → only ONE period-crossing.

    Regression: an earlier implementation incremented per-event, so
    5 symbols at the same ts gave 5 increments per period. With the
    fix, all events in the same ``corr_timeframe`` bucket collapse to
    one tick.
    """
    rm, _, _, _ = _make_min_variance_rm_with_closes(
        ['A', 'B', 'C'], n_bars=120, step_size=3,
    )
    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)
    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    # 2 distinct timestamps × 3 symbols = 6 events.
    # New logic: 2 period-crossings, step_size=3 → 0 recalcs.
    # OLD (buggy) logic: 6 increments → 2 recalcs.
    for ts in [datetime(2026, 1, 1), datetime(2026, 1, 2)]:
        for sym in ['A', 'B', 'C']:
            rm.update_bar(_bar(symbol=sym, ts=ts))
    assert calls == []

    # One more distinct timestamp → 3 period-crossings → exactly 1 recalc.
    for sym in ['A', 'B', 'C']:
        rm.update_bar(_bar(symbol=sym, ts=datetime(2026, 1, 3)))
    assert len(calls) == 1


def test_auto_recalc_counts_corr_timeframe_periods_not_base_bars():
    """Many sub-corr_timeframe events within one period count as ONE crossing.

    With ``corr_timeframe='1d'``, 24 hourly bars on the same calendar day
    must produce one period-crossing (the first hour, opening the day).
    """
    rm, _, _, _ = _make_min_variance_rm_with_closes(
        ['A', 'B'], n_bars=120, step_size=2,
    )
    calls: List[int] = []
    original = rm.calculate_instrument_weight

    def spy(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)
    rm.calculate_instrument_weight = spy                        # type: ignore[assignment]

    # 24 hourly bars × 2 symbols on 2026-01-01 — all in the same daily period.
    # The fix collapses both axes (multi-symbol AND sub-period base) to one
    # period crossing for the entire day.
    for hour in range(24):
        for sym in ['A', 'B']:
            rm.update_bar(_bar(
                symbol=sym,
                ts=datetime(2026, 1, 1, hour, 0, 0),
            ))
    assert calls == []                                          # 1 period-crossing only, < step_size=2

    # Two hours × 2 symbols on 2026-01-02 → second period-crossing → step_size=2 fires.
    for hour in range(2):
        for sym in ['A', 'B']:
            rm.update_bar(_bar(
                symbol=sym,
                ts=datetime(2026, 1, 2, hour, 0, 0),
            ))
    assert len(calls) == 1
