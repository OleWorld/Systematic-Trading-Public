"""
Unit + integration tests for ``Orchestrator`` (multi-strategy aggregation).

Covers:
* Combination math — equal-weight average, explicit weights, FDM applied
  only with >= 2 contributors, ``+/-FORECAST_CAP`` clamp.
* Partial coverage / warmup — renormalization over the contributing
  subset, ``is_warmed_up`` flipping on the first contributor, a symbol
  covered by no warmed child staying ``None``, and all-zero-weight
  contributors yielding ``None``.
* Constructor validation — empty strategies, weight key mismatch /
  negative / non-sum-to-1, non-positive ``fdm``, missing child surface.
* Drop-in — real ``EWMACStrategy`` + ``RSIMRStrategy`` aggregated and
  cross-checked against a manual weighted-sum x FDM recomputation, and a
  ``VolTargetingRiskManager`` accepting the orchestrator in the strategy
  slot and sizing off the combined forecast.

Run from repo root:  pytest tests/test_orchestrator.py -v
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd
import pytest

from config import uniform_registry
from event import BarEvent
from orchestrator import Orchestrator
from riskmanager import VolTargetingRiskManager
from strategy import EWMACStrategy, RSIMRStrategy, Strategy


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class _StubStrategy:
    """Controllable forecast oracle implementing the orchestrator's child surface.

    Exposes ``symbol_list`` / ``get_forecast`` / ``is_warmed_up`` /
    ``update_bar``. ``get_forecast`` returns ``None`` (warmup) for any
    symbol whose forecast hasn't been set; ``is_warmed_up`` tracks that.
    """

    def __init__(self, forecasts: Dict[str, Optional[float]],
                 symbol_list: List[str]):
        self._forecasts: Dict[str, Optional[float]] = dict(forecasts)
        self.symbol_list: List[str] = list(symbol_list)
        self.update_calls: List[BarEvent] = []

    def update_bar(self, event: BarEvent) -> None:
        self.update_calls.append(event)

    def get_forecast(self, symbol: str) -> Optional[float]:
        return self._forecasts.get(symbol)

    def is_warmed_up(self, symbol: str) -> bool:
        return self._forecasts.get(symbol) is not None

    def set_forecast(self, symbol: str, value: Optional[float]) -> None:
        self._forecasts[symbol] = value


class _StepwiseDataHandler:
    """Reveals one bar at a time from a pre-built daily DataFrame.

    Copied from ``tests/test_ewmac.py`` per the project's strategy-testing
    convention. ``advance(i)`` exposes ``frame.iloc[: i]``.
    """

    def __init__(self, frame: pd.DataFrame, symbol: str, exec_tf: str = '1d'):
        self._frame = frame
        self._symbol = symbol
        self._exec_tf = exec_tf
        self._end = 0

    def advance(self, end: int) -> None:
        self._end = end

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> list:
        if symbol != self._symbol:
            return []
        if timeframe is not None and timeframe != self._exec_tf:
            return []
        slice_ = self._frame.iloc[: self._end].tail(n)
        return [_BarRow(timestamp=ts, close=float(row['Close']))
                for ts, row in slice_.iterrows()]


class _BarRow:
    """Lightweight stand-in exposing ``.timestamp`` + ``.close``."""

    def __init__(self, timestamp, close: float):
        self.timestamp = timestamp
        self.close = close


def _bar_event(symbol: str, ts, close: float) -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=ts,
        open=close, high=close, low=close, close=close, volume=1.0,
        period='1d', is_forming=False,
    )


def _build_frame(closes: List[float], start: datetime) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [start + timedelta(days=i) for i in range(len(closes))]
    )
    return pd.DataFrame({
        'Open': closes, 'High': closes, 'Low': closes,
        'Close': closes, 'Volume': [1.0] * len(closes),
    }, index=idx)


_START = datetime(2026, 1, 1)


def _one_bar(orch: Orchestrator, symbol: str, close: float = 100.0,
             ts=None) -> None:
    """Drive a single completed bar through the orchestrator."""
    orch.update_bar(_bar_event(symbol, ts or _START, close))


# ──────────────────────────────────────────────
# Combination math
# ──────────────────────────────────────────────

def test_equal_weight_average_of_two_contributors():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'BTC': 60.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b})           # equal weights, fdm=1
    _one_bar(orch, 'BTC')
    # 0.5*40 + 0.5*60 = 50, fdm=1 (>=2 contributors but fdm default 1.0)
    assert math.isclose(orch.get_forecast('BTC'), 50.0, rel_tol=1e-12)


def test_explicit_weights_blend():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'BTC': 60.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, weights={'a': 0.25, 'b': 0.75})
    _one_bar(orch, 'BTC')
    # 0.25*40 + 0.75*60 = 55
    assert math.isclose(orch.get_forecast('BTC'), 55.0, rel_tol=1e-12)


def test_fdm_applied_with_two_contributors():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'BTC': 60.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, fdm=1.2)
    _one_bar(orch, 'BTC')
    # weighted_sum = 50, effective_fdm = 1.2 → 60
    assert math.isclose(orch.get_forecast('BTC'), 60.0, rel_tol=1e-12)


def test_fdm_not_applied_with_single_contributor():
    # 'a' covers BTC, 'b' covers ETH only → BTC has one contributor.
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'ETH': 60.0}, ['ETH'])
    orch = Orchestrator({'a': a, 'b': b}, fdm=1.5)
    _one_bar(orch, 'BTC')
    # Lone contributor: no diversification uplift, fdm forced to 1.0.
    assert math.isclose(orch.get_forecast('BTC'), 40.0, rel_tol=1e-12)
    row = orch.get_records('BTC').iloc[0]
    assert row['n_contributors'] == 1
    assert math.isclose(row['fdm'], 1.0)


def test_combined_forecast_clamped_to_cap():
    a = _StubStrategy({'BTC': 100.0}, ['BTC'])
    b = _StubStrategy({'BTC': 100.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, fdm=2.0)
    _one_bar(orch, 'BTC')
    # 2.0 * 100 = 200 → clamped to +FORECAST_CAP (100).
    assert orch.get_forecast('BTC') == Strategy.FORECAST_CAP


def test_combined_forecast_clamped_to_negative_cap():
    a = _StubStrategy({'BTC': -100.0}, ['BTC'])
    b = _StubStrategy({'BTC': -100.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, fdm=2.0)
    _one_bar(orch, 'BTC')
    assert orch.get_forecast('BTC') == -Strategy.FORECAST_CAP


# ──────────────────────────────────────────────
# Partial coverage / warmup / renormalization
# ──────────────────────────────────────────────

def test_renormalize_over_available_during_warmup():
    # Both cover BTC; b not warmed yet → only a contributes at full weight.
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'BTC': None}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, weights={'a': 0.3, 'b': 0.7})
    _one_bar(orch, 'BTC')
    assert orch.is_warmed_up('BTC')
    # Only a contributes; weight renormalizes to 1.0 → 40.
    assert math.isclose(orch.get_forecast('BTC'), 40.0, rel_tol=1e-12)

    # Now b warms up → both contribute with renormalized 0.3 / 0.7.
    b.set_forecast('BTC', 80.0)
    _one_bar(orch, 'BTC', ts=_START + timedelta(days=1))
    # 0.3*40 + 0.7*80 = 68
    assert math.isclose(orch.get_forecast('BTC'), 68.0, rel_tol=1e-12)


def test_symbol_with_no_warmed_contributor_stays_none():
    a = _StubStrategy({'BTC': None}, ['BTC'])
    orch = Orchestrator({'a': a})
    _one_bar(orch, 'BTC')
    assert orch.get_forecast('BTC') is None
    assert not orch.is_warmed_up('BTC')
    row = orch.get_records('BTC').iloc[0]
    assert row['n_contributors'] == 0
    assert pd.isna(row['forecast'])
    assert row['fdm'] is None


def test_all_zero_weight_contributors_yield_none():
    # a (weight 0) covers BTC; b (weight 1) covers ETH only.
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'ETH': 60.0}, ['ETH'])
    orch = Orchestrator({'a': a, 'b': b}, weights={'a': 0.0, 'b': 1.0})
    _one_bar(orch, 'BTC')
    # Sole contributor carries zero weight → no effective forecast.
    assert orch.get_forecast('BTC') is None
    assert not orch.is_warmed_up('BTC')


def test_unknown_symbol_returns_none():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    orch = Orchestrator({'a': a})
    assert orch.get_forecast('XRP') is None
    assert not orch.is_warmed_up('XRP')


def test_is_warmed_up_monotone():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    orch = Orchestrator({'a': a})
    assert not orch.is_warmed_up('BTC')
    _one_bar(orch, 'BTC')
    assert orch.is_warmed_up('BTC')
    # A subsequent bar keeps it warmed.
    _one_bar(orch, 'BTC', ts=_START + timedelta(days=1))
    assert orch.is_warmed_up('BTC')


# ──────────────────────────────────────────────
# Symbol-list union + child driving
# ──────────────────────────────────────────────

def test_symbol_list_is_sorted_union():
    a = _StubStrategy({}, ['BTC', 'ETH'])
    b = _StubStrategy({}, ['ETH', 'SOL'])
    orch = Orchestrator({'a': a, 'b': b})
    assert orch.symbol_list == ['BTC', 'ETH', 'SOL']


def test_update_bar_drives_every_child():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'BTC': 60.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b})
    ev = _bar_event('BTC', _START, 100.0)
    orch.update_bar(ev)
    assert a.update_calls == [ev]
    assert b.update_calls == [ev]


def test_update_bar_drives_children_even_for_unmanaged_symbol():
    # Children are forwarded every event (they self-filter); the
    # orchestrator only skips its own aggregation for unmanaged symbols.
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    orch = Orchestrator({'a': a})
    ev = _bar_event('ETH', _START, 100.0)   # ETH not in any child universe
    orch.update_bar(ev)
    assert a.update_calls == [ev]
    assert orch.get_records('ETH').empty


# ──────────────────────────────────────────────
# Weights API
# ──────────────────────────────────────────────

def test_default_weights_are_equal():
    strats = {k: _StubStrategy({}, ['BTC']) for k in ('a', 'b', 'c')}
    orch = Orchestrator(strats)
    assert all(math.isclose(w, 1.0 / 3.0) for w in orch.strategy_weights.values())
    assert math.isclose(sum(orch.strategy_weights.values()), 1.0, abs_tol=1e-12)


def test_calculate_strategy_weights_resets_to_equal():
    a = _StubStrategy({}, ['BTC'])
    b = _StubStrategy({}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b}, weights={'a': 0.9, 'b': 0.1})
    orch.strategy_weights = {'a': 0.0, 'b': 1.0}
    orch.calculate_strategy_weights()
    assert orch.strategy_weights == {'a': 0.5, 'b': 0.5}


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────

def test_get_records_columns_and_per_label():
    a = _StubStrategy({'BTC': 40.0}, ['BTC'])
    b = _StubStrategy({'ETH': 60.0}, ['ETH'])   # does not cover BTC
    orch = Orchestrator({'a': a, 'b': b})
    _one_bar(orch, 'BTC')
    row = orch.get_records('BTC').iloc[0]
    for col in ('open', 'high', 'low', 'close', 'volume',
                'forecast', 'fdm', 'n_contributors',
                'forecast_a', 'weight_a', 'forecast_b', 'weight_b'):
        assert col in row.index
    assert math.isclose(row['forecast_a'], 40.0)
    assert math.isclose(row['weight_a'], 1.0)        # renormalized lone weight
    assert pd.isna(row['forecast_b'])                # b does not cover BTC
    assert row['weight_b'] == 0.0


def test_get_records_empty_for_unknown_symbol():
    orch = Orchestrator({'a': _StubStrategy({'BTC': 40.0}, ['BTC'])})
    assert orch.get_records('NOPE').empty


# ──────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────

def test_rejects_empty_strategies():
    with pytest.raises(ValueError, match="non-empty"):
        Orchestrator({})


def test_rejects_non_dict_strategies():
    with pytest.raises(ValueError, match="non-empty"):
        Orchestrator([_StubStrategy({}, ['BTC'])])      # type: ignore[arg-type]


def test_rejects_weight_key_mismatch():
    a = _StubStrategy({}, ['BTC'])
    b = _StubStrategy({}, ['BTC'])
    with pytest.raises(ValueError, match="weights keys"):
        Orchestrator({'a': a, 'b': b}, weights={'a': 0.5, 'c': 0.5})


def test_rejects_negative_weight():
    a = _StubStrategy({}, ['BTC'])
    b = _StubStrategy({}, ['BTC'])
    with pytest.raises(ValueError, match="finite and >= 0"):
        Orchestrator({'a': a, 'b': b}, weights={'a': 1.2, 'b': -0.2})


def test_rejects_weights_not_summing_to_one():
    a = _StubStrategy({}, ['BTC'])
    b = _StubStrategy({}, ['BTC'])
    with pytest.raises(ValueError, match="sum to 1"):
        Orchestrator({'a': a, 'b': b}, weights={'a': 0.5, 'b': 0.4})


def test_rejects_non_positive_fdm():
    a = _StubStrategy({}, ['BTC'])
    with pytest.raises(ValueError, match="fdm"):
        Orchestrator({'a': a}, fdm=0.0)
    with pytest.raises(ValueError, match="fdm"):
        Orchestrator({'a': a}, fdm=-1.0)
    with pytest.raises(ValueError, match="fdm"):
        Orchestrator({'a': a}, fdm=float('nan'))


def test_rejects_child_missing_surface():
    class _Bad:
        symbol_list = ['BTC']
        # no get_forecast / update_bar
    with pytest.raises(TypeError, match="symbol_list"):
        Orchestrator({'bad': _Bad()})


# ──────────────────────────────────────────────
# Drop-in: real EWMAC + RSIMR aggregation
# ──────────────────────────────────────────────

def _drive_pair(orch: Orchestrator, dh: _StepwiseDataHandler,
                frame: pd.DataFrame, symbol: str) -> None:
    for i in range(len(frame)):
        dh.advance(i + 1)
        ts = frame.index[i]
        close = float(frame['Close'].iloc[i])
        orch.update_bar(_bar_event(symbol, ts, close))


def test_real_strategies_combined_matches_manual_recomputation():
    symbol = 'BTC'
    rng = np.random.default_rng(7)
    closes = list(100.0 + np.cumsum(rng.normal(0, 1.0, size=120)))
    frame = _build_frame(closes, _START)
    dh = _StepwiseDataHandler(frame, symbol)

    ewmac = EWMACStrategy(
        dh, [symbol], variations={'2_4': {'fast': 2, 'slow': 4}},
        fdm=1.0, vol_lookback=5, forecast_scalar_lookback=5,
    )
    rsimr = RSIMRStrategy(
        dh, [symbol], variations={'3': {'window': 3}},
        fdm=1.0, forecast_scalar_lookback=5,
    )
    weights = {'ewmac': 0.6, 'rsimr': 0.4}
    fdm = 1.1
    orch = Orchestrator({'ewmac': ewmac, 'rsimr': rsimr},
                        weights=weights, fdm=fdm)
    _drive_pair(orch, dh, frame, symbol)

    fe = ewmac.get_forecast(symbol)
    fr = rsimr.get_forecast(symbol)
    assert fe is not None and fr is not None        # both warmed by bar 120
    cap = Strategy.FORECAST_CAP
    manual = fdm * (weights['ewmac'] * fe + weights['rsimr'] * fr)
    manual = max(-cap, min(cap, manual))
    assert math.isclose(orch.get_forecast(symbol), manual, rel_tol=1e-9)


# ──────────────────────────────────────────────
# Drop-in: risk manager accepts the orchestrator
# ──────────────────────────────────────────────

class _FakePortfolio:
    def __init__(self, balance=100_000.0, positions=None):
        self._balance = balance
        self.positions = positions if positions is not None else {}
        self.submitted: List[dict] = []

    def get_price(self, symbol):
        return None

    def calculate_balance(self):
        return self._balance

    def projected_position(self, symbol):
        # No pending-order ledger in the double: projected == realized.
        return self.positions.get(symbol, 0.0)

    def submit_order(self, symbol, quantity, direction, timestamp,
                     order_type, price=None):
        self.submitted.append({
            'symbol': symbol, 'quantity': quantity, 'direction': direction,
        })
        return None


class _FakeVol:
    def __init__(self, vols):
        self._vols = dict(vols)

    def update(self, event):
        pass

    def get_annual_vol(self, symbol):
        return self._vols.get(symbol)


class _FakeDH:
    timeframes = {'1d': 500}

    def get_latest_bars_df(self, symbol, n=1, timeframe=None):
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def count_bars(self, symbol, timeframe=None):
        return 0


def test_risk_manager_accepts_orchestrator_and_sizes_off_combined_forecast():
    from event import Direction

    a = _StubStrategy({'BTC': 50.0}, ['BTC'])
    b = _StubStrategy({'BTC': 50.0}, ['BTC'])
    orch = Orchestrator({'a': a, 'b': b})           # combined 50

    pf = _FakePortfolio(positions={'BTC': 0.0})
    rm = VolTargetingRiskManager(
        pf, orch, _FakeVol({'BTC': 8000.0}),
        data_handler=_FakeDH(),
        instruments=uniform_registry(['BTC']),
        annual_target_vol=0.25,
        vol_target_mode='percent_volatility',
        position_buffer=0.0,
        corr_step_size=0,                           # freeze weights for the test
    )
    assert rm.strategy is orch
    rm.instrument_weight = {'BTC': 1.0}             # pin a tradable weight

    ev = _bar_event('BTC', _START, 100.0)
    orch.update_bar(ev)                             # engine order: source first
    rm.update_bar(ev)
    # cash = 100k * 1 * 1 * 0.25 * (50/50) = 25_000 ; qty = 25_000 / 8_000
    assert len(pf.submitted) == 1
    assert pf.submitted[0]['direction'] == Direction.BUY
    assert math.isclose(pf.submitted[0]['quantity'], 25_000.0 / 8_000.0,
                        rel_tol=1e-12)
