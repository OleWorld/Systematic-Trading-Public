"""
Unit tests for the ``Strategy`` ABC and the ``sizing_with_probability`` helper.

Strategy ABC tests use a minimal FakeDataHandler stub and a scripted
DummyStrategy subclass. Strategies write a forecast dict to
``self.forecasts[symbol]`` (read by the risk manager via
``strategy.get_forecast(symbol)`` on every completed bar).

Run from the repo root:  pytest tests/test_strategy.py -v
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest

from event import BarEvent
from strategy import Strategy


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────

class FakeDataHandler:
    """Minimal data handler — get_latest_bars returns a canned DataFrame."""

    def __init__(self, frame: Optional[pd.DataFrame] = None):
        self._frame = frame if frame is not None else pd.DataFrame()

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> pd.DataFrame:
        return self._frame.tail(n)

    def get_latest_bar(self, symbol: str,
                       timeframe: Optional[str] = None) -> Optional[Any]:
        return None


class DummyStrategy(Strategy):
    """Concrete Strategy with a caller-supplied calculate_forecast body.

    Lets each test script exactly what calculate_forecast does without
    having to subclass for every scenario.
    """

    def __init__(self, data_handler, symbol_list,
                 forecast_fn: Optional[Callable[['DummyStrategy', BarEvent],
                                                Optional[Dict[str, Any]]]] = None):
        super().__init__(data_handler, symbol_list)
        self._forecast_fn = forecast_fn

    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        if self._forecast_fn is None:
            return None
        return self._forecast_fn(self, event)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _bar(symbol: str = 'BTC', ts: Optional[datetime] = None,
         open_: float = 1.0, high: float = 2.0, low: float = 0.5,
         close: float = 1.5, volume: float = 10.0) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timestamp=ts if ts is not None else datetime(2026, 1, 1, 12, 0, 0),
        open=open_, high=high, low=low, close=close, volume=volume,
        period='1h', is_forming=False,
    )


def _build(forecast_fn=None, symbols=('BTC',)):
    dh = FakeDataHandler()
    s = DummyStrategy(dh, list(symbols), forecast_fn=forecast_fn)
    return s


# ──────────────────────────────────────────────
# ABC instantiation guard
# ──────────────────────────────────────────────

def test_strategy_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Strategy(FakeDataHandler(), ['BTC'])  # type: ignore[abstract]


# ──────────────────────────────────────────────
# Forecast cache initialization
# ──────────────────────────────────────────────

def test_forecasts_initialized_to_zero_for_each_symbol():
    s = _build(symbols=('BTC', 'ETH', 'SOL'))
    assert s.forecasts == {'BTC': 0.0, 'ETH': 0.0, 'SOL': 0.0}


def test_get_forecast_default_is_zero():
    s = _build()
    assert s.get_forecast('BTC') == 0.0
    assert s.get_forecast('UNKNOWN') == 0.0


# ──────────────────────────────────────────────
# update_bar — symbol filtering
# ──────────────────────────────────────────────

def test_update_bar_ignores_symbol_not_in_list():
    calls = []

    def fn(self, event):
        calls.append(event.symbol)
        return None

    s = _build(forecast_fn=fn, symbols=('BTC',))
    s.update_bar(_bar(symbol='ETH'))
    assert calls == []
    assert s.get_records('ETH').empty
    assert s.get_records('BTC').empty


# ──────────────────────────────────────────────
# update_bar — row construction
# ──────────────────────────────────────────────

def test_update_bar_records_ohlcv_when_no_extras():
    s = _build(forecast_fn=None)
    ts = datetime(2026, 1, 1, 12, 0, 0)
    s.update_bar(_bar(symbol='BTC', ts=ts, open_=1.0, high=2.0, low=0.5,
                      close=1.5, volume=10.0))
    rec = s.get_records('BTC')
    assert len(rec) == 1
    row = rec.iloc[0]
    assert row['open'] == 1.0
    assert row['high'] == 2.0
    assert row['low'] == 0.5
    assert row['close'] == 1.5
    assert row['volume'] == 10.0
    assert rec.index[0] == ts
    # No 'forecast' or 'signal' column when calculate_forecast returns None.
    assert 'forecast' not in rec.columns


def test_update_bar_merges_extras_from_calculate_forecast():
    def fn(self, event):
        return {'sma': 42.0, 'rsi': 70.0}

    s = _build(forecast_fn=fn)
    s.update_bar(_bar())
    rec = s.get_records('BTC')
    assert rec.iloc[0]['sma'] == 42.0
    assert rec.iloc[0]['rsi'] == 70.0


def test_update_bar_does_not_have_events_queue_attribute():
    """Strategy no longer holds an events_queue; the attribute is gone."""
    def fn(self, event):
        return {'forecast': 50.0}

    s = _build(forecast_fn=fn)
    s.update_bar(_bar())
    assert not hasattr(s, 'events_queue')


# ──────────────────────────────────────────────
# Forecast handling
# ──────────────────────────────────────────────

def test_forecast_in_extras_updates_cache_and_record():
    def fn(self, event):
        return {'forecast': 75.0}

    s = _build(forecast_fn=fn)
    s.update_bar(_bar())
    assert s.get_forecast('BTC') == 75.0
    rec = s.get_records('BTC')
    assert rec.iloc[0]['forecast'] == 75.0


def test_forecast_clamps_to_plus_100():
    def fn(self, event):
        return {'forecast': 250.0}

    s = _build(forecast_fn=fn)
    s.update_bar(_bar())
    assert s.get_forecast('BTC') == 100.0
    assert s.get_records('BTC').iloc[0]['forecast'] == 100.0


def test_forecast_clamps_to_minus_100():
    def fn(self, event):
        return {'forecast': -250.0}

    s = _build(forecast_fn=fn)
    s.update_bar(_bar())
    assert s.get_forecast('BTC') == -100.0
    assert s.get_records('BTC').iloc[0]['forecast'] == -100.0


def test_forecast_zero_clears_to_flat():
    def fn_long(self, event):
        return {'forecast': 100.0}

    s = _build(forecast_fn=fn_long)
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 0, 0, 0)))
    assert s.get_forecast('BTC') == 100.0

    # Swap function to return 0.0 — the cached forecast should follow.
    s._forecast_fn = lambda self, e: {'forecast': 0.0}
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 1, 0, 0)))
    assert s.get_forecast('BTC') == 0.0


def test_forecast_nan_skips_cache_update():
    """NaN forecasts (warmup) are recorded as-is but do NOT update the cached value."""
    def fn_first(self, event):
        return {'forecast': 50.0}

    s = _build(forecast_fn=fn_first)
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 0, 0, 0)))
    assert s.get_forecast('BTC') == 50.0

    # Now feed a NaN forecast — cache should remain at 50.0.
    s._forecast_fn = lambda self, e: {'forecast': float('nan')}
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 1, 0, 0)))
    assert s.get_forecast('BTC') == 50.0
    rec = s.get_records('BTC')
    # Recorded NaN value preserved for diagnostics.
    assert pd.isna(rec.iloc[1]['forecast'])


def test_returning_none_leaves_cache_unchanged():
    """calculate_forecast returning None records OHLCV-only and leaves the
    cached forecast at its prior value."""
    def fn_first(self, event):
        return {'forecast': 75.0}

    s = _build(forecast_fn=fn_first)
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 0, 0, 0)))
    assert s.get_forecast('BTC') == 75.0

    s._forecast_fn = lambda self, e: None
    s.update_bar(_bar(ts=datetime(2026, 1, 1, 1, 0, 0)))
    assert s.get_forecast('BTC') == 75.0          # unchanged


# ──────────────────────────────────────────────
# get_records — empty and populated
# ──────────────────────────────────────────────

def test_get_records_empty_when_no_rows_recorded():
    s = _build()
    df = s.get_records('BTC')
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_records_returns_dataframe_with_datetime_index():
    s = _build()
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 13, 0, 0)
    s.update_bar(_bar(ts=t0))
    s.update_bar(_bar(ts=t1))
    df = s.get_records('BTC')
    assert len(df) == 2
    assert list(df.index) == [t0, t1]
    assert 'close' in df.columns


# ──────────────────────────────────────────────
# sizing_with_probability pure-function tests (unchanged)
# ──────────────────────────────────────────────

def test_sizing_with_probability_at_pivot_is_zero():
    # Default num_classes=2: the pivot is p=0.5.
    assert Strategy.sizing_with_probability(0.5) == 0.0
    # num_classes=3: the pivot is p=1/3.
    assert abs(Strategy.sizing_with_probability(1.0 / 3.0, num_classes=3)) < 1e-12


def test_sizing_with_probability_below_pivot_is_negative():
    assert Strategy.sizing_with_probability(0.4) < 0.0


def test_sizing_with_probability_boundaries():
    assert Strategy.sizing_with_probability(1.0) == 1.0
    assert Strategy.sizing_with_probability(1.5) == 1.0
    assert Strategy.sizing_with_probability(0.0) == -1.0
    assert Strategy.sizing_with_probability(-0.5) == -1.0


def test_sizing_with_probability_reference_values():
    assert abs(Strategy.sizing_with_probability(0.875) - 0.7432) < 1e-3
    assert abs(Strategy.sizing_with_probability(0.750) - 0.4363) < 1e-3
    assert abs(Strategy.sizing_with_probability(0.625) - 0.2041) < 1e-3


def test_sizing_with_probability_monotonic():
    samples = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99]
    mags = [Strategy.sizing_with_probability(p) for p in samples]
    assert all(a <= b + 1e-12 for a, b in zip(mags, mags[1:]))
