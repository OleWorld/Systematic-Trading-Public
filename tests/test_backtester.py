"""
Unit tests for the ``Backtester`` event-loop engine.

The engine is the orchestration heart of the system — it drains the
events queue and routes each event to the right consumer — but it was
previously exercised only by the ``backtests/test_ewmac.py`` smoke
script, which pytest does not collect. A regression in the wiring (wrong
processing order, a mis-routed event, a swallowed unknown event, a
leaked bar-timestamp context) would pass the rest of the suite.

These tests pin the documented contract directly, with recording stubs
for the six wired modules:

- Bar-processing order: portfolio → execution → strategy → risk_manager.
- ``OrderEvent`` → ``execution.execute_order``.
- ``FillEvent`` → ``portfolio.update_fill``.
- An unknown event type raises ``TypeError`` (no silent fall-through).
- The per-bar timestamp ContextVar is set while a ``BarEvent`` is being
  processed and cleared once the loop finishes.
- The loop terminates when ``data_handler.continue_backtest`` goes False
  and the queue is drained.

Run from the repo root:  pytest tests/test_backtester.py -v
"""

import queue as thread_queue
from datetime import datetime
from typing import Any, List, Optional

import pytest

from backtester import Backtester
from event import BarEvent, Direction, FillEvent, OrderEvent, OrderType
from logging_setup import clear_current_bar_timestamp
from logging_setup._context import _current_bar_ts, _format_ts


DEFAULT_TS = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture(autouse=True)
def _reset_bar_ts():
    """Reset the bar-timestamp ContextVar so engine state never leaks
    between tests (the engine clears it at the end of ``run`` but a test
    that asserts mid-run state should start from a known baseline)."""
    clear_current_bar_timestamp()
    yield
    clear_current_bar_timestamp()


# ──────────────────────────────────────────────
# Recording doubles — every consumer logs its call into a shared list so
# the *interleaving* across modules is observable, not just per-module.
# ──────────────────────────────────────────────

class RecordingDataHandler:
    """Emits a fixed list of items (one per ``update_bar``) onto the queue,
    then flips ``continue_backtest`` off — mirroring ``HistoricDataHandler``."""

    def __init__(self, events_queue, items: List[Any]):
        self._queue = events_queue
        self._items = list(items)
        self._i = 0
        self.continue_backtest = True

    def update_bar(self) -> None:
        if self._i < len(self._items):
            self._queue.put(self._items[self._i])
            self._i += 1
        if self._i >= len(self._items):
            self.continue_backtest = False


class RecordingPortfolio:
    """Logs update_bar / update_fill; captures the bar-timestamp ContextVar
    value seen at the moment ``update_bar`` runs."""

    def __init__(self, call_log: List[Any]):
        self._log = call_log
        self.seen_bar_ts: List[str] = []

    def update_bar(self, event: BarEvent) -> None:
        self._log.append('portfolio.update_bar')
        self.seen_bar_ts.append(_current_bar_ts.get())

    def update_fill(self, event: FillEvent) -> None:
        self._log.append(('portfolio.update_fill', event))

    def finalize(self) -> None:
        # Post-run hook the engine calls once after the event loop drains.
        # No-op here (and deliberately NOT appended to the call log — the
        # routing tests pin the per-event call sequence, not the epilogue).
        return None


class RecordingExecution:
    """Logs update_bar / execute_order. Optionally emits one FillEvent the
    first time ``execute_order`` is called (to exercise fill routing)."""

    def __init__(self, call_log: List[Any], events_queue,
                 emit_fill: Optional[FillEvent] = None):
        self._log = call_log
        self._queue = events_queue
        self._emit_fill = emit_fill

    def update_bar(self, event: BarEvent) -> None:
        self._log.append('execution.update_bar')

    def execute_order(self, order: OrderEvent) -> None:
        self._log.append(('execution.execute_order', order))
        if self._emit_fill is not None:
            self._queue.put(self._emit_fill)
            self._emit_fill = None


class RecordingStrategy:
    def __init__(self, call_log: List[Any]):
        self._log = call_log

    def update_bar(self, event: BarEvent) -> None:
        self._log.append('strategy.update_bar')


class RecordingRiskManager:
    """Logs update_bar. Optionally emits one OrderEvent the first time
    ``update_bar`` is called (to exercise order routing)."""

    def __init__(self, call_log: List[Any], events_queue,
                 emit_order: Optional[OrderEvent] = None):
        self._log = call_log
        self._queue = events_queue
        self._emit_order = emit_order

    def update_bar(self, event: BarEvent) -> None:
        self._log.append('risk_manager.update_bar')
        if self._emit_order is not None:
            self._queue.put(self._emit_order)
            self._emit_order = None


# ──────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────

def _bar(symbol: str = 'BTC', ts: Optional[datetime] = None) -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=ts if ts is not None else DEFAULT_TS,
        open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
        period='1d', is_forming=False,
    )


def _order() -> OrderEvent:
    return OrderEvent(
        symbol='BTC', order_type=OrderType.MKT, quantity=1.0,
        direction=Direction.BUY, timestamp=DEFAULT_TS, order_id='O1',
    )


def _fill() -> FillEvent:
    return FillEvent(
        timestamp=DEFAULT_TS, symbol='BTC', exchange='SIM', quantity=1.0,
        direction=Direction.BUY, fill_notional=100.0, commission=0.0,
        order_id='O1',
    )


def _build(items: List[Any], *, emit_order: Optional[OrderEvent] = None,
           emit_fill: Optional[FillEvent] = None):
    """Wire a Backtester over a real queue with recording doubles.

    Returns ``(bt, call_log, portfolio)``.
    """
    q = thread_queue.Queue()
    call_log: List[Any] = []
    dh = RecordingDataHandler(q, items)
    portfolio = RecordingPortfolio(call_log)
    execution = RecordingExecution(call_log, q, emit_fill=emit_fill)
    strategy = RecordingStrategy(call_log)
    risk_manager = RecordingRiskManager(call_log, q, emit_order=emit_order)
    bt = Backtester(
        events_queue=q, data_handler=dh, strategy=strategy,
        portfolio=portfolio, risk_manager=risk_manager,
        execution_handler=execution,
    )
    return bt, call_log, portfolio


# ──────────────────────────────────────────────
# Bar-consumer processing order
# ──────────────────────────────────────────────

def test_bar_consumers_called_in_documented_order():
    """A single BarEvent drives the four consumers in the exact documented
    order: portfolio → execution → strategy → risk_manager."""
    bt, call_log, _ = _build([_bar()])
    bt.run()
    assert call_log == [
        'portfolio.update_bar',
        'execution.update_bar',
        'strategy.update_bar',
        'risk_manager.update_bar',
    ]


def test_order_preserved_across_multiple_bars():
    """Two bars → the 4-consumer cycle repeats once per bar, in order."""
    bt, call_log, _ = _build([_bar(ts=datetime(2026, 1, 1)),
                              _bar(ts=datetime(2026, 1, 2))])
    bt.run()
    one_cycle = [
        'portfolio.update_bar', 'execution.update_bar',
        'strategy.update_bar', 'risk_manager.update_bar',
    ]
    assert call_log == one_cycle * 2


# ──────────────────────────────────────────────
# Event-type routing
# ──────────────────────────────────────────────

def test_order_event_routed_to_execute_order():
    """An OrderEvent emitted by the risk manager is drained in the same
    inner loop and routed to execution.execute_order — after the four
    bar-consumers have run."""
    order = _order()
    bt, call_log, _ = _build([_bar()], emit_order=order)
    bt.run()
    assert call_log == [
        'portfolio.update_bar',
        'execution.update_bar',
        'strategy.update_bar',
        'risk_manager.update_bar',
        ('execution.execute_order', order),
    ]


def test_fill_event_routed_to_update_fill():
    """Full chain in one bar: risk_manager emits an OrderEvent →
    execution.execute_order emits a FillEvent → portfolio.update_fill."""
    order = _order()
    fill = _fill()
    bt, call_log, _ = _build([_bar()], emit_order=order, emit_fill=fill)
    bt.run()
    assert call_log == [
        'portfolio.update_bar',
        'execution.update_bar',
        'strategy.update_bar',
        'risk_manager.update_bar',
        ('execution.execute_order', order),
        ('portfolio.update_fill', fill),
    ]


def test_unknown_event_type_raises_type_error():
    """A non-event object on the queue must raise TypeError, never fall
    through silently."""
    bt, _, _ = _build([object()])
    with pytest.raises(TypeError, match="Unknown event type"):
        bt.run()


# ──────────────────────────────────────────────
# Per-bar timestamp ContextVar lifecycle
# ──────────────────────────────────────────────

def test_bar_timestamp_set_during_processing():
    """The engine sets the bar-timestamp ContextVar before driving the
    consumers, so the portfolio sees the bar's timestamp while processing."""
    ts = datetime(2026, 3, 15, 9, 30, 0)
    bt, _, portfolio = _build([_bar(ts=ts)])
    bt.run()
    assert portfolio.seen_bar_ts == [_format_ts(ts)]


def test_bar_timestamp_cleared_after_run():
    """The ContextVar is reset to the dash sentinel once the loop ends, so
    post-backtest log lines aren't stamped with a stale bar timestamp."""
    bt, _, _ = _build([_bar()])
    bt.run()
    assert _current_bar_ts.get() == '-'


# ──────────────────────────────────────────────
# Loop termination
# ──────────────────────────────────────────────

def test_empty_stream_processes_nothing_and_terminates():
    """No bars → no consumer calls, clean return (no hang)."""
    bt, call_log, _ = _build([])
    bt.run()
    assert call_log == []
