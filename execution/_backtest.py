"""BacktestExecution — simulated exchange for backtesting."""

import logging
import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from event import BarEvent, OrderEvent, FillEvent, OrderType, Direction
from execution._base import ExecutionHandler

if TYPE_CHECKING:  # avoid a config<->execution import cycle at runtime
    from config import InstrumentConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Backtest Execution
# ──────────────────────────────────────────────

class BacktestExecution(ExecutionHandler):
    """
    Simulated exchange for backtesting.

    The execution handler's only job is to decide whether the requested
    fill condition (price/volume) is met on the current bar. It does not
    perform any margin or solvency checks — those are owned entirely by
    the portfolio, which reacts to the resulting ``FillEvent``. As a
    consequence, this class holds no reference to the portfolio.

    Fill timing (single model): a MKT order fills on the bar that generated
    the signal, at that bar's close; a LMT order always RESTS — it joins
    ``pending_orders`` and is evaluated from the next bar onward. A LMT is
    never tested against its own signal bar's range: the order comes into
    existence at that bar's close, so its high/low printed before the order
    could have been on the book — filling there would be look-ahead. Because
    strategies compute forecasts from **finalized** values only (data
    through the previous completed bar), a same-bar close fill is achievable
    in live trading (e.g. a market-on-close order placed during the bar
    using the already-final forecast) — the effective signal→fill latency is
    one full bar of information. (The old ``fill_on='next_open'`` mode was
    removed as redundant: it only moved the fill from close(*t*) to
    open(*t+1*) while the signal already lagged a bar; callers wanting more
    delay should lag the forecast itself.)

    Cross-symbol orders (the portfolio's margin-call liquidations): an
    order can arrive for a symbol whose bar for the decision period has
    not streamed yet — its last seen bar is *stale* (an earlier
    timestamp). Filling against that stale bar would trade at a past price
    and timestamp, so such orders are deferred to ``pending_orders`` and
    fill on the symbol's next bar: at that bar's **close** when it belongs
    to the same period the order was decided in (signal-close semantics),
    or at its **open** when the symbol had no bar in the decision period
    (effectively next-open for that symbol).

    Limit orders that carry over to later bars use the gap-favorable
    ``min(limit, bar.open)`` / ``max(limit, bar.open)`` convention because at
    that point they are already resting on the book at the new bar's open.
    """

    def __init__(self, events_queue: queue.Queue[Any],
                 instruments: Dict[str, "InstrumentConfig"],
                 exchange_name: str = 'BACKTEST'):
        self.events_queue = events_queue
        # Per-symbol metadata registry: slippage / commission models and the
        # point_value used to compute true dollar notional for rate-commission.
        self.instruments = instruments
        self.exchange_name = exchange_name

        self.pending_orders: Dict[str, OrderEvent] = {}
        self._current_bars: Dict[str, BarEvent] = {}

    def execute_order(self, event: OrderEvent) -> None:
        """
        Route a new order. A MKT order fills immediately at the current
        (signal-generation) bar's close. A LMT order always RESTS: it joins
        ``pending_orders`` and is evaluated from the next bar onward via
        ``_try_fill`` — never against its own signal bar's range, whose
        high/low printed before the order existed (filling there would be
        look-ahead; the close is the first moment the order can be on the
        book). An order for a symbol whose current bar predates the order's
        own timestamp (a cross-symbol submission, e.g. a margin-call
        liquidation decided while another symbol's bar was being processed)
        is likewise deferred to ``pending_orders`` instead of filling at
        the stale bar's past price/timestamp.
        """
        bar = self._current_bars.get(event.symbol)
        if bar is None:
            raise RuntimeError(
                f"No current bar available for {event.symbol!r} at execute_order time; "
                "an order requires a bar to have been observed first."
            )
        if event.timestamp is not None and bar.timestamp < event.timestamp:
            # Stale bar: this symbol's bar for the decision period hasn't
            # streamed yet. Defer — it fills on the symbol's next bar via
            # the standard pending-orders path (see _try_fill).
            self.pending_orders[event.order_id] = event
            return
        if event.order_type == OrderType.MKT:
            self._emit_fill(event, bar.close, bar)
        elif event.order_type == OrderType.LMT:
            # Rests on the book; evaluated from the next bar via _try_fill.
            self.pending_orders[event.order_id] = event
        else:
            raise ValueError(f"Unexpected order_type: {event.order_type!r}")

    def update_bar(self, event: BarEvent) -> None:
        """
        Process a new bar: store it, then attempt to fill any pending orders
        for this symbol. OHLC fields are guaranteed non-NaN by the
        ``DataHandler`` gate.
        """
        self._current_bars[event.symbol] = event

        to_fill: List[str] = []

        for order_id, order in self.pending_orders.items():
            if order.symbol != event.symbol:
                continue

            fill_price = self._try_fill(order, event)
            if fill_price is not None:
                self._emit_fill(order, fill_price, event)
                to_fill.append(order_id)

        for order_id in to_fill:
            del self.pending_orders[order_id]

    def _try_fill(self, order: OrderEvent, bar: BarEvent) -> Optional[float]:
        """
        Fill price for a pending order evaluated on a later-arriving bar.

        MKT here means a *deferred cross-symbol* order (e.g. a margin-call
        liquidation queued while the symbol's bar for the decision period
        had not streamed yet): the symbol's own bar for that period fills
        at its close (signal-close semantics — same-period bars close at
        the same wall-clock time the decision was made), while a
        later-period bar fills at its open (the order was resting; the
        open is the first available price). LMT uses gap-favorable
        ``min/max`` with ``bar.open`` because the order is resting on the
        book at the new bar's open.
        """
        if order.order_type == OrderType.MKT:
            if order.timestamp is not None and bar.timestamp == order.timestamp:
                return bar.close
            return bar.open
        elif order.order_type == OrderType.LMT:
            if order.direction == Direction.BUY and bar.low <= order.price:
                return min(order.price, bar.open)
            if order.direction == Direction.SELL and bar.high >= order.price:
                return max(order.price, bar.open)
            return None  # Limit order not filled this bar
        else:
            raise ValueError(f"Unexpected order_type: {order.order_type!r}")

    def _emit_fill(self, order: OrderEvent, base_price: float, bar: BarEvent) -> None:
        """Create and enqueue a FillEvent with per-symbol slippage and commission applied.

        Slippage models taker market impact, so it applies to MKT fills
        only: a limit order fills at its limit price (or gap-favorably
        better) by definition — slipping it would produce a fill worse
        than the limit, which a real book cannot do.

        ``fill_notional`` stays in price space (``qty * fill_price``) — the
        portfolio applies ``point_value`` when converting to dollar PnL.
        Commission, however, is charged in dollars, so the contract multiplier
        is passed to the rate-commission notional here.
        """
        cfg = self.instruments[order.symbol]
        if order.order_type == OrderType.MKT:
            fill_price = cfg.slippage.apply(base_price, order.direction)
        elif order.order_type == OrderType.LMT:
            fill_price = base_price
        else:
            raise ValueError(f"Unexpected order_type: {order.order_type!r}")
        qty = order.quantity

        fill_notional = qty * fill_price
        commission = cfg.commission.calculate(qty, fill_price, cfg.point_value)

        fill = FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            exchange=self.exchange_name,
            quantity=qty,
            direction=order.direction,
            fill_notional=fill_notional,
            commission=commission,
            order_id=order.order_id,
        )
        self.events_queue.put(fill)
