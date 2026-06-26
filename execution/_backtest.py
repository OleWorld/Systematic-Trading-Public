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

    Fill-timing modes (``fill_on``):

    - ``'signal_close'`` (default): orders fill on the bar that generated the
      signal. MKT fills at that bar's close; LMT fills at the limit price if
      the bar's range satisfied it, otherwise it joins ``pending_orders`` and
      is evaluated on subsequent bars. This collapses signal -> order -> fill
      into a single bar — i.e. **zero latency** between signal generation and
      order placement. Acceptable as a backtest idealization, but it is not
      representative of live execution.
    - ``'next_open'``: all orders queue in ``pending_orders`` and fill on the
      next bar's open (MKT) or when the bar's range satisfies the limit (LMT).
      A more conservative, 1-bar-delayed model.

    Limit orders that carry over to later bars use the gap-favorable
    ``min(limit, bar.open)`` / ``max(limit, bar.open)`` convention because at
    that point they are already resting on the book at the new bar's open.
    """

    def __init__(self, events_queue: queue.Queue[Any],
                 instruments: Dict[str, "InstrumentConfig"],
                 fill_on: str,
                 exchange_name: str = 'BACKTEST'):
        self.events_queue = events_queue
        # Per-symbol metadata registry: slippage / commission models and the
        # point_value used to compute true dollar notional for rate-commission.
        self.instruments = instruments
        self.exchange_name = exchange_name
        if fill_on not in ('signal_close', 'next_open'):
            raise ValueError(
                f"Unknown fill_on: '{fill_on}'. Must be 'signal_close' or 'next_open'."
            )
        self.fill_on = fill_on

        self.pending_orders: Dict[str, OrderEvent] = {}
        self._current_bars: Dict[str, BarEvent] = {}

    def execute_order(self, event: OrderEvent) -> None:
        """
        Route a new order. Under ``'signal_close'`` we attempt an immediate
        fill against the current (signal-generation) bar; if a LMT order's
        range is not satisfied, it falls through to ``pending_orders``.
        Under ``'next_open'`` all orders are queued for the next bar.
        """
        if self.fill_on == 'next_open':
            self.pending_orders[event.order_id] = event
            return

        if self.fill_on == 'signal_close':
            bar = self._current_bars.get(event.symbol)
            if bar is None:
                raise RuntimeError(
                    f"No current bar available for {event.symbol!r} at execute_order time; "
                    "signal_close mode requires a bar to have been observed first."
                )
            fill_price = self._try_fill_same_bar(event, bar)
            if fill_price is not None:
                self._emit_fill(event, fill_price, bar)
            else:
                # LMT that didn't satisfy the signal bar's range — wait for
                # later bars to fill via the standard pending-orders path.
                self.pending_orders[event.order_id] = event
            return

        raise ValueError(f"Unexpected fill_on: {self.fill_on!r}")

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

    def _try_fill_same_bar(self, order: OrderEvent, bar: BarEvent) -> Optional[float]:
        """
        Fill price for an order arriving on its own signal bar. MKT fills at
        the bar's close; LMT fills at the limit price when the bar's range
        satisfies it. Gap-favorable pricing does not apply here because the
        order was not on the book at the bar's open.
        """
        if order.order_type == OrderType.MKT:
            return bar.close
        elif order.order_type == OrderType.LMT:
            if order.direction == Direction.BUY and bar.low <= order.price:
                return order.price
            if order.direction == Direction.SELL and bar.high >= order.price:
                return order.price
            return None
        else:
            raise ValueError(f"Unexpected order_type: {order.order_type!r}")

    def _try_fill(self, order: OrderEvent, bar: BarEvent) -> Optional[float]:
        """
        Fill price for a pending order carried over to a later bar. MKT fills
        at the bar's open (only reachable under ``fill_on='next_open'``). LMT
        uses gap-favorable ``min/max`` with ``bar.open`` because the order is
        resting on the book at the new bar's open.
        """
        if order.order_type == OrderType.MKT:
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

        ``fill_notional`` stays in price space (``qty * fill_price``) — the
        portfolio applies ``point_value`` when converting to dollar PnL.
        Commission, however, is charged in dollars, so the contract multiplier
        is passed to the rate-commission notional here.
        """
        cfg = self.instruments[order.symbol]
        fill_price = cfg.slippage.apply(base_price, order.direction)
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
