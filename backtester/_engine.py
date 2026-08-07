"""Backtester event-loop engine."""

import logging

from event import BarEvent, OrderEvent, FillEvent
from logging_setup import clear_current_bar_timestamp, set_current_bar_timestamp

logger = logging.getLogger(__name__)


class Backtester:
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.

    Bar-processing order on each ``BarEvent`` — six consumers:
        portfolio.update_bar     → execution.update_bar
                                  → strategy.update_bar        (updates forecast cache)
                                  → universe_manager.update_bar (refreshes this bar's symbol)
                                  → correlation_manager.update_bar (cadenced refresh;
                                                                    returns Optional[CorrelationEvent])
                                  → risk_manager.update_bar     (reads strategy.get_forecast,
                                                                 submits resize order)

    Between the correlation manager's update and the risk manager's own
    ``update_bar``, the engine dispatches any pending ``UniverseEvent``s
    (drained from ``universe_manager.drain_events()``) and the
    ``CorrelationEvent`` (if one was returned this bar) synchronously to
    ``risk_manager.on_universe_event`` / ``on_correlation_event`` —
    universe events first, then the correlation event, so a symbol's
    liveness transition is visible before any weight recompute that
    transition may have triggered. This dispatch is INLINE, never via
    the FIFO events queue: a queued notification would only be drained on
    a *later* iteration of the inner loop, i.e. after this bar's sizing
    had already run on stale universe/weight state — same-timestamp
    sibling symbols would then size on stale weights while the triggering
    symbol alone got the fresh ones. Inline dispatch guarantees every
    symbol sized on this bar sees the same fresh state.

    The risk manager runs *last* in the bar chain so it sees this bar's
    freshly-updated forecast AND freshly-updated universe/weights.
    ``OrderEvent`` and ``FillEvent`` stages drain in subsequent iterations
    of the inner event loop.

    Callers wire each module explicitly and pass them in. See
    ``backtests/sample_backtest/backtest_ewmac_crypto.py`` for a worked
    example.
    """
    def __init__(self, events_queue, data_handler, strategy, portfolio,
                 risk_manager, execution_handler,
                 universe_manager, correlation_manager):
        self.events = events_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler
        self.universe_manager = universe_manager
        self.correlation_manager = correlation_manager

    def run(self):
        """
        Execute the backtest event loop.

        ``DataHandler`` emits ``BarEvent``s. Each bar drives the six bar-
        consumers (portfolio, execution, strategy, universe_manager,
        correlation_manager, risk_manager) in order, with any
        ``UniverseEvent``s/``CorrelationEvent`` dispatched inline to the
        risk manager between the correlation manager's update and the
        risk manager's own ``update_bar`` (see the class docstring for the
        inline-vs-queued rationale). The risk manager may emit
        ``OrderEvent``s, which the execution handler consumes to produce
        ``FillEvent``s, which the portfolio applies.
        """
        logger.info("Starting backtest...")

        while self.data_handler.continue_backtest:
            self.data_handler.update_bar()

            while not self.events.empty():
                event = self.events.get(False)

                if isinstance(event, BarEvent):
                    set_current_bar_timestamp(event.timestamp)
                    logger.debug(
                        "[BAR] %s | O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
                        event.symbol,
                        event.open, event.high, event.low, event.close, event.volume,
                    )
                    self.portfolio.update_bar(event)
                    self.execution_handler.update_bar(event)
                    self.strategy.update_bar(event)
                    self.universe_manager.update_bar(event)
                    corr_event = self.correlation_manager.update_bar(event)
                    # Inline dispatch — NEVER via the FIFO queue: a queued
                    # notification would land after this bar's sizing, so
                    # the triggering symbol would size on stale weights
                    # while same-timestamp siblings size on fresh ones.
                    for u_evt in self.universe_manager.drain_events():
                        self.risk_manager.on_universe_event(u_evt)
                    if corr_event is not None:
                        self.risk_manager.on_correlation_event(corr_event)
                    self.risk_manager.update_bar(event)

                elif isinstance(event, OrderEvent):
                    logger.info(
                        "[ORDER] %s %s %.6f %s @ %s | id=%s",
                        event.order_type.value, event.direction.value, event.quantity,
                        event.symbol, event.price, event.order_id,
                    )
                    self.execution_handler.execute_order(event)

                elif isinstance(event, FillEvent):
                    fill_price = event.fill_notional / event.quantity if event.quantity else 0.0
                    logger.info(
                        "[FILL] %s %.6f %s @ %.2f | commission=%.4f | id=%s",
                        event.direction.value, event.quantity, event.symbol,
                        fill_price, event.commission, event.order_id,
                    )
                    self.portfolio.update_fill(event)

                else:
                    raise TypeError(f"Unknown event type: {type(event).__name__}")

        clear_current_bar_timestamp()
        logger.info("Backtest complete.")
