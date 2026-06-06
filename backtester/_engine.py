"""Backtester event-loop engine."""

import logging

from event import BarEvent, OrderEvent, FillEvent
from logging_setup import clear_current_bar_timestamp, set_current_bar_timestamp

logger = logging.getLogger(__name__)


class Backtester:
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.

    Bar-processing order on each ``BarEvent``:
        portfolio.update_bar  → execution.update_bar
                              → strategy.update_bar     (updates forecast cache)
                              → risk_manager.update_bar (reads strategy.get_forecast,
                                                         submits resize order)

    The risk manager runs *last* so it sees this bar's freshly-updated
    forecast. ``OrderEvent`` and ``FillEvent`` stages drain in subsequent
    iterations of the inner event loop.

    Callers wire each module explicitly and pass them in. See
    ``backtests/test_ewmac.py`` for a worked example.
    """
    def __init__(self, events_queue, data_handler, strategy, portfolio,
                 risk_manager, execution_handler):
        self.events = events_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler

    def run(self):
        """
        Execute the backtest event loop.

        ``DataHandler`` emits ``BarEvent``s. Each bar drives the four bar-
        consumers (portfolio, execution, strategy, risk_manager) in order;
        the risk manager may emit ``OrderEvent``s, which the execution
        handler consumes to produce ``FillEvent``s, which the portfolio
        applies.
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
