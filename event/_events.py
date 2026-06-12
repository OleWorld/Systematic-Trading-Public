"""Event dataclasses flowing through the backtester event loop."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from event._enums import OrderType, Direction


@dataclass
class Event:
    """Base event class for isinstance dispatch in the event loop."""
    pass


@dataclass
class BarEvent(Event):
    """
    Market data event — emitted when a new bar (candle) is available.

    Attributes:
        period: Bar timeframe string, e.g. '1m', '1h', '4h', '1d'.
        is_forming: True for live bars still accumulating ticks.
            Forming bars flow through the full bar pipeline
            (portfolio -> execution -> strategy -> risk_manager) just like
            completed bars. The risk manager itself gates on
            ``is_forming`` to avoid intra-period resize thrash; strategies
            recompute forecasts from ``iloc[-2]``-based finalized values
            so forming-bar processing is idempotent.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    period: str = "1d"
    is_forming: bool = False


@dataclass
class OrderEvent(Event):
    """
    Order event — emitted by Portfolio to send an order to ExecutionHandler.

    Attributes:
        order_type: OrderType.MKT (market) or OrderType.LMT (limit).
        direction: Direction.BUY or Direction.SELL.
        order_id: Auto-generated UUID if not provided.
        price: Required for limit orders, ignored for market orders.
        is_liquidation: True for orders submitted by the portfolio's
            solvency-enforcement path. Liquidation orders are exempt from
            the FIFO cancel pass that fires when account_balance < 0, so
            they are not cancelled by the very mechanism that submitted them.
    """
    symbol: str
    order_type: OrderType
    quantity: float
    direction: Direction
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    is_liquidation: bool = False


@dataclass
class FillEvent(Event):
    """
    Fill event — emitted by ExecutionHandler when an order is filled.

    Attributes:
        direction: Direction.BUY or Direction.SELL.
        fill_notional: Total fill value (quantity * fill_price).
    """
    timestamp: datetime
    symbol: str
    exchange: str
    quantity: float
    direction: Direction
    fill_notional: float
    commission: float = 0.0
    order_id: Optional[str] = None
