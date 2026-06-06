"""event — Event dataclasses and enums for the bar/order/fill pipeline.

Strategies update an internal forecast cache (read by the risk manager
on each completed bar) instead of emitting discrete signal events; only
``BarEvent``/``OrderEvent``/``FillEvent`` flow through the event loop.

Submodules (internal):
    _enums   OrderType, Direction
    _events  Event ABC, BarEvent, OrderEvent, FillEvent
"""

from event._enums import OrderType, Direction
from event._events import Event, BarEvent, OrderEvent, FillEvent

__all__ = [
    "OrderType",
    "Direction",
    "Event",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
]
