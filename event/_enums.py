"""Enum types used across event dataclasses."""

from enum import Enum


class OrderType(Enum):
    """Order type: market or limit."""
    MKT = "MKT"
    LMT = "LMT"


class Direction(Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
