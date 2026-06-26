"""SlippageModel — configurable fill-price slippage for the backtest exchange.

Slippage is applied in **price space** (it shifts the fill price), so it needs
no contract-multiplier (``point_value``) awareness: the multiplier enters later,
when the slipped price difference is converted to dollar PnL by the portfolio.
"""

from dataclasses import dataclass

from event import Direction


@dataclass
class SlippageModel:
    """
    Configurable slippage applied to fill prices.

    Modes:
        'pct'      — percentage of price (e.g., 0.001 = 0.1%)
        'absolute' — fixed value per unit (e.g., 0.50 tick size)
    """
    mode: str
    value: float

    def __post_init__(self):
        if self.mode not in ('pct', 'absolute'):
            raise ValueError(f"Unknown SlippageModel mode: '{self.mode}'. Must be 'pct' or 'absolute'.")

    def apply(self, price: float, direction: Direction) -> float:
        # ``abs(price)`` for the pct branch so slippage is always a positive
        # magnitude — at negative prices (e.g. WTI 2020) the raw product
        # ``price * value`` would push the fill in the wrong direction.
        if self.mode == 'pct':
            slip = abs(price) * self.value
        elif self.mode == 'absolute':
            slip = self.value
        else:
            raise ValueError(f"Unknown SlippageModel mode: '{self.mode}'. Must be 'pct' or 'absolute'.")

        if direction == Direction.BUY:
            return price + slip
        elif direction == Direction.SELL:
            return price - slip
        else:
            raise ValueError(f"Unexpected direction: {direction!r}")
