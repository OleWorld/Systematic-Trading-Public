"""CommissionModel — configurable per-fill commission for the backtest exchange.

The ``'rate'`` mode charges bps on the **true dollar notional**
``abs(quantity * point_value * fill_price)``, so it takes the instrument's
contract multiplier (``point_value``). The ``'per_contract'`` mode is, by
definition, independent of both price and multiplier.
"""

from dataclasses import dataclass

import math


@dataclass
class CommissionModel:
    """
    Configurable commission applied per fill.

    Modes:
        'per_contract' — fixed $ per contract: ``abs(quantity) * value``
                         (default — futures brokers charge per contract,
                         independent of price level and contract multiplier;
                         safe for negative/zero prices)
        'rate'         — fraction of notional: ``abs(quantity * point_value *
                         fill_price) * value`` (e.g., 0.0004 = 4bps taker fee —
                         the crypto/equity convention)
    """
    mode: str = 'per_contract'
    value: float = 0.0

    def __post_init__(self):
        if self.mode not in ('rate', 'per_contract'):
            raise ValueError(
                f"Unknown CommissionModel mode: '{self.mode}'. "
                "Must be 'rate' or 'per_contract'."
            )
        # Enforces the "always non-negative" contract in calculate()'s
        # docstring: a negative value would PAY the trader per fill.
        if not (math.isfinite(self.value) and self.value >= 0):
            raise ValueError(
                f"CommissionModel value must be finite and >= 0, got {self.value}"
            )

    def calculate(self, quantity: float, fill_price: float,
                  point_value: float = 1.0) -> float:
        """Commission in $ for a fill (always non-negative).

        ``point_value`` is the contract multiplier (dollars of PnL per 1.0
        price move per contract); it scales the ``'rate'`` notional and is
        ignored by ``'per_contract'``. Defaults to ``1.0`` so legacy two-arg
        calls are unchanged.
        """
        if self.mode == 'rate':
            return abs(quantity * point_value * fill_price) * self.value
        elif self.mode == 'per_contract':
            return abs(quantity) * self.value
        else:
            raise ValueError(
                f"Unknown CommissionModel mode: '{self.mode}'. "
                "Must be 'rate' or 'per_contract'."
            )
