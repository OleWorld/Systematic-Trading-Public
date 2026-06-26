"""MarginModel — pluggable margin requirements for the backtest portfolio.

The portfolio consults a ``MarginModel`` everywhere it needs a margin number:

- **initial margin** — what a position locks; gates new orders via the
  available-balance check.
- **maintenance margin** — the equity floor that drives the margin-call
  liquidation trigger in ``check_solvency``.
- **max_abs_position** — the inverse (largest position a margin budget can
  carry) used when an order must be scaled down.

All three take the instrument's contract multiplier (``point_value``), so
margin is computed on the **true dollar notional** ``abs(qty * point_value *
price)``. Concentrating that arithmetic — and its ``price == 0`` edge — in one
place keeps the portfolio from ever special-casing a zero reference price.

Per-symbol margin is now expressed by giving each instrument its own model
instance in the ``InstrumentConfig`` registry (``config/_instrument.py``);
there is no longer a single model holding per-symbol dicts. The methods are
therefore symbol-agnostic — they take only ``(quantity, price, point_value)``.

``PortfolioMarginModel`` is the only concrete model: one universal
leverage/rate (the default; reproduces the old ``abs(qty*price) / leverage``
math exactly at ``point_value=1``). A per-contract *dollar* margin model
remains future work.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class MarginModel(ABC):
    """
    Abstract margin model consumed by ``BacktestPortfolio``.

    Symbol-agnostic: per-symbol margin is achieved by assigning a distinct
    model to each instrument's ``InstrumentConfig``. Each method takes the
    instrument's ``point_value`` (contract multiplier) so margin is computed
    on ``abs(quantity * point_value * price)``.
    """

    @abstractmethod
    def initial_margin(self, quantity: float, price: float,
                       point_value: float = 1.0) -> float:
        """Margin required to hold ``quantity`` at ``price`` (>= 0)."""
        raise NotImplementedError

    @abstractmethod
    def maintenance_margin(self, quantity: float, price: float,
                           point_value: float = 1.0) -> float:
        """Maintenance-margin floor for ``quantity`` at ``price`` (>= 0)."""
        raise NotImplementedError

    @abstractmethod
    def max_abs_position(self, margin_budget: float, price: float,
                         point_value: float = 1.0) -> float:
        """
        Largest ``abs(position)`` whose initial margin fits ``margin_budget``
        at ``price``. Returns ``inf`` when ``price == 0`` (a zero-priced
        position locks no margin), so callers never divide by zero.
        """
        raise NotImplementedError


@dataclass
class PortfolioMarginModel(MarginModel):
    """
    Universal-leverage cross-margin model: one ``initial_margin_rate`` and one
    ``maintenance_margin_rate`` applied to every position's notional.

    This is *not* SPAN/TIMS-style risk-netting "portfolio margin" — the name
    follows the project's convention, meaning a single account-wide leverage
    rather than per-symbol margins.

    Margin = ``abs(quantity * point_value * price) * rate``. With
    ``initial_margin_rate = 1 / leverage`` and ``point_value = 1`` this
    reproduces the legacy ``abs(quantity * price) / leverage`` formula exactly.
    ``abs`` keeps margin a positive magnitude for negative-priced instruments
    (e.g. WTI 2020).

    ``maintenance_margin_rate`` defaults to ``0.0``, which reproduces the
    legacy "liquidate only when account_balance < 0" behaviour (total
    maintenance margin is 0, so the trigger collapses to ``< 0``). Set it to a
    positive value below ``initial_margin_rate`` for a realistic earlier
    margin call.
    """

    initial_margin_rate: float
    maintenance_margin_rate: float = 0.0

    def __post_init__(self) -> None:
        # ``not (...)`` comparisons reject NaN (NaN fails every ordering, so
        # the guard fires) — matching the config / risk-manager validation style.
        if not (0.0 < self.initial_margin_rate <= 1.0):
            raise ValueError(
                f"initial_margin_rate must be in (0, 1], got "
                f"{self.initial_margin_rate}. (It is a fraction of notional; "
                f"1/leverage for leverage >= 1.)"
            )
        if not (0.0 <= self.maintenance_margin_rate <= self.initial_margin_rate):
            raise ValueError(
                f"maintenance_margin_rate must be in "
                f"[0, initial_margin_rate={self.initial_margin_rate}], got "
                f"{self.maintenance_margin_rate}. (Maintenance margin is the "
                f"lower liquidation floor; it cannot exceed the initial margin.)"
            )

    @classmethod
    def from_leverage(cls, leverage: float,
                      maintenance_margin_rate: float = 0.0) -> "PortfolioMarginModel":
        """
        Build from a leverage scalar: ``initial_margin_rate = 1 / leverage``.
        The backward-compatible bridge for callers that still pass ``leverage``.

        ``leverage`` must be ``>= 1``: ``1 / leverage`` is the fraction of
        notional posted as margin, so ``leverage < 1`` would require posting
        more than 100% of notional, which ``initial_margin_rate``'s ``<= 1`` cap
        disallows. Rejecting it here gives a leverage-worded error instead of
        leaking the derived-rate ``ValueError`` from ``__post_init__``.
        """
        if not (leverage >= 1.0):
            raise ValueError(
                f"leverage must be >= 1, got {leverage}. "
                f"(Margin = notional / leverage; leverage < 1 implies posting "
                f"more than 100% of notional, which this model's rate cap of "
                f"1.0 disallows.)"
            )
        return cls(initial_margin_rate=1.0 / leverage,
                   maintenance_margin_rate=maintenance_margin_rate)

    def initial_margin(self, quantity: float, price: float,
                       point_value: float = 1.0) -> float:
        return abs(quantity * point_value * price) * self.initial_margin_rate

    def maintenance_margin(self, quantity: float, price: float,
                           point_value: float = 1.0) -> float:
        return abs(quantity * point_value * price) * self.maintenance_margin_rate

    def max_abs_position(self, margin_budget: float, price: float,
                         point_value: float = 1.0) -> float:
        if price == 0:
            return float('inf')
        return margin_budget / (point_value * abs(price) * self.initial_margin_rate)
