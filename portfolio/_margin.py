"""MarginModel — pluggable margin requirements for the backtest portfolio.

The portfolio consults a ``MarginModel`` everywhere it needs a margin number:

- **initial margin** — what a position locks; gates new orders via the
  available-balance check.
- **maintenance margin** — the equity floor that drives the margin-call
  liquidation trigger in ``check_solvency``.
- **max_abs_position** — the inverse (largest position a margin budget can
  carry) used when an order must be scaled down.

Concentrating all three in one object keeps the divide-by-price arithmetic —
and its ``price == 0`` edge — in a single place, so the portfolio never has to
special-case a zero reference price.

Two concrete models (this is a *pure* margin-formula axis — both stay
cross-margin; pooling/liquidation behaviour is unchanged):

- ``PortfolioMarginModel`` — one universal leverage/rate across every symbol
  (the default; reproduces the old ``abs(qty*price) / leverage`` math exactly).
- ``SingleMarginModel`` — per-symbol margin requirements (different margin per
  symbol). Scaffold/stub for now; methods raise ``NotImplementedError``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict


class MarginModel(ABC):
    """
    Abstract margin model consumed by ``BacktestPortfolio``.

    All three methods take ``symbol`` so a per-symbol model
    (``SingleMarginModel``) fits the same interface as the universal-rate
    default; the universal model simply ignores it.
    """

    @abstractmethod
    def initial_margin(self, symbol: str, quantity: float, price: float) -> float:
        """Margin required to hold ``quantity`` of ``symbol`` at ``price`` (>= 0)."""
        raise NotImplementedError

    @abstractmethod
    def maintenance_margin(self, symbol: str, quantity: float, price: float) -> float:
        """Maintenance-margin floor for ``quantity`` of ``symbol`` at ``price`` (>= 0)."""
        raise NotImplementedError

    @abstractmethod
    def max_abs_position(self, symbol: str, margin_budget: float,
                         price: float) -> float:
        """
        Largest ``abs(position)`` of ``symbol`` whose initial margin fits
        ``margin_budget`` at ``price``. Returns ``inf`` when ``price == 0``
        (a zero-priced position locks no margin), so callers never divide by
        zero.
        """
        raise NotImplementedError


@dataclass
class PortfolioMarginModel(MarginModel):
    """
    Universal-leverage cross-margin model: one ``initial_margin_rate`` and one
    ``maintenance_margin_rate`` applied to every symbol's notional.

    This is *not* SPAN/TIMS-style risk-netting "portfolio margin" — the name
    follows the project's ``margin_mode='portfolio_margin'`` convention,
    meaning a single account-wide leverage rather than per-symbol margins.

    Margin = ``abs(quantity * price) * rate``. With
    ``initial_margin_rate = 1 / leverage`` this reproduces the legacy
    ``abs(quantity * price) / leverage`` formula exactly. ``abs`` keeps margin
    a positive magnitude for negative-priced instruments (e.g. WTI 2020).

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
        """
        if not (leverage > 0.0):
            raise ValueError(f"leverage must be > 0, got {leverage}")
        return cls(initial_margin_rate=1.0 / leverage,
                   maintenance_margin_rate=maintenance_margin_rate)

    def initial_margin(self, symbol: str, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.initial_margin_rate

    def maintenance_margin(self, symbol: str, quantity: float, price: float) -> float:
        return abs(quantity * price) * self.maintenance_margin_rate

    def max_abs_position(self, symbol: str, margin_budget: float,
                         price: float) -> float:
        if price == 0:
            return float('inf')
        return margin_budget / (abs(price) * self.initial_margin_rate)


@dataclass
class SingleMarginModel(MarginModel):
    """
    Per-symbol margin model (``margin_mode='single_margin'``): different margin
    requirements for different symbols.

    SCAFFOLD / STUB — not yet wired. The intended shape carries per-symbol
    rates, e.g. ``initial_margin_rates[symbol]`` / ``maintenance_margin_rates[symbol]``
    (or per-contract dollar margins), so each instrument can be margined on its
    own leverage. Mirrors the ``portfolio/_live.py`` placeholder convention:
    defined for the extension point, methods raise until implemented.
    """

    initial_margin_rates: Dict[str, float] = field(default_factory=dict)
    maintenance_margin_rates: Dict[str, float] = field(default_factory=dict)

    def initial_margin(self, symbol: str, quantity: float, price: float) -> float:
        raise NotImplementedError(
            "SingleMarginModel is not yet wired — future work."
        )

    def maintenance_margin(self, symbol: str, quantity: float, price: float) -> float:
        raise NotImplementedError(
            "SingleMarginModel is not yet wired — future work."
        )

    def max_abs_position(self, symbol: str, margin_budget: float,
                         price: float) -> float:
        raise NotImplementedError(
            "SingleMarginModel is not yet wired — future work."
        )
