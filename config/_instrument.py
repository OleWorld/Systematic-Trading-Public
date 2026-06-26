"""InstrumentConfig — per-symbol trading metadata for the backtest engine.

Where ``BacktestConfig`` holds *run-level* infrastructure (dates, capital,
vol-target knobs, fill timing), ``InstrumentConfig`` holds the *per-symbol*
economics the money path needs to trade real futures:

- ``point_value`` — the contract multiplier. A $1 move in the underlying
  changes PnL by ``point_value`` dollars per contract (WTI crude = 1000 bbl,
  so ``point_value = 1000``). Dollar value is ``qty * point_value * price``;
  ``point_value = 1`` reproduces the simplified crypto-perp accounting.
- ``fractional`` — whether fractional lots are allowed. ``True`` for crypto;
  ``False`` for futures, which trade in whole contracts only (the risk manager
  rounds to whole lots and the portfolio enforces it at submission).
- ``slippage`` / ``commission`` / ``margin`` — the per-symbol cost and margin
  models (each instrument can be margined and charged on its own terms).

The engine consumes a **registry** ``Dict[str, InstrumentConfig]``: the
portfolio, execution handler, and risk manager look up each symbol's config.
Use ``uniform_registry`` to build a homogeneous registry (e.g. the crypto
basket) in a single call.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from execution import SlippageModel, CommissionModel
from portfolio import MarginModel, PortfolioMarginModel


def _default_slippage() -> SlippageModel:
    # Futures-first default: absolute $-per-unit, zero cost.
    return SlippageModel(mode='absolute', value=0.0)


def _default_commission() -> CommissionModel:
    # Futures-first default: per-contract, zero cost.
    return CommissionModel(mode='per_contract', value=0.0)


def _default_margin() -> MarginModel:
    # leverage=1 → full-notional initial margin, no maintenance floor.
    return PortfolioMarginModel.from_leverage(1.0)


@dataclass
class InstrumentConfig:
    """
    Per-symbol trading metadata.

    Attributes
    ----------
    point_value
        Contract multiplier (dollars of PnL per 1.0 price move per contract).
        Must be ``> 0``. Default ``1.0`` (the simplified crypto convention).
    fractional
        ``True`` if fractional lots are allowed (crypto); ``False`` for
        whole-lot-only futures. Default ``True``.
    slippage
        Per-symbol ``SlippageModel`` (price-space). Default absolute/0.0.
    commission
        Per-symbol ``CommissionModel``. Default per_contract/0.0.
    margin
        Per-symbol ``MarginModel``. Default
        ``PortfolioMarginModel.from_leverage(1.0)``.

    Raises
    ------
    ValueError
        If ``point_value`` is not strictly positive (NaN rejected).
    """

    point_value: float = 1.0
    fractional: bool = True
    slippage: SlippageModel = field(default_factory=_default_slippage)
    commission: CommissionModel = field(default_factory=_default_commission)
    margin: MarginModel = field(default_factory=_default_margin)

    def __post_init__(self) -> None:
        # ``not (... > 0)`` rejects NaN too (NaN fails the comparison).
        if not (self.point_value > 0):
            raise ValueError(
                f"point_value must be > 0, got {self.point_value}. "
                "(It is the contract multiplier: dollars of PnL per 1.0 "
                "price move per contract.)"
            )


def uniform_registry(
    symbols: List[str],
    *,
    point_value: float = 1.0,
    fractional: bool = True,
    slippage: Optional[SlippageModel] = None,
    commission: Optional[CommissionModel] = None,
    margin: Optional[MarginModel] = None,
) -> Dict[str, InstrumentConfig]:
    """
    Build a registry that applies the same ``InstrumentConfig`` to every
    symbol — the one-liner for a homogeneous book (e.g. the crypto basket).

    Every symbol maps to a single shared ``InstrumentConfig`` instance (treat
    it as immutable). The cost/margin models are stateless, so sharing their
    references is safe. Pass distinct ``InstrumentConfig`` objects directly in
    a hand-built dict for a heterogeneous book.

    Parameters
    ----------
    symbols
        Non-empty list of unique symbol strings.
    point_value, fractional, slippage, commission, margin
        Forwarded to the shared ``InstrumentConfig``. ``None`` model arguments
        fall back to the ``InstrumentConfig`` defaults.

    Raises
    ------
    ValueError
        If ``symbols`` is empty or contains duplicates (or via
        ``InstrumentConfig`` validation).
    """
    if not symbols:
        raise ValueError("symbols must not be empty.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique (duplicate symbol found).")

    cfg = InstrumentConfig(
        point_value=point_value,
        fractional=fractional,
        slippage=slippage if slippage is not None else _default_slippage(),
        commission=commission if commission is not None else _default_commission(),
        margin=margin if margin is not None else _default_margin(),
    )
    return {s: cfg for s in symbols}
