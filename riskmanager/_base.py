"""RiskManager ABC + structural-typing Protocols for the dependencies it relies on."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from event import BarEvent, OrderEvent, OrderType, Direction


# ──────────────────────────────────────────────
# Portfolio dependency (structural typing)
# ──────────────────────────────────────────────

class _PortfolioLike(Protocol):
    """Subset of the Portfolio surface that RiskManager relies on."""

    positions: Dict[str, float]

    def get_price(self, symbol: str) -> Optional[float]: ...

    def calculate_balance(self) -> float: ...

    def submit_order(self, symbol: str, quantity: float, direction: Direction,
                     timestamp, order_type: OrderType,
                     price: Optional[float] = None) -> Optional[OrderEvent]: ...


# ──────────────────────────────────────────────
# DataHandler dependency (rolling-window history)
# ──────────────────────────────────────────────

class _DataHandlerLike(Protocol):
    """Subset of the DataHandler surface that VolTargetingRiskManager reads.

    Used to pull a trailing window of closes for the inline correlation-matrix
    derivation in ``calculate_instrument_weight`` (``mode='min_variance'``,
    ``corr_matrix=None``). ``timeframes`` is read at construction time to
    validate that the configured ``corr_timeframe`` is registered.
    """

    timeframes: Dict[str, int]

    def get_latest_bars(self, symbol: str, n: int = 1,
                        timeframe: Optional[str] = None) -> pd.DataFrame: ...


# ──────────────────────────────────────────────
# Strategy dependency (forecast oracle)
# ──────────────────────────────────────────────

class _StrategyLike(Protocol):
    """Subset of the Strategy surface that RiskManager reads from.

    The risk manager calls ``get_forecast(symbol)`` on every completed
    bar to derive the target position. Strategies no longer emit
    SignalEvents — they update an internal forecast cache that the risk
    manager reads here.

    ``symbol_list`` is read at risk-manager construction time to seed
    the equal-weight ``instrument_weight`` dict.

    ``is_warmed_up(symbol)`` is the strategy's measured end-of-warmup
    signal (True once the first non-NaN forecast has been cached) —
    consumed by ``VolTargetingRiskManager.get_live_symbols`` as
    the strategy gate of the universe liveness check.
    """

    symbol_list: List[str]

    def get_forecast(self, symbol: str) -> Optional[float]: ...

    def is_warmed_up(self, symbol: str) -> bool: ...


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class RiskManager(ABC):
    """
    Abstract base class for forecast-aware position sizing.

    On every completed bar the engine calls ``update_bar`` with the
    ``BarEvent``. Implementations read the forecast cache from their
    bound strategy via ``strategy.get_forecast(symbol)``, derive the
    target position, and submit an order to bring the realized position
    to that target. Forming bars are skipped (gate on
    ``event.is_forming`` inside ``update_bar``) to avoid intra-period
    resize thrash.

    The risk manager is the sole position-sizing authority — it owns
    every order submitted to the portfolio.

    Subclasses implement two abstract hooks:

    * ``update_bar(event)`` — the engine entry point.
    * ``_compute_target_qty(event)`` — the target-derivation pipeline
      (pure math; no side effects). Returns a diagnostic-rich dict
      that ``update_bar`` splices into its per-bar row before handling
      submit / dead-band / at-target.

    The base owns the per-symbol diagnostic buffer (``self._records``),
    a default ``_record_row`` appender, and ``get_records`` —
    subclasses may override ``_record_row`` (calling
    ``super()._record_row(...)`` first) to add side effects such as a
    DEBUG log line.
    """

    def __init__(self, portfolio: _PortfolioLike, strategy: _StrategyLike):
        """Bind dependencies and initialise the diagnostic-row buffer.

        Parameters
        ----------
        portfolio
            Portfolio surface (positions, balance, submit_order).
        strategy
            Strategy exposing ``get_forecast(symbol)`` and
            ``symbol_list``.
        """
        self.portfolio = portfolio
        self.strategy = strategy
        # Per-symbol diagnostic buffer; subclasses populate via _record_row.
        self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @abstractmethod
    def update_bar(self, event: BarEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def _compute_target_qty(self, event: BarEvent) -> Dict[str, Any]:
        """Derive the target position quantity for ``event.symbol``.

        Owns the entire target-derivation pipeline — fetching sigma /
        price / forecast / weights, building any intermediate values,
        and applying the target formula. Owns *target-derivation* skip
        reasons (e.g. the ``'warmup_*'`` family / ``'zero_vol'`` /
        ``'zero_weight'`` for Carver; ``'no_price'`` / ``'warmup_forecast'``
        for Simple). Has no side effects on the portfolio or the records
        buffer.

        Returns a dict that ``update_bar`` splices into the diagnostic
        row. Required keys (every subclass):

        * ``target_qty``: ``Optional[float]`` — ``None`` when skipped.
        * ``skip_reason``: ``Optional[str]`` — ``None`` on success.

        Subclass-specific intermediate keys (sigma, weights,
        annual_cash_target, price, ...) may also appear; ``update_bar``
        splices them in via ``row.update(...)``. Fields not yet
        computed when an early-exit fires should be present as
        ``None`` so the row schema stays uniform across branches.

        ``update_bar`` owns *post-target* concerns: ``current_qty``,
        ``trade_qty``, ``at_target`` / ``dead_band`` decisions, and
        the ``submit_order`` call.
        """
        raise NotImplementedError

    def _record_row(self, symbol: str, row: Dict[str, Any]) -> None:
        """Append one per-bar diagnostic row to the per-symbol buffer.

        Subclasses may override to add side effects (e.g. emit a
        DEBUG log line) by calling ``super()._record_row(...)`` first.
        """
        self._records[symbol].append(row)

    def get_records(self, symbol: str) -> pd.DataFrame:
        """Return recorded sizing rows for ``symbol`` as a DataFrame.

        Indexed by ``timestamp``. Empty DataFrame for unknown symbols
        or before the first completed bar.
        """
        rows = self._records.get(symbol)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df
