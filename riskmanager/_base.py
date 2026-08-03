"""RiskManager ABC + structural-typing Protocols for the dependencies it relies on."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import pandas as pd

from event import BarEvent, OrderEvent, OrderType, Direction


# ──────────────────────────────────────────────
# Portfolio dependency (structural typing)
# ──────────────────────────────────────────────

class _PortfolioLike(Protocol):
    """Subset of the Portfolio surface that RiskManager relies on.

    ``projected_position`` is the realized position plus the signed
    quantities of in-flight (pending) MKT orders — the baseline risk
    managers size their resize diff against, so an order already on its
    way to fill (e.g. a same-bar margin-call liquidation) is never
    double-traded.
    """

    positions: Dict[str, float]

    def get_price(self, symbol: str) -> Optional[float]: ...

    def calculate_balance(self) -> float: ...

    def projected_position(self, symbol: str) -> float: ...

    def submit_order(self, symbol: str, quantity: float, direction: Direction,
                     timestamp, order_type: OrderType,
                     price: Optional[float] = None,
                     is_liquidation: bool = False,
                     fill_on_next_bar: bool = False) -> Optional[OrderEvent]: ...


# ──────────────────────────────────────────────
# UniverseManager dependency (tradable-universe state)
# ──────────────────────────────────────────────

class _UniverseManagerLike(Protocol):
    """Read surface of ``universe.UniverseManager`` the RiskManager consumes.

    The universe manager is the single source of truth for symbol
    liveness; the risk manager only *reads* it (marks are pushed by
    policy sources such as the correlation manager, never by the RM).

    ``status(symbol)`` returns a defensive-copy ``UniverseStatus``
    (``live`` / ``excluded`` / canonically-ordered ``reasons``) — read on
    every sized bar to drive the universal not-live rule and to label the
    skip ladder with the symbol's primary recorded reason. Typed ``Any``
    so ``riskmanager`` need not import ``universe``.
    ``get_live_symbols()`` returns the live subset in
    ``strategy.symbol_list`` order — the introspection counterpart used
    by wiring/diagnostic code.
    """

    def status(self, symbol: str) -> Any: ...

    def get_live_symbols(self) -> List[str]: ...


# ──────────────────────────────────────────────
# Strategy dependency (forecast oracle)
# ──────────────────────────────────────────────

class _StrategyLike(Protocol):
    """Subset of the Strategy surface that RiskManager reads from.

    The risk manager calls ``get_forecast(symbol)`` on every completed
    bar to derive the target position. Strategies no longer emit
    SignalEvents — they update an internal forecast cache that the risk
    manager reads here.

    ``symbol_list`` is read at risk-manager construction time to build
    the default ``InstrumentConfig`` registry.

    ``is_warmed_up(symbol)`` is the strategy's measured end-of-warmup
    signal (True once the first non-NaN forecast has been cached) —
    consumed by ``universe.UniverseManager`` as the strategy gate of the
    liveness check (reason ``'warmup_forecast'``), not by the risk
    manager itself.
    """

    symbol_list: List[str]

    def get_forecast(self, symbol: str) -> Optional[float]: ...

    def is_warmed_up(self, symbol: str) -> bool: ...


# ──────────────────────────────────────────────
# Orchestrator dependency (multi-strategy forecast source)
# ──────────────────────────────────────────────

class _OrchestratorLike(Protocol):
    """Read surface of an ``orchestrator.Orchestrator`` the RiskManager reads.

    A superset of ``_StrategyLike``: an orchestrator combines several
    strategies' forecasts into one combined forecast per symbol and presents
    the same three forecast-source members the risk manager consumes, plus
    ``get_budget_groups`` — the budget-group structure
    ``{label: (budget_weight, universe)}`` that
    ``VolTargetingRiskManager.on_correlation_event`` uses to build
    strategy-budgeted instrument weights (sum-of-books). The risk manager
    detects the extra method via ``hasattr`` at recalc time, so a single
    strategy and a multi-strategy orchestrator remain interchangeable as
    forecast sources (its ``strategy`` parameter is typed
    ``Union[_StrategyLike, _OrchestratorLike]``); a bare ``Strategy``
    simply gets one implicit group.
    """

    symbol_list: List[str]

    def get_forecast(self, symbol: str) -> Optional[float]: ...

    def is_warmed_up(self, symbol: str) -> bool: ...

    def get_budget_groups(self) -> Dict[str, Tuple[float, List[str]]]: ...


# A forecast source is either a single strategy or a multi-strategy
# orchestrator — both expose ``symbol_list`` / ``get_forecast`` /
# ``is_warmed_up``.
_ForecastSourceLike = Union[_StrategyLike, _OrchestratorLike]


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

    def __init__(self, portfolio: _PortfolioLike,
                 strategy: _ForecastSourceLike):
        """Bind dependencies and initialise the diagnostic-row buffer.

        Parameters
        ----------
        portfolio
            Portfolio surface (positions, balance, submit_order).
        strategy
            Forecast source — either a single ``Strategy`` or a
            multi-strategy ``Orchestrator`` (both expose
            ``get_forecast(symbol)``, ``symbol_list`` and
            ``is_warmed_up(symbol)``). Stored as ``self.strategy``.
        """
        self.portfolio = portfolio
        self.strategy = strategy
        # Per-symbol diagnostic buffer; subclasses populate via _record_row.
        self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @abstractmethod
    def update_bar(self, event: BarEvent) -> None:
        raise NotImplementedError

    def on_universe_event(self, event) -> None:
        """React to a universe transition (engine-dispatched inline).

        Default: no-op. ``VolTargetingRiskManager`` overrides to flatten
        positions on not-live edges and re-normalize weights.
        """
        return None

    def on_correlation_event(self, event) -> None:
        """React to a correlation refresh (engine-dispatched inline).

        Default: no-op. ``VolTargetingRiskManager`` overrides to rebuild
        instrument weights and the IDM from the event payload.
        """
        return None

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
