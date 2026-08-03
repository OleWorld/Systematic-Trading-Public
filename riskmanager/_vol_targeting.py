"""VolTargetingRiskManager — forecast-aware Carver vol-targeting sizer.

Implements Carver's vol-targeting framework (Systematic Trading, Ch. 10)
in **cash-vol** form. The risk target is a dollar amount of vol per
period; the instrument's dollar vol per period divides that target to
give the size:

    # vol_target_mode='dollar_volatility' (default — institutional futures
    # convention: a fixed annual $ vol budget, like a drawdown limit that
    # resets yearly instead of compounding with the account):
    annual_cash_target = IDM × instrument_weight
                                × annual_target_vol × (forecast / TARGET_AVG_ABS_FORECAST)

    # vol_target_mode='percent_volatility' (Carver's original form — the
    # vol budget is a fraction of *current* account equity, so position
    # sizes compound as the account grows/shrinks):
    annual_cash_target = capital × IDM × instrument_weight
                                × annual_target_vol × (forecast / TARGET_AVG_ABS_FORECAST)

    daily_cash_target  = annual_cash_target / sqrt(days_per_year)
    target_qty         = daily_cash_target / daily_price_vol
                       = annual_cash_target / annual_price_vol

where:
    capital                = portfolio.calculate_balance()              (account equity)
    IDM                    = instrument diversification multiplier      (constructor;
                             auto-updated from each CorrelationEvent)
    instrument_weight      = per-symbol capital weight                  (self.instrument_weight)
                             (strategy-budgeted sum-of-books when an
                             orchestrator supplies budget groups)
    annual_target_vol  = annualized vol target                      (constructor; REQUIRED —
                             $ amount in dollar mode, e.g. 250_000;
                             fraction of equity in percent mode, e.g. 0.25 = 25 %)
    annual_price_vol   = annualized stdev of price changes ($-units)  (VolEstimator)
    forecast               = strategy.get_forecast(symbol) ∈ [-FORECAST_CAP, +FORECAST_CAP]

The two equalities for ``target_qty`` are algebraically equivalent: the
``sqrt(days_per_year)`` factors in the daily-cash and daily-price-vol
forms cancel, so we implement the cleaner annualized form (no need to
plumb ``days_per_year`` / days_convention into the risk manager). The
daily-cash intermediate is preserved here for readers — it's the natural
mental model when running on daily bars.

Working in price (cash) units instead of percentage units generalizes
cleanly to instruments where percent change is undefined or meaningless
— futures spreads (price can cross zero), instruments quoted in
basis-point terms, synthetic legs. For positive-price single
instruments the result is identical to the old
``annual_target_vol / σ_pct`` form (``σ_$ = price × σ_pct``).

``TARGET_AVG_ABS_FORECAST`` and ``FORECAST_CAP`` are project-wide
constants on ``Strategy`` (default ``50.0`` and ``100.0``). The
``forecast / TARGET_AVG_ABS_FORECAST`` factor rescales from the project's
``±FORECAST_CAP`` forecast convention to Carver's ``±20`` (Carver divides
by 10; our 5×-larger scale gets a 5×-larger denominator). At
``|forecast| = TARGET_AVG_ABS_FORECAST`` the factor is 1.0 — exactly
Carver's vol-target notional. At ``|forecast| = FORECAST_CAP`` (= 2 ×
target by design) the factor is 2.0, doubling the size. At
``forecast = 0`` the target is zero (flat).

**Scope**: this module owns *allocation* (instrument weights + IDM) and
*sizing* (forecast → order quantity). It does NOT own the tradable
universe or correlation estimation — those live in ``universe/`` and
``correlation/`` and reach the risk manager as events:

* ``on_correlation_event(event)`` — the sole weight-recalc entry point.
  A ``CorrelationEvent`` carries a ready-made ρ (or a degeneracy reason)
  and the live-symbol snapshot; the manager turns it into
  ``self.instrument_weight`` per ``instrument_weight_mode``
  (``'equal_weight'`` / ``'min_variance'`` / ``'risk_parity'``, budget-
  grouped when an orchestrator supplies budget groups) and updates
  ``self.idm`` from the same matrix (capped at ``idm_cap``). No ρ is
  derived here, no data handler is held, and there is no cadence state —
  the ``CorrelationManager`` owns all three. Weights start **empty** and
  populate on the first event.
* ``on_universe_event(event)`` — reacts only to a not-live edge
  (``prev_live=True`` -> ``live=False``): flattens any held/pending
  position with a ``fill_on_next_bar=True`` MKT order (the flatten can
  never fill retroactively against the bar that produced the event), and
  pops the symbol out of ``self.instrument_weight``, proportionally
  rescaling the survivors and recomputing ``self.idm`` from the cached
  matrix's principal submatrix over the survivors (when coherent — see
  ``_rescale_weights_after_pop``). Live edges are no-ops: the symbol
  simply waits for the next correlation refresh to earn a weight.

Liveness is read from the injected ``UniverseManager``:
``universe_manager.status(symbol)`` gives ``live`` plus the
canonically-ordered ``reasons`` list. The sizing rule is **universal**:
a not-live symbol's target is 0, evaluated AHEAD of the sigma ladder so
a dead sigma can never block the exit. A held position is flattened via
the normal submit path; a flat one records the symbol's *primary*
recorded reason (``'warmup_forecast'`` / ``'warmup_history'`` /
``'constant_price'`` / ``'delisted'`` / …) as ``skip_reason``. This
applies to every not-live cause — there is no hold-and-warn exception
and no per-reason flatten policy.

Strategy weighting is **not** a risk-manager concern: with multiple
strategies an ``orchestrator.Orchestrator`` owns the per-strategy
weights and bakes them into the single combined forecast it hands the
risk manager (the ``strategy`` parameter accepts a single ``Strategy``
or an ``Orchestrator`` interchangeably). Its optional
``get_budget_groups()`` additionally makes the weight build
strategy-budgeted (sum-of-books; see ``_grouped_weights``).

On every completed bar the manager:
1. Updates the vol estimator.
2. Computes the target quantity per the formula above.
3. Submits a MKT order for ``target_qty - (current_qty +
   pending_mkt_order_quantity)`` — the diff against the *projected*
   position (realized + in-flight MKT orders, e.g. a same-bar
   margin-call liquidation), so an order already on its way to fill is
   never double-traded — if the diff is above the configured dead-band
   (``position_buffer``, Carver §10.7).

Skips on warmup (``sigma is None``), zero/negligible vol (``sigma <
_MIN_SIGMA_REL × |close|`` — guards the EWMA estimator's asymptotic
decay on a flat symbol), a live-but-unweighted symbol
(``'waiting_weight_recalc'`` — awaiting the next correlation refresh),
or forming bars. A zero combined weight is NOT a skip — it is a target
of 0, so a held position is flattened (a flat one is simply labelled
``'zero_weight'``). Any skip that would strand a *held* position emits a
WARNING. Idempotent: a stable forecast on consecutive bars produces no
further orders once the position matches the target.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid a config<->riskmanager import cycle at module load
    from config import InstrumentConfig

from analytics import (
    diversification_multiplier, equal_weight, min_variance, risk_parity,
)
from event import BarEvent, CorrelationEvent, OrderType, Direction, UniverseEvent
from riskmanager._base import (
    RiskManager, _OrchestratorLike, _PortfolioLike, _StrategyLike,
    _UniverseManagerLike,
)
from strategy import Strategy
from volatility import VolEstimator

logger = logging.getLogger(__name__)

# Relative negligible-volatility floor: an annualized sigma below this
# fraction of the bar's ``|close|`` is treated as zero vol (``'zero_vol'``
# skip). Guards the EWMA estimator's geometric decay, which approaches —
# but never reaches — exact 0.0 on a flat-after-activity symbol; without
# a floor the near-zero divisor would explode the target to the
# margin-capped maximum. 1e-6 (annualized vol of 0.0001 % of price) is
# orders of magnitude below any tradable instrument's vol, so real-data
# sizing is unaffected. At ``close == 0`` the relative term collapses to
# the exact-zero check.
_MIN_SIGMA_REL = 1e-6

# CorrelationEvent reasons whose payload carries no usable matrix but a
# usable live set: weights degrade to (budget-grouped) equal weight over
# ``event.live_symbols`` — the ρ=1 degenerate case — and the IDM is left
# untouched rather than being recomputed from a matrix that doesn't exist.
_DEGENERATE_CORR_REASONS = (
    'insufficient_observations',   # too few valid return observations
    'too_few_symbols',             # < 2 non-constant columns survived
    'nan_fallback',                # NaN in the estimated matrix (safety net)
)


class VolTargetingRiskManager(RiskManager):
    """Forecast-aware cash-vol-targeting sizer (Carver's framework).

    Owns one weight dict:

    - ``instrument_weight``: per-symbol capital weight. Starts **empty**
      and is rebuilt on every ``CorrelationEvent`` (see
      ``on_correlation_event``) per ``instrument_weight_mode``
      (``'equal_weight'`` — 1/N — or the ``analytics`` optimizers
      ``'min_variance'`` / ``'risk_parity'`` run on the event's matrix),
      budget-grouped when the forecast source exposes
      ``get_budget_groups()``. The dict may also be overwritten directly
      (research hook) — meaningful for **live symbols only**, since a
      not-live symbol is flattened by the universal rule regardless of
      the weight it carries.

    ``self.idm`` is auto-updated from each ``'ok'`` event's matrix
    (``analytics.diversification_multiplier``, clamped to ``idm_cap``),
    so weights and IDM always describe the same book; the last ``'ok'``
    matrix is cached as ``self._corr_matrix_cache``.

    Strategy weighting is **not** a risk-manager concern — an
    ``orchestrator.Orchestrator`` (passed in the ``strategy`` slot in
    place of a single ``Strategy``) owns the per-strategy weights and
    bakes them into the combined forecast.

    Universe state is read from the injected ``UniverseManager``; the
    risk manager never mutates it. A not-live symbol targets 0 ahead of
    the sigma ladder (universal not-live rule — see the module
    docstring).

    Per-bar diagnostic log analogous to ``Strategy.get_records``: every
    completed bar appends one row to ``self._records[symbol]`` and emits
    a DEBUG log line. Columns capture all sizing inputs and
    intermediates (``forecast``, ``sigma``, ``instrument_weight``,
    ``capital``, ``idm``, ``annual_target_vol``, ``vol_target_mode``,
    ``position_buffer``, ``annual_cash_target``, ``target_qty``,
    ``current_qty`` (realized position), ``pending_mkt_order_quantity``
    (signed sum of in-flight MKT orders — the resize diff targets
    ``current_qty + pending_mkt_order_quantity``, the projected
    position), ``trade_qty``, ``buffer_threshold``) plus ``submitted``
    (bool) and ``skip_reason`` — ``None`` when an order was submitted;
    the RM-owned labels ``'warmup_volatility'`` (sigma not ready),
    ``'zero_vol'``, ``'waiting_weight_recalc'`` (live but absent from
    ``instrument_weight``, i.e. awaiting the next correlation refresh),
    ``'warmup_forecast'`` (the post-weighting None-forecast backstop),
    ``'zero_weight'`` (a zero weight with the position *already flat* —
    a *held* position at zero weight is flattened instead, recording
    ``submitted=True`` / ``skip_reason=None``), ``'dead_band'``, and
    ``'at_target'``; or, for a not-live symbol that is already flat, the
    symbol's **primary universe reason itself** (the first entry of
    ``UniverseStatus.reasons``: ``'warmup_forecast'`` /
    ``'warmup_history'`` / an exclusion mark such as
    ``'constant_price'`` / ``'delisted'``). A skip that strands a held
    position additionally emits a WARNING. Symbols absent from
    ``instrument_weight`` leave the row's ``instrument_weight`` ``None``.
    Universe history is NOT duplicated here — join
    ``universe_manager.get_transition_log()`` on timestamp for it.
    Read via ``risk_manager.get_records(symbol)``.
    """

    def __init__(
        self,
        portfolio: _PortfolioLike,
        strategy: Union[_StrategyLike, _OrchestratorLike],
        vol_estimator: VolEstimator,
        universe_manager: _UniverseManagerLike,
        instruments: Optional[Dict[str, "InstrumentConfig"]] = None,
        idm: float = 1.0,
        idm_cap: Optional[float] = 2.5,
        annual_target_vol: Optional[float] = None,
        vol_target_mode: str = 'dollar_volatility',
        position_buffer: float = 0.25,
        instrument_weight_mode: str = 'equal_weight',
    ):
        """
        Parameters
        ----------
        portfolio
            Portfolio surface (positions, balance, submit_order).
        strategy
            Forecast source — a single ``Strategy`` or a multi-strategy
            ``orchestrator.Orchestrator`` (both expose
            ``get_forecast(symbol)``, ``symbol_list`` and
            ``is_warmed_up(symbol)``). Read on every completed bar
            (forecast) and at construction (symbol_list, for the default
            instruments registry). An orchestrator's optional
            ``get_budget_groups()`` additionally makes the weight build
            strategy-budgeted.
        vol_estimator
            ``VolEstimator`` providing ``get_annual_vol(symbol)`` in
            price (cash) units. Updated by ``update_bar`` on every
            completed bar.
        universe_manager
            ``universe.UniverseManager`` — the single source of truth for
            symbol liveness. Read-only from here:
            ``status(symbol).live`` drives the universal not-live rule
            and ``status(symbol).reasons[0]`` labels the skip. Exclusion
            marks are pushed by policy sources (e.g. the correlation
            manager), never by the risk manager.
        instruments
            Per-symbol ``InstrumentConfig`` registry
            (``Dict[str, InstrumentConfig]``). Two fields are read during
            sizing: ``point_value`` (the sizing divisor — ``target_qty =
            annual_cash_target / (point_value * sigma)``) and ``fractional``
            (whole-lot rounding of the target for futures). Default ``None``
            builds a uniform ``point_value=1`` / ``fractional=True`` registry
            over ``strategy.symbol_list`` — the identity reproducing the
            simplified crypto sizing. Pass the SAME registry handed to the
            portfolio and execution handler for a futures book.
        idm
            Instrument diversification multiplier (Carver Ch. 8).
            Default ``1.0``. Must be ``> 0``. Auto-updated from every
            ``CorrelationEvent`` carrying a matrix (reason ``'ok'``),
            clamped to ``idm_cap`` when the cap is enabled; the
            constructor default applies only until the first such event.
            Must not exceed ``idm_cap`` when the cap is enabled.
        idm_cap
            Upper bound applied to ``self.idm`` whenever it is
            auto-updated from a correlation event. Default ``2.5``
            (Carver's recommended maximum): the IDM multiplies every
            position linearly, so correlation-estimation noise must not
            translate into unbounded leverage. ``None`` disables the cap.
            Must be ``>= 1.0`` when not ``None`` (the DM is
            mathematically ``>= 1`` for a fully-allocated long-only
            weight vector). Direct assignments to ``self.idm`` by
            subclasses or downstream code are NOT capped — the same
            owner-may-overwrite convention as the weight dict.
        annual_target_vol
            Annualized volatility target (Carver's ``τ``). REQUIRED —
            no default; its units depend on ``vol_target_mode``:
            a dollar amount (must be ``> 0``, e.g. ``250_000`` = $250k
            of annual vol) under ``'dollar_volatility'``, or a fraction
            of current account equity (must be in ``(0, 1)``, e.g.
            ``0.25`` = 25 %) under ``'percent_volatility'``.
        vol_target_mode
            How ``annual_target_vol`` is interpreted. One of:

            * ``'dollar_volatility'`` (default) — fixed annual dollar
              vol budget; the cash target does NOT scale with account
              equity (institutional futures convention: the risk/
              drawdown limit is a dollar number reset periodically,
              not a compounding fraction).
            * ``'percent_volatility'`` — Carver's original form; the
              cash target is ``capital × τ`` re-read from the portfolio
              every bar, so sizes compound as the account grows/shrinks.
        position_buffer
            Carver §10.7 dead-band: skip the order if
            ``|trade_qty| <= position_buffer * |target_qty|``. Default
            ``0.25`` (ignore rebalances smaller than 25 % of the target
            position; reduces overtrading on small vol/price flickers).
            Set to ``0.0`` to trade every gap. Must be in ``[0, 1)``.
        instrument_weight_mode
            Weighting scheme applied to every incoming
            ``CorrelationEvent``. One of ``'equal_weight'`` (default),
            ``'min_variance'``, or ``'risk_parity'``. Both corr-based
            schemes run correlation-only — the equal-vol convention,
            exactly equivalent to optimizing the covariance under equal
            per-instrument vols and the right assumption here since
            sizing already divides by each instrument's σ. Under
            ``'equal_weight'`` the weights are 1/N but the event's matrix
            still updates the IDM, so a large decorrelated book earns its
            leverage instead of freezing at the constructor ``idm``.

        Raises
        ------
        ValueError
            On invalid constructor parameters.
        """
        # ``not (>)`` instead of ``<=`` so NaN is rejected too (mirrors
        # the idm_cap check below).
        if not (idm > 0):
            raise ValueError(f"idm must be > 0, got {idm}")
        if idm_cap is not None:
            # ``not (>=)`` instead of ``<`` so NaN is rejected too —
            # min(idm, nan) would silently never cap.
            if not (idm_cap >= 1.0):
                raise ValueError(
                    f"idm_cap must be >= 1.0 or None to disable, got "
                    f"{idm_cap}. (DM = 1/sqrt(w'rho w) >= 1 for sum-to-1 "
                    f"non-negative weights, so a sub-1 cap would always bind.)"
                )
            if idm > idm_cap:
                raise ValueError(
                    f"idm ({idm}) exceeds idm_cap ({idm_cap}); pass a "
                    f"smaller starting idm or raise/disable the cap "
                    f"(idm_cap=None)."
                )
        if vol_target_mode not in ('dollar_volatility', 'percent_volatility'):
            raise ValueError(
                f"Unknown vol_target_mode: {vol_target_mode!r}. "
                "Must be 'dollar_volatility' or 'percent_volatility'."
            )
        if annual_target_vol is None:
            raise ValueError(
                "annual_target_vol must be supplied explicitly (no "
                "default): a dollar amount under 'dollar_volatility' or "
                "a fraction in (0, 1) under 'percent_volatility'."
            )
        if vol_target_mode == 'percent_volatility':
            if not (0 < annual_target_vol < 1):
                raise ValueError(
                    f"annual_target_vol must be in (0, 1) under "
                    f"'percent_volatility', got {annual_target_vol}"
                )
        elif vol_target_mode == 'dollar_volatility':
            if annual_target_vol <= 0:
                raise ValueError(
                    f"annual_target_vol must be > 0 under "
                    f"'dollar_volatility', got {annual_target_vol}"
                )
        else:
            raise ValueError(
                f"Unexpected vol_target_mode: {vol_target_mode!r}"
            )
        if not (0.0 <= position_buffer < 1.0):
            raise ValueError(
                f"position_buffer must be in [0, 1), got {position_buffer}"
            )
        if instrument_weight_mode not in ('equal_weight', 'min_variance',
                                          'risk_parity'):
            raise ValueError(
                f"Unknown instrument_weight_mode: {instrument_weight_mode!r}. "
                "Must be 'equal_weight', 'min_variance', or 'risk_parity'."
            )
        super().__init__(portfolio, strategy)
        self.vol_estimator = vol_estimator
        self.universe_manager = universe_manager
        # Per-symbol point_value (sizing divisor) and fractional flag (whole-lot
        # rounding). Defaults to a uniform pv=1 / fractional=True registry over
        # the strategy's universe — the identity that reproduces the simplified
        # crypto sizing. Pass the SAME registry given to the portfolio/execution
        # for futures (point_value != 1, fractional=False).
        if instruments is None:
            from config import uniform_registry  # lazy: avoid import cycle
            instruments = uniform_registry(list(strategy.symbol_list))
        self.instruments = instruments
        self.idm = idm
        self.idm_cap = idm_cap
        # Narrowed to float by the None-rejection above.
        self.annual_target_vol: float = annual_target_vol
        self.vol_target_mode = vol_target_mode
        self.position_buffer = position_buffer
        self.instrument_weight_mode = instrument_weight_mode
        # Per-symbol capital weights. Start EMPTY — populated by the first
        # CorrelationEvent (there is no construction-time recalc: at
        # construction no bars have streamed, so the universe is empty
        # anyway). May also be overwritten directly (research hook).
        self.instrument_weight: Dict[str, float] = {}
        # Last 'ok' matrix, cached for the exclusion-pop IDM recompute.
        self._corr_matrix_cache: Optional[pd.DataFrame] = None

    # ── Event handlers (engine-dispatched, synchronous) ──────────────

    def on_correlation_event(self, event: CorrelationEvent) -> None:
        """Rebuild ``instrument_weight`` (and the IDM) from a refresh.

        The sole weight-recalc entry point. Dispatches on
        ``event.reason``:

        * ``'ok'`` — budget-grouped weights over ``event.matrix.index``
          per ``instrument_weight_mode``, then the IDM from the same
          matrix (capped at ``idm_cap``); the matrix is cached for the
          exclusion-pop recompute.
        * ``'insufficient_observations'`` / ``'too_few_symbols'`` /
          ``'nan_fallback'`` — no usable matrix: budget-grouped **equal
          weight** over ``event.live_symbols`` (the ρ=1 degenerate case),
          WARNING naming the reason; IDM left as-is EXCEPT the
          singleton-live sub-case, which resets it to ``1.0`` (same rule
          as ``'singleton'`` below).
        * ``'empty_universe'`` — clear the weights (INFO); IDM untouched.
        * ``'singleton'`` — ``{symbol: 1.0}`` with ``idm = 1.0``
          (precedence over budget grouping: the live-subset
          renormalization convention gives the sole live symbol full
          weight whatever its group's budget).

        Raises ``ValueError`` on an unknown reason (house enum-dispatch
        rule — a new reason must be handled explicitly, never fall
        through to the wrong branch).
        """
        live = list(event.live_symbols)
        if event.reason == 'ok':
            self.instrument_weight = self._grouped_weights(
                self.instrument_weight_mode, list(event.matrix.index),
                event.matrix,
            )
            self._update_idm_from_corr(event.matrix)
            self._corr_matrix_cache = event.matrix
        elif event.reason in _DEGENERATE_CORR_REASONS:
            logger.warning(
                "correlation refresh degenerate (%s): equal-weight fallback "
                "over %d live symbols; IDM left as-is (except the "
                "singleton-live case below, which resets it to 1.0)",
                event.reason, len(live),
            )
            if not live:
                self.instrument_weight = {}
            elif len(live) == 1:
                self.instrument_weight = {live[0]: 1.0}
                self.idm = 1.0
            else:
                self.instrument_weight = self._grouped_weights(
                    self.instrument_weight_mode, live, None)
        elif event.reason == 'empty_universe':
            logger.info("correlation refresh: empty universe; "
                        "instrument_weight cleared")
            self.instrument_weight = {}
        elif event.reason == 'singleton':
            if live:
                self.instrument_weight = {live[0]: 1.0}
                self.idm = 1.0
            else:
                # Sole candidate was a marked constant — nothing weightable.
                self.instrument_weight = {}
        else:
            raise ValueError(
                f"Unexpected CorrelationEvent.reason: {event.reason!r}"
            )

    def on_universe_event(self, event: UniverseEvent) -> None:
        """Not-live edge: same-bar flatten + weight pop/rescale (spec §7.2).

        Only a not-live EDGE (``event.prev_live is True`` and
        ``event.live is False``) triggers anything; every other
        transition (already not-live, a live edge, a reason-only churn
        while staying live/not-live) is a no-op — the symbol either was
        already flattened/popped on its original not-live edge or is
        not affected.

        The flatten order carries ``fill_on_next_bar=True`` — it rests
        and fills on the symbol's next bar event, never retroactively
        against the bar that produced this event. ``update_bar``'s
        sizing re-asserts the flatten as an idempotent backstop (covers
        a margin-call cancel pass voiding this order). A flat/zero
        projected position submits nothing.

        Popping the symbol out of ``self.instrument_weight`` (when
        present) triggers ``_rescale_weights_after_pop``, which
        proportionally renormalizes the survivors and recomputes the
        IDM from the cached matrix's principal submatrix when coherent.
        Live edges are no-ops: the symbol simply picks up a weight at
        the next correlation refresh.
        """
        if not (event.prev_live and not event.live):
            return
        symbol = event.symbol
        projected = self.portfolio.projected_position(symbol)
        if projected != 0:
            logger.warning(
                "%s went not-live (%s) holding %.6f projected — flattening "
                "on this bar (fills on the symbol's next bar event)",
                symbol, event.reasons, projected,
            )
            direction = Direction.SELL if projected > 0 else Direction.BUY
            self.portfolio.submit_order(
                symbol=symbol, quantity=abs(projected), direction=direction,
                timestamp=event.timestamp, order_type=OrderType.MKT,
                fill_on_next_bar=True,
            )
        if symbol in self.instrument_weight:
            self.instrument_weight.pop(symbol)
            self._rescale_weights_after_pop()

    # ── Weight machinery ─────────────────────────────────────────────

    def _budget_groups(self) -> Dict[str, Tuple[float, List[str]]]:
        """Resolve the budget-group structure from the forecast source.

        ``strategy.get_budget_groups()`` when the source exposes it (an
        ``Orchestrator``); otherwise one implicit group over the whole
        universe — the bare-``Strategy`` case, under which the grouped
        weight math collapses to the ungrouped form exactly. Budget values
        are guarded (finite, >= 0): a bad value draws a WARNING and the
        budgets fall back to equal over all groups. Overall scale is NOT
        validated — the live-group renormalization in ``_grouped_weights``
        divides by the live-group budget sum, so a not-quite-sum-to-1
        overwrite degrades gracefully.
        """
        get_groups = getattr(self.strategy, 'get_budget_groups', None)
        if not callable(get_groups):
            return {'__all__': (1.0, list(self.strategy.symbol_list))}
        groups = get_groups()
        bad = sorted(
            label for label, (weight, _) in groups.items()
            if not np.isfinite(weight) or weight < 0
        )
        if bad:
            logger.warning(
                "get_budget_groups returned non-finite/negative budget(s) "
                "for %s; falling back to equal budgets over all %d groups",
                bad, len(groups),
            )
            m = len(groups)
            return {
                label: (1.0 / m, symbols)
                for label, (_, symbols) in groups.items()
            }
        return groups

    def _grouped_weights(
        self, mode: str, kept: List[str],
        corr_matrix: Optional[pd.DataFrame],
        dead_log_level: int = logging.INFO,
    ) -> Dict[str, float]:
        """Sum-of-books instrument weights over the ``kept`` labels.

        Per budget group (``_budget_groups``): intersect the group's
        declared universe with ``kept`` (order taken from ``kept``), run
        the within-group weight scheme over the members — ``mode``'s
        optimizer on the principal submatrix of ``corr_matrix``, or equal
        weight when ``mode='equal_weight'`` or ``corr_matrix is None``
        (the ρ=1 degenerate fallback) — scale by the group's budget
        renormalized over the groups that have members, and sum:
        ``w(s) = Σᵢ W'ᵢ·vᵢ(s)``. The result sums to 1 by construction and
        a symbol covered by several groups draws budget from each owner
        (sum-of-books; Carver's sub-system aggregation). Groups with no
        ``kept`` members drop out with their budget redistributed, logged
        at ``dead_log_level`` — INFO by default (expected warmup staging
        while symbols go live); callers may raise it for paths where a
        dead group would be unexpected.
        If every surviving group carries budget 0, equal budgets over the
        survivors apply (WARNING). A group covering all of ``kept``
        consumes ``corr_matrix`` as-is — single-group runs (bare
        ``Strategy`` / full overlap) stay byte-identical to the ungrouped
        path, including optimizer-validator behavior on the supplied
        matrix. With >= 2 surviving groups, one INFO line records each
        group's configured vs. renormalized budget share per recalc.
        Kept symbols covered by no group at all keep their seeded 0.0
        weight and draw a WARNING (only reachable from a malformed custom
        source — the real ``Orchestrator``'s union universe always
        covers ``kept``).
        """
        groups = self._budget_groups()
        members = {
            label: [s for s in kept if s in set(universe)]
            for label, (_, universe) in groups.items()
        }
        alive = {label: syms for label, syms in members.items() if syms}
        if not alive:
            # Defensive: only reachable if get_budget_groups universes fail
            # to cover strategy.symbol_list (a malformed custom source).
            logger.warning(
                "no budget group covers any of the %d weightable symbols; "
                "falling back to ungrouped equal weight",
                len(kept),
            )
            return equal_weight(kept)
        covered = {s for syms in members.values() for s in syms}
        uncovered = [s for s in kept if s not in covered]
        if uncovered:
            logger.warning(
                "%d weightable symbol(s) %s are covered by no budget group "
                "(malformed get_budget_groups universe?); they keep weight "
                "0.0 this recalc",
                len(uncovered), uncovered,
            )
        dead = sorted(set(groups) - set(alive))
        if dead:
            logger.log(
                dead_log_level,
                "budget group(s) %s have no weightable symbols this recalc; "
                "their budget is redistributed over the %d remaining "
                "group(s)",
                dead, len(alive),
            )
        total = sum(groups[label][0] for label in alive)
        if total <= 0:
            logger.warning(
                "all %d weightable budget group(s) carry zero budget; "
                "falling back to equal budgets across them",
                len(alive),
            )
            budgets = {label: 1.0 / len(alive) for label in alive}
        else:
            budgets = {label: groups[label][0] / total for label in alive}
        if len(alive) >= 2:
            logger.info(
                "budget groups (configured -> renormalized live share): %s",
                {label: (groups[label][0], round(budgets[label], 6))
                 for label in alive},
            )
        combined: Dict[str, float] = {s: 0.0 for s in kept}
        for label, group_syms in alive.items():
            if len(group_syms) == 1:
                within = {group_syms[0]: 1.0}
            elif mode == 'equal_weight' or corr_matrix is None:
                within = equal_weight(group_syms)
            else:
                # Full-cover group: pass the matrix through untouched so
                # the ungrouped contract (incl. validator behavior on
                # explicit matrices) is preserved byte-identically.
                sub = (corr_matrix if group_syms == kept
                       else corr_matrix.loc[group_syms, group_syms])
                if mode == 'min_variance':
                    within = min_variance(sub)
                elif mode == 'risk_parity':
                    within = risk_parity(sub)
                else:
                    raise ValueError(f"Unexpected mode: {mode!r}")
            for symbol, v in within.items():
                combined[symbol] += budgets[label] * v
        return combined

    def _update_idm_from_corr(self, corr_matrix: pd.DataFrame) -> None:
        """Auto-update ``self.idm`` from ``corr_matrix`` and the current
        ``self.instrument_weight``.

        Carver's ``DM = 1 / sqrt(wᵀρw)`` via
        ``analytics.diversification_multiplier``, clamped to ``idm_cap`` when
        the cap is enabled. Shared by every corr-consuming weight mode —
        ``'equal_weight'`` feeds the same ρ even though its weights are 1/N,
        so a diversified book still earns its leverage — keeping weights and
        IDM coherent. The cap is leverage policy: it applies regardless of
        which mode produced the weights.
        """
        idm = diversification_multiplier(self.instrument_weight, corr_matrix)
        if self.idm_cap is not None:
            idm = min(idm, self.idm_cap)
        self.idm = idm

    def _rescale_weights_after_pop(self) -> None:
        """Proportionally renormalize survivors to sum 1 (simple rescale —
        the next scheduled refresh re-optimizes properly) and recompute the
        IDM from the cached matrix's principal submatrix when possible.

        Called immediately after ``on_universe_event`` pops a symbol out
        of ``self.instrument_weight``. If no survivors remain, this is a
        no-op (the dict is already empty). A single survivor gets weight
        1.0 and ``idm = 1.0`` (a lone instrument earns no diversification
        uplift — mirrors ``analytics.diversification_multiplier`` at
        N=1). Otherwise the IDM is recomputed from
        ``self._corr_matrix_cache``'s principal submatrix over the
        survivors — but ONLY when that cache exists AND every survivor is
        one of its labels; a survivor absent from the cache (e.g. it went
        live only after the last correlation refresh, or the cache is
        simply ``None`` because no refresh has landed yet) means the
        cached ρ can no longer describe the current book, so the IDM is
        left untouched rather than computed from a stale/incoherent
        matrix.
        """
        survivors = list(self.instrument_weight)
        if not survivors:
            return
        total = sum(self.instrument_weight.values())
        if total > 0:
            self.instrument_weight = {
                s: w / total for s, w in self.instrument_weight.items()
            }
        else:
            self.instrument_weight = {
                s: 1.0 / len(survivors) for s in survivors
            }
        if len(survivors) == 1:
            self.idm = 1.0
            return
        cache = self._corr_matrix_cache
        if cache is None or not set(survivors).issubset(set(cache.index)):
            return                     # no coherent rho — leave IDM as-is
        sub = cache.loc[survivors, survivors]
        self._update_idm_from_corr(sub)

    # ── Sizing ───────────────────────────────────────────────────────

    def update_bar(self, event: BarEvent) -> None:
        """Update sizing inputs and resize the position to the Carver target.

        Skips forming bars (one resize per completed bar). Delegates
        target-qty derivation (and *target-derivation* skip reasons —
        ``'warmup_volatility'`` / ``'zero_vol'`` /
        ``'waiting_weight_recalc'`` / ``'warmup_forecast'``) to
        ``_compute_target_qty``; owns *post-target* concerns
        (``'at_target'`` / ``'dead_band'`` / ``'zero_weight'`` and the
        not-live relabel / submit). A target-derivation skip that strands
        a *held* position emits a WARNING (the RM is never silently blind
        to an open position). A zero combined weight is not a skip here —
        it arrives as ``target_qty = 0`` and flattens a held position via
        the submit path; a **not-live** symbol arrives the same way
        (ahead of the sigma ladder) so a dead sigma cannot block the
        exit, and its flat rows are relabelled with its primary universe
        reason. Universe state is read fresh from the universe manager
        (the engine refreshes it earlier in this bar's chain). Records
        one diagnostic row per *completed* bar — including every
        early-exit branch — into ``self._records[symbol]`` via
        ``_record_row``, which also emits a DEBUG log line.
        """
        if event.is_forming:
            return

        symbol = event.symbol

        # Update vol estimator first so sigma reflects this bar.
        self.vol_estimator.update(event)

        # Universe state as of this bar: the engine has already run the
        # universe manager (and any correlation refresh) for this event,
        # so the status is fresh at the decision point.
        status = self.universe_manager.status(symbol)
        forecast = self.strategy.get_forecast(symbol)
        capital = self.portfolio.calculate_balance()
        current_qty = self.portfolio.positions.get(symbol, 0.0)
        # Signed sum of in-flight (pending) MKT orders — e.g. a same-bar
        # margin-call liquidation submitted by the portfolio earlier in
        # this bar's processing. The resize diff targets the projected
        # end-state ``current_qty + pending_mkt_order_quantity`` so an
        # order already on its way to fill is never double-traded. The
        # two components are recorded separately for debuggability.
        pending_mkt_order_quantity = (
            self.portfolio.projected_position(symbol) - current_qty
        )

        # Seed the diagnostic row with always-known inputs;
        # _compute_target_qty supplies sigma / weights /
        # annual_cash_target / target_qty / skip_reason via row.update.
        row: Dict[str, Any] = {
            'timestamp': event.timestamp,
            'symbol': symbol,
            'forecast': forecast,
            'sigma': None,
            'instrument_weight': None,
            'capital': capital,
            'idm': self.idm,
            'annual_target_vol': self.annual_target_vol,
            'vol_target_mode': self.vol_target_mode,
            'position_buffer': self.position_buffer,
            'annual_cash_target': None,
            'target_qty': None,
            'current_qty': current_qty,
            'pending_mkt_order_quantity': pending_mkt_order_quantity,
            'trade_qty': None,
            'buffer_threshold': None,
            'submitted': False,
            'skip_reason': None,
        }
        row.update(self._compute_target_qty(event))

        if row['skip_reason'] is not None:
            # A target-derivation skip (warmup_volatility / zero_vol /
            # waiting_weight_recalc / warmup_forecast) means we cannot
            # compute a well-defined target this bar. Harmless when flat, but
            # if we are HOLDING a position the risk manager is leaving it
            # unmanaged — surface it loudly so it can never go unnoticed.
            # (zero_weight and the not-live rule no longer land here: both
            # flow through as target_qty=0 and flatten via the submit path.)
            # A position with a liquidation already in flight projects to 0
            # (current + pending) — it is being managed, so no warning.
            if current_qty + pending_mkt_order_quantity != 0:
                logger.warning(
                    "%s: holding %s contracts (%s pending) but skipping "
                    "resize (%s) — position is unmanaged this bar",
                    symbol, current_qty, pending_mkt_order_quantity,
                    row['skip_reason'],
                )
            self._record_row(symbol, row)
            return

        target_qty = row['target_qty']
        # Whole-lot rounding for non-fractional instruments (futures), Carver
        # §10.7: round the continuous target to the nearest contract BEFORE the
        # diff and dead-band so the buffer and the traded size agree. A target
        # that rounds to 0 holds flat (one contract exceeds the vol budget); a
        # held position whose target rounds to 0 is flattened via the diff.
        if not self.instruments[symbol].fractional:
            target_qty = float(round(target_qty))
            row['target_qty'] = target_qty
        trade_qty = target_qty - (current_qty + pending_mkt_order_quantity)
        buffer_threshold = self.position_buffer * abs(target_qty)
        row['trade_qty'] = trade_qty
        row['buffer_threshold'] = buffer_threshold

        # Order matters: ``at_target`` (realized position essentially
        # equals target) is checked first so the diagnostic row carries
        # the more informative label. The dead-band check that follows
        # picks up small-but-nonzero diffs. ``target_qty == 0`` (forecast
        # is 0, a zero instrument weight, or a not-live symbol) lands here
        # when also flat; otherwise the dead-band collapses to zero and any
        # nonzero current position triggers a flatten via the submit path.
        if abs(trade_qty) < 1e-12:                # already at target
            # Relabel the flat case by its cause, most specific first: a
            # NOT-LIVE symbol's zero target is the universe exclusion (its
            # primary recorded reason is the honest label); a zero
            # instrument weight is more informative than the generic
            # 'at_target' (the position is flat *because* it carries no
            # weight). instrument_weight is a populated float in the
            # zero-weight case: a live symbol absent from
            # instrument_weight returned earlier as
            # 'waiting_weight_recalc'.
            if not status.live:
                row['skip_reason'] = (status.reasons[0] if status.reasons
                                      else 'at_target')
            elif row['instrument_weight'] == 0:
                row['skip_reason'] = 'zero_weight'
            else:
                row['skip_reason'] = 'at_target'
            self._record_row(symbol, row)
            return
        if target_qty != 0 and abs(trade_qty) <= buffer_threshold:
            row['skip_reason'] = 'dead_band'
            self._record_row(symbol, row)
            return

        row['submitted'] = True
        self._record_row(symbol, row)

        direction = Direction.BUY if trade_qty > 0 else Direction.SELL
        self.portfolio.submit_order(
            symbol=symbol, quantity=abs(trade_qty), direction=direction,
            timestamp=event.timestamp, order_type=OrderType.MKT,
        )

    def _compute_target_qty(self, event: BarEvent) -> Dict[str, Any]:
        """Carver cash-vol target-qty pipeline.

        target_qty = annual_cash_target / annual_price_vol, where
        annual_cash_target = IDM × instrument_weight
        × annual_target_vol × (forecast / TARGET_AVG_ABS_FORECAST),
        additionally scaled by ``capital`` (current account equity) under
        ``vol_target_mode='percent_volatility'`` — see the module
        docstring for the two forms.

        Owns the *target-derivation* skip ladder, in order:

        1. **Universal not-live rule** — a symbol the universe manager
           reports as not live returns ``target_qty = 0`` with no skip,
           AHEAD of the sigma checks: a flatten needs no sigma, and an
           excluded/delisted symbol's sigma is typically dead
           (``'zero_vol'``), which would otherwise strand the exit.
           ``update_bar`` relabels the already-flat case with the
           symbol's primary recorded universe reason.
        2. Sigma checks — ``'warmup_volatility'`` (not ready) /
           ``'zero_vol'``.
        3. Live but unweighted — ``'waiting_weight_recalc'``: gates pass
           but no weight has landed yet (the next ``CorrelationEvent``
           supplies one).
        4. None-forecast backstop — ``'warmup_forecast'``: a weighted
           symbol whose forecast cache is still empty (``get_forecast``
           → ``None``, e.g. after a direct ``instrument_weight``
           overwrite) skips here instead of feeding ``None`` into the
           formula.

        A zero instrument weight (``iw == 0``) is **not** a skip — it
        returns ``target_qty = 0`` (with ``skip_reason`` left ``None``)
        so a held position is flattened by ``update_bar``'s submit path
        and a flat one is relabelled ``'zero_weight'`` there. The
        returned dict is spliced into the diagnostic row by
        ``update_bar`` via ``row.update(...)``; intermediates computed
        before an early-exit fires are populated, those after remain
        ``None``, preserving the row schema across branches.
        """
        symbol = event.symbol
        out: Dict[str, Any] = {
            'target_qty': None, 'skip_reason': None,
            'sigma': None, 'instrument_weight': None,
            'annual_cash_target': None,
        }

        status = self.universe_manager.status(symbol)
        if not status.live:
            # UNIVERSAL NOT-LIVE RULE: target 0 ahead of the sigma ladder
            # (a dead sigma cannot block the exit). Held -> flatten via the
            # normal submit path; flat -> update_bar relabels with the
            # primary recorded reason. Applies to every not-live cause,
            # constant_price included.
            out['annual_cash_target'] = 0.0
            out['target_qty'] = 0.0
            return out

        sigma = self.vol_estimator.get_annual_vol(symbol)
        if sigma is None:
            out['skip_reason'] = 'warmup_volatility'
            return out
        out['sigma'] = sigma
        # Zero OR negligible vol: the relative term catches the EWMA
        # estimator's asymptotic decay on a flat symbol (never exactly 0)
        # before the divide below can explode the target.
        if sigma == 0 or sigma < _MIN_SIGMA_REL * abs(event.close):
            out['skip_reason'] = 'zero_vol'
            return out

        if symbol not in self.instrument_weight:
            # Live but unweighted: gates pass, no weight has landed yet.
            # Weights are rebuilt only on a CorrelationEvent, so a newly
            # live symbol waits for the next refresh. ``instrument_weight``
            # stays None in the diagnostic row — truthful, vs. recording a
            # synthetic 0.0. Note the trigger is weight-membership, not
            # liveness: a directly overwritten weight still sizes (the
            # documented research hook), guarded by the None-forecast
            # backstop below.
            out['skip_reason'] = 'waiting_weight_recalc'
            return out
        iw = self.instrument_weight[symbol]
        out['instrument_weight'] = iw
        if iw == 0:
            # A zero instrument weight means a target of 0 — NOT a reason
            # to skip. Flow through as target_qty=0 so a held position is
            # flattened by the normal submit path (exactly like
            # forecast=0), instead of being silently stranded. ``update_bar``
            # relabels the flat case as 'zero_weight' for diagnostics.
            out['annual_cash_target'] = 0.0
            out['target_qty'] = 0.0
            return out

        forecast = self.strategy.get_forecast(symbol)
        if forecast is None:
            # No forecast cached yet (warmup) despite the symbol carrying a
            # weight — reachable when ``instrument_weight`` is overwritten
            # directly (the research hook bypasses the liveness-driven
            # weighting). Skip before the formula, which would raise on None
            # (``None / TARGET_AVG_ABS_FORECAST`` is a TypeError). Mirrors
            # SimpleRiskManager's guard.
            out['skip_reason'] = 'warmup_forecast'
            return out
        if self.vol_target_mode == 'percent_volatility':
            # Carver's original form: τ is a fraction of *current*
            # account equity, so the cash target compounds with the
            # account.
            capital = self.portfolio.calculate_balance()
            annual_cash_target = (
                capital * self.idm * iw * self.annual_target_vol
                * (forecast / Strategy.TARGET_AVG_ABS_FORECAST)
            )
        elif self.vol_target_mode == 'dollar_volatility':
            # Fixed annual $ vol budget — no capital term (institutional
            # futures convention: the risk limit is a dollar number, not
            # a compounding fraction of equity).
            annual_cash_target = (
                self.idm * iw * self.annual_target_vol
                * (forecast / Strategy.TARGET_AVG_ABS_FORECAST)
            )
        else:
            raise ValueError(
                f"Unexpected vol_target_mode: {self.vol_target_mode!r}"
            )
        out['annual_cash_target'] = annual_cash_target
        # Divide by the per-contract dollar vol: point_value * sigma. sigma is
        # an annualized price-change stdev (price space); the contract
        # multiplier converts it to dollars of vol per contract.
        pv = self.instruments[symbol].point_value
        out['target_qty'] = annual_cash_target / (pv * sigma)
        return out

    def _record_row(self, symbol: str, row: Dict[str, Any]) -> None:
        """Append the diagnostic row and emit the Carver DEBUG log line."""
        super()._record_row(symbol, row)
        action = 'submit' if row['submitted'] else row['skip_reason']
        logger.debug(
            "[CARVER] %s fc=%s sigma=%s iw=%s cap=%.2f "
            "target=%s cur=%.6f pend=%s trade=%s action=%s",
            symbol, row['forecast'], row['sigma'],
            row['instrument_weight'],
            row['capital'], row['target_qty'], row['current_qty'],
            row['pending_mkt_order_quantity'], row['trade_qty'], action,
        )
