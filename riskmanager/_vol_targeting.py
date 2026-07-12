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
    IDM                    = instrument diversification multiplier      (constructor)
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

Instrument weights are owned by the risk manager (no external allocator
dependency). ``self.instrument_weight: Dict[str, float]`` spreads
capital across symbols and defaults to equal-weight ``1/N`` across
``strategy.symbol_list``; it is populated at ``__init__`` by
``calculate_instrument_weight()``, which can be re-called later (or the
dict overwritten directly) to refresh weights — e.g. monthly rebalances,
correlation-driven weight schemes — without per-bar wiring. **Strategy
weighting is not a risk-manager concern**: with multiple strategies an
``orchestrator.Orchestrator`` owns the per-strategy weights and bakes
them into the single combined forecast it hands the risk manager (the
``strategy`` parameter accepts a single ``Strategy`` or an
``Orchestrator`` interchangeably). The risk manager's job is purely
forecast → order quantity.

**Explicit tradable universe**: the manager owns one ``UniverseStatus``
record per symbol (``self._universe``) — the source of truth behind
``get_live_symbols()`` and the sizing skip ladder. Each record carries
two derived axes plus a reason list:

* ``live`` — measured gate liveness: True iff no liveness reason is
  present. The gates: (1) the full ``corr_lookback`` bars at
  ``corr_timeframe`` (data gate — every live member contributes the
  complete correlation window, so the estimation window never shrinks;
  unmet → reason ``'warmup_correlation'``) and (2)
  ``strategy.is_warmed_up(symbol)`` (strategy gate — the measured flag
  the ``Strategy`` base sets when the first non-NaN forecast is cached;
  unmet → reason ``'warmup_forecast'``).
* ``excluded`` — policy layer: ``'constant_price'`` (a symbol whose
  closes were constant across the corr window at the latest recalc has
  no defined correlation, so it is excluded from that recalc's ρ and
  weights (WARNING) and re-enters automatically at a later recalc once
  it moves again — hold-and-warn: a held position is NOT flattened) and
  ``'removed'`` (manual, permanent — ``remove_symbol(symbol)`` is the
  delisting primitive: the weight is popped and a held position is
  flattened on the symbol's next bar, ahead of the sigma checks so a
  dead sigma cannot block the exit). Exclusion reasons are also
  liveness reasons: excluded ⇒ not live.
* ``reasons`` — every cause the symbol can't be sized this bar
  (multiple allowed); empty ⇔ sizable (live AND weighted). A live
  symbol still awaiting the walk-forward recalc for a weight carries
  ``'waiting_weight_recalc'`` — informational, coexists with
  ``live=True`` (the live set feeds the recalc, so waiting symbols stay
  in it).

Statuses are refreshed per-bar for the event symbol and for all symbols
at each recalc; transitions are INFO-logged. Weights are computed over
the **live subset** only and sum to 1 across it; non-live/unweighted
symbols skip sizing with ``skip_reason`` set to the symbol's **primary
recorded reason itself** — the first entry of the canonically-ordered
reason list (``'warmup_forecast'`` / ``'warmup_correlation'`` /
``'constant_price'`` / ``'waiting_weight_recalc'``, plus ``'removed'``
once a removed symbol is flat); there is no separate label vocabulary,
and the full list is exposed per-row in the ``universe_reasons``
diagnostic column. The live set grows monotonically
during a backtest EXCEPT for explicit exclusions (constant-price until
re-entry; removal permanently). ``universe_status(symbol)`` is the
introspection hook. Automatic delisting *detection* is future work.

The manager performs **walk-forward** weight estimation in every mode:
at each recalc point it re-assesses liveness and derives ρ from a
trailing window — it pulls ``corr_lookback`` bars at ``corr_timeframe``
(default ``'1d'``) for every *live* symbol via the data handler,
computes per-bar price changes per ``corr_mode``
(``'absolute_price_chg'`` → ``.diff()``, default — futures-safe for
negative/zero prices; ``'simple_return'`` → ``.pct_change()`` — for
strictly positive-price assets), and feeds ρ to the ``analytics``
optimizers. Under ``'min_variance'`` (exact long-only minimum-variance
QP) / ``'risk_parity'`` (equal risk contribution) ρ drives the
*weights*; under ``'equal_weight'`` the weights are ``1/N`` but ρ is
still consumed for the IDM — so a large decorrelated book earns its
leverage instead of freezing at the constructor ``idm``. Both
corr-based weight modes run correlation-only — the equal-vol convention,
consistent with sizing already dividing by each instrument's σ. The
matching IDM (``analytics.diversification_multiplier``) is updated from
the same ρ on every successful recompute (clamped to ``idm_cap``),
keeping weights and IDM coherent.
``update_bar`` auto-recalls the method every ``corr_step_size``
completed ``corr_timeframe`` periods (default ``30``; multi-symbol bars
at the same timestamp and sub-period base bars are de-duplicated via
``data.get_period_start``); set ``corr_step_size=0`` to disable
auto-recalc — note that in a backtest this freezes the one-shot
``__init__`` result (an empty universe, since no bars have streamed
yet) unless the caller recalcs manually. An empty live set yields an
empty ``instrument_weight`` + INFO log (expected during backtest
warmup); a singleton live set yields ``{symbol: 1.0}`` with
``idm = 1.0``. If data gaps leave fewer than 30 valid return
observations despite full-lookback members, the manager logs a WARNING
and falls back to equal-weight over the live subset (the ρ=1 degenerate
case); ``self.idm`` is left untouched in this branch.

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
decay on a flat symbol), or forming bars. A zero
combined weight is NOT a skip — it is a target of 0, so a held position
is flattened (a flat one is simply labelled ``'zero_weight'``). Any skip
that would strand a *held* position emits a WARNING. Idempotent: a
stable forecast on consecutive bars produces no further orders once the
position matches the target.
"""

import datetime
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid a config<->riskmanager import cycle at module load
    from config import InstrumentConfig

from analytics import (
    correlation_matrix, diversification_multiplier, equal_weight,
    min_variance, risk_parity,
)
from data import get_period_start
from event import BarEvent, OrderType, Direction
from riskmanager._base import (
    RiskManager, _DataHandlerLike, _OrchestratorLike, _PortfolioLike,
    _StrategyLike,
)
from strategy import Strategy
from volatility import VolEstimator

logger = logging.getLogger(__name__)

# Minimum number of valid return observations (rows surviving
# ``.diff()`` / ``.pct_change()`` + ``.dropna()``) required for a stable
# Pearson correlation estimate. Below this, ``calculate_instrument_weight``
# logs a WARNING and falls back to equal-weight (the ρ=1 degenerate case).
_MIN_CORR_OBS = 30

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

# ── Tradable-universe reason vocabulary ──────────────────────────────
# Every cause a symbol can't be sized this bar is recorded in its
# ``UniverseStatus.reasons`` list (multiple allowed, not one
# precedence-picked label). LIVENESS reasons make the symbol not-live;
# the EXCLUSION subset additionally marks a policy exclusion.
# ``'waiting_weight_recalc'`` is deliberately NOT a liveness reason: it
# marks the recalc-lag state (gates passed, awaiting the walk-forward
# recalc for a weight) and coexists with ``live=True`` — the live set
# feeds the recalc, so waiting symbols must stay in it.
_LIVENESS_REASONS = (
    'warmup_forecast',       # strategy gate unmet (transient)
    'warmup_correlation',    # corr_lookback data gate unmet (transient)
    'constant_price',        # zero-variance excluded at the latest recalc
    'removed',               # manual removal (permanent)
)
_EXCLUSION_REASONS = ('constant_price', 'removed')
# Per-reason flatten policy: exclusion reasons in this set flatten a HELD
# position (target 0 ahead of the sigma ladder — a dead sigma must not
# block the exit). 'constant_price' keeps hold-and-warn semantics:
# flattening on temporary constancy would churn; if flatten-on-constant
# is ever wanted it is a one-entry policy change made deliberately.
_FLATTEN_ON = frozenset({'removed'})


@dataclass
class UniverseStatus:
    """Per-symbol tradable-universe record — two orthogonal axes + reasons.

    ``live`` (measured gate liveness) and ``excluded`` (policy layer) are
    **derived** by the update rule (``_refresh_symbol_status``) and never
    set independently, so contradictory states cannot exist:

    * ``live``     == no reason in ``_LIVENESS_REASONS`` is present
    * ``excluded`` == a reason in ``_EXCLUSION_REASONS`` is present
    * ``reasons == []`` ⇔ sizable (live AND present in
      ``instrument_weight``) — the self-maintained invariant; a direct
      ``instrument_weight`` overwrite can violate it externally, which
      the recalc-time consistency check WARNs about (never raises).

    Mutate only through the risk manager's own transitions (the per-bar
    and per-recalc refreshes, ``remove_symbol``); the public introspection
    hook ``universe_status(symbol)`` returns a defensive copy.
    """

    live: bool
    excluded: bool
    reasons: List[str]


class VolTargetingRiskManager(RiskManager):
    """Forecast-aware cash-vol-targeting sizer (Carver's framework).

    Owns one weight dict:

    - ``instrument_weight``: per-symbol capital weight, populated at
      construction by ``calculate_instrument_weight()`` (default 1/N
      across ``strategy.symbol_list``; with ``mode='min_variance'`` or
      ``'risk_parity'`` derived inline from a trailing window of
      returns pulled from ``self.data_handler`` and optimized by the
      ``analytics`` portfolio optimizers).

    Strategy weighting is **not** a risk-manager concern — an
    ``orchestrator.Orchestrator`` (passed in the ``strategy`` slot in
    place of a single ``Strategy``) owns the per-strategy weights and
    bakes them into the combined forecast.

    ``calculate_instrument_weight`` can be re-called or its dict
    overwritten directly to refresh weights without per-bar wiring.
    With a corr-based mode (``'min_variance'`` / ``'risk_parity'``)
    and ``corr_step_size > 0``,
    ``update_bar`` auto-recalls ``calculate_instrument_weight`` every
    ``corr_step_size`` completed ``corr_timeframe`` periods
    (walk-forward; period crossings are detected via
    ``data.get_period_start`` so multi-symbol bars at the same timestamp
    and sub-period base bars contribute one tick per period, not N or 24N);
    the matching IDM is recomputed alongside (clamped to ``idm_cap``).

    Per-bar diagnostic log analogous to ``Strategy.get_records``: every
    completed bar appends one row to ``self._records[symbol]`` and emits
    a DEBUG log line. Columns capture all sizing inputs and
    intermediates (``forecast``, ``sigma``, ``instrument_weight``,
    ``capital``, ``idm``, ``annual_target_vol``,
    ``position_buffer``, ``annual_cash_target``, ``target_qty``,
    ``current_qty`` (realized position), ``pending_mkt_order_quantity``
    (signed sum of in-flight MKT orders — the resize diff targets
    ``current_qty + pending_mkt_order_quantity``, the projected
    position), ``trade_qty``, ``buffer_threshold``) plus
    ``submitted`` (bool) and ``skip_reason`` — ``None`` when an order
    was submitted; ``'warmup_volatility'`` when sigma is not ready; for
    a symbol outside the weighted universe, the symbol's **primary
    universe reason itself**: ``'warmup_forecast'`` (strategy has no
    forecast yet — also fires post-weighting when a *weighted* symbol's
    forecast cache is still empty, e.g. after a direct
    ``instrument_weight`` overwrite that bypassed the liveness gate),
    ``'warmup_correlation'`` (fewer than ``corr_lookback`` bars),
    ``'constant_price'`` (excluded at the latest recalc — zero variance
    over the corr window), or ``'waiting_weight_recalc'`` (both gates
    pass, awaiting the walk-forward recalc for a weight); the
    substantive skips ``'zero_vol'``,
    ``'dead_band'``, ``'at_target'``, and ``'zero_weight'`` — the last
    meaning a zero combined weight with the position *already flat* (a
    *held* position at zero weight is instead flattened, so it records
    ``submitted=True`` / ``skip_reason=None``); or ``'removed'`` — a
    manually removed symbol that is already flat (a *held* removed
    position is flattened first, ahead of the sigma checks). A skip that
    strands a
    held position (``'zero_vol'`` / any warmup or universe reason)
    additionally
    emits a WARNING. Symbols absent
    from ``instrument_weight`` (outside the tradable universe per the
    liveness gate) leave the row's
    ``instrument_weight`` ``None``. Each row also carries the
    universe state: ``universe_live`` (bool) and ``universe_reasons``
    (the symbol's full reason list, comma-joined; ``''`` == sizable).
    Read via ``risk_manager.get_records(symbol)``.
    """

    def __init__(
        self,
        portfolio: _PortfolioLike,
        strategy: Union[_StrategyLike, _OrchestratorLike],
        vol_estimator: VolEstimator,
        data_handler: _DataHandlerLike,
        instruments: Optional[Dict[str, "InstrumentConfig"]] = None,
        idm: float = 1.0,
        idm_cap: Optional[float] = 2.5,
        annual_target_vol: Optional[float] = None,
        vol_target_mode: str = 'dollar_volatility',
        position_buffer: float = 0.25,
        instrument_weight_mode: str = 'equal_weight',
        corr_lookback: int = 60,
        corr_step_size: int = 30,
        corr_timeframe: str = '1d',
        corr_mode: str = 'absolute_price_chg',
        corr_floor: Optional[float] = None,
        corr_shrinkage: Optional[str] = 'ledoit_wolf',
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
            (forecast) and at construction (symbol_list, for the
            equal-weight default).
        vol_estimator
            ``VolEstimator`` providing ``get_annual_vol(symbol)`` in
            price (cash) units. Updated by ``update_bar`` on every
            completed bar.
        data_handler
            Data-handler surface used to pull a trailing window of closes
            for the inline correlation-matrix derivation when
            ``calculate_instrument_weight`` is called with a corr-based
            ``mode`` (``'min_variance'`` / ``'risk_parity'``) and
            ``corr_matrix=None``. ``corr_timeframe`` must be a key of
            ``data_handler.timeframes``.
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
            Default ``1.0``. Must be ``> 0``. Auto-updated whenever a
            corr-based (min-variance or risk-parity) recompute consumes
            a corr matrix (passed or derived), clamped to ``idm_cap``
            when the cap is enabled; the constructor default applies
            only until the first successful recompute. Must not exceed
            ``idm_cap`` when the cap is enabled.
        idm_cap
            Upper bound applied to ``self.idm`` whenever it is
            auto-updated from a corr-based weight recompute — both the
            inline-derivation path and an explicitly passed
            ``corr_matrix``. Default ``2.5`` (Carver's recommended
            maximum): the IDM multiplies every position linearly, so
            correlation-estimation noise must not translate into
            unbounded leverage. ``None`` disables the cap. Must be
            ``>= 1.0`` when not ``None`` (the DM is mathematically
            ``>= 1`` for a fully-allocated long-only weight vector).
            Direct assignments to ``self.idm`` by subclasses or
            downstream code are NOT capped — the same owner-may-
            overwrite convention as the weight dicts.
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
            Default weighting scheme stored on ``self.instrument_weight_mode``
            and used by ``calculate_instrument_weight`` when it is called
            without an explicit ``mode`` (including the call from this
            constructor). One of ``'equal_weight'`` (default),
            ``'min_variance'``, or ``'risk_parity'``. In every mode the
            weights cover the live subset only; at construction time in
            a backtest the deques are empty, so the universe starts
            empty (INFO log, no raise) and fills in at the walk-forward
            recalcs.
        corr_lookback
            Trailing window in ``corr_timeframe`` bars used to pull
            closes for the inline correlation derivation, AND the
            universe liveness threshold: a symbol must carry this many
            bars at ``corr_timeframe`` before it can enter the tradable
            universe (see ``get_live_symbols``). Default ``60``. Must
            be ``>= 32``: the window yields corr_lookback - 1
            price-change observations and the cross-symbol alignment
            trim costs one more, so corr_lookback - 2 must cover the
            30-observation minimum. Must also be ``<=`` the
            ``corr_timeframe`` deque maxlen
            (``data_handler.timeframes[corr_timeframe]``) — otherwise no
            symbol could ever go live.
        corr_step_size
            Auto-recalc cadence: after every ``corr_step_size`` completed
            ``corr_timeframe`` periods, ``update_bar`` re-calls
            ``calculate_instrument_weight`` to refresh weights (and IDM)
            on a walk-forward basis. The cadence is measured in
            ``corr_timeframe`` periods (not raw bar events) — event-stream
            timestamps are bucketed via ``data.get_period_start``, so N
            symbols at the same timestamp and sub-period base bars (e.g.
            base='1h' with corr_timeframe='1d') both contribute exactly
            one period crossing. Default ``30``. Set to ``0`` to disable
            auto-recalc (one-shot at ``__init__`` only). Active in EVERY
            weight mode — under ``'equal_weight'`` the recalc is the
            periodic liveness re-assessment that admits newly-live
            symbols (weights stay 1/N; the IDM still updates).
        corr_timeframe
            Timeframe the data handler reads when assembling the closes
            window. Default ``'1d'``. Must be a key of
            ``data_handler.timeframes``.
        corr_mode
            How per-bar price changes are computed from the closes window
            before correlating. One of ``'absolute_price_chg'`` (default,
            ``.diff()`` — the futures-safe choice: works for contracts
            and synthetic products such as time spreads whose prices can
            be negative or zero, where simple returns are meaningless —
            inf at zero crossings, sign-flipped below zero) or
            ``'simple_return'`` (``.pct_change()`` — for strictly
            positive-price assets such as crypto/equities).
        corr_floor
            Element-wise lower bound applied to the inline-derived
            correlation matrix (see ``_derive_corr_matrix``) before it
            feeds the optimizer and the IDM. Default ``None`` — no
            flooring (the shrunk sample correlations feed through as
            estimated). ``0.0`` is the recommended Carver-style
            setting: negative correlations estimated from a short
            window are mostly sampling noise, and trusting them both
            overweights spuriously anti-correlated instruments
            (min-variance treats them as a free hedge) and inflates the
            IDM; with a ``0.0`` floor and long-only weights the pre-cap
            IDM is bounded by ``sqrt(N)``. Must be in ``[-1.0, 1.0]``
            when not ``None``. NOT applied to an explicitly passed
            ``corr_matrix`` — the caller owns that matrix.
        corr_shrinkage
            Shrinkage estimator applied when deriving the inline
            correlation matrix (see ``_derive_corr_matrix``). Default
            ``'ledoit_wolf'`` — Ledoit-Wolf shrinkage toward scaled
            identity with the closed-form optimal intensity, keeping ρ
            well-conditioned as the instrument count grows toward (or
            past) ``corr_lookback`` observations, where the raw sample
            estimator degrades into noise. Applied *before*
            ``corr_floor`` (estimate → shrink → floor → PSD repair).
            ``None`` disables shrinkage (raw sample correlation, the
            pre-shrinkage behavior). Like ``corr_floor``, estimation
            hygiene for the inline path only — NOT applied to an
            explicitly passed ``corr_matrix``.

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
        if corr_lookback < _MIN_CORR_OBS + 2:
            raise ValueError(
                f"corr_lookback must be >= {_MIN_CORR_OBS + 2}, got "
                f"{corr_lookback}. corr_lookback is the universe liveness "
                f"threshold; the window yields corr_lookback - 1 price-"
                f"change observations, and the cross-symbol alignment trim "
                f"(the event symbol's window ends one bar after the "
                f"laggards', so dropna removes one row at each end of the "
                f"union frame) costs one more — corr_lookback - 2 must "
                f"cover the {_MIN_CORR_OBS}-obs minimum for a stable "
                f"correlation estimate."
            )
        if corr_step_size < 0:
            raise ValueError(
                f"corr_step_size must be >= 0, got {corr_step_size}"
            )
        if corr_timeframe not in data_handler.timeframes:
            raise ValueError(
                f"corr_timeframe '{corr_timeframe}' not registered in "
                f"data_handler.timeframes; available: "
                f"{list(data_handler.timeframes.keys())}"
            )
        maxlen = data_handler.timeframes[corr_timeframe]
        if corr_lookback > maxlen:
            raise ValueError(
                f"corr_lookback ({corr_lookback}) exceeds the "
                f"'{corr_timeframe}' deque maxlen ({maxlen}); no symbol "
                f"could ever accumulate enough bars to pass the liveness "
                f"gate. Increase timeframes['{corr_timeframe}'] or lower "
                f"corr_lookback."
            )
        if corr_mode not in ('simple_return', 'absolute_price_chg'):
            raise ValueError(
                f"Unknown corr_mode: {corr_mode!r}. "
                "Must be 'simple_return' or 'absolute_price_chg'."
            )
        if corr_floor is not None and not (-1.0 <= corr_floor <= 1.0):
            raise ValueError(
                f"corr_floor must be in [-1.0, 1.0] or None to disable, "
                f"got {corr_floor}"
            )
        if corr_shrinkage not in (None, 'ledoit_wolf'):
            raise ValueError(
                f"corr_shrinkage must be None or 'ledoit_wolf', "
                f"got {corr_shrinkage!r}"
            )
        super().__init__(portfolio, strategy)
        self.vol_estimator = vol_estimator
        self.data_handler = data_handler
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
        self.corr_lookback = corr_lookback
        self.corr_step_size = corr_step_size
        self.corr_timeframe = corr_timeframe
        self.corr_mode = corr_mode
        self.corr_floor = corr_floor
        self.corr_shrinkage = corr_shrinkage
        # Default weighting scheme used by ``calculate_instrument_weight``
        # when called without an explicit ``mode``. Set before the
        # construction-time recalc below so the method can read it.
        self.instrument_weight_mode = instrument_weight_mode

        # Walk-forward auto-recalc state driven by ``update_bar``.
        # ``_periods_since_recalc`` counts distinct ``corr_timeframe``
        # periods crossed by the event stream — NOT raw bar events — so
        # multi-symbol bars at the same timestamp (N events / period) and
        # sub-period base bars (e.g. base='1h', corr='1d' → 24 events /
        # period) both contribute exactly one tick per ``corr_timeframe``
        # period. ``_last_seen_period_start`` is the bucket of the most
        # recently observed bar; transitions to a new bucket increment
        # the counter. Reset to 0 on every recalc (success or fallback).
        self._periods_since_recalc: int = 0
        self._last_seen_period_start: Optional[datetime.datetime] = None

        # Per-symbol capital weights. Populated by
        # ``calculate_instrument_weight`` below; may be re-called or the
        # dict overwritten directly to refresh weights without per-bar
        # wiring. (Strategy weighting is owned by the orchestrator, not
        # the risk manager.)
        self.instrument_weight: Dict[str, float] = {}
        # Explicit tradable-universe state: one ``UniverseStatus`` per
        # symbol, the source of truth behind ``get_live_symbols`` and the
        # sizing skip ladder. Initialized here by one evaluation pass —
        # in a backtest the deques are empty and no forecast is cached,
        # so every symbol starts with both warmup reasons set.
        self._universe: Dict[str, UniverseStatus] = {}
        for s in self.strategy.symbol_list:
            self._refresh_symbol_status(s, log=False)
        self.calculate_instrument_weight()

    def _data_gate_met(self, symbol: str) -> bool:
        """True once ``symbol`` carries the full ``corr_lookback`` bars at
        ``corr_timeframe`` — the data half of the liveness gate, and the
        ``warmup_correlation`` boundary. Uses ``count_bars`` (O(1) deque
        length, no DataFrame materialization — this runs per symbol on the
        liveness check); the count includes the current forming bar."""
        return self.data_handler.count_bars(
            symbol, timeframe=self.corr_timeframe,
        ) >= self.corr_lookback

    def _refresh_symbol_status(self, symbol: str,
                               log: bool = True) -> UniverseStatus:
        """Recompute ``symbol``'s ``UniverseStatus`` — the single writer.

        Re-measures the two gate reasons (``'warmup_forecast'`` via
        ``strategy.is_warmed_up``; ``'warmup_correlation'`` via the O(1)
        ``_data_gate_met``), preserves any policy (exclusion) reasons,
        re-derives ``live``/``excluded`` (never set independently), and
        refreshes ``'waiting_weight_recalc'`` from ``symbol in
        instrument_weight`` — so the status is always fresh at the
        decision point and self-heals within one bar after a direct
        weights overwrite. Creates the record on first sight (initial
        evaluation, not logged as a transition); otherwise mutates it in
        place and, when ``log`` is True, INFO-logs a live flip / reason
        change once per transition. Recalc-time callers pass
        ``log=False`` — net changes across a recalc are logged once
        against the pre-recalc snapshot instead, so the provisional
        constant-price re-assessment cannot double-log.
        """
        reasons: List[str] = []
        if not self.strategy.is_warmed_up(symbol):
            reasons.append('warmup_forecast')
        if not self._data_gate_met(symbol):
            reasons.append('warmup_correlation')
        old = self._universe.get(symbol)
        if old is not None:
            reasons.extend(r for r in _EXCLUSION_REASONS if r in old.reasons)
        # Only liveness reasons can be present at this point, so ``live``
        # is simply "no reason yet"; the waiting marker is then appended
        # for live-but-unweighted symbols (it does not affect liveness).
        live = not reasons
        if live and symbol not in self.instrument_weight:
            reasons.append('waiting_weight_recalc')
        excluded = any(r in _EXCLUSION_REASONS for r in reasons)
        if old is None:
            status = UniverseStatus(live=live, excluded=excluded,
                                    reasons=reasons)
            self._universe[symbol] = status
            return status
        if log:
            self._log_status_transition(symbol, old.live, old.reasons,
                                        live, reasons)
        old.live = live
        old.excluded = excluded
        old.reasons = reasons
        return old

    @staticmethod
    def _log_status_transition(symbol: str, old_live: bool,
                               old_reasons: List[str], live: bool,
                               reasons: List[str]) -> None:
        """INFO-log a universe transition (live flip, or reason change at
        unchanged liveness). Silent when nothing changed."""
        if old_live != live:
            logger.info(
                "%s universe: %s -> %s %s", symbol,
                'live' if old_live else 'not live',
                'live' if live else 'not live', reasons,
            )
        elif old_reasons != reasons:
            logger.info(
                "%s universe: reasons %s -> %s", symbol, old_reasons, reasons,
            )

    def _status(self, symbol: str) -> UniverseStatus:
        """Stored status for ``symbol``, lazily evaluated (and stored) on
        first access — covers symbols added to ``strategy.symbol_list``
        after construction."""
        status = self._universe.get(symbol)
        if status is None:
            status = self._refresh_symbol_status(symbol, log=False)
        return status

    def universe_status(self, symbol: str) -> UniverseStatus:
        """Introspection hook: ``symbol``'s universe record as of its last
        evaluation point (its own bars, each weight recalc, removal).
        Returns a defensive copy — universe state is mutated only through
        the risk manager's own transitions."""
        status = self._status(symbol)
        return UniverseStatus(live=status.live, excluded=status.excluded,
                              reasons=list(status.reasons))

    def remove_symbol(self, symbol: str) -> None:
        """Permanently remove ``symbol`` from the tradable universe — the
        manual delisting primitive.

        Adds the exclusion reason ``'removed'`` (⇒ ``live=False``,
        ``excluded=True``), pops the symbol's instrument weight (the
        remaining weights renormalize at the next walk-forward recalc —
        the same lag every weight change already has), and excludes it
        from all future ρ derivations and weight recalcs. A HELD position
        is flattened on the symbol's **next bar** via the normal submit
        path — the flatten runs ahead of the sigma ladder, so a dead
        sigma cannot block the exit; once flat, the symbol's rows record
        ``skip_reason='removed'``. Idempotent.

        Honest limitation: the risk manager only acts in
        ``update_bar(event)`` for the symbol, so a symbol that never
        prints another bar cannot be flattened in the sim (no price, no
        fill) — true delisting settlement remains out of scope.

        Raises ``ValueError`` when ``symbol`` is not in
        ``strategy.symbol_list``.
        """
        if symbol not in self.strategy.symbol_list:
            raise ValueError(
                f"Cannot remove unknown symbol {symbol!r}: not in "
                f"strategy.symbol_list"
            )
        status = self._status(symbol)
        if 'removed' in status.reasons:
            return                                      # idempotent re-call
        status.reasons.append('removed')
        self.instrument_weight.pop(symbol, None)
        self._refresh_symbol_status(symbol, log=False)  # canonicalize + re-derive
        logger.info(
            "%s universe: removed — permanently excluded (reasons %s); "
            "weight popped, a held position flattens on the symbol's "
            "next bar", symbol, status.reasons,
        )

    def get_live_symbols(self) -> List[str]:
        """Return the symbols currently in the tradable universe — a view
        over the explicit universe state, and the input set the weight
        recalc consumes.

        A symbol is *live* when both gates pass — (1) **data gate**: the
        full ``corr_lookback`` bars at ``corr_timeframe``
        (``_data_gate_met``); (2) **strategy gate**:
        ``strategy.is_warmed_up(symbol)`` — AND it is not excluded
        (``'constant_price'`` at the latest recalc, or ``'removed'``).
        Live symbols still awaiting a weight
        (``'waiting_weight_recalc'``) are INCLUDED: the recalc must see
        them or they could never be weighted.

        Order follows ``strategy.symbol_list``. The result grows
        monotonically during a backtest (deques only grow; the warmup
        flag never resets) EXCEPT for explicit exclusions: a
        constant-price symbol drops out until a later recalc re-admits
        it, and a removed symbol drops out permanently.
        """
        return [s for s in self.strategy.symbol_list if self._status(s).live]

    def calculate_instrument_weight(
        self,
        mode: Optional[str] = None,
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> None:
        """Populate ``self.instrument_weight`` according to ``mode``.

        Thin orchestrator over the ``analytics`` portfolio optimizers:
        this method owns the risk-manager concerns — the universe
        liveness gate (``get_live_symbols``), deriving ρ from the data
        handler, the degenerate-universe and data-gap fallbacks, label
        validation against ``strategy.symbol_list``, and the IDM side
        effect — and delegates the weight math itself to
        ``analytics.equal_weight`` / ``analytics.min_variance`` /
        ``analytics.risk_parity``.

        Weights cover the **live subset** only (see ``get_live_symbols``)
        and sum to 1 across it; non-live symbols are absent from the
        dict and skip sizing with their primary recorded universe reason
        as ``skip_reason`` (see ``UniverseStatus``). The method is also
        the recalc-time
        universe bookkeeping point: gate reasons are re-measured for
        every symbol on entry (constant-price marks are provisionally
        cleared — constancy is re-measured by the inline derivation, so
        an excluded symbol re-enters automatically when its window moves
        again), ``'waiting_weight_recalc'`` is refreshed for every symbol
        after the weights land, net status transitions are INFO-logged
        once, and an ``instrument_weight`` that strays outside the live
        set (a direct overwrite) draws a WARNING — never a raise.

        Parameters
        ----------
        mode
            Weighting scheme. When ``None`` (default), falls back to
            ``self.instrument_weight_mode`` (set in the constructor;
            default ``'equal_weight'``). Otherwise one of:

            * ``'equal_weight'`` — ``{symbol: 1/N}`` across the live
              subset (the weights). ρ is still derived inline to update
              the IDM (so a diversified book earns its leverage rather
              than freezing the constructor ``idm``); any passed
              ``corr_matrix`` is ignored — equal_weight owns no
              estimation hook.
            * ``'min_variance'`` — exact long-only minimum-variance
              weights (``min wᵀρw`` s.t. ``Σw = 1``, ``w ≥ 0``), solved
              numerically by ``analytics.min_variance``.
            * ``'risk_parity'`` — equal-risk-contribution weights
              (``analytics.risk_parity``).

            Both corr-based modes run correlation-only — the equal-vol
            convention, which is exactly equivalent to optimizing the
            covariance under equal per-instrument vols and is the right
            assumption here since sizing already divides by each
            instrument's σ.
        corr_matrix
            Optional explicit correlation matrix — the manual/research
            hook. When supplied (corr-based modes), the caller owns the
            **matrix** (no shrinkage / floor / PSD hygiene is applied)
            but NOT the universe: the labels must be a non-empty subset
            of ``self.strategy.symbol_list`` and are then **intersected
            with the live universe** (``get_live_symbols``) — non-live
            labels are dropped with a WARNING, and the optimizer and the
            IDM consume the principal submatrix over the survivors (row
            order is taken from the matrix). An intersection left empty
            behaves like the empty inline universe (empty weight dict +
            INFO log, ``idm`` untouched); a single survivor yields
            ``{symbol: 1.0}`` with ``idm = 1.0``.
            Under budget groups (an orchestrator in the strategy slot)
            the survivors are then weighted **per group** — each group's
            scheme runs on its principal submatrix, scaled by its budget
            renormalized over the groups with survivors and summed
            (sum-of-books); a group whose members were all dropped loses
            its budget to the survivors with a WARNING.
            When ``None``, ρ is derived inline from the data
            handler over the live subset (see ``_derive_corr_matrix``):
            an empty live set — always the case at construction time in
            a backtest since deques are empty — yields an empty weight
            dict + INFO log; a singleton live set yields
            ``{symbol: 1.0}`` with ``idm = 1.0``; and a data-gap
            shortfall (< 30 valid observations despite full-lookback
            members) logs a WARNING and falls back to equal-weight over
            the live subset (the ρ=1 degenerate case).

        Raises
        ------
        ValueError
            On unknown ``mode``; a ``corr_matrix`` (passed or derived)
            failing the ``analytics`` validators (index ≠ columns,
            asymmetric, NaN/inf), empty, or carrying labels outside
            ``self.strategy.symbol_list``; or optimizer solver failure.

        Notes
        -----
        Mutates ``self.instrument_weight`` in place and, whenever a ρ is
        successfully derived (in *every* mode — ``'equal_weight'`` feeds
        the same ρ to the IDM even though its weights are ``1/N``), also
        updates ``self.idm`` via
        ``analytics.diversification_multiplier(...)``, clamped to
        ``idm_cap`` when the cap is enabled, so weights and IDM stay
        coherent. The data-gap equal-weight fallback and the
        empty-universe path leave ``self.idm`` untouched; a singleton
        universe sets ``idm = 1.0``.
        When the forecast source exposes ``get_budget_groups()`` (an
        ``Orchestrator``), every path builds **strategy-budgeted**
        weights: the configured scheme runs within each strategy's book
        and each book's total weight equals its renormalized budget share
        (``w(s) = Σᵢ W'ᵢ·vᵢ(s)`` — see ``_grouped_weights``). A bare
        ``Strategy`` is a single implicit group, reproducing the
        ungrouped math exactly; identical group universes (full overlap)
        also reproduce it, so behavior changes only where universes
        diverge. Safe to re-call any time
        (e.g. monthly rebalances, regime-driven scheme switches);
        ``update_bar`` re-calls this method every ``corr_step_size``
        completed ``corr_timeframe`` periods in every mode (the recalc
        re-assesses liveness even under ``'equal_weight'``).
        """
        if mode is None:
            mode = self.instrument_weight_mode
        # Universe bookkeeping wraps the weight recalc: re-measure gates
        # (+ provisional constant-price clear + incoming consistency
        # check) before, refresh waiting markers + diff-log net
        # transitions + outgoing consistency check after — also on an
        # exception path, so the universe never holds a half-updated
        # state.
        snapshot = {
            s: (st.live, list(st.reasons)) for s, st in self._universe.items()
        }
        self._reassess_universe_for_recalc()
        try:
            self._recalc_weights(mode, corr_matrix)
        finally:
            self._finalize_universe_after_recalc(snapshot)

    def _recalc_weights(self, mode: str,
                        corr_matrix: Optional[pd.DataFrame]) -> None:
        """Weight-scheme dispatch — the body of
        ``calculate_instrument_weight`` (see its docstring for the full
        contract), split out so the universe snapshot / finalize wrapper
        around it stays flat."""
        if mode == 'equal_weight':
            live = self.get_live_symbols()
            if not live:
                self._log_empty_universe(mode)
                self.instrument_weight = {}
                return
            if len(live) == 1:
                # Single-instrument universe: full weight, no
                # diversification credit.
                self.instrument_weight = {live[0]: 1.0}
                self.idm = 1.0
                return
            # Equal weights are scheme-final, but the IDM still measures the
            # book's diversification: derive rho over the live subset and set
            # idm = 1/sqrt(w'rho w) so a large decorrelated universe earns its
            # leverage (up to idm_cap) instead of staying frozen at the
            # constructor value. The IDM scalar is robust to correlation noise
            # (unlike min_variance corner weights). An explicitly passed
            # corr_matrix is ignored here (equal_weight owns no estimation
            # hook); rho is always derived inline — and FIRST, because it
            # also fixes the weighted universe: constant-price symbols are
            # dropped from rho (undefined correlation) and must be equally
            # absent from the weights, or the DM's label check would raise.
            corr_matrix = self._derive_corr_matrix(mode, live)
            if corr_matrix is None:
                # Data-gap / degenerate shortfall — grouped equal weights
                # over the full live subset (budgets still honored); leave
                # self.idm untouched (rho=1 degenerate case), mirroring the
                # corr-based fallback.
                self.instrument_weight = self._grouped_weights(
                    mode, live, None,
                )
                return
            self._mark_constant_price_exclusions(live, corr_matrix.index)
            self.instrument_weight = self._grouped_weights(
                mode, list(corr_matrix.index), corr_matrix,
            )
            self._update_idm_from_corr(corr_matrix)
        elif mode in ('min_variance', 'risk_parity'):
            if corr_matrix is None:
                live = self.get_live_symbols()
                if not live:
                    self._log_empty_universe(mode)
                    self.instrument_weight = {}
                    return
                if len(live) == 1:
                    # Single-instrument universe: full weight, no
                    # diversification credit.
                    self.instrument_weight = {live[0]: 1.0}
                    self.idm = 1.0
                    return
                corr_matrix = self._derive_corr_matrix(mode, live)
                if corr_matrix is None:
                    # Data-gap shortfall — grouped equal-weight fallback over
                    # the live subset (ρ=1 degenerate case, budgets still
                    # honored); IDM intentionally left untouched.
                    self.instrument_weight = self._grouped_weights(
                        mode, live, None,
                    )
                    return
                self._mark_constant_price_exclusions(live, corr_matrix.index)
                dead_level = logging.INFO
            else:
                # Explicit-matrix hook (manual/research). The caller owns
                # the MATRIX (no shrink/floor/PSD hygiene), but the RM owns
                # the UNIVERSE: the labels are intersected with the live
                # set so weights can never land on a symbol whose forecast
                # or corr window isn't ready — un-gated, a weighted
                # symbol with an empty forecast cache reaches sizing and
                # only the None-forecast backstop stands between it and a
                # TypeError.
                if len(corr_matrix.index) == 0:
                    raise ValueError("corr_matrix must not be empty")
                extra = set(corr_matrix.index) - set(self.strategy.symbol_list)
                if extra:
                    raise ValueError(
                        f"corr_matrix labels must be a subset of "
                        f"strategy.symbol_list; extra={sorted(extra)}"
                    )
                live_set = set(self.get_live_symbols())
                kept = [s for s in corr_matrix.index if s in live_set]
                dropped = [s for s in corr_matrix.index if s not in live_set]
                if dropped:
                    logger.warning(
                        "%s: dropping %d non-live symbol(s) from the "
                        "explicit corr_matrix (liveness requires %d bars "
                        "at '%s' plus a warmed-up strategy forecast): %s",
                        mode, len(dropped), self.corr_lookback,
                        self.corr_timeframe, dropped,
                    )
                if not kept:
                    self._log_empty_universe(mode)
                    self.instrument_weight = {}
                    return
                if len(kept) == 1:
                    # Single live instrument after intersection: full
                    # weight, no diversification credit (mirrors the
                    # inline singleton path).
                    self.instrument_weight = {kept[0]: 1.0}
                    self.idm = 1.0
                    return
                if dropped:
                    # Principal submatrix over the surviving labels —
                    # PSD-ness is preserved under principal sub-selection,
                    # so no re-repair is needed. Skipped when nothing was
                    # dropped so an invalid caller matrix (e.g. index !=
                    # columns) still reaches the optimizer's validators
                    # untouched.
                    corr_matrix = corr_matrix.loc[kept, kept]
                dead_level = logging.WARNING
            # Sum-of-books weights: within-group optimization on principal
            # submatrices of the ONE matrix, scaled by renormalized budgets
            # and summed (the mode dispatch lives in _grouped_weights).
            self.instrument_weight = self._grouped_weights(
                mode, list(corr_matrix.index), corr_matrix,
                dead_log_level=dead_level,
            )
            # Auto-update IDM from the same matrix used for weights so the
            # two stay coherent across walk-forward recomputes.
            self._update_idm_from_corr(corr_matrix)
        else:
            raise ValueError(
                f"Unexpected mode: {mode!r} "
                "(expected 'equal_weight', 'min_variance', or 'risk_parity')"
            )

    def _log_empty_universe(self, mode: str) -> None:
        """INFO-log the empty-live-set state (expected during warmup)."""
        logger.info(
            "%s: no live symbols (liveness requires %d bars at '%s' plus "
            "a warmed-up strategy forecast); instrument_weight is empty "
            "until the next recalc",
            mode, self.corr_lookback, self.corr_timeframe,
        )

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
        at ``dead_log_level`` — INFO for the inline path (expected warmup
        staging), WARNING when an explicit ``corr_matrix`` excluded them.
        If every surviving group carries budget 0, equal budgets over the
        survivors apply (WARNING). A group covering all of ``kept``
        consumes ``corr_matrix`` as-is — single-group runs (bare
        ``Strategy`` / full overlap) stay byte-identical to the ungrouped
        path, including optimizer-validator behavior on caller-supplied
        matrices. With >= 2 surviving groups, one INFO line records each
        group's configured vs. renormalized budget share per recalc.
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

    def _reassess_universe_for_recalc(self) -> None:
        """Start-of-recalc universe pass.

        Re-measures both gate reasons for every symbol in
        ``strategy.symbol_list`` (so gate changes made outside the bar
        stream — e.g. research code mutating history between manual
        recalcs — are picked up) and provisionally clears
        ``'constant_price'`` marks: constancy is a per-recalc measurement
        re-taken by the inline derivation, so a previously-excluded
        symbol re-enters the candidate live set here and the mark
        re-lands only if its window is still constant. ``'removed'`` is
        permanent and survives. Ends with the consistency check on the
        INCOMING state — a directly-overwritten ``instrument_weight``
        holding non-live symbols draws a WARNING before it is rebuilt.

        No transition logging here: net changes across the whole recalc
        are logged once by ``_finalize_universe_after_recalc``, so the
        provisional clear of a still-constant symbol stays invisible.
        """
        for s in self.strategy.symbol_list:
            status = self._universe.get(s)
            if status is not None and 'constant_price' in status.reasons:
                status.reasons = [
                    r for r in status.reasons if r != 'constant_price'
                ]
            self._refresh_symbol_status(s, log=False)
        self._warn_if_weights_outside_live()

    def _mark_constant_price_exclusions(self, live: List[str],
                                        kept_labels) -> None:
        """Mark symbols dropped from a successful inline ρ derivation
        (constant closes over the corr window → undefined correlation) as
        excluded with reason ``'constant_price'``.

        Mutation only (the WARNING already fired inside
        ``_derive_corr_matrix``; the net transition is logged by
        ``_finalize_universe_after_recalc``). Only called when a ρ was
        actually produced: the degenerate fallbacks weight the FULL live
        subset — constants included — so recording an exclusion there
        would contradict the weights.
        """
        kept = set(kept_labels)
        for s in live:
            if s not in kept:
                status = self._universe[s]
                if 'constant_price' not in status.reasons:
                    status.reasons.append('constant_price')
                status.live = False
                status.excluded = True

    def _finalize_universe_after_recalc(
        self, snapshot: Dict[str, Any],
    ) -> None:
        """End-of-recalc universe pass.

        Refreshes every symbol's status against the (possibly rebuilt)
        weight dict — newly-weighted symbols shed
        ``'waiting_weight_recalc'``, still-unweighted live symbols keep
        it, and the constant-price marks land in canonical reason order —
        re-runs the consistency check on the OUTGOING state, and
        INFO-logs each symbol whose ``(live, reasons)`` net-changed
        versus the pre-recalc ``snapshot``.
        """
        for s in self.strategy.symbol_list:
            self._refresh_symbol_status(s, log=False)
        self._warn_if_weights_outside_live()
        for s in self.strategy.symbol_list:
            old = snapshot.get(s)
            if old is None:
                continue        # first sighting — initial state, no transition
            status = self._universe[s]
            self._log_status_transition(s, old[0], old[1],
                                        status.live, status.reasons)

    def _warn_if_weights_outside_live(self) -> None:
        """Consistency check: every weighted symbol should be live.

        WARNING, never raise — the documented direct-overwrite convention
        (callers may assign ``instrument_weight`` themselves) must
        survive; the check just makes the resulting liveness-gate bypass
        loud. Sizing still proceeds for such symbols, guarded by the
        None-forecast backstop in ``_compute_target_qty``.
        """
        stray = [
            s for s in self.instrument_weight
            if s not in self._universe or not self._universe[s].live
        ]
        if stray:
            logger.warning(
                "instrument_weight contains %d non-live symbol(s) %s — "
                "likely a direct overwrite bypassing the liveness gate; "
                "sizing proceeds under the None-forecast backstop",
                len(stray), stray,
            )

    def _derive_corr_matrix(
        self, mode: str, symbols: List[str],
    ) -> Optional[pd.DataFrame]:
        """Derive ρ from a trailing window of per-bar price changes.

        Pipeline order: **estimate → shrink → floor → PSD repair**.
        Pulls ``self.corr_lookback`` bars at ``self.corr_timeframe`` for
        each of ``symbols`` (the live subset, per ``get_live_symbols``)
        via ``self.data_handler.get_latest_bars``, computes per-bar
        price changes per ``self.corr_mode``
        (``'simple_return'`` → ``.pct_change().dropna()``;
        ``'absolute_price_chg'`` → ``.diff().dropna()``), **drops
        zero-variance (constant-price) columns** — their correlation to
        anything is undefined (NaN) and would crash the PSD repair /
        optimizers; the excluded symbols are WARNING-logged and the
        returned matrix's labels are then a strict subset of
        ``symbols`` — and calls ``analytics.correlation_matrix`` on the
        result, forwarding ``self.corr_shrinkage`` (Ledoit-Wolf by
        default; the fitted intensity is DEBUG-logged). When
        ``self.corr_floor`` is not ``None``, the matrix is then
        element-wise floored at ``corr_floor``; since element-wise
        clipping does not preserve positive-semidefiniteness in general,
        the floored matrix passes through ``_nearest_psd_correlation``
        before being returned, so the optimizer and the IDM always
        consume the same valid correlation matrix. (All three
        transforms are estimation hygiene: they apply to this inline
        path only, never to an explicitly passed ``corr_matrix``.)

        Returns ``None`` — after logging a WARNING that names ``mode`` —
        when fewer than ``_MIN_CORR_OBS`` (=30) valid observations
        survive (only reachable via data gaps, since every live symbol
        carries the full lookback), when fewer than 2 non-constant
        columns remain, or when the computed matrix still contains NaN
        (safety net); the caller falls back to equal-weight over the
        live subset. Raises ``ValueError`` on an unexpected
        ``self.corr_mode``.
        """
        closes = {
            s: self.data_handler.get_latest_bars_df(
                s, self.corr_lookback,
                timeframe=self.corr_timeframe,
            )['Close']
            for s in symbols
        }
        # ``fill_method=None`` opts out of pandas's deprecated default
        # forward-fill: NaN prices stay NaN and ``.dropna()`` drops them,
        # instead of synthesising a 0 % return from a forward-filled
        # stale close. Defensive — the DataHandler NaN invariant already
        # rules out NaN closes in the bar deques.
        frame = pd.DataFrame(closes)
        if self.corr_mode == 'simple_return':
            returns = frame.pct_change(fill_method=None).dropna()
        elif self.corr_mode == 'absolute_price_chg':
            returns = frame.diff().dropna()
        else:
            raise ValueError(
                f"Unexpected corr_mode: {self.corr_mode!r}"
            )
        if len(returns) < _MIN_CORR_OBS:
            logger.warning(
                "%s: only %d valid return observations "
                "(need >= %d for stable correlation estimate); "
                "falling back to equal-weight (rho=1 degenerate case)",
                mode, len(returns), _MIN_CORR_OBS,
            )
            return None
        # Constant-price symbols have a zero-variance change series — their
        # correlation to anything is undefined (NaN off-diagonals), which
        # would crash the PSD repair / optimizers. Exclude them from this
        # recalc (they re-enter at a later recalc if they move again); an
        # excluded symbol is absent from the weights, skips sizing as
        # 'constant_price', and a held position triggers the
        # stranded-position WARNING.
        variances = returns.var(ddof=0)
        constant = [c for c in returns.columns if variances[c] == 0.0]
        if constant:
            logger.warning(
                "%s: excluding %d constant-price symbol(s) from this "
                "recalc (zero variance over the corr window): %s",
                mode, len(constant), constant,
            )
            returns = returns.drop(columns=constant)
        if returns.shape[1] < 2:
            logger.warning(
                "%s: fewer than 2 non-constant symbols in the corr "
                "window; falling back to equal-weight (rho=1 degenerate "
                "case)",
                mode,
            )
            return None
        corr = correlation_matrix(returns, shrinkage=self.corr_shrinkage)
        if self.corr_shrinkage is not None:
            logger.debug(
                "%s: %s shrinkage intensity %.4f over %d observations",
                mode, self.corr_shrinkage,
                corr.attrs.get('lw_shrinkage', float('nan')), len(returns),
            )
        if corr.isna().any().any():
            # Safety net: never hand NaN to the PSD repair (LinAlgError) or
            # the optimizers (ValueError). Zero-variance filtering above
            # should make this unreachable; data pathologies fall back.
            logger.warning(
                "%s: correlation matrix contains NaN entries despite "
                "zero-variance filtering; falling back to equal-weight",
                mode,
            )
            return None
        if self.corr_floor is not None:
            # Element-wise floor (Carver: zero out spurious negative
            # correlations before weighting). Clipping preserves symmetry
            # and the 1.0 diagonal for any floor <= 1. NOTE: must be an
            # ``is not None`` check — the default 0.0 is falsy.
            corr = corr.clip(lower=self.corr_floor)
        # Element-wise clipping does not preserve PSD in general; the
        # CVXPY optimizers (and the IDM quadratic form) require a valid
        # correlation matrix, so repair here — at the producer — keeps
        # every consumer on the same matrix.
        return self._nearest_psd_correlation(corr)

    @staticmethod
    def _nearest_psd_correlation(corr: pd.DataFrame) -> pd.DataFrame:
        """Project ``corr`` back to a valid (PSD) correlation matrix.

        Cheap ``eigvalsh`` check first: PSD input is returned unchanged
        (the common case — repair only triggers when ``corr_floor``
        clipping actually broke PSD-ness). Otherwise: clip negative
        eigenvalues to zero, reconstruct, rescale to a unit diagonal
        (a congruence transform, so PSD-ness is preserved exactly), and
        re-symmetrize. The result is a small perturbation of the input,
        not a rebuild — labels and the unit diagonal are preserved.
        """
        vals = corr.to_numpy(dtype=float)
        if float(np.linalg.eigvalsh(vals)[0]) >= 0.0:
            return corr
        eigvals, eigvecs = np.linalg.eigh(vals)
        repaired = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        repaired = 0.5 * (repaired + repaired.T)
        np.fill_diagonal(repaired, 1.0)
        return pd.DataFrame(repaired, index=corr.index, columns=corr.columns)

    def _update_idm_from_corr(self, corr_matrix: pd.DataFrame) -> None:
        """Auto-update ``self.idm`` from ``corr_matrix`` and the current
        ``self.instrument_weight``.

        Carver's ``DM = 1 / sqrt(wᵀρw)`` via
        ``analytics.diversification_multiplier``, clamped to ``idm_cap`` when
        the cap is enabled. Shared by every corr-consuming weight mode —
        ``'equal_weight'`` feeds the same ρ even though its weights are 1/N,
        so a diversified book still earns its leverage — keeping weights and
        IDM coherent. The cap is leverage policy: it applies regardless of
        whether ρ was derived inline or passed explicitly.
        """
        idm = diversification_multiplier(self.instrument_weight, corr_matrix)
        if self.idm_cap is not None:
            idm = min(idm, self.idm_cap)
        self.idm = idm

    def update_bar(self, event: BarEvent) -> None:
        """Update sizing inputs and resize the position to the Carver target.

        Skips forming bars (one resize per completed bar). Delegates
        target-qty derivation (and *target-derivation* skip reasons —
        ``'warmup_volatility'`` / ``'zero_vol'`` / the universe reasons)
        to ``_compute_target_qty``; owns *post-target* concerns
        (``'at_target'`` / ``'dead_band'`` / ``'zero_weight'`` and
        ``'removed'`` relabels / submit). A target-derivation skip that
        strands a *held* position
        emits a WARNING (the RM is never silently blind to an open
        position). A zero combined weight is not a skip here — it arrives
        as ``target_qty = 0`` and flattens a held position via the submit
        path; a **removed** symbol arrives the same way (ahead of the
        sigma ladder) so a dead sigma cannot block the exit, and its flat
        rows are relabelled ``'removed'``. Refreshes the event symbol's
        ``UniverseStatus`` first (gate state only changes on a symbol's
        own bars, and the strategy has already processed this event, so
        the status is exactly as fresh as the old inline gate checks).
        Records one diagnostic row per *completed* bar — including
        every early-exit branch — into ``self._records[symbol]`` via
        ``_record_row``, which also emits a DEBUG log line; each row
        carries the universe state as ``universe_live`` (bool) and
        ``universe_reasons`` (the full reason list, comma-joined).
        """
        if event.is_forming:
            return

        symbol = event.symbol
        # Per-bar universe refresh for the event symbol — right before the
        # sizing decision, so the status (incl. 'waiting_weight_recalc')
        # is always fresh at the decision point and self-heals within one
        # bar after a direct instrument_weight overwrite. The recalc block
        # below refreshes ALL symbols whenever it fires.
        self._refresh_symbol_status(symbol)

        # Walk-forward weight recompute on a fixed cadence, measured in
        # ``corr_timeframe`` periods (not raw bar events). Each event is
        # bucketed via ``get_period_start``; the counter only ticks when
        # the bucket changes, so multi-symbol bars at the same timestamp
        # and sub-period base bars don't multi-increment. The recalc runs
        # in EVERY weight mode — even 'equal_weight' needs the periodic
        # liveness re-assessment (newly-live symbols enter the universe
        # at the next recalc). Counter resets whether the recompute
        # succeeded or yielded an empty/fallback universe — by the next
        # attempt the deques have accumulated more bars.
        if self.corr_step_size > 0:
            period_start = get_period_start(event.timestamp, self.corr_timeframe)
            if period_start != self._last_seen_period_start:
                self._last_seen_period_start = period_start
                self._periods_since_recalc += 1
                if self._periods_since_recalc >= self.corr_step_size:
                    self.calculate_instrument_weight()
                    self._periods_since_recalc = 0

        # Update vol estimator first so sigma reflects this bar.
        self.vol_estimator.update(event)

        # Re-read after the recalc block: a recalc that just landed a
        # weight (or marked an exclusion) has already updated this status.
        status = self._universe[symbol]
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
            'universe_live': status.live,
            # Comma-joined string, not the list itself: a list column
            # would trip runlog's sanitize_frame repr fallback; a plain
            # string is parquet-native. '' == sizable (no reasons).
            'universe_reasons': ','.join(status.reasons),
        }
        row.update(self._compute_target_qty(event))

        if row['skip_reason'] is not None:
            # A target-derivation skip (warmup_volatility / zero_vol / a
            # universe reason) means we cannot compute a
            # well-defined target this bar. Harmless when flat, but if we are
            # HOLDING a position the risk manager is leaving it unmanaged —
            # surface it loudly so it can never go unnoticed. (zero_weight no
            # longer lands here: it flows through as target_qty=0 and flattens
            # via the submit path. Flatten-on-universe-exit is deferred to the
            # delisting work; here the target is undefined, so we warn.)
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
        # is 0, OR a zero instrument/strategy weight) lands here when also
        # flat; otherwise the dead-band collapses to zero and any nonzero
        # current position triggers a flatten via the submit path.
        if abs(trade_qty) < 1e-12:                # already at target
            # Relabel the flat case by its cause, most specific first: a
            # REMOVED symbol's zero target is the permanent exclusion
            # (its row weight is None — it was popped); a zero instrument
            # weight is more informative than the generic 'at_target'
            # (the position is flat *because* it carries no weight).
            # instrument_weight is a populated float in the zero-weight
            # case: a symbol absent from instrument_weight returned
            # earlier with its universe reason.
            if 'removed' in status.reasons:
                row['skip_reason'] = 'removed'
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

        1. **Removal flatten** (``_FLATTEN_ON`` exclusion reasons, i.e.
           ``'removed'``) — returns ``target_qty = 0`` with no skip,
           AHEAD of the sigma checks: a flatten needs no sigma, and a
           delisted symbol's sigma is typically dead (``'zero_vol'``),
           which would otherwise strand the exit. ``update_bar``
           relabels the already-flat case ``'removed'``.
        2. Sigma checks — ``'warmup_volatility'`` (not ready) /
           ``'zero_vol'``.
        3. Unweighted symbol — skip with the symbol's primary recorded
           universe reason itself (the first entry of the
           canonically-ordered reason list: ``'warmup_forecast'`` /
           ``'warmup_correlation'`` / ``'constant_price'`` /
           ``'waiting_weight_recalc'``); the full
           reason list is in the row's ``universe_reasons`` column.
        4. None-forecast backstop — ``'warmup_forecast'`` can also fire
           *post-weighting*: a weighted symbol whose forecast cache is
           still empty (``get_forecast`` → ``None`` — e.g. weights
           overwritten directly, bypassing the liveness gate) skips here
           instead of feeding ``None`` into the formula.

        A zero instrument
        weight (``iw == 0``) is **not** a skip — it returns
        ``target_qty = 0`` (with ``skip_reason`` left ``None``) so a held
        position is flattened by ``update_bar``'s submit path and a flat
        one is relabelled ``'zero_weight'`` there. The returned dict is
        spliced into the diagnostic row by ``update_bar`` via
        ``row.update(...)``; intermediates computed before an
        early-exit fires are populated, those after remain ``None``,
        preserving the row schema across branches.
        """
        symbol = event.symbol
        out: Dict[str, Any] = {
            'target_qty': None, 'skip_reason': None,
            'sigma': None, 'instrument_weight': None,
            'annual_cash_target': None,
        }

        status = self._status(symbol)
        if _FLATTEN_ON.intersection(status.reasons):
            # Removed symbol: flatten policy — a zero target flows through
            # the normal submit path exactly like the zero-weight case
            # (no skip), so a held position exits even when sigma is dead.
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
            # Outside the weighted universe — the recorded reason list
            # (refreshed by update_bar just before) names every cause and
            # is built in precedence order (forecast → correlation →
            # exclusions → waiting), so its first entry IS the primary
            # cause; store it directly. The empty-list fallback is
            # defensive only: the per-bar refresh guarantees an unweighted
            # symbol carries at least 'waiting_weight_recalc'.
            # ``instrument_weight`` stays None in the diagnostic row —
            # truthful, vs. recording a synthetic 0.0. Note the trigger is
            # weight-membership, not the reason list: a directly
            # overwritten weight on a non-live symbol still sizes (the
            # documented gate bypass, WARNed at the next recalc), guarded
            # by the None-forecast backstop below.
            out['skip_reason'] = (status.reasons[0] if status.reasons
                                  else 'waiting_weight_recalc')
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
            # directly (the documented convention bypasses the liveness
            # gate) or a matrix hook weighted a symbol whose strategy gate
            # is unmet. Skip before the formula, which would raise on None
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
