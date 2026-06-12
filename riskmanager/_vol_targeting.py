"""CarverVolTargetingRiskManager — forecast-aware Carver vol-targeting sizer.

Implements Carver's vol-targeting framework (Systematic Trading, Ch. 10)
in **cash-vol** form. The risk target is a dollar amount of vol per
period; the instrument's dollar vol per period divides that target to
give the size:

    # vol_target_mode='dollar_volatility' (default — institutional futures
    # convention: a fixed annual $ vol budget, like a drawdown limit that
    # resets yearly instead of compounding with the account):
    annual_cash_target = IDM × strategy_weight × instrument_weight
                                × annualized_target_vol × (forecast / TARGET_AVG_ABS_FORECAST)

    # vol_target_mode='percent_volatility' (Carver's original form — the
    # vol budget is a fraction of *current* account equity, so position
    # sizes compound as the account grows/shrinks):
    annual_cash_target = capital × IDM × strategy_weight × instrument_weight
                                × annualized_target_vol × (forecast / TARGET_AVG_ABS_FORECAST)

    daily_cash_target  = annual_cash_target / sqrt(days_per_year)
    target_qty         = daily_cash_target / daily_price_vol
                       = annual_cash_target / annualized_price_vol

where:
    capital                = portfolio.calculate_balance()              (account equity)
    IDM                    = instrument diversification multiplier      (constructor)
    strategy_weight        = per-strategy capital weight                (self.strategy_weight)
    instrument_weight      = per-symbol capital weight                  (self.instrument_weight)
    annualized_target_vol  = annualized vol target                      (constructor; REQUIRED —
                             $ amount in dollar mode, e.g. 250_000;
                             fraction of equity in percent mode, e.g. 0.25 = 25 %)
    annualized_price_vol   = annualized stdev of price changes ($-units)  (VolEstimator)
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
``annualized_target_vol / σ_pct`` form (``σ_$ = price × σ_pct``).

``TARGET_AVG_ABS_FORECAST`` and ``FORECAST_CAP`` are project-wide
constants on ``Strategy`` (default ``50.0`` and ``100.0``). The
``forecast / TARGET_AVG_ABS_FORECAST`` factor rescales from the project's
``±FORECAST_CAP`` forecast convention to Carver's ``±20`` (Carver divides
by 10; our 5×-larger scale gets a 5×-larger denominator). At
``|forecast| = TARGET_AVG_ABS_FORECAST`` the factor is 1.0 — exactly
Carver's vol-target notional. At ``|forecast| = FORECAST_CAP`` (= 2 ×
target by design) the factor is 2.0, doubling the size. At
``forecast = 0`` the target is zero (flat).

Weights are owned by the risk manager (no external allocator
dependency). ``self.instrument_weight: Dict[str, float]`` spreads
capital across symbols and defaults to equal-weight ``1/N`` across
``strategy.symbol_list``. ``self.strategy_weight: Dict[str, float]``
spreads capital across strategies and is a placeholder today —
``{strategy_class_name: 1.0}`` for the single bound strategy. Both
dicts are populated at ``__init__`` by ``calculate_instrument_weight()``
/ ``calculate_strategy_weight()``; either method can be re-called later
(or its dict overwritten directly) to refresh weights — e.g. monthly
rebalances, correlation-driven weight schemes — without per-bar wiring.

**Universe liveness gating**: with staggered listings, symbols do not
share the full price history, so weights are computed over the **live
subset** only. A symbol is *live* when (1) it has the full
``corr_lookback`` bars at ``corr_timeframe`` (data gate — every live
member contributes the complete correlation window, so the estimation
window never shrinks) and (2) ``strategy.is_warmed_up(symbol)`` is True
(strategy gate — the measured flag the ``Strategy`` base sets when the
first non-NaN forecast is cached). Non-live symbols are absent from
``instrument_weight`` and skip sizing with ``skip_reason='not_live'``;
live weights sum to 1 across the live subset. The live set is monotone
non-decreasing during a backtest. Delisting/universe-exit handling is
future work.

The manager performs **walk-forward** weight estimation in every mode:
at each recalc point it re-assesses liveness, and — under
``'min_variance'`` / ``'risk_parity'`` — pulls ``corr_lookback`` bars
at ``corr_timeframe`` (default ``'1d'``) for every *live* symbol via
the data handler, computes per-bar price changes per ``corr_mode``
(``'absolute_price_chg'`` → ``.diff()``, default — futures-safe for
negative/zero prices; ``'simple_return'`` → ``.pct_change()`` — for
strictly positive-price assets), and derives ρ → weights via the
``analytics`` portfolio optimizers (``analytics.min_variance`` — exact
long-only minimum-variance QP; ``analytics.risk_parity`` — equal risk
contribution). Both run correlation-only — the equal-vol convention,
consistent with sizing already dividing by each instrument's σ. The
matching IDM (``analytics.diversification_multiplier``) is updated from
the same ρ on every successful recompute, keeping the two coherent.
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
3. Submits a MKT order for ``target_qty - current_qty`` if the diff is
   above the configured dead-band (``position_buffer``, Carver §10.7).

Skips on warmup (``sigma is None``), zero vol, zero combined weight, or
forming bars. Idempotent: a stable forecast on consecutive bars produces
no further orders once the position matches the target.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics import (
    correlation_matrix, diversification_multiplier, equal_weight,
    min_variance, risk_parity,
)
from data import get_period_start
from event import BarEvent, OrderType, Direction
from riskmanager._base import (
    RiskManager, _DataHandlerLike, _PortfolioLike, _StrategyLike,
)
from strategy import Strategy
from volatility import VolEstimator

logger = logging.getLogger(__name__)

# Minimum number of valid return observations (rows surviving
# ``.diff()`` / ``.pct_change()`` + ``.dropna()``) required for a stable
# Pearson correlation estimate. Below this, ``calculate_instrument_weight``
# logs a WARNING and falls back to equal-weight (the ρ=1 degenerate case).
_MIN_CORR_OBS = 30


class CarverVolTargetingRiskManager(RiskManager):
    """Forecast-aware cash-vol-targeting sizer (Carver's framework).

    Owns two weight dicts:

    - ``instrument_weight``: per-symbol capital weight, populated at
      construction by ``calculate_instrument_weight()`` (default 1/N
      across ``strategy.symbol_list``; with ``mode='min_variance'`` or
      ``'risk_parity'`` derived inline from a trailing window of
      returns pulled from ``self.data_handler`` and optimized by the
      ``analytics`` portfolio optimizers).
    - ``strategy_weight``: per-strategy capital weight, populated at
      construction by ``calculate_strategy_weight()`` (default
      ``{strategy_class_name: 1.0}`` — placeholder until multi-strategy
      lands).

    Either ``calculate_*`` method can be re-called or its dict
    overwritten directly to refresh weights without per-bar wiring.
    With a corr-based mode (``'min_variance'`` / ``'risk_parity'``)
    and ``corr_step_size > 0``,
    ``update_bar`` auto-recalls ``calculate_instrument_weight`` every
    ``corr_step_size`` completed ``corr_timeframe`` periods
    (walk-forward; period crossings are detected via
    ``data.get_period_start`` so multi-symbol bars at the same timestamp
    and sub-period base bars contribute one tick per period, not N or 24N);
    the matching IDM is recomputed alongside.

    Per-bar diagnostic log analogous to ``Strategy.get_records``: every
    completed bar appends one row to ``self._records[symbol]`` and emits
    a DEBUG log line. Columns capture all sizing inputs and
    intermediates (``forecast``, ``sigma``, ``instrument_weight``,
    ``strategy_weight``, ``capital``, ``idm``, ``annualized_target_vol``,
    ``position_buffer``, ``annual_cash_target``, ``target_qty``,
    ``current_qty``, ``trade_qty``, ``buffer_threshold``) plus
    ``submitted`` (bool) and ``skip_reason`` — ``None`` when an order
    was submitted, otherwise one of ``'warmup'``, ``'zero_vol'``,
    ``'not_live'`` (symbol absent from ``instrument_weight`` — outside
    the current tradable universe per the liveness gate),
    ``'zero_weight'``, ``'dead_band'``, ``'at_target'``. Read via
    ``risk_manager.get_records(symbol)``.
    """

    def __init__(
        self,
        portfolio: _PortfolioLike,
        strategy: _StrategyLike,
        vol_estimator: VolEstimator,
        data_handler: _DataHandlerLike,
        idm: float = 1.0,
        annualized_target_vol: Optional[float] = None,
        vol_target_mode: str = 'dollar_volatility',
        position_buffer: float = 0.25,
        instrument_weight_mode: str = 'equal_weight',
        corr_lookback: int = 60,
        corr_step_size: int = 30,
        corr_timeframe: str = '1d',
        corr_mode: str = 'absolute_price_chg',
        corr_floor: Optional[float] = 0.0,
    ):
        """
        Parameters
        ----------
        portfolio
            Portfolio surface (positions, balance, submit_order).
        strategy
            Strategy exposing ``get_forecast(symbol)`` and ``symbol_list``.
            Read on every completed bar (forecast) and at construction
            (symbol_list, for the equal-weight default).
        vol_estimator
            ``VolEstimator`` providing ``get_annualized_vol(symbol)`` in
            price (cash) units. Updated by ``update_bar`` on every
            completed bar.
        data_handler
            Data-handler surface used to pull a trailing window of closes
            for the inline correlation-matrix derivation when
            ``calculate_instrument_weight`` is called with ``mode='min_variance'``
            and ``corr_matrix=None``. ``corr_timeframe`` must be a key of
            ``data_handler.timeframes``.
        idm
            Instrument diversification multiplier (Carver Ch. 8).
            Default ``1.0``. Must be ``> 0``. Auto-updated whenever a
            min-variance recompute consumes a corr matrix (passed or
            derived); the constructor default applies only until the
            first successful recompute.
        annualized_target_vol
            Annualized volatility target (Carver's ``τ``). REQUIRED —
            no default; its units depend on ``vol_target_mode``:
            a dollar amount (must be ``> 0``, e.g. ``250_000`` = $250k
            of annual vol) under ``'dollar_volatility'``, or a fraction
            of current account equity (must be in ``(0, 1)``, e.g.
            ``0.25`` = 25 %) under ``'percent_volatility'``.
        vol_target_mode
            How ``annualized_target_vol`` is interpreted. One of:

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
            be ``>= 31`` (so the window yields at least 30 price-change
            observations) and ``<=`` the ``corr_timeframe`` deque maxlen
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
            auto-recalc (one-shot at ``__init__`` only). Only active
            under a corr-based ``instrument_weight_mode``
            (``'min_variance'`` / ``'risk_parity'``); no effect under
            ``'equal_weight'``.
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
            feeds the optimizer and the IDM. Default ``0.0`` — Carver's
            practice: negative correlations estimated from a short
            window are mostly sampling noise, and trusting them both
            overweights spuriously anti-correlated instruments
            (min-variance treats them as a free hedge) and inflates the
            IDM. With the default floor and long-only weights the
            pre-cap IDM is bounded by ``sqrt(N)``. ``None`` disables
            flooring. Must be in ``[-1.0, 1.0]`` when not ``None``.
            NOT applied to an explicitly passed ``corr_matrix`` — the
            caller owns that matrix.

        Raises
        ------
        ValueError
            On invalid constructor parameters.
        """
        if idm <= 0:
            raise ValueError(f"idm must be > 0, got {idm}")
        if vol_target_mode not in ('dollar_volatility', 'percent_volatility'):
            raise ValueError(
                f"Unknown vol_target_mode: {vol_target_mode!r}. "
                "Must be 'dollar_volatility' or 'percent_volatility'."
            )
        if annualized_target_vol is None:
            raise ValueError(
                "annualized_target_vol must be supplied explicitly (no "
                "default): a dollar amount under 'dollar_volatility' or "
                "a fraction in (0, 1) under 'percent_volatility'."
            )
        if vol_target_mode == 'percent_volatility':
            if not (0 < annualized_target_vol < 1):
                raise ValueError(
                    f"annualized_target_vol must be in (0, 1) under "
                    f"'percent_volatility', got {annualized_target_vol}"
                )
        elif vol_target_mode == 'dollar_volatility':
            if annualized_target_vol <= 0:
                raise ValueError(
                    f"annualized_target_vol must be > 0 under "
                    f"'dollar_volatility', got {annualized_target_vol}"
                )
        else:
            raise ValueError(
                f"Unexpected vol_target_mode: {vol_target_mode!r}"
            )
        if not (0.0 <= position_buffer < 1.0):
            raise ValueError(
                f"position_buffer must be in [0, 1), got {position_buffer}"
            )
        if corr_lookback < _MIN_CORR_OBS + 1:
            raise ValueError(
                f"corr_lookback must be >= {_MIN_CORR_OBS + 1}, got "
                f"{corr_lookback}. corr_lookback is the universe liveness "
                f"threshold and yields corr_lookback - 1 price-change "
                f"observations, which must cover the {_MIN_CORR_OBS}-obs "
                f"minimum for a stable correlation estimate."
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
        super().__init__(portfolio, strategy)
        self.vol_estimator = vol_estimator
        self.data_handler = data_handler
        self.idm = idm
        # Narrowed to float by the None-rejection above.
        self.annualized_target_vol: float = annualized_target_vol
        self.vol_target_mode = vol_target_mode
        self.position_buffer = position_buffer
        self.corr_lookback = corr_lookback
        self.corr_step_size = corr_step_size
        self.corr_timeframe = corr_timeframe
        self.corr_mode = corr_mode
        self.corr_floor = corr_floor
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

        # Per-symbol / per-strategy capital weights. Populated by the
        # ``calculate_*`` methods below; either may be re-called or its
        # dict overwritten directly to refresh weights without per-bar
        # wiring.
        self.instrument_weight: Dict[str, float] = {}
        self.strategy_weight: Dict[str, float] = {}
        self.calculate_instrument_weight()
        self.calculate_strategy_weight()

    def get_live_symbols(self) -> List[str]:
        """Return the symbols currently in the tradable universe.

        A symbol is *live* when both gates pass:

        1. **Data gate** — it has the full ``corr_lookback`` bars at
           ``corr_timeframe`` (``get_latest_bars`` returns *up to* n
           rows, so ``len == corr_lookback`` means "at least that many
           available"; the count includes the current forming bar).
        2. **Strategy gate** — ``strategy.is_warmed_up(symbol)``: the
           strategy has cached its first non-NaN forecast for the
           symbol, so it can actually trade it.

        Order follows ``strategy.symbol_list``. The result is monotone
        non-decreasing during a backtest (deques only grow; the warmup
        flag never resets).
        """
        return [
            s for s in self.strategy.symbol_list
            if len(self.data_handler.get_latest_bars(
                s, self.corr_lookback, timeframe=self.corr_timeframe,
            )) >= self.corr_lookback
            and self.strategy.is_warmed_up(s)
        ]

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
        dict and skip sizing with ``skip_reason='not_live'``.

        Parameters
        ----------
        mode
            Weighting scheme. When ``None`` (default), falls back to
            ``self.instrument_weight_mode`` (set in the constructor;
            default ``'equal_weight'``). Otherwise one of:

            * ``'equal_weight'`` — ``{symbol: 1/N}`` across the live
              subset. ``corr_matrix`` is ignored if passed.
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
            hook. When supplied (corr-based modes), the **caller owns
            the universe**: the liveness gate is NOT applied, and the
            matrix labels must be a non-empty subset of
            ``self.strategy.symbol_list`` (row order is taken from the
            matrix). When ``None``, ρ is derived inline from the data
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
        Mutates ``self.instrument_weight`` in place and, on successful
        corr-based computes, also updates ``self.idm`` via
        ``analytics.diversification_multiplier(...)`` so weights and IDM
        stay coherent. The equal-weight fallback and empty-universe
        paths leave ``self.idm`` untouched. Safe to re-call any time
        (e.g. monthly rebalances, regime-driven scheme switches);
        ``update_bar`` re-calls this method every ``corr_step_size``
        completed ``corr_timeframe`` periods in every mode (the recalc
        re-assesses liveness even under ``'equal_weight'``).
        """
        if mode is None:
            mode = self.instrument_weight_mode
        if mode == 'equal_weight':
            live = self.get_live_symbols()
            if not live:
                self._log_empty_universe(mode)
                self.instrument_weight = {}
                return
            self.instrument_weight = equal_weight(live)
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
                    # Data-gap shortfall — equal-weight fallback over the
                    # live subset (ρ=1 degenerate case); IDM intentionally
                    # left untouched.
                    self.instrument_weight = equal_weight(live)
                    return
            if len(corr_matrix.index) == 0:
                raise ValueError("corr_matrix must not be empty")
            extra = set(corr_matrix.index) - set(self.strategy.symbol_list)
            if extra:
                raise ValueError(
                    f"corr_matrix labels must be a subset of "
                    f"strategy.symbol_list; extra={sorted(extra)}"
                )
            if mode == 'min_variance':
                self.instrument_weight = min_variance(corr_matrix)
            elif mode == 'risk_parity':
                self.instrument_weight = risk_parity(corr_matrix)
            else:
                raise ValueError(f"Unexpected mode: {mode!r}")
            # Auto-update IDM from the same matrix used for weights so
            # the two stay coherent across walk-forward recomputes.
            self.idm = diversification_multiplier(
                self.instrument_weight, corr_matrix,
            )
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

    def _derive_corr_matrix(
        self, mode: str, symbols: List[str],
    ) -> Optional[pd.DataFrame]:
        """Derive ρ from a trailing window of per-bar price changes.

        Pulls ``self.corr_lookback`` bars at ``self.corr_timeframe`` for
        each of ``symbols`` (the live subset, per ``get_live_symbols``)
        via ``self.data_handler.get_latest_bars``, computes per-bar
        price changes per ``self.corr_mode``
        (``'simple_return'`` → ``.pct_change().dropna()``;
        ``'absolute_price_chg'`` → ``.diff().dropna()``), and calls
        ``analytics.correlation_matrix`` on the result.

        Returns ``None`` — after logging a WARNING that names ``mode`` —
        when fewer than ``_MIN_CORR_OBS`` (=30) valid observations
        survive (only reachable via data gaps, since every live symbol
        carries the full lookback); the caller falls back to
        equal-weight over the live subset. Raises ``ValueError`` on an
        unexpected ``self.corr_mode``.
        """
        closes = {
            s: self.data_handler.get_latest_bars(
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
        return correlation_matrix(returns)

    def calculate_strategy_weight(self) -> None:
        """Populate ``self.strategy_weight`` with the single-strategy placeholder.

        ``analytics.equal_weight`` over the one bound strategy —
        ``{strategy_class_name: 1.0}``. Will become a real ``1/M`` (or
        correlation-driven) allocation once the risk manager holds
        multiple strategies. Mutates in place; safe to re-call any time.
        """
        name = self.strategy.__class__.__name__
        self.strategy_weight = equal_weight([name])

    def update_bar(self, event: BarEvent) -> None:
        """Update sizing inputs and resize the position to the Carver target.

        Skips forming bars (one resize per completed bar). Delegates
        target-qty derivation (and *target-derivation* skip reasons
        ``'warmup'`` / ``'zero_vol'`` / ``'not_live'`` /
        ``'zero_weight'``) to ``_compute_target_qty``; owns *post-target*
        concerns
        (``'at_target'`` / ``'dead_band'`` / submit). Records one
        diagnostic row per *completed* bar — including every early-exit
        branch — into ``self._records[symbol]`` via ``_record_row``,
        which also emits a DEBUG log line.
        """
        if event.is_forming:
            return

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

        symbol = event.symbol
        forecast = self.strategy.get_forecast(symbol)
        capital = self.portfolio.calculate_balance()
        current_qty = self.portfolio.positions.get(symbol, 0.0)

        # Seed the diagnostic row with always-known inputs;
        # _compute_target_qty supplies sigma / weights /
        # annual_cash_target / target_qty / skip_reason via row.update.
        row: Dict[str, Any] = {
            'timestamp': event.timestamp,
            'symbol': symbol,
            'forecast': forecast,
            'sigma': None,
            'instrument_weight': None,
            'strategy_weight': None,
            'capital': capital,
            'idm': self.idm,
            'annualized_target_vol': self.annualized_target_vol,
            'vol_target_mode': self.vol_target_mode,
            'position_buffer': self.position_buffer,
            'annual_cash_target': None,
            'target_qty': None,
            'current_qty': current_qty,
            'trade_qty': None,
            'buffer_threshold': None,
            'submitted': False,
            'skip_reason': None,
        }
        row.update(self._compute_target_qty(event))

        if row['skip_reason'] is not None:
            self._record_row(symbol, row)
            return

        target_qty = row['target_qty']
        trade_qty = target_qty - current_qty
        buffer_threshold = self.position_buffer * abs(target_qty)
        row['trade_qty'] = trade_qty
        row['buffer_threshold'] = buffer_threshold

        # Order matters: ``at_target`` (realized position essentially
        # equals target) is checked first so the diagnostic row carries
        # the more informative label. The dead-band check that follows
        # picks up small-but-nonzero diffs. ``target_qty == 0``
        # (forecast is 0) lands in ``at_target`` when also flat;
        # otherwise the dead-band collapses to zero and any nonzero
        # current position triggers a flatten via the submit path.
        if abs(trade_qty) < 1e-12:                # already at target
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

        target_qty = annual_cash_target / annualized_price_vol, where
        annual_cash_target = IDM × strategy_weight × instrument_weight
        × annualized_target_vol × (forecast / TARGET_AVG_ABS_FORECAST),
        additionally scaled by ``capital`` (current account equity) under
        ``vol_target_mode='percent_volatility'`` — see the module
        docstring for the two forms.

        Owns the *target-derivation* skip ladder (``'warmup'`` /
        ``'zero_vol'`` / ``'not_live'`` / ``'zero_weight'``). The
        returned dict is
        spliced into the diagnostic row by ``update_bar`` via
        ``row.update(...)``; intermediates computed before an
        early-exit fires are populated, those after remain ``None``,
        preserving the row schema across branches.
        """
        symbol = event.symbol
        out: Dict[str, Any] = {
            'target_qty': None, 'skip_reason': None,
            'sigma': None, 'instrument_weight': None,
            'strategy_weight': None, 'annual_cash_target': None,
        }

        sigma = self.vol_estimator.get_annualized_vol(symbol)
        if sigma is None:
            out['skip_reason'] = 'warmup'
            return out
        out['sigma'] = sigma
        if sigma == 0:
            out['skip_reason'] = 'zero_vol'
            return out

        if symbol not in self.instrument_weight:
            # Not in the current tradable universe (liveness gate): no
            # weight assigned. ``instrument_weight`` stays None in the
            # diagnostic row — truthful, vs. recording a synthetic 0.0.
            out['skip_reason'] = 'not_live'
            return out
        iw = self.instrument_weight[symbol]
        sw = self.strategy_weight.get(self.strategy.__class__.__name__, 0.0)
        out['instrument_weight'] = iw
        out['strategy_weight'] = sw
        if iw * sw == 0:
            out['skip_reason'] = 'zero_weight'
            return out

        forecast = self.strategy.get_forecast(symbol)
        if self.vol_target_mode == 'percent_volatility':
            # Carver's original form: τ is a fraction of *current*
            # account equity, so the cash target compounds with the
            # account.
            capital = self.portfolio.calculate_balance()
            annual_cash_target = (
                capital * self.idm * sw * iw * self.annualized_target_vol
                * (forecast / Strategy.TARGET_AVG_ABS_FORECAST)
            )
        elif self.vol_target_mode == 'dollar_volatility':
            # Fixed annual $ vol budget — no capital term (institutional
            # futures convention: the risk limit is a dollar number, not
            # a compounding fraction of equity).
            annual_cash_target = (
                self.idm * sw * iw * self.annualized_target_vol
                * (forecast / Strategy.TARGET_AVG_ABS_FORECAST)
            )
        else:
            raise ValueError(
                f"Unexpected vol_target_mode: {self.vol_target_mode!r}"
            )
        out['annual_cash_target'] = annual_cash_target
        out['target_qty'] = annual_cash_target / sigma
        return out

    def _record_row(self, symbol: str, row: Dict[str, Any]) -> None:
        """Append the diagnostic row and emit the Carver DEBUG log line."""
        super()._record_row(symbol, row)
        action = 'submit' if row['submitted'] else row['skip_reason']
        logger.debug(
            "[CARVER] %s fc=%.2f sigma=%s iw=%s sw=%s cap=%.2f "
            "target=%s cur=%.6f trade=%s action=%s",
            symbol, row['forecast'], row['sigma'],
            row['instrument_weight'], row['strategy_weight'],
            row['capital'], row['target_qty'], row['current_qty'],
            row['trade_qty'], action,
        )
