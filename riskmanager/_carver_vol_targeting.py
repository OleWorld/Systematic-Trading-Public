"""CarverVolTargetingRiskManager — forecast-aware Carver vol-targeting sizer.

Implements Carver's vol-targeting framework (Systematic Trading, Ch. 10)
in **cash-vol** form. The risk target is a dollar amount of vol per
period; the instrument's dollar vol per period divides that target to
give the size:

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
    annualized_target_vol  = annualized vol target                      (constructor; e.g. 0.25 = 25 %)
    annualized_price_vol   = annualized stdev of price changes ($-units)  (VolEstimator)
    forecast               = strategy.get_forecast(symbol) ∈ [-FORECAST_CAP, +FORECAST_CAP]

The two equalities for ``target_qty`` are algebraically equivalent: the
``sqrt(days_per_year)`` factors in the daily-cash and daily-price-vol
forms cancel, so we implement the cleaner annualized form (no need to
plumb ``days_per_year`` / convention into the risk manager). The
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

With ``instrument_weight_mode='min_variance'`` the manager performs
**walk-forward** weight estimation: at each recalc point it pulls
``corr_lookback`` bars (default ``500``) at ``corr_timeframe`` (default
``'1d'``) for every symbol via the data handler, computes per-bar price
changes per ``corr_mode`` (``'simple_return'`` → ``.pct_change()``,
default; ``'absolute_price_chg'`` → ``.diff()`` — for futures/spreads
whose prices can be negative or zero, where percentage returns are
meaningless), and derives ρ → weights via the existing
min-variance formula. The matching IDM (``analytics.diversification_multiplier``)
is updated from the same ρ on every successful recompute, keeping the
two coherent. ``update_bar`` auto-recalls the method every
``corr_step_size`` completed ``corr_timeframe`` periods (default ``30``;
multi-symbol bars at the same timestamp and sub-period base bars are
de-duplicated via ``data.get_period_start``); set ``corr_step_size=0``
to disable auto-recalc. When the deque holds
fewer than 30 valid return observations — always the case at
construction time — the manager logs a WARNING and falls back to
equal-weight (the ρ=1 / risk-parity degenerate case); ``self.idm``
is left untouched in this branch.

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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from analytics import correlation_matrix, diversification_multiplier
from data import get_period_start
from event import BarEvent, OrderType, Direction
from riskmanager._base import (
    RiskManager, _DataHandlerLike, _PortfolioLike, _StrategyLike,
)
from strategy import Strategy
from volatility import VolEstimator

logger = logging.getLogger(__name__)

# Minimum number of valid return observations (rows surviving
# ``pct_change().dropna()``) required for a stable Pearson correlation
# estimate. Below this, ``calculate_instrument_weight`` logs a WARNING
# and falls back to equal-weight (the ρ=1 / risk-parity degenerate case).
_MIN_CORR_OBS = 30


class CarverVolTargetingRiskManager(RiskManager):
    """Forecast-aware cash-vol-targeting sizer (Carver's framework).

    Owns two weight dicts:

    - ``instrument_weight``: per-symbol capital weight, populated at
      construction by ``calculate_instrument_weight()`` (default 1/N
      across ``strategy.symbol_list``; with ``mode='min_variance'``
      derived inline from a trailing window of returns pulled from
      ``self.data_handler``).
    - ``strategy_weight``: per-strategy capital weight, populated at
      construction by ``calculate_strategy_weight()`` (default
      ``{strategy_class_name: 1.0}`` — placeholder until multi-strategy
      lands).

    Either ``calculate_*`` method can be re-called or its dict
    overwritten directly to refresh weights without per-bar wiring.
    With ``mode='min_variance'`` and ``corr_step_size > 0``,
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
        annualized_target_vol: float = 0.25,
        position_buffer: float = 0.25,
        instrument_weight_mode: str = 'equal_weight',
        corr_lookback: int = 500,
        corr_step_size: int = 30,
        corr_timeframe: str = '1d',
        corr_mode: str = 'simple_return',
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
            Annualized volatility target (Carver's ``τ``). Default
            ``0.25`` (25 %, Carver's default for futures). Must be in
            ``(0, 1)``.
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
            constructor). One of ``'equal_weight'`` (default) or
            ``'min_variance'``. With ``'min_variance'``, the construction-
            time call derives the corr matrix from the data handler;
            cold/short deques degrade to equal-weight + WARNING rather
            than raising.
        corr_lookback
            Trailing window in ``corr_timeframe`` bars used to pull
            closes for the inline correlation derivation. Default ``500``.
            Must be ``>= 2``.
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
            auto-recalc (one-shot at ``__init__`` only). Has no effect
            when ``instrument_weight_mode='equal_weight'``.
        corr_timeframe
            Timeframe the data handler reads when assembling the closes
            window. Default ``'1d'``. Must be a key of
            ``data_handler.timeframes``.
        corr_mode
            How per-bar price changes are computed from the closes window
            before correlating. One of ``'simple_return'`` (default,
            ``.pct_change()``) or ``'absolute_price_chg'`` (``.diff()``).
            Use ``'absolute_price_chg'`` for futures contracts and
            synthetic products (e.g. time spreads) whose prices can be
            negative or zero — simple returns are meaningless there
            (inf at zero crossings, sign-flipped below zero).

        Raises
        ------
        ValueError
            On invalid constructor parameters.
        """
        if idm <= 0:
            raise ValueError(f"idm must be > 0, got {idm}")
        if not (0 < annualized_target_vol < 1):
            raise ValueError(
                f"annualized_target_vol must be in (0, 1), got {annualized_target_vol}"
            )
        if not (0.0 <= position_buffer < 1.0):
            raise ValueError(
                f"position_buffer must be in [0, 1), got {position_buffer}"
            )
        if corr_lookback < 2:
            raise ValueError(f"corr_lookback must be >= 2, got {corr_lookback}")
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
        if corr_mode not in ('simple_return', 'absolute_price_chg'):
            raise ValueError(
                f"Unknown corr_mode: {corr_mode!r}. "
                "Must be 'simple_return' or 'absolute_price_chg'."
            )
        super().__init__(portfolio, strategy)
        self.vol_estimator = vol_estimator
        self.data_handler = data_handler
        self.idm = idm
        self.annualized_target_vol = annualized_target_vol
        self.position_buffer = position_buffer
        self.corr_lookback = corr_lookback
        self.corr_step_size = corr_step_size
        self.corr_timeframe = corr_timeframe
        self.corr_mode = corr_mode
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

    def calculate_instrument_weight(
        self,
        mode: Optional[str] = None,
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> None:
        """Populate ``self.instrument_weight`` according to ``mode``.

        Parameters
        ----------
        mode
            Weighting scheme. When ``None`` (default), falls back to
            ``self.instrument_weight_mode`` (set in the constructor;
            default ``'equal_weight'``). Otherwise one of:

            * ``'equal_weight'`` — ``{symbol: 1/N}`` across
              ``self.strategy.symbol_list``. ``corr_matrix`` is ignored
              if passed.
            * ``'min_variance'`` — minimum-variance weights under the
              equal-volatility assumption, derived from the inverse
              correlation matrix:

                  w = (ρ⁻¹ · 1) / (1ᵀ ρ⁻¹ 1)

              Negative raw weights are clipped to ``0`` and the
              survivors are renormalized to sum to ``1`` (Carver
              long-only convention). When ``corr_matrix`` is omitted,
              ρ is derived inline from a trailing window of per-bar
              price changes (per ``self.corr_mode``) pulled from
              ``self.data_handler`` (see below).
        corr_matrix
            Correlation matrix of per-symbol price returns, indexed and
            columned by the same labels as ``self.strategy.symbol_list``
            (set-equal; row order is taken from the matrix). Optional
            for ``mode='min_variance'``: when ``None``, the method pulls
            ``self.corr_lookback`` bars at ``self.corr_timeframe`` for
            each symbol via ``self.data_handler.get_latest_bars``,
            computes per-bar price changes per ``self.corr_mode``
            (``'simple_return'`` → ``.pct_change().dropna()``;
            ``'absolute_price_chg'`` → ``.diff().dropna()``), and
            calls ``analytics.correlation_matrix`` on the result. If
            fewer than ``_MIN_CORR_OBS`` (=30) valid observations
            survive — always the case at construction time in a backtest
            since deques are empty — the method logs a WARNING and
            falls back to equal-weight (the ρ=1 / risk-parity
            degenerate case). Ignored for ``mode='equal_weight'``.

        Raises
        ------
        ValueError
            On unknown ``mode``; ``corr_matrix`` (passed or derived) whose
            index does not equal ``corr_matrix.columns`` or does not
            match ``self.strategy.symbol_list`` as a set; or degenerate
            min-variance weights that all clip to zero.

        Notes
        -----
        Mutates ``self.instrument_weight`` in place and, on successful
        min-variance computes, also updates ``self.idm`` via
        ``analytics.diversification_multiplier(...)`` so weights and IDM
        stay coherent. The equal-weight fallback path leaves ``self.idm``
        untouched. Safe to re-call any time (e.g. monthly rebalances,
        regime-driven scheme switches); ``update_bar`` re-calls this
        method every ``corr_step_size`` completed bars when min-variance
        is active.
        """
        if mode is None:
            mode = self.instrument_weight_mode
        if mode == 'equal_weight':
            syms = self.strategy.symbol_list
            n = len(syms)
            self.instrument_weight = {s: 1.0 / n for s in syms}
        elif mode == 'min_variance':
            if corr_matrix is None:
                # Derive ρ from a trailing window of simple returns pulled
                # from the data handler. Empty/short deques degrade to
                # equal-weight + WARNING (equivalent to assuming ρ=1).
                closes = {
                    s: self.data_handler.get_latest_bars(
                        s, self.corr_lookback,
                        timeframe=self.corr_timeframe,
                    )['Close']
                    for s in self.strategy.symbol_list
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
                        "min_variance: only %d valid return observations "
                        "(need >= %d for stable correlation estimate); "
                        "falling back to equal-weight "
                        "(rho=1 / risk-parity under Carver's equal-vol assumption)",
                        len(returns), _MIN_CORR_OBS,
                    )
                    syms = self.strategy.symbol_list
                    n = len(syms)
                    self.instrument_weight = {s: 1.0 / n for s in syms}
                    return
                corr_matrix = correlation_matrix(returns)
            if not corr_matrix.index.equals(corr_matrix.columns):
                raise ValueError(
                    "corr_matrix.index must equal corr_matrix.columns "
                    "(labels and order)"
                )
            expected = set(self.strategy.symbol_list)
            got = set(corr_matrix.index)
            if got != expected:
                missing = expected - got
                extra = got - expected
                raise ValueError(
                    f"corr_matrix labels must equal strategy.symbol_list "
                    f"as a set; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            inv = np.linalg.inv(corr_matrix.to_numpy())
            raw = inv.sum(axis=1) / inv.sum()
            clipped = np.clip(raw, 0.0, None)
            total = float(clipped.sum())
            if total <= 0:
                raise ValueError(
                    "min_variance produced all-zero weights after clipping; "
                    "review the correlation matrix"
                )
            normalized = clipped / total
            self.instrument_weight = {
                label: float(w)
                for label, w in zip(corr_matrix.index, normalized)
            }
            # Auto-update IDM from the same matrix used for weights so
            # the two stay coherent across walk-forward recomputes.
            self.idm = diversification_multiplier(
                self.instrument_weight, corr_matrix,
            )
        else:
            raise ValueError(
                f"Unexpected mode: {mode!r} "
                "(expected 'equal_weight' or 'min_variance')"
            )

    def calculate_strategy_weight(self) -> None:
        """Populate ``self.strategy_weight`` with the single-strategy placeholder.

        ``{strategy_class_name: 1.0}``. Will become a real ``1/M`` (or
        correlation-driven) allocation once the risk manager holds
        multiple strategies. Mutates in place; safe to re-call any time.
        """
        name = self.strategy.__class__.__name__
        self.strategy_weight = {name: 1.0}

    def update_bar(self, event: BarEvent) -> None:
        """Update sizing inputs and resize the position to the Carver target.

        Skips forming bars (one resize per completed bar). Delegates
        target-qty derivation (and *target-derivation* skip reasons
        ``'warmup'`` / ``'zero_vol'`` / ``'zero_weight'``) to
        ``_compute_target_qty``; owns *post-target* concerns
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
        # and sub-period base bars don't multi-increment. The actual
        # matrix work runs every ``corr_step_size`` periods and only when
        # min-variance is active. Counter resets whether the recompute
        # succeeded or fell back to equal-weight (cold-deque path) — by
        # the next attempt the deque has accumulated more bars.
        if (
            self.corr_step_size > 0
            and self.instrument_weight_mode == 'min_variance'
        ):
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

        target_qty = (capital × IDM × strategy_weight × instrument_weight
                      × annualized_target_vol × (forecast / TARGET_AVG_ABS_FORECAST))
                     / annualized_price_vol

        Owns the *target-derivation* skip ladder (``'warmup'`` /
        ``'zero_vol'`` / ``'zero_weight'``). The returned dict is
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

        iw = self.instrument_weight.get(symbol, 0.0)
        sw = self.strategy_weight.get(self.strategy.__class__.__name__, 0.0)
        out['instrument_weight'] = iw
        out['strategy_weight'] = sw
        if iw * sw == 0:
            out['skip_reason'] = 'zero_weight'
            return out

        capital = self.portfolio.calculate_balance()
        forecast = self.strategy.get_forecast(symbol)
        annual_cash_target = (
            capital * self.idm * sw * iw * self.annualized_target_vol
            * (forecast / Strategy.TARGET_AVG_ABS_FORECAST)
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
