"""
Mode-agnostic strategy base for the event-driven trading system.

Exposes the ``Strategy`` ABC — the shared machinery every strategy
template builds on: the per-symbol forecast cache (clamped to ±100 via
``_commit_forecast_row``), the measured warmup flag, per-bar record
keeping, and the project-wide forecast constants. ``update_bar()`` is
abstract here: each *template* defines how engine bars become
forecasts. The per-event timeseries template lives in
``strategy._timeseries`` (``TimeSeriesStrategy`` — one symbol's bar in,
that symbol's forecast out); a future cross-sectional template
(cross-symbol alignment, batch signal generation) will be a sibling
subclass honoring the same contract.

The forecast dictionary is the strategy's only output. The risk manager
reads ``strategy.get_forecast(symbol)`` on every completed bar to derive
the target position. Strategies are pure "forecast oracles" — they
never enqueue events.

Concrete example strategies live in sibling modules (e.g. ``ewmac``).
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from scipy.stats import norm

from event import BarEvent

__all__ = ['Strategy']

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Structural types for dependencies
# ──────────────────────────────────────────────

class _DataHandlerLike(Protocol):
    """Minimal interface ``Strategy`` subclasses need from the data handler.

    Only ``get_latest_bars`` is consumed — the last *n* ``Bar`` objects at the
    execution timeframe (a plain list, no DataFrame; see
    ``DataHandler.get_latest_bars`` / ``get_latest_bars_df``).
    """
    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> List[Any]: ...


# ──────────────────────────────────────────────
# Strategy ABC
# ──────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract, mode-agnostic base class for trading strategies.

    Owns the machinery every strategy template shares: the per-symbol
    forecast cache, forecast clamping and warmup tracking (via
    ``_commit_forecast_row``), per-bar record keeping, and the
    project-wide forecast constants. ``update_bar()`` is abstract —
    concrete strategies subclass a *template* that implements it:
    ``TimeSeriesStrategy`` (``strategy._timeseries``) for per-event
    single-symbol strategies today; a future cross-sectional template
    will be its sibling.

    Project-wide forecast convention (class constants below):
        ``-FORECAST_CAP`` = max short conviction, ``0`` = flat,
        ``+FORECAST_CAP`` = max long conviction.

    ``TARGET_AVG_ABS_FORECAST`` is the average ``|forecast|`` value each
    strategy is expected to calibrate toward. ``VolTargetingRiskManager``
    divides the forecast by ``TARGET_AVG_ABS_FORECAST`` so that
    ``|forecast| = TARGET_AVG_ABS_FORECAST`` reproduces Carver's basic
    vol-target notional and ``|forecast| = FORECAST_CAP`` doubles it.
    The relationship ``FORECAST_CAP == 2 * TARGET_AVG_ABS_FORECAST`` is
    by design.
    """

    # Project-wide forecast convention. Treated as constants — do not override
    # in subclasses; both ``Strategy.update_bar`` and
    # ``VolTargetingRiskManager`` read them via the class.
    TARGET_AVG_ABS_FORECAST: float = 50.0
    FORECAST_CAP: float = 100.0

    def __init__(self, data_handler: _DataHandlerLike,
                 symbol_list: List[str]):
        """
        Bind dependencies and initialise per-symbol state.

        Parameters
        ----------
        data_handler
            Source of bar lookbacks; must expose ``get_latest_bars``.
        symbol_list
            Symbols this strategy acts on. Bars for other symbols are
            ignored by ``update_bar``.
        """
        self.data_handler = data_handler
        self.symbol_list = symbol_list
        # Per-symbol cached forecast in [-100, +100]. Default None means
        # "no forecast cached yet" (warmup) — distinct from a genuine flat
        # forecast of 0.0. The risk manager reads this dict on every
        # completed bar via get_forecast().
        self.forecasts: Dict[str, Optional[float]] = {s: None for s in symbol_list}
        # Per-symbol warmup flag: False until the first non-NaN forecast
        # is cached for the symbol, then True forever (monotone). This is
        # the *measured* end-of-warmup signal the risk manager's universe
        # liveness gate consumes via ``is_warmed_up`` — no declared
        # bar-count estimate needed.
        self._warmed_up: Dict[str, bool] = {s: False for s in symbol_list}
        # Per-symbol list of row dicts, populated by update_bar() on each bar.
        self._records: Dict[str, List[Dict]] = defaultdict(list)

    @abstractmethod
    def update_bar(self, event: BarEvent) -> None:
        """Consume one engine-dispatched ``BarEvent`` (template hook).

        Engine contract every template must honor:

        - called on EVERY bar event, forming and completed alike;
        - must ignore events whose symbol is outside ``symbol_list``;
        - must be idempotent under forming-bar re-runs (compute from
          finalized values so re-processing the same forming period
          cannot change cached state);
        - commits results via ``_commit_forecast_row`` so clamping,
          caching, warmup flags, and per-bar records stay uniform
          across templates.

        ``TimeSeriesStrategy`` implements the per-event timeseries
        template; a future cross-sectional template will implement an
        alignment-aware version.
        """
        raise NotImplementedError

    def _commit_forecast_row(self, symbol: str, base_row: Dict[str, Any],
                             extras: Optional[Dict[str, Any]]) -> None:
        """Merge ``extras`` into ``base_row``, clamp/cache the forecast, record.

        The shared back half of every template's bar handling: if ``extras``
        carries a ``'forecast'`` that is neither ``None`` nor NaN, clamp it
        to ``±FORECAST_CAP``, write it to ``self.forecasts[symbol]``, and
        flip the symbol's warmup flag; NaN forecasts (warmup) are recorded
        as-is for diagnostics but leave the cached value untouched. The
        merged row is then appended to the per-symbol record buffer.
        ``extras=None`` records ``base_row`` (OHLCV-only) and changes no
        cached state. Never mutates the caller's ``extras`` dict.

        Templates call this once per committed result — the timeseries
        template once per bar event; a future cross-sectional template once
        per symbol per aligned timestamp.
        """
        if extras is not None:
            extras = dict(extras)              # don't mutate caller's dict
            if 'forecast' in extras:
                raw = extras['forecast']
                if raw is not None and not pd.isna(raw):
                    cap = Strategy.FORECAST_CAP
                    clamped = max(-cap, min(cap, float(raw)))
                    self.forecasts[symbol] = clamped
                    extras['forecast'] = clamped
                    # First real forecast ⇒ warmup is over for this symbol.
                    self._warmed_up[symbol] = True
            base_row.update(extras)

        self._record_row(symbol, base_row)

    def get_forecast(self, symbol: str) -> Optional[float]:
        """Return the cached forecast for ``symbol``, or ``None`` if none yet.

        ``None`` means no real forecast has been cached (warmup) — distinct
        from a genuine flat forecast of ``0.0``. The risk manager calls this
        on every completed bar to derive the target position; its liveness
        gate keeps weight off un-warmed symbols, so the sizing arithmetic
        never sees ``None``. Mirrors ``VolEstimator.get_annual_vol`` which
        likewise returns ``None`` while warming up. Unknown symbols return
        ``None``.
        """
        return self.forecasts.get(symbol)

    def is_warmed_up(self, symbol: str) -> bool:
        """Return True once ``symbol`` has produced its first non-NaN forecast.

        Measured, not estimated: the flag flips inside ``update_bar`` at
        the exact moment the first real forecast is written to the cache,
        and never resets (monotone). The risk manager's universe liveness
        gate reads this to keep instrument weight away from symbols whose
        strategy cannot trade yet (e.g. indicator chains still warming
        up). Unknown symbols return ``False``.
        """
        return self._warmed_up.get(symbol, False)

    @staticmethod
    def sizing_with_probability(p: float, num_classes: int = 2) -> float:
        """
        Lopez de Prado's signed bet-size formula (Ch. 10 of *Advances in
        Financial Machine Learning*) for a one-vs-rest classifier with the
        predicted side embedded externally.

        Returns a scalar in ``[-1.0, 1.0]`` — the canonical probability
        mapping; the project-wide forecast scale is ``[-100, +100]``, so
        callers multiply by the predicted side (+1 long, -1 short) AND by
        100 to convert to forecast units. With ``num_classes=2`` (binary
        default) the result is non-negative for ``p >= 0.5``. For
        ``p < 1/num_classes`` the value is negative — meaningful in a
        multi-class OvR setting where the predicted class is below random.

        Parameters
        ----------
        p
            Probability of the predicted class.
        num_classes
            Number of classes in the underlying classifier. Default ``2``
            (binary). The pivot at which ``z = 0`` is ``1/num_classes``.

        Notes
        -----
        Formula:
            z = (p - 1/num_classes) / sqrt(p * (1 - p))
            m = 2 * Φ(z) - 1
        where ``Φ`` is the standard-normal CDF, computed via
        ``scipy.stats.norm.cdf``. The boundary cases ``p >= 1.0`` → ``1``
        and ``p <= 0.0`` → ``-1`` are handled explicitly to avoid a
        divide-by-zero (or complex-valued sqrt) in ``z``.
        """
        if p >= 1.0:
            return 1.0
        if p <= 0.0:
            return -1.0
        z = (p - 1.0 / num_classes) / (p * (1.0 - p)) ** 0.5
        return 2.0 * float(norm.cdf(z)) - 1.0

    def _record_row(self, symbol: str, row: Dict) -> None:
        """Append one per-bar row dict to the per-symbol record buffer."""
        self._records[symbol].append(row)

    def get_records(self, symbol: str) -> pd.DataFrame:
        """
        Return recorded rows for a symbol as a DataFrame indexed by timestamp.

        Returns an empty DataFrame if no rows have been recorded yet.
        """
        if not self._records[symbol]:
            return pd.DataFrame()
        df = pd.DataFrame(self._records[symbol])
        df.set_index('timestamp', inplace=True)
        return df
