"""Rolling-stdev annualized-vol estimator.

Per-symbol price-change series ``close - prior_close`` at the configured
``timeframe`` (default ``'1d'``), fed into the existing
``Stdev(length=lookback)`` indicator. Annualized vol =
``stdev * sqrt(bars_per_year)``. Used by
``CarverVolTargetingRiskManager`` to derive the dollar-volatility divisor
in the cash-vol position-sizing formula.

Closes are sourced from a ``DataHandler`` via
``get_latest_bars(symbol, 2, timeframe=self.timeframe)`` rather than from
the raw ``BarEvent`` close. This decouples sigma's time-scale from the
engine's base timeframe — running with ``base_timeframe='1h'`` and
``timeframe='1d'`` gives the same sigma as ``base_timeframe='1d'`` would,
because the estimator reads the same daily closes either way. The
underlying ``Stdev`` upserts on the forming HTF timestamp, so sigma only
advances at HTF period boundaries; within a period, repeated
``update`` calls overwrite the forming entry.

The estimator output carries the **units of the input series**: feeding
price changes (the standard wiring here) produces an annualized stdev in
price units (e.g. dollars). This generalizes cleanly to instruments where
percent-change is undefined or meaningless — futures spreads (price can
cross zero), instruments quoted in basis-point terms, synthetic legs.
Negative or zero prior closes are handled naturally: ``close - 0 = close``
is a valid price change, no singularity.

Per-symbol state (private):

- ``_stdev[symbol]`` — ``Stdev(length=lookback)`` instance fed one price
  change per completed bar.

Forming bars are ignored to avoid double-counting changes within a single
period (the live data handler emits forming bars on every tick).

The estimator is a helper to the risk manager and not driven by the
``Backtester`` event loop; the risk manager calls ``update(event)``
itself on every completed bar.
"""

import math
from typing import Dict, List, Optional

import pandas as pd

from event import BarEvent
from indicator import Stdev
from volatility._base import VolEstimator, _DataHandlerLike

__all__ = ['RollingStdevVolEstimator']


class RollingStdevVolEstimator(VolEstimator):
    """Rolling sample-stdev (ddof=1) of price changes, scaled by ``sqrt(bars_per_year)``."""

    def __init__(self, symbol_list: List[str], data_handler: _DataHandlerLike,
                 bars_per_year: float, timeframe: str = '1d',
                 lookback: int = 35):
        """Configure per-symbol state.

        Parameters
        ----------
        symbol_list
            Symbols to estimate vol for. Bars for other symbols are
            silently ignored by ``update``.
        data_handler
            Source of bar lookbacks; must expose ``get_latest_bars``. The
            estimator reads the latest 2 bars at ``timeframe`` on every
            completed ``BarEvent`` and computes the next price change
            from them.
        bars_per_year
            Annualization factor — must match ``timeframe``. For the
            ``'calendar'`` days convention (24/7 markets): ``1d`` → 365,
            ``4h`` → ``365 * 6``, ``1h`` → ``365 * 24``. For
            ``'business'``: ``1d`` → 252, ``4h`` → ``252 * 6``, etc.
        timeframe
            Timeframe at which to read closes from ``data_handler``.
            Default ``'1d'``. Must be a timeframe registered with
            ``data_handler`` (otherwise ``get_latest_bars`` raises on
            the first ``update`` call).
        lookback
            Window for the rolling stdev. Default ``35`` — Carver's
            daily-equivalent vol estimate (Systematic Trading p.287
            footnote 161).

        Raises
        ------
        ValueError
            If ``lookback < 2`` (sample stdev needs at least 2 obs) or
            ``bars_per_year <= 0``.
        """
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if bars_per_year <= 0:
            raise ValueError(f"bars_per_year must be > 0, got {bars_per_year}")

        self.symbol_list = list(symbol_list)
        self.data_handler = data_handler
        self.timeframe = timeframe
        self.lookback = lookback
        self.bars_per_year = bars_per_year
        self._sqrt_bpy = math.sqrt(bars_per_year)

        self._stdev: Dict[str, Stdev] = {
            s: Stdev(length=lookback) for s in self.symbol_list
        }

    def update(self, event: BarEvent) -> None:
        """Push one price change at ``self.timeframe`` per completed bar;
        skip forming bars.

        Reads the latest 2 bars at ``self.timeframe`` from the data
        handler and pushes ``forming_close - prev_close`` into the
        underlying ``Stdev`` with the forming HTF timestamp. The
        indicator upserts on that timestamp, so within a single HTF
        period repeated calls overwrite the same forming output entry;
        ``.latest`` (the last finalized stdev) only advances at period
        boundaries.

        Skips when the data handler doesn't yet have 2 bars at
        ``self.timeframe`` for the symbol (still warming up the HTF
        deque).
        """
        if event.is_forming:
            return
        symbol = event.symbol
        if symbol not in self._stdev:
            return
        bars = self.data_handler.get_latest_bars(
            symbol, 2, timeframe=self.timeframe,
        )
        if len(bars) < 2:
            return
        forming_ts = bars.index[-1]
        forming_close = float(bars['Close'].iloc[-1])
        prev_close = float(bars['Close'].iloc[-2])
        price_change = forming_close - prev_close
        self._stdev[symbol].update(forming_ts, price_change)

    def get_annual_vol(self, symbol: str) -> Optional[float]:
        """Return ``stdev.latest['stdev'] * sqrt(bars_per_year)``, or
        ``None`` if no finalized non-NaN stdev is available yet (warmup).

        The returned value carries the units of the input series — price
        units (e.g. dollars) under the standard wiring.

        Reads the last finalized stdev (``.latest``) rather than the
        forming entry, matching the convention used by strategies that
        consume indicator outputs. ``.latest`` reflects the previous
        completed HTF bar's stdev (the forming entry that this bar just
        upserted becomes ``.latest`` once the next HTF period begins).
        """
        stdev = self._stdev.get(symbol)
        if stdev is None:
            return None
        latest = stdev.latest
        if latest is None:
            return None
        sigma = float(latest['stdev'])
        if pd.isna(sigma):
            return None
        return sigma * self._sqrt_bpy
