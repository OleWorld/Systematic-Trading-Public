"""VolEstimator ABC for the forecast-aware risk manager.

Concrete estimators (rolling stdev, EWMA, Yang-Zhang, GARCH, …) implement
``update`` to observe each completed bar and ``get_annualized_vol`` to
return the current annualized volatility per symbol. The risk manager
calls these on every completed bar to derive the dollar-volatility
divisor in the cash-vol position-sizing formula.

``VolEstimator`` is a helper to the risk manager — not an engine-level
bar consumer. The ``Backtester`` event loop never calls it directly;
the risk manager owns it and drives ``update(event)`` itself. The
method name is ``update`` (mirroring ``Indicator.update``) rather than
``update_bar`` (which is reserved for the four modules the engine
drives directly: portfolio, execution, strategy, risk_manager).

Estimators source closes from a ``DataHandler`` at a configurable
``timeframe`` (default ``'1d'``) — decoupled from the engine's base
timeframe so the sigma is on a stable time-scale regardless of how
often ``update`` is called. This mirrors how ``Strategy`` reads
``data_handler.get_latest_bars(symbol, n, timeframe=...)`` for its own
HTF lookbacks. Estimators read ``forming_close - prev_finalized_close``
at the configured timeframe and feed that price change into their
underlying stdev indicator (which upserts on the forming timestamp, so
sigma only advances at HTF period boundaries).

``get_annualized_vol`` returns an annualized stdev of whatever input
series the implementation feeds in. The standard estimators here feed in
price changes (``close - prior_close``), so the returned value is in
price units — used by the Carver risk manager as the annualized $-vol of
the instrument.

Estimators must never propagate ``NaN``: they return ``None`` while warming
up and a clean ``0.0`` if the underlying volatility is genuinely zero.
``None`` is a "not yet" signal that the risk manager treats as "skip
sizing this bar"; ``0.0`` would crash the divide and so the risk manager
also skips on it.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import pandas as pd

from data import parse_timeframe_to_seconds
from event import BarEvent

__all__ = ['VolEstimator', 'bars_per_year', '_DataHandlerLike']


_SECONDS_PER_DAY = 24 * 3600
_DAYS_PER_YEAR = {
    'crypto': 365,    # 24/7 markets
    'tradfi': 252,    # standard equity/futures trading-day convention
}


def bars_per_year(timeframe: str, convention: str) -> float:
    """Return bars-per-year for a timeframe under the chosen convention.

    Used as the annualization factor for ``VolEstimator`` subclasses:
    ``annualized_vol = stdev * sqrt(bars_per_year(timeframe, convention))``.

    The two supported conventions:

    - ``'crypto'`` (365 days/year, 24/7 markets). ``1d`` → 365,
      ``4h`` → 365*6, ``1h`` → 365*24, ``1m`` → 365*1440.
    - ``'tradfi'`` (252 trading days/year). ``1d`` → 252,
      ``4h`` → 252*6, ``1h`` → 252*24, ``1m`` → 252*1440.

    The intraday formulas hold full-time-trading hours (24/day) into the
    bar count for both conventions; for daily data the result is exactly
    the trading-days-per-year number, which is the load-bearing case for
    most systematic strategies. Any timeframe that
    ``parse_timeframe_to_seconds`` recognises is supported.

    Raises
    ------
    ValueError
        If ``convention`` is not ``'crypto'`` or ``'tradfi'``.
    """
    if convention not in _DAYS_PER_YEAR:
        raise ValueError(
            f"Unknown convention: '{convention}'. "
            "Must be 'crypto' (365 days/year) or 'tradfi' (252 days/year)."
        )
    seconds_per_year = _DAYS_PER_YEAR[convention] * _SECONDS_PER_DAY
    return seconds_per_year / parse_timeframe_to_seconds(timeframe)


class _DataHandlerLike(Protocol):
    """Minimal interface vol estimators need from the data handler.

    Mirrors ``strategy._base._DataHandlerLike`` — only ``get_latest_bars``
    is consumed (the estimator reads the last two closes at its
    configured timeframe).
    """

    def get_latest_bars(self, symbol: str, n: int,
                        timeframe: Optional[str] = None) -> pd.DataFrame: ...


class VolEstimator(ABC):
    """Abstract base class for symbol-keyed annualized-vol estimators."""

    @abstractmethod
    def update(self, event: BarEvent) -> None:
        """Observe one completed bar.

        Implementations should ignore ``event.is_forming=True`` bars to
        avoid double-counting observations within a forming period (the
        live data handler emits a forming bar on every tick).
        """
        raise NotImplementedError

    @abstractmethod
    def get_annualized_vol(self, symbol: str) -> Optional[float]:
        """Return the current annualized volatility for ``symbol``.

        Output carries the units of the input series the implementation
        feeds in (price units for the standard estimators here, which
        feed in price changes).

        ``None`` while warming up (insufficient observations to produce a
        finalized estimate). ``0.0`` if the underlying estimate is genuinely
        zero (constant inputs). Never returns ``NaN``.
        """
        raise NotImplementedError
