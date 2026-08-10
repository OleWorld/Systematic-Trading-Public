"""
Timeseries strategy template for the event-driven trading system.

Exposes ``TimeSeriesStrategy`` — the per-event template for strategies
that read ONE symbol's bars and generate a forecast for that symbol
alone (e.g. ``ewmac``, ``rsimr``). ``update_bar()`` filters the event
symbol, builds the OHLCV row, delegates the strategy-specific math to
``calculate_forecast()`` (subclass hook), and commits the result through
``Strategy._commit_forecast_row`` (clamp → cache → warmup flip →
record).

Cross-sectional strategies (cross-symbol bar alignment, batch signal
generation) are a different template: they will subclass ``Strategy``
directly as a sibling of this class, honoring the same forecast-oracle
contract. See ``strategy._base`` for the shared machinery.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

from event import BarEvent
from strategy._base import Strategy

__all__ = ['TimeSeriesStrategy']


class TimeSeriesStrategy(Strategy):
    """
    Per-event timeseries strategy template.

    The backtester calls ``update_bar()`` on each ``BarEvent``. ``update_bar``
    handles symbol filtering, OHLCV recording, and delegates to
    ``calculate_forecast()`` for the strategy-specific math — one symbol's
    bar in, that symbol's forecast out, immediately.

    Subclasses implement ``calculate_forecast()`` — pure forecast
    computation. Use ``self.data_handler.get_latest_bars(symbol, n)`` for
    lookback data. Return a dict of fields to record. If the dict contains
    the ``'forecast'`` key, the value (after clamping to
    ``[-FORECAST_CAP, +FORECAST_CAP]``) is written to
    ``self.forecasts[symbol]`` and recorded in the per-bar log. Returning
    ``None`` records OHLCV-only and leaves the cached forecast unchanged.
    """

    def update_bar(self, event: BarEvent) -> None:
        """Process a BarEvent: filter symbol, run forecast logic, record row."""
        if event.symbol not in self.symbol_list:
            return

        base_row = {
            'timestamp': event.timestamp,
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'volume': event.volume,
        }

        extras = self.calculate_forecast(event)
        self._commit_forecast_row(event.symbol, base_row, extras)

    @abstractmethod
    def calculate_forecast(self, event: BarEvent) -> Optional[Dict[str, Any]]:
        """
        Implement forecast computation.

        Called by ``update_bar()`` for each ``BarEvent`` whose symbol is in
        ``symbol_list``. Return a dict of strategy-specific fields to record
        (indicators, intermediate values, and crucially ``'forecast'`` —
        the signed conviction in ``[-FORECAST_CAP, +FORECAST_CAP]``).
        Return ``None`` to record OHLCV only and leave the cached forecast
        unchanged (e.g. during warmup before any forecast can be computed).

        Do not include OHLCV keys (open, high, low, close, volume, timestamp)
        in the return — the template merges those in automatically.
        """
        raise NotImplementedError
