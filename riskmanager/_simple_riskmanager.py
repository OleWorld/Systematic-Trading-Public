"""SimpleRiskManager — simple forecast-following sizer.

Execution-mode agnostic: the same class is used for backtesting and live.
For calibrated continuous forecasts (e.g. EWMAC) where conviction should
modulate position size, prefer ``CarverVolTargetingRiskManager``.
"""

from typing import Any, Dict

from event import BarEvent, OrderType, Direction
from riskmanager._base import RiskManager, _PortfolioLike, _StrategyLike


class SimpleRiskManager(RiskManager):
    """
    Simple forecast-following sizer.

    Forecast sign drives direction:
        forecast > 0  → target long
        forecast < 0  → target short
        forecast == 0 → flatten

    Forecast magnitude is ignored; the position notional is set by the
    configured sizing mode:
        ``'fixed_notional'``   — target_notional = position_size
        ``'fixed_quantity'``   — target_qty = position_size  (price-independent)
        ``'fixed_equity_pct'`` — target_notional = equity * position_size

    On every completed bar (``event.is_forming = False``), the manager
    reads ``strategy.get_forecast(symbol)``, computes the target
    quantity via ``_compute_target_qty``, and submits a MKT order for
    the difference against the current realized position. If the
    realized position already matches the target (within ``1e-9``), no
    order is submitted.

    Per-bar diagnostic log analogous to ``Strategy.get_records``:
    every completed bar appends one row to ``self._records[symbol]``.
    Columns: ``forecast``, ``price``, ``size_mode``, ``position_size``,
    ``target_qty``, ``current_qty``, ``trade_qty``, ``submitted``
    (bool), and ``skip_reason`` ∈ ``{None, 'no_price', 'at_target'}``.
    Read via ``risk_manager.get_records(symbol)``.

    For calibrated continuous forecasts (e.g. EWMAC), use
    ``CarverVolTargetingRiskManager`` instead — it scales the notional
    by ``forecast / 50`` so conviction translates into position size.
    """

    _MODES = ('fixed_notional', 'fixed_quantity', 'fixed_equity_pct')

    def __init__(self, portfolio: _PortfolioLike, strategy: _StrategyLike,
                 size_mode: str = 'fixed_notional',
                 position_size: float = 10_000.0):
        """
        Parameters
        ----------
        portfolio
            Portfolio instance providing price, balance, positions, and
            ``submit_order``. Margin checking is the portfolio's
            responsibility, not the risk manager's.
        strategy
            Strategy instance exposing ``get_forecast(symbol)``. Read on
            every completed bar to derive the target position.
        size_mode
            One of ``'fixed_notional'``, ``'fixed_quantity'``,
            ``'fixed_equity_pct'``. Validated at construction.
        position_size
            Magnitude interpreted per ``size_mode``.
        """
        if size_mode not in self._MODES:
            raise ValueError(
                f"Unknown size_mode: '{size_mode}'. "
                f"Must be one of {self._MODES}."
            )
        super().__init__(portfolio, strategy)
        self.size_mode = size_mode
        self.position_size = position_size

    def update_bar(self, event: BarEvent) -> None:
        """Resize the position to match the strategy's current forecast.

        Skips forming bars (idempotent across intra-period ticks).
        Delegates target-qty derivation (and the ``'no_price'`` skip)
        to ``_compute_target_qty``; owns the post-target
        ``'at_target'`` check and the submit call. Records one
        diagnostic row per *completed* bar — including early-exit
        branches — into ``self._records[symbol]`` via ``_record_row``.
        """
        if event.is_forming:
            return

        symbol = event.symbol
        forecast = self.strategy.get_forecast(symbol)
        current_qty = self.portfolio.positions.get(symbol, 0.0)

        # Seed the diagnostic row with always-known inputs;
        # _compute_target_qty supplies price / target_qty / skip_reason
        # via row.update.
        row: Dict[str, Any] = {
            'timestamp': event.timestamp,
            'symbol': symbol,
            'forecast': forecast,
            'price': None,
            'size_mode': self.size_mode,
            'position_size': self.position_size,
            'target_qty': None,
            'current_qty': current_qty,
            'trade_qty': None,
            'submitted': False,
            'skip_reason': None,
        }
        row.update(self._compute_target_qty(event))

        if row['skip_reason'] is not None:
            self._record_row(symbol, row)
            return

        target_qty = row['target_qty']
        trade_qty = target_qty - current_qty
        row['trade_qty'] = trade_qty

        if abs(trade_qty) < 1e-9:                 # already at target
            row['skip_reason'] = 'at_target'
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
        """Map forecast sign + sizing mode to a signed target quantity.

        Owns the ``'no_price'`` skip (price missing or zero in the
        portfolio). ``forecast == 0`` returns ``target_qty = 0.0``
        with ``skip_reason = None`` — a valid flat target, not a skip.

        Uses ``abs(price)`` in the divides so negative-priced
        instruments (e.g. WTI 2020) produce a sensible magnitude — the
        sign comes from the forecast.

        Returns a dict with keys ``target_qty``, ``skip_reason``,
        ``price``. Spliced into the diagnostic row by ``update_bar``.
        """
        symbol = event.symbol
        out: Dict[str, Any] = {
            'target_qty': None, 'skip_reason': None, 'price': None,
        }

        price = self.portfolio.get_price(symbol)
        if price is None or price == 0:
            out['skip_reason'] = 'no_price'
            return out
        out['price'] = price

        forecast = self.strategy.get_forecast(symbol)
        if forecast == 0:
            out['target_qty'] = 0.0
            return out

        sign = 1.0 if forecast > 0 else -1.0
        if self.size_mode == 'fixed_notional':
            target_qty = sign * self.position_size / abs(price)
        elif self.size_mode == 'fixed_quantity':
            target_qty = sign * self.position_size
        elif self.size_mode == 'fixed_equity_pct':
            equity = self.portfolio.calculate_balance()
            target_qty = sign * (equity * self.position_size) / abs(price)
        else:
            raise ValueError(
                f"Unknown size_mode: '{self.size_mode}'. "
                f"Must be one of {self._MODES}."
            )
        out['target_qty'] = target_qty
        return out
