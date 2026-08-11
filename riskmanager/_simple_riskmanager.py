"""SimpleRiskManager — simple forecast-following sizer.

Execution-mode agnostic: the same class is used for backtesting and live.
For calibrated continuous forecasts (e.g. EWMAC) where conviction should
modulate position size, prefer ``VolTargetingRiskManager``.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from event import BarEvent, OrderType, Direction
from riskmanager._base import (
    RiskManager, _OrchestratorLike, _PortfolioLike, _StrategyLike,
)

if TYPE_CHECKING:  # avoid a config<->riskmanager import cycle at module load
    from config import InstrumentConfig

logger = logging.getLogger(__name__)


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
    the difference against the *projected* position — the realized
    position plus in-flight pending MKT orders (e.g. a same-bar
    margin-call liquidation), so an order already on its way to fill is
    never double-traded. If the projected position already matches the
    target (within ``1e-9``), no order is submitted.

    Per-bar diagnostic log analogous to ``Strategy.get_records``:
    every completed bar appends one row to ``self._records[symbol]``.
    Columns: ``forecast``, ``price``, ``size_mode``, ``position_size``,
    ``target_qty``, ``current_qty`` (realized position),
    ``pending_mkt_order_quantity`` (signed sum of in-flight MKT orders;
    the resize diff targets ``current_qty + pending_mkt_order_quantity``),
    ``trade_qty``, ``submitted`` (bool), and ``skip_reason`` ∈
    ``{None, 'no_price', 'zero_price', 'warmup_forecast', 'at_target'}``
    (``'no_price'`` — no reference price at all, ``get_price`` returned
    ``None``; ``'zero_price'`` — an exact-zero price under a notional
    sizing mode, whose divide-by-|price| is undefined (``fixed_quantity``
    is price-independent and sizes normally at zero);
    ``'warmup_forecast'`` — the strategy has not cached a forecast yet,
    so ``get_forecast`` returns ``None``). Any skip that strands a *held*
    position emits a WARNING (mirroring ``VolTargetingRiskManager``).
    Read via ``risk_manager.get_records(symbol)``.

    For calibrated continuous forecasts (e.g. EWMAC), use
    ``VolTargetingRiskManager`` instead — it scales the notional
    by ``forecast / 50`` so conviction translates into position size.
    """

    _MODES = ('fixed_notional', 'fixed_quantity', 'fixed_equity_pct')

    def __init__(self, portfolio: _PortfolioLike,
                 strategy: Union[_StrategyLike, _OrchestratorLike],
                 size_mode: str = 'fixed_quantity',
                 position_size: float = 10_000.0,
                 instruments: Optional[Dict[str, "InstrumentConfig"]] = None):
        """
        Parameters
        ----------
        portfolio
            Portfolio instance providing price, balance, positions, and
            ``submit_order``. Margin checking is the portfolio's
            responsibility, not the risk manager's.
        strategy
            Forecast source — a single ``Strategy`` or a multi-strategy
            ``orchestrator.Orchestrator`` — exposing
            ``get_forecast(symbol)``. Read on every completed bar to
            derive the target position.
        size_mode
            One of ``'fixed_quantity'`` (default — futures convention:
            size in contracts), ``'fixed_notional'``,
            ``'fixed_equity_pct'``. Validated at construction.
        position_size
            Magnitude interpreted per ``size_mode``.
        instruments
            Per-symbol ``InstrumentConfig`` registry. ``point_value`` divides
            the notional in the ``'fixed_notional'`` / ``'fixed_equity_pct'``
            modes (qty = notional / (point_value * |price|)); ``fractional``
            rounds the target to whole lots for futures. Default ``None``
            builds a uniform ``point_value=1`` / ``fractional=True`` registry
            over ``strategy.symbol_list`` (the crypto identity). Pass the SAME
            registry given to the portfolio for a futures book.
        """
        if size_mode not in self._MODES:
            raise ValueError(
                f"Unknown size_mode: '{size_mode}'. "
                f"Must be one of {self._MODES}."
            )
        super().__init__(portfolio, strategy)
        self.size_mode = size_mode
        self.position_size = position_size
        if instruments is None:
            from config import uniform_registry  # lazy: avoid import cycle
            instruments = uniform_registry(list(strategy.symbol_list))
        self.instruments = instruments

    def update_bar(self, event: BarEvent) -> None:
        """Resize the position to match the strategy's current forecast.

        Skips forming bars (idempotent across intra-period ticks).
        Delegates target-qty derivation (and the ``'no_price'`` skip)
        to ``_compute_target_qty``; owns the post-target
        ``'at_target'`` check and the submit call. Records one
        diagnostic row per *completed* bar — including early-exit
        branches — into ``self._records[symbol]`` via ``_record_row``.
        Bars for symbols outside ``strategy.symbol_list`` (context symbols)
        are skipped before any state update.
        """
        if event.is_forming:
            return

        symbol = event.symbol
        if symbol not in self._traded_symbols:
            # Context symbol: streamed for strategies to read, never sized.
            return
        forecast = self.strategy.get_forecast(symbol)
        current_qty = self.portfolio.positions.get(symbol, 0.0)
        # Signed sum of in-flight (pending) MKT orders — e.g. a same-bar
        # margin-call liquidation. The resize diff targets the projected
        # end-state ``current_qty + pending_mkt_order_quantity`` so an
        # order already on its way to fill is never double-traded.
        pending_mkt_order_quantity = (
            self.portfolio.projected_position(symbol) - current_qty
        )

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
            'pending_mkt_order_quantity': pending_mkt_order_quantity,
            'trade_qty': None,
            'submitted': False,
            'skip_reason': None,
        }
        row.update(self._compute_target_qty(event))

        if row['skip_reason'] is not None:
            # A skip means no well-defined target this bar. Harmless when
            # flat, but a HELD position is left unmanaged — surface it
            # loudly (mirrors VolTargetingRiskManager). A position with a
            # liquidation already in flight projects to 0, so no warning.
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
        # Whole-lot rounding for non-fractional instruments (futures): round to
        # the nearest contract before the diff so the traded size is an integer.
        if not self.instruments[symbol].fractional:
            target_qty = float(round(target_qty))
            row['target_qty'] = target_qty
        trade_qty = target_qty - (current_qty + pending_mkt_order_quantity)
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

        Owns the ``'no_price'`` skip (``get_price`` returned ``None`` —
        no reference price at all), the ``'zero_price'`` skip (an
        exact-zero price under a notional sizing mode, whose
        divide-by-|price| is undefined; ``fixed_quantity`` is
        price-independent and sizes normally at zero), and the
        ``'warmup_forecast'`` skip (``get_forecast`` returns ``None``
        before the strategy's first cached forecast).
        ``forecast == 0`` returns ``target_qty = 0.0`` with
        ``skip_reason = None`` — a valid flat target, not a skip.

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
        if price is None:
            out['skip_reason'] = 'no_price'
            return out
        out['price'] = price

        forecast = self.strategy.get_forecast(symbol)
        if forecast is None:
            # No forecast cached yet (warmup). Skip before the sign logic,
            # which would raise on None ( ``None > 0`` is a TypeError).
            out['skip_reason'] = 'warmup_forecast'
            return out
        if forecast == 0:
            out['target_qty'] = 0.0
            return out

        # Contract multiplier: dollar notional is qty * point_value * price,
        # so converting a target notional to contracts divides by
        # point_value * |price|. fixed_quantity is already in contracts —
        # price-independent, so it sizes fine even at an exact-zero price
        # (e.g. a spread crossing zero); only the notional modes, whose
        # divide is undefined at 0, skip with 'zero_price'.
        pv = self.instruments[symbol].point_value
        sign = 1.0 if forecast > 0 else -1.0
        if self.size_mode == 'fixed_notional':
            if price == 0:
                out['skip_reason'] = 'zero_price'
                return out
            target_qty = sign * self.position_size / (pv * abs(price))
        elif self.size_mode == 'fixed_quantity':
            target_qty = sign * self.position_size
        elif self.size_mode == 'fixed_equity_pct':
            if price == 0:
                out['skip_reason'] = 'zero_price'
                return out
            equity = self.portfolio.calculate_balance()
            target_qty = sign * (equity * self.position_size) / (pv * abs(price))
        else:
            raise ValueError(
                f"Unknown size_mode: '{self.size_mode}'. "
                f"Must be one of {self._MODES}."
            )
        out['target_qty'] = target_qty
        return out
