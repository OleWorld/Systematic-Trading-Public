"""BacktestPortfolio — simulated cross-margin futures brokerage account."""

import logging
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from event import BarEvent, OrderEvent, FillEvent, OrderType, Direction
from portfolio._base import Portfolio, _DataHandlerLike, _EventsQueueLike

logger = logging.getLogger(__name__)


class BacktestPortfolio(Portfolio):
    """
    Simulated brokerage account for backtesting (cross-margin futures model).

    Margin model (cross-margin futures):
        Cash = wallet balance (initial capital + cumulative realized P&L - cumulative commissions)
        Unrealized P&L = sum(qty * (current_price - avg_cost)) over all positions
        Account balance = cash + unrealized P&L
        Position margin = abs(notional) / leverage
        Available balance = account balance - position_margin - reserved_margin

        Opening a position: cash decreases by commission only (no notional flow)
        Closing a position: cash changes by realized P&L minus commission

    Margin reservation for pending orders splits each order into a
    risk-reducing component (up to abs(projected_position) in the opposite
    direction) and a risk-increasing component. The reducing component
    reserves zero margin; the increasing component reserves
    qty_increasing * price / leverage. The same formula applies to MKT and
    LMT orders — only the reservation price differs (current safe price for
    MKT, limit price for LMT).

    Position baseline for new-order margin checks (``_projected_position``)
    is realized + pending MKT orders only; LMT pendings are excluded
    because they may never fill, so a pending LMT must not be allowed to
    pre-credit margin freedom for sizing other orders. The risk-increasing
    portion of any pending order — MKT or LMT — is still reserved via
    ``_reserved_margin``, so a pending LMT that *adds* to a position keeps
    its capital committed; only the *freeing* assumption is dropped.

    Solvency: the portfolio is solvent while account_balance >= 0. When
    account_balance falls below zero, ``check_solvency`` cancels every
    pending non-liquidation order FIFO and submits MKT liquidation orders
    to fully close every open position in ascending unrealized_pnl order
    (worst position first). Liquidation at mark price does not recover
    account_balance (it converts unrealized loss to realized loss); the
    loop's purpose is to stop further market exposure from making things
    worse, not to magically restore equity. ``available_balance`` going
    negative is tolerated — it just means existing positions are sitting
    at unrealized loss and no new opening orders can be placed.

    Fill-time margin trade-off: margin is reserved at submission price.
    Between submission and fill the price can move arbitrarily — a pending
    MKT BUY queued at $100 under ``fill_on='next_open'`` may fill at $200
    on a gap-up — and the FIFO reservation walk does NOT re-validate at
    fill time. The triggering fill itself can therefore drive
    ``account_balance`` below zero in one bar; ``check_solvency`` catches
    this *after* the fill is booked (cash and realized PnL have already
    moved) and liquidates from there. This is acceptable for backtesting
    under conservative slippage models, but ``LiveExecution`` must
    implement its own pre-fill margin gating (in real markets the exchange
    rejects orders that would breach margin at fill time).
    """

    def __init__(self, events_queue: _EventsQueueLike,
                 data_handler: _DataHandlerLike,
                 symbol_list: List[str],
                 initial_capital: float = 100_000.0,
                 leverage: float = 1.0):
        self.events_queue = events_queue
        self.data_handler = data_handler
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.leverage = leverage

        # Account state
        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {s: 0.0 for s in symbol_list}
        self.avg_cost: Dict[str, float] = {s: 0.0 for s in symbol_list}
        self._latest_prices: Dict[str, float] = {}

        # Pending order tracking (for margin reservation)
        self.pending_orders: Dict[str, OrderEvent] = {}

        # Per-symbol unrealized P&L (updated each bar)
        self.unrealized_pnl: Dict[str, float] = {s: 0.0 for s in symbol_list}

        # Account-level snapshots (updated each bar)
        self.account_balance: float = initial_capital
        self.available_balance: float = initial_capital

        # Per-symbol margin requirements (updated each bar and fill)
        self.margin_requirements: Dict[str, float] = {s: 0.0 for s in symbol_list}

        # Cumulative realized P&L per symbol
        self.realized_pnl: Dict[str, float] = {s: 0.0 for s in symbol_list}

        # Cumulative commission paid across all fills (account-level)
        self.total_commission: float = 0.0

        # Record keeping
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = []
        self.order_log: List[Dict] = []

    # ── Bar update (equity snapshot) ──────────

    def update_bar(self, event: BarEvent) -> None:
        """
        Refresh per-symbol margin and account snapshots from the new bar,
        append the equity curve entry, then run a solvency check (which may
        cancel pending orders and submit liquidation orders if
        account_balance < 0).

        OHLC fields are guaranteed non-NaN by the ``DataHandler`` gate;
        no defensive check is needed here.
        """
        self._latest_prices[event.symbol] = event.close

        # Update margin requirements for this symbol at current price
        self.margin_requirements[event.symbol] = (
            abs(self.positions[event.symbol] * event.close) / self.leverage
        )

        self._refresh_snapshot()

        prior_balance = (
            self.equity_curve[-1]['account_balance']
            if self.equity_curve
            else self.initial_capital
        )
        if prior_balance > 0:
            simple_return = (self.account_balance - prior_balance) / prior_balance
            log_return = (
                math.log(self.account_balance / prior_balance)
                if self.account_balance > 0
                else float('nan')
            )
        else:
            simple_return = float('nan')
            log_return = float('nan')

        self.equity_curve.append({
            'timestamp': event.timestamp,
            'symbol': event.symbol,
            'cash': self.cash,
            'unrealized_pnl': dict(self.unrealized_pnl),
            'realized_pnl': dict(self.realized_pnl),
            'account_balance': self.account_balance,
            'simple_return': simple_return,
            'log_return': log_return,
            'position_margin': self._position_margin(),
            'margin_requirements': dict(self.margin_requirements),
            'available_balance': self.available_balance,
            'positions': dict(self.positions),
            'total_commission': self.total_commission,
        })

        self.check_solvency(event.timestamp)

    # ── Order submission ──────────────────────

    def submit_order(self, symbol: str, quantity: float, direction: Direction,
                     timestamp: Any, order_type: OrderType,
                     price: Optional[float] = None,
                     is_liquidation: bool = False) -> Optional[OrderEvent]:
        """
        Receive an order request from the risk manager (the "trader").
        The portfolio (the "exchange") performs a margin check, scales the
        quantity down if necessary, then queues the OrderEvent for execution.
        Returns the emitted OrderEvent, or None if rejected.

        ``is_liquidation`` marks the order as a solvency-driven liquidation;
        such orders are exempt from the FIFO cancel pass in ``check_solvency``.
        """
        ref_price = self._validate_order_params(symbol, quantity, order_type, price)
        if ref_price is None:
            return None

        # Margin check — scale down if insufficient capital
        original_qty = quantity
        quantity = self._apply_margin_check(symbol, quantity, direction, ref_price)
        if quantity <= 0:
            logger.warning(
                "[ORDER REJECTED] %s %s %s | requested=%.6f | Reason: insufficient margin",
                symbol, order_type.value, direction.value, original_qty,
            )
            return None
        if quantity != original_qty:
            logger.warning(
                "[ORDER SCALED] %s %s %s | requested=%.6f -> approved=%.6f | Reason: insufficient margin",
                symbol, order_type.value, direction.value,
                original_qty, quantity,
            )

        order = OrderEvent(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            direction=direction,
            price=price,
            timestamp=timestamp,
            is_liquidation=is_liquidation,
        )

        self.pending_orders[order.order_id] = order
        self.order_log.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'order_type': order.order_type.value,
            'direction': direction.value,
            'quantity': quantity,
            'order_id': order.order_id,
            'is_liquidation': is_liquidation,
        })
        self.events_queue.put(order)
        return order

    def _validate_order_params(self, symbol: str, quantity: float,
                               order_type: OrderType,
                               price: Optional[float]) -> Optional[float]:
        """
        Validate order parameters and resolve a reference price.

        Returns the reference price on success, or None if the order is invalid.
        Raises ValueError if a MKT order is given a price.
        """
        if order_type == OrderType.MKT and price is not None:
            raise ValueError("MKT order cannot have a price.")
        if quantity <= 0:
            return None
        ref_price = price if price is not None else self.get_price(symbol)
        # Negative prices are tolerated (e.g. WTI 2020 settled at -$37);
        # downstream margin formulas use ``abs(price)``. Only exact zero is
        # rejected — it would divide-by-zero in the margin-scaling path.
        if ref_price is None or pd.isna(ref_price) or ref_price == 0:
            return None
        return ref_price

    # ── Fill -> Update state ──────────────────

    def update_fill(self, event: FillEvent) -> None:
        """
        Process a fill and update position, avg cost, cash, and margin.

        State transitions by direction and existing position:
            BUY + long/flat  -> add to long, update weighted avg cost
            BUY + short      -> cover short, realize P&L (avg_cost - fill)
                                if overfilled: flip to long at fill price
            SELL + short/flat -> add to short, update weighted avg cost
            SELL + long       -> close long, realize P&L (fill - avg_cost)
                                if overfilled: flip to short at fill price

        Cash changes only by realized P&L minus commission (futures model).
        After applying the fill, run a solvency check.
        """
        symbol = event.symbol
        fill_price = event.fill_notional / event.quantity if event.quantity != 0 else 0.0

        new_pos, new_avg_cost, realized = self._apply_fill_to_position(
            symbol, event.quantity, event.direction, fill_price, event.fill_notional)

        # Update state
        self.avg_cost[symbol] = new_avg_cost
        self.cash += realized - event.commission
        self.positions[symbol] = new_pos
        self.margin_requirements[symbol] = abs(new_pos * fill_price) / self.leverage
        self.realized_pnl[symbol] += realized
        self.total_commission += event.commission

        # Release reserved margin
        if event.order_id and event.order_id in self.pending_orders:
            del self.pending_orders[event.order_id]

        self.trade_log.append({
            'timestamp': event.timestamp,
            'symbol': symbol,
            'direction': event.direction.value,
            'quantity': event.quantity,
            'fill_price': fill_price,
            'fill_notional': event.fill_notional,
            'commission': event.commission,
            'realized_pnl': realized,
            'position_after': new_pos,
            'cash_after': self.cash,
            'order_id': event.order_id,
        })

        self.check_solvency(event.timestamp)

    def _apply_fill_to_position(self, symbol: str, qty: float, direction: Direction,
                                fill_price: float, fill_notional: float
                                ) -> tuple:
        """
        Pure position math: compute new position, avg cost, and realized P&L.

        Returns:
            (new_pos, new_avg_cost, realized_pnl)
        """
        old_pos = self.positions.get(symbol, 0.0)
        old_avg_cost = self.avg_cost[symbol]

        if direction == Direction.BUY:
            new_pos = old_pos + qty
            if old_pos >= 0:
                # Adding to long or opening long
                total_cost = old_avg_cost * old_pos + fill_notional
                new_avg_cost = total_cost / new_pos if new_pos != 0 else 0.0
                return new_pos, new_avg_cost, 0.0
            else:
                # Covering short
                cover_qty = min(qty, abs(old_pos))
                realized = cover_qty * (old_avg_cost - fill_price)
                if new_pos > 0:
                    return new_pos, fill_price, realized  # Flipped to long
                elif new_pos == 0:
                    return new_pos, 0.0, realized
                else:
                    return new_pos, old_avg_cost, realized  # Still short

        elif direction == Direction.SELL:
            new_pos = old_pos - qty
            if old_pos <= 0:
                # Adding to short or opening short
                total_cost = old_avg_cost * abs(old_pos) + fill_notional
                new_avg_cost = total_cost / abs(new_pos) if new_pos != 0 else 0.0
                return new_pos, new_avg_cost, 0.0
            else:
                # Closing long
                sell_qty = min(qty, old_pos)
                realized = sell_qty * (fill_price - old_avg_cost)
                if new_pos < 0:
                    return new_pos, fill_price, realized  # Flipped to short
                elif new_pos == 0:
                    return new_pos, 0.0, realized
                else:
                    return new_pos, old_avg_cost, realized  # Still long

        else:
            raise ValueError(f"Unexpected direction: {direction!r}")

    # ── Margin check ──────────────────────────

    def calculate_available_balance(self) -> float:
        """Available balance = account balance - position margin - reserved margin."""
        return (self.calculate_balance()
                - self._position_margin()
                - self._reserved_margin())

    def _apply_margin_check(self, symbol: str, qty: float, direction: Direction,
                            price: float) -> float:
        """
        First-stage margin check at order submission. Returns the max
        approvable quantity (full ``qty`` if sufficient margin, scaled
        down otherwise).

        Position baseline is the *projected* position (current + pending
        **MKT** orders for this symbol — LMT pendings are excluded because
        they are conditional fills). A same-bar order sequence such as
        ``FLATTEN + OPEN_OPPOSITE`` is therefore evaluated against its
        true end-state rather than the stale realized position so long as
        both legs are MKT (which is the live case — same-bar flips emit
        MKT orders). Marginal cost of the new order is taken against
        ``max(realized_margin, projected_margin_now)`` so that capital
        already locked in the existing position is correctly attributed
        and not double-counted with what pending MKT fills will free.

        Margin formula is uniform for MKT and LMT (no LMT worst-case);
        the only difference is the reference price (current safe price
        for MKT, limit price for LMT, which the caller resolves).
        """
        pos = self._projected_position(symbol)
        return self._scale_qty_to_margin(
            symbol, pos, qty, direction, price, exclude_order_id=None,
        )

    def _scale_qty_to_margin(self, symbol: str, pos: float, qty: float,
                             direction: Direction, price: float,
                             exclude_order_id: Optional[str]) -> float:
        """
        Compute the marginal margin demand of placing ``qty`` at ``price``
        against the baseline ``max(realized_margin, projected_margin_now)``
        and scale down if the available balance can't cover it.

        Same formula for MKT and LMT — the worst-case
        ``max(abs(new_pos), qty)`` clause for LMT has been removed.
        """
        realized_margin = self.margin_requirements.get(symbol, 0.0)
        projected_margin_now = abs(pos * price) / self.leverage
        baseline = max(realized_margin, projected_margin_now)

        new_margin_raw = self._calculate_new_margin(pos, qty, direction, price)
        delta = max(realized_margin, new_margin_raw) - baseline

        if delta <= 0:
            return qty  # Risk-neutral or reducing

        available = (self.calculate_balance()
                     - self._position_margin()
                     - self._reserved_margin(exclude_order_id=exclude_order_id))
        if delta <= available:
            return qty

        # Scale down so the new symbol margin tops out at baseline + available.
        # ``abs(price)`` so this stays correct for negative-priced instruments
        # (e.g. WTI 2020); margin is a magnitude, position-bound is a magnitude.
        max_new_margin = max(0.0, baseline + available)
        max_abs_new_pos = max_new_margin * self.leverage / abs(price)
        if direction == Direction.BUY:
            max_qty = max_abs_new_pos - pos
        elif direction == Direction.SELL:
            max_qty = max_abs_new_pos + pos
        else:
            raise ValueError(f"Unexpected direction: {direction!r}")
        return max(0.0, min(qty, max_qty))

    # ── Margin calculations ───────────────────

    def _projected_position(self, symbol: str,
                            exclude_order_id: Optional[str] = None) -> float:
        """
        Realized position plus signed quantities of all pending **MKT**
        orders for this symbol (optionally excluding one order). LMT
        pendings are intentionally excluded: they are conditional fills
        (price may never reach the limit), so projecting them would let
        an unfilled LMT pre-credit margin freedom that may never
        materialize. MKT pendings are guaranteed to fill on the next
        eligible bar by construction, so they are safe to project.

        Used as the position baseline for forward-looking margin checks
        so that same-bar order sequences (e.g. FLATTEN + OPEN_OPPOSITE,
        which are emitted as MKT) are evaluated against their projected
        end-state. Risk-increasing LMT pendings still consume capital
        via ``_reserved_margin``; this method only governs the position
        baseline, not the reservation ledger.
        """
        proj = self.positions.get(symbol, 0.0)
        for order in self.pending_orders.values():
            if order.order_id == exclude_order_id:
                continue
            if order.symbol != symbol:
                continue
            if order.order_type == OrderType.LMT:
                continue
            proj += self._dir_sign(order.direction) * order.quantity
        return proj

    def _reserved_margin(self, exclude_order_id: Optional[str] = None) -> float:
        """
        Sum of per-order margin reservations for all pending orders
        (optionally excluding one), walked in FIFO insertion order so each
        order's risk-reducing/increasing split is measured against the
        running projected position that includes earlier pending orders.

        Per-order reservation = ``risk_increasing_qty * abs(price) / leverage``,
        where ``risk_increasing_qty = qty - min(qty, max(0, -dir_sign * running_pos))``.
        ``abs(price)`` keeps the reservation a magnitude for negative-priced
        instruments (e.g. WTI 2020).
        Risk-reducing pending orders contribute zero. Same formula for
        MKT (priced at current safe price) and LMT (priced at the limit).
        Pending orders for which no safe price is available contribute
        zero (reservation deferred until a price is known).
        """
        running_pos: Dict[str, float] = dict(self.positions)
        reserved = 0.0
        for oid, order in self.pending_orders.items():
            if oid == exclude_order_id:
                continue
            if order.order_type == OrderType.LMT:
                price = order.price
            elif order.order_type == OrderType.MKT:
                price = self.get_price(order.symbol)
            else:
                raise ValueError(f"Unexpected order_type: {order.order_type!r}")
            if price is None:
                continue

            pos = running_pos.get(order.symbol, 0.0)
            dir_sign = self._dir_sign(order.direction)
            reducing_capacity = max(0.0, -dir_sign * pos)
            reducing = min(order.quantity, reducing_capacity)
            increasing = order.quantity - reducing
            reserved += increasing * abs(price) / self.leverage
            running_pos[order.symbol] = pos + dir_sign * order.quantity
        return reserved

    def _position_margin(self) -> float:
        """Total margin locked by realized open positions across all symbols."""
        return sum(self.margin_requirements.values())

    # ── Solvency enforcement ──────────────────

    def check_solvency(self, timestamp: Any) -> None:
        """
        Enforce solvency. Trigger: ``account_balance < 0`` (cash + unrealized
        PnL has gone negative; the account is blown up). When triggered:
          1. Cancel every pending non-liquidation order FIFO (stop new exposure).
          2. Submit MKT liquidation orders to fully close every open position
             in ascending ``unrealized_pnl`` order (worst first). Liquidation
             orders carry ``is_liquidation=True`` so the next tick's FIFO
             cancel pass will not cancel them, and a duplicate-submission
             guard prevents re-queuing on a symbol that already has a
             pending liquidation.
          3. Log an ``[INSOLVENT]`` warning with the timestamp and account
             balance.

        Liquidation at mark price does not recover ``account_balance`` (it
        only converts unrealized loss into realized loss), so the loop's
        exit condition is "no positions left," not "balance restored."
        Called from ``update_bar`` and ``update_fill``.
        """
        self._refresh_snapshot()
        if self.account_balance >= 0:
            return

        logger.warning(
            "[INSOLVENT] account_balance=%.2f | cancelling pending and "
            "liquidating all positions",
            self.account_balance,
        )
        self._cancel_pending_non_liquidation()
        self._liquidate_all_positions(timestamp)
        self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        """Recompute unrealized PnL, account_balance, and available_balance."""
        self._calculate_unrealized_pnl()
        self.account_balance = self.calculate_balance()
        self.available_balance = (self.account_balance
                                  - self._position_margin()
                                  - self._reserved_margin())

    def _cancel_pending_non_liquidation(self) -> None:
        """Cancel every pending order that isn't a liquidation order, FIFO."""
        for oid in list(self.pending_orders.keys()):
            order = self.pending_orders[oid]
            if order.is_liquidation:
                continue
            logger.warning(
                "[ORDER CANCELLED] %s %s %.6f %s | id=%s | Reason: account insolvent",
                order.symbol, order.order_type.value, order.quantity,
                order.direction.value, oid,
            )
            del self.pending_orders[oid]

    def _liquidate_all_positions(self, timestamp: Any) -> None:
        """
        Submit MKT liquidation orders for every non-zero position, in order
        of ascending ``unrealized_pnl`` (worst first). Full-position close
        per symbol. Symbols that already have a pending liquidation order
        are skipped (no duplicates). Symbols without a safe price are
        skipped with a warning and will be retried on a future tick.

        Caveat on the duplicate-submission guard: the guard is "any pending
        liquidation for this symbol blocks a new one." If a prior liquidation
        is stuck pending — e.g. queued under ``fill_on='next_open'`` and the
        next bars are all NaN-skipped, or the position has grown via direct
        state mutation in a test — no replacement order is enqueued. There
        is no automatic re-arm; in production this would require operator
        intervention.
        """
        symbols_with_pending_liquidation = {
            o.symbol for o in self.pending_orders.values() if o.is_liquidation
        }
        open_syms = [s for s, q in self.positions.items() if q != 0]
        open_syms.sort(key=lambda s: self.unrealized_pnl.get(s, 0.0))
        for sym in open_syms:
            if sym in symbols_with_pending_liquidation:
                continue
            price = self.get_price(sym)
            if price is None:
                logger.warning(
                    "[LIQUIDATION SKIPPED] %s | no price available yet", sym,
                )
                continue
            qty = abs(self.positions[sym])
            direction = Direction.SELL if self.positions[sym] > 0 else Direction.BUY
            self.submit_order(
                symbol=sym, quantity=qty, direction=direction,
                timestamp=timestamp, order_type=OrderType.MKT,
                is_liquidation=True,
            )

    # ── Helpers ───────────────────────────────

    @staticmethod
    def _dir_sign(direction: Direction) -> float:
        """Return +1.0 for BUY, -1.0 for SELL."""
        if direction == Direction.BUY:
            return 1.0
        elif direction == Direction.SELL:
            return -1.0
        else:
            raise ValueError(f"Unexpected direction: {direction!r}")

    def _calculate_new_margin(self, pos: float, qty: float, direction: Direction,
                              price: float) -> float:
        """
        Projected margin requirement after applying an order to baseline ``pos``.

        Caller passes the appropriate baseline (raw realized position via
        ``self.positions[symbol]`` or the projected-through-pending position
        via ``self._projected_position(symbol, ...)``) depending on use case.

        Uniform formula for MKT and LMT: ``abs(new_pos) * price / leverage``.
        """
        new_pos = pos + self._dir_sign(direction) * qty
        return abs(new_pos * price) / self.leverage

    def _calculate_unrealized_pnl(self) -> None:
        """Recalculate per-symbol unrealized P&L from current prices."""
        for sym, qty in self.positions.items():
            if qty != 0:
                price = self.get_price(sym)
                if price is None:
                    continue  # Keep previous unrealized_pnl for this symbol
                self.unrealized_pnl[sym] = qty * (price - self.avg_cost[sym])
            else:
                self.unrealized_pnl[sym] = 0.0

    def calculate_balance(self) -> float:
        """Account balance = wallet cash + total unrealized P&L."""
        return self.cash + sum(self.unrealized_pnl.values())

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Return the latest known price for ``symbol``: the cached price from
        the most recent ``update_bar``, or a fallback close fetched from the
        data handler if nothing has been cached yet. Returns ``None`` when no
        bars are available (cold start). The returned price is guaranteed
        non-NaN by the ``DataHandler`` gate.
        """
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
        bars = self.data_handler.get_latest_bars(symbol, 1)
        if len(bars) > 0:
            price = bars['Close'].iloc[-1]
            self._latest_prices[symbol] = price
            return price
        return None

    # ── Order management ──────────────────────

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order and release its reserved margin."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]

    # ── Record getters ────────────────────────

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Return the per-bar equity snapshot history as a DataFrame indexed by
        timestamp. Empty DataFrame if no bars have been processed yet.
        """
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df

    def get_trade_log(self) -> pd.DataFrame:
        """
        Return the per-fill trade log as a DataFrame (one row per
        ``FillEvent`` processed). Empty DataFrame if no fills have occurred.
        """
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def get_order_log(self) -> pd.DataFrame:
        """
        Return the per-submission order log as a DataFrame (one row per
        accepted ``OrderEvent``, including liquidation orders). Empty
        DataFrame if no orders have been submitted.
        """
        if not self.order_log:
            return pd.DataFrame()
        return pd.DataFrame(self.order_log)
