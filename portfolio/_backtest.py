"""BacktestPortfolio — simulated cross-margin futures brokerage account."""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from event import BarEvent, OrderEvent, FillEvent, OrderType, Direction
from portfolio._base import Portfolio, _DataHandlerLike, _EventsQueueLike

if TYPE_CHECKING:  # avoid a config<->portfolio import cycle at runtime
    from config import InstrumentConfig

logger = logging.getLogger(__name__)


class BacktestPortfolio(Portfolio):
    """
    Simulated brokerage account for backtesting (cross-margin futures model).

    Margin model (cross-margin futures):
        Cash = wallet balance (initial capital + cumulative realized P&L - cumulative commissions)
        Unrealized P&L = sum(qty * point_value * (current_price - avg_cost)) over all positions
        Account balance = cash + unrealized P&L
        Position margin = sum of per-symbol initial margin from each instrument's ``MarginModel``
        Available balance = account balance - position_margin - reserved_margin

    Per-symbol economics come from the ``instruments`` registry
    (``Dict[str, InstrumentConfig]``, see ``config/_instrument.py``): each
    symbol carries its own ``point_value`` (contract multiplier — dollar value
    is ``qty * point_value * price``), ``fractional`` flag (whole-lot
    enforcement at order submission), and ``MarginModel``. The default
    ``PortfolioMarginModel`` applies one universal rate
    (``initial_margin = abs(notional) * (1/leverage)`` at ``point_value=1``,
    reproducing the legacy formula exactly) plus a separate (lower)
    maintenance-margin rate that drives the liquidation trigger. Use
    ``config.uniform_registry(...)`` to build a homogeneous registry, or hand
    a per-symbol dict for a heterogeneous book.

        Opening a position: cash decreases by commission only (no notional flow)
        Closing a position: cash changes by realized P&L minus commission

    Margin reservation for pending orders splits each order into a
    risk-reducing component (up to abs(projected_position) in the opposite
    direction) and a risk-increasing component. The reducing component
    reserves zero margin; the increasing component reserves the symbol's
    ``MarginModel.initial_margin(qty_increasing, price, point_value)``. The
    same formula applies to MKT and LMT orders — only the reservation price
    differs (current safe price for MKT, limit price for LMT) — and, since
    pending orders may span symbols, the model and ``point_value`` are looked
    up per order.

    Position baseline for new-order margin checks (``projected_position``,
    also the baseline risk managers size against) is realized + pending MKT
    orders only; LMT pendings are excluded because they may never fill, so
    a pending LMT must not be allowed to pre-credit margin freedom for
    sizing other orders. The risk-increasing portion of any pending order —
    MKT or LMT — is still reserved via ``_reserved_margin``, so a pending
    LMT that *adds* to a position keeps its capital committed; only the
    *freeing* assumption is dropped.

    Solvency / margin call: the portfolio is solvent while
    ``account_balance >= total maintenance margin`` (the sum of the
    ``MarginModel``'s per-symbol maintenance margins). When account_balance
    falls below that floor, ``check_solvency`` cancels every pending
    non-liquidation order FIFO and submits MKT liquidation orders to fully
    close every open position in ascending unrealized_pnl order (worst
    position first). With the default ``maintenance_margin_rate=0`` the floor
    is 0, so the trigger collapses to the legacy ``account_balance < 0``
    (total wipeout); a positive rate fires the realistic earlier margin call.
    Liquidation at mark price does not recover account_balance (it converts
    unrealized loss to realized loss); the loop's purpose is to stop further
    market exposure from making things worse, not to magically restore equity.
    ``available_balance`` going negative is tolerated — it just means existing
    positions are sitting at unrealized loss and no new opening orders can be
    placed. While it is negative, the risk-neutral margin-check shortcut is
    also revoked: an order that merely *replaces* margin demand (e.g. a
    same-bar re-open netted against an in-flight liquidation) is scaled to
    what the balance actually backs instead of passing at full size —
    pure risk-reduction still always passes in full. (Liquidate-all-
    worst-first is kept; partial liquidation — closing only enough to
    restore the maintenance floor — is future work.)

    Fill-time margin trade-off: margin is reserved at submission price.
    Between submission and fill the price can move — a resting LMT, or a
    deferred cross-symbol MKT (e.g. a margin-call liquidation waiting for
    its symbol's own bar), may fill on a later bar at a very different
    price — and the FIFO reservation walk does NOT re-validate at fill
    time. The triggering fill itself can therefore drive
    ``account_balance`` below zero in one bar; ``check_solvency`` catches
    this *after* the fill is booked (cash and realized PnL have already
    moved) and liquidates from there. This is acceptable for backtesting
    under conservative slippage models, but ``LiveExecution`` must
    implement its own pre-fill margin gating (in real markets the exchange
    rejects orders that would breach margin at fill time).

    Cancelled orders: ``cancel_order`` and the margin-call cancel pass
    remove an order from this portfolio's ledger and release its reserved
    margin, but the simulated exchange holds its own pending book and an
    already-emitted ``OrderEvent`` cannot be recalled from the events
    queue. Any ``FillEvent`` that later arrives for a cancelled order id
    is therefore **voided** in ``update_fill`` (WARNING log, no state
    change) so a cancelled order can never move positions or cash.
    """

    def __init__(self, events_queue: _EventsQueueLike,
                 data_handler: _DataHandlerLike,
                 symbol_list: List[str],
                 instruments: Dict[str, "InstrumentConfig"],
                 initial_capital: float = 100_000.0):
        self.events_queue = events_queue
        self.data_handler = data_handler
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        # Per-symbol metadata registry (point_value, fractional, and the
        # per-symbol slippage/commission/margin models). Every traded symbol
        # must be covered — margin and PnL math read its config per symbol.
        missing = set(symbol_list) - set(instruments)
        if missing:
            raise ValueError(
                f"instruments registry is missing config for symbols: "
                f"{sorted(missing)}"
            )
        self.instruments = instruments

        # Account state
        self.cash: float = initial_capital
        self.positions: Dict[str, float] = {s: 0.0 for s in symbol_list}
        self.avg_cost: Dict[str, float] = {s: 0.0 for s in symbol_list}
        self._latest_prices: Dict[str, float] = {}

        # Pending order tracking (for margin reservation)
        self.pending_orders: Dict[str, OrderEvent] = {}

        # Order ids cancelled via cancel_order / the margin-call cancel
        # pass. The simulated exchange may still emit a fill for such an
        # order (its book and the events queue are not recalled);
        # update_fill voids those fills instead of booking them. An id is
        # discarded once its fill has been voided (at most one fill per
        # order id), keeping the set small.
        self._cancelled_order_ids: set = set()

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

        # Record keeping. ``equity_curve`` holds one LIGHTWEIGHT row per
        # BarEvent (account-level scalars only). The heavy per-symbol dicts
        # (positions / unrealized_pnl / realized_pnl / margin_requirements)
        # are NOT copied per event — they would cost O(N) per event and
        # O(N²×T) overall with multi-symbol bars. Instead one snapshot per
        # timestamp is kept in ``_pnl_snapshots`` (overwritten on each event,
        # so it retains that timestamp's last-event state) and re-attached by
        # ``get_equity_curve``. Net memory: O(N×T).
        self.equity_curve: List[Dict] = []
        self._pnl_snapshots: Dict[Any, Dict[str, Dict[str, float]]] = {}
        self.trade_log: List[Dict] = []
        self.order_log: List[Dict] = []

    # ── Bar update (equity snapshot) ──────────

    def update_bar(self, event: BarEvent) -> None:
        """
        Refresh per-symbol margin and account snapshots from the new bar,
        run the solvency check (which may cancel pending orders and submit
        liquidation orders), THEN append the lightweight per-event equity
        row and per-timestamp snapshot — so a margin-call bar's row
        reflects post-enforcement state.

        OHLC fields are guaranteed non-NaN by the ``DataHandler`` gate;
        no defensive check is needed here.
        """
        self._latest_prices[event.symbol] = event.close

        # Update margin requirements for this symbol at current price
        cfg = self.instruments[event.symbol]
        self.margin_requirements[event.symbol] = cfg.margin.initial_margin(
            self.positions[event.symbol], event.close, cfg.point_value,
        )

        # Per-event fast path: only THIS symbol's price changed since the
        # last event, so only its unrealized entry needs recomputing —
        # identical numbers to a full-universe recompute at O(1) instead
        # of O(N) (see _update_symbol_unrealized's invariant note).
        self._update_symbol_unrealized(event.symbol)
        self._refresh_balances()

        # Solvency FIRST, then record: the margin-call cancel pass
        # releases reserved margin (available_balance changes), so
        # appending the row before enforcement left a margin-call bar's
        # row one event stale. Positions cannot change inside enforcement
        # (liquidations rest via fill_on_next_bar) and prices don't move,
        # so maintenance_margin computed here stays valid for the row.
        maintenance_margin = self._maintenance_margin()
        self._enforce_solvency(event.timestamp,
                               maintenance_margin=maintenance_margin)

        prior_balance = (
            self.equity_curve[-1]['account_balance']
            if self.equity_curve
            else self.initial_capital
        )
        simple_return, log_return = self._period_returns(prior_balance)

        # Lightweight per-event row: account-level scalars only.
        self.equity_curve.append({
            'timestamp': event.timestamp,
            'symbol': event.symbol,
            'cash': self.cash,
            'account_balance': self.account_balance,
            'simple_return': simple_return,
            'log_return': log_return,
            'position_margin': self._position_margin(),
            'maintenance_margin': maintenance_margin,
            'available_balance': self.available_balance,
            'total_commission': self.total_commission,
        })
        # Per-timestamp per-symbol snapshot, overwritten on each event so it
        # retains this timestamp's last-event state (the value that
        # last-row-per-timestamp consumers read). One snapshot per timestamp,
        # not per event — get_equity_curve re-attaches it as object columns.
        self._pnl_snapshots[event.timestamp] = {
            'positions': dict(self.positions),
            'unrealized_pnl': dict(self.unrealized_pnl),
            'realized_pnl': dict(self.realized_pnl),
            'margin_requirements': dict(self.margin_requirements),
        }

    # ── Order submission ──────────────────────

    def submit_order(self, symbol: str, quantity: float, direction: Direction,
                     timestamp: Any, order_type: OrderType,
                     price: Optional[float] = None,
                     is_liquidation: bool = False,
                     fill_on_next_bar: bool = False) -> Optional[OrderEvent]:
        """
        Receive an order request from the risk manager (the "trader").
        The portfolio (the "exchange") performs a margin check, scales the
        quantity down if necessary, then queues the OrderEvent for execution.
        Returns the emitted OrderEvent, or None if rejected.

        ``is_liquidation`` marks the order as a solvency-driven liquidation;
        such orders are exempt from the FIFO cancel pass in ``check_solvency``.

        ``fill_on_next_bar`` is forwarded onto the emitted ``OrderEvent`` —
        see ``OrderEvent.fill_on_next_bar`` for the fill contract it enforces
        at execution time (the order never fills against a bar dispatched
        before it existed).
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
        # Whole-lot enforcement (authoritative safety net): non-fractional
        # instruments (futures) trade in integer contracts only. Truncate
        # toward zero so the rounded order never exceeds the margin-approved
        # size; a sub-1-lot target can't be traded, so drop the order.
        if not self.instruments[symbol].fractional:
            quantity = float(math.floor(quantity))
            if quantity <= 0:
                logger.info(
                    "[ORDER REJECTED] %s %s %s | requested=%.6f | Reason: "
                    "sub-1-lot for whole-lot instrument",
                    symbol, order_type.value, direction.value, original_qty,
                )
                return None
        if quantity != original_qty:
            logger.warning(
                "[ORDER SCALED] %s %s %s | requested=%.6f -> approved=%.6f | Reason: insufficient margin or whole-lot rounding",
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
            fill_on_next_bar=fill_on_next_bar,
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
            'fill_on_next_bar': fill_on_next_bar,
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
        # downstream margin formulas use ``abs(price)``. Exact zero is also
        # tolerated now: the ``MarginModel`` owns the only divide-by-price and
        # returns ``inf`` (no position bound) at price 0, since a zero-priced
        # position locks no margin. Only None / NaN (no usable reference price)
        # are rejected.
        if ref_price is None or pd.isna(ref_price):
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

        Fills for cancelled orders are **voided**: the portfolio cancelled
        the order (releasing its reserved margin), but the simulated
        exchange's book / the events queue may still deliver the fill —
        booking it would resurrect an order the account no longer backs.

        After applying the fill (and the solvency check), the last equity
        row and its timestamp's snapshot are re-synced when they belong to
        this fill's timestamp — see _sync_last_equity_row.
        """
        if (event.order_id is not None
                and event.order_id in self._cancelled_order_ids):
            self._cancelled_order_ids.discard(event.order_id)
            logger.warning(
                "[FILL VOIDED] %s %.6f %s | id=%s | Reason: order was cancelled",
                event.direction.value, event.quantity, event.symbol,
                event.order_id,
            )
            return

        symbol = event.symbol
        cfg = self.instruments[symbol]
        pv = cfg.point_value
        fill_price = event.fill_notional / event.quantity if event.quantity != 0 else 0.0

        # ``_apply_fill_to_position`` returns ``realized`` in PRICE space
        # (per-unit PnL * qty); the contract multiplier converts it to dollars.
        # avg_cost stays a per-unit price (fill_notional = qty * price), so the
        # weighted-average math inside is pv-independent.
        new_pos, new_avg_cost, realized = self._apply_fill_to_position(
            symbol, event.quantity, event.direction, fill_price, event.fill_notional)
        dollar_realized = realized * pv

        # Update state
        self.avg_cost[symbol] = new_avg_cost
        self.cash += dollar_realized - event.commission
        self.positions[symbol] = new_pos
        self.margin_requirements[symbol] = cfg.margin.initial_margin(
            new_pos, fill_price, pv,
        )
        self.realized_pnl[symbol] += dollar_realized
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
            'realized_pnl': dollar_realized,
            'position_after': new_pos,
            'cash_after': self.cash,
            'order_id': event.order_id,
        })

        # Only this symbol's position/avg-cost changed — incremental
        # refresh, then the solvency check (skipping the public wrapper's
        # redundant full recompute).
        self._update_symbol_unrealized(symbol)
        self._refresh_balances()
        self._enforce_solvency(event.timestamp)

        # Fill-synchronous recording: the equity row for this fill's bar
        # was appended BEFORE the fill booked (portfolio.update_bar runs
        # first in the engine's bar chain; fills drain later from the
        # queue), so without a re-sync the fill's cash/commission/position
        # effects would surface under the NEXT row's timestamp. The last
        # row is structurally this fill's bar in every engine flow
        # (same-bar close fills book before the next bar streams; deferred
        # fills book right after their own symbol's bar chain) — the
        # timestamp guard covers direct calls outside the engine flow,
        # where rewriting a past-labeled row with future state would be
        # wrong. Runs AFTER _enforce_solvency so a fill-triggered margin
        # call's cancel pass is captured in the row too.
        if (self.equity_curve
                and self.equity_curve[-1]['timestamp'] == event.timestamp):
            self._sync_last_equity_row()
        elif self.equity_curve:
            logger.warning(
                "[ROW-SYNC SKIPPED] fill at %s but last equity row is at "
                "%s — row left as-is. Unreachable in engine flows (the "
                "portfolio appends its row before any fill for that bar "
                "can exist): either a direct out-of-flow call, or the "
                "recording invariant broke.",
                event.timestamp, self.equity_curve[-1]['timestamp'],
            )

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
        pos = self.projected_position(symbol)
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

        A risk-neutral-or-reducing order (``delta <= 0``) is approved at
        full size only while ``available >= 0``. Below the floor the
        order goes through the scaling branch like any other: pure
        risk-reduction still passes in full (``max_qty >= |pos|`` even at
        a zero budget — liquidations are never blocked), but the
        re-open/flip portion is capped by what the balance actually
        backs. Without that gate, a margin-call bar's in-flight
        liquidation nets the projected position to 0 and a same-bar
        full-size re-open scores ``delta == 0`` — a free pass for an
        insolvent account.

        Same formula for MKT and LMT — the worst-case
        ``max(abs(new_pos), qty)`` clause for LMT has been removed.
        """
        cfg = self.instruments[symbol]
        realized_margin = self.margin_requirements.get(symbol, 0.0)
        projected_margin_now = cfg.margin.initial_margin(pos, price, cfg.point_value)
        baseline = max(realized_margin, projected_margin_now)

        new_margin_raw = self._calculate_new_margin(symbol, pos, qty, direction, price)
        delta = max(realized_margin, new_margin_raw) - baseline

        available = (self.calculate_balance()
                     - self._position_margin()
                     - self._reserved_margin(exclude_order_id=exclude_order_id))
        if delta <= 0:
            if available >= 0:
                return qty  # Risk-neutral or reducing; account above its floor
            # Below the floor (negative available), the risk-neutral free
            # pass is revoked: netting against an in-flight liquidation
            # makes a same-bar full-size RE-OPEN look risk-neutral
            # (projected position 0, new margin == old margin), which
            # would let an insolvent account rebuild its position for
            # free. Fall through to the scaling branch instead — it still
            # grants any pure risk-reduction in full (``max_qty =
            # max_abs_new_pos + |pos| >= |pos|`` covers a full close even
            # at a zero margin budget, so liquidations are never blocked)
            # but balance-caps the re-open/flip portion.
        elif delta <= available:
            return qty

        # Scale down so the new symbol margin tops out at baseline + available.
        # The model owns the divide-by-price (and its price==0 edge: it returns
        # ``inf``, i.e. no position bound — a zero-priced position locks no
        # margin). ``max_abs_position`` is a magnitude, correct for
        # negative-priced instruments (e.g. WTI 2020).
        max_new_margin = max(0.0, baseline + available)
        max_abs_new_pos = cfg.margin.max_abs_position(
            max_new_margin, price, cfg.point_value,
        )
        if direction == Direction.BUY:
            max_qty = max_abs_new_pos - pos
        elif direction == Direction.SELL:
            max_qty = max_abs_new_pos + pos
        else:
            raise ValueError(f"Unexpected direction: {direction!r}")
        return max(0.0, min(qty, max_qty))

    # ── Margin calculations ───────────────────

    def projected_position(self, symbol: str,
                           exclude_order_id: Optional[str] = None) -> float:
        """
        Realized position plus signed quantities of all pending **MKT**
        orders for this symbol (optionally excluding one order — an
        internal hook used by the margin checks) — i.e. the position the
        account will hold once in-flight MKT orders (e.g. a same-bar
        margin-call liquidation) fill. LMT pendings are intentionally
        excluded: they are conditional fills (price may never reach the
        limit), so projecting them would let an unfilled LMT pre-credit
        margin freedom that may never materialize. MKT pendings are
        guaranteed to fill on the next eligible bar by construction, so
        they are safe to project. Unknown symbols project ``0.0``.

        Consumers: the forward-looking margin checks use it as the
        position baseline so same-bar order sequences (e.g. FLATTEN +
        OPEN_OPPOSITE, which are emitted as MKT) are evaluated against
        their projected end-state; risk managers size their resize diff
        against it so an in-flight liquidation is never double-traded.
        NOTE for any future LMT-emitting risk manager: because resting
        LMT orders are excluded here, sizing diffs against this baseline
        would double-count them — such an RM must track its own resting
        orders. Risk-increasing LMT pendings still consume capital via
        ``_reserved_margin``; this method only governs the position
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

        Per-order reservation = ``margin.initial_margin(risk_increasing_qty,
        price, point_value)``, where ``risk_increasing_qty = qty -
        min(qty, max(0, -dir_sign * running_pos))``. The model's ``abs`` keeps
        the reservation a magnitude for negative-priced instruments (e.g. WTI
        2020). The pending orders may span DIFFERENT symbols, so the margin
        model and ``point_value`` are looked up per order (``order.symbol``).
        Risk-reducing pending orders contribute zero. Same formula for
        MKT (priced at current safe price) and LMT (priced at the limit).
        Pending orders for which no safe price is available contribute
        zero (reservation deferred until a price is known).
        """
        if not self.pending_orders:
            # Common case (no in-flight orders): skip the O(N)
            # positions-dict copy below — the sum is 0 regardless.
            return 0.0
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
            cfg = self.instruments[order.symbol]
            reserved += cfg.margin.initial_margin(
                increasing, price, cfg.point_value,
            )
            running_pos[order.symbol] = pos + dir_sign * order.quantity
        return reserved

    def _position_margin(self) -> float:
        """Total initial margin locked by realized open positions across all symbols."""
        return sum(self.margin_requirements.values())

    def _maintenance_margin(self) -> float:
        """
        Total maintenance margin across all open positions at current prices —
        the equity floor below which ``check_solvency`` triggers a margin call.

        Mirrors ``_position_margin`` but consults the model's
        ``maintenance_margin`` and reads the current mark per symbol; symbols
        with no available price are skipped (their margin is unknown this tick).
        With the default ``maintenance_margin_rate=0`` this returns 0.0, so the
        liquidation trigger collapses to the legacy ``account_balance < 0``.
        """
        total = 0.0
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.get_price(sym)
            if price is None:
                continue
            cfg = self.instruments[sym]
            total += cfg.margin.maintenance_margin(qty, price, cfg.point_value)
        return total

    # ── Solvency enforcement ──────────────────

    def check_solvency(self, timestamp: Any) -> None:
        """
        Enforce the maintenance-margin call. Trigger:
        ``account_balance < total maintenance margin`` (the equity backing the
        positions has fallen below the exchange's maintenance floor). With the
        default ``maintenance_margin_rate=0`` the floor is 0, so this is exactly
        the legacy ``account_balance < 0`` (total wipeout); a positive rate
        fires the realistic earlier margin call. When triggered:
          1. Cancel every pending non-liquidation order FIFO (stop new exposure).
          2. Submit MKT liquidation orders to fully close every open position
             in ascending ``unrealized_pnl`` order (worst first). Liquidation
             orders carry ``is_liquidation=True`` so the next tick's FIFO
             cancel pass will not cancel them, and a duplicate-submission
             guard prevents re-queuing on a symbol that already has a
             pending liquidation.
          3. Log a ``[MARGIN CALL]`` warning with the account balance and the
             maintenance-margin floor it breached.

        Liquidation at mark price does not recover ``account_balance`` (it
        only converts unrealized loss into realized loss), so the loop's
        exit condition is "no positions left," not "balance restored."

        This public entry point performs a FULL cache refresh first, so it
        is safe to call after arbitrary direct state mutation (tests,
        research hooks). The engine-driven hot paths (``update_bar`` /
        ``update_fill``) keep their caches fresh incrementally and call
        ``_enforce_solvency`` directly, skipping the redundant refresh.
        """
        self._refresh_snapshot()
        self._enforce_solvency(timestamp)

    def _enforce_solvency(self, timestamp: Any,
                          maintenance_margin: Optional[float] = None) -> None:
        """Margin-call trigger + liquidation flow, assuming FRESH caches.

        Callers must guarantee ``unrealized_pnl`` / ``account_balance``
        reflect current prices and positions (the hot paths refresh
        incrementally before calling; the public ``check_solvency`` does a
        full refresh). ``maintenance_margin`` may be passed when the
        caller already computed it this event (``update_bar`` shares it
        with the equity row) — state must not have mutated since.
        """
        if maintenance_margin is None:
            maintenance_margin = self._maintenance_margin()
        if self.account_balance >= maintenance_margin:
            return

        logger.warning(
            "[MARGIN CALL] account_balance=%.2f < maintenance_margin=%.2f | "
            "cancelling pending and liquidating all positions",
            self.account_balance, maintenance_margin,
        )
        self._cancel_pending_non_liquidation()
        self._liquidate_all_positions(timestamp)
        self._refresh_snapshot()

    def _refresh_snapshot(self) -> None:
        """Recompute unrealized PnL (full universe), account_balance, and
        available_balance — the self-healing full refresh. Used by the
        public ``check_solvency`` and the post-liquidation re-sync; the
        per-event hot paths use the incremental
        ``_update_symbol_unrealized`` + ``_refresh_balances`` pair instead."""
        self._calculate_unrealized_pnl()
        self._refresh_balances()

    def _refresh_balances(self) -> None:
        """Recompute ``account_balance`` / ``available_balance`` from the
        (already-current) per-symbol ``unrealized_pnl`` cache — the summing
        half of ``_refresh_snapshot`` without the O(N) per-symbol
        unrealized recompute."""
        self.account_balance = self.calculate_balance()
        self.available_balance = (self.account_balance
                                  - self._position_margin()
                                  - self._reserved_margin())

    def _update_symbol_unrealized(self, symbol: str) -> None:
        """Recompute ``unrealized_pnl[symbol]`` only — O(1) per event.

        Same formula as ``_calculate_unrealized_pnl`` restricted to one
        symbol. INVARIANT this relies on: a symbol's unrealized inputs
        (``positions``, ``avg_cost``, ``_latest_prices``) mutate only at
        that symbol's own ``update_bar`` (price) and ``update_fill``
        (position / avg cost), and both call this method immediately —
        so every OTHER symbol's cached value is already current and a
        full-universe recompute would produce identical numbers at O(N)
        cost. Code that mutates portfolio state directly (tests, research
        hooks) must re-sync via ``_refresh_snapshot`` or go through the
        public ``check_solvency``.
        """
        qty = self.positions[symbol]
        if qty != 0:
            price = self.get_price(symbol)
            if price is None:
                return  # Keep previous unrealized_pnl for this symbol
            pv = self.instruments[symbol].point_value
            self.unrealized_pnl[symbol] = qty * pv * (price - self.avg_cost[symbol])
        else:
            self.unrealized_pnl[symbol] = 0.0

    def _period_returns(self, prior_balance: float) -> tuple:
        """``(simple_return, log_return)`` of the current ``account_balance``
        against ``prior_balance``.

        Both are NaN when the prior balance is non-positive; ``log_return``
        is additionally NaN when the current balance is non-positive.
        Shared by ``update_bar`` (prior = previous equity row) and
        ``_sync_last_equity_row`` (prior = second-to-last row) so the two
        stay lock-stepped.
        """
        if prior_balance > 0:
            simple_return = (self.account_balance - prior_balance) / prior_balance
            log_return = (
                math.log(self.account_balance / prior_balance)
                if self.account_balance > 0
                else float('nan')
            )
            return simple_return, log_return
        return float('nan'), float('nan')

    def _sync_last_equity_row(self) -> None:
        """Overwrite the LAST equity row's scalars and its timestamp's
        per-symbol snapshot from current account state (returns recomputed
        against the same prior-row baseline via ``_period_returns``).

        Called by ``update_fill`` (fill-synchronous recording: the row for
        a bar is appended before that bar's fills book, so each booked
        fill re-syncs it) — the sole recording reconciliation since the
        end-of-run ``finalize`` hook was removed as a proven no-op
        (2026-08, oracle-verified on the sample smoke runs). Assumes
        FRESH caches — the caller refreshes incrementally first. No-op
        on an empty curve.
        """
        if not self.equity_curve:
            return
        last = self.equity_curve[-1]
        prior_balance = (
            self.equity_curve[-2]['account_balance']
            if len(self.equity_curve) >= 2
            else self.initial_capital
        )
        simple_return, log_return = self._period_returns(prior_balance)
        last.update({
            'cash': self.cash,
            'account_balance': self.account_balance,
            'simple_return': simple_return,
            'log_return': log_return,
            'position_margin': self._position_margin(),
            'maintenance_margin': self._maintenance_margin(),
            'available_balance': self.available_balance,
            'total_commission': self.total_commission,
        })
        self._pnl_snapshots[last['timestamp']] = {
            'positions': dict(self.positions),
            'unrealized_pnl': dict(self.unrealized_pnl),
            'realized_pnl': dict(self.realized_pnl),
            'margin_requirements': dict(self.margin_requirements),
        }

    def _cancel_pending_non_liquidation(self) -> None:
        """Cancel every pending order that isn't a liquidation order, FIFO.

        Cancelled ids are registered so a fill the simulated exchange may
        still emit for them is voided by ``update_fill``.
        """
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
            self._cancelled_order_ids.add(oid)

    def _liquidate_all_positions(self, timestamp: Any) -> None:
        """
        Submit MKT liquidation orders for every non-zero position, in order
        of ascending ``unrealized_pnl`` (worst first). Full-position close
        per symbol. Symbols that already have a pending liquidation order
        are skipped (no duplicates). Symbols without a safe price are
        skipped with a warning and will be retried on a future tick.

        Every liquidation order carries ``fill_on_next_bar=True``: it always
        rests and fills on the symbol's next bar event — at that bar's open
        when the decision-period bar has already streamed (never
        retroactively against a bar that already closed), or at the
        decision-period bar's own close when that bar arrives late (the
        symbol's bar for this period had not streamed yet). See
        ``OrderEvent.fill_on_next_bar`` / ``BacktestExecution._try_fill`` for
        the fill-timing contract.

        Caveat on the duplicate-submission guard: the guard is "any pending
        liquidation for this symbol blocks a new one." If a prior liquidation
        is stuck pending — every liquidation now defers to the symbol's next
        bar, so this is the ordinary case, not just the decision-period-bar
        gap — and the symbol never prints again, or the position has grown
        via direct state mutation in a test, no replacement order is
        enqueued. There is no automatic re-arm; in production this would
        require operator intervention.
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
                is_liquidation=True, fill_on_next_bar=True,
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

    def _calculate_new_margin(self, symbol: str, pos: float, qty: float,
                              direction: Direction, price: float) -> float:
        """
        Projected margin requirement after applying an order to baseline ``pos``.

        Caller passes the appropriate baseline (raw realized position via
        ``self.positions[symbol]`` or the projected-through-pending position
        via ``self.projected_position(symbol, ...)``) depending on use case.

        Uniform formula for MKT and LMT: the model's initial margin on the
        projected position.
        """
        new_pos = pos + self._dir_sign(direction) * qty
        cfg = self.instruments[symbol]
        return cfg.margin.initial_margin(new_pos, price, cfg.point_value)

    def _calculate_unrealized_pnl(self) -> None:
        """Recalculate per-symbol unrealized P&L (in dollars) from current prices.

        Dollar PnL = ``qty * point_value * (price - avg_cost)`` — avg_cost is a
        per-unit price, so the contract multiplier converts the price move to
        dollars.
        """
        for sym, qty in self.positions.items():
            if qty != 0:
                price = self.get_price(sym)
                if price is None:
                    continue  # Keep previous unrealized_pnl for this symbol
                pv = self.instruments[sym].point_value
                self.unrealized_pnl[sym] = qty * pv * (price - self.avg_cost[sym])
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
        if bars:
            price = bars[-1].close
            self._latest_prices[symbol] = price
            return price
        return None

    # ── Order management ──────────────────────

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order and release its reserved margin.

        The id is registered as cancelled so a fill the simulated exchange
        may still emit for it is voided by ``update_fill`` instead of
        booked.
        """
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self._cancelled_order_ids.add(order_id)

    # ── Record getters ────────────────────────

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Return the per-bar equity snapshot history as a DataFrame indexed by
        timestamp. Empty DataFrame if no bars have been processed yet.

        Account-level columns (``cash``, ``account_balance``, ``simple_return``,
        ``log_return``, ``position_margin``, ``maintenance_margin``,
        ``available_balance``, ``total_commission``) are recorded once per
        BarEvent. The per-symbol
        dict columns (``positions``, ``unrealized_pnl``, ``realized_pnl``,
        ``margin_requirements``) are re-attached from the per-timestamp
        snapshot (one per timestamp, last-event-wins): every event row of a
        given timestamp therefore carries that timestamp's *final* snapshot,
        matching the documented "collapse to one row per timestamp (last
        wins)" consumption. The attached dicts are shared references (the
        snapshots live once per timestamp), so this does not re-bloat memory.
        """
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        for col in ('positions', 'unrealized_pnl', 'realized_pnl',
                    'margin_requirements'):
            df[col] = [self._pnl_snapshots[ts][col] for ts in df.index]
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
