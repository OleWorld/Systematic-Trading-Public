"""BacktestConfig — single source of truth for all backtest infrastructure parameters."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BacktestConfig:
    """
    Parameter holder for a backtest run.

    Centralizes the *run-level* infrastructure parameters (capital,
    timeframes, vol-target knobs). The actual backtest date range is not
    configured here — the engine streams exactly the data supplied, and
    ``runlog.save_run`` records the realized range in the manifest.
    Construction validates only the **structural coherence of its own
    fields** — non-empty ``symbols``, ``base_timeframe`` registered in
    ``timeframes``, higher timeframes strictly larger than base. Domain
    values (vol-target knobs, corr knobs, size modes, ...) are validated
    by the module that consumes them, at wiring time — e.g.
    ``CorrelationManager.__init__`` raises on a bad ``corr_lookback`` (the
    same value also feeds ``UniverseManager.__init__`` as
    ``min_history_bars``).
    **Per-symbol economics** (point_value, fractional, slippage, commission,
    margin/leverage) live in ``InstrumentConfig`` (``config/_instrument.py``),
    supplied as a registry to the portfolio, execution, and risk manager.
    Callers read the fields off this object as they wire each module manually
    (see ``backtests/sample_backtest/backtest_ewmac_sample.py``).
    """

    # --- Data ---
    symbols: List[str]
    base_timeframe: str                                         # streaming TF (e.g. '1m')
    days_convention: str                                        # 'calendar' (365 days/year, 24/7) or 'business' (252 trading days/year)
    timeframes: Dict[str, int] = field(default_factory=dict)    # {tf: maxlen} e.g. {'1m': 500, '1h': 500, '4h': 200}

    # --- Portfolio ---
    # NOTE: per-symbol economics (point_value, fractional, slippage,
    # commission, margin/leverage) live in InstrumentConfig
    # (config/_instrument.py), supplied as a registry to the portfolio,
    # execution, and risk manager — NOT here.
    initial_capital: float = 100_000.0

    # --- Risk / Sizing ---
    # Carver vol-targeting knobs consumed by `VolTargetingRiskManager`
    # (annual_target_vol, vol_target_mode, position_buffer,
    # instrument_weight_mode, idm_cap). ``idm`` is not in config — pass it
    # directly to the risk manager constructor if a non-default value is
    # needed. The corr_* knobs below instead feed `UniverseManager`
    # (corr_lookback -> min_history_bars, corr_timeframe -> history_timeframe)
    # and `CorrelationManager` (all six) — not the risk manager directly.
    annual_target_vol: Optional[float] = None  # Carver's τ — $ amount ('dollar_volatility') or fraction in (0,1) ('percent_volatility'); required (validated) when wiring VolTargetingRiskManager, may stay None for SimpleRiskManager runs
    vol_target_mode: str = 'dollar_volatility'     # 'dollar_volatility' (fixed annual $ vol budget) or 'percent_volatility' (fraction of equity)
    position_buffer: float = 0.25        # Carver §10.7 dead-band (0.0 to trade every gap)
    instrument_weight_mode: str = 'equal_weight'   # 'equal_weight', 'min_variance', or 'risk_parity'
    corr_lookback: int = 60          # corr trailing window (CorrelationManager.lookback) AND universe liveness threshold (UniverseManager.min_history_bars), in corr_timeframe bars; >= 32, <= deque maxlen
    corr_step_size: int = 30              # CorrelationManager auto-recalc cadence in completed corr_timeframe periods; 0 disables
    corr_timeframe: str = '1d'            # data-handler timeframe read by both UniverseManager (history_timeframe) and CorrelationManager (timeframe)
    corr_mode: str = 'absolute_price_chg' # CorrelationManager.mode — 'absolute_price_chg' (futures-safe: negative/zero prices) or 'simple_return' (positive-price assets)
    corr_floor: Optional[float] = None    # CorrelationManager.floor — element-wise floor on the inline-derived rho; None (default) disables; 0.0 is the recommended Carver-style setting (zero out spurious negative correlations; bounds pre-cap IDM by sqrt(N))
    corr_shrinkage: Optional[str] = 'ledoit_wolf'  # CorrelationManager.shrinkage — shrinkage on the inline-derived rho ('ledoit_wolf' — well-conditioned at high N); None disables (raw sample corr)
    idm_cap: Optional[float] = 2.5       # VolTargetingRiskManager.idm_cap — cap on the auto-updated IDM; None disables (Carver's 2.5; >= 1.0 since DM >= 1 for long-only sum-to-1 weights)

    # NOTE: size_mode and position_size are consumed only by
    # SimpleRiskManager (sign-of-forecast follower). Ignored when
    # wiring VolTargetingRiskManager.
    size_mode: str = 'fixed_quantity'   # 'fixed_quantity' (contracts — futures default), 'fixed_notional', 'fixed_equity_pct'
    position_size: float = 10_000.0

    # --- Execution ---
    # NOTE: slippage / commission are per-symbol (InstrumentConfig). There is
    # no run-level fill-timing knob: orders fill on the signal bar (MKT at
    # close, LMT at the limit if touched) — see execution/_backtest.py.

    def __post_init__(self):
        if not self.symbols:
            raise ValueError("symbols list must not be empty.")

        # Default timeframes to {base_timeframe: 500} if empty
        if not self.timeframes:
            self.timeframes = {self.base_timeframe: 500}

        if self.base_timeframe not in self.timeframes:
            raise ValueError(
                f"base_timeframe '{self.base_timeframe}' must be a key in timeframes dict."
            )

        # Import here to avoid circular dependency at module level
        from data import parse_timeframe_to_seconds
        base_secs = parse_timeframe_to_seconds(self.base_timeframe)
        for tf in self.timeframes:
            tf_secs = parse_timeframe_to_seconds(tf)
            if tf != self.base_timeframe and tf_secs <= base_secs:
                raise ValueError(
                    f"Timeframe '{tf}' must be strictly larger than "
                    f"base_timeframe '{self.base_timeframe}'."
                )

        # Domain-value validation (vol-target knobs, corr knobs, size
        # modes, days_convention) is owned by the consuming modules and
        # fires at wiring time: VolTargetingRiskManager.__init__ (vol-target
        # knobs), UniverseManager.__init__ / CorrelationManager.__init__
        # (corr knobs), SimpleRiskManager.__init__, volatility.bars_per_year.
