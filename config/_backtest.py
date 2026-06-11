"""BacktestConfig — single source of truth for all backtest infrastructure parameters."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BacktestConfig:
    """
    Validated parameter holder for a backtest run.

    Centralizes the infrastructure parameters (data window, portfolio
    sizing, vol-target knobs, slippage / commission, fill timing) and
    validates them at construction. Callers read the fields off this
    object as they wire each module manually (see
    ``backtests/test_ewmac.py`` for a worked example).
    """

    # --- Data ---
    symbols: List[str]
    start_date: str
    end_date: str
    base_timeframe: str                                         # streaming TF (e.g. '1m')
    convention: str                                             # 'crypto' (365 days/year, 24/7) or 'tradfi' (252 days/year)
    timeframes: Dict[str, int] = field(default_factory=dict)    # {tf: maxlen} e.g. {'1m': 500, '1h': 500, '4h': 200}

    # --- Portfolio ---
    initial_capital: float = 100_000.0
    leverage: float = 1.0

    # --- Risk / Sizing ---
    # Carver vol-targeting knobs consumed by `CarverVolTargetingRiskManager`.
    # ``idm`` is not in config — pass it directly to the risk manager
    # constructor if a non-default value is needed.
    annualized_target_vol: float = 0.25  # Carver's τ; annualized vol target (default 0.25 = 25 %)
    position_buffer: float = 0.25        # Carver §10.7 dead-band (0.0 to trade every gap)
    instrument_weight_mode: str = 'equal_weight'   # 'equal_weight' or 'min_variance'
    corr_lookback: int = 500              # trailing window for correlation (in corr_timeframe bars)
    corr_step_size: int = 30              # auto-recalc cadence in completed bars; 0 disables
    corr_timeframe: str = '1d'            # data-handler timeframe to read closes from
    corr_mode: str = 'simple_return'      # 'simple_return' or 'absolute_price_chg' (futures/spreads with negative/zero prices)

    # NOTE: size_mode and position_size are consumed only by
    # SimpleRiskManager (sign-of-forecast follower). Ignored when
    # wiring CarverVolTargetingRiskManager.
    size_mode: str = 'fixed_notional'   # 'fixed_notional', 'fixed_quantity', 'fixed_equity_pct'
    position_size: float = 10_000.0

    # --- Execution ---
    slippage_mode: str = 'pct'          # 'pct' or 'absolute'
    slippage_value: float = 0.0
    commission_rate: float = 0.0
    fill_on: str = 'signal_close'       # 'signal_close' or 'next_open'

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

        if self.slippage_mode not in ('pct', 'absolute'):
            raise ValueError(
                f"Unknown slippage_mode: '{self.slippage_mode}'. "
                "Must be 'pct' or 'absolute'."
            )
        if self.fill_on not in ('signal_close', 'next_open'):
            raise ValueError(
                f"Unknown fill_on: '{self.fill_on}'. "
                "Must be 'signal_close' or 'next_open'."
            )
        if self.size_mode not in ('fixed_notional', 'fixed_quantity', 'fixed_equity_pct'):
            raise ValueError(
                f"Unknown size_mode: '{self.size_mode}'. "
                "Must be 'fixed_notional', 'fixed_quantity', or 'fixed_equity_pct'."
            )
        if self.convention not in ('crypto', 'tradfi'):
            raise ValueError(
                f"Unknown convention: '{self.convention}'. "
                "Must be 'crypto' (365 days/year, 24/7) or "
                "'tradfi' (252 days/year)."
            )
        # Mirror CarverVolTargetingRiskManager constructor validation so
        # bad values fail at config construction, not deep in the wiring.
        if not (0 < self.annualized_target_vol < 1):
            raise ValueError(
                f"annualized_target_vol must be in (0, 1), "
                f"got {self.annualized_target_vol}"
            )
        if not (0.0 <= self.position_buffer < 1.0):
            raise ValueError(
                f"position_buffer must be in [0, 1), got {self.position_buffer}"
            )
        if self.instrument_weight_mode not in ('equal_weight', 'min_variance'):
            raise ValueError(
                f"Unknown instrument_weight_mode: '{self.instrument_weight_mode}'. "
                "Must be 'equal_weight' or 'min_variance'."
            )
        if self.corr_lookback < 2:
            raise ValueError(
                f"corr_lookback must be >= 2, got {self.corr_lookback}"
            )
        if self.corr_step_size < 0:
            raise ValueError(
                f"corr_step_size must be >= 0, got {self.corr_step_size}"
            )
        if self.corr_mode not in ('simple_return', 'absolute_price_chg'):
            raise ValueError(
                f"Unknown corr_mode: {self.corr_mode!r}. "
                "Must be 'simple_return' or 'absolute_price_chg'."
            )
