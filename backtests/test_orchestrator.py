import sys
import os
import logging
import queue
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())

from logging_setup import configure_logging
configure_logging(level=logging.WARNING)

from analytics import backtest_stats
from config import BacktestConfig, uniform_registry
from data import HistoricDataHandler
from strategy import EWMACStrategy, RSIMRStrategy
from orchestrator import Orchestrator
from portfolio import BacktestPortfolio, PortfolioMarginModel
from execution import BacktestExecution, SlippageModel, CommissionModel
from volatility import EWMAVolEstimator, bars_per_year
from riskmanager import VolTargetingRiskManager
from backtester import Backtester


# --- Load market data: {symbol: OHLCV DataFrame} ---
# Same bundled crypto-perp basket as the per-strategy smoke runners. The
# orchestrator runs EWMAC and RSIMR together, combining their forecasts.
sample_csv = os.path.join(os.path.dirname(__file__), 'sample_data', 'crypto_1d.csv')
_raw = pd.read_csv(sample_csv)
_raw['timestamp'] = pd.to_datetime(_raw['timestamp'], utc=True)
_grouped = {sym: g for sym, g in _raw.groupby('symbol')}
stables = ['USDC_USDT:USDT', 'USTC_USDT:USDT']
for sym in stables:
    del _grouped[sym]
_symbols = list(str(x) for x in _grouped.keys())

# --- Config (validated parameter holder) ---
config = BacktestConfig(
    symbols=_symbols,
    start_date='2021-01-01',
    end_date='2026-04-23',
    base_timeframe='1d',
    days_convention='calendar',             # crypto is 24/7 → 365 d/y
    timeframes={'1d': 500},

    instrument_weight_mode='risk_parity',
    corr_mode='absolute_price_chg',
    corr_lookback=256,
    corr_timeframe='1d',

    initial_capital=10_000_000,
    vol_target_mode='dollar_volatility',
    annual_target_vol=1_000_000,            # $1M annual vol
    position_buffer=0.25,

    fill_on='signal_close',
)

instruments = uniform_registry(
    config.symbols,
    point_value=1.0,
    fractional=True,
    slippage=SlippageModel('absolute', 0.0),
    commission=CommissionModel('per_contract', 0.0),
    margin=PortfolioMarginModel.from_leverage(10.0, maintenance_margin_rate=0.05),
)


data = {
    sym: _grouped[sym].set_index('timestamp')[['Open', 'High', 'Low', 'Close', 'Volume']]
    for sym in config.symbols
}

# --- Manual module wiring ---
events_queue = queue.Queue()

data_handler = HistoricDataHandler(
    events_queue, config.symbols,
    base_timeframe=config.base_timeframe,
    timeframes=config.timeframes,
    data=data,
)

# Two strategies over the SAME universe; each keeps its own forecast logic
# and reads the shared data handler exactly as it would standalone.
ewmac = EWMACStrategy(
    data_handler, config.symbols,
    lookback_pairs=[(4, 16), (16, 64), (32, 128)],
    weights=[0.42, 0.16, 0.42],
    fdm=1.12,
    vol_lookback=25,
    forecast_scalar_lookback=500,
)
rsimr = RSIMRStrategy(
    data_handler, config.symbols,
    rsi_windows=[3, 14, 28],
    weights=[0.50, 0.25, 0.25],
    fdm=1.0,
    forecast_scalar_lookback=256,
)

# The orchestrator owns per-strategy weighting and aggregates both
# forecasts into one combined forecast per symbol (Carver weighted-sum x
# FDM, capped). It occupies the strategy slot of BOTH the risk manager and
# the Backtester — no engine or risk-manager change is needed.
orchestrator = Orchestrator(
    {'ewmac': ewmac, 'rsimr': rsimr},
    weights={'ewmac': 0.5, 'rsimr': 0.5},   # equal default; shown explicitly
    fdm=1.15,                               # cross-strategy diversification uplift
)

portfolio = BacktestPortfolio(
    events_queue, data_handler, config.symbols,
    instruments=instruments,
    initial_capital=config.initial_capital,
)

vol_timeframe = '1d'
vol_estimator = EWMAVolEstimator(
    config.symbols, data_handler=data_handler,
    bars_per_year=bars_per_year(vol_timeframe, config.days_convention),
    timeframe=vol_timeframe, span=36,
)

risk_manager = VolTargetingRiskManager(
    portfolio, orchestrator, vol_estimator,   # orchestrator in the strategy slot
    data_handler=data_handler,
    instruments=instruments,
    annual_target_vol=config.annual_target_vol,
    vol_target_mode=config.vol_target_mode,
    position_buffer=config.position_buffer,
    instrument_weight_mode=config.instrument_weight_mode,
    corr_lookback=config.corr_lookback,
    corr_step_size=config.corr_step_size,
    corr_timeframe=config.corr_timeframe,
    corr_mode=config.corr_mode,
    corr_floor=config.corr_floor,
    corr_shrinkage=config.corr_shrinkage,
    idm_cap=config.idm_cap,
)

execution = BacktestExecution(
    events_queue,
    instruments=instruments,
    fill_on=config.fill_on,
)

bt = Backtester(events_queue, data_handler, orchestrator, portfolio,
                risk_manager, execution)

# --- Run ---
bt.run()

# --- Portfolio results ---
portfolio = bt.portfolio

equity_df = portfolio.get_equity_curve()
trade_df = portfolio.get_trade_log()
order_df = portfolio.get_order_log()

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

_fc_symbol = 'BTC_USDT:USDT'  # representative symbol for forecast diagnostics

# =====================================================================
#  1. PORTFOLIO SUMMARY
# =====================================================================
print(f"\n{'='*80}")
print("  PORTFOLIO SUMMARY")
print(f"{'='*80}")

print(f"  Initial capital: ${portfolio.initial_capital:,.2f}")
print(f"  Final cash:      ${portfolio.cash:,.2f}")
if not equity_df.empty:
    final_balance = equity_df['account_balance'].iloc[-1]
    pnl = final_balance - portfolio.initial_capital
    print(f"  Final balance:   ${final_balance:,.2f}")
    print(f"  Total P&L:       ${pnl:,.2f} ({pnl/portfolio.initial_capital*100:.2f}%)")
    print(f"  Available balance:   ${equity_df['available_balance'].iloc[-1]:,.2f}")
    print(f"  Total commission: ${portfolio.total_commission:,.2f}")
    print(f"\n--- Return summary ---")
    print(equity_df[['simple_return', 'log_return']].describe().to_string())

print(f"\n--- Final allocator state ---")
print(f"  IDM:                {bt.risk_manager.idm:.4f}")
print(f"  Strategy weights:   {orchestrator.strategy_weights}  fdm={orchestrator.fdm}")

# =====================================================================
#  2. BACKTEST STATISTICS
# =====================================================================
print(f"\n{'='*80}")
print("  BACKTEST STATISTICS")
print(f"{'='*80}")

stats = backtest_stats(
    equity_df, trade_df,
    initial_capital=config.initial_capital,
    timeframe=config.base_timeframe,
    days_convention=config.days_convention,
)
print(stats.to_string())

# =====================================================================
#  3. TRADES SUMMARY
# =====================================================================
print(f"\n{'='*80}")
print("  TRADES SUMMARY")
print(f"{'='*80}")

print(f"  Orders placed: {len(order_df)}")
print(f"  Trades filled: {len(trade_df)}")

# =====================================================================
#  4. ORCHESTRATOR / FORECAST SUMMARY
# =====================================================================
print(f"\n{'='*80}")
print("  ORCHESTRATOR FORECAST SUMMARY")
print(f"{'='*80}")

orch_records = bt.strategy.get_records(_fc_symbol)        # bt.strategy is the orchestrator
combined = (
    orch_records['forecast'].dropna()
    if not orch_records.empty else pd.Series(dtype=float)
)
if len(combined) > 0:
    print(f"\n--- Combined forecast diagnostics ({_fc_symbol}) ---")
    print(f"  Non-NaN combined forecasts: {len(combined)}")
    print(f"  Mean |combined forecast|:   {np.mean(np.abs(combined)):.2f}")
    print(f"  Min / Max:                  {combined.min():.2f} / {combined.max():.2f}")

# Per-child forecasts remain reachable through the orchestrator.
for label in orchestrator.strategies:
    child_records = orchestrator.strategies[label].get_records(_fc_symbol)
    child_fc = (
        child_records['forecast'].dropna()
        if not child_records.empty else pd.Series(dtype=float)
    )
    if len(child_fc) > 0:
        print(f"  [{label:>6}] non-NaN: {len(child_fc):>4}  "
              f"mean |f|: {np.mean(np.abs(child_fc)):.2f}")

riskmanager_records = bt.risk_manager.get_records(_fc_symbol)
if not riskmanager_records.empty:
    skip_counts = riskmanager_records['skip_reason'].value_counts(dropna=False).to_dict()
    print(f"\n--- Risk-manager diagnostics ({_fc_symbol}) ---")
    print(f"  Rows recorded:     {len(riskmanager_records)}")
    print(f"  Submitted orders:  {int(riskmanager_records['submitted'].sum())}")
    print(f"  skip_reason counts: {skip_counts}")

# --- Per-bar record tables (last 10 bars, full frame) ---
if not orch_records.empty:
    print(f"\n--- Orchestrator records: last 10 bars ({_fc_symbol}) ---")
    print(orch_records.tail(10).to_string())
if not riskmanager_records.empty:
    print(f"\n--- Risk-manager records: last 10 bars ({_fc_symbol}) ---")
    print(riskmanager_records.tail(10).to_string())
