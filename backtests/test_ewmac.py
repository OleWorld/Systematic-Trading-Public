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
from config import BacktestConfig
from data import HistoricDataHandler
from strategy import EWMACStrategy
from portfolio import BacktestPortfolio
from execution import BacktestExecution, SlippageModel, CommissionModel
from volatility import EWMAVolEstimator, bars_per_year
from riskmanager import CarverVolTargetingRiskManager
from backtester import Backtester
from plotting import plot_strategy

# --- Config (validated parameter holder) ---
# EWMAC defaults need ~756 daily bars of warmup (256-day slow EMA + 500-bar
# forecast-scalar SMA). The 2021-2025 window gives ~1825 daily bars — plenty
# for warmup AND post-warmup signal emission.
config = BacktestConfig(
    symbols=['BTC_USDT:USDT', 'BNB_USDT:USDT', 'SOL_USDT:USDT', 'DOGE_USDT:USDT', 'ETH_USDT:USDT'],
    instrument_weight_mode = 'min_variance',
    corr_mode='simple_return',              # crypto-style: positive prices, pct-change correlations
    start_date='2021-01-01',
    end_date='2025-12-31',
    base_timeframe='1d',
    days_convention='calendar',
    timeframes={'1d': 500},
    initial_capital=1_000_000.0,
    leverage=10.0,
    annualized_target_vol=0.5,
    vol_target_mode='percent_volatility',   # crypto-style: τ is a fraction of current equity
    position_buffer=0.25,
    slippage_mode='pct',
    slippage_value=0.001,
    commission_mode='rate',                 # crypto-style: bps on notional
    commission_value=0.001,
    fill_on='signal_close',
)

# --- Load market data: {symbol: OHLCV DataFrame} ---
# The user supplies their own data as a {symbol: DataFrame} dict. Here we load a
# bundled CSV of daily bars; each frame is indexed by a tz-aware DatetimeIndex
# with Open/High/Low/Close/Volume columns. Built in config.symbols order.
sample_csv = os.path.join(os.path.dirname(__file__), 'sample_data', 'crypto_1d.csv')
_raw = pd.read_csv(sample_csv)
_raw['timestamp'] = pd.to_datetime(_raw['timestamp'], utc=True)
_grouped = {sym: g for sym, g in _raw.groupby('symbol')}
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

strategy = EWMACStrategy(
    data_handler, config.symbols,
    lookback_pairs=[(4, 16), (16, 64), (32, 128)],
    weights=[0.42, 0.16, 0.42],
    fdm=1.12,
    vol_lookback=25,
    forecast_scalar_lookback=500,
)

portfolio = BacktestPortfolio(
    events_queue, data_handler, config.symbols,
    initial_capital=config.initial_capital,
    leverage=config.leverage,
)

vol_timeframe = '1d'
vol_estimator = EWMAVolEstimator(
    config.symbols, data_handler=data_handler,
    bars_per_year=bars_per_year(vol_timeframe, config.days_convention),
    timeframe=vol_timeframe, span=36,
)

risk_manager = CarverVolTargetingRiskManager(
    portfolio, strategy, vol_estimator,
    data_handler=data_handler,
    annualized_target_vol=config.annualized_target_vol,
    vol_target_mode=config.vol_target_mode,
    position_buffer=config.position_buffer,
    instrument_weight_mode=config.instrument_weight_mode,
    corr_lookback=config.corr_lookback,
    corr_step_size=config.corr_step_size,
    corr_timeframe=config.corr_timeframe,
    corr_mode=config.corr_mode,
)

execution = BacktestExecution(
    events_queue,
    slippage_model=SlippageModel(config.slippage_mode, config.slippage_value),
    commission_model=CommissionModel(mode=config.commission_mode,
                                     value=config.commission_value),
    fill_on=config.fill_on,
)

bt = Backtester(events_queue, data_handler, strategy, portfolio,
                risk_manager, execution)

# --- Run ---
bt.run()

# --- Portfolio results ---
portfolio = bt.portfolio

equity_df = portfolio.get_equity_curve()
trade_df = portfolio.get_trade_log()
order_df = portfolio.get_order_log()

# Print wide record frames in full: show every column, one row per line.
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

print(f"\n--- Positions & P&L by symbol ---")
for sym in config.symbols:
    print(f"  {sym} position: {portfolio.positions[sym]:.6f} | "
          f"realized P&L: ${portfolio.realized_pnl[sym]:,.2f} | "
          f"unrealized P&L: ${portfolio.unrealized_pnl[sym]:,.2f}")

print(f"\n--- Final allocator state ---")
print(f"  IDM:                {bt.risk_manager.idm:.4f}")
print(f"  Instrument weights:")
for sym, w in bt.risk_manager.instrument_weight.items():
    print(f"    {sym:<22} {w:.4f}")

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
if not trade_df.empty:
    print(f"\n--- Trade Log (last 10) ---")
    print(trade_df.tail(10).to_string(index=False))

# =====================================================================
#  4. FORECAST SUMMARY  (one symbol — all symbols share the same format)
# =====================================================================
print(f"\n{'='*80}")
print("  FORECAST SUMMARY")
print(f"{'='*80}")

# --- Forecast sanity check (post-warmup avg |f| should approach 50). ---
strategy_records = bt.strategy.get_records(_fc_symbol)
post_warmup_forecasts = (
    strategy_records['forecast'].dropna()
    if not strategy_records.empty else pd.Series(dtype=float)
)
if len(post_warmup_forecasts) > 0:
    print(f"\n--- Forecast diagnostics ({_fc_symbol}) ---")
    print(f"  Non-NaN forecasts: {len(post_warmup_forecasts)}")
    print(f"  Mean |forecast|:   {np.mean(np.abs(post_warmup_forecasts)):.2f}  (target ≈ 50)")
    print(f"  Min / Max:         {post_warmup_forecasts.min():.2f} / {post_warmup_forecasts.max():.2f}")

# --- Risk-manager sizing diagnostics ---
riskmanager_records = bt.risk_manager.get_records(_fc_symbol)
if not riskmanager_records.empty:
    skip_counts = riskmanager_records['skip_reason'].value_counts(dropna=False).to_dict()
    print(f"\n--- Risk-manager diagnostics ({_fc_symbol}) ---")
    print(f"  Rows recorded:     {len(riskmanager_records)}")
    print(f"  Submitted orders:  {int(riskmanager_records['submitted'].sum())}")
    print(f"  skip_reason counts: {skip_counts}")

# --- Per-bar record tables (last 10 bars, full frame) ---
if not strategy_records.empty:
    print(f"\n--- Strategy records: last 10 bars ({_fc_symbol}) ---")
    print(strategy_records.tail(10).to_string())
if not riskmanager_records.empty:
    print(f"\n--- Risk-manager records: last 10 bars ({_fc_symbol}) ---")
    print(riskmanager_records.tail(10).to_string())


# import plotly.express as px
# import pandas as pd
# df = bt.strategy.get_records("BTC_USDT:USDT")
# fig = plot_strategy(df,
#                     indicators={'fast_ema_16_64': 1, 'slow_ema_16_64': 1,
#                                 'forecast_16_64': 2, 'forecast_32_128': 2,
#                                 'forecast_64_256': 2, 'forecast': 2},
#                     title='BTC_USDT:USDT EWMAC', timeframe='1d')
# fig.show(config=dict({'scrollZoom':True}))
# fig = px.line(equity_df['realized_pnl'].apply(pd.Series) + equity_df['unrealized_pnl'].apply(pd.Series))
# fig.show()
# fig = px.line(equity_df[['account_balance', 'available_balance']])
# fig.show()