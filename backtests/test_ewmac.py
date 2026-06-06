import sys
import os
import logging
import queue
import pandas as pd
sys.path.insert(0, os.getcwd())

from logging_setup import configure_logging
configure_logging(level=logging.WARNING)

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
    start_date='2021-01-01',
    end_date='2025-12-31',
    base_timeframe='1d',
    convention='crypto',
    timeframes={'1d': 500},
    initial_capital=1_000_000.0,
    leverage=10.0,
    annualized_target_vol=0.5,
    position_buffer=0.25,
    slippage_mode='pct',
    slippage_value=0.001,
    commission_rate=0.001,
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
    lookback_pairs=[(16, 64), (32, 128), (64, 256)],
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
    bars_per_year=bars_per_year(vol_timeframe, config.convention),
    timeframe=vol_timeframe, span=36,
)

risk_manager = CarverVolTargetingRiskManager(
    portfolio, strategy, vol_estimator,
    data_handler=data_handler,
    annualized_target_vol=config.annualized_target_vol,
    position_buffer=config.position_buffer,
    instrument_weight_mode=config.instrument_weight_mode,
    corr_lookback=config.corr_lookback,
    corr_step_size=config.corr_step_size,
    corr_timeframe=config.corr_timeframe,
)

execution = BacktestExecution(
    events_queue,
    slippage_model=SlippageModel(config.slippage_mode, config.slippage_value),
    commission_model=CommissionModel(rate=config.commission_rate),
    fill_on=config.fill_on,
)

bt = Backtester(events_queue, data_handler, strategy, portfolio,
                risk_manager, execution)

# --- Run ---
bt.run()

# --- Portfolio results ---
portfolio = bt.portfolio

print(f"\n{'='*80}")
print("  PORTFOLIO SUMMARY")
print(f"{'='*80}")

equity_df = portfolio.get_equity_curve()
trade_df = portfolio.get_trade_log()
order_df = portfolio.get_order_log()

print(f"  Initial capital: ${portfolio.initial_capital:,.2f}")
print(f"  Final cash:      ${portfolio.cash:,.2f}")
if not equity_df.empty:
    final_balance = equity_df['account_balance'].iloc[-1]
    pnl = final_balance - portfolio.initial_capital
    print(f"  Final balance:   ${final_balance:,.2f}")
    print(f"  Total P&L:       ${pnl:,.2f} ({pnl/portfolio.initial_capital*100:.2f}%)")
    print(f"  Available bal:   ${equity_df['available_balance'].iloc[-1]:,.2f}")
    print(f"\n--- Per-bar returns (last 5 rows) ---")
    print(equity_df[['account_balance', 'simple_return', 'log_return']].tail().to_string())
    print(f"\n--- Return summary ---")
    print(equity_df[['simple_return', 'log_return']].describe().to_string())

print(f"\n  Orders placed: {len(order_df)}")
print(f"  Trades filled: {len(trade_df)}")
print(f"  Total commission: ${portfolio.total_commission:,.2f}")
for sym in config.symbols:
    print(f"  {sym} position: {portfolio.positions[sym]:.6f} | realized P&L: ${portfolio.realized_pnl[sym]:,.2f}")

if not trade_df.empty:
    print(f"\n--- Trade Log (last 10) ---")
    print(trade_df.tail(10).to_string(index=False))

# --- Forecast sanity check (post-warmup avg |f| should approach 50). ---
import numpy as np
strategy_records = bt.strategy.get_records('BTC_USDT:USDT')
post_warmup_forecasts = strategy_records['forecast'].dropna()
if len(post_warmup_forecasts) > 0:
    print(f"\n--- Forecast diagnostics (BTC) ---")
    print(f"  Non-NaN forecasts: {len(post_warmup_forecasts)}")
    print(f"  Mean |forecast|:   {np.mean(np.abs(post_warmup_forecasts)):.2f}  (target ≈ 50)")
    print(f"  Min / Max:         {post_warmup_forecasts.min():.2f} / {post_warmup_forecasts.max():.2f}")

# --- Risk-manager sizing diagnostics ---
riskmanager_records = bt.risk_manager.get_records('BTC_USDT:USDT')
if not riskmanager_records.empty:
    skip_counts = riskmanager_records['skip_reason'].value_counts(dropna=False).to_dict()
    print(f"\n--- Risk-manager diagnostics (BTC) ---")
    print(f"  Rows recorded:     {len(riskmanager_records)}")
    print(f"  Submitted orders:  {int(riskmanager_records['submitted'].sum())}")
    print(f"  skip_reason counts: {skip_counts}")
    print(f"\n  Last 10 bars:")
    print(riskmanager_records[
        ['forecast', 'sigma', 'instrument_weight', 'strategy_weight', 'capital',
         'target_qty', 'current_qty', 'trade_qty',
         'submitted', 'skip_reason']
    ].tail(10).to_string())

print(f"\n--- Final allocator state ---")
print(f"  IDM:                {bt.risk_manager.idm:.4f}")
print(f"  Instrument weights:")
for sym, w in bt.risk_manager.instrument_weight.items():
    print(f"    {sym:<22} {w:.4f}")


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