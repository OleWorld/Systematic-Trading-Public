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
from strategy import RSIMRStrategy
from portfolio import BacktestPortfolio, PortfolioMarginModel
from execution import BacktestExecution, SlippageModel, CommissionModel
from volatility import EWMAVolEstimator, bars_per_year
from riskmanager import VolTargetingRiskManager
from backtester import Backtester
from plotting import plot_strategy


# --- Load market data: {symbol: OHLCV DataFrame} ---
# The user supplies their own data as a {symbol: DataFrame} dict. Here we load a
# bundled CSV of daily bars; each frame is indexed by a tz-aware DatetimeIndex
# with Open/High/Low/Close/Volume columns. Built in config.symbols order.
sample_csv = os.path.join(os.path.dirname(__file__), 'sample_data', 'crypto_1d.csv')
_raw = pd.read_csv(sample_csv)
_raw['timestamp'] = pd.to_datetime(_raw['timestamp'], utc=True)
_grouped = {sym: g for sym, g in _raw.groupby('symbol')}
stables = ['USDC_USDT:USDT', 'USTC_USDT:USDT']
for sym in stables:
    del _grouped[sym]
_symbols = list(str(x) for x in _grouped.keys())

# --- Config (validated parameter holder) ---
# RSIMR needs only a short signal warmup (max RSI window + forecast-scalar SMA;
# ~28 + 256 ≈ 285 daily bars with the defaults below). The 2021-01 → 2026-04
# window gives ~1939 daily bars — plenty for warmup AND post-warmup trading.
#
# This smoke run exercises the engine's FUTURES-FIRST defaults (dollar vol
# target, absolute-price-change correlations, absolute slippage, per-contract
# commission) on the bundled crypto basket, now driven by the RSI mean-reverter
# instead of EWMAC. Only days_convention is data-driven: crypto trades 24/7, so
# 'calendar' (365 d/y) is required for correct vol annualization.
config = BacktestConfig(
    symbols=_symbols,
    start_date='2021-01-01',
    end_date='2026-04-23',
    base_timeframe='1d',
    days_convention='calendar',             # data-driven: crypto is 24/7 → 365 d/y
    timeframes={'1d': 500},

    instrument_weight_mode='risk_parity',  # recommended default: 1/N weights, IDM still derived from rho
    corr_mode='absolute_price_chg',         # futures default: .diff() correlations
    corr_lookback  = 256,
    corr_timeframe = '1d',

    initial_capital=10_000_000,
    vol_target_mode='dollar_volatility',    # futures default: fixed annual $ vol budget
    annual_target_vol=1_000_000,        # $1M annual vol
    position_buffer=0.25,

    fill_on='signal_close',
)

# Per-symbol economics (point_value, fractional, slippage, commission, margin)
# live in an InstrumentConfig registry, NOT BacktestConfig. The crypto basket is
# homogeneous, so a uniform registry is a one-liner: point_value=1 and
# fractional=True reproduce the simplified crypto-perp accounting, with 10x
# leverage / 5% maintenance margin and a 0.1% rate commission.
instruments = uniform_registry(
    config.symbols,
    point_value=1.0,
    fractional=True,
    slippage=SlippageModel('absolute', 0.0),     # futures default: $ per unit
    commission=CommissionModel('rate', 0.001),  # futures default: bps on notional
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

# RSI mean-reverter: three RSI speeds (fast/medium/slow) mapped through arctanh
# into a [-100, +100] forecast (oversold → long, overbought → short), with the
# same dynamic forecast scalar EWMAC uses to drive avg |f| toward 50.
strategy = RSIMRStrategy(
    data_handler, config.symbols,
    rsi_windows=[7, 14, 28],
    weights=[1.0 / 3.0] * 3,
    fdm=1.0,
    forecast_scalar_lookback=256,
)

# The portfolio reads each symbol's point_value (PnL/margin) and MarginModel
# from the instruments registry. For a heterogeneous futures book, hand a
# per-symbol dict instead of the uniform registry built above.
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
    portfolio, strategy, vol_estimator,
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
# symbol = 'DOGE_USDT:USDT'
# df = bt.strategy.get_records(symbol)
# fig = plot_strategy(df,
#                     indicators={'rsi_14': 1, 'forecast': 2},
#                     title=f'{symbol} RSI mean-reversion', timeframe='1d')
# fig.show(config=dict({'scrollZoom':True}), renderer='browser')
