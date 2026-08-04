"""
Validation-suite smoke run: EWMAC fast/slow parameter sweep on a small
crypto subset, exercising param_sweep (+ cell cache), heatmap, walk-forward,
periodic stats, bootstrap, and the Deflated Sharpe Ratio end-to-end.
Invoke from the repo root. Target runtime: ~1-2 minutes.
"""

import logging
import os
import queue
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.getcwd())

from logging_setup import configure_logging
configure_logging(level=logging.WARNING)
# Per-cell progress lines (best-effort: emitted only if the root handler
# passes INFO; harmless if filtered — the print sections are the real output).
logging.getLogger('validation._sweep').setLevel(logging.INFO)

from backtester import Backtester
from config import uniform_registry
from correlation import CorrelationManager
from data import HistoricDataHandler
from execution import BacktestExecution, CommissionModel, SlippageModel
from portfolio import BacktestPortfolio, PortfolioMarginModel
from riskmanager import VolTargetingRiskManager
from strategy import EWMACStrategy
from universe import UniverseManager
from validation import (bootstrap_stats, deflated_sharpe, load_sweep,
                        param_sweep, periodic_stats, walk_forward)
from volatility import EWMAVolEstimator, bars_per_year

# --- Data: 10 longest-history symbols from the bundled daily sample -------
sample_csv = os.path.join(os.path.dirname(__file__), '..', 'sample_data',
                          'crypto_1d.csv')
_raw = pd.read_csv(sample_csv)
_raw['timestamp'] = pd.to_datetime(_raw['timestamp'], utc=True)
_grouped = {sym: g for sym, g in _raw.groupby('symbol')
            if sym not in ('USDC_USDT:USDT', 'USTC_USDT:USDT')}
_counts = pd.Series({sym: len(g) for sym, g in _grouped.items()})
# deterministic selection: row count desc, ties broken alphabetically (58
# symbols share the max history in the bundled CSV — an unstable sort would
# pick an implementation-dependent 10)
# (selection order: count desc; SYMBOLS is then re-sorted alphabetically for
# presentation)
_by_history = sorted(_counts.index, key=lambda s: (-_counts[s], s))
SYMBOLS = sorted(_by_history[:10])
DATA = {
    sym: _grouped[sym].set_index('timestamp')[
        ['Open', 'High', 'Low', 'Close', 'Volume']]
    for sym in SYMBOLS
}
INSTRUMENTS = uniform_registry(
    SYMBOLS, point_value=1.0, fractional=True,
    slippage=SlippageModel('absolute', 0.0),
    commission=CommissionModel('per_contract', 0.0),
    margin=PortfolioMarginModel.from_leverage(10.0,
                                              maintenance_margin_rate=0.05),
)
INITIAL_CAPITAL = 10_000_000


def make_run(fast, slow):
    """One fresh engine graph per cell: single-variation EWMAC(fast, slow)
    on the 10-symbol subset; returns the portfolio (the factory contract)."""
    events_queue = queue.Queue()
    data_handler = HistoricDataHandler(events_queue, SYMBOLS,
                                       base_timeframe='1d',
                                       timeframes={'1d': 500}, data=DATA)
    strategy = EWMACStrategy(
        data_handler, SYMBOLS,
        variations={f'{fast}_{slow}': {'fast': fast, 'slow': slow}},
        forecast_scalar_lookback=256)
    portfolio = BacktestPortfolio(events_queue, data_handler, SYMBOLS,
                                  instruments=INSTRUMENTS,
                                  initial_capital=INITIAL_CAPITAL)
    vol = EWMAVolEstimator(SYMBOLS, data_handler=data_handler,
                           bars_per_year=bars_per_year('1d', 'calendar'),
                           timeframe='1d', span=36)
    # Fresh universe/correlation managers per cell, same as every other
    # stateful module the factory contract requires (param_sweep re-runs
    # this factory once per grid cell).
    universe_manager = UniverseManager(
        strategy, data_handler, min_history_bars=60, history_timeframe='1d')
    correlation_manager = CorrelationManager(
        data_handler, universe_manager,
        lookback=60, step_size=30, timeframe='1d',
        mode='absolute_price_chg', floor=None, shrinkage='ledoit_wolf')
    risk_manager = VolTargetingRiskManager(
        portfolio, strategy, vol, universe_manager=universe_manager,
        instruments=INSTRUMENTS, annual_target_vol=1_000_000,
        instrument_weight_mode='equal_weight')
    execution = BacktestExecution(events_queue, instruments=INSTRUMENTS)
    Backtester(events_queue, data_handler, strategy, portfolio,
               risk_manager, execution, universe_manager,
               correlation_manager).run()
    return portfolio


# --- Sweep (fresh timestamped cache dir; exercises write AND load paths) --
_ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')
CACHE_DIR = os.path.join('results', 'sweeps', f'{_ts}Z_ewmac-sweep-smoke')
GRID = {'fast': [4, 8, 16, 32, 64], 'slow': [16, 32, 64, 128, 256]}

sweep = param_sweep(make_run, grid=GRID,
                    where=lambda fast, slow: fast < slow,
                    timeframe='1d', days_convention='calendar',
                    cache_dir=CACHE_DIR)
print(f"\nSweep cached: {CACHE_DIR}")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(f"\n{'=' * 80}\n  SWEEP TABLE (stats from {sweep.stats_start_resolved})"
      f"\n{'=' * 80}")
print(sweep.table.to_string())

# offline-load parity check — the archive must reproduce the live table
loaded = load_sweep(CACHE_DIR)
pd.testing.assert_frame_equal(sweep.table, loaded.table)
print("\nload_sweep parity: OK")

print(f"\n{'=' * 80}\n  SHARPE HEATMAP\n{'=' * 80}")
hm = sweep.heatmap('Sharpe Ratio')
print(hm.heatmap.to_string())
print(f"best cell (y={hm.y}, x={hm.x}): {hm.best_cell}")

print(f"\n{'=' * 80}\n  WALK-FORWARD (yearly, anchored)\n{'=' * 80}")
wf = walk_forward(sweep, oos='YE')
print(wf.folds.to_string(index=False))
print("\n--- Stitched OOS stats ---")
print(wf.stitched_stats.to_string())

best = sweep.best('Sharpe Ratio')
print(f"\n{'=' * 80}\n  BEST CELL {best}\n{'=' * 80}")

print("\n--- Periodic stats (yearly) ---")
print(periodic_stats(sweep.equity(**best), sweep.trades(**best),
                     initial_capital=sweep.initial_capital(**best),
                     timeframe='1d', days_convention='calendar').to_string())

print("\n--- Bootstrap (stationary, 2000 resamples) ---")
boot = bootstrap_stats(sweep.equity(**best),
                       initial_capital=sweep.initial_capital(**best),
                       timeframe='1d', days_convention='calendar',
                       start=sweep.stats_start_resolved,
                       n_resamples=2000, seed=0)
print(boot.table.to_string())
print(f"block length: {boot.block_length:.1f} bars")

print("\n--- Deflated Sharpe ---")
dsr = deflated_sharpe(sweep)
print(f"winner:            {dsr.winner_params}")
print(f"Sharpe (ann.):     {dsr.sharpe_annualized:.3f}")
print(f"SR0 haircut (ann.): {dsr.sr0_annualized:.3f}  "
      f"(N={dsr.n_trials} trials, var_sr={dsr.var_sr:.5f})")
print(f"DSR:               {dsr.dsr:.3f}  "
      f"(P[true Sharpe > 0 after {dsr.n_trials} trials])")
