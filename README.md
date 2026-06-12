## Quick Start — What the Trader Implements

A trader only needs to care about **3 things**. The system handles everything else.

### 1. Config — Define your backtest parameters

```python
from config import BacktestConfig

config = BacktestConfig(
    symbols=['BTC_USDT', 'BNB_USDT'],
    start_date='2026-01-01',
    end_date='2026-04-04',
    base_timeframe='4h',
    days_convention='calendar',   # 'calendar' (365 d/y, 24/7) or 'business' (252 trading d/y)
    # Timeframes — {tf: maxlen}. Omit for single-TF (defaults to {base: 500}).
    # For multi-TF: timeframes={'1m': 500, '1h': 500, '4h': 200},
    initial_capital=1_000_000.0,
    leverage=5.0,
    annualized_target_vol=250_000.0,        # Carver τ — REQUIRED; units depend on vol_target_mode
    vol_target_mode='dollar_volatility',    # 'dollar_volatility' (fixed annual $ vol budget — default)
                                            # or 'percent_volatility' (fraction of equity, e.g. 0.25)
    position_buffer=0.25,         # Carver §10.7 dead-band (0.0 trades every gap)
    slippage_mode='absolute',     # 'absolute' ($ per unit — default) or 'pct' (% of price)
    slippage_value=0.5,
    commission_mode='per_contract',  # 'per_contract' ($ per contract — default) or 'rate' (bps on notional)
    commission_value=2.5,
    fill_on='signal_close',       # 'signal_close' or 'next_open'
)
```

### 2. Strategy — Implement `calculate_forecast()`

```python
from strategy import Strategy
from indicator import SMA

class MyStrategy(Strategy):
    def __init__(self, data_handler, symbol_list, fast=10, slow=30):
        super().__init__(data_handler, symbol_list)
        self.fast = fast
        self.slow = slow
        self._fast_sma = {s: SMA(length=fast) for s in symbol_list}
        self._slow_sma = {s: SMA(length=slow) for s in symbol_list}

    def calculate_forecast(self, event):
        sym = event.symbol
        self._fast_sma[sym].update(event.timestamp, event.close)
        self._slow_sma[sym].update(event.timestamp, event.close)

        fast = self._fast_sma[sym].latest
        slow = self._slow_sma[sym].latest
        if fast is None or slow is None:
            return None                    # warmup — record OHLCV only

        # Scale the raw crossover to the project ±100 forecast convention.
        # Cap at FORECAST_CAP; the base class also clamps before caching.
        raw = (fast - slow) / slow * 1000.0
        forecast = max(-Strategy.FORECAST_CAP, min(Strategy.FORECAST_CAP, raw))
        return {'fast_sma': fast, 'slow_sma': slow, 'forecast': forecast}
```

**Available inside `calculate_forecast`:**

- `self.data_handler.get_latest_bars(symbol, n)` — lookback DataFrame at base TF. `iloc[-1]` is the **forming** bar (in live: mutates as ticks arrive; in backtest: equals the final bar); `iloc[-2]` is the most recent completed bar.
- `self.data_handler.get_latest_bars(symbol, n, '4h')` — lookback at a registered higher TF. Same convention: `iloc[-1]` is the forming HTF bar (aggregation of completed base bars in the current HTF period); `iloc[-2]` is the most recent completed HTF bar. Use `iloc[-2]` and earlier for logic that must only see closed bars.
- Per-symbol stateful indicators (`SMA`, `EMA`, `KAMA`, `RSI`, `ATR`, ...) fed one scalar per bar via `indicator.update(timestamp, ...)` — read finalized values via `indicator.latest`.
- Return a dict containing a `'forecast'` key in `[-Strategy.FORECAST_CAP, +Strategy.FORECAST_CAP]` (plus any indicators you want recorded). Return `None` during warmup to record OHLCV only and leave the cached forecast unchanged. The risk manager reads `strategy.get_forecast(symbol)` to derive the target position.

**Multi-timeframe**: Register higher TFs via `timeframes` in config. `get_latest_bars(symbol, n, timeframe)` returns `n` bars where the last row is the **forming** HTF bar — the aggregation of completed base bars that fell into the current HTF period. Signal logic that must compare closed bars should read `iloc[-2]` and earlier (e.g. a crossover on completed HTF bars compares `iloc[-3]` vs `iloc[-2]`).

### 3. Position Sizing — Carver vol-targeting (default choice)

`CarverVolTargetingRiskManager` implements Carver's cash-vol framework:

```text
# vol_target_mode='dollar_volatility' (default — fixed annual $ vol budget):
target_qty = (IDM × weight × annualized_target_vol × forecast / 50)
             / annualized_$_vol

# vol_target_mode='percent_volatility' (τ as a fraction of current equity):
target_qty = (capital × IDM × weight × annualized_target_vol × forecast / 50)
             / annualized_$_vol
```

So `|forecast| = 50` reproduces Carver's basic vol target and `|forecast| = 100`
doubles it. The knobs you tune live on `BacktestConfig`:

- `annualized_target_vol` — Carver's τ (REQUIRED, no default). A dollar amount
  (e.g. `250_000`) under `'dollar_volatility'` — the cash-vol budget stays fixed
  as the account grows/shrinks (institutional futures convention: the risk limit
  is a dollar number reset periodically). A fraction in `(0, 1)` (e.g. `0.25`)
  under `'percent_volatility'` — sizes compound with equity.
- `vol_target_mode` — `'dollar_volatility'` (default) or `'percent_volatility'`
- `position_buffer` — Carver §10.7 dead-band (default `0.25`; `0.0` trades every gap)

For simple sign-of-forecast sizing (fixed notional / fixed quantity / fixed
equity fraction), swap in `SimpleRiskManager`:

```python
from riskmanager import SimpleRiskManager

risk_manager = SimpleRiskManager(
    portfolio, strategy,
    size_mode='fixed_quantity',     # default; or 'fixed_notional' / 'fixed_equity_pct'
    position_size=10.0,             # contracts under 'fixed_quantity'
)
```

`size_mode` and `position_size` on `BacktestConfig` are read only by
`SimpleRiskManager`; `CarverVolTargetingRiskManager` ignores them.

### Data — supply your own OHLCV

You provide market data as a `{symbol: DataFrame}` dict — this is the only way data
enters the engine. Each DataFrame is indexed by a timezone-aware `DatetimeIndex` and
exposes `Open`/`High`/`Low`/`Close`/`Volume` columns; sourcing, cleaning, and windowing
the data is up to you. A small bundled sample of daily bars lives at
[backtests/sample_data/crypto_1d.csv](backtests/sample_data/crypto_1d.csv):

```python
import pandas as pd

raw = pd.read_csv('backtests/sample_data/crypto_1d.csv')
raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
data = {sym: g.set_index('timestamp')[['Open', 'High', 'Low', 'Close', 'Volume']]
        for sym, g in raw.groupby('symbol')}
```

### Run — wire the modules and start the loop

The trader instantiates each module explicitly, passes them into
`Backtester(...)`, and calls `run()`. The full pattern lives in
[backtests/test_ewmac.py](backtests/test_ewmac.py); the condensed shape is:

```python
import queue
from data import HistoricDataHandler
from portfolio import BacktestPortfolio
from execution import BacktestExecution, SlippageModel, CommissionModel
from volatility import EWMAVolEstimator, bars_per_year
from riskmanager import CarverVolTargetingRiskManager
from backtester import Backtester

events_queue = queue.Queue()
data_handler = HistoricDataHandler(events_queue, config.symbols,
                                   base_timeframe=config.base_timeframe,
                                   timeframes=config.timeframes, data=data)
strategy     = MyStrategy(data_handler, config.symbols, fast=10, slow=30)
portfolio    = BacktestPortfolio(events_queue, data_handler, config.symbols,
                                 initial_capital=config.initial_capital,
                                 leverage=config.leverage)
vol_estimator = EWMAVolEstimator(config.symbols, data_handler=data_handler,
                                 bars_per_year=bars_per_year('1d', config.days_convention),
                                 timeframe='1d', span=36)
risk_manager  = CarverVolTargetingRiskManager(portfolio, strategy, vol_estimator,
                                              data_handler=data_handler,
                                              annualized_target_vol=config.annualized_target_vol,
                                              vol_target_mode=config.vol_target_mode,
                                              position_buffer=config.position_buffer)
execution     = BacktestExecution(events_queue,
                                  slippage_model=SlippageModel(config.slippage_mode, config.slippage_value),
                                  commission_model=CommissionModel(mode=config.commission_mode,
                                                                   value=config.commission_value),
                                  fill_on=config.fill_on)

bt = Backtester(events_queue, data_handler, strategy, portfolio, risk_manager, execution)
bt.run()

# Access results
equity_df = portfolio.get_equity_curve()
trade_df  = portfolio.get_trade_log()
records   = bt.strategy.get_records('BTC_USDT')
```