## Quick Start — What the Trader Implements

A trader only needs to care about **3 things**. The system handles everything else.

### 1. Config — Define your backtest parameters

```python
from config import BacktestConfig

config = BacktestConfig(
    symbols=['BTC_USDT:USDT', 'BNB_USDT:USDT'],   # CCXT perp naming, as in the bundled sample data
    base_timeframe='4h',
    days_convention='calendar',   # 'calendar' (365 d/y, 24/7) or 'business' (252 trading d/y)
    # Timeframes — {tf: maxlen}. Omit for single-TF (defaults to {base: 500}).
    # For multi-TF: timeframes={'1m': 500, '1h': 500, '4h': 200},
    initial_capital=1_000_000.0,
    annual_target_vol=250_000.0,        # Carver τ — REQUIRED; units depend on vol_target_mode
    vol_target_mode='dollar_volatility',    # 'dollar_volatility' (fixed annual $ vol budget — default)
                                            # or 'percent_volatility' (fraction of equity, e.g. 0.25)
    position_buffer=0.25,         # Carver §10.7 dead-band (0.0 trades every gap)
)
```

`BacktestConfig` holds only **run-level** parameters. The per-symbol economics
(point value, fractional lots, slippage, commission, margin/leverage) live in an
`InstrumentConfig` registry — see the next section.

### 1b. Instruments — per-symbol economics

Each traded symbol carries an `InstrumentConfig`: its `point_value` (contract
multiplier — dollar value is `qty × point_value × price`; `1` for crypto, e.g.
`1000` for WTI crude), `fractional` flag (`True` for crypto, `False` for
whole-lot-only futures), and its own slippage / commission / margin models.
The engine consumes a registry `{symbol: InstrumentConfig}`. For a homogeneous
book, `uniform_registry` builds it in one call:

```python
from config import uniform_registry
from execution import SlippageModel, CommissionModel
from portfolio import PortfolioMarginModel

instruments = uniform_registry(
    config.symbols,
    point_value=1.0, fractional=True,                       # crypto-perp defaults
    slippage=SlippageModel('absolute', 0.5),                # $ per unit
    commission=CommissionModel('per_contract', 2.5),        # $ per contract
    margin=PortfolioMarginModel.from_leverage(5.0, maintenance_margin_rate=0.025),
)
```

For a futures book with different multipliers/margins per contract, hand a
per-symbol dict of `InstrumentConfig` instead. Your **strategy and risk-manager
logic still size in contracts** — the point-value conversion and whole-lot
rounding happen in the back-end.

### 2. Strategy — Subclass `TimeSeriesStrategy`, implement `calculate_forecast()`

```python
from strategy import TimeSeriesStrategy
from indicator import SMA

class MyStrategy(TimeSeriesStrategy):
    def __init__(self, data_handler, symbol_list, fast=10, slow=30):
        super().__init__(data_handler, symbol_list)
        self.fast = fast
        self.slow = slow
        self._fast_sma = {s: SMA(window=fast) for s in symbol_list}
        self._slow_sma = {s: SMA(window=slow) for s in symbol_list}

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
        cap = TimeSeriesStrategy.FORECAST_CAP
        forecast = max(-cap, min(cap, raw))
        return {'fast_sma': fast, 'slow_sma': slow, 'forecast': forecast}
```

`TimeSeriesStrategy` is the per-event template for strategies that read one
symbol's bars and forecast that symbol alone; the mode-agnostic `Strategy`
base underneath owns the forecast cache, clamping, warmup tracking, and
per-bar records (a future cross-sectional template will subclass it
directly).

**Available inside `calculate_forecast`:**

- `self.data_handler.get_latest_bars(symbol, n)` — lookback DataFrame at base TF. `iloc[-1]` is the **forming** bar (in live: mutates as ticks arrive; in backtest: equals the final bar); `iloc[-2]` is the most recent completed bar.
- `self.data_handler.get_latest_bars(symbol, n, '4h')` — lookback at a registered higher TF. Same convention: `iloc[-1]` is the forming HTF bar (aggregation of completed base bars in the current HTF period); `iloc[-2]` is the most recent completed HTF bar. Use `iloc[-2]` and earlier for logic that must only see closed bars.
- Per-symbol stateful indicators (`SMA`, `EMA`, `KAMA`, `RSI`, `ATR`, ...) fed one scalar per bar via `indicator.update(timestamp, ...)` — read finalized values via `indicator.latest`.
- Return a dict containing a `'forecast'` key in `[-Strategy.FORECAST_CAP, +Strategy.FORECAST_CAP]` (plus any indicators you want recorded). Return `None` during warmup to record OHLCV only and leave the cached forecast unchanged. The risk manager reads `strategy.get_forecast(symbol)` to derive the target position.

**Multi-timeframe**: Register higher TFs via `timeframes` in config. `get_latest_bars(symbol, n, timeframe)` returns `n` bars where the last row is the **forming** HTF bar — the aggregation of completed base bars that fell into the current HTF period. Signal logic that must compare closed bars should read `iloc[-2]` and earlier (e.g. a crossover on completed HTF bars compares `iloc[-3]` vs `iloc[-2]`).

### 3. Position Sizing — Carver vol-targeting (default choice)

`VolTargetingRiskManager` implements Carver's cash-vol framework:

```text
# vol_target_mode='dollar_volatility' (default — fixed annual $ vol budget):
target_qty = (IDM × weight × annual_target_vol × forecast / 50)
             / annual_$_vol

# vol_target_mode='percent_volatility' (τ as a fraction of current equity):
target_qty = (capital × IDM × weight × annual_target_vol × forecast / 50)
             / annual_$_vol
```

So `|forecast| = 50` reproduces Carver's basic vol target and `|forecast| = 100`
doubles it. The knobs you tune live on `BacktestConfig`:

- `annual_target_vol` — Carver's τ (REQUIRED, no default). A dollar amount
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
`SimpleRiskManager`; `VolTargetingRiskManager` ignores them.

### Data — supply your own OHLCV

You provide market data as a `{symbol: DataFrame}` dict — this is the only way data
enters the engine. Each DataFrame is indexed by a timezone-aware `DatetimeIndex`
(**UTC enforced**: a naive index raises `ValueError` at construction; other timezones
are converted to UTC) and exposes `Open`/`High`/`Low`/`Close`/`Volume` columns;
sourcing, cleaning, and windowing the data is up to you. A small bundled sample of
daily bars lives at
[backtests/sample_data/sample_1d.csv](backtests/sample_data/sample_1d.csv):

```python
import pandas as pd

raw = pd.read_csv('backtests/sample_data/sample_1d.csv')
raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
data = {sym: g.set_index('timestamp')[['Open', 'High', 'Low', 'Close', 'Volume']]
        for sym, g in raw.groupby('symbol')}
```

**Alternative data (optional).** Non-OHLCV series — funding rates, open
interest, and the like — ride along as named per-symbol *alt feeds*:
`alt_data={feed: {symbol: df}}`, each frame a tz-aware `DatetimeIndex`
plus numeric columns (column names become field names). Timestamps mean
*"the moment the value became known"*. Records are merged into the same
time-sorted stream as bars and stored in rolling windows — no events are
emitted; a strategy reads the latest values inside `calculate_forecast`
via `data_handler.get_latest_alt(symbol, feed, n)` (or
`get_latest_alt_df` / `count_alt`). When the bar at open-time *T* is
processed, a feed's window contains exactly the records with `ts ≤ T`. A
feed doesn't have to cover every symbol — uncovered symbols simply never
warm up for that strategy. A shared series (e.g. refinery utilization
across several oil futures) is broadcast at wiring time:

```python
alt_data = {'refinery_util': {sym: util_df for sym in ['CL', 'RB', 'HO']}}
data_handler = HistoricDataHandler(events_queue, config.symbols,
                                   base_timeframe=config.base_timeframe,
                                   timeframes=config.timeframes,
                                   data=data, alt_data=alt_data)
```

### Run — wire the modules and start the loop

The trader instantiates each module explicitly, passes them into
`Backtester(...)`, and calls `run()`. Two more modules feed the risk manager's
sizing and are required by `Backtester` alongside the rest: `universe.UniverseManager`
(tracks which symbols are currently tradable — strategy warmup + minimum
price history) and `correlation.CorrelationManager` (estimates the instrument
correlation matrix on a walk-forward cadence, used for instrument weights and
the diversification multiplier). The full pattern lives in
[backtests/sample_backtest/backtest_ewmac_sample.py](backtests/sample_backtest/backtest_ewmac_sample.py); the condensed shape is:

```python
import queue
from data import HistoricDataHandler
from portfolio import BacktestPortfolio
from execution import BacktestExecution, SlippageModel, CommissionModel
from volatility import EWMAVolEstimator, bars_per_year
from universe import UniverseManager
from correlation import CorrelationManager
from riskmanager import VolTargetingRiskManager
from backtester import Backtester

events_queue = queue.Queue()
data_handler = HistoricDataHandler(events_queue, config.symbols,
                                   base_timeframe=config.base_timeframe,
                                   timeframes=config.timeframes, data=data)
strategy     = MyStrategy(data_handler, config.symbols, fast=10, slow=30)
# `instruments` is the registry built in step 1b — the SAME object is passed to
# the portfolio, risk manager, and execution handler.
portfolio    = BacktestPortfolio(events_queue, data_handler, config.symbols,
                                 instruments=instruments,
                                 initial_capital=config.initial_capital)
vol_estimator = EWMAVolEstimator(config.symbols, data_handler=data_handler,
                                 bars_per_year=bars_per_year('1d', config.days_convention),
                                 timeframe='1d', span=36)

# Tradable-universe liveness and correlation estimation are engine-driven
# modules of their own — both required by Backtester, both reach the risk
# manager as events (never a per-bar concern for the RM itself).
universe_manager    = UniverseManager(strategy, data_handler,
                                      min_history_bars=60, history_timeframe='1d')
correlation_manager = CorrelationManager(data_handler, universe_manager,
                                         lookback=60, step_size=30, timeframe='1d')

risk_manager  = VolTargetingRiskManager(portfolio, strategy, vol_estimator,
                                              universe_manager=universe_manager,
                                              instruments=instruments,
                                              annual_target_vol=config.annual_target_vol,
                                              vol_target_mode=config.vol_target_mode,
                                              position_buffer=config.position_buffer)
execution     = BacktestExecution(events_queue,
                                  instruments=instruments)

bt = Backtester(events_queue, data_handler, strategy, portfolio, risk_manager,
                execution, universe_manager, correlation_manager)
bt.run()

# Access results
equity_df = portfolio.get_equity_curve()
trade_df  = portfolio.get_trade_log()
records   = bt.strategy.get_records('BTC_USDT:USDT')
```