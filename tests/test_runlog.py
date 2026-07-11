"""
Unit + integration tests for the ``runlog`` run-archive package.

Unit tier drives ``save_run``/``load_run``/``list_runs`` with duck-typed
fakes (hand-built frames matching the real getters' shapes: non-unique
tz-aware timestamp index, dict columns with shared per-timestamp refs,
None-bearing object columns). Integration tier wires a real mini-backtest
(HistoricDataHandler + EWMACStrategy + BacktestPortfolio +
SimpleRiskManager + BacktestExecution) and round-trips the archive.

Run from repo root:  pytest tests/test_runlog.py -v
"""

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from runlog import (RunRecord, list_runs, load_run,
                    replace_with_retry, save_run)
from runlog._save import _resolve_run_id
from runlog._serialize import (
    decode_stats,
    encode_stats,
    sanitize_frame,
    serialize_instruments,
    slugify,
)


# ──────────────────────────────────────────────
# Fixture data + duck-typed fakes
# ──────────────────────────────────────────────

TS = [pd.Timestamp('2026-01-01', tz='UTC'),
      pd.Timestamp('2026-01-02', tz='UTC'),
      pd.Timestamp('2026-01-03', tz='UTC')]
SYMBOLS = ['AAA', 'BBB']


def _build_equity_curve() -> pd.DataFrame:
    """Mimic ``BacktestPortfolio.get_equity_curve()``: two rows per
    timestamp (one per symbol) with a non-unique timestamp index, and the
    four dict columns shared (same object) across a timestamp's rows —
    the last-event-wins snapshot."""
    rows, dict_cells = [], {}
    for i, ts in enumerate(TS):
        snapshot = {
            'positions': {'AAA': float(i), 'BBB': -float(i)},
            'unrealized_pnl': {'AAA': 1.5 * i, 'BBB': -0.5 * i},
            'realized_pnl': {'AAA': 10.0 * i, 'BBB': 0.0},
            'margin_requirements': {'AAA': 100.0 * i, 'BBB': 50.0 * i},
        }
        dict_cells[ts] = snapshot
        for j, sym in enumerate(SYMBOLS):
            rows.append({
                'timestamp': ts, 'symbol': sym,
                'cash': 1000.0 + 10 * i + j,
                'account_balance': 1000.0 + 20 * i + j,
                'simple_return': 0.01 * j if i else float('nan'),
                'log_return': 0.01 * j if i else float('nan'),
                'position_margin': 5.0 * i,
                'maintenance_margin': 1.0 * i,
                'available_balance': 900.0 + 20 * i,
                'total_commission': 2.0 * i,
            })
    df = pd.DataFrame(rows).set_index('timestamp')
    for col in ('positions', 'unrealized_pnl', 'realized_pnl',
                'margin_requirements'):
        df[col] = [dict_cells[ts][col] for ts in df.index]
    return df


class _FakePortfolio:
    """Duck-typed BacktestPortfolio double with hand-built frames."""

    def __init__(self, empty: bool = False):
        self._empty = empty
        self.initial_capital = 1000.0
        self.cash = 1040.0
        self.total_commission = 4.0
        self.account_balance = 1042.0
        self.available_balance = 940.0
        self.positions = {'AAA': 2.0, 'BBB': -2.0}
        self.avg_cost = {'AAA': 10.0, 'BBB': 20.0}
        self.realized_pnl = {'AAA': 20.0, 'BBB': 0.0}
        self.unrealized_pnl = {'AAA': 3.0, 'BBB': -1.0}
        self.margin_requirements = {'AAA': 200.0, 'BBB': 100.0}

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame() if self._empty else _build_equity_curve()

    def get_trade_log(self) -> pd.DataFrame:
        if self._empty:
            return pd.DataFrame()
        return pd.DataFrame([
            {'timestamp': TS[1], 'symbol': 'AAA', 'direction': 'BUY',
             'quantity': 2.0, 'fill_price': 10.0, 'fill_notional': 20.0,
             'commission': 2.0, 'realized_pnl': 0.0, 'position_after': 2.0,
             'cash_after': 1018.0, 'order_id': 'O1'},
            {'timestamp': TS[2], 'symbol': 'BBB', 'direction': 'SELL',
             'quantity': 2.0, 'fill_price': 20.0, 'fill_notional': 40.0,
             'commission': 2.0, 'realized_pnl': 20.0, 'position_after': -2.0,
             'cash_after': 1040.0, 'order_id': 'O2'},
        ])

    def get_order_log(self) -> pd.DataFrame:
        if self._empty:
            return pd.DataFrame()
        return pd.DataFrame([
            {'timestamp': TS[1], 'symbol': 'AAA', 'order_type': 'MKT',
             'direction': 'BUY', 'quantity': 2.0, 'order_id': 'O1',
             'is_liquidation': False},
        ])


class _FakeStrategy:
    """Duck-typed Strategy double: timestamp-indexed per-symbol records
    with per-variation forecast/weight columns."""

    def __init__(self, symbols: List[str] = None, empty: bool = False):
        self.symbol_list = list(symbols if symbols is not None else SYMBOLS)
        self._empty = empty
        self.variations = {'v1': {'window': 5}}
        self.weights = {'v1': 1.0}
        self.fdm = 1.0

    def get_records(self, symbol: str) -> pd.DataFrame:
        if self._empty or symbol not in self.symbol_list:
            return pd.DataFrame()
        df = pd.DataFrame({
            'timestamp': TS,
            'close': [10.0, 11.0, 12.0],
            'forecast': [float('nan'), 5.0, -7.5],
            'forecast_v1': [float('nan'), 5.0, -7.5],
            'weight_v1': [1.0, 1.0, 1.0],
        })
        return df.set_index('timestamp')


class _FakeOrchestrator:
    """Duck-typed Orchestrator double (detected by the ``strategies``
    dict) combining two child fakes."""

    def __init__(self):
        self.strategies: Dict[str, _FakeStrategy] = {
            'ewmac': _FakeStrategy(['AAA', 'BBB']),
            'rsimr': _FakeStrategy(['BBB']),
        }
        self.symbol_list = sorted({s for c in self.strategies.values()
                                   for s in c.symbol_list})
        self.strategy_weights = {'ewmac': 0.5, 'rsimr': 0.5}
        self.fdm = 1.2

    def get_records(self, symbol: str) -> pd.DataFrame:
        df = pd.DataFrame({
            'timestamp': TS,
            'forecast': [float('nan'), 4.0, -6.0],
            'fdm': [None, 1.2, 1.2],
            'n_contributors': [0, 2, 2],
            'forecast_ewmac': [float('nan'), 5.0, -7.5],
            'weight_ewmac': [0.0, 0.5, 0.5],
            'forecast_rsimr': [float('nan'), 3.0, -4.5],
            'weight_rsimr': [0.0, 0.5, 0.5],
        })
        return df.set_index('timestamp')


class _FakeRiskManager:
    """Duck-typed VolTargetingRiskManager double: None-bearing object
    columns exactly as the real per-bar rows carry them."""

    def __init__(self, carver: bool = True, empty: bool = False):
        self._empty = empty
        if carver:
            self.idm = 1.5
            self.instrument_weight = {'AAA': 0.5, 'BBB': 0.5}

    def get_live_symbols(self) -> List[str]:
        return list(SYMBOLS)

    def get_records(self, symbol: str) -> pd.DataFrame:
        if self._empty:
            return pd.DataFrame()
        df = pd.DataFrame({
            'timestamp': TS,
            'symbol': [symbol] * 3,
            'forecast': [None, 5.0, -7.5],
            'sigma': [None, 2.0, 2.1],
            'instrument_weight': [None, 0.5, 0.5],
            'submitted': [False, True, True],
            'skip_reason': ['warmup_volatility', None, None],
        })
        return df.set_index('timestamp')


class _MinimalRiskManager:
    """SimpleRiskManager-like double: records only, no idm/weights/live."""

    def get_records(self, symbol: str) -> pd.DataFrame:
        return _FakeRiskManager().get_records(symbol)


@dataclasses.dataclass
class _MiniConfig:
    """Tiny stand-in config carrying only what save_run reads."""
    base_timeframe: str = '1d'
    days_convention: str = 'calendar'
    initial_capital: float = 1000.0


def _save_fake_run(tmp_path, **overrides):
    """Save one fake single-strategy run into ``tmp_path`` and return the
    RunRecord."""
    kwargs = dict(
        portfolio=_FakePortfolio(),
        strategy=_FakeStrategy(),
        risk_manager=_FakeRiskManager(),
        config=_MiniConfig(),
        root=tmp_path,
    )
    kwargs.update(overrides)
    return save_run(**kwargs)


# ──────────────────────────────────────────────
# Layout + manifest
# ──────────────────────────────────────────────

def test_save_creates_layout_and_manifest(tmp_path):
    record = _save_fake_run(tmp_path, label='unit run', notes='hello',
                            extra={'seed': 42})
    assert isinstance(record, RunRecord)
    assert record.path.parent == tmp_path
    assert record.path.name.endswith('_unit-run')      # slugged label
    for rel in ('manifest.json', 'portfolio/equity_curve.parquet',
                'portfolio/pnl_snapshots.parquet',
                'portfolio/trade_log.parquet', 'portfolio/order_log.parquet',
                'strategies/_FakeStrategy.parquet', 'riskmanager.parquet'):
        assert (record.path / rel).is_file(), rel
    assert not (record.path / 'orchestrator.parquet').exists()
    assert not record.path.name.endswith('.tmp')

    m = record.manifest
    assert m['schema_version'] == 1
    assert m['run']['label'] == 'unit run'              # original, unslugged
    assert m['run']['notes'] == 'hello'
    assert m['run']['extra'] == {'seed': 42}
    assert m['counts'] == {'n_symbols': 2, 'n_trades': 2, 'n_orders': 1,
                           'n_equity_rows': 6}
    assert m['portfolio']['initial_capital'] == 1000.0
    assert m['portfolio']['positions'] == {'AAA': 2.0, 'BBB': -2.0}
    assert m['risk_manager']['idm'] == 1.5
    assert m['risk_manager']['live_symbols'] == SYMBOLS
    assert m['system']['kind'] == 'strategy'
    assert m['system']['children']['_FakeStrategy']['weights'] == {'v1': 1.0}
    assert m['config']['days_convention'] == 'calendar'


def test_minimal_risk_manager_fields_absent(tmp_path):
    record = _save_fake_run(tmp_path, risk_manager=_MinimalRiskManager())
    rm = record.manifest['risk_manager']
    assert 'idm' not in rm and 'instrument_weight' not in rm
    assert 'live_symbols' not in rm


def test_instruments_deduped_by_identity():
    shared = _MiniConfig()   # any object with the expected attrs missing → None
    groups = serialize_instruments({'AAA': shared, 'BBB': shared})
    assert len(groups) == 1
    assert groups[0]['symbols'] == ['AAA', 'BBB']


# ──────────────────────────────────────────────
# Table round trips
# ──────────────────────────────────────────────

def test_equity_curve_round_trip(tmp_path):
    record = _save_fake_run(tmp_path)
    loaded = record.equity_curve()
    source = _build_equity_curve().drop(columns=[
        'positions', 'unrealized_pnl', 'realized_pnl', 'margin_requirements'])
    assert list(loaded.columns) == list(source.columns)
    assert loaded.index.name == 'timestamp'
    pd.testing.assert_frame_equal(loaded, source, check_dtype=False)


def test_pnl_snapshots_tidy_schema(tmp_path):
    record = _save_fake_run(tmp_path)
    snaps = record.pnl_snapshots()
    assert list(snaps.columns) == ['timestamp', 'symbol', 'position',
                                   'realized_pnl', 'unrealized_pnl',
                                   'margin_requirement']
    # One row per (timestamp x symbol).
    assert len(snaps) == len(TS) * len(SYMBOLS)
    cell = snaps[(snaps['timestamp'] == TS[2]) & (snaps['symbol'] == 'AAA')]
    assert float(cell['position'].iloc[0]) == 2.0
    assert float(cell['realized_pnl'].iloc[0]) == 20.0
    cell = snaps[(snaps['timestamp'] == TS[1]) & (snaps['symbol'] == 'BBB')]
    assert float(cell['unrealized_pnl'].iloc[0]) == -0.5
    assert float(cell['margin_requirement'].iloc[0]) == 50.0


def test_equity_curve_full_reconstruction(tmp_path):
    record = _save_fake_run(tmp_path)
    full = record.equity_curve_full()
    source = _build_equity_curve()
    assert list(full.columns) == list(source.columns)
    assert len(full) == len(source)
    for i in range(len(source)):
        for col in ('positions', 'unrealized_pnl', 'realized_pnl',
                    'margin_requirements'):
            assert full[col].iloc[i] == source[col].iloc[i], (i, col)


def test_trade_and_order_log_round_trip(tmp_path):
    record = _save_fake_run(tmp_path)
    trades = record.trade_log()
    pd.testing.assert_frame_equal(trades, _FakePortfolio().get_trade_log(),
                                  check_dtype=False)
    orders = record.order_log()
    assert orders['order_type'].tolist() == ['MKT']
    assert orders['is_liquidation'].tolist() == [False]


def test_riskmanager_records_none_columns(tmp_path):
    record = _save_fake_run(tmp_path)
    rm = record.riskmanager_records()
    # Long-table shape: leading symbol column + timestamp as a column.
    assert rm.columns[0] == 'symbol'
    assert 'timestamp' in rm.columns
    assert len(rm) == len(TS) * len(SYMBOLS)
    one = rm[rm['symbol'] == 'AAA'].reset_index(drop=True)
    # None-bearing object columns land as float64 + NaN / string + null.
    assert pd.isna(one['sigma'].iloc[0]) and one['sigma'].iloc[1] == 2.0
    assert one['skip_reason'].iloc[0] == 'warmup_volatility'
    assert pd.isna(one['skip_reason'].iloc[1])
    assert bool(one['submitted'].iloc[1]) is True


def test_strategy_records_round_trip(tmp_path):
    record = _save_fake_run(tmp_path)
    assert record.strategy_labels == ['_FakeStrategy']
    assert record.has_orchestrator is False
    rec = record.strategy_records()          # label omitted: single strategy
    assert set(rec['symbol']) == set(SYMBOLS)
    assert 'weight_v1' in rec.columns
    assert (rec['weight_v1'] == 1.0).all()
    with pytest.raises(ValueError, match='unknown strategy label'):
        record.strategy_records('nope')
    with pytest.raises(ValueError, match='bare Strategy'):
        record.orchestrator_records()


# ──────────────────────────────────────────────
# Orchestrator runs
# ──────────────────────────────────────────────

def test_orchestrator_run(tmp_path):
    orch = _FakeOrchestrator()
    record = _save_fake_run(tmp_path, strategy=orch)
    assert record.has_orchestrator is True
    assert sorted(record.strategy_labels) == ['ewmac', 'rsimr']
    for rel in ('orchestrator.parquet', 'strategies/ewmac.parquet',
                'strategies/rsimr.parquet'):
        assert (record.path / rel).is_file(), rel
    combined = record.orchestrator_records()
    assert {'forecast_ewmac', 'weight_ewmac', 'n_contributors'} <= set(combined.columns)
    child = record.strategy_records('rsimr')
    assert set(child['symbol']) == {'BBB'}   # rsimr only covers BBB
    with pytest.raises(ValueError, match='pass label='):
        record.strategy_records()            # ambiguous with 2 strategies
    assert record.manifest['system']['strategy_weights'] == {'ewmac': 0.5,
                                                             'rsimr': 0.5}
    assert record.manifest['system']['fdm'] == 1.2


# ──────────────────────────────────────────────
# Stats JSON codec
# ──────────────────────────────────────────────

def test_stats_codec_round_trip():
    stats = pd.Series({
        'Start': pd.Timestamp('2026-01-01', tz='UTC'),
        'End': pd.NaT,
        'Duration': pd.Timedelta(days=2, hours=3),
        'Return [%]': 12.5,
        'Sharpe Ratio': float('nan'),
        '# Fills': 7,
        'Note': 'ok',
    }, dtype=object)
    payload = json.loads(json.dumps(encode_stats(stats), allow_nan=False))
    back = decode_stats(payload)
    assert list(back.index) == list(stats.index)         # order preserved
    assert back['Start'] == stats['Start']
    assert back['End'] is pd.NaT
    assert back['Duration'] == stats['Duration']
    assert back['Return [%]'] == 12.5
    assert isinstance(back['Sharpe Ratio'], float) and pd.isna(back['Sharpe Ratio'])
    assert back['# Fills'] == 7.0
    assert back['Note'] == 'ok'


def test_saved_stats_loads(tmp_path):
    record = _save_fake_run(tmp_path)
    stats = record.stats()
    assert record.manifest['analytics']['status']['stats'] == 'ok'
    assert 'Equity Final [$]' in stats.index


# ──────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────

def test_empty_run_saves_cleanly(tmp_path):
    record = _save_fake_run(
        tmp_path,
        portfolio=_FakePortfolio(empty=True),
        strategy=_FakeStrategy(empty=True),
        risk_manager=_FakeRiskManager(empty=True),
    )
    empties = record.manifest['empty_tables']
    assert 'portfolio/trade_log.parquet' in empties
    assert 'portfolio/equity_curve.parquet' in empties
    assert record.trade_log().empty
    assert record.equity_curve().empty
    assert record.strategy_records().empty
    assert record.equity_curve_full().empty


def test_analytics_failure_never_loses_raw_archive(tmp_path, monkeypatch):
    import runlog._save as save_mod

    def _boom(*args, **kwargs):
        raise RuntimeError('injected stats failure')

    monkeypatch.setattr(save_mod, 'backtest_stats', _boom)
    record = _save_fake_run(tmp_path)
    assert record.manifest['analytics']['status']['stats'] == 'error'
    assert 'injected stats failure' in record.manifest['analytics']['errors']['stats']
    # Raw archive fully intact and loadable.
    assert not record.equity_curve().empty
    assert not record.riskmanager_records().empty
    with pytest.raises(ValueError, match='injected stats failure'):
        record.stats()


def test_config_none_skips_stats_and_turnover(tmp_path):
    record = _save_fake_run(tmp_path, config=None)
    status = record.manifest['analytics']['status']
    assert status['stats'] == 'skipped'
    assert status['turnover'] == 'skipped'
    assert record.manifest['config'] is None
    with pytest.raises(ValueError, match='skipped'):
        record.turnover()
    # Raw archive unaffected.
    assert not record.trade_log().empty


def test_validation_errors(tmp_path):
    with pytest.raises(TypeError, match='portfolio lacks'):
        save_run(portfolio=object(), strategy=_FakeStrategy(),
                 risk_manager=_FakeRiskManager(), root=tmp_path)
    with pytest.raises(TypeError, match='label must be'):
        _save_fake_run(tmp_path, label=123)
    with pytest.raises(TypeError, match='extra must be'):
        _save_fake_run(tmp_path, extra=[1, 2])
    assert list_runs(tmp_path).empty         # nothing was half-written


# ──────────────────────────────────────────────
# Run identity + history browsing
# ──────────────────────────────────────────────

def test_slugify_and_run_id_collision(tmp_path):
    assert slugify('BTC/USDT:USDT run') == 'BTC-USDT-USDT-run'
    with pytest.raises(ValueError):
        slugify(':::')
    created = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
    first = _resolve_run_id(tmp_path, created, 'x')
    (tmp_path / first).mkdir()
    second = _resolve_run_id(tmp_path, created, 'x')
    assert second == f'{first}_2'
    (tmp_path / f'{second}.tmp').mkdir()     # in-flight save also collides
    assert _resolve_run_id(tmp_path, created, 'x') == f'{first}_3'


def test_list_runs(tmp_path):
    r1 = _save_fake_run(tmp_path, label='first')
    r2 = _save_fake_run(tmp_path, label='second')
    (tmp_path / 'not-a-run').mkdir()                       # manifest-less
    (tmp_path / '20260101T000000Z_crashed.tmp').mkdir()    # stale tmp
    runs = list_runs(tmp_path)
    assert len(runs) == 2
    assert list(runs.columns) == ['run_id', 'created_utc', 'label',
                                  'n_symbols', 'n_trades', 'initial_capital',
                                  'final_balance', 'schema_version']
    assert runs['label'].tolist() == ['first', 'second']
    assert runs['created_utc'].is_monotonic_increasing
    assert runs['final_balance'].tolist() == [1042.0, 1042.0]
    assert load_run(r1.path).manifest == r1.manifest
    with pytest.raises(ValueError, match='not a run archive'):
        load_run(tmp_path / 'not-a-run')


# ──────────────────────────────────────────────
# sanitize_frame unit coverage
# ──────────────────────────────────────────────

def test_sanitize_frame_object_columns():
    df = pd.DataFrame({
        'num': pd.Series([None, 1.0, 2], dtype=object),
        'txt': pd.Series(['a', None, 'b'], dtype=object),
        'flag': pd.Series([True, None, False], dtype=object),
        'allnone': pd.Series([None, None, None], dtype=object),
        'mixed': pd.Series([1, 'x', None], dtype=object),
    })
    out = sanitize_frame(df)
    assert out['num'].dtype == 'float64' and pd.isna(out['num'].iloc[0])
    assert out['txt'].dtype == object
    assert str(out['flag'].dtype) == 'boolean'
    assert out['allnone'].dtype == 'float64'
    assert out['mixed'].iloc[1] == repr('x')  # repr fallback


# ──────────────────────────────────────────────
# Integration: real mini-backtest → archive → reload
# ──────────────────────────────────────────────

def test_integration_real_mini_backtest(tmp_path):
    import queue as thread_queue

    from backtester import Backtester
    from config import BacktestConfig, uniform_registry
    from data import HistoricDataHandler
    from execution import BacktestExecution
    from portfolio import BacktestPortfolio
    from riskmanager import SimpleRiskManager
    from strategy import EWMACStrategy

    rng = np.random.default_rng(7)
    idx = pd.date_range('2026-01-01', periods=60, freq='D', tz='UTC')
    data = {}
    for sym, base in (('AAA', 100.0), ('BBB', 50.0)):
        closes = base + np.cumsum(rng.normal(0, 1.0, len(idx)))
        data[sym] = pd.DataFrame({
            'Open': closes, 'High': closes + 0.5, 'Low': closes - 0.5,
            'Close': closes, 'Volume': 1.0,
        }, index=idx)

    config = BacktestConfig(
        symbols=list(data), start_date='2026-01-01', end_date='2026-03-02',
        base_timeframe='1d', days_convention='calendar',
        timeframes={'1d': 100}, initial_capital=1_000_000.0,
        annual_target_vol=100_000.0,
        size_mode='fixed_quantity', position_size=10.0,
    )
    instruments = uniform_registry(config.symbols)
    events_queue = thread_queue.Queue()
    data_handler = HistoricDataHandler(
        events_queue, config.symbols, base_timeframe=config.base_timeframe,
        timeframes=config.timeframes, data=data,
    )
    strategy = EWMACStrategy(
        data_handler, config.symbols,
        variations={'2_4': {'fast': 2, 'slow': 4}},
        vol_lookback=5, forecast_scalar_lookback=5,
    )
    portfolio = BacktestPortfolio(
        events_queue, data_handler, config.symbols,
        instruments=instruments, initial_capital=config.initial_capital,
    )
    risk_manager = SimpleRiskManager(
        portfolio, strategy, size_mode=config.size_mode,
        position_size=config.position_size, instruments=instruments,
    )
    execution = BacktestExecution(events_queue, instruments=instruments)
    Backtester(events_queue, data_handler, strategy, portfolio,
               risk_manager, execution).run()

    record = save_run(
        portfolio=portfolio, strategy=strategy, risk_manager=risk_manager,
        config=config, instruments=instruments, root=tmp_path,
        label='mini',
    )

    # Counts reconcile with the live objects.
    assert record.manifest['counts']['n_trades'] == len(portfolio.get_trade_log())
    assert record.manifest['counts']['n_trades'] > 0
    assert record.manifest['counts']['n_equity_rows'] == len(portfolio.get_equity_curve())

    # Scalar equity round trip against the live frame.
    live_eq = portfolio.get_equity_curve()
    loaded_eq = record.equity_curve()
    pd.testing.assert_frame_equal(
        loaded_eq,
        live_eq.drop(columns=['positions', 'unrealized_pnl', 'realized_pnl',
                              'margin_requirements']),
        check_dtype=False,
    )

    # Full reconstruction matches the live dict columns row-by-row.
    full = record.equity_curve_full()
    for col in ('positions', 'unrealized_pnl', 'realized_pnl',
                'margin_requirements'):
        assert all(full[col].iloc[i] == live_eq[col].iloc[i]
                   for i in range(len(live_eq))), col

    # Strategy records carry the new per-variation weight column.
    rec = record.strategy_records()
    assert 'weight_2_4' in rec.columns
    assert set(rec['symbol']) == set(config.symbols)
    live_rec = strategy.get_records('AAA')
    loaded_aaa = rec[rec['symbol'] == 'AAA'].drop(columns='symbol')
    loaded_aaa = loaded_aaa.set_index('timestamp')
    pd.testing.assert_frame_equal(loaded_aaa, live_rec, check_dtype=False)

    # RM records + derived analytics saved and loadable.
    assert not record.riskmanager_records().empty
    stats = record.stats()
    assert record.manifest['analytics']['status']['stats'] == 'ok'
    assert float(stats['Equity Final [$]']) == pytest.approx(
        float(live_eq['account_balance'].iloc[-1]))
    assert record.manifest['analytics']['status']['attribution'] == 'ok'
    assert not record.attribution_long().empty
    # Turnover table indexed by symbol with the trailing Average row.
    turnover = record.turnover()
    assert record.manifest['analytics']['status']['turnover'] == 'ok'
    assert 'Average' in turnover.index

    # Offline re-analysis on the saved copy reproduces the live stats.
    from analytics import backtest_stats
    recomputed = backtest_stats(
        record.equity_curve_full(), record.trade_log(),
        initial_capital=config.initial_capital,
        timeframe=config.base_timeframe,
        days_convention=config.days_convention,
    )
    assert float(recomputed['Net PnL [$]']) == pytest.approx(
        float(stats['Net PnL [$]']))
    assert float(recomputed['Sharpe Ratio']) == pytest.approx(
        float(stats['Sharpe Ratio']), nan_ok=True)


def test_sanitize_frame_is_re_exported():
    """sanitize_frame is public API (consumed by validation's sweep cache)."""
    import runlog
    from runlog._serialize import sanitize_frame as private

    assert runlog.sanitize_frame is private
    assert 'sanitize_frame' in runlog.__all__


# ──────────────────────────────────────────────
# Accessor copy semantics (F8)
# ──────────────────────────────────────────────

def test_run_record_accessor_mutation_does_not_poison_cache(tmp_path):
    rec = _save_fake_run(tmp_path)
    first = rec.trade_log()
    first['injected'] = 999.0                # new column
    if len(first):
        first.iloc[0, first.columns.get_loc('injected')] = -1.0
    second = rec.trade_log()
    assert 'injected' not in second.columns


def test_run_record_reads_parquet_once_per_table(tmp_path, monkeypatch):
    rec = _save_fake_run(tmp_path)
    calls = {'n': 0}
    real = pd.read_parquet
    def counting(path, *a, **kw):
        calls['n'] += 1
        return real(path, *a, **kw)
    monkeypatch.setattr('runlog._load.pd.read_parquet', counting)
    rec.trade_log()
    rec.trade_log()
    assert calls['n'] == 1                   # cached after the first read


# ──────────────────────────────────────────────
# replace_with_retry (F3): transient Windows rename locks
# ──────────────────────────────────────────────

def test_replace_with_retry_recovers_from_transient_lock(tmp_path,
                                                         monkeypatch, caplog):
    calls = {'n': 0}
    real_replace = os.replace
    def flaky(src, dst):
        calls['n'] += 1
        if calls['n'] <= 2:
            raise PermissionError(5, 'Access is denied')
        return real_replace(src, dst)
    monkeypatch.setattr('runlog._serialize.os.replace', flaky)
    monkeypatch.setattr('runlog._serialize.time.sleep', lambda s: None)
    src = tmp_path / 'a'
    src.mkdir()
    dst = tmp_path / 'b'
    with caplog.at_level(logging.WARNING, logger='runlog._serialize'):
        replace_with_retry(src, dst)
    assert calls['n'] == 3
    assert dst.is_dir() and not src.exists()
    assert any('retrying' in rec.message for rec in caplog.records)


def test_replace_with_retry_reraises_after_exhausted_attempts(tmp_path,
                                                              monkeypatch):
    def always_locked(src, dst):
        raise PermissionError(5, 'Access is denied')
    monkeypatch.setattr('runlog._serialize.os.replace', always_locked)
    monkeypatch.setattr('runlog._serialize.time.sleep', lambda s: None)
    src = tmp_path / 'a'
    src.mkdir()
    with pytest.raises(PermissionError):
        replace_with_retry(src, tmp_path / 'b', attempts=3)
