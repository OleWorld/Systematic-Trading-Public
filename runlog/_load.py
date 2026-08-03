"""
``load_run`` / ``list_runs`` / ``RunRecord`` — read a saved run archive.

``RunRecord`` is the handle over one run folder written by
``runlog.save_run``: the manifest is loaded eagerly (it is small), every
table is read lazily on first access and cached. Tables listed in the
manifest's ``empty_tables`` (skipped at save time because they had no
rows) load as empty DataFrames; analytics artifacts that were skipped or
failed at save time raise a ``ValueError`` naming the recorded reason.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ._serialize import decode_stats

logger = logging.getLogger(__name__)

#: Tidy pnl_snapshots column -> equity-curve dict-column name, in the
#: original ``get_equity_curve`` column order.
_SNAPSHOT_RESTORE = {
    'position': 'positions',
    'unrealized_pnl': 'unrealized_pnl',
    'realized_pnl': 'realized_pnl',
    'margin_requirement': 'margin_requirements',
}

_LIST_RUNS_COLUMNS = ['run_id', 'created_utc', 'label', 'start', 'end',
                      'n_symbols', 'n_trades', 'initial_capital',
                      'final_balance', 'schema_version']


class RunRecord:
    """
    Handle over one saved run folder.

    Attributes: ``path`` (the run directory) and ``manifest`` (the parsed
    ``manifest.json`` dict). All table accessors are lazy and cached on
    first read; each call returns a fresh copy, so callers may mutate
    freely.
    """

    def __init__(self, path: Any):
        """Open the run folder at ``path``; raises ``ValueError`` if it is
        not a run archive (no ``manifest.json``)."""
        self.path = Path(path)
        manifest_path = self.path / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(
                f"{self.path} is not a run archive (no manifest.json)"
            )
        with open(manifest_path, encoding='utf-8') as fh:
            self.manifest: Dict[str, Any] = json.load(fh)
        self._cache: Dict[str, Any] = {}

    def __repr__(self) -> str:
        run = self.manifest.get('run', {})
        return f"RunRecord(run_id={run.get('run_id')!r}, path={str(self.path)!r})"

    # ── Internal readers ─────────────────────────────────────────────

    def _read_table(self, relpath: str, *,
                    missing_reason: Optional[str] = None) -> pd.DataFrame:
        """
        Read one parquet table (cached; each call returns a fresh copy so
        caller mutation can never reach the cache). Paths listed in the
        manifest's ``empty_tables`` return an empty DataFrame; a missing
        file raises ``ValueError`` (with ``missing_reason`` appended when
        given).
        """
        if relpath in self._cache:
            return self._cache[relpath].copy()
        if relpath in self.manifest.get('empty_tables', []):
            df = pd.DataFrame()
        else:
            file = self.path / relpath
            if not file.is_file():
                detail = f" — {missing_reason}" if missing_reason else ""
                raise ValueError(
                    f"{relpath} not found in run archive {self.path}{detail}"
                )
            df = pd.read_parquet(file)
        self._cache[relpath] = df
        return df.copy()

    def _analytics_reason(self, name: str) -> Optional[str]:
        """Manifest-recorded skip/error reason for one analytics artifact."""
        analytics = self.manifest.get('analytics', {})
        status = analytics.get('status', {}).get(name)
        if status == 'ok' or status is None:
            return None
        error = analytics.get('errors', {}).get(name)
        return f"analytics {name!r} was {status}" + (f": {error}" if error else "")

    def _read_analytics(self, name: str, relpath: str) -> pd.DataFrame:
        """Read one analytics table, raising the recorded reason if it was
        skipped or failed at save time."""
        reason = self._analytics_reason(name)
        if reason is not None:
            raise ValueError(f"{relpath} unavailable — {reason}")
        return self._read_table(relpath, missing_reason=reason)

    # ── Portfolio tables ─────────────────────────────────────────────

    def equity_curve(self) -> pd.DataFrame:
        """Per-event scalar equity curve, indexed by ``timestamp``
        (non-unique: one row per BarEvent). Dict columns live in
        ``pnl_snapshots()`` / ``equity_curve_full()``."""
        df = self._read_table('portfolio/equity_curve.parquet')
        return df if df.empty else df.set_index('timestamp')

    def pnl_snapshots(self) -> pd.DataFrame:
        """Tidy per-(timestamp x symbol) snapshot table: ``timestamp,
        symbol, position, realized_pnl, unrealized_pnl,
        margin_requirement``."""
        return self._read_table('portfolio/pnl_snapshots.parquet')

    def equity_curve_full(self) -> pd.DataFrame:
        """
        Reconstruct the original ``portfolio.get_equity_curve()`` shape:
        the scalar equity curve with the four per-symbol dict columns
        (``positions``, ``unrealized_pnl``, ``realized_pnl``,
        ``margin_requirements``) re-attached per timestamp — suitable for
        re-running ``pnl_attribution`` / ``turnover_stats`` offline.
        """
        eq = self.equity_curve()
        snaps = self.pnl_snapshots()
        if eq.empty or snaps.empty:
            return eq
        per_ts: Dict[Any, Dict[str, dict]] = {}
        for ts, group in snaps.groupby('timestamp'):
            symbols = group['symbol'].tolist()
            per_ts[ts] = {
                dict_col: dict(zip(symbols, group[tidy_col].astype(float)))
                for tidy_col, dict_col in _SNAPSHOT_RESTORE.items()
            }
        eq = eq.copy()
        for dict_col in _SNAPSHOT_RESTORE.values():
            eq[dict_col] = [per_ts[ts][dict_col] for ts in eq.index]
        return eq

    def trade_log(self) -> pd.DataFrame:
        """Per-fill trade log (RangeIndex), as saved."""
        return self._read_table('portfolio/trade_log.parquet')

    def order_log(self) -> pd.DataFrame:
        """Per-accepted-order log (RangeIndex), as saved."""
        return self._read_table('portfolio/order_log.parquet')

    # ── Record tables ────────────────────────────────────────────────

    @property
    def strategy_labels(self) -> List[str]:
        """Labels of the archived strategies (one per record table)."""
        return list(self.manifest['system']['labels'])

    @property
    def has_orchestrator(self) -> bool:
        """True when the run's strategy slot held an Orchestrator."""
        return self.manifest['system']['kind'] == 'orchestrator'

    def strategy_records(self, label: Optional[str] = None) -> pd.DataFrame:
        """
        One strategy's long record table (``symbol``, ``timestamp``, plus
        the strategy's diagnostic columns). ``label`` may be omitted only
        when the run holds exactly one strategy; unknown labels raise
        ``ValueError``.
        """
        labels_map = self.manifest['system']['labels']
        if label is None:
            if len(labels_map) == 1:
                label = next(iter(labels_map))
            else:
                raise ValueError(
                    f"run holds {len(labels_map)} strategies "
                    f"({sorted(labels_map)}); pass label= explicitly"
                )
        elif label not in labels_map:
            raise ValueError(
                f"unknown strategy label {label!r}; available: {sorted(labels_map)}"
            )
        return self._read_table(f'strategies/{labels_map[label]}')

    def orchestrator_records(self) -> pd.DataFrame:
        """The orchestrator's combined long record table; raises
        ``ValueError`` on a single-strategy run."""
        if not self.has_orchestrator:
            raise ValueError(
                "this run was saved with a bare Strategy (no orchestrator "
                "records); use strategy_records() instead"
            )
        return self._read_table('orchestrator.parquet')

    def riskmanager_records(self) -> pd.DataFrame:
        """The risk manager's long sizing-diagnostics table."""
        return self._read_table('riskmanager.parquet')

    def universe_transitions(self) -> pd.DataFrame:
        """The universe manager's transition log (one row per initial
        evaluation + per status transition; empty for runs saved without a
        universe manager)."""
        return self._read_table('universe.parquet')

    # ── Derived analytics ────────────────────────────────────────────

    def stats(self) -> pd.Series:
        """The saved ``backtest_stats`` Series (Timestamps/Timedeltas/NaN
        restored, label order preserved)."""
        reason = self._analytics_reason('stats')
        if reason is not None:
            raise ValueError(f"analytics/stats.json unavailable — {reason}")
        cache_key = 'analytics/stats.json'
        if cache_key not in self._cache:
            file = self.path / cache_key
            if not file.is_file():
                raise ValueError(
                    f"{cache_key} not found in run archive {self.path}"
                )
            with open(file, encoding='utf-8') as fh:
                self._cache[cache_key] = decode_stats(json.load(fh))
        return self._cache[cache_key]

    def attribution_long(self) -> pd.DataFrame:
        """The tidy ``pnl_attribution(...).long`` frame
        (``timestamp, symbol, strategy, variation, pnl``)."""
        return self._read_analytics('attribution', 'analytics/attribution_long.parquet')

    def attribution_total(self) -> pd.Series:
        """The per-bar system gross PnL Series
        (``pnl_attribution(...).total``), indexed by timestamp."""
        df = self._read_analytics('attribution', 'analytics/attribution_total.parquet')
        if df.empty:
            return pd.Series(dtype=float, name='pnl')
        return df.set_index('timestamp')['pnl']

    def turnover(self) -> pd.DataFrame:
        """The per-instrument ``turnover_stats`` table, indexed by symbol
        (includes the final ``'Average'`` row)."""
        df = self._read_analytics('turnover', 'analytics/turnover.parquet')
        return df if df.empty else df.set_index('symbol')


def load_run(path: Any) -> RunRecord:
    """Open the run archive at ``path`` (a run directory) and return its
    ``RunRecord``; raises ``ValueError`` if no manifest is found."""
    return RunRecord(path)


def list_runs(root: Any = 'results/runs') -> pd.DataFrame:
    """
    Cheap run-history browser: one row per archived run under ``root``,
    read from manifests only (no parquet touched), sorted by
    ``created_utc``. Columns: ``run_id, created_utc, label, start, end,
    n_symbols, n_trades, initial_capital, final_balance,
    schema_version``. ``start``/``end`` are the ACTUAL derived backtest
    range (``None`` for pre-schema-2 manifests, which predate
    ``data_range``). In-flight or crashed ``*.tmp`` folders and
    manifest-less directories are skipped. Returns an empty fixed-schema
    frame when ``root`` is missing or holds no runs.
    """
    root = Path(root)
    rows = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.endswith('.tmp'):
                continue
            manifest_path = child / 'manifest.json'
            if not manifest_path.is_file():
                continue
            try:
                with open(manifest_path, encoding='utf-8') as fh:
                    manifest = json.load(fh)
            except Exception:
                logger.warning("Skipping unreadable manifest at %s", manifest_path)
                continue
            run = manifest.get('run', {})
            counts = manifest.get('counts', {})
            portfolio = manifest.get('portfolio', {})
            data_range = manifest.get('data_range') or {}
            rows.append({
                'run_id': run.get('run_id', child.name),
                'created_utc': run.get('created_utc'),
                'label': run.get('label'),
                'start': data_range.get('start'),
                'end': data_range.get('end'),
                'n_symbols': counts.get('n_symbols'),
                'n_trades': counts.get('n_trades'),
                'initial_capital': portfolio.get('initial_capital'),
                'final_balance': portfolio.get('account_balance'),
                'schema_version': manifest.get('schema_version'),
            })
    df = pd.DataFrame(rows, columns=_LIST_RUNS_COLUMNS)
    if not df.empty:
        df['created_utc'] = pd.to_datetime(df['created_utc'], format='ISO8601')
        df = df.sort_values('created_utc').reset_index(drop=True)
    return df
