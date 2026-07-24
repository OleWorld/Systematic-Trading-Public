"""
``save_run`` — archive one finished backtest run to disk.

Writes a self-contained run folder under ``root`` (default
``results/runs/``): the portfolio's equity curve / PnL snapshots / trade
log / order log, one long per-symbol record table per strategy (plus the
orchestrator's combined table on multi-strategy runs), the risk manager's
sizing records, best-effort derived analytics, and a ``manifest.json``
capturing config, end-of-run state, and environment metadata.

Robustness contract: raw tables are written first and each derived
analytics artifact is computed inside its own ``try/except`` — an
analytics bug can never lose the raw archive. Everything is staged in a
``<run_id>.tmp`` directory and atomically renamed to ``<run_id>`` after
the manifest lands, so a crash mid-save never leaves a half-visible run.

All inputs are duck-typed (an ``Orchestrator`` is detected by its
``strategies`` dict, mirroring ``analytics._attribution``) — this module
imports nothing from ``config``/``portfolio``/``strategy``/``riskmanager``.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics import backtest_stats, pnl_attribution, turnover_stats

from ._load import RunRecord, load_run
from ._serialize import (
    SCHEMA_VERSION,
    encode_stats,
    json_safe,
    replace_with_retry,
    sanitize_frame,
    serialize_config,
    serialize_instruments,
    slugify,
)

logger = logging.getLogger(__name__)

#: Dict columns re-attached by ``BacktestPortfolio.get_equity_curve`` —
#: normalized into the tidy ``pnl_snapshots`` table instead of being
#: serialized as object cells. Maps equity-curve column -> tidy column.
_SNAPSHOT_FIELDS = {
    'positions': 'position',
    'realized_pnl': 'realized_pnl',
    'unrealized_pnl': 'unrealized_pnl',
    'margin_requirements': 'margin_requirement',
}


def save_run(*, portfolio: Any, strategy: Any, risk_manager: Any,
             config: Any = None, instruments: Optional[Dict[str, Any]] = None,
             root: Any = 'results/runs', label: Optional[str] = None,
             notes: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None) -> RunRecord:
    """
    Archive a finished backtest run and return a ``RunRecord`` handle.

    Parameters (all keyword-only, duck-typed):
      portfolio     — exposes ``get_equity_curve()``, ``get_trade_log()``,
                      ``get_order_log()`` and the end-of-run state attrs
                      (``initial_capital``, ``cash``, ...).
      strategy      — a ``Strategy`` or an ``Orchestrator`` (detected by a
                      dict-valued ``strategies`` attribute); must expose
                      ``symbol_list`` and ``get_records(symbol)``.
      risk_manager  — exposes ``get_records(symbol)``; ``idm`` /
                      ``instrument_weight`` / ``get_live_symbols`` are
                      snapshotted when present.
      config        — optional ``BacktestConfig`` dataclass. Without it the
                      raw archive still saves, but ``backtest_stats`` and
                      ``turnover_stats`` are skipped with a WARNING (they
                      need ``base_timeframe``/``days_convention``).
      instruments   — optional ``{symbol: InstrumentConfig}`` registry,
                      flattened into the manifest (documentation-grade).
      root          — archive root directory (default ``results/runs``).
      label         — optional human label, slug-appended to the run id.
      notes / extra — free text / JSON-able dict stored in the manifest.

    Returns ``load_run(<final path>)`` — every successful save proves the
    archive loads back. Raises ``TypeError``/``ValueError`` on genuinely
    broken inputs (missing duck-type surface, bad param types, unwritable
    root); derived-analytics failures are recorded in the manifest instead
    of raised.
    """
    _validate_inputs(portfolio, strategy, risk_manager, label, notes, extra)

    created = datetime.now(timezone.utc)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = _resolve_run_id(root, created, label)
    tmp_dir = root / f'{run_id}.tmp'
    tmp_dir.mkdir()
    logger.info("Archiving run to %s", root / run_id)

    empty_tables: List[str] = []

    # ── Raw portfolio tables (first — never lost to later failures) ──
    equity_df = portfolio.get_equity_curve()
    scalar_eq = equity_df.drop(columns=list(_SNAPSHOT_FIELDS), errors='ignore')
    _write_table(scalar_eq, tmp_dir, 'portfolio/equity_curve.parquet', empty_tables)
    _write_table(_build_pnl_snapshots(equity_df), tmp_dir,
                 'portfolio/pnl_snapshots.parquet', empty_tables)
    trade_df = portfolio.get_trade_log()
    order_df = portfolio.get_order_log()
    _write_table(trade_df, tmp_dir, 'portfolio/trade_log.parquet', empty_tables)
    _write_table(order_df, tmp_dir, 'portfolio/order_log.parquet', empty_tables)

    # ── Per-source record tables ──
    is_orchestrator = isinstance(getattr(strategy, 'strategies', None), dict)
    if is_orchestrator:
        children: Dict[str, Any] = strategy.strategies
        _write_table(_records_long_table(strategy, strategy.symbol_list),
                     tmp_dir, 'orchestrator.parquet', empty_tables)
    else:
        children = {type(strategy).__name__: strategy}
    labels_map = _label_filenames(children)
    for child_label, child in children.items():
        _write_table(_records_long_table(child, child.symbol_list), tmp_dir,
                     f'strategies/{labels_map[child_label]}', empty_tables)
    _write_table(_records_long_table(risk_manager, strategy.symbol_list),
                 tmp_dir, 'riskmanager.parquet', empty_tables)

    # ── Derived analytics (best-effort, each in its own guard) ──
    analytics_status, analytics_errors = _save_analytics(
        tmp_dir, empty_tables, equity_df, trade_df, portfolio, strategy, config,
    )

    # ── Manifest (last = completeness marker), then atomic reveal ──
    manifest = _build_manifest(
        run_id=run_id, created=created, label=label, notes=notes, extra=extra,
        config=config, instruments=instruments, portfolio=portfolio,
        strategy=strategy, risk_manager=risk_manager, children=children,
        labels_map=labels_map, is_orchestrator=is_orchestrator,
        equity_df=equity_df, trade_df=trade_df, order_df=order_df,
        empty_tables=empty_tables, analytics_status=analytics_status,
        analytics_errors=analytics_errors,
    )
    with open(tmp_dir / 'manifest.json', 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)
    final_dir = root / run_id
    replace_with_retry(tmp_dir, final_dir)
    logger.info("Run archived: %s", final_dir)
    return load_run(final_dir)


# ── Validation & identity ────────────────────────────────────────────────

def _validate_inputs(portfolio: Any, strategy: Any, risk_manager: Any,
                     label: Optional[str], notes: Optional[str],
                     extra: Optional[dict]) -> None:
    """Fail fast on missing duck-type surface or bad param types."""
    for attr in ('get_equity_curve', 'get_trade_log', 'get_order_log'):
        if not hasattr(portfolio, attr):
            raise TypeError(f"portfolio lacks required method {attr!r}")
    for attr in ('symbol_list', 'get_records'):
        if not hasattr(strategy, attr):
            raise TypeError(f"strategy lacks required attribute {attr!r}")
    if not hasattr(risk_manager, 'get_records'):
        raise TypeError("risk_manager lacks required method 'get_records'")
    if label is not None and not isinstance(label, str):
        raise TypeError(f"label must be a str or None, got {type(label).__name__}")
    if notes is not None and not isinstance(notes, str):
        raise TypeError(f"notes must be a str or None, got {type(notes).__name__}")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError(f"extra must be a dict or None, got {type(extra).__name__}")


def _resolve_run_id(root: Path, created: datetime, label: Optional[str]) -> str:
    """
    Build the run id ``<UTC ts>Z[_<label slug>]`` (colon-free — Windows-safe
    and lexicographically chronological) and suffix ``_2``, ``_3``, ... on a
    same-second collision with an existing run (or in-flight ``.tmp``).
    """
    base_id = created.strftime('%Y%m%dT%H%M%SZ')
    if label is not None:
        base_id = f'{base_id}_{slugify(label)}'
    run_id, n = base_id, 2
    while (root / run_id).exists() or (root / f'{run_id}.tmp').exists():
        run_id = f'{base_id}_{n}'
        n += 1
    return run_id


def _label_filenames(children: Dict[str, Any]) -> Dict[str, str]:
    """Map each strategy label to a unique slug-sanitized parquet filename."""
    labels_map: Dict[str, str] = {}
    used: set = set()
    for child_label in children:
        slug, n = slugify(child_label), 2
        while slug in used:
            slug = f'{slugify(child_label)}-{n}'
            n += 1
        used.add(slug)
        labels_map[child_label] = f'{slug}.parquet'
    return labels_map


# ── Table builders ───────────────────────────────────────────────────────

def _write_table(df: Optional[pd.DataFrame], tmp_dir: Path, relpath: str,
                 empty_tables: List[str]) -> None:
    """
    Sanitize and write one table; an empty (or ``None``) frame is skipped
    and recorded in ``empty_tables`` instead (the loader returns an empty
    DataFrame for listed paths).
    """
    if df is None or df.empty:
        empty_tables.append(relpath)
        return
    target = tmp_dir / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitize_frame(df).to_parquet(target, index=False)


def _build_pnl_snapshots(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the equity curve's four per-symbol dict columns into the tidy
    schema ``timestamp, symbol, position, realized_pnl, unrealized_pnl,
    margin_requirement`` — one row per (timestamp x symbol). The dict
    columns are deduped to one row per timestamp first (they are
    last-event-wins shared references across a timestamp's event rows).
    """
    if equity_df.empty or not all(c in equity_df.columns for c in _SNAPSHOT_FIELDS):
        return pd.DataFrame()
    snap = equity_df[list(_SNAPSHOT_FIELDS)].groupby(level=0).last()
    expanded: Dict[str, pd.DataFrame] = {
        col: pd.DataFrame(snap[col].tolist(), index=snap.index)
        for col in _SNAPSHOT_FIELDS
    }
    symbols = sorted(set().union(*(e.columns for e in expanded.values())))
    if not symbols:
        return pd.DataFrame()
    n_sym = len(symbols)
    tidy = {
        'timestamp': snap.index.repeat(n_sym),
        'symbol': np.tile(symbols, len(snap.index)),
    }
    for col, out_name in _SNAPSHOT_FIELDS.items():
        aligned = expanded[col].reindex(columns=symbols)
        tidy[out_name] = aligned.to_numpy(dtype=float).ravel()
    return pd.DataFrame(tidy)


def _data_range(equity_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """First/last equity-curve timestamps as ISO strings — the ACTUAL
    backtest range (the curve carries one row per streamed BarEvent, so it
    covers the full run even when the book stays flat). Both ``None`` for
    an empty run."""
    if equity_df.empty:
        return {'start': None, 'end': None}
    return {'start': pd.Timestamp(equity_df.index[0]).isoformat(),
            'end': pd.Timestamp(equity_df.index[-1]).isoformat()}


def _records_long_table(source: Any, symbols: List[str]) -> pd.DataFrame:
    """
    Concat ``source.get_records(symbol)`` across ``symbols`` into one long
    table with a leading ``symbol`` column and ``timestamp`` reset to a
    regular column. Symbols with empty records (never live) are skipped.
    """
    frames = []
    for symbol in symbols:
        rec = source.get_records(symbol)
        if rec is None or rec.empty:
            continue
        rec = rec.reset_index()
        if 'symbol' in rec.columns:
            # Risk-manager rows already carry the symbol — just lead with it.
            rec = rec[['symbol'] + [c for c in rec.columns if c != 'symbol']]
        else:
            rec.insert(0, 'symbol', symbol)
        frames.append(rec)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# ── Derived analytics ────────────────────────────────────────────────────

def _save_analytics(tmp_dir: Path, empty_tables: List[str],
                    equity_df: pd.DataFrame, trade_df: pd.DataFrame,
                    portfolio: Any, strategy: Any, config: Any):
    """
    Compute and persist the three derived artifacts, each inside its own
    guard. Returns ``(status, errors)`` dicts for the manifest: status
    values are ``'ok'`` / ``'skipped'`` / ``'error'``; errors carry the
    ``repr`` of the raising exception.
    """
    status: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    if config is None:
        logger.warning(
            "No config passed to save_run — skipping backtest_stats and "
            "turnover_stats (they need base_timeframe/days_convention)"
        )
        status['stats'] = status['turnover'] = 'skipped'
    else:
        try:
            stats = backtest_stats(
                equity_df, trade_df,
                initial_capital=portfolio.initial_capital,
                timeframe=config.base_timeframe,
                days_convention=config.days_convention,
            )
            (tmp_dir / 'analytics').mkdir(exist_ok=True)
            with open(tmp_dir / 'analytics/stats.json', 'w', encoding='utf-8') as fh:
                json.dump(encode_stats(stats), fh, indent=2, allow_nan=False)
            status['stats'] = 'ok'
        except Exception as exc:
            logger.exception("backtest_stats failed — raw archive unaffected")
            status['stats'] = 'error'
            errors['stats'] = repr(exc)
        try:
            turnover = turnover_stats(
                equity_df, trade_df,
                timeframe=config.base_timeframe,
                days_convention=config.days_convention,
            )
            turnover = turnover.rename_axis('symbol').reset_index()
            _write_table(turnover, tmp_dir, 'analytics/turnover.parquet',
                         empty_tables)
            status['turnover'] = 'ok'
        except Exception as exc:
            logger.exception("turnover_stats failed — raw archive unaffected")
            status['turnover'] = 'error'
            errors['turnover'] = repr(exc)

    try:
        attribution = pnl_attribution(equity_df, strategy)
        _write_table(attribution.long, tmp_dir,
                     'analytics/attribution_long.parquet', empty_tables)
        total = attribution.total.rename('pnl').rename_axis('timestamp')
        _write_table(total.reset_index(), tmp_dir,
                     'analytics/attribution_total.parquet', empty_tables)
        status['attribution'] = 'ok'
    except Exception as exc:
        logger.exception("pnl_attribution failed — raw archive unaffected")
        status['attribution'] = 'error'
        errors['attribution'] = repr(exc)

    return status, errors


# ── Manifest ─────────────────────────────────────────────────────────────

def _build_manifest(*, run_id: str, created: datetime, label: Optional[str],
                    notes: Optional[str], extra: Optional[dict], config: Any,
                    instruments: Optional[Dict[str, Any]], portfolio: Any,
                    strategy: Any, risk_manager: Any,
                    children: Dict[str, Any], labels_map: Dict[str, str],
                    is_orchestrator: bool, equity_df: pd.DataFrame,
                    trade_df: pd.DataFrame, order_df: pd.DataFrame,
                    empty_tables: List[str], analytics_status: Dict[str, str],
                    analytics_errors: Dict[str, str]) -> Dict[str, Any]:
    """Assemble the JSON-safe manifest dict (see the module docstring)."""
    portfolio_state = {'class': type(portfolio).__name__}
    for attr in ('initial_capital', 'cash', 'total_commission',
                 'account_balance', 'available_balance', 'positions',
                 'avg_cost', 'realized_pnl', 'unrealized_pnl',
                 'margin_requirements'):
        if hasattr(portfolio, attr):
            portfolio_state[attr] = json_safe(getattr(portfolio, attr))

    rm_state = {'class': type(risk_manager).__name__}
    for attr in ('idm', 'instrument_weight'):
        if hasattr(risk_manager, attr):
            rm_state[attr] = json_safe(getattr(risk_manager, attr))
    if hasattr(risk_manager, 'get_live_symbols'):
        rm_state['live_symbols'] = json_safe(risk_manager.get_live_symbols())

    system: Dict[str, Any] = {
        'kind': 'orchestrator' if is_orchestrator else 'strategy',
        'class': type(strategy).__name__,
        'labels': labels_map,
        'children': {
            child_label: {
                'class': type(child).__name__,
                'variations': json_safe(getattr(child, 'variations', None)),
                'weights': json_safe(getattr(child, 'weights', None)),
                'fdm': json_safe(getattr(child, 'fdm', None)),
            }
            for child_label, child in children.items()
        },
    }
    if is_orchestrator:
        system['strategy_weights'] = json_safe(
            getattr(strategy, 'strategy_weights', None))
        system['fdm'] = json_safe(getattr(strategy, 'fdm', None))

    return {
        'schema_version': SCHEMA_VERSION,
        'run': {
            'run_id': run_id,
            'label': label,
            'notes': notes,
            'created_utc': created.isoformat(),
            'extra': json_safe(extra) if extra is not None else None,
        },
        'data_range': _data_range(equity_df),
        'environment': {
            'python': sys.version.split()[0],
            'pandas': pd.__version__,
            'pyarrow': _pyarrow_version(),
            'git_commit': _git_commit(),
        },
        'config': serialize_config(config),
        'instruments': serialize_instruments(instruments),
        'portfolio': portfolio_state,
        'risk_manager': rm_state,
        'system': system,
        'counts': {
            'n_symbols': len(strategy.symbol_list),
            'n_trades': len(trade_df),
            'n_orders': len(order_df),
            'n_equity_rows': len(equity_df),
        },
        'empty_tables': empty_tables,
        'analytics': {'status': analytics_status, 'errors': analytics_errors},
    }


def _pyarrow_version() -> Optional[str]:
    """pyarrow version string, or None if the import fails."""
    try:
        import pyarrow
        return pyarrow.__version__
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    """Best-effort current git commit hash; None on any failure."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None
