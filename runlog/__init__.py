"""
Run record-keeping package — persist each backtest run's results to disk.

``save_run(...)`` is called once after ``bt.run()`` (zero engine changes)
and writes a self-contained folder under ``results/runs/<run_id>/``:
Parquet tables for the equity curve, tidy PnL snapshots, trade/order
logs, per-strategy / orchestrator / risk-manager record tables, derived
analytics (``backtest_stats`` / ``pnl_attribution`` / ``turnover_stats``,
best-effort), and a ``manifest.json`` capturing config, end-of-run state,
and environment metadata. ``load_run(path)`` returns a ``RunRecord``
handle for offline analysis; ``list_runs(root)`` browses the run history
from manifests alone.

Callers do ``from runlog import save_run, load_run, list_runs, sanitize_frame``.
"""

from ._load import RunRecord, list_runs, load_run
from ._save import save_run
from ._serialize import sanitize_frame

__all__ = ['RunRecord', 'list_runs', 'load_run', 'sanitize_frame', 'save_run']
