"""
Statistical validation of backtest results (research layer).

Block-bootstrap CIs/p-values, parameter-sweep heatmaps, slice-once
walk-forward, per-period stats, and the Deflated Sharpe Ratio. Operates on
``portfolio.get_equity_curve()`` output (live or archived) and caller-supplied
run factories — never imports the engine. Callers do
``from validation import ...``.
"""

from ._bootstrap import (BootstrapResult, bootstrap_stats,
                         politis_white_block_length)
from ._deflated import (DeflatedSharpeResult, deflated_sharpe,
                        probabilistic_sharpe)
from ._heatmap import ParamHeatmap
from ._periodic import periodic_stats
from ._sweep import SweepResult, load_sweep, param_sweep
from ._walkforward import WalkForwardResult, walk_forward

__all__ = [
    'BootstrapResult', 'DeflatedSharpeResult', 'ParamHeatmap', 'SweepResult',
    'WalkForwardResult', 'bootstrap_stats', 'deflated_sharpe', 'load_sweep',
    'param_sweep', 'periodic_stats', 'politis_white_block_length',
    'probabilistic_sharpe', 'walk_forward',
]
