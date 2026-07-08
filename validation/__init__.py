"""
Statistical validation of backtest results (research layer).

Block-bootstrap CIs/p-values, parameter-sweep heatmaps, slice-once
walk-forward, per-period stats, and the Deflated Sharpe Ratio. Operates on
``portfolio.get_equity_curve()`` output (live or archived) and caller-supplied
run factories — never imports the engine. Callers do
``from validation import ...``.
"""

__all__ = []
