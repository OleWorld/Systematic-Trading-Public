"""Unit tests for ``plotting.plot_strategy``.

Structural assertions on the returned ``go.Figure`` (trace types, y-data,
volume-bar colors) — no rendering. Focus is the ``chart`` parameter:
``'candlestick'`` (default) vs ``'line'`` (settle-price-only futures data,
where the data layer's non-NaN OHLC requirement means O=H=L=C=settle and
candles would render degenerately).

Run from the repo root:  pytest tests/test_plotting_strategy.py -v
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from plotting import plot_strategy


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _records_df(closes, *, settle_only: bool = False) -> pd.DataFrame:
    """Strategy ``get_records``-shaped frame from a list of closes.

    ``settle_only=True`` mimics settle-price futures data fed through the
    data layer: O=H=L=C=settle.
    """
    idx = pd.date_range('2024-01-01', periods=len(closes), freq='D', tz='UTC')
    closes = pd.Series([float(c) for c in closes], index=idx)
    if settle_only:
        opens, highs, lows = closes, closes, closes
    else:
        opens = closes.shift(1).fillna(closes.iloc[0])
        highs = pd.concat([opens, closes], axis=1).max(axis=1) + 1.0
        lows = pd.concat([opens, closes], axis=1).min(axis=1) - 1.0
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'volume': 10.0,
    }, index=idx)


def _price_trace(fig: go.Figure):
    """The trace named 'Price' (the row-1 price trace)."""
    return next(t for t in fig.data if t.name == 'Price')


def _volume_trace(fig: go.Figure) -> go.Bar:
    return next(t for t in fig.data if isinstance(t, go.Bar))


# ──────────────────────────────────────────────
# chart-mode dispatch
# ──────────────────────────────────────────────

def test_default_chart_is_candlestick():
    fig = plot_strategy(_records_df([100, 101, 99]))
    assert isinstance(_price_trace(fig), go.Candlestick)


def test_line_chart_uses_scatter_of_close():
    df = _records_df([100, 101, 99], settle_only=True)
    fig = plot_strategy(df, chart='line')
    trace = _price_trace(fig)
    assert isinstance(trace, go.Scatter)
    assert list(trace.y) == list(df['close'])


def test_unknown_chart_rejected():
    with pytest.raises(ValueError, match="chart"):
        plot_strategy(_records_df([100, 101]), chart='ohlc')


# ──────────────────────────────────────────────
# Volume-bar coloring
# ──────────────────────────────────────────────

def test_line_mode_volume_colored_by_close_vs_prior_close():
    """Settle-only data has close == open, so the candlestick rule
    (close < open → red) would paint every bar green. Line mode colors by
    close vs prior close instead; the first bar (no prior) is neutral."""
    df = _records_df([100, 101, 99], settle_only=True)
    fig = plot_strategy(df, chart='line')
    colors = list(_volume_trace(fig).marker.color)
    assert colors[0] == '#9e9e9e'       # no prior close → neutral
    assert colors[1] == '#26a69a'       # 101 > 100 → up/green
    assert colors[2] == '#ef5350'       # 99 < 101 → down/red


def test_candlestick_mode_volume_coloring_unchanged():
    df = _records_df([100, 101, 99])
    fig = plot_strategy(df)
    colors = list(_volume_trace(fig).marker.color)
    # close vs open: bar0 flat (not <) → green, bar1 up → green, bar2 down → red
    assert colors == ['#26a69a', '#26a69a', '#ef5350']


# ──────────────────────────────────────────────
# Interaction with resampling
# ──────────────────────────────────────────────

def test_line_mode_with_resample_on_settle_only_frame():
    """O=H=L=C frames round-trip through the OHLCV resample agg and still
    render as a line of the bucket-last close."""
    df = _records_df(range(100, 114), settle_only=True)
    fig = plot_strategy(df, timeframe='1w', chart='line')
    trace = _price_trace(fig)
    assert isinstance(trace, go.Scatter)
    assert len(trace.y) < len(df)       # actually aggregated
