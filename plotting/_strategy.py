"""Candlestick/line + indicator + signal-marker chart for strategy records."""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import resample


def plot_strategy(df: pd.DataFrame,
                  timeframe: Optional[str] = None,
                  chart: str = 'candlestick',
                  indicators: Optional[Dict[str, int]] = None,
                  volume: bool = True,
                  title: str = "Strategy Chart",
                  signal_offset: float = 0) -> go.Figure:
    """
    Plot a price chart from a strategy get_records() DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of strategy.get_records(symbol). Expected columns:
        open, high, low, close, volume, and any indicator/signal columns.
    timeframe : str, optional
        Target timeframe for the chart (e.g. ``'1h'``, ``'4h'``, ``'1d'``).
        ``None`` (default) plots the records at the base timeframe of ``df``.
        When set, OHLCV is aggregated as ``first/max/min/last/sum`` and
        indicator/forecast columns take their last value per bucket
        (as-of semantics). Signal lists are concatenated within each bucket
        so multiple signals firing in one bucket all appear on the chart.
        Bucket alignment matches ``data.resample``.
    chart : str
        Price-panel style: ``'candlestick'`` (default) or ``'line'``
        (a line of ``close``). Use ``'line'`` for settle-price-only
        futures data — the data layer requires non-NaN OHLC, so such
        data is fed with O=H=L=C=settle and candles render degenerately.
        In line mode the volume bars are colored by close vs *prior*
        close (close vs open is always flat when O=C); the first bar,
        having no prior close, renders neutral gray.
    indicators : dict of str -> int, optional
        Mapping of column name to 1-indexed panel number. Panel 1 is the
        price candlestick panel; panel 2 is the first sub-panel directly
        below it; panel 3 is below that; and so on. Indicators on panel 1
        are overlaid on price (e.g. moving averages); indicators on panels
        >= 2 get their own y-axis (e.g. percent-rank, RSI).
        Panel numbers must form a contiguous range; panel 1 may be
        omitted (since the price candlestick always occupies it), so
        ``{'a': 2, 'b': 3}`` is allowed (no overlays, two sub-panels)
        but a gap like ``{'a': 1, 'b': 3}`` raises ``ValueError``. If
        ``None``, all non-OHLCV/signal/forecast columns are auto-detected
        and placed on panel 1.
    volume : bool
        Whether to show a volume subplot. When enabled, volume is rendered
        in the bottom-most row, below any indicator sub-panels.
    title : str
        Chart title.
    signal_offset : float
        Fractional vertical offset for signal markers (e.g. ``0.002`` for
        0.2% above close). Signal markers always render on the price panel.

    Returns
    -------
    go.Figure
    """
    if chart not in ('candlestick', 'line'):
        raise ValueError(
            f"Unknown chart: {chart!r}. Must be 'candlestick' or 'line'."
        )
    if timeframe is not None and not df.empty:
        df = _resample_records(df, timeframe)

    ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}
    meta_cols = {'signal', 'forecast'}

    # Auto-detect indicator columns if not specified, all on panel 1
    if indicators is None:
        indicators = {c: 1 for c in df.columns if c not in ohlcv_cols | meta_cols}

    # Validate panel numbers form a contiguous range. Panel 1 (the price
    # candlestick) is always rendered, so it may be omitted from the
    # ``indicators`` dict — accept either ``[1, 2, ..., k]`` (overlays on
    # price + k-1 sub-panels) or ``[2, ..., k]`` (no overlays, k-1
    # sub-panels). Gaps elsewhere are still rejected.
    panel_nums = sorted(set(indicators.values()))
    if panel_nums:
        expected_with_one = list(range(1, panel_nums[-1] + 1))
        expected_without_one = list(range(2, panel_nums[-1] + 1))
        if panel_nums not in (expected_with_one, expected_without_one):
            raise ValueError(
                f"Panel numbers must be contiguous (panel 1 may be omitted "
                f"since it's the always-present price panel); got {panel_nums}"
            )

    # Group indicators by panel (sorted within each panel for stable color cycle)
    panels: dict[int, list[str]] = {}
    for col, panel in sorted(indicators.items(), key=lambda kv: (kv[1], kv[0])):
        panels.setdefault(panel, []).append(col)

    n_indicator_panels = panel_nums[-1] - 1 if panel_nums else 0
    n_rows = 1 + n_indicator_panels + (1 if volume else 0)
    volume_row = n_rows  # volume always last

    # Row heights: price 0.5, each indicator panel 0.2, volume 0.15, normalized to 1
    raw_heights = [0.5] + [0.2] * n_indicator_panels + ([0.15] if volume else [])
    total = sum(raw_heights)
    row_heights = [h / total for h in raw_heights]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
    )

    # Price trace on panel (row 1)
    if chart == 'candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)
    elif chart == 'line':
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'], mode='lines', name='Price',
            line=dict(width=1.5, color='#26a69a'),
        ), row=1, col=1)
    else:
        raise ValueError(f"Unexpected chart: {chart!r}")

    # Indicator overlays — route to assigned panel's row
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336', '#00BCD4']
    color_idx = 0
    for panel in sorted(panels):
        for col in panels[panel]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], mode='lines',
                    name=col, line=dict(width=1.5, color=colors[color_idx % len(colors)]),
                ), row=panel, col=1)
                color_idx += 1

    # Signal markers — always on price panel; signal column contains a list of signal names or None
    if 'signal' in df.columns:
        signals = df[df['signal'].notna()]

        def _has_signal(col, name):
            return col.apply(lambda x: name in x)

        long_signals = signals[_has_signal(signals['signal'], 'OPEN_LONG')]
        short_signals = signals[_has_signal(signals['signal'], 'OPEN_SHORT')]
        close_long = signals[_has_signal(signals['signal'], 'CLOSE_LONG')]
        close_short = signals[_has_signal(signals['signal'], 'CLOSE_SHORT')]
        flatten = signals[_has_signal(signals['signal'], 'FLATTEN')]

        _add_markers(fig, long_signals, 'close', 'triangle-up', '#26a69a', 'Open Long', signal_offset)
        _add_markers(fig, short_signals, 'close', 'triangle-down', '#ef5350', 'Open Short', signal_offset)
        _add_markers(fig, close_long, 'close', 'x', '#80cbc4', 'Close Long', signal_offset)
        _add_markers(fig, close_short, 'close', 'x', '#ef9a9a', 'Close Short', signal_offset)
        _add_markers(fig, flatten, 'close', 'diamond', '#FFEB3B', 'Flatten', signal_offset)

    # Volume bars on the last row
    if volume:
        if chart == 'candlestick':
            vol_colors = ['#ef5350' if c < o else '#26a69a'
                          for o, c in zip(df['open'], df['close'])]
        elif chart == 'line':
            # Settle-only frames have O=C, so color by close vs prior
            # close instead; the first bar has no prior → neutral gray.
            prev = df['close'].shift()
            vol_colors = [
                '#9e9e9e' if pd.isna(p) else ('#ef5350' if c < p else '#26a69a')
                for p, c in zip(prev, df['close'])
            ]
        else:
            raise ValueError(f"Unexpected chart: {chart!r}")
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='Volume',
            marker_color=vol_colors, opacity=0.5, showlegend=False,
        ), row=volume_row, col=1)

    height = 500 + 200 * (n_indicator_panels + (1 if volume else 0))
    fig.update_layout(
        title=title, template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=height, margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='Price', row=1, col=1)
    for panel in range(2, 2 + n_indicator_panels):
        fig.update_yaxes(title_text=', '.join(panels.get(panel, [])),
                         row=panel, col=1)
    if volume:
        fig.update_yaxes(title_text='Volume', row=volume_row, col=1)

    return fig


def _add_markers(fig: go.Figure, df: pd.DataFrame,
                 price_col: str, symbol: str, color: str,
                 name: str, offset_pct: float) -> None:
    """Add signal markers slightly offset from the price for visibility.

    The offset is a fraction of ``|price|`` ADDED to the price (not a
    multiplication by ``1 + offset``) so a positive offset always shifts
    markers upward — multiplicative offsets invert direction when the
    price is negative (e.g. WTI 2020, futures spreads). Identical to the
    historical behavior for positive prices.
    """
    if df.empty:
        return
    y_vals = df[price_col] + df[price_col].abs() * offset_pct
    fig.add_trace(go.Scatter(
        x=df.index, y=y_vals, mode='markers', name=name,
        marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color='white')),
    ), row=1, col=1)


_OHLCV_AGG = {'open': 'first', 'high': 'max', 'low': 'min',
              'close': 'last', 'volume': 'sum'}


def _aggregate_signals(values: pd.Series) -> Optional[List[str]]:
    """Concat all non-None signal lists in a bucket; return None if all empty."""
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        out.extend(v)
    return out if out else None


def _resample_records(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a strategy records DataFrame to a higher timeframe.

    Bucketing is delegated to ``data.resample`` so chart boundaries
    match the rest of the system. OHLCV is aggregated standardly; every
    other column takes ``last`` (as-of), except ``signal`` which
    concatenates per-bucket lists.
    """
    agg: dict[str, Any] = {}
    for col in df.columns:
        if col in _OHLCV_AGG:
            agg[col] = _OHLCV_AGG[col]
        elif col == 'signal':
            agg[col] = _aggregate_signals
        else:
            agg[col] = 'last'

    resampled = resample(df, timeframe, agg)
    if resampled.empty:
        return resampled
    resampled.dropna(subset=['open'], inplace=True)
    return resampled
