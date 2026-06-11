"""Unit tests for ``BacktestConfig`` validation.

Focused on the correlation / walk-forward fields added for the inline
min-variance derivation in ``CarverVolTargetingRiskManager`` —
``corr_lookback``, ``corr_step_size``, ``corr_timeframe``.

Run from the repo root:  pytest tests/test_config_backtest.py -v
"""

import pytest

from config import BacktestConfig


def _kwargs(**overrides):
    """Minimal valid kwargs for ``BacktestConfig`` construction."""
    base = dict(
        symbols=['BTC'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        base_timeframe='1d',
        convention='crypto',
    )
    base.update(overrides)
    return base


def test_default_corr_fields():
    """Defaults match the documented production stack (500 / 30 / '1d')."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_lookback == 500
    assert cfg.corr_step_size == 30
    assert cfg.corr_timeframe == '1d'


def test_corr_lookback_below_two_rejected():
    with pytest.raises(ValueError, match="corr_lookback"):
        BacktestConfig(**_kwargs(corr_lookback=1))
    with pytest.raises(ValueError, match="corr_lookback"):
        BacktestConfig(**_kwargs(corr_lookback=0))


def test_corr_step_size_negative_rejected():
    with pytest.raises(ValueError, match="corr_step_size"):
        BacktestConfig(**_kwargs(corr_step_size=-1))


def test_corr_step_size_zero_accepted():
    """``0`` is a valid value — disables auto-recalc."""
    cfg = BacktestConfig(**_kwargs(corr_step_size=0))
    assert cfg.corr_step_size == 0


def test_default_corr_mode():
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_mode == 'simple_return'


def test_unknown_corr_mode_rejected():
    with pytest.raises(ValueError, match="corr_mode"):
        BacktestConfig(**_kwargs(corr_mode='log_return'))
