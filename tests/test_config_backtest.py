"""Unit tests for ``BacktestConfig``.

``BacktestConfig`` validates only the structural coherence of its own
fields (non-empty ``symbols``, ``base_timeframe`` registered in
``timeframes``, higher timeframes strictly larger than base). Domain
values (vol-target knobs, corr knobs, size modes, ``days_convention``)
are validated by the consuming modules at wiring time —
``VolTargetingRiskManager.__init__``, ``SimpleRiskManager.__init__``,
``volatility.bars_per_year`` — and their rejection tests live with those
modules. Here we pin the defaults and the structural checks, plus one
contract test asserting config does NOT re-validate domain values.

Run from the repo root:  pytest tests/test_config_backtest.py -v
"""

import pytest

from config import BacktestConfig


def _kwargs(**overrides):
    """Minimal valid kwargs for ``BacktestConfig`` construction."""
    base = dict(
        symbols=['BTC'],
        base_timeframe='1d',
        days_convention='calendar',
    )
    base.update(overrides)
    return base


# ── structural checks (config-owned) ─────────────────────────────────


def test_empty_symbols_rejected():
    with pytest.raises(ValueError, match="symbols"):
        BacktestConfig(**_kwargs(symbols=[]))


def test_timeframes_default_to_base_with_maxlen_500():
    cfg = BacktestConfig(**_kwargs())
    assert cfg.timeframes == {'1d': 500}


def test_base_timeframe_missing_from_timeframes_rejected():
    with pytest.raises(ValueError, match="base_timeframe"):
        BacktestConfig(**_kwargs(base_timeframe='1h', timeframes={'1d': 100}))


def test_timeframe_not_larger_than_base_rejected():
    """Every non-base timeframe must be strictly larger than base."""
    with pytest.raises(ValueError, match="strictly larger"):
        BacktestConfig(**_kwargs(
            base_timeframe='1d', timeframes={'1d': 500, '4h': 200},
        ))


def test_higher_timeframes_accepted():
    cfg = BacktestConfig(**_kwargs(
        base_timeframe='1h', timeframes={'1h': 500, '4h': 200, '1d': 100},
    ))
    assert set(cfg.timeframes) == {'1h', '4h', '1d'}


# ── defaults (documented production stack) ───────────────────────────


def test_default_corr_fields():
    """Defaults match the documented production stack (60 / 30 / '1d')."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_lookback == 60
    assert cfg.corr_step_size == 30
    assert cfg.corr_timeframe == '1d'


def test_default_vol_target_mode_is_dollar_volatility():
    cfg = BacktestConfig(**_kwargs())
    assert cfg.vol_target_mode == 'dollar_volatility'


def test_default_annual_target_vol_is_none():
    """tau has no config-level default or requirement — the
    VolTargetingRiskManager constructor enforces it at wiring time, so a
    SimpleRiskManager run may leave it None."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.annual_target_vol is None


def test_default_corr_mode():
    """Futures-first default: absolute price changes."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_mode == 'absolute_price_chg'


def test_default_size_mode_is_fixed_quantity():
    """Futures-first default: size in contracts, not notional dollars."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.size_mode == 'fixed_quantity'


def test_no_start_end_date_fields():
    """start_date/end_date were removed 2026-07: the actual backtest range
    is derived by runlog.save_run from the equity curve, not user-typed."""
    cfg = BacktestConfig(**_kwargs())
    assert not hasattr(cfg, 'start_date')
    assert not hasattr(cfg, 'end_date')


def test_default_corr_floor_and_idm_cap():
    """Defaults: no rho floor (None), cap IDM at Carver's 2.5."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_floor is None
    assert cfg.idm_cap == 2.5


def test_default_corr_shrinkage():
    """Estimation hygiene on by default: Ledoit-Wolf shrink the inline rho."""
    assert BacktestConfig(**_kwargs()).corr_shrinkage == 'ledoit_wolf'


# ── ownership contract ───────────────────────────────────────────────


def test_domain_values_are_not_validated_at_config_construction():
    """Config is a parameter holder: values the consuming modules would
    reject must construct fine here and fail at wiring time instead
    (VolTargetingRiskManager / SimpleRiskManager / bars_per_year own the
    checks). Guards against re-introducing mirrored validation."""
    cfg = BacktestConfig(**_kwargs(
        days_convention='crypto',           # bars_per_year would reject
        annual_target_vol=None,             # VolTargetingRiskManager would reject
        vol_target_mode='bogus',            # VolTargetingRiskManager would reject
        position_buffer=5.0,                # VolTargetingRiskManager would reject
        instrument_weight_mode='hrp',       # VolTargetingRiskManager would reject
        corr_lookback=5,                    # VolTargetingRiskManager would reject
        corr_step_size=-1,                  # VolTargetingRiskManager would reject
        corr_mode='log_return',             # VolTargetingRiskManager would reject
        corr_floor=9.0,                     # VolTargetingRiskManager would reject
        corr_shrinkage='oas',               # VolTargetingRiskManager would reject
        idm_cap=0.1,                        # VolTargetingRiskManager would reject
        size_mode='martingale',             # SimpleRiskManager would reject
    ))
    assert cfg.corr_lookback == 5
    assert cfg.annual_target_vol is None
