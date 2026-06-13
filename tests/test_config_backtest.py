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
        days_convention='calendar',
        annual_target_vol=0.25,     # required since the futures-first refactor
    )
    base.update(overrides)
    return base


def test_default_corr_fields():
    """Defaults match the documented production stack (60 / 30 / '1d')."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_lookback == 60
    assert cfg.corr_step_size == 30
    assert cfg.corr_timeframe == '1d'


def test_corr_lookback_below_31_rejected():
    """corr_lookback doubles as the universe liveness threshold and must
    yield >= 30 price-change observations (lookback - 1)."""
    for bad in (30, 2, 1, 0):
        with pytest.raises(ValueError, match="corr_lookback"):
            BacktestConfig(**_kwargs(corr_lookback=bad))


def test_corr_lookback_of_31_accepted():
    cfg = BacktestConfig(**_kwargs(corr_lookback=31))
    assert cfg.corr_lookback == 31


def test_corr_lookback_above_deque_maxlen_rejected():
    """corr_lookback > the corr_timeframe deque maxlen → no symbol could
    ever pass the liveness gate; rejected at config construction."""
    with pytest.raises(ValueError, match="maxlen"):
        BacktestConfig(**_kwargs(
            timeframes={'1d': 100}, corr_lookback=101,
        ))


def test_corr_step_size_negative_rejected():
    with pytest.raises(ValueError, match="corr_step_size"):
        BacktestConfig(**_kwargs(corr_step_size=-1))


def test_corr_step_size_zero_accepted():
    """``0`` is a valid value — disables auto-recalc."""
    cfg = BacktestConfig(**_kwargs(corr_step_size=0))
    assert cfg.corr_step_size == 0


def test_instrument_weight_mode_accepts_all_three_schemes():
    for mode in ('equal_weight', 'min_variance', 'risk_parity'):
        cfg = BacktestConfig(**_kwargs(instrument_weight_mode=mode))
        assert cfg.instrument_weight_mode == mode


def test_unknown_instrument_weight_mode_rejected():
    with pytest.raises(ValueError, match="instrument_weight_mode"):
        BacktestConfig(**_kwargs(instrument_weight_mode='bogus'))


def test_default_vol_target_mode_is_dollar_volatility():
    cfg = BacktestConfig(**_kwargs())
    assert cfg.vol_target_mode == 'dollar_volatility'


def test_none_annual_target_vol_rejected():
    """tau has no default — omitting it must fail at config construction."""
    kwargs = _kwargs()
    kwargs.pop('annual_target_vol', None)
    with pytest.raises(ValueError, match="annual_target_vol"):
        BacktestConfig(**kwargs)


def test_percent_mode_range_validation():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="annual_target_vol"):
            BacktestConfig(**_kwargs(
                vol_target_mode='percent_volatility',
                annual_target_vol=bad,
            ))


def test_dollar_mode_accepts_large_tau():
    cfg = BacktestConfig(**_kwargs(
        vol_target_mode='dollar_volatility',
        annual_target_vol=250_000.0,
    ))
    assert cfg.annual_target_vol == 250_000.0


def test_dollar_mode_rejects_non_positive_tau():
    with pytest.raises(ValueError, match="annual_target_vol"):
        BacktestConfig(**_kwargs(
            vol_target_mode='dollar_volatility',
            annual_target_vol=0.0,
        ))


def test_unknown_vol_target_mode_rejected():
    with pytest.raises(ValueError, match="vol_target_mode"):
        BacktestConfig(**_kwargs(vol_target_mode='notional_volatility'))


def test_old_convention_values_rejected():
    """Hard rename guard: pre-rename 'crypto'/'tradfi' must fail loudly."""
    with pytest.raises(ValueError, match="days_convention"):
        BacktestConfig(**_kwargs(days_convention='crypto'))
    with pytest.raises(ValueError, match="days_convention"):
        BacktestConfig(**_kwargs(days_convention='tradfi'))


def test_default_corr_mode():
    """Futures-first default: absolute price changes."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_mode == 'absolute_price_chg'


def test_default_size_mode_is_fixed_quantity():
    """Futures-first default: size in contracts, not notional dollars."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.size_mode == 'fixed_quantity'


def test_default_slippage_mode_is_absolute():
    """Futures-first default: slippage in $ per unit (ticks), not % of price."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.slippage_mode == 'absolute'


def test_default_commission_mode_is_per_contract():
    """Futures-first default: commission in $ per contract."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.commission_mode == 'per_contract'
    assert cfg.commission_value == 0.0


def test_unknown_commission_mode_rejected():
    with pytest.raises(ValueError, match="commission_mode"):
        BacktestConfig(**_kwargs(commission_mode='bps'))


def test_unknown_corr_mode_rejected():
    with pytest.raises(ValueError, match="corr_mode"):
        BacktestConfig(**_kwargs(corr_mode='log_return'))


def test_default_corr_floor_and_idm_cap():
    """Futures-first defaults: floor rho at 0, cap IDM at Carver's 2.5."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_floor == 0.0
    assert cfg.idm_cap == 2.5


def test_corr_floor_outside_range_rejected():
    for bad in (1.5, -1.5, float('nan')):
        with pytest.raises(ValueError, match="corr_floor"):
            BacktestConfig(**_kwargs(corr_floor=bad))


def test_corr_floor_none_and_bounds_accepted():
    assert BacktestConfig(**_kwargs(corr_floor=None)).corr_floor is None
    assert BacktestConfig(**_kwargs(corr_floor=-1.0)).corr_floor == -1.0
    assert BacktestConfig(**_kwargs(corr_floor=1.0)).corr_floor == 1.0
    assert BacktestConfig(**_kwargs(corr_floor=0)).corr_floor == 0


def test_idm_cap_below_one_rejected():
    for bad in (0.99, 0.0, -2.5, float('nan')):
        with pytest.raises(ValueError, match="idm_cap"):
            BacktestConfig(**_kwargs(idm_cap=bad))


def test_idm_cap_one_and_none_accepted():
    assert BacktestConfig(**_kwargs(idm_cap=1.0)).idm_cap == 1.0
    assert BacktestConfig(**_kwargs(idm_cap=None)).idm_cap is None


def test_default_corr_shrinkage():
    """Estimation hygiene on by default: Ledoit-Wolf shrink the inline rho."""
    assert BacktestConfig(**_kwargs()).corr_shrinkage == 'ledoit_wolf'


def test_corr_shrinkage_invalid_rejected():
    for bad in ('oas', 'constant_correlation', ''):
        with pytest.raises(ValueError, match="corr_shrinkage"):
            BacktestConfig(**_kwargs(corr_shrinkage=bad))


def test_corr_shrinkage_none_accepted():
    assert BacktestConfig(**_kwargs(corr_shrinkage=None)).corr_shrinkage is None
