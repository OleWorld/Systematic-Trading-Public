"""
Unit tests for ``config/_instrument.py`` (InstrumentConfig + uniform_registry).

Pins the per-symbol metadata bundle's defaults (backward-compatible
``point_value=1`` / ``fractional=True``), its validation, and the
``uniform_registry`` factory that builds a homogeneous registry in one call.

Run from the repo root:  pytest tests/test_instrument.py -v
"""

import pytest

from config import InstrumentConfig, uniform_registry
from execution import SlippageModel, CommissionModel
from portfolio import MarginModel, PortfolioMarginModel


# ──────────────────────────────────────────────
# InstrumentConfig — defaults & construction
# ──────────────────────────────────────────────

def test_defaults_are_backward_compatible():
    """Default config reproduces today's crypto behaviour: pv=1, fractional."""
    cfg = InstrumentConfig()
    assert cfg.point_value == 1.0
    assert cfg.fractional is True
    assert isinstance(cfg.slippage, SlippageModel)
    assert isinstance(cfg.commission, CommissionModel)
    assert isinstance(cfg.margin, MarginModel)
    # default commission is the futures-first per_contract/0.0
    assert cfg.commission.mode == 'per_contract'


def test_explicit_construction_holds_values():
    s = SlippageModel('pct', 0.001)
    c = CommissionModel('rate', 0.0004)
    m = PortfolioMarginModel.from_leverage(20.0, maintenance_margin_rate=0.02)
    cfg = InstrumentConfig(point_value=1000.0, fractional=False,
                           slippage=s, commission=c, margin=m)
    assert cfg.point_value == 1000.0
    assert cfg.fractional is False
    assert cfg.slippage is s
    assert cfg.commission is c
    assert cfg.margin is m


@pytest.mark.parametrize("bad_pv", [0.0, -1.0, float('nan')])
def test_non_positive_point_value_raises(bad_pv):
    with pytest.raises(ValueError):
        InstrumentConfig(point_value=bad_pv)


# ──────────────────────────────────────────────
# uniform_registry
# ──────────────────────────────────────────────

def test_uniform_registry_covers_every_symbol():
    reg = uniform_registry(['BTC', 'ETH', 'SOL'])
    assert set(reg.keys()) == {'BTC', 'ETH', 'SOL'}
    for cfg in reg.values():
        assert isinstance(cfg, InstrumentConfig)


def test_uniform_registry_applies_shared_metadata():
    s = SlippageModel('absolute', 0.5)
    c = CommissionModel('per_contract', 2.5)
    m = PortfolioMarginModel.from_leverage(10.0, maintenance_margin_rate=0.05)
    reg = uniform_registry(
        ['ES', 'NQ'], point_value=50.0, fractional=False,
        slippage=s, commission=c, margin=m,
    )
    for cfg in reg.values():
        assert cfg.point_value == 50.0
        assert cfg.fractional is False
        assert cfg.slippage is s
        assert cfg.commission is c
        assert cfg.margin is m


def test_uniform_registry_defaults_match_instrument_defaults():
    reg = uniform_registry(['BTC'])
    cfg = reg['BTC']
    assert cfg.point_value == 1.0
    assert cfg.fractional is True


def test_uniform_registry_empty_symbols_raises():
    with pytest.raises(ValueError):
        uniform_registry([])


def test_uniform_registry_duplicate_symbols_raises():
    with pytest.raises(ValueError):
        uniform_registry(['BTC', 'BTC'])


# ──────────────────────────────────────────────
# Frozen dataclass (F21)
# ──────────────────────────────────────────────

import dataclasses


def test_instrument_config_is_frozen():
    cfg = InstrumentConfig(point_value=1000.0, fractional=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.point_value = 2000.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.fractional = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.slippage = None


def test_uniform_registry_still_shares_one_instance():
    reg = uniform_registry(['A', 'B', 'C'])
    assert reg['A'] is reg['B'] is reg['C']   # dedupe-by-identity preserved
