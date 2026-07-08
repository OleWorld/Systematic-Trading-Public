"""Tests for validation._bootstrap — resampling machinery + bootstrap_stats."""

import numpy as np
import pandas as pd
import pytest

from validation._bootstrap import (_flat_top_window, _resample_indices,
                                   politis_white_block_length)


def _ar1(t, phi, seed=0, scale=1.0):
    """AR(1) series x_t = phi*x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, scale, size=t)
    x = np.empty(t)
    x[0] = eps[0]
    for i in range(1, t):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _lag1_autocorr(m):
    """Mean lag-1 autocorrelation across rows of a 2-D array."""
    a, b = m[:, :-1], m[:, 1:]
    ac = ((a - a.mean(axis=1, keepdims=True))
          * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
    return float((ac / (a.std(axis=1) * b.std(axis=1))).mean())


def test_flat_top_window_hand_values():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    assert list(_flat_top_window(x)) == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]


def test_block_length_larger_for_persistent_series():
    white = politis_white_block_length(_ar1(1000, phi=0.0, seed=1))
    persistent = politis_white_block_length(_ar1(1000, phi=0.9, seed=1))
    assert persistent > white
    assert persistent > 10.0
    assert white < 5.0


def test_block_length_bounds_and_degenerate():
    t = 500
    b = politis_white_block_length(_ar1(t, phi=0.95, seed=2))
    assert 1.0 <= b <= min(3.0 * np.sqrt(t), t / 3.0)
    assert politis_white_block_length(np.zeros(100)) == 1.0   # constant
    assert politis_white_block_length(np.array([1.0, 2.0])) == 1.0  # too short


def test_block_length_deterministic():
    x = _ar1(800, phi=0.5, seed=3)
    assert politis_white_block_length(x) == politis_white_block_length(x)


def test_indices_shapes_and_ranges():
    rng = np.random.default_rng(0)
    for method in ('stationary', 'circular', 'iid'):
        idx = _resample_indices(rng, t=50, b=7, method=method, block_length=5.0)
        assert idx.shape == (7, 50)
        assert idx.min() >= 0 and idx.max() < 50


def test_unknown_method_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _resample_indices(rng, t=50, b=2, method='jackknife', block_length=5.0)


def test_block_methods_preserve_autocorrelation_iid_destroys_it():
    x = _ar1(1000, phi=0.8, seed=4)
    rng = np.random.default_rng(5)
    stat = x[_resample_indices(rng, 1000, 200, 'stationary', 20.0)]
    circ = x[_resample_indices(rng, 1000, 200, 'circular', 20.0)]
    iid = x[_resample_indices(rng, 1000, 200, 'iid', 20.0)]
    assert _lag1_autocorr(stat) > 0.5
    assert _lag1_autocorr(circ) > 0.5
    assert abs(_lag1_autocorr(iid)) < 0.1


def test_stationary_blocks_are_contiguous_runs():
    """Within a resample row, consecutive indices mostly step by +1 (mod T)."""
    rng = np.random.default_rng(6)
    idx = _resample_indices(rng, t=200, b=50, method='stationary',
                            block_length=10.0)
    steps = (idx[:, 1:] - idx[:, :-1]) % 200
    frac_contiguous = float((steps == 1).mean())
    assert 0.8 < frac_contiguous < 0.95   # ~1 - 1/block_length
