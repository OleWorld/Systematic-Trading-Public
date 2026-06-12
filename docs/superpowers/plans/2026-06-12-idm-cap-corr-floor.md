# IDM Cap & Correlation Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Carver risk-policy knobs to `CarverVolTargetingRiskManager` — `corr_floor=0.0` (element-wise floor on the inline-derived correlation matrix) and `idm_cap=2.5` (clamp on the auto-updated IDM) — mirrored on `BacktestConfig`.

**Architecture:** Both knobs live on the risk manager (Approach A of the approved spec at `docs/superpowers/specs/2026-06-12-idm-cap-corr-floor-design.md`); the `analytics` package is untouched. The floor is an *estimation-hygiene* concern: it applies only in `_derive_corr_matrix` (the inline path), never to an explicitly passed `corr_matrix`. The cap is *leverage policy*: it applies at the single `self.idm` auto-assignment site in `calculate_instrument_weight`, regardless of where ρ came from.

**Tech Stack:** Python 3 / pandas / pytest. No new dependencies.

---

## Repo facts the executor must know

- **Python interpreter:** `python` on PATH is a Windows Store stub. Always use the conda env by full path. From a bash shell: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe`.
- **Run tests from the repo root** (`c:\Projects\Systematic-Trading-Public`): `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/ -v`. Repo root must be the CWD (tests rely on it being on `sys.path`).
- **DIRTY WORKING TREE WARNING:** `backtests/test_ewmac.py` and `strategy/ewmac.py` carry pre-existing uncommitted user edits that are NOT part of this work. Never `git add` either file. Task 6 modifies `backtests/test_ewmac.py` but leaves it **uncommitted** — flag this to the user at the end.
- **CLAUDE.md is gitignored** (local-only). Edit it in Task 6 but do not attempt to commit it.
- Stage files individually by path in every commit (`git add <file> <file>`), never `git add -A` / `git add .`.

## File structure

| File | Change |
|---|---|
| `riskmanager/_vol_targeting.py` | Constructor params + validation (`corr_floor`, `idm_cap`); floor in `_derive_corr_matrix`; cap at the IDM auto-update in `calculate_instrument_weight`; docstrings. |
| `config/_backtest.py` | Two new fields + mirrored validation in `__post_init__`. |
| `tests/test_riskmanager_carver.py` | New constructor-validation, floor-behavior, and cap-behavior tests; floor-mirroring updates to three existing tests. |
| `tests/test_config_backtest.py` | New default/validation tests for the two config fields. |
| `backtests/test_ewmac.py` | Pass the two new config fields through to the RM constructor (working-tree only, not committed). |
| `CLAUDE.md` | Document the two knobs (local-only, not committed). |

---

### Task 1: `corr_floor` constructor param + validation

**Files:**
- Modify: `riskmanager/_vol_targeting.py` (constructor, ~lines 191–410)
- Test: `tests/test_riskmanager_carver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_riskmanager_carver.py`, after `test_explicit_corr_matrix_over_subset_is_accepted` (~line 1175). The `_build_rm` helper (defined ~line 900) forwards `**kwargs` to the constructor.

```python
# ──────────────────────────────────────────────
# corr_floor (element-wise ρ floor, Carver: zero out spurious
# negative correlations) — constructor validation
# ──────────────────────────────────────────────

def test_constructor_corr_floor_defaults_to_zero():
    """Futures-first default: floor the inline-derived ρ at 0 (Carver)."""
    rm = _build_rm(['BTC'])
    assert rm.corr_floor == 0.0


def test_constructor_rejects_corr_floor_outside_minus_one_one():
    for bad in (1.5, -1.5, 2.0):
        with pytest.raises(ValueError, match="corr_floor"):
            _build_rm(['BTC'], corr_floor=bad)


def test_constructor_corr_floor_accepts_none_and_bounds():
    """None disables flooring; the closed interval ends are valid."""
    assert _build_rm(['BTC'], corr_floor=None).corr_floor is None
    assert _build_rm(['BTC'], corr_floor=-1.0).corr_floor == -1.0
    assert _build_rm(['BTC'], corr_floor=1.0).corr_floor == 1.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k corr_floor`
Expected: 3 FAILED — `TypeError: __init__() got an unexpected keyword argument 'corr_floor'` (the defaults test fails on the missing attribute).

- [ ] **Step 3: Implement**

In `riskmanager/_vol_targeting.py`:

(a) Add the parameter to the constructor signature after `corr_mode` (~line 205):

```python
        corr_mode: str = 'absolute_price_chg',
        corr_floor: Optional[float] = 0.0,
```

(b) Add validation immediately after the existing `corr_mode` check (the `if corr_mode not in (...)` block ending ~line 373):

```python
        if corr_floor is not None and not (-1.0 <= corr_floor <= 1.0):
            raise ValueError(
                f"corr_floor must be in [-1.0, 1.0] or None to disable, "
                f"got {corr_floor}"
            )
```

(c) Store it alongside the other corr knobs (after `self.corr_mode = corr_mode`, ~line 385):

```python
        self.corr_floor = corr_floor
```

(d) Add the docstring entry in the constructor's Parameters section, directly after the `corr_mode` entry (~line 302):

```
        corr_floor
            Element-wise lower bound applied to the inline-derived
            correlation matrix (see ``_derive_corr_matrix``) before it
            feeds the optimizer and the IDM. Default ``0.0`` — Carver's
            practice: negative correlations estimated from a short
            window are mostly sampling noise, and trusting them both
            overweights spuriously anti-correlated instruments
            (min-variance treats them as a free hedge) and inflates the
            IDM. With the default floor and long-only weights the
            pre-cap IDM is bounded by ``sqrt(N)``. ``None`` disables
            flooring. Must be in ``[-1.0, 1.0]`` when not ``None``.
            NOT applied to an explicitly passed ``corr_matrix`` — the
            caller owns that matrix.
```

(Behavioral application of the floor comes in Task 3; this task only adds the validated, stored parameter.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k corr_floor`
Expected: 3 PASSED.

- [ ] **Step 5: Run the full RM test file (no regressions)**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add riskmanager/_vol_targeting.py tests/test_riskmanager_carver.py
git commit -m "Add corr_floor constructor param to CarverVolTargetingRiskManager

Validated [-1, 1]-or-None knob, default 0.0 (Carver: zero out spurious
negative correlations). Stored only; applied in _derive_corr_matrix in a
follow-up commit.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `idm_cap` constructor param + validation + `idm <= idm_cap` coherence check

**Files:**
- Modify: `riskmanager/_vol_targeting.py` (constructor)
- Test: `tests/test_riskmanager_carver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_riskmanager_carver.py` after the Task 1 tests:

```python
# ──────────────────────────────────────────────
# idm_cap (Carver's 2.5 leverage-policy cap) — constructor validation
# ──────────────────────────────────────────────

def test_constructor_idm_cap_defaults_to_two_point_five():
    rm = _build_rm(['BTC'])
    assert rm.idm_cap == 2.5


def test_constructor_rejects_idm_cap_below_one():
    """DM = 1/sqrt(w'ρw) >= 1 for sum-to-1 non-negative weights, so a
    sub-1 cap would always bind — a config error worth failing on."""
    for bad in (0.99, 0.5, 0.0, -1.0):
        with pytest.raises(ValueError, match="idm_cap"):
            _build_rm(['BTC'], idm_cap=bad)


def test_constructor_idm_cap_accepts_one_and_none():
    assert _build_rm(['BTC'], idm_cap=1.0).idm_cap == 1.0
    assert _build_rm(['BTC'], idm_cap=None).idm_cap is None


def test_constructor_rejects_idm_above_idm_cap():
    """A starting idm above the cap is a contradiction → raise, never
    silently clamp. Lifting or disabling the cap makes the same idm valid."""
    with pytest.raises(ValueError, match="idm_cap"):
        _build_rm(['BTC'], idm=3.0)                 # default cap 2.5
    assert _build_rm(['BTC'], idm=3.0, idm_cap=None).idm == 3.0
    assert _build_rm(['BTC'], idm=3.0, idm_cap=3.5).idm == 3.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k idm_cap`
Expected: 4 FAILED (`unexpected keyword argument 'idm_cap'` / missing attribute / no raise).

- [ ] **Step 3: Implement**

In `riskmanager/_vol_targeting.py`:

(a) Add the parameter directly after `idm` in the signature (~line 197):

```python
        idm: float = 1.0,
        idm_cap: Optional[float] = 2.5,
```

(Safe insertion: every call site in the repo and tests passes `idm` and later params by keyword.)

(b) Extend the validation block — replace the existing `if idm <= 0:` check (~lines 309–310) with:

```python
        if idm <= 0:
            raise ValueError(f"idm must be > 0, got {idm}")
        if idm_cap is not None:
            if idm_cap < 1.0:
                raise ValueError(
                    f"idm_cap must be >= 1.0 or None to disable, got "
                    f"{idm_cap}. (DM = 1/sqrt(w'rho w) >= 1 for sum-to-1 "
                    f"non-negative weights, so a sub-1 cap would always bind.)"
                )
            if idm > idm_cap:
                raise ValueError(
                    f"idm ({idm}) exceeds idm_cap ({idm_cap}); pass a "
                    f"smaller starting idm or raise/disable the cap "
                    f"(idm_cap=None)."
                )
```

(c) Store it next to `self.idm` (~line 377):

```python
        self.idm = idm
        self.idm_cap = idm_cap
```

(d) Docstring entry, directly after the `idm` parameter entry (~line 231). Also append one sentence to the existing `idm` entry: `Must not exceed ``idm_cap`` when the cap is enabled.`

```
        idm_cap
            Upper bound applied to ``self.idm`` whenever it is
            auto-updated from a corr-based weight recompute — both the
            inline-derivation path and an explicitly passed
            ``corr_matrix``. Default ``2.5`` (Carver's recommended
            maximum): the IDM multiplies every position linearly, so
            correlation-estimation noise must not translate into
            unbounded leverage. ``None`` disables the cap. Must be
            ``>= 1.0`` when not ``None`` (the DM is mathematically
            ``>= 1`` for a fully-allocated long-only weight vector).
            Direct assignments to ``self.idm`` by subclasses or
            downstream code are NOT capped — the same owner-may-
            overwrite convention as the weight dicts.
```

(Behavioral application of the cap comes in Task 4.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k idm_cap`
Expected: 4 PASSED.

- [ ] **Step 5: Run the full RM test file (no regressions)**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -q`
Expected: all pass. (Existing tests use `idm` values of 1.0/1.5/1.7/2.0 — all under the 2.5 default cap.)

- [ ] **Step 6: Commit**

```bash
git add riskmanager/_vol_targeting.py tests/test_riskmanager_carver.py
git commit -m "Add idm_cap constructor param to CarverVolTargetingRiskManager

>= 1.0-or-None knob, default 2.5 (Carver's maximum). Constructor idm
must not exceed the cap (raise, not clamp). Stored only; applied at the
IDM auto-update site in a follow-up commit.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Apply the floor in `_derive_corr_matrix` + make existing manual-expectation tests floor-aware

**Files:**
- Modify: `riskmanager/_vol_targeting.py` (`_derive_corr_matrix`, ~lines 576–625)
- Modify: `tests/test_riskmanager_carver.py` (three existing tests at ~lines 1252–1280, ~1312–1349, ~1372–1405)
- Test: `tests/test_riskmanager_carver.py` (new behavior tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_riskmanager_carver.py` (after the Task 2 tests). Note: `corr_lookback=61` → 60 diff observations ≥ the 30-obs minimum, and the 61-bar closes make every symbol pass the data gate exactly.

```python
# ──────────────────────────────────────────────
# corr_floor — behavioral (inline derivation path)
# ──────────────────────────────────────────────

def _anti_correlated_closes():
    """Three 61-bar series: A and B with strongly negative
    absolute-price-change correlation (ρ ≈ -0.8 by construction), C
    independent of both. Deterministic (seeded)."""
    rng = np.random.default_rng(seed=7)
    idx = pd.date_range('2024-01-01', periods=61, freq='D')
    chg_a = rng.normal(0.0, 1.0, 60)
    chg_b = -0.8 * chg_a + 0.6 * rng.normal(0.0, 1.0, 60)
    chg_c = rng.normal(0.0, 1.0, 60)

    def prices(chg):
        return pd.Series(
            100.0 + np.concatenate([[0.0], np.cumsum(chg)]), index=idx,
        )

    return {'A': prices(chg_a), 'B': prices(chg_b), 'C': prices(chg_c)}


def _floor_rm(closes, **kwargs):
    """Min-variance RM over ``closes`` with corr_lookback=61 (all live)."""
    dh = FakeDataHandler(closes=closes)
    return CarverVolTargetingRiskManager(
        FakePortfolio(), FakeStrategy(symbol_list=list(closes)),
        FakeVolEstimator(), data_handler=dh,
        instrument_weight_mode='min_variance',
        corr_lookback=61, annualized_target_vol=0.25, **kwargs,
    )


def test_derived_corr_matrix_is_floored_at_zero_by_default():
    """The inline derivation clips ρ at corr_floor; with corr_floor=None
    the raw (verifiably negative) matrix flows through."""
    closes = _anti_correlated_closes()
    rm = _floor_rm(closes)
    symbols = list(closes)
    floored = rm._derive_corr_matrix('min_variance', symbols)
    off_diag = floored.values[~np.eye(len(floored), dtype=bool)]
    assert off_diag.min() >= 0.0
    # Sanity: the raw matrix really is negative somewhere, otherwise
    # this test proves nothing.
    rm.corr_floor = None
    raw = rm._derive_corr_matrix('min_variance', symbols)
    assert raw.values[~np.eye(len(raw), dtype=bool)].min() < -0.5


def test_corr_floor_prevents_overweighting_of_anti_correlated_pair():
    """min_variance on the raw matrix (corr_floor=None) treats the A/B
    anti-correlation as a free hedge and starves C; the default floor
    removes the spurious credit. Floored IDM respects the sqrt(N) bound;
    the raw IDM exceeds the floored one (idm_cap disabled to expose it)."""
    closes = _anti_correlated_closes()
    floored = _floor_rm(closes)                          # corr_floor=0.0 default
    raw = _floor_rm(closes, corr_floor=None, idm_cap=None)
    assert raw.instrument_weight['C'] < floored.instrument_weight['C']
    assert floored.idm <= math.sqrt(3.0) + 1e-9
    assert raw.idm > floored.idm
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k "floored_at_zero or prevents_overweighting"`
Expected: 2 FAILED (floor not applied yet — `off_diag.min() >= 0.0` fails; the floored/raw runs produce identical weights/IDM).

- [ ] **Step 3: Implement the floor**

In `riskmanager/_vol_targeting.py`, `_derive_corr_matrix` (~line 625), replace the final line

```python
        return correlation_matrix(returns)
```

with:

```python
        corr = correlation_matrix(returns)
        if self.corr_floor is not None:
            # Element-wise floor (Carver: zero out spurious negative
            # correlations before weighting). Clipping preserves symmetry
            # and the 1.0 diagonal for any floor <= 1. NOTE: must be an
            # ``is not None`` check — the default 0.0 is falsy.
            corr = corr.clip(lower=self.corr_floor)
        return corr
```

Also update the `_derive_corr_matrix` docstring: after the sentence ending "and calling ``analytics.correlation_matrix`` on the result.", add:

```
        When ``self.corr_floor`` is not ``None``, the resulting matrix
        is element-wise floored at ``corr_floor`` before being returned
        (estimation hygiene: applies to this inline path only, never to
        an explicitly passed ``corr_matrix``).
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k "floored_at_zero or prevents_overweighting"`
Expected: 2 PASSED.

- [ ] **Step 5: Make the three existing manual-expectation tests floor-aware**

These tests replay the IDM equation from a manually computed corr matrix; they must now mirror the default floor (and would otherwise fail whenever the seeded sample ρ is negative).

(a) `test_min_variance_derives_corr_from_filled_deques_and_auto_updates_idm` (~line 1278) — replace:

```python
    expected_corr = trailing.pct_change(fill_method=None).dropna().corr()
```

with:

```python
    expected_corr = (
        trailing.pct_change(fill_method=None).dropna().corr()
        .clip(lower=0.0)                # mirror the RM's default corr_floor
    )
```

(b) `test_returns_used_are_simple_pct_change` (~lines 1335–1336) — replace:

```python
    expected_corr = pd.DataFrame(closes).pct_change(fill_method=None).dropna().corr()
    rho = expected_corr.loc['A', 'B']
```

with:

```python
    expected_corr = (
        pd.DataFrame(closes).pct_change(fill_method=None).dropna().corr()
        .clip(lower=0.0)                # mirror the RM's default corr_floor
    )
    rho = expected_corr.loc['A', 'B']
```

(The closed-form sanity `1/sqrt((1+ρ)/2)` further down stays valid because `rho` is read from the clipped matrix.)

(c) `test_absolute_price_chg_mode_uses_diff` (~lines 1392–1393) — replace:

```python
    expected_corr = pd.DataFrame(closes).diff().dropna().corr()
    rho = expected_corr.loc['A', 'B']
```

with:

```python
    expected_corr_raw = pd.DataFrame(closes).diff().dropna().corr()
    expected_corr = expected_corr_raw.clip(lower=0.0)   # mirror the RM's default corr_floor
    # The mode-disambiguation sanity below needs the UNfloored value
    # (both modes' ρ could clip to the same 0.0).
    rho = expected_corr_raw.loc['A', 'B']
```

(The `expected_idm` replay in that test keeps using `expected_corr`; the `rho` vs `pct_rho` sanity keeps using raw values.)

- [ ] **Step 6: Run the full RM test file (no regressions)**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -q`
Expected: all pass. If any *other* test fails by replaying a manually computed corr matrix, apply the same `.clip(lower=0.0)` mirroring shown in Step 5.

- [ ] **Step 7: Commit**

```bash
git add riskmanager/_vol_targeting.py tests/test_riskmanager_carver.py
git commit -m "Floor inline-derived correlation matrix at corr_floor

_derive_corr_matrix clips rho element-wise (default floor 0.0) before
it feeds the optimizer and the IDM, so spurious negative correlations
from the short estimation window neither overweight 'free hedge' pairs
nor inflate the IDM. Explicit corr_matrix path stays raw. Existing
manual-expectation tests mirror the floor.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Apply the cap at the IDM auto-update site

**Files:**
- Modify: `riskmanager/_vol_targeting.py` (`calculate_instrument_weight`, ~lines 556–560)
- Test: `tests/test_riskmanager_carver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_riskmanager_carver.py`. (`_corr_df` is the existing helper at ~line 919; 7 mutually uncorrelated labels give raw DM = √7 ≈ 2.6458 > 2.5.)

```python
# ──────────────────────────────────────────────
# idm_cap — behavioral (auto-update site)
# ──────────────────────────────────────────────

def test_idm_capped_at_default_two_point_five():
    """7 uncorrelated instruments → raw DM = sqrt(7) ≈ 2.6458; the stored
    idm is clamped to the default 2.5. Exercised via the explicit
    corr_matrix path: the cap is leverage policy and applies regardless
    of where the matrix came from."""
    labels = [f'S{i}' for i in range(7)]
    rm = _build_rm(labels)
    corr = _corr_df(labels, off_diag=0.0)
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    assert rm.idm == 2.5


def test_idm_cap_none_disables_capping():
    labels = [f'S{i}' for i in range(7)]
    rm = _build_rm(labels, idm_cap=None)
    corr = _corr_df(labels, off_diag=0.0)
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=corr)
    assert math.isclose(rm.idm, math.sqrt(7.0), rel_tol=1e-6)


def test_idm_cap_applies_under_risk_parity_too():
    """The cap sits at the shared assignment site, after the mode dispatch."""
    labels = [f'S{i}' for i in range(7)]
    rm = _build_rm(labels)
    corr = _corr_df(labels, off_diag=0.0)
    rm.calculate_instrument_weight(mode='risk_parity', corr_matrix=corr)
    assert rm.idm == 2.5


def test_explicit_corr_matrix_is_not_floored():
    """The research hook owns the matrix: negative entries reach the
    optimizer raw. A/B at ρ=-0.5 with C uncorrelated → exact long-only
    min-variance is (0.4, 0.4, 0.2): the 'hedged' pair out-weights C.
    A (wrongly) floored matrix would yield w_C >= w_A instead."""
    rm = _build_rm(['A', 'B', 'C'])
    rho = pd.DataFrame(
        [[1.0, -0.5, 0.0],
         [-0.5, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
    )
    rm.calculate_instrument_weight(mode='min_variance', corr_matrix=rho)
    w = rm.instrument_weight
    assert math.isclose(w['A'], 0.4, abs_tol=1e-6)
    assert math.isclose(w['B'], 0.4, abs_tol=1e-6)
    assert math.isclose(w['C'], 0.2, abs_tol=1e-6)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k "idm_capped or cap_none_disables or cap_applies_under or not_floored"`
Expected: `test_idm_capped_at_default_two_point_five` and `test_idm_cap_applies_under_risk_parity_too` FAIL (idm ≈ 2.6458, uncapped); the other two PASS already (they pin current behavior — keep them as regression guards).

- [ ] **Step 3: Implement the cap**

In `riskmanager/_vol_targeting.py`, `calculate_instrument_weight` (~lines 556–560), replace:

```python
            # Auto-update IDM from the same matrix used for weights so
            # the two stay coherent across walk-forward recomputes.
            self.idm = diversification_multiplier(
                self.instrument_weight, corr_matrix,
            )
```

with:

```python
            # Auto-update IDM from the same matrix used for weights so
            # the two stay coherent across walk-forward recomputes. The
            # cap is leverage policy: it applies regardless of whether
            # the matrix was derived inline or passed explicitly.
            idm = diversification_multiplier(
                self.instrument_weight, corr_matrix,
            )
            if self.idm_cap is not None:
                idm = min(idm, self.idm_cap)
            self.idm = idm
```

Also update the `calculate_instrument_weight` docstring Notes section: change "also updates ``self.idm`` via ``analytics.diversification_multiplier(...)``" to "also updates ``self.idm`` via ``analytics.diversification_multiplier(...)``, clamped to ``idm_cap`` when the cap is enabled".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -v -k "idm_capped or cap_none_disables or cap_applies_under or not_floored"`
Expected: 4 PASSED.

- [ ] **Step 5: Run the full RM test file (no regressions)**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_riskmanager_carver.py -q`
Expected: all pass. (Existing IDM expectations are √2 ≈ 1.414 or 2-asset values ≤ √2 — all under 2.5; the Task 3 floor additionally guarantees inline-path IDM ≤ √N.)

- [ ] **Step 6: Commit**

```bash
git add riskmanager/_vol_targeting.py tests/test_riskmanager_carver.py
git commit -m "Cap auto-updated IDM at idm_cap

calculate_instrument_weight clamps the diversification multiplier to
idm_cap (default 2.5, Carver's maximum) at the shared assignment site —
inline and explicit corr_matrix paths alike. Direct self.idm overwrites
remain uncapped by convention.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `BacktestConfig` fields + mirrored validation

**Files:**
- Modify: `config/_backtest.py`
- Test: `tests/test_config_backtest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_backtest.py`:

```python
def test_default_corr_floor_and_idm_cap():
    """Futures-first defaults: floor rho at 0, cap IDM at Carver's 2.5."""
    cfg = BacktestConfig(**_kwargs())
    assert cfg.corr_floor == 0.0
    assert cfg.idm_cap == 2.5


def test_corr_floor_outside_range_rejected():
    for bad in (1.5, -1.5):
        with pytest.raises(ValueError, match="corr_floor"):
            BacktestConfig(**_kwargs(corr_floor=bad))


def test_corr_floor_none_and_bounds_accepted():
    assert BacktestConfig(**_kwargs(corr_floor=None)).corr_floor is None
    assert BacktestConfig(**_kwargs(corr_floor=-1.0)).corr_floor == -1.0
    assert BacktestConfig(**_kwargs(corr_floor=1.0)).corr_floor == 1.0


def test_idm_cap_below_one_rejected():
    for bad in (0.99, 0.0, -2.5):
        with pytest.raises(ValueError, match="idm_cap"):
            BacktestConfig(**_kwargs(idm_cap=bad))


def test_idm_cap_one_and_none_accepted():
    assert BacktestConfig(**_kwargs(idm_cap=1.0)).idm_cap == 1.0
    assert BacktestConfig(**_kwargs(idm_cap=None)).idm_cap is None
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_config_backtest.py -v -k "corr_floor or idm_cap"`
Expected: 5 FAILED — `TypeError: __init__() got an unexpected keyword argument`.

- [ ] **Step 3: Implement**

In `config/_backtest.py`:

(a) Add the fields after `corr_mode` (~line 42):

```python
    corr_floor: Optional[float] = 0.0    # element-wise floor on the inline-derived rho; None disables (Carver: zero out spurious negative correlations)
    idm_cap: Optional[float] = 2.5       # cap on the auto-updated IDM; None disables (Carver's 2.5; >= 1.0 since DM >= 1 for long-only sum-to-1 weights)
```

(b) Add mirrored validation in `__post_init__`, after the `corr_mode` check (~line 175):

```python
        if self.corr_floor is not None and not (-1.0 <= self.corr_floor <= 1.0):
            raise ValueError(
                f"corr_floor must be in [-1.0, 1.0] or None to disable, "
                f"got {self.corr_floor}"
            )
        if self.idm_cap is not None and self.idm_cap < 1.0:
            raise ValueError(
                f"idm_cap must be >= 1.0 or None to disable, got "
                f"{self.idm_cap}. (DM = 1/sqrt(w'rho w) >= 1 for sum-to-1 "
                f"non-negative weights, so a sub-1 cap would always bind.)"
            )
```

(c) Update the dataclass comment block at ~lines 32–34 (the `# Carver vol-targeting knobs ...` comment): the note "``idm`` is not in config — pass it directly to the risk manager constructor if a non-default value is needed." stays accurate — `idm_cap` IS in config, the starting `idm` is not. No change needed beyond adding the two fields.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/test_config_backtest.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add config/_backtest.py tests/test_config_backtest.py
git commit -m "Mirror corr_floor and idm_cap on BacktestConfig

Same validation as the CarverVolTargetingRiskManager constructor so bad
values fail at config construction, not deep in the wiring.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Smoke-runner wiring, CLAUDE.md, full verification

**Files:**
- Modify: `backtests/test_ewmac.py` (~line 118) — **DO NOT COMMIT** (carries unrelated user WIP)
- Modify: `CLAUDE.md` — **DO NOT COMMIT** (gitignored, local-only)

- [ ] **Step 1: Pass the knobs through in the smoke runner**

In `backtests/test_ewmac.py`, the `CarverVolTargetingRiskManager(...)` call (~lines 108–119), after `corr_mode=config.corr_mode,` add:

```python
    corr_floor=config.corr_floor,
    idm_cap=config.idm_cap,
```

- [ ] **Step 2: Update CLAUDE.md**

(a) In the `riskmanager/` → `_vol_targeting.py` bullet, extend the default-constructor sentence: change

`Default constructor: idm=1.0, annualized_target_vol=None (required), vol_target_mode='dollar_volatility', position_buffer=0.25, corr_lookback=60, corr_step_size=30, corr_timeframe='1d', corr_mode='absolute_price_chg'.`

to

`Default constructor: idm=1.0, idm_cap=2.5, annualized_target_vol=None (required), vol_target_mode='dollar_volatility', position_buffer=0.25, corr_lookback=60, corr_step_size=30, corr_timeframe='1d', corr_mode='absolute_price_chg', corr_floor=0.0.`

and, in the same bullet, after the sentence describing the IDM auto-update ("On every successful corr-based compute … the equal-weight-fallback path leaves `self.idm` untouched."), insert:

`Two Carver guardrails bound estimation noise: **corr_floor** (default 0.0; None disables; must be in [-1, 1]) element-wise floors the *inline-derived* ρ in _derive_corr_matrix before it feeds the optimizer and the IDM — spurious negative correlations from the short window neither overweight "free hedge" pairs nor inflate the IDM, and with the default floor the long-only quadratic form guarantees pre-cap IDM ≤ √N (the explicitly-passed corr_matrix hook is NOT floored — caller owns the matrix); **idm_cap** (default 2.5 — Carver's maximum; None disables; must be ≥ 1.0, and the constructor idm must not exceed it) clamps self.idm at the auto-update site in every corr-based compute, inline or explicit — direct self.idm overwrites stay uncapped. Floor = estimation hygiene (inline path only); cap = leverage policy (every auto-assignment).`

(b) In the **Notes for Future Agents** → "Default risk-manager stack" bullet, extend the BacktestConfig knob list: change

`The Carver knobs vol_target_mode, annualized_target_vol, position_buffer (…), instrument_weight_mode (…), corr_lookback (…), corr_step_size (default 30), corr_timeframe (default '1d'), and corr_mode (…) all live on BacktestConfig`

to include, after the `corr_mode` entry:

`, corr_floor (default 0.0 — element-wise floor on the inline-derived ρ; None disables), and idm_cap (default 2.5 — cap on the auto-updated IDM; None disables)`

(c) Design doc reference: in the same `_vol_targeting.py` bullet, optionally append `See docs/superpowers/specs/2026-06-12-idm-cap-corr-floor-design.md for the floor/cap design rationale.`

- [ ] **Step 3: Run the full test suite**

Run from repo root: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe -m pytest tests/ -q`
Expected: all pass, zero failures.

- [ ] **Step 4: Run the smoke backtest**

Run from repo root: `/c/Users/lizxd/anaconda3/envs/VibeCoding/python.exe backtests/test_ewmac.py`
Expected: completes without error; the printed `IDM:` line (portfolio summary section) shows a value ≤ 2.5.

- [ ] **Step 5: Verify nothing unintended is staged, then report**

```bash
git status --short
```

Expected: `backtests/test_ewmac.py` and `strategy/ewmac.py` modified-but-uncommitted (user WIP + our two wiring lines); CLAUDE.md changes invisible (gitignored); nothing staged.

**Report to the user:** the smoke-runner pass-through lines are intentionally left uncommitted because `backtests/test_ewmac.py` carries their unrelated uncommitted edits (`lookback_pairs`, weights, fdm overrides); they should commit that file themselves when their WIP is ready.

---

## Self-review (completed at plan time)

- **Spec coverage:** floor semantics (Task 1 + 3), cap semantics (Task 2 + 4), config mirror (Task 5), wiring + CLAUDE.md (Task 6), all six spec test bullets mapped (floor default → T3; floor disabled → T3; research hook unfloored + capped → T4; cap binds / None disables → T4; constructor validation → T1/T2; config validation → T5; fixture review → T3 Step 5).
- **Placeholder scan:** every code step carries the actual code; no TBDs.
- **Type consistency:** `corr_floor: Optional[float]`, `idm_cap: Optional[float]` consistent across RM constructor, config fields, and tests; `_build_rm`/`_floor_rm` kwargs flow through `**kwargs`.
