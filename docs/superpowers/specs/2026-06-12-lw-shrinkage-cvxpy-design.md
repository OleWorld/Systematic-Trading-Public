# Ledoit-Wolf Shrinkage & CVXPY Optimizer Migration — Design

**Date:** 2026-06-12
**Status:** Approved
**Scope:** `analytics/_correlation.py`, `analytics/_portfolio_optimizer.py`, `riskmanager/_vol_targeting.py`, `config/_backtest.py`, `backtests/test_ewmac.py`, tests, CLAUDE.md.

## Problem

The portfolio is about to grow from 5 crypto perps to a larger instrument universe. Two problems follow:

1. **Estimation noise.** `CarverVolTargetingRiskManager._derive_corr_matrix` estimates ρ from only `corr_lookback=60` observations. As N grows toward (and past) the window length, the sample correlation matrix becomes noisy and ill-conditioned — eventually singular when N exceeds the observation count. Ledoit-Wolf shrinkage (sample covariance shrunk toward scaled identity with the closed-form optimal intensity) keeps the estimate well-conditioned and positive-definite at any N.
2. **Solver robustness.** `analytics._portfolio_optimizer` uses scipy SLSQP / L-BFGS-B. The honest case for CVXPY, settled during brainstorming: at this problem size CVXPY is *not* faster — canonicalization overhead exceeds SLSQP's entire solve time for an N≤100 QP, and solves only run at the `corr_step_size` walk-forward cadence anyway. Its value is robustness (CLARABEL/OSQP are purpose-built convex solvers that handle near-singular matrices far better than SLSQP) and expressiveness (future constraint sets — max-weight caps, market-neutral `Σw=0, Σ|w|=1` — become one-line changes).

## Decision record

- **Both optimizers migrate to CVXPY** (`min_variance`, `risk_parity`), not just the QP. One solver stack; `scipy.optimize` leaves the module.
- **Ledoit-Wolf via scikit-learn** (`sklearn.covariance.LedoitWolf`, the 2004 scaled-identity-target estimator), not an in-house implementation. sklearn was installed into the `VibeCoding` env for this work; cvxpy 1.9.1 + CLARABEL were already present. No requirements file exists — deps are recorded in CLAUDE.md.
- **Exposure:** new knob `corr_shrinkage: Optional[str] = 'ledoit_wolf'` — default **ON** — as an RM constructor param mirrored on `BacktestConfig`, applied in the inline `_derive_corr_matrix` path only. An explicitly passed `corr_matrix=` stays caller-owned, the same contract as `corr_floor`. Shrinkage is estimation hygiene, so it sits with the other estimation-hygiene knob on the RM; the matrix math itself lives in `analytics`.
- **PSD supersession.** The 2026-06-12 IDM-cap/corr-floor spec argued the floored matrix's possible non-PSD-ness was moot because every consumer evaluates the quadratic form only on the long-only feasible region. CVXPY breaks that argument: DCP certification of `quad_form` is *global*, not feasible-region-local. This work therefore adds explicit PSD handling at both boundaries — producer (post-floor eigenvalue-clip repair in the RM, so the optimizer and the IDM consume the same clean matrix) and consumer (validation in `_build_sigma`, so explicitly passed matrices fail loudly instead of silently feeding a non-convex QP, which is what SLSQP used to accept).

## Design

### 1. `analytics/_correlation.py` — shrinkage option on `correlation_matrix`

Extend the single existing entry point (no parallel function — the RM already calls this):

```python
correlation_matrix(values, *, lookback=None, method='pearson', shrinkage=None)
```

- `shrinkage ∈ {None, 'ledoit_wolf'}`; strict validation (`ValueError` on anything else). `'ledoit_wolf'` requires `method='pearson'` (LW is a covariance-based estimator) — `ValueError` otherwise.
- LW path: **lazy import** of `sklearn.covariance.LedoitWolf` inside the branch with a crisp `ImportError` message — keeps `from analytics import ...` (and hence the whole project) importable without sklearn when shrinkage is off.
- Fit on `window.dropna()` — LW needs complete rows (listwise deletion), unlike pandas's pairwise handling on the unshrunk path; the docstring documents the difference. Convert shrunk covariance → correlation: `D⁻¹ Σ D⁻¹` with `D = sqrt(diag(Σ))`, then `np.clip(..., -1.0, 1.0)` for numerical hygiene and an exact-1.0 diagonal restore; rebuild as a labeled DataFrame.
- Diagnostics without API break: the fitted intensity is attached as `result.attrs['lw_shrinkage'] = float(lw.shrinkage_)`; the RM logs it per recalc.

### 2. `analytics/_portfolio_optimizer.py` — scipy → CVXPY

- `from scipy.optimize import minimize` → top-level `import cvxpy as cp`. Public signatures unchanged (`corr_matrix, vols=None, *, tol, max_iter`).
- **`min_variance`:** `cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), [cp.sum(w) == 1, w >= 0])`, `solver=cp.CLARABEL`. `psd_wrap` skips CVXPY's eigen-check — safe because `_build_sigma` now performs its own PSD validation/repair (below).
- **`risk_parity`:** Spinu's strictly convex form stays, expressed in CVXPY: `cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(sigma)) - cp.sum(cp.log(w)))`, no explicit bounds (the log domain enforces `w > 0`), CLARABEL (exponential cone). The interior-point solution is then refined by a short **Newton polish** on the first-order system `Σw − 1/w = 0` (SPD Hessian `Σ + diag(1/w²)`, quadratic convergence, step-halving for positivity) before the `w / w.sum()` normalization — exp-cone solves deliver only ~tolerance-level weight accuracy, and the polish closes the gap to ~machine precision so the documented ≈1e-6/1e-9 output precision holds. *(Added during implementation: discovered when the ERC golden tests demanded tighter weights than CLARABEL's exp-cone path provides at any stable tolerance.)*
- **PSD validation/repair in `_build_sigma`** (new): after building Σ, `np.linalg.eigvalsh`; `λ_min >= -1e-8` → clip negative eigenvalues to 0, reconstruct, symmetrize (numerical-dust projection); `λ_min < -1e-8` → `ValueError` ("materially non-PSD"). Previously such a matrix flowed silently into SLSQP as a non-convex QP.
- **`tol` / `max_iter`:** retained with identical validation; forwarded to CLARABEL as `max_iter` and `tol_gap_abs`/`tol_gap_rel`/`tol_feas`. **Defaults (corrected during implementation — the draft assumed 1e-12 was uniformly unreachable for interior-point, which measurement disproved for the QP):** `min_variance` keeps `tol=1e-12` (CLARABEL reaches it cleanly on QPs at these sizes); `risk_parity` uses `tol=1e-9` for the CVXPY stage (1e-12 trips CLARABEL's `optimal_inaccurate` reduced-accuracy fallback on some exp-cone instances) and relies on the Newton polish for final precision. Existing test tolerances (≈1e-6/1e-9 asserts) hold.
- **Failure contract preserved:** catch `cp.SolverError`; accept only `prob.status == cp.OPTIMAL`; anything else, including `OPTIMAL_INACCURATE`, → `ValueError(f"... solver failed: {status}")` — strict, as today.

### 3. `riskmanager/_vol_targeting.py` — `corr_shrinkage` knob + post-floor PSD repair

- New constructor param `corr_shrinkage: Optional[str] = 'ledoit_wolf'`, validated at construction against `{None, 'ledoit_wolf'}`.
- `_derive_corr_matrix` pipeline order (documented in its docstring): pull closes → diff/pct_change → `_MIN_CORR_OBS` gate → `correlation_matrix(returns, shrinkage=self.corr_shrinkage)` → `corr_floor` clip → **PSD repair** → return.
- PSD repair: private helper `_nearest_psd_correlation(corr)` — cheap `eigvalsh` check first; only when the floor actually broke PSD (`λ_min < 0`): clip eigenvalues at 0, reconstruct, rescale to unit diagonal, re-symmetrize. Guarantees every inline-derived matrix feeding the optimizer AND `diversification_multiplier` is a clean correlation matrix (preserves the "weights and IDM from the same matrix" invariant).
- DEBUG-log the LW intensity from `corr.attrs` when present (one line per walk-forward recalc).

### 4. `config/_backtest.py`

`corr_shrinkage: Optional[str] = 'ledoit_wolf'` on `BacktestConfig`, validated in `__post_init__`, same pattern as `corr_floor`/`idm_cap`.

### 5. `backtests/test_ewmac.py`

Pass `corr_shrinkage=config.corr_shrinkage` to the RM constructor (same pattern as the existing corr_floor/idm_cap wiring).

## Testing

- The existing optimizer suite (~33 tests) is the regression net: all golden/closed-form/validation tests must pass **unchanged**. The `max_iter=1` failure-injection approach carries over to CLARABEL without monkeypatching; the one fixture change is `test_min_variance_solver_failure_raises`, whose pre-CVXPY matrix turned out to be materially non-PSD (λ_min ≈ −0.27 — it merely stalled SLSQP) and is now rejected up front by the PSD validation, so failure is injected on a valid matrix instead.
- New shrinkage tests in `tests/test_analytics_correlation.py`: golden vs a direct sklearn `LedoitWolf` fit; output is a valid correlation matrix (unit diagonal, symmetric, PD, entries in [-1, 1]); off-diagonals pulled toward 0 vs the sample estimate; `attrs['lw_shrinkage']` in (0, 1]; `lookback` respected; validation raises (unknown value; `method != 'pearson'`).
- New PSD tests: `_build_sigma` repairs an eps-non-PSD matrix and raises on a materially non-PSD one; RM-level adversarial case (≥3 assets where `clip(lower=0)` breaks PSD) gets repaired before feeding the optimizer.
- RM knob tests: `corr_shrinkage=None` reproduces the pre-change matrix bit-for-bit; default forwards `'ledoit_wolf'`; constructor rejects invalid values; `BacktestConfig` mirrors.

## Out of scope (explicitly deferred)

- Shrinkage of the optional `vols=` covariance path — the RM stack is correlation-only by design; the `vols` hook is unchanged.
- Constant-correlation (LW 2003) or OAS shrinkage targets — sklearn's identity-target LW only.
- Market-neutral / max-weight constraint schemes — future work the CVXPY migration deliberately makes cheap.
- Delisting/universe-exit handling (pre-existing future work, unaffected).

## Verification (results)

1. `python -m pytest tests/ -v` (VibeCoding interpreter) — **598 passed**; optimizer goldens unchanged.
2. `python backtests/test_ewmac.py` (default, shrinkage ON) — completes clean: Return +17.59%, Sharpe 0.24, IDM at the 2.5 cap, weights sum to 1 and ≥ 0.
3. **Correction to the draft's "OFF = no-op vs main" claim — it was wrong and could never hold.** The matrix-level no-op *is* proven: `test_corr_shrinkage_none_reproduces_raw_sample_corr` shows `correlation_matrix(..., shrinkage=None)` is bit-identical to the old raw `pd.DataFrame.corr`. But the *smoke* output cannot match `main`, because the inline RM path now always runs `_nearest_psd_correlation`, and `main` was silently feeding **non-PSD** floored matrices to scipy SLSQP (a non-convex QP scipy accepted without complaint). Measured on the real 532-instrument universe (60-obs window): raw correlation is rank 58/532 (cond ≈ 1e302); floored-at-0 has **λ_min = −1.083**; Ledoit-Wolf-shrunk is full-rank (cond ≈ 196) at only 0.029 intensity. So the new `corr_shrinkage=None` smoke run returns **−42.8%** — the high-dimensional failure mode of the raw sample estimator, working as intended and demonstrating *why* shrinkage is the default. `main`'s +12.58% was a non-convex-QP artifact on a broken matrix, not a trustworthy baseline.
4. REPL stress: `min_variance` / `risk_parity` solve a 100×100 near-singular sample correlation cleanly under CLARABEL (`test_*_solves_near_singular_100x100`), the robustness claim the migration was made for.
