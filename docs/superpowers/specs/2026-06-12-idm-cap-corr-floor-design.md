# IDM Cap & Correlation Floor — Design

**Date:** 2026-06-12
**Status:** Approved
**Scope:** `riskmanager/_vol_targeting.py`, `config/_backtest.py`, `backtests/test_ewmac.py`, tests, CLAUDE.md. The `analytics` package is untouched.

## Problem

Two leverage amplifiers in `CarverVolTargetingRiskManager` are currently unbounded against correlation-estimation noise:

1. **Unbounded IDM.** `self.idm` multiplies every position linearly. It is auto-set to `1 / sqrt(wᵀρw)` on each corr-based weight recompute; noisy ρ estimates that drag average correlation toward zero or negative inflate the IDM — and therefore leverage on the entire book — without limit.
2. **Spurious negative correlations overweight instruments.** The inline ρ is estimated from a short trailing window (`corr_lookback=60` by default → standard error ≈ 1/√60 ≈ 0.13). `min_variance` rewards negative correlation aggressively: a spuriously anti-correlated pair is overweighted as a free hedge. When the regime snaps back (correlations rise toward +1 in stress), the portfolio is maximally concentrated exactly when diversification vanishes. The same negative entries simultaneously shrink `wᵀρw` and inflate the IDM — two amplifiers fed by one bad estimate.

## Decision record (the floor debate)

**Resolution: floor the inline-derived correlation matrix at 0, on by default, configurable.** This is also Carver's documented practice (zero negative correlations before computing weights and the DM; cap the IDM at 2.5).

For flooring: (a) at 60 observations, negative readings are mostly sampling noise; (b) the error cost is asymmetric — ignoring a true hedge costs a little efficiency, trusting a false hedge costs a drawdown; (c) negative entries compound through both the optimizer weights and the IDM.

Against flooring: (a) genuine structural hedges lose diversification credit; (b) elementwise clipping is statistically crude vs. shrinkage; (c) clipped matrices are not PSD in general.

Why the objections dissolve in this stack: every downstream consumer is **long-only** (`w ≥ 0` in `min_variance`, `risk_parity`, and the `diversification_multiplier` contract). With non-negative weights and a floored-at-0 matrix, `wᵀρw = Σwᵢ² + Σ(non-negative cross terms) ≥ Σwᵢ² > 0`, so the quadratic form is strictly positive on the entire feasible region — the PSD concern is moot — and the IDM is bounded by √N before the cap even applies. The 60-bar window cannot reliably distinguish a structural hedge from regime noise anyway; shrinkage can be added later without undoing the floor.

**Placement decision:** both knobs live on the RM (Approach A), not in `analytics`. Flooring and capping are Carver risk-policy decisions, and the RM is the orchestrator that owns RM-specific concerns. The guiding rule: **the floor is an estimation-hygiene concern** (applies only where the RM estimates ρ itself), **the cap is leverage policy** (applies to every auto-assigned IDM regardless of where ρ came from).

## Design

### `corr_floor: Optional[float] = 0.0` (RM constructor + `BacktestConfig`)

- **Validation:** when not `None`, must be in `[-1.0, 1.0]`; otherwise `ValueError`. `None` disables flooring. Mirrored in `BacktestConfig.__post_init__`.
- **Application point:** exactly one — `_derive_corr_matrix`, as `correlation_matrix(returns).clip(lower=self.corr_floor)` before returning. Clipping preserves symmetry and leaves the 1.0 diagonal alone (for any floor ≤ 1), so the existing `analytics` validators pass unchanged.
- **Coherence:** the floored matrix flows to both the optimizer and `diversification_multiplier` — the existing "weights and IDM computed from the same matrix" invariant holds.
- **Not floored:** the explicit `corr_matrix=` research hook (caller owns the matrix), and the equal-weight / data-gap fallback paths (no matrix involved).
- **Mode-agnostic:** applies under both `corr_mode` settings; the floor is a matrix transform, independent of how price changes were computed.
- **Implementation gotcha:** the disable check must be `if self.corr_floor is not None`, never truthiness — the default `0.0` is falsy.

### `idm_cap: Optional[float] = 2.5` (RM constructor + `BacktestConfig`)

- **Validation:** when not `None`, must be `>= 1.0`; otherwise `ValueError`. (DM is mathematically ≥ 1 for sum-to-1 non-negative weights, so a sub-1 cap would always bind — a config error worth failing on.) `None` disables capping. Mirrored in `BacktestConfig.__post_init__`.
- **Constructor coherence check:** the starting `idm` must satisfy `idm <= idm_cap` when the cap is not `None`; violation raises `ValueError` (strict-validation convention — `idm=4.0, idm_cap=2.5` is a contradiction, not something to silently clamp).
- **Application point:** the auto-update site in `calculate_instrument_weight`: `self.idm = min(self.idm_cap, diversification_multiplier(self.instrument_weight, corr_matrix))` when the cap is not `None`. Applies in **both** the inline-derivation path and the explicit `corr_matrix=` path — the cap is leverage policy regardless of where ρ came from.
- **Unaffected paths:** the singleton-live-set path (`idm = 1.0`, trivially under any valid cap), the equal-weight fallback and empty-universe paths (IDM untouched, as today), and direct `self.idm` overwrites by subclasses/downstream code (uncapped — same "owner may overwrite" convention as the weight dicts; documented in the docstring).
- **Interaction with the floor:** with the default `corr_floor=0.0`, the pre-cap IDM is already ≤ √N, so the cap can only bind for universes of N ≥ 7 (√6 ≈ 2.45 < 2.5 < √7 ≈ 2.65).

### Config & wiring

- `BacktestConfig` gains both fields with mirrored validation (matching the existing "fail at config construction, not deep in the wiring" pattern).
- `backtests/test_ewmac.py` passes both through to the RM constructor explicitly, alongside the other corr knobs, for visibility.
- CLAUDE.md riskmanager section and default-stack notes updated to describe both knobs.

### Tests

`tests/test_riskmanager_carver.py`:

1. **Floor active by default:** anti-correlated synthetic closes → derived matrix has no negative entries; `min_variance` weights ≈ equal; IDM ≤ √N.
2. **Floor disabled:** same fixture with `corr_floor=None` → raw negative ρ flows through; the anti-correlated pair is overweighted relative to (1); IDM exceeds the floored-case IDM.
3. **Floor skips the research hook:** explicit `corr_matrix` containing negative entries → weights reflect the raw matrix (no flooring), but the resulting IDM is still capped.
4. **Cap binds:** N ≥ 7 mutually uncorrelated symbols (raw DM ≈ √7 ≈ 2.65) → `idm == 2.5`; with `idm_cap=None` → `idm ≈ 2.65`.
5. **Validation:** `corr_floor=1.5` / `-1.5` raises; `idm_cap=0.5` raises; `idm=3.0, idm_cap=2.5` raises.
6. **Fixture review:** existing tests that assert IDM or corr-based weights get a pass — the new default floor changes behavior wherever a fixture happens to produce negative ρ.

`tests/test_config_backtest.py`: mirrored validation raises for the same invalid values; valid values round-trip.

### Error handling

All failures are `ValueError` with descriptive messages, per project convention. No new exception types. No logging changes — the capped IDM lands in the existing per-bar records' `idm` column automatically.

## Out of scope

- Correlation shrinkage (Ledoit-Wolf-style λ·ρ̂ + (1−λ)·ρ̄) — a future, statistically softer alternative to the hard floor.
- Floor/cap parameters on the `analytics` functions themselves (Approach B) — promote the one-liner to a helper only if a research-side need materializes.
- Per-contract margin, delisting/universe-exit, multi-strategy weights — tracked elsewhere.
