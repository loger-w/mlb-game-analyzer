# MLB `config.py` 2-Knob Calibration — Design

**Date:** 2026-05-29
**Status:** approved (design); pending spec review → implementation plan

## Goal

Re-fit the team-level model's coefficients to data so its **probabilities are calibrated** (match observed outcomes), using the now-operational point-in-time backtest harness. Fit exactly **two** coefficients; hold the rest at current priors. Objective = probability calibration (Gaussian likelihood / log-loss), NOT directional hit-rate (base-rate-dominated) and NOT direct edge-maximization (overfits n≈300).

## What we fit vs hold

| Coefficient | Action | Rationale |
|-------------|--------|-----------|
| `LEAGUE_RG` (4.4) | **FIT** — μ-level divisor | Observed mean μ_total≈8.1 ran high vs under-heavy outcomes |
| `SIGMA_TEAM` (3.0) | **FIT** — per-team scoring SD; `SIGMA=SIGMA_TEAM·√2` | Controls probability confidence → directly affects whether edge is signal or noise |
| `RECENT_W`, `SP_W`, `BP_W`, `FIP_CONSTANT`, `RECENT_N` | **HOLD** at priors | Anti-overfit: n is small; only fit what we have clear miscalibration evidence for |

## Data

- **Backfill** 2026-03-26 → 2026-04-30 via `backfill.py` (relief index already covers the season → predictions only, no rebuild), then `fetch_results.py --month 2026-03` and `--month 2026-04`.
- μ/σ fitting needs **actual scores only, not odds** → uses the full April+May sample with results (~800–900 games).
- Odds-dependent validation (log-loss, reliability, edge hit) stays on May's ~292 games.

## Method — re-fit from frozen inputs (zero refetch)

Each `features.json` froze the *inputs* (`home_rs_recent/season`, `away_rs_recent/season`, `home_starter.fip`, `away_starter.fip`, `home_bullpen_era`, `away_bullpen_era`, `park_factor`). The fitter recomputes μ and probabilities for candidate knob values purely in-memory — no API calls.

Per-game model inputs are reconstructed as:
- `home_rs = RECENT_W·rs_recent + (1−RECENT_W)·rs_season` (RECENT_W held at prior), same for away.
- `home_starter_fip = frozen fip` if non-null, else fallback = candidate `LEAGUE_RG` (mirrors `assemble_inputs`); same for away. (Few null-FIP games.)
- `bullpen_era`, `park_factor` = frozen.

**Stage 1 — fit `LEAGUE_RG`** (uses all games with results): μ_total is (near-)proportional to 1/`LEAGUE_RG`. Solve numerically (bisection over `LEAGUE_RG ∈ [2.0, 8.0]`, monotone) for the value where **mean(predicted μ_total) = mean(actual total)**. Bisection (not closed-form) so the rare null-FIP games — whose μ is not exactly ∝1/L because the fallback also scales — are handled exactly. No external deps.

**Stage 2 — fit `SIGMA_TEAM`** (uses all games with results; uses the Stage-1 `LEAGUE_RG`): the model treats each team's score as `N(μ_i, SIGMA_TEAM²)` independent, so both margin and total have variance `2·SIGMA_TEAM²`. Estimate from residuals `r = actual − μ` (RMSE around 0, which conservatively folds any small μ bias into spread):

```
SIGMA_TEAM = sqrt( ( mean(r_total²) + mean(r_margin²) ) / 4 )
```

This is the Gaussian MLE for σ and uses the continuous scores (more efficient + less overfit than tuning to over/under hit-rate).

## Validation (May, odds games)

Report **before vs after** (priors vs fitted knobs):
- RL log-loss and O/U log-loss (lower = better calibrated).
- Reliability: bucket predicted probability (e.g. deciles), compare to actual frequency.
- Edge-calibration hit rates (the RL 46.9% / O/U 52.3% baseline) — does fitting move them toward/above 50%?
- **Stability check:** fit knobs on **April-only**, evaluate on **May**; confirm the fitted values are close to the all-data fit (not sample-specific). This is the lightweight overfit guard (no full CV).

## Components / files

- **Modify `scripts/run_model.py`** — add optional, backward-compatible params so the fitter reuses the model math (one source of truth, no duplication):
  - `expected_runs(..., league_rg: float | None = None)` → uses arg or `config.LEAGUE_RG`.
  - `cover_prob_home(..., sigma: float | None = None)` and `over_prob(..., sigma: float | None = None)` → uses arg or `config.SIGMA`.
- **Create `scripts/fit_config.py`** — pure functions + CLI:
  - `load_fit_rows(months) -> list[dict]` (read frozen inputs + result per backfilled game).
  - `recompute_mu(row, league_rg) -> (mu_total, mu_margin)` (reuses `run_model`).
  - `fit_league_rg(rows) -> float` (Stage 1 bisection).
  - `fit_sigma_team(rows, league_rg) -> float` (Stage 2 RMSE).
  - `eval_calibration(rows_with_odds, league_rg, sigma_team) -> dict` (log-loss/reliability/edge; needs odds).
  - CLI prints **proposed values + before/after table**; `--write` applies to `config.py` only when invoked.
- **Create `scripts/tests/test_fit_config.py`** — TDD on the pure fns with synthetic games (known μ/σ → recover the knobs).

## Out of scope

- Other coefficients (held as priors); no full cross-validation / multi-param grid search.
- Doubleheader dirs still overwrite (2 May cases) — pre-existing limitation.
- `config.py` is changed only when the user approves the proposed numbers (via `--write` or manual edit).
