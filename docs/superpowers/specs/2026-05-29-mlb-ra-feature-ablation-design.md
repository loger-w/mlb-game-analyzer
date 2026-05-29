# MLB RA-Defense Feature Ablation — Design

**Date:** 2026-05-29
**Status:** approved (design); pending spec review → implementation plan

## Goal

Test, by an honest time-split, whether adding a **team run-allowed (RA) defense signal** to the team-level model improves its out-of-sample ability to forecast the RL / O-U bet outcomes — measured *relative to the market*. Build the ablation as a reusable, model-read-only harness so later candidate features plug in the same way. This is the first cycle of the "minimal-first, prove-by-ablation" feature program; it directly follows the 2-knob calibration ([[mlb-config-calibration]]) which left the conclusion: the current feature set (RS + starter FIP + bullpen ERA + PF) is ~market-parity, so edge must come from features.

## Background: why RA, why now

- The model's opponent run-prevention term is **pitcher-only**: `defense = SP_W·starterFIP + BP_W·bullpenERA` (rate stats). Team RA (runs allowed per game) is the holistic *result* of run-prevention (full staff + fielding + sequencing) and is **already frozen** in every `features.json` (`home_ra_recent/season`, `away_ra_recent/season`) but **unused** by the model.
- RA needs **zero new fetching** → cheapest possible first ablation, and it validates the harness end-to-end before we invest in features that require new point-in-time data (offense/lineup, starter recent-form, bullpen fatigue).

## How the feature enters the model — one new knob `w_ra`

The opponent's defense term gains a blend with that opponent's RA:

```
ra_blend(team)   = RECENT_W·ra_recent + (1 − RECENT_W)·ra_season        # reuse held prior RECENT_W
defense(team)    = (1 − w_ra)·pitch_today(team) + w_ra·ra_blend(team)
```

where `pitch_today(team) = SP_W·starter_fip + BP_W·bullpen_era` (unchanged). Then μ is computed as today:

```
mu_home = home_rs_blend · defense(away) / LEAGUE_RG · PF/100
mu_away = away_rs_blend · defense(home) / LEAGUE_RG · PF/100
```

(The away team's defense suppresses the home offense, and vice-versa — same wiring as today, only `defense()` changes.)

- **`w_ra = 0` reproduces today's model exactly** → it is the ablation baseline.
- `w_ra ∈ [0, 1]`. Both blended terms are on a runs/game scale → dimensionally clean.
- The feature adds **exactly one** parameter (`w_ra`); the RA recency blend reuses the existing `RECENT_W`.

## Data split

- **Train = March + April** games with `result.json` (≈468). Odds not required for training.
- **Test = May** games that have **both** odds and result (the 292 RL / 281 non-push O-U games).
- **Strict temporal separation:** May games are never in train — not even May games that lack odds — so the fitted `w_ra` cannot see the test period. No leakage.

## Training objective (odds-free) → choosing `w_ra`

April has almost no odds, so the training objective is **score-prediction quality**, not bet log-loss:

For each candidate `w_ra` on a grid `{0.00, 0.05, 0.10, …, 0.80}`:
1. Recompute μ for all train games with that `w_ra`.
2. Refit `LEAGUE_RG` on train by mean-matching total (bisection; μ_total still ∝ 1/`LEAGUE_RG`, so monotone — the existing `fit_league_rg` works).
3. Refit `SIGMA_TEAM` on train = residual MLE `sqrt((mean r_total² + mean r_margin²)/4)` (existing `fit_sigma_team`).
4. **Training score `= SIGMA_TEAM(w_ra)`** — lower means μ tracks actual scores better (this equals maximizing the Gaussian score log-likelihood, since with σ at its MLE the maximized likelihood is monotone-decreasing in σ).

Pick `w_ra* = argmin_w SIGMA_TEAM_train(w)`. If `w_ra* == 0`, RA does not help even in-sample → **reject immediately** (no OOS step needed). Otherwise freeze `params* = {w_ra*, LEAGUE_RG(w_ra*), SIGMA_TEAM(w_ra*)}` from train.

Rationale: fitting `w_ra` to better predict actual runs (not to fit bet outcomes) is the anti-overfit choice, and it's what the data allows given odds sparsity on train. Better μ → better probabilities is the model's premise; the time-split test checks whether that premise pays off OOS.

## Test (market-relative) → accept rule

Evaluate **two** frozen models on the held-out May games: **baseline** `params0 = fit on train at w_ra=0`, and **candidate** `params*`. For each model compute per-bet log-loss:

- **RL** (all 292): `p = cover_prob_home(mu_margin, rl_home_point, σ)`, outcome `y = 1[actual_margin > −rl_home_point]`, `ll = −[y·ln p + (1−y)·ln(1−p)]` (p clamped to [1e-9, 1−1e-9]).
- **O-U** (281, push excluded): `p = over_prob(mu_total, total_line, σ)`, `y = 1[actual_total > total_line]`.
- **Market log-loss** (reference): same formula with `p = rl_home_no_vig` / `over_no_vig`.

Headline metric = **pooled bet log-loss** over (RL bets ∪ O-U bets). Report RL and O-U separately too.

**Per-feature accept gate:** candidate's pooled OOS log-loss < baseline's pooled OOS log-loss, and the improvement exceeds **1 standard error** of the paired per-bet difference (`SE = std(ll_base − ll_cand)/√n_bets`; RL and O-U bets pooled, same bets for both models). Pooling correlated RL/O-U bets in the SE is a mild approximation — noted, acceptable for a go/no-go gate.

**North-star (always reported, not a gate):** the **gap to market** = `model_pooled_log_loss − market_pooled_log_loss`, for both baseline and candidate. Accept narrows the gap; the program's goal is gap ≤ 0 (model out-forecasts Pinnacle). Splitting "improve vs baseline" (gate) from "beat market" (goal) avoids rejecting every incremental feature for failing to single-handedly beat Pinnacle.

## Components / files

- **Create `scripts/ablation.py`** — pure fns + CLI, model-read-only:
  - `recompute_mu_ra(row, league_rg, w_ra) -> (mu_total, mu_margin)` — generalizes `fit_config.recompute_mu` with the RA blend (`w_ra=0` ≡ `recompute_mu`).
  - `fit_params(train_rows, w_ra) -> {"w_ra","league_rg","sigma_team"}` — mean-match L then residual-MLE σ at fixed `w_ra`. To stay DRY, `fit_config.fit_league_rg`/`fit_sigma_team` gain an **optional `mu_fn` param defaulting to `recompute_mu`** (backward-compatible — existing callers unaffected); `ablation.fit_params` passes `mu_fn = lambda r, L: recompute_mu_ra(r, L, w_ra)`.
  - `select_w_ra(train_rows, grid) -> (w_ra*, train_table)` — argmin σ over grid.
  - `eval_logloss(test_rows, params) -> {"rl":[...],"ou":[...],"market_rl":[...],"market_ou":[...]}` — per-bet log-loss arrays (for means + paired SE).
  - `ablate_ra(train_rows, test_rows, grid) -> dict` — orchestrates baseline vs candidate, computes pooled means, gap-to-market, paired SE, verdict.
  - CLI `main`: loads rows, prints the baseline-vs-candidate table + verdict, and writes a record to `analysis-data/backtest/ablation-ra-2026.md`. **No `--write`; never edits `config.py`/`run_model.py`.**
- **Modify `scripts/fit_config.py`** — extend `load_fit_rows` row dict with `home_ra_recent`, `home_ra_season`, `away_ra_recent`, `away_ra_season` (read from `inputs`; already present in v2 `features.json`). Existing consumers ignore the new keys (backward-compatible).
- **Create `scripts/tests/test_ablation.py`** — TDD: synthetic rows where RA carries known signal → `select_w_ra` recovers `w_ra*>0`; where RA is pure noise → `w_ra*≈0`; `recompute_mu_ra(…, w_ra=0)` equals `fit_config.recompute_mu`; `eval_logloss` math on a hand-checkable row; `ablate_ra` verdict logic (accept when candidate clearly better, reject when within noise).

## Execution (operational, after code is green)

1. Full test suite green.
2. `python scripts/ablation.py` → reads existing frozen Mar–May data (already on disk from prior backfills), prints + writes the ablation report. No new fetching, no model mutation.
3. Read the verdict: fitted `w_ra*`, baseline-vs-candidate OOS log-loss (RL / O-U / pooled), gap-to-market for each, paired SE, accept/reject.
4. Report to user. **If RA passes**, baking `w_ra` into the live model (`run_model.py` + `config.py`) is a **separate** decision/plan. If it fails, zero production change.

## Out of scope

- Other candidate features (offense/lineup OPS·wOBA, starter recent-form, bullpen fatigue) — later cycles, same harness.
- No general feature-registry/plugin framework (YAGNI) — RA-specific with clean seams the next feature can copy.
- No change to the live model in this spec; no `--write`.
- Doubleheader dirs still overwrite (pre-existing, negligible).
