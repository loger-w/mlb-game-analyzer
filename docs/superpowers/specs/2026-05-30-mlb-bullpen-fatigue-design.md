# MLB Bullpen Short-Rest Fatigue — Design

**Date:** 2026-05-30
**Status:** approved (design); pending spec review → implementation plan

## Goal

Test whether **bullpen short-rest fatigue** (relief innings thrown in the trailing 2 days) carries any out-of-sample edge, via **two paths**: (A) a μ-penalty through the (now generalized) ablation harness — expected to REJECT; (B) the real test — a pre-registered "heavy-usage tail" **bet filter**, judged by hit-rate **and** CLV on the model's positive-edge bets, tail vs non-tail. This is feature-program cycle #2 (after RA was ablation-rejected); same minimal-first, prove-OOS, anti-overfit discipline. Honest prior: likely inconclusive (the tail is ~5% of games → small n); the durable wins are a clean test and a **feature-agnostic ablation harness** for the features after this.

## The fatigue signal — zero new fetch, zero re-backfill

`bullpen.relief_ip_last_k(team_id, year, as_of, k=2, cache_dir=DEFAULT_CACHE_DIR, index=None)` sums relief IP over the `k` days **before** `as_of` from the **already-cached relief index** (`analysis-data/backtest/cache/relief_index_{year}.json`, keyed by team_id → `[{date, er, ip}]`, rebuilt point-in-time from boxscores dated before the game → leakage-free). Returns `0.0` if the team/window is empty. `index` may be passed to avoid reloading per call.

Team_id is resolved from the matchup dir name (`AWAY@HOME` abbreviations) via the existing `_team_resolver.resolve_team_id`. Each fit/CLV row is enriched **once** with `home_fat_ip` / `away_fat_ip` so all downstream logic is pure (no per-call I/O). Verified distribution (team-games from 4/15): trailing-2d relief IP mean ~6, p90 ~10, **p95 ~12** → the `≥12` tail is genuinely the extreme ~5%.

## Path A — μ penalty via the ablation harness (expected REJECT)

**Penalty form:** a team's bullpen ERA is bumped by `w_fat × fat_ip`:
`defense(team) = SP_W·starterFIP + BP_W·(bullpen_era + w_fat·fat_ip)`.
`w_fat = 0` reproduces today's model exactly (baseline). One knob, **no baseline-subtraction param** — mean-matched `LEAGUE_RG` re-centers the level, so only the cross-game *differential* in load carries signal. The fatigued team's bumped defense raises the *opponent's* expected runs (same wiring as RA).

**Protocol (identical to RA):** fit `w_fat` on Mar–Apr by minimizing residual σ (odds-free), OOS-test on May per-bet log-loss vs the no-feature baseline, accept only if pooled OOS log-loss beats baseline by >1 SE. Grid `FAT_W_GRID` (e.g. `0.00 .. 0.30` step `0.02`; small because `fat_ip` is ~6–12, so the penalty is `w_fat × ~10`).

**Generalize the ablation harness (the §2 the user approved):** `ablation.py`'s core gains an injectable μ-recompute function (default `recompute_mu_ra`, backward-compatible), and its RA-specific result keys are renamed feature-neutral:
- `fit_params(rows, w, recompute=recompute_mu_ra) -> {"w", "league_rg", "sigma_team"}`
- `select_w(rows, grid, recompute=recompute_mu_ra) -> (w_star, table)`
- `eval_logloss(rows, params, recompute=recompute_mu_ra)` (reads `params["w"]`)
- `ablate(train, test, grid, recompute=recompute_mu_ra) -> {"w_star", "baseline"{w,league_rg,sigma_team,rl_ll,ou_ll,pooled_ll}, "candidate"{…}, "pooled_improve", "pooled_se", "accept", "market_pooled_ll", "gap_baseline", "gap_candidate", "train_table"}`
RA behavior is unchanged (same math); only the param + key names change. `ablation.render_report` and RA tests update `w_ra*`→`w*`, `w_ra`→`w`. **Re-verify after refactor: `python scripts/ablation.py` still yields `w_star=0` (RA REJECT).** `fatigue.recompute_mu_fatigue(row, league_rg, w_fat)` is passed as `recompute`.

## Path B — heavy-usage tail filter (the real test)

- **Tail flag (pre-registered, fixed):** `TAIL_IP = 12.0`; a game is **tail** iff `max(home_fat_ip, away_fat_ip) ≥ TAIL_IP` (≥1 taxed bullpen).
- **Bets:** the model's **positive-edge** bets, defined as in `compute_edge_calibration`: RL → `home_rl_pp > 0` ⇒ bet home-cover; OU → `over_pp > 0` ⇒ bet over (push excluded). Pool RL + OU positive-edge bets.
- **Two outcomes per bet:** (1) **hit** = did that side win (home covered / over hit); (2) **CLV** = signed no-vig pp move of that side from entry→close (`clv.find_entry_close_snapshots` + `clv.clv_pp` in the **edge direction**, headroom subset only).
- **Compare tail vs non-tail:** report n, hit-rate, mean CLV for each subset (pooled; RL/OU split if n permits).
- **Accept only if BOTH** hit-rate and CLV differ clearly and consistently and clear noise (tail bets clearly worse → a suppression rule; clearly better → fatigue side mispriced). Report `n` prominently; **if the tail subset is tiny (likely), report "inconclusive" — do not force a verdict.** This is the anti-overfit guard for a small subset on one season.

## Components / files

- **`scripts/bullpen.py`** — add `relief_ip_last_k`.
- **`scripts/ablation.py`** — generalize core to injectable `recompute` + feature-neutral keys (above); RA unchanged behaviorally.
- **`scripts/fatigue.py`** (new) — `team_ids_from_matchup(matchup) -> (away_id|None, home_id|None)`; `add_fatigue_to_rows(rows, year, k=2, cache_dir=…)` (enrich `home_fat_ip`/`away_fat_ip`, load index once); `recompute_mu_fatigue(row, league_rg, w_fat)`; `FAT_W_GRID`; `TAIL_IP=12.0`; `fatigue_filter_report(rows, snapshots_dir)` (Path B split); `render_report(...)`; `main` CLI — runs both paths, prints verdicts, writes `analysis-data/backtest/ablation-fatigue-2026.md`. **Read-only**: no `--write`, no edits to `config.py`/`run_model.py`.
- **Tests** — `scripts/tests/test_fatigue.py` (signal join, `recompute_mu_fatigue` w=0≡baseline + monotonicity, `add_fatigue_to_rows`, `fatigue_filter_report` split logic on synthetic rows + tmp snapshots); extend `scripts/tests/test_bullpen.py` (`relief_ip_last_k` window/empty); update `scripts/tests/test_ablation.py` for the generalized keys.

## Execution (operational, after code green)

1. Full suite green; **re-confirm RA ablation still REJECTs** (`w_star=0`) post-refactor.
2. `python scripts/fatigue.py` — reads frozen Mar–May rows + cached relief index + snapshots (no network, no model mutation). Prints Path A verdict (expect REJECT) + Path B tail-vs-non-tail hit/CLV; writes the record file.
3. Read verdicts; report honestly (incl. tail n). Do **not** modify the live model. If by some chance both paths show a real, consistent effect, baking it in is a **separate** decision/plan.

## Out of scope

- Starter short-leash co-condition (user chose single-condition tail).
- Re-backfill / freezing fatigue into `features.json` (computed live from the cache instead).
- RL/OU asymmetry of fatigue (a tired pen helps the opponent specifically) — approximated by a game-level tail flag; noted limitation.
- Acting on the result / any live-model change.
