# MLB CLV Measuring Stick — Design

**Date:** 2026-05-30
**Status:** approved (design); pending spec review → implementation plan

## Goal

Build a **read-only** closing-line-value (CLV) measuring stick: for each backfilled game where the model had a pick (its positive-edge side), did Pinnacle's no-vig line for **that side** move toward us between the earliest pre-commence snapshot (entry) and the latest pre-commence snapshot (close)?

**Why this is the #1 priority** (from the edge-roadmap workflow): the model's RL/O-U bet outcomes are ~coin-flip (log-loss ≈ ln2), so 292 noisy outcomes can't tell us whether a future feature carries real signal. CLV is a *leading* indicator — if the model's edge is real, its positive-edge picks should see the line drift their way *before* the game is played. A trustworthy CLV stick is the prerequisite for honestly judging every future feature. This spec builds **only the measuring stick** (A); forward snapshot capture (B) is a separate follow-on.

## Established constraints (verified this session / by the workflow)

- `closing_line.py` hard-codes the `12-00-ET` snapshot. `odds_compare.find_latest_snapshot_for_game` already picks the *latest* pre-commence snapshot — which is exactly why backfilled `features.json` entries have ~no CLV headroom (the entry already *is* the latest). So CLV must be computed from the **snapshot files directly**, independent of which snapshot `features.json` froze.
- Snapshot files `odds/odds_snapshots/{date}_{HH-MM}-ET.json`: top-level `snapshot_time_utc`, `snapshot_time_et`; `games[]` each with `home_team`, `away_team` (full names, matching `features.json` `game.home`/`game.away`), `commence_utc`, `bookmakers.pinnacle.{ml,ou,rl}`.
- **No intra-day odds before 2026-04-28** → CLV is measurable only on ~late-April + May games that have ≥2 distinct pre-commence snapshots. Report this honestly.
- The archived "close" is a soft proxy (median ~71 min pre-commence). CLV measured here is therefore a **lower-bound-ish, flattered** read; surface the entry→close time gap so its realness is visible.

## CLV definition

- **Entry** = earliest snapshot for the game's ET date with `snap_ts < commence` and matching teams. **Close** = latest such snapshot. Strict `snap_ts < commence` excludes the 22:00-ET last-of-day snapshot for night games (the day-rollover/post-commence trap that produced a spurious result during the workflow).
- **Pick side** per market from frozen edges: RL → `home` if `home_rl_pp > 0`, `away` if `< 0`, `None` if `0`/missing; O-U → `over` if `over_pp > 0`, `under` if `< 0`, `None` if `0`/missing.
- **CLV_pp** = `(no_vig[pick] at close − no_vig[pick] at entry) × 100`. Positive ⇒ market moved **toward** our pick. Defined only on the **headroom subset** (`entry_ts` strictly `< close_ts` and the pick-side no-vig present at both).
- **Honest approximation (baked in, documented):** when the **point** moves (totals can; RL is ~always fixed ±1.5), comparing no-vig prob at different lines is approximate. Report Δpoint and the **% of games where the point was stable** so the approximation is transparent. Rigorous same-bet CLV needs alternate-line capture = B-territory (out of scope).

## Aggregation / outputs

On the headroom subset, per market (RL, O-U):
- `n`, mean & median signed CLV_pp, **share > 0** (fraction where line moved our way).
- `corr(|edge_pp|, CLV_pp)` — does bigger model conviction predict a bigger favorable line move? (Pearson; `None` if n < 3.)
- Stratified by **entry hour (ET)**: `{hour: (n, mean_clv)}` — exposes timing effects.
- **Diagnostics:** `n_headroom / n_total`, median minutes(entry→close), % point-stable.

Reading: mean CLV ≳ 0 beyond noise ⇒ the model's edge signal has predictive CLV (encouraging); mean CLV ≈ 0 ⇒ no CLV, consistent with the established no-alpha finding. The stick does not assert alpha; it *measures honestly*.

## Components / files

- **`scripts/lib/closing_line.py`** — ADD `find_entry_close_snapshots(snapshots_dir, date, home_team, away_team) -> tuple[dict|None, dict|None]`: scan `{date}_*-ET.json`, collect games matching teams with `snap_ts < commence` (reuse `_parse_iso_utc`), attach `snapshot_time_utc`/`snapshot_time_et` to each game dict, return `(earliest_game, latest_game)` by `snap_ts` (both `None` if none qualify; the two may be the same dict when only one qualifies). Existing `find_closing_snapshot_for_game` untouched (its tests must stay green).
- **`scripts/lib/clv.py`** (new) — pure core + one I/O-driving fn:
  - `pick_sides(row) -> {"rl": "home"|"away"|None, "ou": "over"|"under"|None}` from `home_rl_pp`/`over_pp`.
  - `no_vig_for(game, market, side) -> float | None` (uses `closing_line.extract_pinnacle_rl_no_vig` / `extract_pinnacle_no_vig`).
  - `point_for(game, market) -> float | None` (RL `home_point`; O-U `total_line`).
  - `clv_pp(entry_game, close_game, market, side) -> float | None` = `round((no_vig_close − no_vig_entry) * 100, 2)`; `None` if either side missing.
  - `compute_clv_rows(df, snapshots_dir) -> list[dict]`: per df row uses `home_team`/`away_team`/`date`, finds entry/close, returns `{date, matchup, has_headroom, entry_hour, minutes_gap, rl_pick, rl_clv, rl_edge_pp, rl_point_stable, ou_pick, ou_clv, ou_edge_pp, ou_point_stable}`. `has_headroom = entry and close and entry_ts < close_ts`.
  - `aggregate_clv(rows) -> dict`: the outputs above (per-market means/median/share>0/corr, entry-hour table, diagnostics).
- **`scripts/lib/load.py`** — add `home_team`, `away_team` (from `features.json` `game.home`/`game.away`) to the row dict (needed for snapshot matching). Backward-compatible (existing consumers ignore new keys).
- **`scripts/backtest.py`** — after `rl/ou/edge`, compute `clv = aggregate_clv(compute_clv_rows(df, SNAPSHOTS_DIR))`; pass to render. `SNAPSHOTS_DIR = SKILL_ROOT/"odds"/"odds_snapshots"`.
- **`scripts/lib/render.py`** — add a "CLV(領先指標)" section with the per-market numbers, entry-hour table, diagnostics, and the honest caveats (soft close, point-move approximation, no pre-4/28 data).
- **Tests** — `scripts/tests/test_clv.py` (pure fns on synthetic snapshot/row dicts: pick_sides; no_vig_for; clv_pp sign & magnitude; compute_clv_rows headroom logic incl. equal-ts → no headroom and post-commence exclusion; aggregate_clv means/share/corr/point-stable). Extend `scripts/tests/test_closing_line.py` for `find_entry_close_snapshots` (earliest/latest pick, post-commence exclusion, no-match → (None,None), single-snapshot → equal entry/close).

## Error handling

Missing snapshot file / no team match / unparseable commence → that game contributes no CLV (excluded from aggregates, counted in `n_total` but not `n_headroom`). Missing pick-side no-vig at entry or close → that market's CLV is `None` for the game. No `--write`, no model mutation, no new fetching (reads existing snapshot files only).

## Out of scope

- Forward snapshot capture / earlier entry-freeze (subsystem B).
- Alternate-line / same-bet rigorous CLV (needs capturing alt lines).
- Acting on CLV (bet selection); this is measurement only.
- Any change to the model, config, or live prediction flow.
