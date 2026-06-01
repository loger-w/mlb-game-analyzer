# MLB Edge Threshold Sweep — Design

**Date:** 2026-06-01
**Status:** approved (design); pending spec review → implementation plan

## Goal

Add an **edge threshold sweep** to the v2 backtest report: for a set of edge thresholds (pp), simulate "only bet when `|edge| ≥ t`, on the side the model favors" and report, per market (RL, O/U) and per threshold, the **hit rate** (outcome) alongside the **CLV** (leading indicator). This answers the user's question directly: *does raising the bet threshold (only taking high-conviction bets) improve results, or is the edge magnitude uninformative?*

**Why:** the existing report has two coverage points — full-sample direction hit (no threshold) and the positive-edge calibration subset (`edge > 0`, one side only). Neither shows how results behave as the bet threshold tightens. The backtest already measured `corr(|edge|, CLV) ≈ 0`, which *predicts* a flat sweep; the sweep makes that prediction explicit and falsifiable across thresholds rather than at a single cut.

## Bet semantics (approved this session)

- **Two-sided.** Per market the model's pick side is `sign(edge)`: RL → `home` if `home_rl_pp > 0`, `away` if `< 0`; O/U → `over` if `over_pp > 0`, `under` if `< 0`. `edge == 0`/missing/nan ⇒ no pick (game excluded). This matches `clv.pick_sides` exactly — the sweep must not invent a second pick rule.
- **Threshold filter** = `|edge_pp| ≥ t`. `t = 0` therefore means "all non-zero-edge bets" (the natural base row), **not** "all games" — games with no pick never enter.
- A bet **hits** when the picked side covers: RL `home` pick hits iff `actual_margin > −rl_home_point`; `away` pick hits iff not. O/U `over` pick hits iff `actual_total > total_line`; `under` iff not. **Push** (`actual_total == total_line`) excluded from O/U, as in `compute_ou_metrics`.

## Established constraints (verified this session)

- **Two different denominators, never merged.** Hit rate is computed on the `_valid` subset (odds + result present, ~292). CLV is computed only on the **headroom subset** from `compute_clv_rows` (entry strictly before close, pick-side no-vig present, ~270). The same threshold row therefore shows a **bet count** and a separate **CLV count**. This mirrors the existing report (edge calibration n=153 vs CLV n=270).
- **Pick direction is reused, not reinvented.** Hit-rate side = `sign(edge_pp)`; CLV side already = `pick_sides` (same rule). Both must agree by construction.
- **`edge == 0` carries no pick.** Reuse the `clv._finite_nonzero` semantics (finite & non-zero) so a 0/nan edge never counts at any threshold, including `t = 0`.
- CLV inherits all existing caveats (soft "close" proxy, no intra-day odds pre-2026-04-28, point-move approximation) — the sweep adds no new CLV machinery, it only re-aggregates `compute_clv_rows` output on threshold subsets.

## Thresholds

Default `THRESHOLDS = [0, 1, 2, 3, 4]` (pp), defined in `backtest.py`. Integer steps are enough to see the trend; the two-sided rule keeps the `t = 0` base near full-sample so higher thresholds retain usable n.

## Outputs

Per market (RL, O/U), one row per threshold:

- `n_bets` — bets passing the filter on the `_valid` subset.
- `hit_rate` — fraction of those that hit (`None` if `n_bets == 0`).
- `n_clv` — bets passing the filter on the headroom subset.
- `clv_mean` — mean signed CLV_pp on that subset (`None` if `n_clv == 0`).
- `share_pos` — fraction with CLV_pp > 0 (line moved our way).

Report section (inserted between "edge 校準" and "CLV"):

```
## edge 門檻掃描(雙向:|edge|≥門檻,下 model pick 側)

RL:
| 門檻  | 注數 | 命中率 | CLV注數 | CLV mean | 往我方 |
|-------|------|--------|---------|----------|--------|
| ≥0pp  | …    | …      | …       | …        | …      |
| ≥1pp  | …    | …      | …       | …        | …      |
| ≥2pp  | …    | …      | …       | …        | …      |
| ≥3pp  | …    | …      | …       | …        | …      |
| ≥4pp  | …    | …      | …       | …        | …      |

O/U:(同欄位)

> 判讀:命中率與 CLV 要「同向往上」才算門檻撈出 edge;
> 命中率升但 CLV≈0/負 = 高門檻只是小樣本雜訊,非真 alpha。
```

Reading: a real edge would show hit rate **and** CLV both rising with the threshold. Hit rate rising while CLV stays ≈0/negative = small-sample noise at the tail, not alpha — consistent with the measured `corr(|edge|,CLV) ≈ 0`.

## Components / files

- **`scripts/lib/metrics.py`** — ADD `compute_threshold_sweep(df, thresholds) -> dict`. Reuse `_valid`. For each market build the candidate frame (edge finite & non-zero, plus the market's existing notna guards and O/U push exclusion), derive pick via `sign(edge)`, hit via the cover rule above; loop thresholds filtering `abs(edge) >= t`. Returns `{"thresholds": [...], "rl": [{"t", "n_bets", "hit_rate"}, ...], "ou": [...]}`. Existing `compute_rl_metrics`/`compute_ou_metrics`/`compute_edge_calibration` untouched.
- **`scripts/lib/clv.py`** — ADD `aggregate_clv_by_threshold(rows, thresholds) -> dict`. `rows` = `compute_clv_rows(...)` output. For each market keep `has_headroom` rows with `<market>_clv` not None and `<market>_edge_pp` finite & non-zero (reuse `_finite_nonzero`); per threshold filter `abs(edge_pp) >= t` and reuse the `_stats` shape (`n`, `mean`, `share_pos`). Returns `{"rl": [{"t","n","mean","share_pos"}, ...], "ou": [...]}`. Existing `aggregate_clv` untouched.
- **`scripts/backtest.py`** — `cmd_run`: compute `clv_rows = compute_clv_rows(df.to_dict("records"), SNAPSHOTS_DIR)` **once**, reuse for both `aggregate_clv(clv_rows)` and `aggregate_clv_by_threshold(clv_rows, THRESHOLDS)` (avoids scanning snapshot files twice). Compute `sweep = compute_threshold_sweep(df, THRESHOLDS)`. Pass both to render. Define `THRESHOLDS` here.
- **`scripts/lib/render.py`** — ADD `render_threshold_section(sweep, sweep_clv) -> str` joining hit-rate (from `sweep`) and CLV (from `sweep_clv`) into the two tables; call it from `render_report` between the edge-calibration block and the CLV block. New optional params default to `None` (section omitted when absent), keeping existing callers/tests green.
- **Tests** —
  - `scripts/tests/test_backtest_metrics_v2.py`: `compute_threshold_sweep` on a synthetic df — two-sided pick correctness (a negative-edge `away` pick that hits), threshold filtering monotonically drops n, push excluded from O/U, `n_bets == 0 → hit_rate None`.
  - `scripts/tests/test_clv.py`: `aggregate_clv_by_threshold` on synthetic clv rows — threshold filter on `abs(edge_pp)`, None-CLV/no-headroom rows excluded, mean & share_pos correct, zero-edge never counts at `t = 0`.
  - `scripts/tests/test_backtest_e2e.py` (exists): assert the new section header appears in the rendered report.

## Error handling

`n_bets`/`n_clv` of 0 → `hit_rate`/`clv_mean`/`share_pos` rendered as "—" (reuse `render._pct`/`_f` None handling). No `--write`, no model/config mutation, no new fetching — reads the same df and snapshot files the backtest already loads.

## Out of scope

- ROI / staking returns (would need an odds/price assumption); hit rate + CLV suffice to judge alpha.
- Acting on the sweep (bet selection); this is measurement only.
- Any change to the model, config, σ/weights, or live prediction flow.
- Per-edge-bucket (binned) view; this is cumulative `≥ t` by design. Revisit only if the cumulative view proves ambiguous.
