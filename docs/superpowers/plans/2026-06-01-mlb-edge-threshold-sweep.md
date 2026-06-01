# MLB Edge Threshold Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an edge-threshold sweep to the v2 backtest report — per market (RL, O/U) and per `|edge| ≥ t` threshold, show bet count + hit rate (outcome) alongside CLV count + mean + share-toward-us (leading indicator).

**Architecture:** Three pure functions sliced along existing layers — `metrics.compute_threshold_sweep` (hit rate on the `_valid` subset), `clv.aggregate_clv_by_threshold` (CLV on the headroom subset), `render.render_threshold_section` (joins both into two markdown tables). `backtest.py` computes `compute_clv_rows` once and feeds both CLV aggregators. Bet side is `sign(edge)` (two-sided), identical to `clv.pick_sides`; `edge == 0`/nan never bets.

**Tech Stack:** Python 3.13, pandas, pytest. Tests live in `scripts/tests/` and self-insert `scripts/` on `sys.path`, importing `from lib import <module>`.

**Spec:** `docs/superpowers/specs/2026-06-01-mlb-edge-threshold-sweep-design.md`

**How to run tests (Windows PowerShell, from repo root):**
`python -m pytest scripts/tests/<file>::<test> -v`

---

### Task 1: `metrics.compute_threshold_sweep` — hit rate per threshold

**Files:**
- Modify: `scripts/lib/metrics.py` (add function at end; existing functions untouched)
- Test: `scripts/tests/test_backtest_metrics_v2.py` (add helper + 2 tests)

- [ ] **Step 1: Write the failing tests**

Add to the END of `scripts/tests/test_backtest_metrics_v2.py`:

```python
def _sweep_df():
    # row0: home edge +8, margin 3 (>1.5 → home covers) → pick home, HIT
    # row1: home edge -5 (pick away), margin 0 (home no-cover → away covers) → HIT; ou edge -4 (under), total 7<8.5 under → HIT
    # row2: home edge +2 (pick home), margin 1 (<1.5 no-cover) → MISS; ou edge +2 (over), total 10>8.5 over → HIT
    # row3: home edge 0 (no pick); ou edge +3 but PUSH (total==line) → excluded from O/U
    return pd.DataFrame([
        {"home_rl_pp": 8.0, "rl_home_point": -1.5, "actual_margin": 3,
         "over_pp": 6.0, "total_line": 8.5, "actual_total": 7,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": -5.0, "rl_home_point": -1.5, "actual_margin": 0,
         "over_pp": -4.0, "total_line": 8.5, "actual_total": 7,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": 2.0, "rl_home_point": -1.5, "actual_margin": 1,
         "over_pp": 2.0, "total_line": 8.5, "actual_total": 10,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": 0.0, "rl_home_point": -1.5, "actual_margin": 5,
         "over_pp": 3.0, "total_line": 8.5, "actual_total": 8.5,
         "result_missing": False, "odds_missing": False},
    ])


def test_threshold_sweep_two_sided_and_filtering():
    out = metrics.compute_threshold_sweep(_sweep_df(), [0, 2, 3])
    assert out["thresholds"] == [0, 2, 3]
    rl = {r["t"]: r for r in out["rl"]}
    # row3 home edge 0 → no pick → excluded everywhere. Candidates: 8, -5, 2
    assert rl[0]["n_bets"] == 3 and round(rl[0]["hit_rate"], 3) == round(2 / 3, 3)
    # t=3 keeps |edge|>=3 → 8 (home pick HIT), -5 (away pick HIT) → 2/2; proves two-sided away pick counts
    assert rl[3]["n_bets"] == 2 and rl[3]["hit_rate"] == 1.0
    ou = {r["t"]: r for r in out["ou"]}
    # push row excluded → 3 O/U bets (6, -4, 2)
    assert ou[0]["n_bets"] == 3
    # t=3 → |over_pp|>=3 → 6 (over MISS), -4 (under HIT) → 1/2
    assert ou[3]["n_bets"] == 2 and ou[3]["hit_rate"] == 0.5


def test_threshold_sweep_empty_bucket_hit_none():
    out = metrics.compute_threshold_sweep(_sweep_df(), [99])
    assert out["rl"][0]["n_bets"] == 0 and out["rl"][0]["hit_rate"] is None
    assert out["ou"][0]["n_bets"] == 0 and out["ou"][0]["hit_rate"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_backtest_metrics_v2.py -v`
Expected: the two new tests FAIL with `AttributeError: module 'lib.metrics' has no attribute 'compute_threshold_sweep'`. Existing 3 tests still PASS.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/lib/metrics.py`:

```python
def compute_threshold_sweep(df: pd.DataFrame, thresholds) -> dict:
    """雙向下注:|edge|≥t 下 model pick 側(sign(edge)),逐門檻命中率。
    RL 用 home_rl_pp(pick home/away)、O/U 用 over_pp(pick over/under,排 push)。"""
    v = _valid(df)

    rl = v[v["home_rl_pp"].notna() & (v["home_rl_pp"] != 0)
           & v["actual_margin"].notna() & v["rl_home_point"].notna()].copy()
    rl_home_cover = rl["actual_margin"] > (-rl["rl_home_point"])
    rl_hit = (rl["home_rl_pp"] > 0) == rl_home_cover   # pick home & cover, or pick away & no-cover

    ou = v[v["over_pp"].notna() & (v["over_pp"] != 0)
           & v["actual_total"].notna() & v["total_line"].notna()].copy()
    ou = ou[ou["actual_total"] != ou["total_line"]]    # exclude push
    ou_over = ou["actual_total"] > ou["total_line"]
    ou_hit = (ou["over_pp"] > 0) == ou_over

    def _sweep(frame, edge_col, hit):
        rows = []
        for t in thresholds:
            mask = frame[edge_col].abs() >= t
            n = int(mask.sum())
            rows.append({"t": t, "n_bets": n,
                         "hit_rate": float(hit[mask].mean()) if n else None})
        return rows

    return {"thresholds": list(thresholds),
            "rl": _sweep(rl, "home_rl_pp", rl_hit),
            "ou": _sweep(ou, "over_pp", ou_hit)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_backtest_metrics_v2.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/metrics.py scripts/tests/test_backtest_metrics_v2.py
git commit -m "feat(backtest): threshold sweep hit-rate (two-sided |edge|>=t)"
```

---

### Task 2: `clv.aggregate_clv_by_threshold` — CLV per threshold

**Files:**
- Modify: `scripts/lib/clv.py` (add function at end; reuse `_finite_nonzero`, `_stats`)
- Test: `scripts/tests/test_clv.py` (add 1 test)

- [ ] **Step 1: Write the failing test**

Add to the END of `scripts/tests/test_clv.py`:

```python
def test_aggregate_clv_by_threshold():
    rows = [
        {"has_headroom": True, "rl_clv": 2.0, "rl_edge_pp": 1.0, "ou_clv": 5.0, "ou_edge_pp": 4.0},
        {"has_headroom": True, "rl_clv": -1.0, "rl_edge_pp": 3.0, "ou_clv": 1.0, "ou_edge_pp": 2.0},
        {"has_headroom": True, "rl_clv": 0.5, "rl_edge_pp": -3.0, "ou_clv": -2.0, "ou_edge_pp": -1.0},
        {"has_headroom": True, "rl_clv": 9.0, "rl_edge_pp": 0.0, "ou_clv": 9.0, "ou_edge_pp": 0.0},   # edge 0 → never counts
        {"has_headroom": False, "rl_clv": None, "rl_edge_pp": None, "ou_clv": None, "ou_edge_pp": None},
    ]
    out = clv.aggregate_clv_by_threshold(rows, [0, 2, 3])
    rl = {r["t"]: r for r in out["rl"]}
    # candidates (headroom, clv not None, edge finite & nonzero): edges 1, 3, -3 → 3 rows (edge 0 + no-headroom excluded)
    assert rl[0]["n"] == 3
    assert abs(rl[0]["mean"] - (2.0 - 1.0 + 0.5) / 3) < 1e-3
    assert abs(rl[0]["share_pos"] - 2 / 3) < 1e-3      # 2.0>0, -1.0 no, 0.5>0
    # t=2 → |edge|>=2 → edges 3, -3 → clv -1.0, 0.5 → n=2
    assert rl[2]["n"] == 2
    assert rl[3]["n"] == 2
    ou = {r["t"]: r for r in out["ou"]}
    # ou candidates: edges 4, 2, -1 (edge 0 + no-headroom excluded) → 3
    assert ou[0]["n"] == 3
    # t=2 → |edge|>=2 → 4, 2 → clv 5.0, 1.0 → n=2
    assert ou[2]["n"] == 2 and abs(ou[2]["mean"] - 3.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_clv.py::test_aggregate_clv_by_threshold -v`
Expected: FAIL with `AttributeError: module 'lib.clv' has no attribute 'aggregate_clv_by_threshold'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/lib/clv.py`:

```python
def aggregate_clv_by_threshold(rows: list, thresholds) -> dict:
    """headroom 子集上,逐門檻(|edge_pp|≥t)的 CLV 彙總。重用 _finite_nonzero / _stats。
    回傳每市場一個 list:[{t, n, mean, share_pos}, ...](對齊 thresholds 順序)。"""
    hr = [r for r in rows if r["has_headroom"]]

    def _sweep(clv_key, edge_key):
        cand = [r for r in hr if r[clv_key] is not None and _finite_nonzero(r[edge_key])]
        out = []
        for t in thresholds:
            sub = [r for r in cand if abs(r[edge_key]) >= t]
            s = _stats([r[clv_key] for r in sub])
            out.append({"t": t, "n": s["n"], "mean": s["mean"], "share_pos": s["share_pos"]})
        return out

    return {"rl": _sweep("rl_clv", "rl_edge_pp"),
            "ou": _sweep("ou_clv", "ou_edge_pp")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_clv.py -v`
Expected: all CLV tests PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/clv.py scripts/tests/test_clv.py
git commit -m "feat(backtest): CLV aggregation per edge threshold"
```

---

### Task 3: `render.render_threshold_section` + wire into `render_report`

**Files:**
- Modify: `scripts/lib/render.py` (add helper `_clvpp`, add `render_threshold_section`, extend `render_report` signature + body)
- Test: `scripts/tests/test_backtest_metrics_v2.py` (add 1 render test)

- [ ] **Step 1: Write the failing test**

Add to the END of `scripts/tests/test_backtest_metrics_v2.py`:

```python
def test_render_threshold_section_present():
    from lib import render
    sweep = {"thresholds": [0, 2],
             "rl": [{"t": 0, "n_bets": 100, "hit_rate": 0.5}, {"t": 2, "n_bets": 40, "hit_rate": 0.55}],
             "ou": [{"t": 0, "n_bets": 90, "hit_rate": 0.52}, {"t": 2, "n_bets": 30, "hit_rate": None}]}
    sweep_clv = {"rl": [{"t": 0, "n": 95, "mean": -0.2, "share_pos": 0.44},
                        {"t": 2, "n": 38, "mean": 0.1, "share_pos": 0.5}],
                 "ou": [{"t": 0, "n": 88, "mean": 0.07, "share_pos": 0.45},
                        {"t": 2, "n": 0, "mean": None, "share_pos": None}]}
    text = render.render_threshold_section(sweep, sweep_clv)
    assert "edge 門檻掃描" in text
    assert "≥0pp" in text and "≥2pp" in text
    assert "RL" in text and "O/U" in text
    assert "—" in text   # None hit_rate / mean / share render as em dash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_backtest_metrics_v2.py::test_render_threshold_section_present -v`
Expected: FAIL with `AttributeError: module 'lib.render' has no attribute 'render_threshold_section'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/lib/render.py`, add `_clvpp` right after the existing `_f` helper (around line 12):

```python
def _clvpp(x) -> str:
    return f"{x:+.2f}pp" if isinstance(x, (int, float)) else "—"
```

Then add `render_threshold_section` right before `render_report`:

```python
def render_threshold_section(sweep: dict, sweep_clv: dict) -> str:
    def _table(hits, clvs):
        lines = ["| 門檻 | 注數 | 命中率 | CLV注數 | CLV mean | 往我方 |",
                 "|------|------|--------|---------|----------|--------|"]
        for h, c in zip(hits, clvs):
            lines.append(
                f"| ≥{h['t']}pp | {h['n_bets']} | {_pct(h['hit_rate'])} "
                f"| {c['n']} | {_clvpp(c['mean'])} | {_pct(c['share_pos'])} |")
        return "\n".join(lines)
    return (
        "## edge 門檻掃描(雙向:|edge|≥門檻,下 model pick 側)\n\n"
        "RL:\n" + _table(sweep["rl"], sweep_clv["rl"]) + "\n\n"
        "O/U:\n" + _table(sweep["ou"], sweep_clv["ou"]) + "\n\n"
        "> 判讀:命中率與 CLV 要「同向往上」才算門檻撈出 edge;\n"
        "> 命中率升但 CLV≈0/負 = 高門檻只是小樣本雜訊,非真 alpha。\n"
    )
```

Then change the `render_report` signature (currently line 36-37) from:

```python
def render_report(*, df: pd.DataFrame, rl: dict, ou: dict, edge: dict,
                  month: str, out_path: Path, clv: dict | None = None) -> None:
```

to:

```python
def render_report(*, df: pd.DataFrame, rl: dict, ou: dict, edge: dict,
                  month: str, out_path: Path, clv: dict | None = None,
                  sweep: dict | None = None, sweep_clv: dict | None = None) -> None:
```

And in `render_report`'s body, insert the sweep block immediately AFTER the edge-calibration lines and BEFORE the `if clv is not None:` block. The edge-calibration block ends with `"",` (the blank line after the O/U 正 edge line). Insert there:

```python
    if sweep is not None and sweep_clv is not None:
        lines += [render_threshold_section(sweep, sweep_clv), ""]
    if clv is not None:
        lines += [render_clv_section(clv), ""]
```

(The `if clv is not None:` block already exists — add the `if sweep ...` block right before it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_backtest_metrics_v2.py scripts/tests/test_clv.py -v`
Expected: all PASS (the existing `test_render_clv_section_present` in test_clv.py must stay green — signature change is backward-compatible via defaults).

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/render.py scripts/tests/test_backtest_metrics_v2.py
git commit -m "feat(backtest): render edge-threshold-sweep section"
```

---

### Task 4: Wire into `backtest.py`, run full suite, regenerate report

**Files:**
- Modify: `scripts/backtest.py` (imports, `THRESHOLDS`, `cmd_run` body, `render_report` call)

- [ ] **Step 1: Update imports and add THRESHOLDS**

In `scripts/backtest.py`, change the metrics import (line 23) from:

```python
from lib.metrics import compute_rl_metrics, compute_ou_metrics, compute_edge_calibration
```

to:

```python
from lib.metrics import (
    compute_rl_metrics, compute_ou_metrics, compute_edge_calibration,
    compute_threshold_sweep,
)
```

Change the clv import (line 25) from:

```python
from lib.clv import compute_clv_rows, aggregate_clv
```

to:

```python
from lib.clv import compute_clv_rows, aggregate_clv, aggregate_clv_by_threshold
```

Add after `SNAPSHOTS_DIR = ...` (line 27):

```python
THRESHOLDS = [0, 1, 2, 3, 4]   # edge pp 門檻掃描
```

- [ ] **Step 2: Update `cmd_run` to compute the sweep and reuse clv_rows**

In `cmd_run`, replace the metrics+clv block (currently lines 39-42):

```python
    rl = compute_rl_metrics(df)
    ou = compute_ou_metrics(df)
    edge = compute_edge_calibration(df)
    clv = aggregate_clv(compute_clv_rows(df.to_dict("records"), SNAPSHOTS_DIR)) if len(df) else None
```

with:

```python
    rl = compute_rl_metrics(df)
    ou = compute_ou_metrics(df)
    edge = compute_edge_calibration(df)
    sweep = compute_threshold_sweep(df, THRESHOLDS) if len(df) else None
    clv_rows = compute_clv_rows(df.to_dict("records"), SNAPSHOTS_DIR) if len(df) else []
    clv = aggregate_clv(clv_rows) if clv_rows else None
    sweep_clv = aggregate_clv_by_threshold(clv_rows, THRESHOLDS) if clv_rows else None
```

And update the `render_report(...)` call (currently line 46) from:

```python
    render_report(df=df, rl=rl, ou=ou, edge=edge, month=args.month, out_path=report_path, clv=clv)
```

to:

```python
    render_report(df=df, rl=rl, ou=ou, edge=edge, month=args.month, out_path=report_path,
                  clv=clv, sweep=sweep, sweep_clv=sweep_clv)
```

- [ ] **Step 3: Run the FULL test suite**

Run: `python -m pytest scripts/tests/ -v`
Expected: all tests PASS (no regressions across the whole suite).

- [ ] **Step 4: Regenerate the May report and eyeball the new section**

Run: `python scripts/backtest.py run --month 2026-05`
Expected: console prints `Valid (odds+result): 292 / 344` (or current count); `analysis-data/backtest/2026-05-report.md` now contains a `## edge 門檻掃描` section with RL and O/U tables, 5 rows each (≥0/1/2/3/4 pp), inserted between 「edge 校準」 and 「CLV」.

Then Read `analysis-data/backtest/2026-05-report.md` and sanity-check: `t=0` bet counts should be near the full valid sample (two-sided), and counts should be monotonically non-increasing as the threshold rises.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest.py analysis-data/backtest/2026-05-report.md
git commit -m "feat(backtest): wire edge-threshold sweep into report + regenerate 2026-05"
```

---

## Self-Review

**Spec coverage:**
- Two-sided `sign(edge)` bet semantics → Task 1 (`rl_hit`/`ou_hit` via `(edge>0)==cover`), test asserts two-sided away pick hits. ✓
- `|edge| ≥ t` filter, `t=0` = all non-zero-edge bets → Task 1 candidate frame excludes `edge==0`; Task 2 reuses `_finite_nonzero`. ✓
- Push excluded from O/U → Task 1 `ou = ou[ou["actual_total"] != ou["total_line"]]`, test row3 push excluded. ✓
- Two separate denominators (hit on `_valid` ~292, CLV on headroom ~270) → hit rate in metrics, CLV in clv; render shows `注數` and `CLV注數` as separate columns. ✓
- Reuse `compute_clv_rows` once → Task 4 computes `clv_rows` once, feeds both aggregators. ✓
- Section between edge-calibration and CLV → Task 3 inserts `if sweep` block before `if clv`. ✓
- Thresholds `[0,1,2,3,4]` → Task 4 `THRESHOLDS`. ✓
- n=0 → None rendered as "—" → Task 1/2 emit None; `_pct`/`_clvpp` map None to "—"; Task 3 test asserts "—" present. ✓
- Out of scope (ROI, bet action, model/config change, binning) → none added. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; test bodies are concrete with hand-computed expected values. ✓

**Type consistency:** `compute_threshold_sweep` returns `{"thresholds", "rl":[{t,n_bets,hit_rate}], "ou":[...]}`; `aggregate_clv_by_threshold` returns `{"rl":[{t,n,mean,share_pos}], "ou":[...]}`; `render_threshold_section(sweep, sweep_clv)` reads exactly those keys (`h['n_bets']`/`h['hit_rate']`, `c['n']`/`c['mean']`/`c['share_pos']`); `backtest.py` passes `sweep=`/`sweep_clv=` matching `render_report`'s new params. Consistent across tasks. ✓

**Note vs spec:** Spec mentioned `test_backtest_e2e.py (exists)` for the section-appears assertion — that file does NOT exist (only a stale `.pyc`). This plan covers the section via a `render_threshold_section` unit test in `test_backtest_metrics_v2.py`, matching the existing `test_render_clv_section_present` convention in `test_clv.py`.
