# MLB `config.py` 2-Knob Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit two coefficients — `LEAGUE_RG` (run level) and `SIGMA_TEAM` (confidence) — to the backfilled point-in-time sample so the model's probabilities are calibrated, holding all other coefficients at their priors.

**Architecture:** Each `features.json` froze the model *inputs*, so we recompute μ and probabilities for any candidate knob values purely in-memory (no API calls). Stage 1 solves `LEAGUE_RG` by bisection so mean predicted total matches mean actual total; Stage 2 sets `SIGMA_TEAM` to the Gaussian MLE of the score residuals. A CLI prints before/after calibration (log-loss, reliability, edge) on the odds-bearing May games and optionally writes the two values into `config.py`.

**Tech Stack:** Python 3, `pytest`, stdlib `statistics.NormalDist` / `math` (no new deps). Spec: `docs/superpowers/specs/2026-05-29-mlb-config-calibration-design.md`.

**Commit policy:** Repo owner commits manually. Treat `Commit` steps as checkpoints — run only if the owner asks; otherwise leave staged.

---

## File Structure

- **Modify `scripts/run_model.py`** — add optional `league_rg=` to `expected_runs`, optional `sigma=` to `cover_prob_home` and `over_prob` (default to `config`; backward-compatible). One source of truth for the model math so the fitter reuses it.
- **Modify `scripts/tests/test_run_model.py`** — tests that the new optional args change μ / probabilities and that omitting them preserves current behavior.
- **Create `scripts/fit_config.py`** — pure fns (`recompute_mu`, `fit_league_rg`, `fit_sigma_team`, `eval_calibration`, `rewrite_config_text`) + I/O loader `load_fit_rows` + CLI `main`.
- **Create `scripts/tests/test_fit_config.py`** — TDD for every pure fn with synthetic games, plus a tmp-dir test for `load_fit_rows`.

**Row schema** produced by `load_fit_rows` and consumed by the pure fns (define once, used by all tasks):
```python
# one dict per backfilled game that has a result.json
{
  "date": "2026-05-12", "matchup": "NYY@BAL",
  "home_rs_recent": float, "home_rs_season": float,
  "away_rs_recent": float, "away_rs_season": float,
  "home_starter_fip": float | None, "away_starter_fip": float | None,
  "home_bullpen_era": float, "away_bullpen_era": float, "park_factor": float,
  "actual_total": int, "actual_margin": int,        # home_score+away_score, home_score-away_score
  "has_odds": bool,
  "rl_home_point": float | None, "rl_home_no_vig": float | None,
  "total_line": float | None, "over_no_vig": float | None,
}
```

---

## Task 1: Parametrize `run_model` (optional `league_rg`, `sigma`)

**Files:**
- Modify: `scripts/run_model.py`
- Test: `scripts/tests/test_run_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_run_model.py`:

```python
def test_expected_runs_league_rg_override_changes_mu():
    import run_model, config
    base = run_model.expected_runs(4.5, 4.0, 4.0, 4.2, 100.0)
    override = run_model.expected_runs(4.5, 4.0, 4.0, 4.2, 100.0, league_rg=config.LEAGUE_RG * 2)
    # mu scales as 1/league_rg → doubling league_rg halves each mu
    assert abs(override[0] - base[0] / 2) < 1e-9
    assert abs(override[1] - base[1] / 2) < 1e-9


def test_cover_prob_home_sigma_override_changes_prob():
    import run_model
    base = run_model.cover_prob_home(0.5, -1.5)
    wide = run_model.cover_prob_home(0.5, -1.5, sigma=100.0)
    # huge sigma → prob pulled toward 0.5
    assert abs(wide - 0.5) < abs(base - 0.5)


def test_over_prob_sigma_override_changes_prob():
    import run_model
    base = run_model.over_prob(8.0, 8.5)
    wide = run_model.over_prob(8.0, 8.5, sigma=100.0)
    assert abs(wide - 0.5) < abs(base - 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_run_model.py -q`
Expected: FAIL — `expected_runs() got an unexpected keyword argument 'league_rg'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/run_model.py`, change three function signatures/bodies:

```python
def expected_runs(home_rs: float, away_rs: float,
                  home_pitch: float, away_pitch: float,
                  pf: float, league_rg: float | None = None) -> tuple[float, float]:
    """期望得分。league_rg 省略時用 config.LEAGUE_RG。"""
    lg = config.LEAGUE_RG if league_rg is None else league_rg
    pf_mult = pf / 100.0
    mu_home = home_rs * away_pitch / lg * pf_mult
    mu_away = away_rs * home_pitch / lg * pf_mult
    return mu_home, mu_away


def cover_prob_home(mu_margin: float, rl_point_home: float, sigma: float | None = None) -> float:
    """P(主隊過 RL)。sigma 省略時用 config.SIGMA。"""
    s = config.SIGMA if sigma is None else sigma
    z = (-rl_point_home - mu_margin) / s
    return 1.0 - _N.cdf(z)


def over_prob(mu_total: float, total_line: float, sigma: float | None = None) -> float:
    """P(Over)。sigma 省略時用 config.SIGMA。"""
    s = config.SIGMA if sigma is None else sigma
    z = (total_line - mu_total) / s
    return 1.0 - _N.cdf(z)
```

(`predict()` calls these with no extra args → still uses `config` defaults. No other change.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_run_model.py -q`
Expected: PASS (all, including the 3 new).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/run_model.py scripts/tests/test_run_model.py
git commit -m "feat(run_model): optional league_rg/sigma overrides for fitting"
```

---

## Task 2: `recompute_mu` + `load_fit_rows`

**Files:**
- Create: `scripts/fit_config.py`
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/tests/test_fit_config.py`:

```python
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fit_config
import config


def _row(**kw):
    base = dict(home_rs_recent=4.5, home_rs_season=4.5, away_rs_recent=4.0, away_rs_season=4.0,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                away_bullpen_era=4.0, park_factor=100.0, actual_total=9, actual_margin=1,
                has_odds=False, rl_home_point=None, rl_home_no_vig=None,
                total_line=None, over_no_vig=None, date="2026-05-01", matchup="A@B")
    base.update(kw)
    return base


def test_recompute_mu_matches_manual():
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             park_factor=100.0)
    # rs blend = 4.4; pitch = 0.6*4.0+0.4*4.0 = 4.0; mu = 4.4*4.0/4.4*1.0 = 4.0 each
    mt, mm = fit_config.recompute_mu(r, league_rg=4.4)
    assert abs(mt - 8.0) < 1e-9
    assert abs(mm - 0.0) < 1e-9


def test_recompute_mu_null_fip_uses_league_rg_fallback():
    r = _row(home_starter_fip=None, away_starter_fip=None)
    # should not raise; null fip → fallback = league_rg
    mt, mm = fit_config.recompute_mu(r, league_rg=4.4)
    assert mt > 0


def test_load_fit_rows_reads_inputs_and_result(tmp_path):
    d = tmp_path / "2026-05-01" / "A@B"
    d.mkdir(parents=True)
    feats = {"schema_version": 2,
             "inputs": {"home_rs_recent": 4.5, "home_rs_season": 4.4,
                        "away_rs_recent": 4.0, "away_rs_season": 4.1,
                        "home_starter": {"fip": 3.5}, "away_starter": {"fip": None},
                        "home_bullpen_era": 3.9, "away_bullpen_era": 4.1, "park_factor": 101.0},
             "odds": {"rl": {"home_point": -1.5, "home_no_vig": 0.40},
                      "total": {"line": 8.5, "over_no_vig": 0.48}}}
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"home_score": 6, "away_score": 3, "total": 9}), encoding="utf-8")

    rows = fit_config.load_fit_rows({"2026-05"}, data_dir=tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["home_starter_fip"] == 3.5
    assert r["away_starter_fip"] is None
    assert r["actual_total"] == 9
    assert r["actual_margin"] == 3
    assert r["has_odds"] is True
    assert r["rl_home_point"] == -1.5
    assert r["total_line"] == 8.5


def test_load_fit_rows_skips_without_result(tmp_path):
    d = tmp_path / "2026-05-02" / "C@D"
    d.mkdir(parents=True)
    (d / "features.json").write_text(json.dumps({"schema_version": 2, "inputs": {}}), encoding="utf-8")
    rows = fit_config.load_fit_rows({"2026-05"}, data_dir=tmp_path)
    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_fit_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fit_config'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/fit_config.py`:

```python
#!/usr/bin/env python3
"""Fit 2 個係數(LEAGUE_RG, SIGMA_TEAM)使模型機率校準。從凍結的 features.json 重算,零 refetch。

用法:
  python scripts/fit_config.py                 # 提案 + before/after,不改檔
  python scripts/fit_config.py --write          # 同時把兩個值寫回 config.py
  python scripts/fit_config.py --months 2026-04,2026-05
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"
CONFIG_PATH = SCRIPT_DIR / "config.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
import run_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def recompute_mu(row: dict, league_rg: float) -> tuple[float, float]:
    """從凍結 inputs 以 candidate league_rg 重算 (mu_total, mu_margin)。RECENT_W 等用 config 先驗。"""
    home_rs = config.RECENT_W * row["home_rs_recent"] + (1 - config.RECENT_W) * row["home_rs_season"]
    away_rs = config.RECENT_W * row["away_rs_recent"] + (1 - config.RECENT_W) * row["away_rs_season"]
    home_fip = row["home_starter_fip"] if row["home_starter_fip"] is not None else league_rg
    away_fip = row["away_starter_fip"] if row["away_starter_fip"] is not None else league_rg
    home_pitch = run_model.pitch_today(home_fip, row["home_bullpen_era"])
    away_pitch = run_model.pitch_today(away_fip, row["away_bullpen_era"])
    mu_home, mu_away = run_model.expected_runs(home_rs, away_rs, home_pitch, away_pitch,
                                               row["park_factor"], league_rg=league_rg)
    return mu_home + mu_away, mu_home - mu_away


def load_fit_rows(months: set, data_dir: Path = ANALYSIS_DATA_DIR) -> list:
    """讀 features.json(schema 2)+ result.json → fit row 清單。無 result 或非 v2 略過。"""
    rows = []
    for date_dir in sorted(Path(data_dir).iterdir()):
        if not date_dir.is_dir() or date_dir.name.endswith(".local-backup"):
            continue
        if date_dir.name[:7] not in months:
            continue
        for m in sorted(date_dir.iterdir()):
            if not m.is_dir():
                continue
            fp = m / "features.json"
            rp = m / "result.json"
            if not fp.exists() or not rp.exists():
                continue
            try:
                feats = json.loads(fp.read_text(encoding="utf-8"))
                result = json.loads(rp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if feats.get("schema_version") != 2:
                continue
            inp = feats.get("inputs") or {}
            if not inp:
                continue
            odds = feats.get("odds") or {}
            rl = (odds.get("rl") or {}) if odds else {}
            total = (odds.get("total") or {}) if odds else {}
            rows.append({
                "date": date_dir.name, "matchup": m.name,
                "home_rs_recent": inp["home_rs_recent"], "home_rs_season": inp["home_rs_season"],
                "away_rs_recent": inp["away_rs_recent"], "away_rs_season": inp["away_rs_season"],
                "home_starter_fip": (inp.get("home_starter") or {}).get("fip"),
                "away_starter_fip": (inp.get("away_starter") or {}).get("fip"),
                "home_bullpen_era": inp["home_bullpen_era"], "away_bullpen_era": inp["away_bullpen_era"],
                "park_factor": inp["park_factor"],
                "actual_total": result["home_score"] + result["away_score"],
                "actual_margin": result["home_score"] - result["away_score"],
                "has_odds": bool(odds),
                "rl_home_point": rl.get("home_point"), "rl_home_no_vig": rl.get("home_no_vig"),
                "total_line": total.get("line"), "over_no_vig": total.get("over_no_vig"),
            })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_fit_config.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): load frozen rows + recompute_mu"
```

---

## Task 3: `fit_league_rg` (Stage 1 bisection)

**Files:**
- Modify: `scripts/fit_config.py`
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_fit_league_rg_recovers_level():
    # All rows identical, no null FIP → mu_total ∝ 1/L exactly.
    # At L0=4.4 mu_total=8.0 (see recompute test). Want mean actual total = 8.0 → L stays ~4.4.
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                 away_bullpen_era=4.0, park_factor=100.0, actual_total=8) for _ in range(20)]
    L = fit_config.fit_league_rg(rows)
    assert abs(L - 4.4) < 0.05


def test_fit_league_rg_higher_when_actual_lower():
    # actual total 7.0 < predicted 8.0 at L=4.4 → need higher L to lower mu
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                 away_bullpen_era=4.0, park_factor=100.0, actual_total=7) for _ in range(20)]
    L = fit_config.fit_league_rg(rows)
    assert L > 4.4
    # mean predicted at fitted L should ≈ 7.0
    mean_mu = sum(fit_config.recompute_mu(r, L)[0] for r in rows) / len(rows)
    assert abs(mean_mu - 7.0) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k fit_league_rg -q`
Expected: FAIL — `AttributeError: module 'fit_config' has no attribute 'fit_league_rg'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fit_config.py`:

```python
def fit_league_rg(rows: list, lo: float = 2.0, hi: float = 8.0, iters: int = 50) -> float:
    """二分搜尋 LEAGUE_RG 使 mean(預測 mu_total) = mean(實際 total)。mu_total 隨 L 單調遞減。"""
    target = sum(r["actual_total"] for r in rows) / len(rows)

    def mean_mu_total(L: float) -> float:
        return sum(recompute_mu(r, L)[0] for r in rows) / len(rows)

    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_mu_total(mid) > target:   # 預測偏高 → 提高 L 以壓低 mu
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -k fit_league_rg -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): fit_league_rg via bisection (mean-match totals)"
```

---

## Task 4: `fit_sigma_team` (Stage 2 residual MLE)

**Files:**
- Modify: `scripts/fit_config.py`
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_fit_sigma_team_recovers_spread():
    # At L=4.4 each row predicts mu_total=8.0, mu_margin=0.0.
    # Build rows whose residuals have a known RMS. Use symmetric pairs so totals & margins
    # each have residual magnitude exactly k → SIGMA_TEAM = sqrt((k^2 + k^2)/4) = k/sqrt(2).
    k = 4.0
    rows = []
    for _ in range(10):
        rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                         home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                         away_bullpen_era=4.0, park_factor=100.0,
                         actual_total=int(8 + k), actual_margin=int(0 + k)))
        rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                         home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                         away_bullpen_era=4.0, park_factor=100.0,
                         actual_total=int(8 - k), actual_margin=int(0 - k)))
    s = fit_config.fit_sigma_team(rows, league_rg=4.4)
    assert abs(s - k / math.sqrt(2)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k fit_sigma_team -q`
Expected: FAIL — `AttributeError: module 'fit_config' has no attribute 'fit_sigma_team'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fit_config.py`:

```python
def fit_sigma_team(rows: list, league_rg: float) -> float:
    """SIGMA_TEAM = sqrt( (mean(r_total^2)+mean(r_margin^2)) / 4 )，r = 實際 - 預測(高斯 MLE)。"""
    sse = 0.0
    for r in rows:
        mt, mm = recompute_mu(r, league_rg)
        sse += (r["actual_total"] - mt) ** 2 + (r["actual_margin"] - mm) ** 2
    n = len(rows)
    return round(math.sqrt(sse / (4 * n)), 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -k fit_sigma_team -q`
Expected: PASS.

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): fit_sigma_team via residual MLE"
```

---

## Task 5: `eval_calibration` (log-loss + edge; needs odds)

**Files:**
- Modify: `scripts/fit_config.py`
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_eval_calibration_basic():
    # One game, with odds. At L=4.4, sigma_team chosen so SIGMA=4.24.
    # mu_margin=0, rl_home_point=-1.5 → p_cover = P(margin>1.5) with mean 0 → <0.5.
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             park_factor=100.0, actual_total=9, actual_margin=3, has_odds=True,
             rl_home_point=-1.5, rl_home_no_vig=0.40, total_line=8.5, over_no_vig=0.48)
    out = fit_config.eval_calibration([r], league_rg=4.4, sigma_team=3.0)
    assert out["n_rl"] == 1
    assert out["n_ou"] == 1                      # 9 != 8.5, not a push
    assert out["rl_log_loss"] > 0
    assert out["ou_log_loss"] > 0


def test_eval_calibration_excludes_push_and_no_odds():
    r_push = _row(has_odds=True, rl_home_point=-1.5, rl_home_no_vig=0.4,
                  total_line=9.0, over_no_vig=0.48, actual_total=9, actual_margin=2)  # total==line → push
    r_noodds = _row(has_odds=False)
    out = fit_config.eval_calibration([r_push, r_noodds], league_rg=4.4, sigma_team=3.0)
    assert out["n_ou"] == 0       # push excluded
    assert out["n_rl"] == 1       # the push row still has RL; no-odds row excluded everywhere
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k eval_calibration -q`
Expected: FAIL — `AttributeError: module 'fit_config' has no attribute 'eval_calibration'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fit_config.py`:

```python
def _clamp(p: float, eps: float = 1e-9) -> float:
    return min(max(p, eps), 1 - eps)


def eval_calibration(rows: list, league_rg: float, sigma_team: float) -> dict:
    """在有盤口的列上算 RL/O-U log-loss、正 edge 命中率。SIGMA = sigma_team*sqrt(2)。"""
    sigma = sigma_team * math.sqrt(2)
    rl_ll, ou_ll = [], []
    rl_edge_y, ou_edge_y = [], []
    for r in rows:
        if not r["has_odds"] or r["rl_home_point"] is None:
            continue
        mt, mm = recompute_mu(r, league_rg)
        # RL
        p_cover = _clamp(run_model.cover_prob_home(mm, r["rl_home_point"], sigma=sigma))
        y = 1.0 if r["actual_margin"] > -r["rl_home_point"] else 0.0
        rl_ll.append(-(y * math.log(p_cover) + (1 - y) * math.log(1 - p_cover)))
        if (p_cover - r["rl_home_no_vig"]) * 100 > 0:
            rl_edge_y.append(y)
        # O/U (exclude push)
        if r["total_line"] is not None and r["actual_total"] != r["total_line"]:
            p_over = _clamp(run_model.over_prob(mt, r["total_line"], sigma=sigma))
            yo = 1.0 if r["actual_total"] > r["total_line"] else 0.0
            ou_ll.append(-(yo * math.log(p_over) + (1 - yo) * math.log(1 - p_over)))
            if (p_over - r["over_no_vig"]) * 100 > 0:
                ou_edge_y.append(yo)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_rl": len(rl_ll), "rl_log_loss": _mean(rl_ll),
        "n_ou": len(ou_ll), "ou_log_loss": _mean(ou_ll),
        "rl_pos_edge_n": len(rl_edge_y), "rl_pos_edge_hit": _mean(rl_edge_y),
        "ou_pos_edge_n": len(ou_edge_y), "ou_pos_edge_hit": _mean(ou_edge_y),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -k eval_calibration -q`
Expected: PASS.

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): eval_calibration (log-loss + edge)"
```

---

## Task 6: `rewrite_config_text` + CLI `main`

**Files:**
- Modify: `scripts/fit_config.py`
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_rewrite_config_text_replaces_only_two_values_and_keeps_comments():
    text = (
        "import math\n"
        "LEAGUE_RG = 4.4        # 聯盟每場均分\n"
        "RECENT_W = 0.35        # RS blend\n"
        "SIGMA_TEAM = 3.0       # 單隊單場得分 SD(歷史先驗)\n"
        "SIGMA = SIGMA_TEAM * math.sqrt(2)\n"
    )
    out = fit_config.rewrite_config_text(text, league_rg=4.75, sigma_team=3.42)
    assert "LEAGUE_RG = 4.75        # 聯盟每場均分" in out
    assert "SIGMA_TEAM = 3.42       # 單隊單場得分 SD(歷史先驗)" in out
    assert "RECENT_W = 0.35        # RS blend" in out          # untouched
    assert "SIGMA = SIGMA_TEAM * math.sqrt(2)" in out          # untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k rewrite_config_text -q`
Expected: FAIL — `AttributeError: module 'fit_config' has no attribute 'rewrite_config_text'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fit_config.py`:

```python
def rewrite_config_text(text: str, league_rg: float, sigma_team: float) -> str:
    """只改 LEAGUE_RG / SIGMA_TEAM 的數值,保留行內註解與其餘內容。"""
    text = re.sub(r"^(LEAGUE_RG = )[\d.]+", rf"\g<1>{league_rg}", text, flags=re.M)
    text = re.sub(r"^(SIGMA_TEAM = )[\d.]+", rf"\g<1>{sigma_team}", text, flags=re.M)
    return text


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit LEAGUE_RG + SIGMA_TEAM(機率校準)")
    p.add_argument("--months", default="2026-03,2026-04,2026-05",
                   help="逗號分隔 YYYY-MM(μ/σ fit 用全部;校準驗證自動取有盤口者)")
    p.add_argument("--write", action="store_true", help="把擬合值寫回 config.py")
    args = p.parse_args(argv)

    months = set(args.months.split(","))
    rows = load_fit_rows(months)
    if not rows:
        print("沒有可用資料(features.json + result.json)。先跑 backfill + fetch_results。", file=sys.stderr)
        return 1

    train = [r for r in rows]                                   # μ/σ:全部有 result 的列
    may_odds = [r for r in rows if r["date"][:7] == "2026-05" and r["has_odds"]]
    apr = [r for r in rows if r["date"][:7] in ("2026-03", "2026-04")]

    L_fit = fit_league_rg(train)
    S_fit = fit_sigma_team(train, L_fit)

    print(f"樣本:fit(有 result)={len(train)}  五月有盤口(驗證)={len(may_odds)}  三四月={len(apr)}")
    print(f"\n提案係數:")
    print(f"  LEAGUE_RG : {config.LEAGUE_RG} → {L_fit}")
    print(f"  SIGMA_TEAM: {config.SIGMA_TEAM} → {S_fit}")

    before = eval_calibration(may_odds, config.LEAGUE_RG, config.SIGMA_TEAM)
    after = eval_calibration(may_odds, L_fit, S_fit)
    print(f"\n五月校準 before → after(log-loss 越低越好):")
    print(f"  RL log-loss : {_fmt(before['rl_log_loss'])} → {_fmt(after['rl_log_loss'])}  (n={after['n_rl']})")
    print(f"  OU log-loss : {_fmt(before['ou_log_loss'])} → {_fmt(after['ou_log_loss'])}  (n={after['n_ou']})")
    print(f"  RL +edge 命中: {_fmt(before['rl_pos_edge_hit'])} → {_fmt(after['rl_pos_edge_hit'])}  (n={after['rl_pos_edge_n']})")
    print(f"  OU +edge 命中: {_fmt(before['ou_pos_edge_hit'])} → {_fmt(after['ou_pos_edge_hit'])}  (n={after['ou_pos_edge_n']})")

    if apr:
        L_apr = fit_league_rg(apr)
        S_apr = fit_sigma_team(apr, L_apr)
        may_apr = eval_calibration(may_odds, L_apr, S_apr)
        print(f"\n穩定性(三四月 fit → 五月驗證):L={L_apr} σ={S_apr}  "
              f"RL ll={_fmt(may_apr['rl_log_loss'])} OU ll={_fmt(may_apr['ou_log_loss'])}")

    if args.write:
        new_text = rewrite_config_text(CONFIG_PATH.read_text(encoding="utf-8"), L_fit, S_fit)
        CONFIG_PATH.write_text(new_text, encoding="utf-8")
        print(f"\n[WRITE] 已更新 {CONFIG_PATH}(LEAGUE_RG={L_fit}, SIGMA_TEAM={S_fit})")
    else:
        print(f"\n(未寫檔。確認後加 --write 套用,或手動改 config.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -q`
Expected: PASS (all fit_config tests).

- [ ] **Step 5: Full suite**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 6: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): config rewrite + CLI before/after report"
```

---

## Task 7: Execute — backfill April, fit, review (operational, not TDD)

**Files:** none (operational). Produces April `features.json`/`result.json` and the fit report (stdout).

- [ ] **Step 1: Full suite green**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 2: Backfill late-March + April** (relief index already covers it → predictions only; long — consider background)

Run: `python scripts/backfill.py --start 2026-03-26 --end 2026-04-30`
Expected: one `[YYYY-MM-DD] N 場` line per day; writes `features.json` per game (most April games have `odds=null` — expected, μ/σ fit doesn't need odds).

- [ ] **Step 3: Fetch results for March + April**

Run: `python scripts/fetch_results.py --month 2026-03` then `python scripts/fetch_results.py --month 2026-04`
Expected: per-day `fetched=N matched=M`.

- [ ] **Step 4: Run the fitter (no write)**

Run: `python scripts/fit_config.py`
Expected: prints sample sizes, proposed `LEAGUE_RG`/`SIGMA_TEAM`, before→after May calibration (log-loss should drop or hold; edge hit moves toward/above 50%), and the April→May stability line.

- [ ] **Step 5: Sanity-check the proposal**

Confirm: fitted `LEAGUE_RG` is plausible (~4–5), `SIGMA_TEAM` plausible (~3–5), after-fit log-loss ≤ before, and the April→May stability fit is close to the all-data fit. If after-fit is worse or the stability fit diverges wildly, STOP and report — do not `--write`.

- [ ] **Step 6: Apply (only on user approval)**

The fitted numbers change the **live** model. Present them to the user; apply only when approved:
Run: `python scripts/fit_config.py --write`
Then re-freeze + re-backtest to see the effect end-to-end:
Run: `python scripts/backtest.py run --month 2026-05`
(Note: the existing May `features.json` froze μ/probabilities under the OLD config; to reflect new config in the backtest, re-run `python scripts/backfill.py --start 2026-05-01 --end 2026-05-25` first, then `fetch_results --month 2026-05`, then backtest.)

> **Known limitations (document, do not fix here):** changing `LEAGUE_RG` shifts the missing-FIP fallback for the few null-FIP games (negligible). Doubleheader dirs still overwrite. April odds are sparse, so the edge/log-loss validation stays a May-only check.

---

## Self-Review

**Spec coverage:** objective=calibration (log-loss/MLE) ✓ (Tasks 4–5); fit `LEAGUE_RG` mean-match ✓ (Task 3); fit `SIGMA_TEAM` residual MLE ✓ (Task 4); hold other coeffs (only these two parametrized; `recompute_mu` uses `config.RECENT_W` etc.) ✓; backfill April + results ✓ (Task 7); re-fit from frozen inputs, zero refetch ✓ (`recompute_mu`); validation before/after + April→May stability ✓ (Task 6 main); run_model optional args ✓ (Task 1); CLI proposes, `--write` applies ✓ (Task 6); does not auto-edit config without `--write` ✓.

**Placeholder scan:** none — all steps have full code/commands.

**Type/name consistency:** row dict keys defined in File Structure and produced by `load_fit_rows` (Task 2) are consumed identically by `recompute_mu`/`fit_league_rg`/`fit_sigma_team`/`eval_calibration` (Tasks 2–5). Signatures: `recompute_mu(row, league_rg)`, `fit_league_rg(rows)`, `fit_sigma_team(rows, league_rg)`, `eval_calibration(rows, league_rg, sigma_team)`, `rewrite_config_text(text, league_rg, sigma_team)`, `load_fit_rows(months, data_dir)` — consistent across tasks. `expected_runs(..., league_rg=None)`, `cover_prob_home(..., sigma=None)`, `over_prob(..., sigma=None)` match between Task 1 and their use in Tasks 2/5.
