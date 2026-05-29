# MLB RA-Defense Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-read-only ablation harness and use it to test, by an honest March–April→May time-split, whether blending team RA into the defense term (one knob `w_ra`) improves out-of-sample RL/O-U forecasting relative to the market.

**Architecture:** Reuse the frozen `features.json` rows (zero refetch). Generalize the μ recompute with an RA blend (`w_ra=0` ≡ today's model). Fit `w_ra` on Mar–Apr by minimizing the score-residual σ (odds-free), then evaluate the frozen weights on held-out May by per-bet log-loss vs the market. Accept only if it beats the `w_ra=0` baseline OOS by >1 SE. The harness never edits the live model.

**Tech Stack:** Python 3, `pytest`, stdlib `math`/`statistics.NormalDist`. Spec: `docs/superpowers/specs/2026-05-29-mlb-ra-feature-ablation-design.md`. Reuses `scripts/fit_config.py` + `scripts/run_model.py`.

**Commit policy:** Repo owner commits manually. Treat `Commit` steps as checkpoints — run only if the owner asks; otherwise leave staged.

---

## File Structure

- **Modify `scripts/fit_config.py`** — (a) add 4 RA fields to the `load_fit_rows` row dict; (b) add optional `mu_fn` param to `fit_league_rg` and `fit_sigma_team` (default = `recompute_mu`, backward-compatible) so the ablation reuses them with an RA-aware μ.
- **Modify `scripts/tests/test_fit_config.py`** — assert `load_fit_rows` returns the RA fields; assert `mu_fn` injection works.
- **Create `scripts/ablation.py`** — `recompute_mu_ra`, `fit_params`, `select_w_ra`, `eval_logloss`, `ablate_ra`, CLI `main`. Model-read-only (no edits to `config.py`/`run_model.py`).
- **Create `scripts/tests/test_ablation.py`** — TDD for every pure fn with synthetic rows.

**Row schema addition** (produced by `load_fit_rows`, consumed by `recompute_mu_ra`): the existing row dict gains `home_ra_recent`, `home_ra_season`, `away_ra_recent`, `away_ra_season` (floats, from `features.json["inputs"]`).

**`params` dict** (produced by `fit_params`, consumed by `eval_logloss`): `{"w_ra": float, "league_rg": float, "sigma_team": float}`.

---

## Task 1: Load RA fields into fit rows

**Files:**
- Modify: `scripts/fit_config.py` (the `rows.append({...})` dict in `load_fit_rows`)
- Test: `scripts/tests/test_fit_config.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_load_fit_rows_includes_ra_fields(tmp_path):
    d = tmp_path / "2026-05-01" / "A@B"
    d.mkdir(parents=True)
    feats = {"schema_version": 2,
             "inputs": {"home_rs_recent": 4.5, "home_rs_season": 4.4,
                        "away_rs_recent": 4.0, "away_rs_season": 4.1,
                        "home_ra_recent": 3.8, "home_ra_season": 4.2,
                        "away_ra_recent": 5.1, "away_ra_season": 4.7,
                        "home_starter": {"fip": 3.5}, "away_starter": {"fip": 4.0},
                        "home_bullpen_era": 3.9, "away_bullpen_era": 4.1, "park_factor": 101.0}}
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"home_score": 6, "away_score": 3, "total": 9}), encoding="utf-8")

    rows = fit_config.load_fit_rows({"2026-05"}, data_dir=tmp_path)
    r = rows[0]
    assert r["home_ra_recent"] == 3.8
    assert r["home_ra_season"] == 4.2
    assert r["away_ra_recent"] == 5.1
    assert r["away_ra_season"] == 4.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k includes_ra -q`
Expected: FAIL with `KeyError: 'home_ra_recent'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/fit_config.py`, inside `load_fit_rows`, add the four RA keys to the appended dict (right after the four RS keys):

```python
                "away_rs_recent": inp["away_rs_recent"], "away_rs_season": inp["away_rs_season"],
                "home_ra_recent": inp["home_ra_recent"], "home_ra_season": inp["home_ra_season"],
                "away_ra_recent": inp["away_ra_recent"], "away_ra_season": inp["away_ra_season"],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -k includes_ra -q`
Expected: PASS.

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): load frozen RA fields into fit rows"
```

---

## Task 2: Inject `mu_fn` into `fit_league_rg` / `fit_sigma_team`

**Files:**
- Modify: `scripts/fit_config.py` (`fit_league_rg`, `fit_sigma_team`)
- Test: `scripts/tests/test_fit_config.py`

**Context:** Both functions currently hardcode `recompute_mu(r, L)`. Adding an optional `mu_fn` lets the ablation pass an RA-aware μ while every existing caller (and `main`) is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fit_config.py`:

```python
def test_fit_functions_accept_mu_fn():
    # A custom mu_fn that always predicts total=10, margin=0, ignoring league_rg.
    def fake_mu(row, league_rg):
        return 10.0, 0.0
    rows = [_row(actual_total=10, actual_margin=0) for _ in range(5)]
    # sigma should be ~0 because fake_mu predicts actuals exactly
    s = fit_config.fit_sigma_team(rows, league_rg=4.4, mu_fn=fake_mu)
    assert s == 0.0
    # league_rg bisection still returns a value in range (fake_mu ignores L, mean_mu const=10 == target 10)
    L = fit_config.fit_league_rg(rows, mu_fn=fake_mu)
    assert 2.0 <= L <= 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fit_config.py -k accept_mu_fn -q`
Expected: FAIL with `fit_sigma_team() got an unexpected keyword argument 'mu_fn'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/fit_config.py`, change both signatures + internal calls:

```python
def fit_league_rg(rows: list, lo: float = 2.0, hi: float = 8.0, iters: int = 50, mu_fn=None) -> float:
    """二分搜尋 LEAGUE_RG 使 mean(預測 mu_total) = mean(實際 total)。mu_fn 省略時用 recompute_mu。"""
    mu_fn = mu_fn or recompute_mu
    target = sum(r["actual_total"] for r in rows) / len(rows)

    def mean_mu_total(L: float) -> float:
        return sum(mu_fn(r, L)[0] for r in rows) / len(rows)

    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_mu_total(mid) > target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def fit_sigma_team(rows: list, league_rg: float, mu_fn=None) -> float:
    """SIGMA_TEAM = sqrt( (mean(r_total^2)+mean(r_margin^2)) / 4 )。mu_fn 省略時用 recompute_mu。"""
    mu_fn = mu_fn or recompute_mu
    sse = 0.0
    for r in rows:
        mt, mm = mu_fn(r, league_rg)
        sse += (r["actual_total"] - mt) ** 2 + (r["actual_margin"] - mm) ** 2
    n = len(rows)
    return round(math.sqrt(sse / (4 * n)), 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_fit_config.py -q`
Expected: PASS (all — existing fit tests still green since default is `recompute_mu`).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fit_config.py scripts/tests/test_fit_config.py
git commit -m "feat(fit_config): optional mu_fn injection for fit_league_rg/fit_sigma_team"
```

---

## Task 3: `recompute_mu_ra`

**Files:**
- Create: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_ablation.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ablation
import fit_config
import config


def _row(**kw):
    base = dict(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                away_bullpen_era=4.0, park_factor=100.0, actual_total=9, actual_margin=1,
                has_odds=False, rl_home_point=None, rl_home_no_vig=None,
                total_line=None, over_no_vig=None, date="2026-05-01", matchup="A@B")
    base.update(kw)
    return base


def test_recompute_mu_ra_w0_equals_baseline():
    r = _row(home_rs_recent=4.8, home_rs_season=4.8, away_rs_recent=4.2, away_rs_season=4.2,
             home_starter_fip=4.5, away_starter_fip=3.6, home_bullpen_era=4.2, away_bullpen_era=4.0,
             home_ra_recent=3.0, home_ra_season=3.0, away_ra_recent=6.0, away_ra_season=6.0,
             park_factor=100.0)
    base = fit_config.recompute_mu(r, league_rg=4.4)
    ra0 = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.0)
    assert abs(ra0[0] - base[0]) < 1e-9
    assert abs(ra0[1] - base[1]) < 1e-9


def test_recompute_mu_ra_blends_in_ra():
    # All rs=4.4, pitch=4.0 (fip=bp=4.0). away RA blend high (6.0) → suppresses home offense less?
    # defense(away) = (1-w)*4.0 + w*6.0. Higher defense → higher mu_home (worse run prevention).
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             away_ra_recent=6.0, away_ra_season=6.0,   # away gives up more runs
             home_ra_recent=4.0, home_ra_season=4.0, park_factor=100.0)
    mt0, _ = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.0)
    mt5, _ = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.5)
    assert mt5 > mt0   # blending in away's high RA raises expected total
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_ablation.py -k recompute_mu_ra -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ablation'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/ablation.py`:

```python
#!/usr/bin/env python3
"""RA-defense 特徵 ablation(model-read-only)。把球隊 RA 摻進防禦項(w_ra),
用三~四月以得分殘差 fit w_ra,五月 OOS 以 vs-market log-loss 裁決。不改線上模型。

用法:
  python scripts/ablation.py
  python scripts/ablation.py --train 2026-03,2026-04 --test 2026-05
"""
import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
import run_model
import fit_config

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

W_RA_GRID = [round(0.05 * i, 2) for i in range(17)]   # 0.00 .. 0.80


def _ra_blend(recent: float, season: float) -> float:
    return config.RECENT_W * recent + (1 - config.RECENT_W) * season


def recompute_mu_ra(row: dict, league_rg: float, w_ra: float) -> tuple[float, float]:
    """μ with RA blended into each side's defense term. w_ra=0 ≡ fit_config.recompute_mu."""
    home_rs = config.RECENT_W * row["home_rs_recent"] + (1 - config.RECENT_W) * row["home_rs_season"]
    away_rs = config.RECENT_W * row["away_rs_recent"] + (1 - config.RECENT_W) * row["away_rs_season"]
    home_fip = row["home_starter_fip"] if row["home_starter_fip"] is not None else league_rg
    away_fip = row["away_starter_fip"] if row["away_starter_fip"] is not None else league_rg
    home_pitch = run_model.pitch_today(home_fip, row["home_bullpen_era"])
    away_pitch = run_model.pitch_today(away_fip, row["away_bullpen_era"])
    home_def = (1 - w_ra) * home_pitch + w_ra * _ra_blend(row["home_ra_recent"], row["home_ra_season"])
    away_def = (1 - w_ra) * away_pitch + w_ra * _ra_blend(row["away_ra_recent"], row["away_ra_season"])
    mu_home, mu_away = run_model.expected_runs(home_rs, away_rs, home_def, away_def,
                                               row["park_factor"], league_rg=league_rg)
    return mu_home + mu_away, mu_home - mu_away
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_ablation.py -k recompute_mu_ra -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "feat(ablation): recompute_mu_ra (RA-blended defense, w_ra=0 ≡ baseline)"
```

---

## Task 4: `fit_params` + `select_w_ra`

**Files:**
- Modify: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_ablation.py`:

```python
def test_fit_params_returns_league_and_sigma():
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                 home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                 park_factor=100.0, actual_total=8, actual_margin=0) for _ in range(20)]
    p = ablation.fit_params(rows, w_ra=0.0)
    assert p["w_ra"] == 0.0
    assert abs(p["league_rg"] - 4.4) < 0.05    # mean total 8 at L=4.4 (mu_total=8)
    assert p["sigma_team"] == 0.0              # μ predicts actuals exactly


def test_select_w_ra_recovers_signal():
    # Construct rows where actual total tracks away RA: higher away RA → higher actual total.
    # Baseline (w=0) μ ignores RA → residuals; blending RA in reduces residuals → σ drops.
    rows = []
    for ra, tot in [(3.0, 6), (4.0, 8), (5.0, 10), (6.0, 12)]:
        for _ in range(10):
            rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                             home_starter_fip=4.0, away_starter_fip=4.0,
                             home_bullpen_era=4.0, away_bullpen_era=4.0,
                             away_ra_recent=ra, away_ra_season=ra,
                             home_ra_recent=ra, home_ra_season=ra,
                             park_factor=100.0, actual_total=tot, actual_margin=0))
    w_star, table = ablation.select_w_ra(rows, ablation.W_RA_GRID)
    assert w_star > 0.0                         # RA signal present → nonzero weight chosen
    # σ at w_star should be <= σ at w=0
    sig0 = dict(table)[0.0]
    assert dict(table)[w_star] <= sig0


def test_select_w_ra_rejects_noise():
    # RA identical across all rows (no signal) → w=0 is as good as any → argmin picks 0.0.
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                 home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                 park_factor=100.0, actual_total=8 + (i % 3), actual_margin=0) for i in range(30)]
    w_star, table = ablation.select_w_ra(rows, ablation.W_RA_GRID)
    assert w_star == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_ablation.py -k "fit_params or select_w_ra" -q`
Expected: FAIL with `AttributeError: module 'ablation' has no attribute 'fit_params'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/ablation.py`:

```python
def fit_params(rows: list, w_ra: float) -> dict:
    """在固定 w_ra 下,mean-match 出 league_rg、殘差 MLE 出 sigma_team。"""
    mu_fn = lambda r, L: recompute_mu_ra(r, L, w_ra)
    L = fit_config.fit_league_rg(rows, mu_fn=mu_fn)
    s = fit_config.fit_sigma_team(rows, L, mu_fn=mu_fn)
    return {"w_ra": w_ra, "league_rg": L, "sigma_team": s}


def select_w_ra(rows: list, grid: list) -> tuple:
    """回 (w_ra*, [(w, sigma_train)...])。w_ra* = argmin σ_train(訓練得分殘差最小)。
    平手時偏好較小的 w(0 優先,符合『沒幫助就不加』)。"""
    table = []
    for w in grid:
        table.append((w, fit_params(rows, w)["sigma_team"]))
    w_star = min(table, key=lambda t: (t[1], t[0]))[0]
    return w_star, table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_ablation.py -k "fit_params or select_w_ra" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "feat(ablation): fit_params + select_w_ra (train-set score-fit)"
```

---

## Task 5: `eval_logloss`

**Files:**
- Modify: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_ablation.py`:

```python
def test_eval_logloss_returns_per_bet_arrays():
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
             park_factor=100.0, actual_total=9, actual_margin=3, has_odds=True,
             rl_home_point=-1.5, rl_home_no_vig=0.40, total_line=8.5, over_no_vig=0.48)
    out = ablation.eval_logloss([r], {"w_ra": 0.0, "league_rg": 4.4, "sigma_team": 3.0})
    assert len(out["rl"]) == 1 and out["rl"][0] > 0
    assert len(out["ou"]) == 1 and out["ou"][0] > 0
    assert len(out["market_rl"]) == 1 and len(out["market_ou"]) == 1


def test_eval_logloss_skips_push_and_no_odds():
    r_push = _row(has_odds=True, rl_home_point=-1.5, rl_home_no_vig=0.4,
                  total_line=9.0, over_no_vig=0.48, actual_total=9, actual_margin=2)
    r_noodds = _row(has_odds=False)
    out = ablation.eval_logloss([r_push, r_noodds], {"w_ra": 0.0, "league_rg": 4.4, "sigma_team": 3.0})
    assert len(out["ou"]) == 0      # push excluded
    assert len(out["rl"]) == 1      # push row keeps RL; no-odds row excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_ablation.py -k eval_logloss -q`
Expected: FAIL with `AttributeError: module 'ablation' has no attribute 'eval_logloss'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/ablation.py`:

```python
def _clamp(p: float, eps: float = 1e-9) -> float:
    return min(max(p, eps), 1 - eps)


def _ll(p: float, y: float) -> float:
    p = _clamp(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def eval_logloss(rows: list, params: dict) -> dict:
    """每注 log-loss 陣列(model 與 market)。只取有盤口者;O-U 排除 push。"""
    sigma = params["sigma_team"] * math.sqrt(2)
    L, w = params["league_rg"], params["w_ra"]
    out = {"rl": [], "ou": [], "market_rl": [], "market_ou": []}
    for r in rows:
        if not r["has_odds"] or r["rl_home_point"] is None:
            continue
        mt, mm = recompute_mu_ra(r, L, w)
        p_cov = run_model.cover_prob_home(mm, r["rl_home_point"], sigma=sigma)
        y = 1.0 if r["actual_margin"] > -r["rl_home_point"] else 0.0
        out["rl"].append(_ll(p_cov, y))
        out["market_rl"].append(_ll(r["rl_home_no_vig"], y))
        if r["total_line"] is not None and r["actual_total"] != r["total_line"]:
            p_ov = run_model.over_prob(mt, r["total_line"], sigma=sigma)
            yo = 1.0 if r["actual_total"] > r["total_line"] else 0.0
            out["ou"].append(_ll(p_ov, yo))
            out["market_ou"].append(_ll(r["over_no_vig"], yo))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_ablation.py -k eval_logloss -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "feat(ablation): eval_logloss (per-bet model+market log-loss)"
```

---

## Task 6: `ablate_ra` (orchestrate + verdict)

**Files:**
- Modify: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_ablation.py`:

```python
import math as _m


def _odds_row(away_ra, total, margin):
    # May-style test row with odds. Line fixed; actual total/margin vary with away RA.
    return _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                away_ra_recent=away_ra, away_ra_season=away_ra, home_ra_recent=4.4, home_ra_season=4.4,
                park_factor=100.0, actual_total=total, actual_margin=margin, has_odds=True,
                rl_home_point=-1.5, rl_home_no_vig=0.41, total_line=8.5, over_no_vig=0.50)


def test_ablate_ra_structure_and_keys():
    train = [_row(away_ra_recent=4.4, away_ra_season=4.4, actual_total=8, actual_margin=0) for _ in range(30)]
    test = [_odds_row(4.4, 9, 1) for _ in range(20)]
    out = ablation.ablate_ra(train, test, ablation.W_RA_GRID)
    for k in ("w_ra_star", "baseline", "candidate", "pooled_improve", "pooled_se",
              "accept", "gap_baseline", "gap_candidate"):
        assert k in out
    assert out["baseline"]["w_ra"] == 0.0
    # verdict is a bool
    assert isinstance(out["accept"], bool)


def test_ablate_ra_accepts_when_candidate_clearly_better():
    # Train: actual total tracks away RA strongly → RA reduces train σ → w_ra*>0.
    train = []
    for ra, tot in [(2.5, 5), (4.0, 8), (5.5, 11), (7.0, 14)]:
        for _ in range(15):
            train.append(_row(away_ra_recent=ra, away_ra_season=ra, home_ra_recent=ra, home_ra_season=ra,
                              actual_total=tot, actual_margin=0))
    # Test (May): same RA→total relationship, with odds; line 8.5 fixed.
    test = []
    for ra, tot in [(2.5, 5), (4.0, 8), (5.5, 11), (7.0, 14)]:
        for _ in range(15):
            test.append(_odds_row(ra, tot, 1))
    out = ablation.ablate_ra(train, test, ablation.W_RA_GRID)
    assert out["w_ra_star"] > 0.0
    assert out["candidate"]["pooled_ll"] < out["baseline"]["pooled_ll"]   # RA helps OOS
    assert out["accept"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_ablation.py -k ablate_ra -q`
Expected: FAIL with `AttributeError: module 'ablation' has no attribute 'ablate_ra'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/ablation.py`:

```python
def _pooled(ev: dict) -> list:
    return ev["rl"] + ev["ou"]


def _pooled_market(ev: dict) -> list:
    return ev["market_rl"] + ev["market_ou"]


def _mean(xs: list):
    return sum(xs) / len(xs) if xs else None


def ablate_ra(train_rows: list, test_rows: list, grid: list) -> dict:
    """baseline(w=0) vs candidate(w*) 的 OOS 比較。accept = OOS pooled log-loss 改善 > 1 SE。"""
    w_star, train_table = select_w_ra(train_rows, grid)
    p_base = fit_params(train_rows, 0.0)
    p_cand = fit_params(train_rows, w_star)

    ev_base = eval_logloss(test_rows, p_base)
    ev_cand = eval_logloss(test_rows, p_cand)

    base_pool = _pooled(ev_base)
    cand_pool = _pooled(ev_cand)
    mkt_pool = _pooled_market(ev_base)   # market 與 model 無關,base/cand 相同

    diffs = [b - c for b, c in zip(base_pool, cand_pool)]   # >0 表示 candidate 較好
    n = len(diffs)
    improve = _mean(diffs) or 0.0
    if n > 1:
        mean_d = improve
        var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")

    accept = (w_star > 0.0) and (improve > se)   # 改善為正且超過 1 SE

    def _summ(p, ev):
        return {"w_ra": p["w_ra"], "league_rg": p["league_rg"], "sigma_team": p["sigma_team"],
                "rl_ll": _mean(ev["rl"]), "ou_ll": _mean(ev["ou"]), "pooled_ll": _mean(_pooled(ev))}

    return {
        "w_ra_star": w_star,
        "train_table": train_table,
        "baseline": _summ(p_base, ev_base),
        "candidate": _summ(p_cand, ev_cand),
        "pooled_improve": improve,
        "pooled_se": se,
        "accept": accept,
        "market_pooled_ll": _mean(mkt_pool),
        "gap_baseline": (_mean(base_pool) - _mean(mkt_pool)) if mkt_pool else None,
        "gap_candidate": (_mean(cand_pool) - _mean(mkt_pool)) if mkt_pool else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_ablation.py -k ablate_ra -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "feat(ablation): ablate_ra orchestration + accept verdict"
```

---

## Task 7: CLI `main` (report + record file)

**Files:**
- Modify: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_ablation.py`:

```python
def test_render_report_contains_verdict_and_numbers():
    result = {
        "w_ra_star": 0.25, "train_table": [(0.0, 3.5), (0.25, 3.4)],
        "baseline": {"w_ra": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                     "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
        "candidate": {"w_ra": 0.25, "league_rg": 4.1, "sigma_team": 3.40,
                      "rl_ll": 0.68, "ou_ll": 0.69, "pooled_ll": 0.685},
        "pooled_improve": 0.010, "pooled_se": 0.004, "accept": True,
        "market_pooled_ll": 0.690, "gap_baseline": 0.005, "gap_candidate": -0.005,
    }
    text = ablation.render_report(result, train_n=468, test_n=292)
    assert "w_ra*" in text and "0.25" in text
    assert "ACCEPT" in text                       # verdict surfaced
    assert "0.685" in text                        # candidate pooled ll
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_ablation.py -k render_report -q`
Expected: FAIL with `AttributeError: module 'ablation' has no attribute 'render_report'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/ablation.py`:

```python
def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render_report(result: dict, train_n: int, test_n: int) -> str:
    b, c = result["baseline"], result["candidate"]
    verdict = "ACCEPT" if result["accept"] else "REJECT"
    lines = [
        "# RA-defense ablation — 2026 (train Mar–Apr → test May)",
        "",
        f"訓練={train_n} 場  測試(有盤口)={test_n} 注場",
        f"選出 w_ra* = {result['w_ra_star']}",
        "",
        "| 模型 | w_ra | league_rg | sigma_team | RL ll | OU ll | pooled ll |",
        "|------|------|-----------|------------|-------|-------|-----------|",
        f"| baseline | {b['w_ra']} | {b['league_rg']} | {b['sigma_team']} | {_f(b['rl_ll'])} | {_f(b['ou_ll'])} | {_f(b['pooled_ll'])} |",
        f"| candidate | {c['w_ra']} | {c['league_rg']} | {c['sigma_team']} | {_f(c['rl_ll'])} | {_f(c['ou_ll'])} | {_f(c['pooled_ll'])} |",
        "",
        f"OOS pooled 改善(baseline − candidate)= {_f(result['pooled_improve'])} ± {_f(result['pooled_se'])} (1 SE)",
        f"**判決:{verdict}**(接受條件:改善 > 1 SE)",
        "",
        f"離市場差距(pooled ll − market {_f(result['market_pooled_ll'])}):"
        f" baseline {_f(result['gap_baseline'])} → candidate {_f(result['gap_candidate'])}",
        "",
        "> 北極星=差距≤0(打敗市場)。此判決僅決定 RA 是否值得進模型;"
        "baking 進 config/run_model 是另一個決定。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="RA-defense 特徵 ablation(read-only)")
    p.add_argument("--train", default="2026-03,2026-04", help="逗號分隔 YYYY-MM")
    p.add_argument("--test", default="2026-05", help="逗號分隔 YYYY-MM(取有盤口者)")
    args = p.parse_args(argv)

    train_rows = [r for r in fit_config.load_fit_rows(set(args.train.split(",")))]
    test_rows = [r for r in fit_config.load_fit_rows(set(args.test.split(","))) if r["has_odds"]]
    if not train_rows or not test_rows:
        print("資料不足:確認三~五月已 backfill + fetch_results。", file=sys.stderr)
        return 1

    result = ablate_ra(train_rows, test_rows, W_RA_GRID)
    report = render_report(result, train_n=len(train_rows), test_n=len(test_rows))
    print(report)
    out_path = SKILL_ROOT / "analysis-data" / "backtest" / "ablation-ra-2026.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[record] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_ablation.py -q`
Expected: PASS (all ablation tests).

- [ ] **Step 5: Full suite**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 6: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "feat(ablation): CLI report + record file"
```

---

## Task 8: Execute the ablation (operational, not TDD)

**Files:** none (operational). Reads existing frozen Mar–May data; writes `analysis-data/backtest/ablation-ra-2026.md`. No model mutation.

- [ ] **Step 1: Full suite green**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 2: Run the ablation**

Run: `python scripts/ablation.py`
Expected: prints the baseline-vs-candidate table + verdict; writes the record file. Fast (no network — all from frozen `features.json` + `result.json`).

- [ ] **Step 3: Read the verdict**

Inspect output: fitted `w_ra*`, baseline vs candidate OOS log-loss (RL / O-U / pooled), `pooled_improve ± SE`, ACCEPT/REJECT, and gap-to-market for baseline → candidate.

- [ ] **Step 4: Sanity checks**

Confirm: baseline `pooled_ll` ≈ the known ~0.69 (ln 2) from the calibration work (sanity that the harness reproduces the established baseline); `w_ra*` is in `[0, 0.8]`; if `w_ra*==0`, verdict must be REJECT; SE is finite and positive. If baseline `pooled_ll` is wildly off ~0.69, STOP — the harness disagrees with the prior backtest and must be debugged before trusting the verdict.

- [ ] **Step 5: Report to user**

Summarize: did RA pass OOS? By how much vs noise? Did it narrow the gap to market? **Do not** modify `config.py`/`run_model.py`. If RA passed, the next step (baking it into the live model) is a separate decision/plan for the user.

> **Known limitations (document, do not fix here):** training objective is score-residual (odds-free), a proxy for bet calibration; pooled RL+O-U SE treats correlated bets as independent (mild); one month of OOS test is thin, so a borderline verdict should be treated as "not proven," consistent with the minimal-first discipline.

---

## Self-Review

**Spec coverage:** RA blend with one knob `w_ra`, `w_ra=0`≡baseline (Task 3) ✓; train=Mar+Apr / test=May-with-odds, strict separation (Task 7 `main`) ✓; odds-free training objective = min σ on train (Task 4 `select_w_ra`) ✓; OOS market-relative per-bet log-loss (Task 5) ✓; accept gate = OOS pooled improvement > 1 SE, north-star gap-to-market reported (Task 6) ✓; reuse `fit_league_rg`/`fit_sigma_team` via `mu_fn` (Task 2) ✓; load RA fields (Task 1) ✓; model-read-only, no `--write` (Task 7 `main`, Task 8) ✓; TDD with signal/noise synthetic rows (Tasks 3–7) ✓; execute + sanity vs ~0.69 baseline (Task 8) ✓.

**Placeholder scan:** none — every code/test step is complete.

**Type/name consistency:** row dict gains `home_ra_recent/season`, `away_ra_recent/season` (Task 1) and is read by `recompute_mu_ra` (Task 3). `params` dict `{w_ra, league_rg, sigma_team}` produced by `fit_params` (Task 4), consumed by `eval_logloss` (Task 5) and `render_report` (Task 7). `select_w_ra` returns `(w_ra*, table)` where table is `[(w, sigma)]` — consumed in tests (Task 4) and stored as `train_table` (Task 6). `ablate_ra` result keys (`w_ra_star`, `baseline`, `candidate`, `pooled_improve`, `pooled_se`, `accept`, `market_pooled_ll`, `gap_baseline`, `gap_candidate`, `train_table`) defined in Task 6 and consumed by `render_report` (Task 7) — consistent. `fit_league_rg(rows, ..., mu_fn=None)` / `fit_sigma_team(rows, league_rg, mu_fn=None)` signatures (Task 2) match the `mu_fn=lambda r,L: recompute_mu_ra(r,L,w_ra)` call in `fit_params` (Task 4).
