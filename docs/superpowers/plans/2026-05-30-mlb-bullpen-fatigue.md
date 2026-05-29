# MLB Bullpen Short-Rest Fatigue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test bullpen short-rest fatigue (trailing-2-day relief IP) two ways — a μ-penalty via the (generalized) ablation harness, and a pre-registered heavy-usage tail bet-filter judged by hit-rate + CLV — without touching the live model.

**Architecture:** The fatigue signal is computed live from the already-cached relief index (zero new fetch, leakage-free), team_id resolved from the matchup abbreviations. Path A reuses the ablation harness, which we first generalize to accept an injectable μ-recompute fn (RA unchanged). Path B reuses the CLV snapshot finder to split the model's positive-edge bets by tail vs non-tail. Read-only; expect REJECT/inconclusive; the durable win is a feature-agnostic harness.

**Tech Stack:** Python 3, `pytest`, stdlib. Spec: `docs/superpowers/specs/2026-05-30-mlb-bullpen-fatigue-design.md`. Reuses `scripts/bullpen.py`, `scripts/ablation.py`, `scripts/fit_config.py`, `scripts/lib/clv.py`, `scripts/lib/load.py`, `scripts/_team_resolver.py`.

**Commit policy:** Repo owner commits manually. Treat `Commit` steps as checkpoints — run only if asked; otherwise leave staged.

---

## File Structure

- **`scripts/bullpen.py`** — ADD `relief_ip_last_k`.
- **`scripts/ablation.py`** — generalize core (inject `recompute`, rename `w_ra`→`w` / `w_ra_star`→`w_star`, `select_w_ra`→`select_w`, `ablate_ra`→`ablate`); `recompute_mu_ra` + RA behavior unchanged.
- **`scripts/fatigue.py`** (new) — `team_ids_from_matchup`, `add_fatigue_to_rows`, `recompute_mu_fatigue`, `FAT_W_GRID`, `TAIL_IP`, `fatigue_filter_report`, `render_report`, `main`.
- **Tests** — extend `test_bullpen.py`, update `test_ablation.py` (renames), new `test_fatigue.py`.

**Row sources:** Path A uses `fit_config.load_fit_rows` rows (model inputs: rs/fip/bullpen/pf + odds + actuals). Path B uses `lib.load.build_dataframe_for_month` records (edges `home_rl_pp`/`over_pp` + team names + actuals + odds no-vig). Both get enriched with `home_fat_ip`/`away_fat_ip` by the same `add_fatigue_to_rows`.

---

## Task 1: `bullpen.relief_ip_last_k`

**Files:**
- Modify: `scripts/bullpen.py`
- Test: `scripts/tests/test_bullpen.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_bullpen.py`:

```python
def test_relief_ip_last_k_sums_window():
    idx = {"141": [{"date": "2026-05-01", "er": 0, "ip": 3.0},
                   {"date": "2026-05-02", "er": 0, "ip": 2.0},
                   {"date": "2026-05-04", "er": 0, "ip": 5.0}]}
    # as_of 05-04, k=2 → window [05-02, 05-04) → only 05-02 (2.0); 05-01 outside, 05-04 excluded
    assert bullpen.relief_ip_last_k(141, 2026, "2026-05-04", k=2, index=idx) == 2.0
    # k=3 → [05-01, 05-04) → 3.0 + 2.0 = 5.0
    assert bullpen.relief_ip_last_k(141, 2026, "2026-05-04", k=3, index=idx) == 5.0
    # unknown team → 0.0
    assert bullpen.relief_ip_last_k(999, 2026, "2026-05-04", k=2, index=idx) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_bullpen.py -k last_k -q`
Expected: FAIL with `AttributeError: module 'bullpen' has no attribute 'relief_ip_last_k'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/bullpen.py`, add `from datetime import datetime, timedelta` to the imports, then add at end:

```python
def relief_ip_last_k(team_id, year, as_of, k=2, cache_dir=DEFAULT_CACHE_DIR, index=None) -> float:
    """team_id 在 as_of 前 k 天(不含 as_of 當日)的後援投球局數總和。index 可注入避免重載。"""
    if index is None:
        index = load_or_build_index(year, needed_through=as_of, cache_dir=cache_dir)
    a = datetime.strptime(as_of, "%Y-%m-%d")
    lo = a - timedelta(days=k)
    tot = 0.0
    for e in index.get(str(team_id), []):
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        if lo <= d < a:
            tot += e["ip"]
    return round(tot, 4)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_bullpen.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/bullpen.py scripts/tests/test_bullpen.py
git commit -m "feat(bullpen): relief_ip_last_k (trailing-k-day relief IP from cached index)"
```

---

## Task 2: Generalize the ablation harness

**Files:**
- Modify: `scripts/ablation.py`
- Test: `scripts/tests/test_ablation.py`

**Context:** make the μ-recompute fn injectable and rename RA-specific keys to feature-neutral. RA math is unchanged; only signatures/key names change.

- [ ] **Step 1: Update the tests first (they encode the new API)**

In `scripts/tests/test_ablation.py`, apply these exact replacements:
- `ablation.fit_params(rows, w_ra=0.0)` → `ablation.fit_params(rows, 0.0)`
- `p["w_ra"]` → `p["w"]`
- `ablation.select_w_ra(` → `ablation.select_w(`
- `ablation.ablate_ra(` → `ablation.ablate(`
- `out["w_ra_star"]` → `out["w_star"]`
- `out["baseline"]["w_ra"]` → `out["baseline"]["w"]`
- In `test_render_report_contains_verdict_and_numbers`: change the result dict keys `"w_ra_star"` → `"w_star"`, and inside `baseline`/`candidate` `"w_ra"` → `"w"`; change the assertion `assert "w_ra*" in text` → `assert "w*" in text`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_ablation.py -q`
Expected: FAIL (e.g. `fit_params() got an unexpected keyword`/`KeyError: 'w'`/`AttributeError: ... 'select_w'`).

- [ ] **Step 3: Rewrite the four functions + render in `scripts/ablation.py`**

Replace `fit_params`, `select_w_ra`, `eval_logloss`, `ablate_ra` with:

```python
def fit_params(rows: list, w: float, recompute=recompute_mu_ra) -> dict:
    """在固定 w 下,mean-match league_rg、殘差 MLE sigma_team。recompute 可換特徵(預設 RA)。"""
    mu_fn = lambda r, L: recompute(r, L, w)
    L = fit_config.fit_league_rg(rows, mu_fn=mu_fn)
    s = fit_config.fit_sigma_team(rows, L, mu_fn=mu_fn)
    return {"w": w, "league_rg": L, "sigma_team": s}


def select_w(rows: list, grid: list, recompute=recompute_mu_ra) -> tuple:
    """回 (w*, [(w, sigma_train)...])。w* = argmin σ_train,平手偏好較小 w。"""
    table = [(w, fit_params(rows, w, recompute)["sigma_team"]) for w in grid]
    w_star = min(table, key=lambda t: (t[1], t[0]))[0]
    return w_star, table


def eval_logloss(rows: list, params: dict, recompute=recompute_mu_ra) -> dict:
    """每注 log-loss 陣列(model 與 market)。只取有盤口者;O-U 排除 push。"""
    sigma = params["sigma_team"] * math.sqrt(2)
    L, w = params["league_rg"], params["w"]
    out = {"rl": [], "ou": [], "market_rl": [], "market_ou": []}
    for r in rows:
        if not r["has_odds"] or r["rl_home_point"] is None:
            continue
        mt, mm = recompute(r, L, w)
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


def ablate(train_rows: list, test_rows: list, grid: list, recompute=recompute_mu_ra) -> dict:
    """baseline(w=0) vs candidate(w*) OOS 比較。accept = OOS pooled log-loss 改善 > 1 SE。"""
    w_star, train_table = select_w(train_rows, grid, recompute)
    p_base = fit_params(train_rows, 0.0, recompute)
    p_cand = fit_params(train_rows, w_star, recompute)

    ev_base = eval_logloss(test_rows, p_base, recompute)
    ev_cand = eval_logloss(test_rows, p_cand, recompute)

    base_pool = _pooled(ev_base)
    cand_pool = _pooled(ev_cand)
    mkt_pool = _pooled_market(ev_base)

    diffs = [b - c for b, c in zip(base_pool, cand_pool)]
    n = len(diffs)
    improve = _mean(diffs) or 0.0
    if n > 1:
        var = sum((d - improve) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")

    accept = (w_star > 0.0) and (improve > se)

    def _summ(p, ev):
        return {"w": p["w"], "league_rg": p["league_rg"], "sigma_team": p["sigma_team"],
                "rl_ll": _mean(ev["rl"]), "ou_ll": _mean(ev["ou"]), "pooled_ll": _mean(_pooled(ev))}

    return {
        "w_star": w_star, "train_table": train_table,
        "baseline": _summ(p_base, ev_base), "candidate": _summ(p_cand, ev_cand),
        "pooled_improve": improve, "pooled_se": se, "accept": accept,
        "market_pooled_ll": _mean(mkt_pool),
        "gap_baseline": (_mean(base_pool) - _mean(mkt_pool)) if mkt_pool else None,
        "gap_candidate": (_mean(cand_pool) - _mean(mkt_pool)) if mkt_pool else None,
    }
```

In `render_report`, replace the three RA-named reads: `result['w_ra_star']` → `result['w_star']`; `b['w_ra']` → `b['w']`; `c['w_ra']` → `c['w']`; and the literal `選出 w_ra* =` → `選出 w* =`.

In `main`, replace `result = ablate_ra(train_rows, test_rows, W_RA_GRID)` → `result = ablate(train_rows, test_rows, W_RA_GRID)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_ablation.py scripts/tests/test_fit_config.py -q`
Expected: PASS (all).

- [ ] **Step 5: Re-verify RA still REJECTs (behavior unchanged)**

Run: `python scripts/ablation.py 2>&1 | grep -E "w\* =|判決"`
Expected: `選出 w* = 0.0` and `判決:REJECT` (same RA result as before the refactor).

- [ ] **Step 6: Commit** (checkpoint)

```bash
git add scripts/ablation.py scripts/tests/test_ablation.py
git commit -m "refactor(ablation): feature-agnostic core (injectable recompute, neutral keys)"
```

---

## Task 3: `fatigue.py` — signal join + μ recompute

**Files:**
- Create: `scripts/fatigue.py`
- Test: `scripts/tests/test_fatigue.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/tests/test_fatigue.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fatigue
import fit_config


def _row(**kw):
    base = dict(date="2026-05-04", matchup="A@H",
                home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                park_factor=100.0, home_fat_ip=0.0, away_fat_ip=0.0)
    base.update(kw)
    return base


def test_recompute_mu_fatigue_w0_equals_baseline():
    r = _row(home_fat_ip=12.0, away_fat_ip=8.0)
    base = fit_config.recompute_mu(r, league_rg=4.4)
    f0 = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.0)
    assert abs(f0[0] - base[0]) < 1e-9 and abs(f0[1] - base[1]) < 1e-9


def test_recompute_mu_fatigue_tired_pen_raises_total():
    # away pen tired (high fat_ip) → away defense worse → home scores more → total up
    r = _row(away_fat_ip=12.0, home_fat_ip=0.0)
    t0, _ = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.0)
    t1, _ = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.05)
    assert t1 > t0


def test_team_ids_from_matchup_strips_doubleheader(monkeypatch):
    monkeypatch.setattr(fatigue, "resolve_team_id", lambda a: {"A": 1, "H": 2}[a])
    assert fatigue.team_ids_from_matchup("A@H") == (1, 2)
    assert fatigue.team_ids_from_matchup("A@H-G2") == (1, 2)   # suffix stripped


def test_add_fatigue_to_rows_enriches(monkeypatch):
    monkeypatch.setattr(fatigue, "resolve_team_id", lambda a: {"A": 100, "H": 200}[a])
    idx = {"100": [{"date": "2026-05-02", "er": 0, "ip": 4.0}],
           "200": [{"date": "2026-05-03", "er": 0, "ip": 13.0}]}
    rows = [dict(date="2026-05-04", matchup="A@H")]
    out = fatigue.add_fatigue_to_rows(rows, 2026, k=2, index=idx)
    assert out[0]["away_fat_ip"] == 4.0    # team A (away), 05-02 in [05-02,05-04)
    assert out[0]["home_fat_ip"] == 13.0   # team H (home), 05-03 in window
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_fatigue.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fatigue'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/fatigue.py`:

```python
#!/usr/bin/env python3
"""牛棚短休疲勞:近 2 天後援 IP。Path A=μ 懲罰(經一般化 ablation 台子),
Path B=重用尾巴過濾器(正 edge 注的命中率+CLV,tail vs non-tail)。對線上模型唯讀。

用法:
  python scripts/fatigue.py
  python scripts/fatigue.py --train 2026-03,2026-04 --test 2026-05
"""
import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bullpen
import fit_config
from _team_resolver import resolve_team_id
from lib.closing_line import _parse_iso_utc, find_entry_close_snapshots
from lib import clv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

FAT_W_GRID = [round(0.02 * i, 2) for i in range(16)]   # 0.00 .. 0.30
TAIL_IP = 12.0
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"


def team_ids_from_matchup(matchup):
    """'AWAY@HOME'(可含 -G1/-G2)→ (away_id, home_id)。失敗 → (None, None)。"""
    try:
        away_abbr, _, rest = matchup.partition("@")
        home_abbr = rest.split("-")[0]
        return resolve_team_id(away_abbr.strip()), resolve_team_id(home_abbr.strip())
    except Exception:
        return None, None


def add_fatigue_to_rows(rows, year, k=2, cache_dir=bullpen.DEFAULT_CACHE_DIR, index=None):
    """每列加 home_fat_ip / away_fat_ip(近 k 天後援 IP)。index 預設讀快取(不重建)。"""
    if index is None:
        dates = [r["date"] for r in rows if r.get("date")]
        through = max(dates) if dates else f"{year}-05-25"
        index = bullpen.load_or_build_index(year, needed_through=through, cache_dir=cache_dir)
    out = []
    for r in rows:
        away_id, home_id = team_ids_from_matchup(r.get("matchup", ""))
        d = r.get("date")
        hf = bullpen.relief_ip_last_k(home_id, year, d, k, index=index) if (home_id and d) else 0.0
        af = bullpen.relief_ip_last_k(away_id, year, d, k, index=index) if (away_id and d) else 0.0
        r2 = dict(r)
        r2["home_fat_ip"] = hf
        r2["away_fat_ip"] = af
        out.append(r2)
    return out


def recompute_mu_fatigue(row, league_rg, w_fat):
    """μ with 疲勞懲罰:該隊 bullpen_era += w_fat × fat_ip。w_fat=0 ≡ fit_config.recompute_mu。"""
    r2 = dict(row)
    r2["home_bullpen_era"] = row["home_bullpen_era"] + w_fat * row.get("home_fat_ip", 0.0)
    r2["away_bullpen_era"] = row["away_bullpen_era"] + w_fat * row.get("away_fat_ip", 0.0)
    return fit_config.recompute_mu(r2, league_rg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_fatigue.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fatigue.py scripts/tests/test_fatigue.py
git commit -m "feat(fatigue): trailing-2d relief IP signal + mu penalty recompute"
```

---

## Task 4: `fatigue.py` — Path B tail filter

**Files:**
- Modify: `scripts/fatigue.py`
- Test: `scripts/tests/test_fatigue.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_fatigue.py`:

```python
import json


def _write_two_snaps(tmp_path, over_entry, over_close):
    for slot, snap_utc, over_nv in [("12-00-ET", "2026-05-04T16:00:00Z", over_entry),
                                     ("18-00-ET", "2026-05-04T21:00:00Z", over_close)]:
        data = {"snapshot_time_utc": snap_utc, "snapshot_time_et": f"2026-05-04 {slot[:2]}:00 ET",
                "games": [{"home_team": "Hh", "away_team": "Aa", "game_date_et": "2026-05-04",
                           "commence_utc": "2026-05-04T22:00:00Z",
                           "bookmakers": {"pinnacle": {
                               "ml": {"Aa": {"no_vig_pct": 39.0}, "Hh": {"no_vig_pct": 61.0}},
                               "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                                      "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                               "rl": {"Hh": {"point": -1.5, "no_vig_pct": 40.0},
                                      "Aa": {"point": 1.5, "no_vig_pct": 60.0}}}}}]}
        (tmp_path / f"2026-05-04_{slot}.json").write_text(json.dumps(data), encoding="utf-8")


def _drow(**kw):
    base = dict(date="2026-05-04", matchup="Aa@Hh", home_team="Hh", away_team="Aa",
                home_rl_pp=None, over_pp=2.0, rl_home_point=-1.5, total_line=8.5,
                actual_margin=1, actual_total=10, home_fat_ip=0.0, away_fat_ip=0.0)
    base.update(kw)
    return base


def test_fatigue_filter_splits_tail_and_reports(tmp_path):
    _write_two_snaps(tmp_path, over_entry=50.0, over_close=55.0)  # over CLV +5 in over direction
    rows = [
        _drow(home_fat_ip=13.0, over_pp=2.0, actual_total=10),  # tail (pen≥12), over edge, over hit (10>8.5)
        _drow(away_fat_ip=4.0, over_pp=2.0, actual_total=7),    # non-tail, over edge, over miss (7<8.5)
    ]
    out = fatigue.fatigue_filter_report(rows, tmp_path)
    assert out["tail"]["n"] == 1 and out["tail"]["hit_rate"] == 1.0
    assert out["non_tail"]["n"] == 1 and out["non_tail"]["hit_rate"] == 0.0
    assert out["tail"]["clv_mean"] == 5.0          # over no-vig rose 50→55
    assert out["tail"]["n_clv"] == 1


def test_fatigue_filter_skips_non_positive_edge_and_push():
    # over_pp<=0 → no over bet; home_rl_pp None → no rl bet → empty
    rows = [_drow(over_pp=-1.0, home_rl_pp=None)]
    out = fatigue.fatigue_filter_report(rows, "nonexistent_dir")
    assert out["tail"]["n"] == 0 and out["non_tail"]["n"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_fatigue.py -k filter -q`
Expected: FAIL with `AttributeError: module 'fatigue' has no attribute 'fatigue_filter_report'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fatigue.py`:

```python
def _edge_clv(row, snapshots_dir, market, side):
    """正 edge 方向的 CLV(close−entry pp)。無 headroom / 無盤 → None。"""
    entry, close = find_entry_close_snapshots(snapshots_dir, row.get("date"),
                                              row.get("home_team"), row.get("away_team"))
    if entry is None or close is None:
        return None
    e_ts = _parse_iso_utc(entry.get("snapshot_time_utc", ""))
    c_ts = _parse_iso_utc(close.get("snapshot_time_utc", ""))
    if e_ts is None or c_ts is None or not (e_ts < c_ts):
        return None
    return clv.clv_pp(entry, close, market, side)


def _bet_summ(bets):
    n = len(bets)
    if n == 0:
        return {"n": 0, "hit_rate": None, "clv_mean": None, "n_clv": 0}
    hit = sum(1 for h, _ in bets if h) / n
    clvs = [c for _, c in bets if c is not None]
    return {"n": n, "hit_rate": round(hit, 3),
            "clv_mean": (round(sum(clvs) / len(clvs), 3) if clvs else None), "n_clv": len(clvs)}


def fatigue_filter_report(rows, snapshots_dir):
    """正 edge 注(home_rl_pp>0→主過盤;over_pp>0→Over)按尾巴(max fat_ip≥TAIL_IP)分組,
    各報 n / 命中率 / CLV。對齊 compute_edge_calibration 的正 edge 定義。"""
    tail, non = [], []
    for r in rows:
        bets = []
        e = r.get("home_rl_pp")
        if isinstance(e, (int, float)) and math.isfinite(e) and e > 0:
            mg, pt = r.get("actual_margin"), r.get("rl_home_point")
            if mg is not None and pt is not None:
                bets.append((mg > -pt, _edge_clv(r, snapshots_dir, "rl", "home")))
        o = r.get("over_pp")
        if isinstance(o, (int, float)) and math.isfinite(o) and o > 0:
            tt, ln = r.get("actual_total"), r.get("total_line")
            if tt is not None and ln is not None and tt != ln:
                bets.append((tt > ln, _edge_clv(r, snapshots_dir, "ou", "over")))
        is_tail = max(r.get("home_fat_ip", 0.0) or 0.0, r.get("away_fat_ip", 0.0) or 0.0) >= TAIL_IP
        (tail if is_tail else non).extend(bets)
    return {"tail": _bet_summ(tail), "non_tail": _bet_summ(non)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_fatigue.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/fatigue.py scripts/tests/test_fatigue.py
git commit -m "feat(fatigue): Path B tail filter (positive-edge hit-rate + CLV, tail vs non-tail)"
```

---

## Task 5: `fatigue.py` — report + CLI

**Files:**
- Modify: `scripts/fatigue.py`
- Test: `scripts/tests/test_fatigue.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fatigue.py`:

```python
def test_render_report_has_both_paths():
    path_a = {"w_star": 0.0,
              "baseline": {"w": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                           "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
              "candidate": {"w": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                            "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
              "pooled_improve": 0.0, "pooled_se": 0.004, "accept": False}
    path_b = {"tail": {"n": 18, "hit_rate": 0.5, "clv_mean": -0.2, "n_clv": 15},
              "non_tail": {"n": 240, "hit_rate": 0.49, "clv_mean": 0.0, "n_clv": 220}}
    text = fatigue.render_report(path_a, path_b, train_n=468, test_n=292, valid_n=258)
    assert "Path A" in text and "Path B" in text
    assert "REJECT" in text
    assert "18" in text          # tail n surfaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_fatigue.py -k render -q`
Expected: FAIL with `AttributeError: module 'fatigue' has no attribute 'render_report'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/fatigue.py`:

```python
def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _fmt_subset(name, s):
    return (f"- {name}:n={s['n']}｜命中率 {_pct(s['hit_rate'])}"
            f"｜CLV mean {_f(s['clv_mean'], 3)}pp(n_clv={s['n_clv']})")


def render_report(path_a, path_b, train_n, test_n, valid_n):
    b, c = path_a["baseline"], path_a["candidate"]
    verdict = "ACCEPT" if path_a["accept"] else "REJECT"
    lines = [
        "# 牛棚短休疲勞 ablation — 2026",
        "",
        f"## Path A:μ 懲罰(w_fat × 近2天後援IP 加進牛棚ERA)",
        f"_train Mar–Apr={train_n}｜test May(有盤口)={test_n}_",
        f"- w_fat* = {path_a['w_star']}",
        f"- pooled log-loss:baseline {_f(b['pooled_ll'])} → candidate {_f(c['pooled_ll'])}"
        f"(改善 {_f(path_a['pooled_improve'])} ± {_f(path_a['pooled_se'])} SE)",
        f"- **判決:{verdict}**(接受條件:OOS 改善 > 1 SE)",
        "",
        f"## Path B:尾巴過濾器(任一隊近2天後援IP ≥ {TAIL_IP})— 正 edge 注,valid={valid_n}",
        _fmt_subset("尾巴(tail)", path_b["tail"]),
        _fmt_subset("非尾巴", path_b["non_tail"]),
        "",
        "> 尾巴 n 小屬正常;命中率與 CLV 要同向且離噪音才算數,否則 inconclusive。",
        "> 對線上模型唯讀;此判決僅決定是否值得進一步,baking 是另一個決定。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    import ablation
    from lib.load import build_dataframe_for_month
    p = argparse.ArgumentParser(description="牛棚疲勞 ablation(read-only)")
    p.add_argument("--train", default="2026-03,2026-04")
    p.add_argument("--test", default="2026-05")
    args = p.parse_args(argv)
    year = int(args.test[:4])

    train = add_fatigue_to_rows(fit_config.load_fit_rows(set(args.train.split(","))), year)
    test = add_fatigue_to_rows(fit_config.load_fit_rows(set(args.test.split(","))), year)
    test_odds = [r for r in test if r["has_odds"]]
    if not train or not test_odds:
        print("資料不足:確認三~五月已 backfill + fetch_results。", file=sys.stderr)
        return 1
    path_a = ablation.ablate(train, test_odds, FAT_W_GRID, recompute=recompute_mu_fatigue)

    dfrows = add_fatigue_to_rows(build_dataframe_for_month(args.test).to_dict("records"), year)
    valid = [r for r in dfrows if not r["odds_missing"] and not r["result_missing"]]
    path_b = fatigue_filter_report(valid, SNAPSHOTS_DIR)

    report = render_report(path_a, path_b, len(train), len(test_odds), len(valid))
    print(report)
    out_path = SKILL_ROOT / "analysis-data" / "backtest" / "ablation-fatigue-2026.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[record] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_fatigue.py -q`
Expected: PASS (all).

- [ ] **Step 5: Full suite**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 6: Commit** (checkpoint)

```bash
git add scripts/fatigue.py scripts/tests/test_fatigue.py
git commit -m "feat(fatigue): both-path report + CLI"
```

---

## Task 6: Execute (operational, not TDD)

**Files:** none (operational). Writes `analysis-data/backtest/ablation-fatigue-2026.md`. No model mutation.

- [ ] **Step 1: Full suite green + RA still REJECTs**

Run: `python -m pytest scripts/tests -q` (expect all pass), then
Run: `python scripts/ablation.py 2>&1 | grep -E "w\* =|判決"` (expect `w* = 0.0`, `REJECT`).

- [ ] **Step 2: Run the fatigue test**

Run: `python scripts/fatigue.py`
Expected: prints Path A (w_fat*, baseline→candidate pooled ll, verdict — expect REJECT) and Path B (tail vs non-tail n / hit-rate / CLV); writes the record file. Reads frozen rows + cached index + snapshots only (no network).

- [ ] **Step 3: Read verdicts + sanity checks**

Confirm: Path A `w_fat*` plausibly `0.0` (REJECT), or if >0 the OOS improvement is checked vs 1 SE. Path B tail `n` is small (likely a few dozen bets) — if tiny, treat as **inconclusive**. Check tail vs non-tail: only a clear, consistent gap in BOTH hit-rate and CLV (beyond the small-n noise) is meaningful.

- [ ] **Step 4: Report to user**

Summarize honestly: Path A verdict; Path B tail-vs-non-tail (with n caveat); whether anything is actionable (almost certainly not). **Do not** modify `config.py`/`run_model.py`. If both paths somehow show a real consistent effect, baking it in is a separate decision/plan.

> **Known limitations (document, do not fix here):** tail subset is small (one season, ~5% of games) → low power; RL/OU fatigue asymmetry approximated by a game-level flag; signal computed live from the cache (no re-backfill), so it depends on the relief index covering the dates (it does, through 2026-05-25).

---

## Self-Review

**Spec coverage:** signal `relief_ip_last_k` from cached index, leakage-free (Task 1) ✓; Path A μ penalty `bullpen_era += w_fat·fat_ip`, w=0≡baseline, via generalized harness, expect REJECT (Tasks 2–3, 5) ✓; harness generalization = injectable recompute + neutral keys, RA unchanged + re-verified (Task 2) ✓; Path B pre-registered tail `≥12`, positive-edge per `compute_edge_calibration`, hit-rate + CLV tail vs non-tail, inconclusive-if-tiny (Tasks 4–5) ✓; team_id from matchup abbr (Task 3) ✓; row enrichment once, pure downstream (Task 3) ✓; read-only / no `--write` / no model mutation (Task 5–6) ✓; honest low-power note (Tasks 5–6) ✓. Starter co-condition, re-backfill, RL/OU asymmetry — correctly out of scope.

**Placeholder scan:** none — every code/test step is complete (Task 2's test edits are exact string replacements).

**Type/name consistency:** `relief_ip_last_k(team_id, year, as_of, k, cache_dir, index)` (Task 1) called by `add_fatigue_to_rows` (Task 3). Generalized ablation: `fit_params(rows, w, recompute)`, `select_w(rows, grid, recompute)`, `eval_logloss(rows, params, recompute)` (reads `params["w"]`), `ablate(train, test, grid, recompute)` returning `w_star` + `baseline/candidate{w,league_rg,sigma_team,rl_ll,ou_ll,pooled_ll}` (Task 2) — consumed by `fatigue.main` via `ablate(..., recompute=recompute_mu_fatigue)` and `render_report` reads `path_a["w_star"]`/`baseline["pooled_ll"]` (Task 5). `recompute_mu_fatigue(row, league_rg, w_fat)` (Task 3) matches the `recompute(r, L, w)` call shape in the generalized harness (Task 2). `fatigue_filter_report(rows, snapshots_dir) -> {tail, non_tail}` with `_bet_summ` keys `{n, hit_rate, clv_mean, n_clv}` (Task 4) consumed by `render_report._fmt_subset` (Task 5). `add_fatigue_to_rows` sets `home_fat_ip`/`away_fat_ip` (Task 3) read by `recompute_mu_fatigue` (Task 3) and `fatigue_filter_report` tail flag (Task 4).
