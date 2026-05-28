# MLB 極簡預測重構（Model C / team-level）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用結果型 team-level 模型(球隊平均得分 × 對方先發FIP+牛棚 / 聯盟 × PF)取代壞掉的 xwOBA×FIP 路線,輸出 RL + O/U + vs 市場 edge,由一支 orchestrator 統一流程,並凍結輸入+盤口供回測。

**Architecture:** `predict_game.py`(orchestrator,單場/`--all`)串 `fetch_inputs`(抓資料)→ `run_model`(純函數算期望得分+機率)→ `odds_compare`(no-vig + edge)→ `report`(凍結 features.json + 產 prediction.md)。所有先驗係數集中於 `config.py`。

**Tech Stack:** Python 3.13, stdlib `statistics.NormalDist`, `requests`, MLB Stats API, pytest。沿用既有 `_team_resolver` / `park_factors_lib` / `lib/closing_line` / `odds/fetch_odds` snapshot。

Spec: `docs/superpowers/specs/2026-05-29-mlb-minimal-prediction-rebuild-design.md`

**全程指令前綴:** `PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)`;測試在 `scripts/` 下跑 `$PYTHON -m pytest`。

> **實作備註(對 spec 的兩處細化,執行者請遵循):**
> 1. 輕量打線抓取**併進 `fetch_inputs.py`**(函式 `fetch_lineup_light`),故 `lineup_analyzer.py` **完全退役**(spec §10 原寫「瘦身」;改為退役 + 函式內聯,更貼合 orchestrator+模組架構,凍結行為不變)。
> 2. FIP 採標準含 HBP 版:`(13×HR + 3×(BB+HBP) − 2×K)/IP + C`(MLB API 有 `hitByPitch` 欄位)。

---

## Phase 0 — Scaffolding

### Task 0: `config.py` 先驗係數

**Files:**
- Create: `scripts/config.py`
- Test: `scripts/tests/test_config.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_config.py
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

def test_constants_present_and_typed():
    assert config.LEAGUE_RG == 4.4
    assert config.RECENT_W == 0.35
    assert config.SP_W == 0.6 and config.BP_W == 0.4
    assert config.SIGMA_TEAM == 3.0
    assert config.FIP_CONSTANT == 3.10
    assert config.RECENT_N == 10
    assert config.MIN_IP == 10
    # weights coherent
    assert math.isclose(config.SP_W + config.BP_W, 1.0)

def test_sigma_derived():
    assert math.isclose(config.SIGMA, 3.0 * math.sqrt(2), rel_tol=1e-9)
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'config'`）

- [ ] **Step 3: 實作 `scripts/config.py`**

```python
# scripts/config.py
"""所有預測先驗係數集中於此。回測後重新擬合只改這裡。"""
import math

LEAGUE_RG = 4.4        # 聯盟每場均分
RECENT_W = 0.35        # RS blend:近期權重(季 = 1 - RECENT_W)
SP_W = 0.6             # 期望得分 — 先發權重(約 6/9 局)
BP_W = 0.4             # 期望得分 — 牛棚權重(約 3/9 局)
SIGMA_TEAM = 3.0       # 單隊單場得分 SD(歷史先驗)
SIGMA = SIGMA_TEAM * math.sqrt(2)   # margin / total SD ≈ 4.24
FIP_CONSTANT = 3.10    # FIP 聯盟正規化常數
RECENT_N = 10          # 近期窗口場數
MIN_IP = 10            # 先發 IP 低於此 → FIP 不穩,用聯盟替代

def constants_snapshot() -> dict:
    """凍結進 features.json 的當下係數值(重現用)。"""
    return {
        "league_rg": LEAGUE_RG, "recent_w": RECENT_W,
        "sp_w": SP_W, "bp_w": BP_W, "sigma_team": SIGMA_TEAM,
        "fip_constant": FIP_CONSTANT, "recent_n": RECENT_N, "min_ip": MIN_IP,
    }
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_config.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/config.py scripts/tests/test_config.py
git commit -m "feat(predict): config.py 先驗係數集中"
```

---

## Phase 1 — `run_model.py`（數學核心，純函數）

### Task 1: 期望得分 `pitch_today` + `expected_runs`

**Files:**
- Create: `scripts/run_model.py`
- Test: `scripts/tests/test_run_model.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_run_model.py
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_model

def test_pitch_today():
    # 0.6*3.6 + 0.4*4.0 = 3.76
    assert math.isclose(run_model.pitch_today(3.6, 4.0), 3.76, rel_tol=1e-9)
    # 0.6*4.5 + 0.4*4.2 = 4.38
    assert math.isclose(run_model.pitch_today(4.5, 4.2), 4.38, rel_tol=1e-9)

def test_expected_runs_worked_example():
    mu_home, mu_away = run_model.expected_runs(
        home_rs=4.8, away_rs=4.2,
        home_pitch=4.38, away_pitch=3.76, pf=100,
    )
    assert round(mu_home, 2) == 4.10   # 4.8*3.76/4.4
    assert round(mu_away, 2) == 4.18   # 4.2*4.38/4.4

def test_park_factor_scales_both_sides():
    a = run_model.expected_runs(4.5, 4.5, 4.4, 4.4, pf=100)
    b = run_model.expected_runs(4.5, 4.5, 4.4, 4.4, pf=110)
    assert b[0] > a[0] and b[1] > a[1]
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_run_model.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'run_model'`）

- [ ] **Step 3: 實作 `run_model.py`（第一段）**

```python
# scripts/run_model.py
"""確定性預測:team-level 期望得分 → RL / O/U / ML 機率。純函數,零 I/O。"""
from statistics import NormalDist
import config

_N = NormalDist()


def pitch_today(starter_fip: float, bullpen_era: float) -> float:
    """今日防守力(每9局)= SP_W×先發FIP + BP_W×牛棚ERA。"""
    return config.SP_W * starter_fip + config.BP_W * bullpen_era


def expected_runs(home_rs: float, away_rs: float,
                  home_pitch: float, away_pitch: float,
                  pf: float) -> tuple[float, float]:
    """期望得分。home_pitch/away_pitch 為各隊今日防守力(pitch_today 輸出)。

    μ_home = 主隊RS × 對方(away)防守力 / 聯盟 × PF/100
    """
    pf_mult = pf / 100.0
    mu_home = home_rs * away_pitch / config.LEAGUE_RG * pf_mult
    mu_away = away_rs * home_pitch / config.LEAGUE_RG * pf_mult
    return mu_home, mu_away
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_run_model.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model.py scripts/tests/test_run_model.py
git commit -m "feat(predict): run_model 期望得分公式"
```

### Task 2: 機率函數（RL / Over / ML）

**Files:**
- Modify: `scripts/run_model.py`
- Test: `scripts/tests/test_run_model.py`

- [ ] **Step 1: 加 failing test**

```python
# 追加到 scripts/tests/test_run_model.py
def test_probabilities_worked_example():
    mu_margin = 4.10 - 4.18   # -0.08
    mu_total = 4.10 + 4.18    # 8.28
    # 主 -1.5 過盤
    assert round(run_model.cover_prob_home(mu_margin, rl_point_home=-1.5), 3) == 0.355
    # Over 8.5
    assert round(run_model.over_prob(mu_total, total_line=8.5), 3) == 0.479
    # 主 ML
    assert round(run_model.home_ml_prob(mu_margin), 3) == 0.492

def test_cover_prob_home_dog_line():
    # 主 +1.5:margin > -1.5 才 cover,機率應 > 主 -1.5
    mm = 0.0
    assert run_model.cover_prob_home(mm, +1.5) > run_model.cover_prob_home(mm, -1.5)

def test_over_under_complement():
    p_over = run_model.over_prob(8.28, 8.5)
    assert math.isclose(p_over + run_model.over_prob(8.28, 8.5), 2 * p_over)
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_run_model.py::test_probabilities_worked_example -v`
Expected: FAIL（`AttributeError: module 'run_model' has no attribute 'cover_prob_home'`）

- [ ] **Step 3: 實作機率函數（追加到 `run_model.py`）**

```python
# 追加到 scripts/run_model.py

def cover_prob_home(mu_margin: float, rl_point_home: float) -> float:
    """P(主隊過 RL)。主隊 cover 條件:margin > −rl_point_home。
    主 −1.5 → P(margin>1.5);主 +1.5 → P(margin>−1.5)。"""
    z = (-rl_point_home - mu_margin) / config.SIGMA
    return 1.0 - _N.cdf(z)


def over_prob(mu_total: float, total_line: float) -> float:
    """P(Over):P(total > 線)。"""
    z = (total_line - mu_total) / config.SIGMA
    return 1.0 - _N.cdf(z)


def home_ml_prob(mu_margin: float) -> float:
    """P(主隊勝)= P(margin > 0)。內部用,不輸出給使用者。"""
    return _N.cdf(mu_margin / config.SIGMA)
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_run_model.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model.py scripts/tests/test_run_model.py
git commit -m "feat(predict): run_model RL/Over/ML 機率(常態近似)"
```

### Task 3: 頂層 `predict()` 組裝

**Files:**
- Modify: `scripts/run_model.py`
- Test: `scripts/tests/test_run_model.py`

- [ ] **Step 1: 加 failing test**

```python
# 追加到 scripts/tests/test_run_model.py
def test_predict_assembles_output():
    out = run_model.predict(
        home_rs=4.8, away_rs=4.2,
        home_starter_fip=4.5, away_starter_fip=3.6,
        home_bullpen_era=4.2, away_bullpen_era=4.0,
        pf=100, rl_point_home=-1.5, total_line=8.5,
    )
    assert round(out["mu_home"], 2) == 4.10
    assert round(out["mu_away"], 2) == 4.18
    assert round(out["mu_margin"], 2) == -0.08
    assert round(out["mu_total"], 2) == 8.28
    assert round(out["p_home_cover_rl"], 3) == 0.355
    assert round(out["p_over"], 3) == 0.479
    assert round(out["p_home_ml"], 3) == 0.492

def test_predict_deterministic():
    kw = dict(home_rs=4.8, away_rs=4.2, home_starter_fip=4.5, away_starter_fip=3.6,
              home_bullpen_era=4.2, away_bullpen_era=4.0, pf=100,
              rl_point_home=-1.5, total_line=8.5)
    assert run_model.predict(**kw) == run_model.predict(**kw)
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_run_model.py::test_predict_assembles_output -v`
Expected: FAIL（`AttributeError: ... 'predict'`）

- [ ] **Step 3: 實作 `predict()`（追加到 `run_model.py`）**

```python
# 追加到 scripts/run_model.py

def predict(*, home_rs: float, away_rs: float,
            home_starter_fip: float, away_starter_fip: float,
            home_bullpen_era: float, away_bullpen_era: float,
            pf: float, rl_point_home: float | None, total_line: float | None) -> dict:
    """完整模型輸出。rl_point_home / total_line 可為 None(無盤口時機率仍算 ML)。

    機率由「2 位小數的 μ」推導,使顯示的 μ 與機率內部一致、可重現
    (否則 raw μ 會讓機率第 3 位小數與 worked example 47.9% 不符)。
    """
    home_pitch = pitch_today(home_starter_fip, home_bullpen_era)
    away_pitch = pitch_today(away_starter_fip, away_bullpen_era)
    mu_home_raw, mu_away_raw = expected_runs(home_rs, away_rs, home_pitch, away_pitch, pf)
    mu_home = round(mu_home_raw, 2)
    mu_away = round(mu_away_raw, 2)
    mu_margin = round(mu_home - mu_away, 2)
    mu_total = round(mu_home + mu_away, 2)
    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "mu_margin": mu_margin,
        "mu_total": mu_total,
        "p_home_ml": round(home_ml_prob(mu_margin), 4),
        "p_home_cover_rl": (round(cover_prob_home(mu_margin, rl_point_home), 4)
                            if rl_point_home is not None else None),
        "p_over": (round(over_prob(mu_total, total_line), 4)
                   if total_line is not None else None),
    }
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_run_model.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model.py scripts/tests/test_run_model.py
git commit -m "feat(predict): run_model.predict 組裝完整輸出"
```

---

## Phase 2 — `fetch_inputs.py`（資料抓取，salvage 自 fetch_game_data）

### Task 4: 純計算 `calc_fip` + `parse_ip` + `rs_blend`

**Files:**
- Create: `scripts/fetch_inputs.py`
- Test: `scripts/tests/test_fetch_inputs.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_fetch_inputs.py
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_inputs as fi

def test_parse_ip_thirds():
    assert fi.parse_ip("123.1") == 123 + 1/3
    assert fi.parse_ip("50.2") == 50 + 2/3
    assert fi.parse_ip("12.0") == 12.0
    assert fi.parse_ip("0") == 0.0

def test_calc_fip_standard():
    # (13*15 + 3*(40+5) - 2*180)/180 + 3.10
    # = (195 + 135 - 360)/180 + 3.10 = (-30)/180 + 3.10 = -0.1667 + 3.10 = 2.93
    assert fi.calc_fip(hr=15, bb=40, hbp=5, k=180, ip=180.0) == 2.93

def test_calc_fip_min_ip_fallback():
    # IP < MIN_IP(10) → None(呼叫端 fallback 聯盟)
    assert fi.calc_fip(hr=1, bb=2, hbp=0, k=5, ip=4.0) is None

def test_rs_blend():
    # 0.35*5.0 + 0.65*4.0 = 1.75 + 2.6 = 4.35
    assert math.isclose(fi.rs_blend(recent=5.0, season=4.0), 4.35, rel_tol=1e-9)
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_fetch_inputs.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'fetch_inputs'`）

- [ ] **Step 3: 實作純函數段（`fetch_inputs.py` 開頭）**

```python
# scripts/fetch_inputs.py
"""抓單場模型輸入 + 凍結用打線快照。純計算與 I/O 分離,純計算可單測。"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from _team_resolver import resolve_team_id, team_abbr
from park_factors_lib import runs_pf

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_ip(ip_str) -> float:
    """MLB inningsPitched 字串 → float(.1=1/3, .2=2/3)。"""
    whole, _, frac = str(ip_str).partition(".")
    thirds = {"1": 1/3, "2": 2/3}.get(frac, 0.0)
    return int(whole or 0) + thirds


def calc_fip(*, hr: int, bb: int, hbp: int, k: int, ip: float) -> float | None:
    """標準 FIP(含 HBP)。IP < config.MIN_IP → None(樣本太小)。"""
    if ip < config.MIN_IP:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + config.FIP_CONSTANT, 2)


def rs_blend(recent: float, season: float) -> float:
    """RS 近期/整季加權混合。"""
    return config.RECENT_W * recent + (1 - config.RECENT_W) * season
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_fetch_inputs.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_inputs.py scripts/tests/test_fetch_inputs.py
git commit -m "feat(predict): fetch_inputs 純計算(FIP/parse_ip/rs_blend)"
```

### Task 5: 抓取層（schedule / RS-RA / FIP / 牛棚 / PF / 打線）

**Files:**
- Modify: `scripts/fetch_inputs.py`
- Test: `scripts/tests/test_fetch_inputs.py`（assemble 純函數測試 + 一個網路 smoke 步驟）

- [ ] **Step 1: 加 failing test（純組裝函數）**

```python
# 追加到 scripts/tests/test_fetch_inputs.py
def test_assemble_inputs_pure():
    raw = {
        "game": {"date": "2026-05-29", "game_pk": 778001, "venue": "Coors Field",
                 "home": {"team": "Colorado Rockies", "team_id": 115,
                          "probable_pitcher": "P A", "probable_pitcher_id": 1},
                 "away": {"team": "Arizona Diamondbacks", "team_id": 109,
                          "probable_pitcher": "P B", "probable_pitcher_id": 2}},
        "home_rs_recent": 5.0, "home_rs_season": 4.0,
        "away_rs_recent": 4.2, "away_rs_season": 4.4,
        "home_ra_recent": 5.5, "home_ra_season": 5.0,
        "away_ra_recent": 4.0, "away_ra_season": 4.1,
        "home_starter": {"name": "P A", "id": 1, "fip": 4.5, "ip": 60.0,
                         "k": 55, "bb": 20, "hbp": 3, "hr": 8},
        "away_starter": {"name": "P B", "id": 2, "fip": 3.6, "ip": 70.0,
                         "k": 80, "bb": 18, "hbp": 2, "hr": 6},
        "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
        "park_factor": 112.0,
        "lineup_frozen": {"source": "projected", "home": [], "away": []},
    }
    out = fi.assemble_inputs(raw)
    assert out["home_rs_blend"] == round(fi.rs_blend(5.0, 4.0), 3)
    assert out["away_rs_blend"] == round(fi.rs_blend(4.2, 4.4), 3)
    assert out["park_factor"] == 112.0
    # raw 透傳(供 features.json 凍結)
    assert out["raw"]["home_starter"]["fip"] == 4.5
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_fetch_inputs.py::test_assemble_inputs_pure -v`
Expected: FAIL（`AttributeError: ... 'assemble_inputs'`）

- [ ] **Step 3: 實作抓取層 + 組裝（追加到 `fetch_inputs.py`）**

```python
# 追加到 scripts/fetch_inputs.py

def fetch_schedule_game(date: str, home_id: int, away_id: int) -> dict | None:
    """抓當日賽程,回傳該 matchup 的 game dict(含 probablePitcher)。"""
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher(note)"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    r.raise_for_status()
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            h = g["teams"]["home"]["team"]["id"]
            a = g["teams"]["away"]["team"]["id"]
            if h == home_id and a == away_id:
                return g
    return None


def _team_rs_ra(team_id: int, before_date: str) -> dict:
    """近 RECENT_N 場 + 整季的 RS/RA per game。沿用 schedule+linescore 模式。"""
    def _games(start_days_back: int | None):
        end = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
        start = (datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=start_days_back)
                 if start_days_back else datetime(end.year, 3, 20))
        params = {"sportId": 1, "teamId": team_id, "startDate": start.strftime("%Y-%m-%d"),
                  "endDate": end.strftime("%Y-%m-%d"), "hydrate": "linescore"}
        r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
        r.raise_for_status()
        rows = []
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                if g["status"]["abstractGameState"] != "Final" or g.get("gameType", "R") != "R":
                    continue
                h = g["teams"]["home"]; a = g["teams"]["away"]
                is_home = h["team"]["id"] == team_id
                me, opp = (h, a) if is_home else (a, h)
                rows.append((me.get("score") or 0, opp.get("score") or 0, g["gameDate"][:10]))
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows

    season = _games(None)
    recent = season[:config.RECENT_N]

    def _per_game(rows, idx):
        return round(sum(r[idx] for r in rows) / len(rows), 2) if rows else config.LEAGUE_RG

    return {
        "rs_recent": _per_game(recent, 0), "ra_recent": _per_game(recent, 1),
        "rs_season": _per_game(season, 0), "ra_season": _per_game(season, 1),
    }


def fetch_starter(mlbam_id: int | None, name: str, year: int) -> dict:
    """抓先發季成績組件並算 FIP。id 缺 / 無成績 → fip=None(呼叫端 fallback)。"""
    base = {"name": name, "id": mlbam_id, "fip": None,
            "ip": None, "k": None, "bb": None, "hbp": None, "hr": None}
    if not mlbam_id:
        return base
    try:
        r = requests.get(f"{MLB_API_BASE}/people/{mlbam_id}/stats",
                         params={"stats": "season", "group": "pitching", "season": year},
                         timeout=10)
        r.raise_for_status()
        splits = (r.json().get("stats") or [{}])[0].get("splits") or []
        if not splits:
            return base
        s = splits[0]["stat"]
        ip = parse_ip(s.get("inningsPitched", "0"))
        k = int(s.get("strikeOuts", 0)); bb = int(s.get("baseOnBalls", 0))
        hbp = int(s.get("hitByPitch", 0)); hr = int(s.get("homeRuns", 0))
        base.update(ip=ip, k=k, bb=bb, hbp=hbp, hr=hr,
                    fip=calc_fip(hr=hr, bb=bb, hbp=hbp, k=k, ip=ip))
        return base
    except Exception as e:
        print(f"[fetch_inputs] starter {mlbam_id} 失敗:{e}", file=sys.stderr)
        return base


def fetch_bullpen_era(team_id: int, year: int) -> float:
    """牛棚 ERA(sitCodes=rp)。失敗 → 4.00。"""
    try:
        r = requests.get(f"{MLB_API_BASE}/teams/{team_id}/stats",
                         params={"stats": "statSplits", "group": "pitching",
                                 "season": year, "sitCodes": "rp"}, timeout=10)
        r.raise_for_status()
        for sg in r.json().get("stats", []):
            for sp in sg.get("splits", []):
                era = sp.get("stat", {}).get("era")
                if era is not None:
                    return float(era)
    except Exception:
        pass
    return 4.00


def fetch_lineup_light(team_id: int, game_pk: int, year: int) -> list[dict]:
    """凍結用輕量打線:有官方先發打線就抓 9 人(order/name/id),否則回 []。
    只取名單與打序;進階攻擊值留空(v1 不進模型,凍結為日後 ablation 保留欄位)。"""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10)
        r.raise_for_status()
        box = r.json().get("liveData", {}).get("boxscore", {}).get("teams", {})
        side = "home" if box.get("home", {}).get("team", {}).get("id") == team_id else "away"
        order = box.get(side, {}).get("battingOrder", []) or []
        players = box.get(side, {}).get("players", {})
        out = []
        for i, pid in enumerate(order[:9]):
            p = players.get(f"ID{pid}", {})
            out.append({"order": i + 1, "name": p.get("person", {}).get("fullName"),
                        "id": pid, "ops": None, "woba": None})
        return out
    except Exception:
        return []


def assemble_inputs(raw: dict) -> dict:
    """純組裝:raw(各抓取結果) → run_model 可吃的扁平 dict + 透傳 raw 供凍結。"""
    return {
        "home_rs_blend": round(rs_blend(raw["home_rs_recent"], raw["home_rs_season"]), 3),
        "away_rs_blend": round(rs_blend(raw["away_rs_recent"], raw["away_rs_season"]), 3),
        "home_starter_fip": raw["home_starter"]["fip"] if raw["home_starter"]["fip"] is not None else config.LEAGUE_RG,
        "away_starter_fip": raw["away_starter"]["fip"] if raw["away_starter"]["fip"] is not None else config.LEAGUE_RG,
        "home_bullpen_era": raw["home_bullpen_era"],
        "away_bullpen_era": raw["away_bullpen_era"],
        "park_factor": raw["park_factor"],
        "raw": raw,
    }


def fetch_inputs(date: str, away: str, home: str) -> dict:
    """主入口:抓齊一場的輸入。回傳 assemble_inputs(raw) 結果(含 raw 供凍結)。"""
    home_id = resolve_team_id(home)
    away_id = resolve_team_id(away)
    year = int(date[:4])

    game = fetch_schedule_game(date, home_id, away_id)
    if game is None:
        raise ValueError(f"找不到 {away}@{home} 於 {date} 的例行賽")

    gi_home = game["teams"]["home"]; gi_away = game["teams"]["away"]
    game_pk = game["gamePk"]
    venue = game["venue"]["name"]
    home_pp = gi_home.get("probablePitcher", {})
    away_pp = gi_away.get("probablePitcher", {})

    home_form = _team_rs_ra(home_id, date)
    away_form = _team_rs_ra(away_id, date)
    home_starter = fetch_starter(home_pp.get("id"), home_pp.get("fullName", "TBD"), year)
    away_starter = fetch_starter(away_pp.get("id"), away_pp.get("fullName", "TBD"), year)

    raw = {
        "game": {"date": date, "game_pk": game_pk, "venue": venue,
                 "home": {"team": gi_home["team"]["name"], "team_id": home_id,
                          "probable_pitcher": home_starter["name"], "probable_pitcher_id": home_pp.get("id")},
                 "away": {"team": gi_away["team"]["name"], "team_id": away_id,
                          "probable_pitcher": away_starter["name"], "probable_pitcher_id": away_pp.get("id")}},
        "home_rs_recent": home_form["rs_recent"], "home_rs_season": home_form["rs_season"],
        "away_rs_recent": away_form["rs_recent"], "away_rs_season": away_form["rs_season"],
        "home_ra_recent": home_form["ra_recent"], "home_ra_season": home_form["ra_season"],
        "away_ra_recent": away_form["ra_recent"], "away_ra_season": away_form["ra_season"],
        "home_starter": home_starter, "away_starter": away_starter,
        "home_bullpen_era": fetch_bullpen_era(home_id, year),
        "away_bullpen_era": fetch_bullpen_era(away_id, year),
        "park_factor": runs_pf(venue),
        "lineup_frozen": {"source": "official",
                          "home": fetch_lineup_light(home_id, game_pk, year),
                          "away": fetch_lineup_light(away_id, game_pk, year)},
    }
    if not raw["lineup_frozen"]["home"] and not raw["lineup_frozen"]["away"]:
        raw["lineup_frozen"]["source"] = "projected"
    return assemble_inputs(raw)
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_fetch_inputs.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 網路 smoke（手動驗證一次,非自動測試）**

Run: `$PYTHON -c "import sys; sys.path.insert(0,'scripts'); import fetch_inputs as fi, json; print(json.dumps(fi.fetch_inputs('2026-05-28','ARI','COL'), ensure_ascii=False)[:400])"`
Expected: 印出含 `home_rs_blend` / `away_starter_fip` 等鍵的 JSON 片段(數值合理、非全 fallback)。若當天該 matchup 不存在,換一組當日真有的 matchup。

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_inputs.py scripts/tests/test_fetch_inputs.py
git commit -m "feat(predict): fetch_inputs 抓取層(RS-RA/FIP/牛棚/PF/打線)"
```

---

## Phase 3 — `odds_compare.py`（no-vig + edge）

### Task 6: `lib/closing_line.py` 加 RL 抽取

**Files:**
- Modify: `scripts/lib/closing_line.py`
- Test: `scripts/tests/test_odds_compare.py`（新檔,本 task 先測 closing_line）
- Fixture: 重用 `odds/odds_snapshots/2026-05-27_*-ET.json`（執行時挑一個存在且含 pinnacle.rl 的）

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_odds_compare.py
import sys, json
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from lib.closing_line import extract_pinnacle_rl_no_vig

def _sample_game_with_rl():
    return {
        "home_team": "Colorado Rockies", "away_team": "Arizona Diamondbacks",
        "bookmakers": {"pinnacle": {"rl": {
            "Colorado Rockies": {"point": -1.5, "no_vig_pct": 38.0},
            "Arizona Diamondbacks": {"point": 1.5, "no_vig_pct": 62.0},
        }}},
    }

def test_extract_rl_no_vig():
    out = extract_pinnacle_rl_no_vig(_sample_game_with_rl())
    assert out["home_point"] == -1.5
    assert round(out["home_no_vig"], 3) == 0.380
    assert out["away_point"] == 1.5
    assert round(out["away_no_vig"], 3) == 0.620

def test_extract_rl_missing_returns_none():
    assert extract_pinnacle_rl_no_vig({"home_team": "X", "away_team": "Y",
                                        "bookmakers": {}}) is None
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_odds_compare.py -v`
Expected: FAIL（`ImportError: cannot import name 'extract_pinnacle_rl_no_vig'`）

- [ ] **Step 3: 加 `extract_pinnacle_rl_no_vig`（追加到 `scripts/lib/closing_line.py`）**

```python
# 追加到 scripts/lib/closing_line.py

def extract_pinnacle_rl_no_vig(game: dict) -> dict | None:
    """抽 Pinnacle RL 的 home/away point + no-vig 機率。缺 → None。"""
    pinn = game.get("bookmakers", {}).get("pinnacle")
    if not pinn:
        return None
    rl = pinn.get("rl", {})
    home = game.get("home_team"); away = game.get("away_team")
    h = rl.get(home, {}); a = rl.get(away, {})
    if "no_vig_pct" not in h or "no_vig_pct" not in a:
        return None
    if "point" not in h or "point" not in a:
        return None
    return {
        "home_point": float(h["point"]), "home_no_vig": h["no_vig_pct"] / 100.0,
        "away_point": float(a["point"]), "away_no_vig": a["no_vig_pct"] / 100.0,
    }
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_odds_compare.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/closing_line.py scripts/tests/test_odds_compare.py
git commit -m "feat(odds): closing_line 加 RL no-vig 抽取"
```

### Task 7: `odds_compare.py` — 找最新 snapshot + 算 edge

**Files:**
- Create: `scripts/odds_compare.py`
- Test: `scripts/tests/test_odds_compare.py`

- [ ] **Step 1: 加 failing test**

```python
# 追加到 scripts/tests/test_odds_compare.py
import odds_compare as oc

def test_compute_edges():
    model = {"p_home_cover_rl": 0.355, "p_over": 0.479}
    market = {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                     "away_point": 1.5, "away_no_vig": 0.59},
              "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}}
    edges = oc.compute_edges(model, market)
    # (0.355 - 0.41)*100 = -5.5
    assert round(edges["home_rl_pp"], 1) == -5.5
    # (0.479 - 0.52)*100 = -4.1
    assert round(edges["over_pp"], 1) == -4.1

def test_compute_edges_none_market():
    assert oc.compute_edges({"p_home_cover_rl": 0.5, "p_over": 0.5}, None) == {
        "home_rl_pp": None, "over_pp": None}
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_odds_compare.py::test_compute_edges -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'odds_compare'`）

- [ ] **Step 3: 實作 `odds_compare.py`**

```python
# scripts/odds_compare.py
"""找預測當下的最新 Pinnacle snapshot,抽 RL+總分 no-vig,算 vs model 的 edge。"""
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    _parse_iso_utc, extract_pinnacle_no_vig, extract_pinnacle_rl_no_vig,
)


def find_latest_snapshot_for_game(date: str, home_team: str, away_team: str,
                                  snapshots_dir: Path = SNAPSHOTS_DIR) -> tuple[dict | None, str | None]:
    """掃 odds_snapshots,挑「snapshot_time 最新且 < 開球」且含此 matchup 的那筆。"""
    best = None  # (snap_ts, game_dict, filename)
    for f in sorted(snapshots_dir.glob(f"{date}_*-ET.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
        if snap_ts is None:
            continue
        for g in data.get("games", []):
            if g.get("home_team") != home_team or g.get("away_team") != away_team:
                continue
            commence = _parse_iso_utc(g.get("commence_utc", ""))
            if commence is None or snap_ts >= commence:
                continue
            if best is None or snap_ts > best[0]:
                best = (snap_ts, g, f.name)
    if best is None:
        return None, None
    return best[1], best[2]


def market_from_snapshot(game: dict) -> dict | None:
    """從 snapshot game 抽 RL + 總分 no-vig。任一缺 → None。"""
    ml_total = extract_pinnacle_no_vig(game)   # 已含 total_line / over_no_vig / under_no_vig
    rl = extract_pinnacle_rl_no_vig(game)
    if ml_total is None or rl is None:
        return None
    return {
        "rl": rl,
        "total": {"line": ml_total["total_line"],
                  "over_no_vig": ml_total["over_no_vig"],
                  "under_no_vig": ml_total["under_no_vig"]},
    }


def compute_edges(model: dict, market: dict | None) -> dict:
    """edge(pp) = (model 機率 − 市場 no-vig) × 100。market None → 全 None。"""
    if not market:
        return {"home_rl_pp": None, "over_pp": None}
    home_rl_pp = None
    if model.get("p_home_cover_rl") is not None:
        home_rl_pp = round((model["p_home_cover_rl"] - market["rl"]["home_no_vig"]) * 100, 1)
    over_pp = None
    if model.get("p_over") is not None:
        over_pp = round((model["p_over"] - market["total"]["over_no_vig"]) * 100, 1)
    return {"home_rl_pp": home_rl_pp, "over_pp": over_pp}
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_odds_compare.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_compare.py scripts/tests/test_odds_compare.py
git commit -m "feat(odds): odds_compare 找最新 snapshot + edge"
```

---

## Phase 4 — `report.py`（凍結 + 渲染）

### Task 8: `report.py` — features.json + prediction.md

**Files:**
- Create: `scripts/report.py`
- Test: `scripts/tests/test_report.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_report.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report

def _bundle():
    return {
        "inputs": {"raw": {"game": {"date": "2026-05-29", "game_pk": 778001,
                                    "venue": "Coors Field",
                                    "home": {"team": "Colorado Rockies"},
                                    "away": {"team": "Arizona Diamondbacks"}},
                           "home_starter": {"name": "P A", "fip": 4.5},
                           "away_starter": {"name": "P B", "fip": 3.6},
                           "lineup_frozen": {"source": "official", "home": [], "away": []},
                           "home_ra_recent": 5.5, "home_ra_season": 5.0,
                           "away_ra_recent": 4.0, "away_ra_season": 4.1},
                   "home_rs_blend": 4.8, "away_rs_blend": 4.2,
                   "home_starter_fip": 4.5, "away_starter_fip": 3.6,
                   "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
                   "park_factor": 112.0},
        "model": {"mu_home": 4.10, "mu_away": 4.18, "mu_margin": -0.08, "mu_total": 8.28,
                  "p_home_ml": 0.492, "p_home_cover_rl": 0.355, "p_over": 0.479},
        "market": {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                          "away_point": 1.5, "away_no_vig": 0.59},
                   "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}},
        "edges": {"home_rl_pp": -5.5, "over_pp": -4.1},
        "snapshot_file": "2026-05-29_15-00-ET.json",
    }

def test_build_features_schema(tmp_path):
    feats = report.build_features(_bundle())
    assert feats["schema_version"] == 2
    assert feats["game"]["home"] == "Colorado Rockies"
    assert feats["inputs"]["home_ra_season"] == 5.0     # RA 凍結但不進模型
    assert feats["lineup_frozen"]["source"] == "official"
    assert feats["model"]["p_over"] == 0.479
    assert feats["odds"]["rl"]["home_point"] == -1.5
    assert feats["edges"]["over_pp"] == -4.1
    assert "constants_used" in feats["model"]

def test_render_prediction_md_has_rl_ou_no_ml():
    md = report.render_prediction_md(_bundle())
    assert "RL HOME" in md and "Over" in md
    assert "35.5%" in md or "0.355" in md
    assert "Money line" not in md and "Moneyline" not in md  # 不得出現 ML

def test_write_outputs(tmp_path):
    paths = report.write_outputs(_bundle(), out_dir=tmp_path)
    assert (tmp_path / "features.json").exists()
    assert (tmp_path / "prediction.md").exists()
    data = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_report.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'report'`）

- [ ] **Step 3: 實作 `report.py`**

```python
# scripts/report.py
"""凍結 features.json(回測/ablation) + 產 prediction.md(AI 敘事素材)。"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config


def build_features(b: dict) -> dict:
    """組 features.json(schema v2)。b = orchestrator 收集的 bundle。"""
    inp = b["inputs"]; raw = inp["raw"]; g = raw["game"]
    market = b.get("market")
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": {"date": g["date"], "game_pk": g["game_pk"],
                 "home": g["home"]["team"], "away": g["away"]["team"], "venue": g["venue"]},
        "inputs": {
            "home_rs_recent": raw["home_rs_recent"], "home_rs_season": raw["home_rs_season"],
            "away_rs_recent": raw["away_rs_recent"], "away_rs_season": raw["away_rs_season"],
            "home_ra_recent": raw["home_ra_recent"], "home_ra_season": raw["home_ra_season"],
            "away_ra_recent": raw["away_ra_recent"], "away_ra_season": raw["away_ra_season"],
            "home_starter": raw["home_starter"], "away_starter": raw["away_starter"],
            "home_bullpen_era": inp["home_bullpen_era"], "away_bullpen_era": inp["away_bullpen_era"],
            "park_factor": inp["park_factor"], "league_rg_used": config.LEAGUE_RG,
        },
        "lineup_frozen": raw["lineup_frozen"],
        "model": {**b["model"], "constants_used": config.constants_snapshot()},
        "odds": ({"snapshot_file": b.get("snapshot_file"),
                  "rl": market["rl"], "total": market["total"]} if market else None),
        "edges": b["edges"],
    }


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _nv(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def render_prediction_md(b: dict) -> str:
    g = b["inputs"]["raw"]["game"]; m = b["model"]; mk = b.get("market"); e = b["edges"]
    home = g["home"]["team"]; away = g["away"]["team"]
    lines = [
        f"## {away} @ {home} — {g['date']}",
        f"- 期望得分:HOME {m['mu_home']} / AWAY {m['mu_away']}(total {m['mu_total']})",
        "",
        "| 市場 | 線 | model 機率 | 市場 no-vig | edge(pp) |",
        "|------|----|-----------|-------------|----------|",
    ]
    if mk:
        rl = mk["rl"]; tot = mk["total"]
        p_home = m["p_home_cover_rl"]; p_away = (1 - p_home) if p_home is not None else None
        p_over = m["p_over"]; p_under = (1 - p_over) if p_over is not None else None
        e_rl = e["home_rl_pp"]; e_ov = e["over_pp"]
        e_rl_a = (-e_rl) if isinstance(e_rl, (int, float)) else None
        e_ov_u = (-e_ov) if isinstance(e_ov, (int, float)) else None
        lines += [
            f"| RL HOME | {rl['home_point']:+} | {_pct(p_home)} | {_nv(rl['home_no_vig'])} | {e_rl:+} |",
            f"| RL AWAY | {rl['away_point']:+} | {_pct(p_away)} | {_nv(rl['away_no_vig'])} | {e_rl_a:+} |",
            f"| Over | {tot['line']} | {_pct(p_over)} | {_nv(tot['over_no_vig'])} | {e_ov:+} |",
            f"| Under | {tot['line']} | {_pct(p_under)} | {_nv(tot['under_no_vig'])} | {e_ov_u:+} |",
            "",
            f"- 所用盤口 snapshot:{b.get('snapshot_file', '—')}",
        ]
    else:
        lines += [
            f"| RL HOME | — | {_pct(m['p_home_cover_rl'])} | — | — |",
            f"| Over | — | {_pct(m['p_over'])} | — | — |",
            "",
            "- ⚠️ 無盤口可比(snapshot 缺或未開盤),只輸出 model 機率。",
        ]
    lines += [
        "",
        "<!-- AI 敘事:哪邊有正 edge、量級、需注意什麼。"
        "不喊「下哪邊」、不硬掰 EV%、只談 RL 與 O/U(不輸出勝負盤)。 -->",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(b: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = build_features(b)
    (out_dir / "features.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "prediction.md").write_text(render_prediction_md(b), encoding="utf-8")
    return {"features": out_dir / "features.json", "prediction": out_dir / "prediction.md"}
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_report.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/report.py scripts/tests/test_report.py
git commit -m "feat(predict): report 凍結 features.json + prediction.md"
```

---

## Phase 5 — `predict_game.py`（orchestrator）

### Task 9: 單場流程

**Files:**
- Create: `scripts/predict_game.py`
- Test: `scripts/tests/test_predict_game.py`

- [ ] **Step 1: 寫 failing test（純組裝 `run_one_from_inputs`，不打網路）**

```python
# scripts/tests/test_predict_game.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_game as pg

def _inputs():
    return {
        "home_rs_blend": 4.8, "away_rs_blend": 4.2,
        "home_starter_fip": 4.5, "away_starter_fip": 3.6,
        "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
        "park_factor": 100.0,
        "raw": {"game": {"date": "2026-05-29", "game_pk": 1, "venue": "X",
                         "home": {"team": "H"}, "away": {"team": "A"}},
                "home_starter": {"fip": 4.5}, "away_starter": {"fip": 3.6},
                "lineup_frozen": {"source": "projected", "home": [], "away": []},
                "home_ra_recent": 5.0, "home_ra_season": 5.0,
                "away_ra_recent": 4.0, "away_ra_season": 4.0},
    }

def test_run_one_from_inputs_no_market():
    bundle = pg.run_one_from_inputs(_inputs(), market=None, snapshot_file=None)
    assert round(bundle["model"]["mu_home"], 2) == 4.10
    assert bundle["edges"]["home_rl_pp"] is None   # 無 market
    assert bundle["model"]["p_home_cover_rl"] is None

def test_run_one_from_inputs_with_market():
    market = {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                     "away_point": 1.5, "away_no_vig": 0.59},
              "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}}
    bundle = pg.run_one_from_inputs(_inputs(), market=market, snapshot_file="s.json")
    assert round(bundle["model"]["p_home_cover_rl"], 3) == 0.355
    assert round(bundle["edges"]["over_pp"], 1) == -4.1
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_predict_game.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'predict_game'`）

- [ ] **Step 3: 實作 `predict_game.py`**

```python
# scripts/predict_game.py
#!/usr/bin/env python3
"""Orchestrator:單場 (--matchup) 或當日全部 (--all)。串 fetch_inputs → run_model
→ odds_compare → report。"""
import argparse
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_model
import report as report_mod
from fetch_inputs import fetch_inputs, MLB_API_BASE
from odds_compare import find_latest_snapshot_for_game, market_from_snapshot, compute_edges
from _team_resolver import resolve_team_id, team_abbr

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def run_one_from_inputs(inputs: dict, market: dict | None, snapshot_file: str | None) -> dict:
    """純組裝:inputs(+ 可選 market) → 完整 bundle(model/market/edges/snapshot)。"""
    rl_point = market["rl"]["home_point"] if market else None
    total_line = market["total"]["line"] if market else None
    model = run_model.predict(
        home_rs=inputs["home_rs_blend"], away_rs=inputs["away_rs_blend"],
        home_starter_fip=inputs["home_starter_fip"], away_starter_fip=inputs["away_starter_fip"],
        home_bullpen_era=inputs["home_bullpen_era"], away_bullpen_era=inputs["away_bullpen_era"],
        pf=inputs["park_factor"], rl_point_home=rl_point, total_line=total_line,
    )
    edges = compute_edges(model, market)
    return {"inputs": inputs, "model": model, "market": market,
            "edges": edges, "snapshot_file": snapshot_file}


def predict_one(date: str, away: str, home: str, suffix: str = "") -> Path:
    """完整單場:抓資料 + 盤口 + 算 + 寫檔。回傳 out_dir。"""
    inputs = fetch_inputs(date, away, home)
    home_team = inputs["raw"]["game"]["home"]["team"]
    away_team = inputs["raw"]["game"]["away"]["team"]
    snap_game, snap_file = find_latest_snapshot_for_game(date, home_team, away_team)
    market = market_from_snapshot(snap_game) if snap_game else None
    bundle = run_one_from_inputs(inputs, market, snap_file)
    out_dir = SKILL_ROOT / "analysis-data" / date / f"{away}@{home}{suffix}"
    report_mod.write_outputs(bundle, out_dir)
    print(f"[OK] {away}@{home}{suffix} → {out_dir}", file=sys.stderr)
    return out_dir


def predict_all(date: str) -> list[Path]:
    """當日全部例行賽。逐場跑,單場失敗只記錄不中斷。"""
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher(note)"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            home_id = g["teams"]["home"]["team"]["id"]
            away_id = g["teams"]["away"]["team"]["id"]
            home = team_abbr(home_id, g["teams"]["home"]["team"]["name"])
            away = team_abbr(away_id, g["teams"]["away"]["team"]["name"])
            try:
                out.append(predict_one(date, away, home))
            except Exception as e:
                print(f"[SKIP] {away}@{home}:{e}", file=sys.stderr)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="MLB team-level 預測")
    p.add_argument("--date", required=True, help="ET 開打日 YYYY-MM-DD")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--matchup", help="AWAY@HOME(縮寫)")
    grp.add_argument("--all", action="store_true", help="當日全部例行賽")
    args = p.parse_args(argv)
    if args.all:
        predict_all(args.date)
    else:
        away, _, home = args.matchup.partition("@")
        predict_one(args.date, away.strip(), home.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_predict_game.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 端到端 smoke（手動，挑當日真有的 matchup）**

Run: `$PYTHON scripts/predict_game.py --date 2026-05-28 --matchup ARI@COL`
Expected: stderr 印 `[OK] ARI@COL → analysis-data/2026-05-28/ARI@COL`;該資料夾出現 `features.json` + `prediction.md`;打開 prediction.md 有 RL/Over 表、無 Money line。

- [ ] **Step 6: Commit**

```bash
git add scripts/predict_game.py scripts/tests/test_predict_game.py
git commit -m "feat(predict): predict_game orchestrator(單場 + --all)"
```

---

## Phase 6 — 回測改接 features.json

### Task 10: `lib/load.py` 改讀 features.json

**Files:**
- Modify: `scripts/lib/load.py`（整檔改寫 `_build_row` + `build_dataframe_for_month`）
- Test: `scripts/tests/test_backtest_load_v2.py`

- [ ] **Step 1: 寫 failing test（用 tmp 目錄擺一個 features.json + result.json）**

```python
# scripts/tests/test_backtest_load_v2.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import load

def _write_game(base: Path, date: str, matchup: str):
    d = base / date / matchup
    d.mkdir(parents=True)
    (d / "features.json").write_text(json.dumps({
        "schema_version": 2,
        "game": {"date": date, "game_pk": 1, "home": "H", "away": "A", "venue": "X"},
        "model": {"mu_total": 8.28, "p_home_cover_rl": 0.355, "p_over": 0.479,
                  "p_home_ml": 0.492},
        "odds": {"snapshot_file": "s.json",
                 "rl": {"home_point": -1.5, "home_no_vig": 0.41,
                        "away_point": 1.5, "away_no_vig": 0.59},
                 "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}},
        "edges": {"home_rl_pp": -5.5, "over_pp": -4.1},
    }, ensure_ascii=False), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({
        "winner": "HOME", "home_score": 6, "away_score": 3, "total": 9,
    }, ensure_ascii=False), encoding="utf-8")

def test_load_features_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "ANALYSIS_DATA_DIR", tmp_path)
    _write_game(tmp_path, "2026-05-29", "A@H")
    df = load.build_dataframe_for_month("2026-05")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["p_over"] == 0.479
    assert row["over_pp"] == -4.1
    assert row["actual_total"] == 9
    assert row["actual_margin"] == 3      # home 6 - away 3
    assert row["result_missing"] == False
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_backtest_load_v2.py -v`
Expected: FAIL（KeyError / 舊 schema 欄位 `skill_total` 不存在,或讀不到 features.json）

- [ ] **Step 3: 改寫 `scripts/lib/load.py`**

```python
# scripts/lib/load.py 整檔改寫
"""讀 features.json(v2)+ result.json → 每場一列 DataFrame。預測值已凍結,不重算。"""
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"

_MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})(?:-(?:G?\d+))?$")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_row(date: str, matchup_dir: Path) -> Optional[dict]:
    feats = _read_json(matchup_dir / "features.json")
    if feats is None or feats.get("schema_version") != 2:
        return None
    model = feats.get("model", {})
    odds = feats.get("odds") or {}
    edges = feats.get("edges", {})
    result = _read_json(matchup_dir / "result.json")

    rl = odds.get("rl") or {}
    total = odds.get("total") or {}
    actual_margin = actual_total = None
    if result is not None:
        actual_margin = result.get("home_score", 0) - result.get("away_score", 0)
        actual_total = result.get("total")

    return {
        "date": date, "matchup": matchup_dir.name,
        "game_pk": feats.get("game", {}).get("game_pk"),
        # model
        "mu_total": model.get("mu_total"),
        "p_home_cover_rl": model.get("p_home_cover_rl"),
        "p_over": model.get("p_over"),
        "p_home_ml": model.get("p_home_ml"),
        # market
        "rl_home_point": rl.get("home_point"),
        "rl_home_no_vig": rl.get("home_no_vig"),
        "total_line": total.get("line"),
        "over_no_vig": total.get("over_no_vig"),
        # edges
        "home_rl_pp": edges.get("home_rl_pp"),
        "over_pp": edges.get("over_pp"),
        # actual
        "actual_winner": result.get("winner") if result else None,
        "actual_total": actual_total,
        "actual_margin": actual_margin,
        # status
        "odds_missing": odds is None or not odds,
        "result_missing": result is None,
    }


def build_dataframe_for_month(month: str, days_filter: Optional[set] = None) -> pd.DataFrame:
    rows = []
    for date_dir in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        if days_filter is not None and date_dir.name not in days_filter:
            continue
        for matchup_dir in sorted(date_dir.iterdir()):
            if not matchup_dir.is_dir() or not _MATCHUP_RE.match(matchup_dir.name):
                continue
            row = _build_row(date_dir.name, matchup_dir)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_backtest_load_v2.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/load.py scripts/tests/test_backtest_load_v2.py
git commit -m "refactor(backtest): load 改讀 features.json v2"
```

### Task 11: `lib/metrics.py` 改成 RL / O/U / edge 指標

**Files:**
- Modify: `scripts/lib/metrics.py`（整檔改寫）
- Modify: `scripts/backtest.py`（移除已刪指標的呼叫;見 Step 3）
- Test: `scripts/tests/test_backtest_metrics_v2.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_backtest_metrics_v2.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from lib import metrics

def _df():
    return pd.DataFrame([
        # RL: home cover if actual_margin > -point(-1.5→>1.5). margin=3 → home covers.
        {"p_home_cover_rl": 0.355, "rl_home_point": -1.5, "actual_margin": 3,
         "p_over": 0.479, "total_line": 8.5, "actual_total": 9,
         "home_rl_pp": -5.5, "over_pp": -4.1, "result_missing": False, "odds_missing": False},
        {"p_home_cover_rl": 0.60, "rl_home_point": -1.5, "actual_margin": 0,
         "p_over": 0.60, "total_line": 8.5, "actual_total": 7,
         "home_rl_pp": 8.0, "over_pp": 6.0, "result_missing": False, "odds_missing": False},
    ])

def test_rl_hit_rate():
    out = metrics.compute_rl_metrics(_df())
    # game1: model 主過盤機率 0.355<0.5 → 預測「主不過」;實際 margin 3 → 主過盤 → miss
    # game2: model 0.60>0.5 → 預測「主過」;實際 margin 0(<1.5) → 主沒過 → miss
    assert out["n"] == 2
    assert round(out["rl_hit_rate"], 3) == 0.000

def test_ou_hit_rate():
    out = metrics.compute_ou_metrics(_df())
    # game1: model p_over 0.479<0.5 → 預測 Under;實際 9 > 8.5 → Over → miss
    # game2: model 0.60>0.5 → 預測 Over;實際 7 < 8.5 → Under → miss
    assert out["n"] == 2
    assert round(out["ou_hit_rate"], 3) == 0.000

def test_edge_calibration_positive_side():
    out = metrics.compute_edge_calibration(_df())
    assert "rl_pos_edge_n" in out and "ou_pos_edge_n" in out
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `$PYTHON -m pytest tests/test_backtest_metrics_v2.py -v`
Expected: FAIL（`AttributeError: ... 'compute_rl_metrics'`）

- [ ] **Step 3: 改寫 `scripts/lib/metrics.py`**

```python
# scripts/lib/metrics.py 整檔改寫
"""回測指標:RL 過盤命中、O/U 命中、edge 校準。讀 lib.load 產的 DataFrame。"""
import numpy as np
import pandas as pd


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["result_missing"]) & (~df["odds_missing"])].copy()


def compute_rl_metrics(df: pd.DataFrame) -> dict:
    """model 預測『主過盤』(p>0.5) 是否命中(實際 margin > −point)。"""
    v = _valid(df)
    v = v[v["p_home_cover_rl"].notna() & v["rl_home_point"].notna() & v["actual_margin"].notna()]
    if len(v) == 0:
        return {"n": 0, "rl_hit_rate": None}
    pred_home_cover = v["p_home_cover_rl"] > 0.5
    actual_home_cover = v["actual_margin"] > (-v["rl_home_point"])
    hit = (pred_home_cover == actual_home_cover).mean()
    return {"n": int(len(v)), "rl_hit_rate": float(hit)}


def compute_ou_metrics(df: pd.DataFrame) -> dict:
    """model 預測 Over(p>0.5) 是否命中。排除 push(actual == line)。"""
    v = _valid(df)
    v = v[v["p_over"].notna() & v["total_line"].notna() & v["actual_total"].notna()]
    v = v[v["actual_total"] != v["total_line"]]
    if len(v) == 0:
        return {"n": 0, "ou_hit_rate": None}
    pred_over = v["p_over"] > 0.5
    actual_over = v["actual_total"] > v["total_line"]
    hit = (pred_over == actual_over).mean()
    return {"n": int(len(v)), "ou_hit_rate": float(hit)}


def compute_edge_calibration(df: pd.DataFrame) -> dict:
    """正 edge 那側是否真的較常贏(edge 有沒有預測力)。"""
    v = _valid(df)
    out = {}
    # RL:home_rl_pp>0 → 看好主過盤
    rl = v[v["home_rl_pp"].notna() & v["actual_margin"].notna() & v["rl_home_point"].notna()]
    rl_pos = rl[rl["home_rl_pp"] > 0]
    if len(rl_pos):
        actual = rl_pos["actual_margin"] > (-rl_pos["rl_home_point"])
        out["rl_pos_edge_n"] = int(len(rl_pos))
        out["rl_pos_edge_hit"] = float(actual.mean())
    else:
        out["rl_pos_edge_n"] = 0; out["rl_pos_edge_hit"] = None
    # O/U:over_pp>0 → 看好 Over
    ou = v[v["over_pp"].notna() & v["actual_total"].notna() & v["total_line"].notna()]
    ou = ou[ou["actual_total"] != ou["total_line"]]
    ou_pos = ou[ou["over_pp"] > 0]
    if len(ou_pos):
        actual = ou_pos["actual_total"] > ou_pos["total_line"]
        out["ou_pos_edge_n"] = int(len(ou_pos))
        out["ou_pos_edge_hit"] = float(actual.mean())
    else:
        out["ou_pos_edge_n"] = 0; out["ou_pos_edge_hit"] = None
    return out
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `$PYTHON -m pytest tests/test_backtest_metrics_v2.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 修 `scripts/backtest.py` 改用新指標**

把 `backtest.py` 的 `cmd_run` 內 metrics 呼叫換成新函數(移除 `compute_direction_metrics` / `compute_total_metrics` / `compute_calibration` / `compute_slice_metrics` import,改 import `compute_rl_metrics, compute_ou_metrics, compute_edge_calibration`),並把 `render_report` 呼叫改成傳這三個 dict。`lib/render.py` 對應改成印 RL/OU/edge 三段(渲染細節在本 task 一併改;若超出時間,render 可先印 `dict` 原貌,報告美化另開 task)。

驗證:`$PYTHON scripts/backtest.py run --month 2026-05`(需先有 v2 features.json;5 月舊資料無 v2 → 預期 0 列,正常,going-forward 累積)。
Expected: 不報錯;印出 valid 列數(可能為 0,因 5 月是舊 schema)。

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/metrics.py scripts/backtest.py scripts/lib/render.py scripts/tests/test_backtest_metrics_v2.py
git commit -m "refactor(backtest): metrics 改 RL/OU/edge 校準"
```

---

## Phase 7 — 刪除舊碼 + SKILL 重寫

### Task 12: 退役舊腳本 + 其 tests

**Files:**
- Delete:（見下方清單）

- [ ] **Step 1: 刪除退役檔（完全退役清單）**

```bash
cd "C:/Users/USER/.agents/skills/mlb-game-analyzer"
git rm scripts/pitcher_stats.py scripts/lineup_analyzer.py scripts/dossier_renderer.py \
       scripts/summary_renderer.py scripts/signals_lib.py scripts/lib_tier_v2.py \
       scripts/lib_role_tagging.py scripts/roster_checker.py scripts/merge_game_data.py \
       scripts/scoring_formula.py scripts/predict.py scripts/refresh_baselines.py \
       scripts/backfill_signals.py scripts/fetch_game_data.py scripts/prepare_game.py
git rm reference/matchup-factors.md reference/flags-checklist.md reference/workflow-fundamentals.md
git rm scripts/tests/test_prepare_game.py scripts/tests/test_prepare_game_steps.py \
       scripts/tests/test_summary_renderer.py scripts/tests/test_dossier_renderer.py \
       scripts/tests/test_signals_lib.py scripts/tests/test_tier_v2.py \
       scripts/tests/test_role_tagging.py scripts/tests/test_roster_checker.py \
       scripts/tests/test_triggers.py scripts/tests/test_backfill_signals.py \
       scripts/tests/test_md_format.py scripts/tests/test_baseline_schema.py \
       scripts/tests/test_fetch_game_data.py
```
> 若上面某檔已不存在(名稱微異),用 `git status` 對照後逐一 `git rm` 實際存在者;不要 `git rm` 到 Phase 0-6 新建的檔。
> 也檢查 `scripts/tests/test_predict.py`(舊 predict 的測試)是否存在,存在則一併 `git rm`。

- [ ] **Step 2: 確認沒有殘留 import 指向已刪模組**

Run（Grep tool 或）: `$PYTHON -c "import subprocess,sys; sys.exit(0)"` 之後用 ripgrep:
`rg -l 'import (pitcher_stats|lineup_analyzer|signals_lib|merge_game_data|scoring_formula|lib_tier_v2|lib_role_tagging|roster_checker|dossier_renderer|summary_renderer|prepare_game|fetch_game_data)' scripts/`
Expected: 無輸出(除了已刪的測試已隨之移除)。若有殘留,修正該 import 或一併處理。

- [ ] **Step 3: 跑全測試確認綠燈**

Run: `cd scripts && $PYTHON -m pytest -q`
Expected: 全綠(只剩新模組 + 保留模組的測試:test_config / test_run_model / test_fetch_inputs / test_odds_compare / test_report / test_predict_game / test_backtest_load_v2 / test_backtest_metrics_v2 / test_team_resolver / closing_line 相關)。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: 退役舊預測/敘事腳本(約 6500 行)"
```

### Task 13: 重寫 SKILL.md

**Files:**
- Modify: `SKILL.md`（整檔改寫）

- [ ] **Step 1: 整檔改寫 `SKILL.md`**

```markdown
---
name: mlb-game-analyzer
description: Use when the user asks for MLB single-game or full-day (ET) RL / O-U prediction — team-level expected-runs model vs current Pinnacle line (edge in pp). Outputs RL + O/U only, not money line.
---

# MLB Game Predictor — RL / O-U 預測

## Step 0：建立 ET_NOW（必跑、1 次 tool call）

```bash
$PYTHON -c "from datetime import datetime; from zoneinfo import ZoneInfo; n=datetime.now(ZoneInfo('America/New_York')); print(n.strftime('%Y-%m-%d %H:%M %Z'))"
```
記為 `ET_NOW`。相對日期(今天/明天/昨天)一律以 `ET_NOW.date()` 解析,**不信 system currentDate**。

## Step 1：解析 intent

- 單場 or 當日(ET)全部?
- 使用者是否要「先抓盤口 / 最新 odds」?→ 先跑一次 fetch_odds 存 snapshot。

## Step 2：跑指令

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)

# (可選)先抓當下盤口:
python odds/fetch_odds.py

# 單場:
$PYTHON scripts/predict_game.py --date {ET-YYYY-MM-DD} --matchup {AWAY}@{HOME}

# 當日全部:
$PYTHON scripts/predict_game.py --date {ET-YYYY-MM-DD} --all
```
輸出寫到 `analysis-data/{date}/{AWAY}@{HOME}/`(`features.json` + `prediction.md`)。隊名用英文縮寫(KC / LAA / NYY)。

## Step 3：給使用者

讀該資料夾的 `prediction.md` 念給使用者:RL + O/U 的 model 機率、市場 no-vig、edge(pp)。
**AI 只敘事**:指出哪邊有正 edge、量級、需注意什麼。
⛔ 不喊「下哪邊」、不硬掰 EV%、**不提 Money line**、不改任何數字。

## 注意

⚠️ 係數(σ 等)尚未經回測重新擬合前,edge 數字僅供觀察、不可當下注依據。
```

- [ ] **Step 2: 驗證 SKILL.md 無殘留舊內容**

Run（Grep）: `rg -i 'xwoba|信心|HIGH 80|prepare_game|dossier|信號|tier' SKILL.md`
Expected: 無輸出。

- [ ] **Step 3: Commit**

```bash
git add SKILL.md
git commit -m "docs(skill): SKILL.md 縮為路由+指令+念結果"
```

---

## Self-Review（執行前作者已核對）

- **Spec 覆蓋**:模型(§5→Task 1-3)、fetch_inputs+FIP+RS(§5.1-5.2→Task 4-5)、odds edge(§6-7→Task 6-7)、features.json(§8→Task 8)、orchestrator 單場/--all(§4→Task 9)、回測改 schema(§11→Task 10-11)、刪除(§10→Task 12)、SKILL(§9→Task 13)。打線凍結(§3/§8)在 Task 5 `fetch_lineup_light` + Task 8 features schema。✓
- **型別一致**:`run_model.predict(...)` 鍵(`mu_home/mu_away/mu_margin/mu_total/p_home_ml/p_home_cover_rl/p_over`)在 Task 3 定義,Task 8/9/10 沿用同名。`compute_edges` 回 `home_rl_pp/over_pp`,Task 8/10/11 一致。`market` 結構(`rl.{home_point,home_no_vig,...}` / `total.{line,over_no_vig,under_no_vig}`)Task 7 定義,Task 8/9 沿用。✓
- **無 placeholder**:各 step 均含實際程式碼/指令/預期輸出。Task 11 Step 5 的 render 美化標為「可另開 task」是範圍說明、非邏輯缺口(指標已完整測試)。

---

## 已知尾項（非阻塞,future）

- Poisson/Skellam 機率升級(spec §14)。
- σ_team / 權重用 going-forward 回測重新擬合(spec §15 gate)。
- ablation:用凍結的 lineup / RA 重跑比較(spec §11)。
- `lib/render.py` RL/OU/edge 報告美化(若 Task 11 只做 dict 原貌)。
- `odds/analyze_smart_money.py` + `odds/reports/`:獨立保留,未來決定去留(spec §10)。
