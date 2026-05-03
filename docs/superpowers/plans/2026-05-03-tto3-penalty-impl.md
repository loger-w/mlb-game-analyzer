# TTO3 Penalty Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mlb-game-analyzer` skill 既有 8 個 derived signals 之上新增第 9 個 `tto3_penalty`，量化先發投手第三輪面對打者的 OPS 衰退幅度，dossier `## 投手對決` table 加 visible row 並走 `## 🎯 訊號摘要` + `### 額外信號` 的標準 surface。

**Architecture:** 沿用 `fetch_platoon_splits` 的 MLB Stats API `statSplits` 路徑加 `fetch_tto_splits`（season + career fallback）。新 signal 是 pure function 落在 `signals_lib.py` 與既有 8 條 contract 一致；`_HALF_LIFE_BY_NAME` 增第 9 條 `structural`。`compute_all_signals` per-pitcher loop 加一行；`signals_for_bundle` cache 自動覆蓋 dossier / summary 兩個 surface。**不**動 `merge_game_data.py`、`scoring_formula.py`、Flag 體系。

**Tech Stack:** Python 3.11+、`requests`、`pytest` + `monkeypatch` + `unittest.mock.MagicMock`（既有）、MLB Stats API `statSplits` / `careerStatSplits`。

**Spec reference:** `docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md`

---

## ⚠️ Plan B Amendment（2026-05-03）

Task 1（spike）執行後證實 Plan A 路徑（MLB Stats API `statSplits + sitCodes`）不可行：MLB API 不曝光 Times-Through-Order 資料。已切到 **Plan B**：用 `pybaseball.statcast_pitcher` 拉整季逐球資料自行聚合 TTO 桶。

**任務編號**：1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12（Task 5 在 Plan B 修訂時併入 Task 4 — 編號刻意保留 gap，避免 Tasks 6-12 大量 cross-reference 跟著動）。

詳見 spec `docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md` §0 Plan B Amendment。

---

## File Structure（Plan B）

**修改**：
- `scripts/pitcher_stats.py` — 新 `fetch_tto_splits()` + 3 個 helpers（`_pa_outcome_aggregates()`、`_compute_tto_from_statcast()`、`_has_sufficient_tto3()`）；main 路徑接入 + JSON 寫入
- `scripts/signals_lib.py` — 新 `signal_tto3_penalty()`；`_HALF_LIFE_BY_NAME` 加第 9 條；`compute_all_signals` per-pitcher loop 加一行
- `scripts/dossier_renderer.py` — 新 `_render_tto_splits_cell()` helper；`## 投手對決` table caller 加 row
- `reference/matchup-factors.md` — §Signals 加 §9 條目；半衰期表 structural 列加 `tto3_penalty`
- `CHANGELOG.md` — 移除 line 50 過時條目（`wRC+ / Stuff+ — FanGraphs API non-free，不引入`）；最頂端加新版區塊
- `scripts/tests/test_pitcher_stats.py` — append `_pa_outcome_aggregates` / `_compute_tto_from_statcast` / `fetch_tto_splits` 系列測試
- `scripts/tests/test_signals_lib.py` — append signal_tto3_penalty 系列測試
- `scripts/tests/test_dossier_renderer.py` — append TTO row 系列測試

**不動**：`scoring_formula.py`、`merge_game_data.py`、`flags-checklist.md`、其他 signal 邏輯、`prepare_game.py`、`fetch_game_data.py`、`roster_checker.py`

**現有測試 baseline**：439 tests

**目標**：439 → ~457 tests（+18）

---

## Test Execution

從 `scripts/` 目錄跑：
```bash
cd scripts
python -m pytest tests/ -v
```

或單檔：
```bash
python -m pytest tests/test_pitcher_stats.py -v
python -m pytest tests/test_signals_lib.py -v
python -m pytest tests/test_dossier_renderer.py -v
```

或單測試函式：
```bash
python -m pytest tests/test_signals_lib.py::test_tto3_penalty_fires_ops_medium -v
```

---

## Task 1: Spike — 驗證 MLB API TTO sitCode 字串（DONE 2026-05-03）

**Status**: ✅ **COMPLETED**（spike 在 controller session 已執行；下游 implementer **不需要**重跑）

**結果摘要**：
- 三組候選 sitCode（`ot1,ot2,ot3` / `1,2,3` / `1f,2f,3f`）測試 → 全部 0 splits 回傳
- `careerStatSplits` endpoint 同樣 0 splits
- MLB API `/situationCodes` 元資料端點 602 個 codes 無任何 TTO 切面
- 結論：MLB Stats API **不曝光** Times-Through-Order 切面，**Plan A 路徑死**

**Plan B 可行性確認**（同 spike）：
- `pybaseball.statcast_pitcher('2025-04-01', '2025-04-15', 669373)` → 271 pitches、72 PAs
- 必要欄位 `at_bat_number / batter / pitcher / events / description / game_pk / inning` 全 PRESENT
- events 分布合理（field_out 31, strikeout 23, single 11, walk 3, double 2, home_run 1, force_out 1）

**動作**：
- spec `docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md` 已加 §0 Plan B Amendment + §5.1/§5.3/§5.5 全部就地改寫；commit 已包含
- 本 plan Tasks 2/3/4 都是 Plan B 內容；Task 5 已併入 Task 4（編號 gap 刻意保留）
- 下游 Tasks 2-4 不需要再做 spike，直接照 spec §5.1 Plan B 程式碼實作

---

## Task 2: `_pa_outcome_aggregates` helper

**Files:**
- Modify: `scripts/pitcher_stats.py`（新 helper，append 在 `fetch_platoon_splits` 之後）
- Test: `scripts/tests/test_pitcher_stats.py`（appended）

純函式：把一個 PA-level pandas DataFrame slice（一行一 PA，含 `events` 欄）轉成 `{ops, k_pct, bb_pct, bf}` 字典。OBP / SLG 由 events 計數合成（PA 不直接給 OPS）。

- [ ] **Step 1: 寫 4 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py` 結尾：

```python
# ---------------------------------------------------------------------------
# TTO splits helpers (Plan B — Statcast pitch-by-pitch aggregation)
# ---------------------------------------------------------------------------
import pandas as _pd_mod


def _pa_df(events: list[str]):
    """Build a tiny PA-level DataFrame for tests."""
    return _pd_mod.DataFrame({"events": events})


def test_pa_outcome_aggregates_all_strikeouts():
    """5 strikeouts → OPS=0、K%=100、BB%=0、BF=5。"""
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df(["strikeout"] * 5))
    assert out["bf"] == 5
    assert out["k_pct"] == 100.0
    assert out["bb_pct"] == 0.0
    assert out["ops"] == 0.0


def test_pa_outcome_aggregates_basic_mix():
    """1 single + 1 walk + 1 K + 1 field_out + 1 home_run → 5 PAs，AB=4。

    Hits: 1 + 1HR = 2H, 1B=1, HR=1, TB = 1 + 4 = 5
    BB=1, K=1
    AVG = 2/4 = 0.500
    OBP = (2 + 1) / (4 + 1) = 0.600   # SF=0
    SLG = 5 / 4 = 1.250
    OPS = 1.850
    K% = 1/5 = 20.0
    BB% = 1/5 = 20.0
    """
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([
        "single", "walk", "strikeout", "field_out", "home_run",
    ]))
    assert out["bf"] == 5
    assert out["k_pct"] == 20.0
    assert out["bb_pct"] == 20.0
    assert abs(out["ops"] - 1.850) < 0.005


def test_pa_outcome_aggregates_handles_sf_and_hbp():
    """SF / HBP 不計 AB；HBP 計入 OBP 分子。

    PAs: 2 single, 1 hit_by_pitch, 1 sac_fly, 1 strikeout, 1 field_out → 6 PAs
    AB = 6 - 0(BB) - 1(HBP) - 1(SF) - 0(SH) = 4
    H = 2, TB = 2
    OBP = (2 + 0 + 1) / (4 + 0 + 1 + 1) = 3/6 = 0.500
    SLG = 2/4 = 0.500
    OPS = 1.000
    """
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([
        "single", "single", "hit_by_pitch", "sac_fly", "strikeout", "field_out",
    ]))
    assert out["bf"] == 6
    assert abs(out["ops"] - 1.000) < 0.005


def test_pa_outcome_aggregates_empty_returns_zero_bf():
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([]))
    assert out["bf"] == 0
    assert out["ops"] is None
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k pa_outcome_aggregates -v
```

預期：4 errors（`AttributeError: module 'pitcher_stats' has no attribute '_pa_outcome_aggregates'`）

- [ ] **Step 3: 實作 `_pa_outcome_aggregates`**

Append 到 `scripts/pitcher_stats.py`，建議放在 `fetch_platoon_splits` 之後（line ~522 之後）：

```python
def _pa_outcome_aggregates(pa_df) -> dict:
    """從 PA-level DataFrame slice（一行一 PA，含 events 欄）算 OPS / K% / BB% / BF。

    OBP / SLG / AVG 由 events 計數 + sabermetric 公式合成（PA 級資料不直接給 OPS）。
    Plan B helper — input 是 statcast_pitcher 經 events.notna() filter 過的 slice。
    """
    bf = len(pa_df)
    if bf == 0:
        return {"ops": None, "k_pct": 0.0, "bb_pct": 0.0, "bf": 0}

    events = pa_df["events"]
    h_singles = int((events == "single").sum())
    h_doubles = int((events == "double").sum())
    h_triples = int((events == "triple").sum())
    h_hrs = int((events == "home_run").sum())
    h = h_singles + h_doubles + h_triples + h_hrs

    bb = int((events == "walk").sum())
    hbp = int((events == "hit_by_pitch").sum())
    k = int(events.isin(["strikeout", "strikeout_double_play"]).sum())
    sf = int(events.isin(["sac_fly", "sac_fly_double_play"]).sum())
    sh = int(events.isin(["sac_bunt", "sacrifice_bunt_double_play"]).sum())

    ab = bf - bb - hbp - sf - sh
    if ab <= 0:
        return {"ops": None,
                "k_pct": round(k / bf * 100, 1),
                "bb_pct": round(bb / bf * 100, 1),
                "bf": bf}

    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0
    tb = h_singles + 2 * h_doubles + 3 * h_triples + 4 * h_hrs
    slg = tb / ab if ab > 0 else 0.0
    ops = obp + slg

    return {
        "ops": round(ops, 3),
        "k_pct": round(k / bf * 100, 1),
        "bb_pct": round(bb / bf * 100, 1),
        "bf": bf,
    }
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k pa_outcome_aggregates -v
```

預期：4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): _pa_outcome_aggregates helper (PA events → OPS/K%/BB%)"
```

---

## Task 3: `_compute_tto_from_statcast` helper

**Files:**
- Modify: `scripts/pitcher_stats.py`
- Test: `scripts/tests/test_pitcher_stats.py`

從 `pybaseball.statcast_pitcher` 拉 pitch-by-pitch DataFrame，依 `(game_pk, batter)` 分組 + `at_bat_number` 排序計算每位打者在該場的 PA ordinal，再 PA 級加總成 `tto1` / `tto2` / `tto3` 桶。

- [ ] **Step 1: 寫 3 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py`：

```python
def _statcast_df(rows: list[dict]):
    """Build a fake statcast_pitcher DataFrame for tests."""
    return _pd_mod.DataFrame(rows)


def test_compute_tto_from_statcast_assigns_ordinals(monkeypatch):
    """1 場、3 打者各面對 3 次 = 9 PAs；TTO ordinal 1/2/3 各 3 BF。"""
    rows = []
    ab_num = 1
    for tto_round in range(3):
        for batter in (101, 102, 103):
            rows.append({
                "game_pk": 778001, "at_bat_number": ab_num,
                "batter": batter, "events": "field_out",
            })
            ab_num += 1

    fake_statcast = lambda *args, **kwargs: _statcast_df(rows)
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import _compute_tto_from_statcast
    out = _compute_tto_from_statcast(669373, 2025, 2025)
    assert "error" not in out
    for bucket in ("tto1", "tto2", "tto3"):
        assert bucket in out
        assert out[bucket]["bf"] == 3


def test_compute_tto_from_statcast_empty_df(monkeypatch):
    """statcast_pitcher 回空 DataFrame → error。"""
    fake_statcast = lambda *args, **kwargs: _pd_mod.DataFrame()
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import _compute_tto_from_statcast
    out = _compute_tto_from_statcast(669373, 2025, 2025)
    assert "error" in out


def test_compute_tto_from_statcast_no_pa_events(monkeypatch):
    """DataFrame 有 pitches 但都沒 events（None）→ error No PA events。"""
    fake_statcast = lambda *args, **kwargs: _statcast_df([
        {"game_pk": 778001, "at_bat_number": 1, "batter": 101, "events": None},
        {"game_pk": 778001, "at_bat_number": 1, "batter": 101, "events": None},
    ])
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import _compute_tto_from_statcast
    out = _compute_tto_from_statcast(669373, 2025, 2025)
    assert "error" in out


def test_compute_tto_from_statcast_pybaseball_raises(monkeypatch):
    """statcast_pitcher 拋 exception → error 帶訊息。"""
    def _raise(*args, **kwargs):
        raise RuntimeError("savant down")
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, _raise, None, None),
    )

    from pitcher_stats import _compute_tto_from_statcast
    out = _compute_tto_from_statcast(669373, 2025, 2025)
    assert "error" in out
    assert "savant down" in out["error"]
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k compute_tto_from_statcast -v
```

預期：4 errors（`_compute_tto_from_statcast` 不存在）

- [ ] **Step 3: 實作 `_compute_tto_from_statcast`**

Append 到 `scripts/pitcher_stats.py` 緊接 `_pa_outcome_aggregates` 之後：

```python
def _compute_tto_from_statcast(mlbam_id: int, year_start: int, year_end: int) -> dict:
    """從 pybaseball Statcast 逐球資料聚合成 TTO1 / TTO2 / TTO3 桶。

    對每個 PA（events 非 null 的 row），在 (game_pk, batter) 群組內依
    at_bat_number 升冪排序，cumcount + 1 即 PA ordinal（1st / 2nd / 3rd PA）。
    超過 3rd（4th+ PA）忽略，因為樣本太稀。
    """
    _, statcast_pitcher_fn, _, _ = _import_pybaseball()
    try:
        start = f"{year_start}-03-20"
        end = f"{year_end}-11-05"
        df = statcast_pitcher_fn(start, end, mlbam_id)
        if df is None or df.empty:
            return {"error": "No Statcast data"}

        pa_df = df[df["events"].notna()].copy()
        if pa_df.empty:
            return {"error": "No PA events in Statcast data"}

        pa_df = pa_df.sort_values(["game_pk", "at_bat_number"])
        pa_df["tto_ordinal"] = pa_df.groupby(["game_pk", "batter"]).cumcount() + 1

        result: dict = {}
        for ordinal in (1, 2, 3):
            bucket = pa_df[pa_df["tto_ordinal"] == ordinal]
            if len(bucket) == 0:
                continue
            result[f"tto{ordinal}"] = _pa_outcome_aggregates(bucket)
        return result if result else {"error": "No TTO buckets computed"}
    except Exception as e:
        return {"error": f"statcast TTO compute failed: {e}"}
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k compute_tto_from_statcast -v
```

預期：4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): _compute_tto_from_statcast (Plan B Statcast aggregation)"
```

---

## Task 4: `fetch_tto_splits` orchestrator + main 路徑接入

**Files:**
- Modify: `scripts/pitcher_stats.py`（orchestrator + main 路徑）
- Test: `scripts/tests/test_pitcher_stats.py`

包 `_compute_tto_from_statcast` 成 fallback orchestrator：season 優先；TTO3 BF < 30 → 5-year career；都不夠 → 回 season（caller 走 small_sample）；都失敗 → `{"error": ...}`。同步把 fetch 接進 pitcher_stats main 路徑（原計畫 Task 5 + Task 6 合併）。

- [ ] **Step 1: 寫 5 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py`：

```python
def _build_full_season_df():
    """Build a statcast DataFrame with TTO3 ≥ 30 BF."""
    rows = []
    ab_num = 1
    for game in range(10):
        for tto_round in range(3):
            for batter in range(101, 105):  # 4 batters per round
                rows.append({
                    "game_pk": 778000 + game,
                    "at_bat_number": ab_num,
                    "batter": batter,
                    "events": "single" if tto_round == 2 else "field_out",
                })
                ab_num += 1
    return _statcast_df(rows)


def _build_thin_df(tto3_bf: int):
    """Build a DataFrame with exactly tto3_bf TTO3 PAs."""
    rows = []
    ab_num = 1
    games_needed = max(1, (tto3_bf + 8) // 9)  # 9 batters per game * 1 TTO3 round
    bf_added = 0
    for game in range(games_needed):
        for tto_round in range(3):
            for batter in range(101, 110):
                if tto_round == 2 and bf_added >= tto3_bf:
                    continue
                rows.append({
                    "game_pk": 778000 + game,
                    "at_bat_number": ab_num,
                    "batter": batter,
                    "events": "field_out",
                })
                ab_num += 1
                if tto_round == 2:
                    bf_added += 1
    return _statcast_df(rows)


def test_fetch_tto_splits_season_full(monkeypatch):
    """Season tto3.bf ≥ 30 → source=season，不打 career。"""
    calls = {"n": 0}

    def fake_statcast(*args, **kwargs):
        calls["n"] += 1
        return _build_full_season_df()

    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import fetch_tto_splits
    out = fetch_tto_splits(669373, 2025)
    assert out["source"] == "season"
    assert out["tto3"]["bf"] >= 30
    assert calls["n"] == 1


def test_fetch_tto_splits_falls_back_to_career(monkeypatch):
    """Season tto3.bf < 30 → 改 career；career 充足 → source=career."""
    calls = {"n": 0}

    def fake_statcast(start_dt, end_dt, mlbam):
        calls["n"] += 1
        # 第一次呼叫（season）→ thin；第二次（career window）→ full
        if calls["n"] == 1:
            return _build_thin_df(15)
        return _build_full_season_df()

    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import fetch_tto_splits
    out = fetch_tto_splits(669373, 2025)
    assert out["source"] == "career"
    assert calls["n"] == 2


def test_fetch_tto_splits_both_thin(monkeypatch):
    """Season + career 都 < 30 BF → 回 season（caller 走 small_sample）。"""
    fake_statcast = lambda *a, **k: _build_thin_df(15)
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import fetch_tto_splits
    out = fetch_tto_splits(669373, 2025)
    assert out["source"] == "season"
    assert out["tto3"]["bf"] < 30


def test_fetch_tto_splits_season_error_career_ok(monkeypatch):
    """Season 失敗 → career 補上。"""
    calls = {"n": 0}

    def fake_statcast(start_dt, end_dt, mlbam):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("season pull failed")
        return _build_full_season_df()

    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, fake_statcast, None, None),
    )

    from pitcher_stats import fetch_tto_splits
    out = fetch_tto_splits(669373, 2025)
    assert out["source"] == "career"
    assert calls["n"] == 2


def test_fetch_tto_splits_both_fail_returns_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("savant down")
    monkeypatch.setattr(
        "pitcher_stats._import_pybaseball",
        lambda: (None, _raise, None, None),
    )

    from pitcher_stats import fetch_tto_splits
    out = fetch_tto_splits(669373, 2025)
    assert "error" in out
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k fetch_tto_splits -v
```

預期：5 errors（`fetch_tto_splits` 不存在）

- [ ] **Step 3: 實作 `fetch_tto_splits` + `_has_sufficient_tto3`**

Append 到 `scripts/pitcher_stats.py` 緊接 `_compute_tto_from_statcast` 之後：

```python
_TTO_MIN_BF = 30  # tto3 bucket 最小 BF；不足走 career fallback


def _has_sufficient_tto3(data: dict) -> bool:
    """data 裡 tto3.bf 是否 ≥ _TTO_MIN_BF。error / 缺 tto3 → False。"""
    if "error" in data:
        return False
    tto3 = data.get("tto3") or {}
    return (tto3.get("bf") or 0) >= _TTO_MIN_BF


def fetch_tto_splits(mlbam_id: int, year: int) -> dict:
    """C2.5：取得投手 Times-Through-Order Splits（TTO1 / TTO2 / TTO3）。

    Plan B：用 pybaseball Statcast pitch-by-pitch 自行聚合。
    Season 優先；TTO3 BF < 30 → silent fallback 5-year career window。
    回傳：
      {
        "source": "season" | "career",
        "tto1": {...}, "tto2": {...}, "tto3": {...},
      }
      或 {"error": "..."} 兩條路徑都失敗時。

    Caller (signal_tto3_penalty) 看 tto3.bf 自行判斷 small_sample。
    """
    season_data = _compute_tto_from_statcast(mlbam_id, year, year)
    if _has_sufficient_tto3(season_data):
        season_data["source"] = "season"
        return season_data

    career_data = _compute_tto_from_statcast(mlbam_id, year - 4, year)
    if _has_sufficient_tto3(career_data):
        career_data["source"] = "career"
        return career_data

    if "error" not in season_data:
        season_data["source"] = "season"
        return season_data
    if "error" not in career_data:
        career_data["source"] = "career"
        return career_data
    return {"error": season_data.get("error", "TTO splits unavailable")}
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k fetch_tto_splits -v
```

預期：5 passed

- [ ] **Step 5: 把 `fetch_tto_splits` 接進 pitcher_stats main 路徑**

`scripts/pitcher_stats.py` 第 887 行（`platoon_splits = fetch_platoon_splits(pitcher_id, args.year)` 那行）後，append 一行：

```python
    platoon_splits = fetch_platoon_splits(pitcher_id, args.year)
    tto_splits = fetch_tto_splits(pitcher_id, args.year)
```

`scripts/pitcher_stats.py` 第 924 行（`"platoon_splits": platoon_splits,` 那行）後，append 一行：

```python
        "platoon_splits": platoon_splits,
        "tto_splits": tto_splits,
```

- [ ] **Step 6: 跑全部既有測試確認沒撞到**

```bash
cd scripts
python -m pytest tests/ -v --tb=short
```

預期：所有既有測試（439 + 13 個 Tasks 2-4 新加 = 452）全 pass。

- [ ] **Step 7: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): fetch_tto_splits orchestrator + main 路徑接入"
```

---

## Task 7: `signal_tto3_penalty` — 第 9 個 derived signal

**Files:**
- Modify: `scripts/signals_lib.py`（在 `signal_core_il_count` 之後、`compute_all_signals` 之前新增 signal）
- Test: `scripts/tests/test_signals_lib.py`（appended）

純函式：把 `tto_splits` dict 轉成 signal contract dict。Fire 條件：OPS Δ ≥ 0.100 OR K% drop ≥ 3pp，TTO3 BF ≥ 30。

- [ ] **Step 1: 寫 7 個 failing tests**

Append 到 `scripts/tests/test_signals_lib.py` 結尾：

```python
# ---------------------------------------------------------------------------
# signal_tto3_penalty — 3rd-time-through-order OPS uplift signal (#9)
# ---------------------------------------------------------------------------

def _make_tto_splits(*, ops1=0.700, ops3=0.810, k1=28.0, k3=23.0,
                     bf3=180, source="season"):
    """Helper：build canonical tto_splits dict for tests."""
    return {
        "source": source,
        "tto1": {"ops": ops1, "k_pct": k1, "bb_pct": 7.0, "bf": 320},
        "tto2": {"ops": (ops1 + ops3) / 2, "k_pct": (k1 + k3) / 2, "bb_pct": 7.5, "bf": 290},
        "tto3": {"ops": ops3, "k_pct": k3, "bb_pct": 8.0, "bf": bf3},
    }


def test_tto3_penalty_fires_ops_medium():
    """OPS Δ +0.110 → fires medium。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.810, k1=28, k3=27))
    _signal_contract(s)
    assert s["fired"] is True
    assert s["severity"] == "medium"
    assert abs(s["value"] - 0.110) < 1e-6
    assert "TTO3 penalty" in s["label"]
    assert s["confidence"] == "data"
    assert s["half_life"] == "structural"


def test_tto3_penalty_fires_ops_high():
    """OPS Δ +0.155 → fires high。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.855, k1=28, k3=27))
    assert s["fired"] is True
    assert s["severity"] == "high"


def test_tto3_penalty_fires_k_drop_only():
    """OPS Δ +0.050（< 0.100 不 fire ops）+ K% Δ -4pp → fires by K trigger。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.750, k1=28, k3=24))
    assert s["fired"] is True
    assert s["severity"] == "medium"  # 不到 high OPS 閾值
    assert "K%" in s["label"]


def test_tto3_penalty_fires_both_ops_and_k():
    """OPS Δ +0.130 + K% Δ -4pp → fires medium，label 同時含兩段。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.830, k1=28, k3=24))
    assert s["fired"] is True
    assert s["severity"] == "medium"
    assert "TTO3 penalty" in s["label"]
    assert "K%" in s["label"]


def test_tto3_penalty_no_fire():
    """OPS Δ +0.060 + K% Δ -1pp → no fire。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.760, k1=28, k3=27))
    assert s["fired"] is False
    assert "value" in s


def test_tto3_penalty_small_sample_below_30_bf():
    """tto3.bf = 25 → no fire + confidence=small_sample。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(ops1=0.700, ops3=0.900, bf3=25))
    assert s["fired"] is False
    assert s["confidence"] == "small_sample"


def test_tto3_penalty_career_source_marks_heuristic():
    """source=career + fire → confidence=heuristic、label 後綴 (career fallback)。"""
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(_make_tto_splits(
        ops1=0.700, ops3=0.810, k1=28, k3=27, source="career",
    ))
    assert s["fired"] is True
    assert s["confidence"] == "heuristic"
    assert "career" in s["label"].lower()


def test_tto3_penalty_handles_none_input():
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty(None)
    assert s["fired"] is False
    assert s["confidence"] == "small_sample"


def test_tto3_penalty_handles_error_input():
    from signals_lib import signal_tto3_penalty
    s = signal_tto3_penalty({"error": "fetch failed"})
    assert s["fired"] is False
    assert s["confidence"] == "small_sample"
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_signals_lib.py -k tto3_penalty -v
```

預期：9 errors（`signal_tto3_penalty` 不存在）

- [ ] **Step 3: 在 `signals_lib.py` 註冊 half_life + 新增 signal function**

第 47–56 行 `_HALF_LIFE_BY_NAME` 加第 9 條：

```python
_HALF_LIFE_BY_NAME = {
    "tier_mismatch": "structural",
    "heat_vs_babip": "short",
    "platoon_advantage": "medium",
    "strong_park": "structural",
    "reverse_platoon": "medium",
    "chain_break": "medium",
    "pitch_mix_concentration": "medium",
    "core_il_count": "short",
    "tto3_penalty": "structural",  # ← 新增
}
```

緊接現有第 8 個 signal `signal_core_il_count`（line ~425）之後，新增第 9 個：

```python
# ---------------------------------------------------------------------------
# 9. tto3_penalty — pitcher's TTO3 OPS uplift vs TTO1 (3rd-time-through curve)
# ---------------------------------------------------------------------------

_TTO3_OPS_DELTA_FIRE = 0.100   # ≥ 0.100 → medium fire
_TTO3_OPS_DELTA_HIGH = 0.150   # ≥ 0.150 → high fire
_TTO3_K_DROP_FIRE = 3.0        # K% drop ≥ 3 percentage points → medium fire
_TTO3_MIN_BF = 30              # require ≥ 30 BF in tto3 bucket


def signal_tto3_penalty(tto_splits: dict | None) -> dict:
    """Surface starters whose TTO3 OPS uplift exceeds league-typical curve.

    Fires when (any of):
      - tto3.ops - tto1.ops ≥ 0.100  → medium (≥ 0.150 → high)
      - tto3.k_pct - tto1.k_pct ≤ -3.0 (K% drop ≥ 3pp) → medium

    half_life: structural (multi-year stuff/arsenal/stamina trait).
    Confidence: data (season) or heuristic (career fallback).
    Small sample: tto3.bf < 30 → no_fire + confidence=small_sample.

    Pre-game data only; AI in summary judges bullpen-load implications.
    Does NOT auto-trigger run value adjustment.
    """
    name = "tto3_penalty"
    if not tto_splits or "error" in tto_splits:
        return _make(name, False, confidence="small_sample")

    tto1 = tto_splits.get("tto1") or {}
    tto3 = tto_splits.get("tto3") or {}
    bf3 = tto3.get("bf") or 0
    if bf3 < _TTO3_MIN_BF:
        return _make(name, False, confidence="small_sample",
                     details={"tto3_bf": bf3})

    ops1 = _to_float(tto1.get("ops"))
    ops3 = _to_float(tto3.get("ops"))
    if ops1 is None or ops3 is None:
        return _make(name, False, confidence="small_sample")

    k1 = _to_float(tto1.get("k_pct"))
    k3 = _to_float(tto3.get("k_pct"))
    has_k = k1 is not None and k3 is not None

    ops_delta = ops3 - ops1
    k_delta = (k3 - k1) if has_k else 0.0

    fired_ops = ops_delta >= _TTO3_OPS_DELTA_FIRE
    fired_k = has_k and k_delta <= -_TTO3_K_DROP_FIRE

    if not (fired_ops or fired_k):
        return _make(name, False, value=round(ops_delta, 3),
                     details={"tto3_bf": bf3,
                              "source": tto_splits.get("source", "season")})

    severity = "high" if ops_delta >= _TTO3_OPS_DELTA_HIGH else "medium"
    source = tto_splits.get("source", "season")
    confidence = "data" if source == "season" else "heuristic"

    label = (
        f"TTO3 penalty:OPS Δ +{ops_delta:.3f}（TTO1 {ops1:.3f} → TTO3 {ops3:.3f}），"
        f"第三輪明顯衰退"
    )
    if fired_k:
        label += f"；K% 從 {k1:.1f}% 掉到 {k3:.1f}%（Δ {k_delta:+.1f}pp）"
    if source == "career":
        label += "(career fallback)"

    return _make(
        name, True, value=round(ops_delta, 3), severity=severity, label=label,
        details={
            "ops_delta": round(ops_delta, 3),
            "k_delta": round(k_delta, 1) if has_k else None,
            "tto1_ops": ops1, "tto3_ops": ops3,
            "tto3_bf": bf3, "source": source,
        },
        confidence=confidence,
    )
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_signals_lib.py -k tto3_penalty -v
```

預期：9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/signals_lib.py scripts/tests/test_signals_lib.py
git commit -m "feat(signals): signal_tto3_penalty (#9) + half_life=structural"
```

---

## Task 8: 把 `signal_tto3_penalty` 接進 `compute_all_signals`

**Files:**
- Modify: `scripts/signals_lib.py`（per-pitcher loop）
- Test: `scripts/tests/test_signals_lib.py`

`compute_all_signals` per-pitcher loop 加一行；同時加一個 schema 完整性 test 確認 `_HALF_LIFE_BY_NAME` 9 條都有。

- [ ] **Step 1: 寫 2 個 failing tests**

Append 到 `scripts/tests/test_signals_lib.py`：

```python
def test_tto3_penalty_in_compute_all_signals():
    """compute_all_signals 對 home + away 各算一次 tto3_penalty。"""
    from signals_lib import compute_all_signals
    bundle = {
        "home_pitcher": {
            "tto_splits": _make_tto_splits(ops1=0.700, ops3=0.810),
        },
        "away_pitcher": {
            "tto_splits": _make_tto_splits(ops1=0.690, ops3=0.730),  # no fire
        },
        "home_lineup": {}, "away_lineup": {}, "merged": {},
    }
    out = compute_all_signals(bundle)
    tto = [s for s in out["signals"] if s["name"] == "tto3_penalty"]
    assert len(tto) == 2
    sides = {s["side"] for s in tto}
    assert sides == {"HOME", "AWAY"}
    home_tto = next(s for s in tto if s["side"] == "HOME")
    away_tto = next(s for s in tto if s["side"] == "AWAY")
    assert home_tto["fired"] is True
    assert away_tto["fired"] is False


def test_half_life_registry_includes_tto3():
    from signals_lib import _HALF_LIFE_BY_NAME
    assert _HALF_LIFE_BY_NAME["tto3_penalty"] == "structural"
    # Confirm 9 條 entry（既有 8 + 新增 tto3_penalty）
    assert len(_HALF_LIFE_BY_NAME) == 9
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_signals_lib.py -k "tto3_penalty_in_compute_all_signals or half_life_registry_includes_tto3" -v
```

預期：1 fail（`compute_all_signals` 還沒呼叫 `signal_tto3_penalty`）+ 1 pass（registry test 已在 Task 7 通過）。

> 註：如 Task 7 step 3 修改 `_HALF_LIFE_BY_NAME` 已包含 9 條，第 2 個 test 會直接 pass。

- [ ] **Step 3: 修改 `compute_all_signals` per-pitcher loop**

`scripts/signals_lib.py` per-pitcher loop（line ~462）加一行：

```python
    # Per-pitcher signals (tier_mismatch, reverse_platoon, pitch_mix_concentration, tto3_penalty)
    for side, p in (("HOME", home_p), ("AWAY", away_p)):
        signals.append(_tag(signal_tier_mismatch(p.get("tier_gap")), side))
        signals.append(_tag(
            signal_reverse_platoon(p.get("platoon_splits"), p.get("pitch_hand")),
            side,
        ))
        statcast = p.get("statcast") or {}
        signals.append(_tag(
            signal_pitch_mix_concentration(statcast.get("pitch_types")),
            side,
        ))
        signals.append(_tag(signal_tto3_penalty(p.get("tto_splits")), side))  # ← 新增
```

- [ ] **Step 4: 跑全部 signals 測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_signals_lib.py -v
```

預期：既有 + Task 7 9 個 + Task 8 2 個 = 全 pass。

- [ ] **Step 5: Commit**

```bash
git add scripts/signals_lib.py scripts/tests/test_signals_lib.py
git commit -m "feat(signals): wire tto3_penalty into compute_all_signals per-pitcher loop"
```

---

## Task 9: Dossier `## 投手對決` table — 加 visible row「TTO splits」

**Files:**
- Modify: `scripts/dossier_renderer.py`（新 helper + table caller）
- Test: `scripts/tests/test_dossier_renderer.py`（appended）

加 `_render_tto_splits_cell(pitcher)` helper；在 `## 投手對決` table 緊接 vs LHB / vs RHB row 之後加一個 row「TTO splits」。

- [ ] **Step 1: 找 dossier 的 `## 投手對決` table 區塊位置**

```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py -k pitcher -v --co
```

或：

```bash
grep -n "vs LHB\|vs RHB\|## 投手對決\|pitcher_table\|matchup" dossier_renderer.py
```

記錄 vs LHB / vs RHB row 渲染的函式 + 行號（後續 Step 3 在那裡 inject）。

- [ ] **Step 2: 寫 3 個 failing tests**

Append 到 `scripts/tests/test_dossier_renderer.py`：

```python
def test_pitcher_table_includes_tto_row_season():
    """tto_splits source=season + 充足樣本 → table row 含 TTO1/2/3 OPS + Δ。"""
    from dossier_renderer import _render_tto_splits_cell
    pitcher = {
        "tto_splits": {
            "source": "season",
            "tto1": {"ops": 0.700, "k_pct": 28.0, "bb_pct": 7.0, "bf": 320},
            "tto2": {"ops": 0.740, "k_pct": 26.5, "bb_pct": 7.5, "bf": 290},
            "tto3": {"ops": 0.810, "k_pct": 23.0, "bb_pct": 8.0, "bf": 180},
        },
    }
    cell = _render_tto_splits_cell(pitcher)
    assert "TTO1" in cell and "TTO3" in cell
    assert ".700" in cell and ".810" in cell
    assert "Δ+0.110" in cell
    assert "180 BF" in cell
    assert "(career)" not in cell


def test_pitcher_table_tto_row_career_suffix():
    """source=career → cell 後綴「(career)」。"""
    from dossier_renderer import _render_tto_splits_cell
    pitcher = {
        "tto_splits": {
            "source": "career",
            "tto1": {"ops": 0.680, "k_pct": 25.0, "bb_pct": 8.0, "bf": 1500},
            "tto2": {"ops": 0.715, "k_pct": 24.0, "bb_pct": 8.5, "bf": 1300},
            "tto3": {"ops": 0.755, "k_pct": 22.0, "bb_pct": 9.0, "bf": 800},
        },
    }
    cell = _render_tto_splits_cell(pitcher)
    assert "(career)" in cell
    assert "Δ+0.075" in cell


def test_pitcher_table_tto_row_small_sample():
    """tto3.bf=20 → 「n/a (sample <30 BF)」。"""
    from dossier_renderer import _render_tto_splits_cell
    pitcher = {
        "tto_splits": {
            "source": "season",
            "tto1": {"ops": 0.700, "bf": 50},
            "tto2": {"ops": 0.740, "bf": 40},
            "tto3": {"ops": 0.810, "bf": 20},
        },
    }
    assert _render_tto_splits_cell(pitcher) == "n/a (sample <30 BF)"


def test_pitcher_table_tto_row_missing_key():
    """投手缺 tto_splits key（schema 向下相容）→ 「n/a」。"""
    from dossier_renderer import _render_tto_splits_cell
    assert _render_tto_splits_cell({}) == "n/a"
    assert _render_tto_splits_cell(None) == "n/a"


def test_pitcher_table_tto_row_error():
    """tto_splits = {error: ...} → 「n/a」。"""
    from dossier_renderer import _render_tto_splits_cell
    pitcher = {"tto_splits": {"error": "fetch failed"}}
    assert _render_tto_splits_cell(pitcher) == "n/a"
```

- [ ] **Step 3: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py -k tto -v
```

預期：5 errors（`_render_tto_splits_cell` 不存在）

- [ ] **Step 4: 實作 `_render_tto_splits_cell` helper**

Append 到 `scripts/dossier_renderer.py`（位置：找一個 helper 集中區塊，比如 `_arsenal_top3_str` 附近；不存在就在檔尾 `if __name__` 之前）：

```python
def _render_tto_splits_cell(pitcher: dict | None) -> str:
    """渲染 ## 投手對決 table 的「TTO splits」 cell。

    格式：`TTO1 .700 / TTO2 .740 / TTO3 .810 (Δ+0.110, 180 BF)`
    source=career → 後綴「(career)」
    缺 key / fetch error → 「n/a」
    tto3.bf < 30 → 「n/a (sample <30 BF)」
    """
    if not pitcher:
        return "n/a"
    tto = pitcher.get("tto_splits")
    if not tto or "error" in tto:
        return "n/a"
    tto1 = tto.get("tto1") or {}
    tto2 = tto.get("tto2") or {}
    tto3 = tto.get("tto3") or {}
    bf3 = tto3.get("bf") or 0
    if bf3 < 30:
        return "n/a (sample <30 BF)"
    o1, o2, o3 = tto1.get("ops"), tto2.get("ops"), tto3.get("ops")
    if o1 is None or o3 is None:
        return "n/a"
    delta = o3 - o1
    suffix = " (career)" if tto.get("source") == "career" else ""
    o2_str = f"{o2:.3f}" if o2 is not None else "?"
    return (
        f"TTO1 {o1:.3f} / TTO2 {o2_str} / TTO3 {o3:.3f} "
        f"(Δ{delta:+.3f}, {bf3} BF){suffix}"
    )
```

- [ ] **Step 5: 把 row 接進 `## 投手對決` table**

依照 Step 1 找到的 vs LHB / vs RHB row 渲染處，在那 row 之後 append：

```python
# 假設既有渲染長這樣：
table_rows.append(f"| vs LHB | {away_lhb} | {home_lhb} |")
table_rows.append(f"| vs RHB | {away_rhb} | {home_rhb} |")
# 緊接著加：
table_rows.append(
    f"| TTO splits | {_render_tto_splits_cell(away_pitcher)} | "
    f"{_render_tto_splits_cell(home_pitcher)} |"
)
```

> 實際變數名稱依現有 dossier_renderer 結構調整。如果 vs LHB / vs RHB row 在 `<details>` 折疊塊裡（spec §5「visible row 不入 `<details>`」），TTO row 必須**外**於 `<details>`，跟 visible 4 row 同層。

- [ ] **Step 6: 加 1 個 integration test 確認 row 真的進 table 輸出**

Append 到 `scripts/tests/test_dossier_renderer.py`：

```python
def test_dossier_pitcher_table_includes_tto_row_in_output():
    """跑完整 render，確認 TTO splits row 文字出現在輸出 markdown。"""
    from dossier_renderer import render_dossier  # 假設這是 entry point
    bundle = {
        "home_pitcher": {
            "name": "Skubal", "pitch_hand": "L",
            "tier_v2": {"score": 90, "tier": "Elite"},
            "tto_splits": {
                "source": "season",
                "tto1": {"ops": 0.650, "bf": 200},
                "tto2": {"ops": 0.690, "bf": 180},
                "tto3": {"ops": 0.720, "bf": 100},
            },
        },
        "away_pitcher": {
            "name": "Cole", "pitch_hand": "R",
            "tier_v2": {"score": 80, "tier": "Strong"},
            "tto_splits": {
                "source": "season",
                "tto1": {"ops": 0.700, "bf": 200},
                "tto2": {"ops": 0.740, "bf": 180},
                "tto3": {"ops": 0.810, "bf": 100},
            },
        },
        "home_lineup": {"lineup": []}, "away_lineup": {"lineup": []},
        "merged": {},
    }
    md = render_dossier(bundle)
    assert "| TTO splits |" in md
    assert "TTO1 .650" in md
    assert "TTO1 .700" in md
```

> 如 entry point 名稱不是 `render_dossier`，依實際 export 名稱調整（`grep "^def" dossier_renderer.py` 找）。

- [ ] **Step 7: 跑全部 dossier 測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py -v
```

預期：既有 + 6 新測試全 pass。

- [ ] **Step 8: Commit**

```bash
git add scripts/dossier_renderer.py scripts/tests/test_dossier_renderer.py
git commit -m "feat(dossier): 投手對決 table 加 TTO splits visible row"
```

---

## Task 10: `reference/matchup-factors.md` — §Signals 加 §9 + 半衰期表

**Files:**
- Modify: `reference/matchup-factors.md`

純 docs 異動。

- [ ] **Step 1: 加 §9 條目**

打開 `reference/matchup-factors.md`，找到第 257–258 行 `#### 8. core_il_count` 區塊結尾。緊接其後（在「Signals 與紀律 Flag 的關係」表之前）插入：

```markdown
#### 9. tto3_penalty（投手）
- 觸發：TTO3 OPS - TTO1 OPS ≥ 0.100 → medium，≥ 0.150 → high；OR K% drop ≥ 3pp
- 樣本：TTO3 BF ≥ 30；season 不足 fallback career（confidence: heuristic）
- 範例：starter TTO1 .700 / TTO3 .810（Δ +0.110）→ 第三輪 OPS 等同聯盟平均打者
- AI 判讀：
  - TTO3 弱（fire）→ 教練可能提早換投，後段牛棚負擔 ↑
  - 同時對手 `core_il_count` fire（牛棚薄）→ 後段失分風險 ↑、總分判讀偏多
  - TTO3 強（不 fire）→ 隱性訊號，AI 可從 dossier `## 投手對決` 表格直接讀「能撐第三輪 → 牛棚消耗少」
- ⛔ **不自動 ±run value**（與 §3 / §8 紀律一致）
```

- [ ] **Step 2: 更新半衰期表 structural 列**

在第 274–278 行半衰期表，把：

```markdown
| structural | （無） | tier_mismatch / strong_park | 多年 / season-to-date 累計，反身慢，**正常引用** |
```

改為：

```markdown
| structural | （無） | tier_mismatch / strong_park / tto3_penalty | 多年 / season-to-date 累計，反身慢，**正常引用** |
```

- [ ] **Step 3: 跑全部測試確認沒撞到（純 docs 但保險）**

```bash
cd scripts
python -m pytest tests/ -v --tb=short
```

預期：所有測試 pass。

- [ ] **Step 4: Commit**

```bash
git add reference/matchup-factors.md
git commit -m "docs(reference): matchup-factors §Signals — 加 §9 tto3_penalty"
```

---

## Task 11: `CHANGELOG.md` — 移除 line 50 過時條目 + 加新版區塊

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 移除過時 line 50 條目**

打開 `CHANGELOG.md`，找到第 50 行：

```markdown
- **wRC+ / Stuff+** — FanGraphs API non-free，不引入
```

刪除整行（5/3 session 已實作 wRC+ commit `df165ab` + Stuff+ commit `ca7d8a1`）。

- [ ] **Step 2: 抓出 Tasks 2 / 3 / 4 / 7 / 8 / 9 / 10 的 commit short hashes**

```bash
cd C:/Users/Loger/.claude/skills/mlb-game-analyzer
git log --oneline -15
```

抓出 7 個 commit short hash（每個 Task 一筆 commit；Task 4 含 orchestrator + main wire 是單一 commit）。Task 1 spike 結果已寫進 spec / plan 並 commit；spec 改 Plan B 也已 commit；這兩個 commit 一併列入 CHANGELOG。

- [ ] **Step 3: 在最頂端加新版區塊**

在現有 `## 2026-05-03 — Path B refactor` 區塊之上插入下面內容；把 `<HASHN>` 佔位符替換為 Step 2 抓到的真實 short hash：

```markdown
## 2026-05-04 — TTO3 penalty signal（signal #9，Plan B）

第 9 個 derived signal，pitcher-side per-game。先發投手第三輪面對打者 OPS
衰退幅度，覆蓋 PR-3 後 line 48「第二批 signals」第一項。Plan A（MLB API
statSplits + sitCodes）spike 後證實 MLB API 不曝光 TTO 切面；改走 Plan B 用
pybaseball Statcast pitch-by-pitch 自行聚合。

- **commit <HASH_SPEC>** `docs(spec)`: TTO3 — Plan B amendment after Plan A spike disproved
- **commit <HASH2>** `feat(pitcher)`: `_pa_outcome_aggregates` helper (PA events → OPS/K%/BB%)
- **commit <HASH3>** `feat(pitcher)`: `_compute_tto_from_statcast` (Plan B Statcast aggregation)
- **commit <HASH4>** `feat(pitcher)`: `fetch_tto_splits` orchestrator + main 路徑接入
- **commit <HASH7>** `feat(signals)`: `signal_tto3_penalty` (#9) + half_life=structural
- **commit <HASH8>** `feat(signals)`: wire tto3_penalty into compute_all_signals
- **commit <HASH9>** `feat(dossier)`: 投手對決 table 加 TTO splits visible row
- **commit <HASH10>** `docs(reference)`: matchup-factors §Signals §9 + 半衰期表

### 紀律保留

- ✅ 信號**不入 scoring formula**（一致 §3 / §8）
- ✅ 既有 8 signals 行為零變動（compute_all_signals 只追加一行）
- ✅ 4 月小樣本 season → 5-year career silent fallback，BF < 30 統一 small_sample no_fire
- ✅ Dossier TTO row 無條件顯示（mirror vs LHB / vs RHB pattern）
- ✅ `merge_game_data.py` / `prepare_game.py` / `scoring_formula.py` / Flag 體系全部不動

### Out of scope（下批）

- TTO4+ penalty（樣本太稀）
- Reliever inheritance penalty
- 動態調整觸發閾值（按 tier 別）— 留至 backtest 階段
- 休息天數 / 上一場用球數（CHANGELOG line 48 第二批 signals 中的另兩項）
```

- [ ] **Step 4: Commit（CHANGELOG 是 Task 11 唯一 commit；Task 11 自身的 hash 不寫進 body）**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): TTO3 penalty signal #9 上線；移除過時 wRC+/Stuff+ 條目"
```

---

## Task 12: End-to-end smoke test — 跑一場真實比賽

**Files:** （無 commit；驗證運行）

驗證 pipeline 端到端產出 dossier 含 TTO row + signal 正確 fire / no_fire。

- [ ] **Step 1: 選一場 5/3 場次**

```bash
ls "C:/Users/Loger/.claude/skills/mlb-game-analyzer/analysis-data/2026-05-03"
```

選任一場（建議含完整 official lineup 的場次）。記錄 home / away abbr。

- [ ] **Step 2: 重跑 prepare_game.py**

```bash
cd C:/Users/Loger/.claude/skills/mlb-game-analyzer
python scripts/prepare_game.py --date 2026-05-03 --home <HOME_ABBR> --away <AWAY_ABBR> --force
```

監看 stderr：
- 若有 `TTO splits unavailable` 警告 → 可能 sitCode 未鎖好或網路問題
- 若無警告 → 進 Step 3

- [ ] **Step 3: 檢查產出檔**

```bash
cd analysis-data/2026-05-03/<AWAY>@<HOME>
cat home_pitcher.json | python -m json.tool | grep -A 8 tto_splits
cat away_pitcher.json | python -m json.tool | grep -A 8 tto_splits
```

預期：每個 pitcher.json 都有 `"tto_splits": {"source": "season"|"career", "tto1": {...}, "tto2": {...}, "tto3": {...}}` 或 `{"error": ...}`。

- [ ] **Step 4: 檢查 dossier**

```bash
grep -A 1 "TTO splits" dossier.md
grep -A 1 "TTO3 penalty" dossier.md
```

預期：
- `| TTO splits | ... |` row 在 `## 投手對決` table 內
- 若 fire：`## 🎯 訊號摘要` 段含 `🟠 TTO3 penalty` 或 `🔴 TTO3 penalty`
- 若 no fire：訊號摘要不含 TTO3 條目（visible row 仍存在）

- [ ] **Step 5: 檢查 summary**

```bash
grep -A 3 "額外信號" summary.md
```

預期：fired TTO3 出現在 `### 額外信號` 段（若 fire），no_fire 則不出現。

- [ ] **Step 6: 全測試 final 確認**

```bash
cd scripts
python -m pytest tests/ -v --tb=short
```

預期：~457 tests 全 pass（439 baseline + ~18 新增 Plan B）。

如果都 pass，task 完成。Task 11 CHANGELOG 區塊裡的 `<HASHN>` 佔位符應該已經在 Task 11 Step 2-3 替換成真實 hash（在 Task 11 commit 前完成），不需事後 amend。

---

## Spec coverage 自我驗證表（Plan B）

| Spec 段落 | 對應 task | 驗證點 |
|---|---|---|
| §2 Goals 1: signal_tto3_penalty 落 signals_lib.py | Task 7 | 單元測試 9 個 |
| §2 Goals 2: fetch_tto_splits（Plan B Statcast 路徑） | Task 4 | 5 個 orchestrator 測試 |
| §2 Goals 3: 4 月 fallback career, heuristic | Task 4 + Task 7 | `test_fetch_tto_splits_falls_back_to_career` + `test_tto3_penalty_career_source_marks_heuristic` |
| §2 Goals 4: dossier visible row | Task 9 | 單元測試 + integration test |
| §2 Goals 5: dossier 訊號摘要 + summary 額外信號 | Task 8（compute_all_signals 接入後 cache 自動帶） | Task 12 smoke §3 / §5 |
| §2 Goals 6: matchup-factors §9 + 半衰期表 | Task 10 | docs review |
| §2 Goals 7: CHANGELOG line 50 清理 | Task 11 | docs review |
| §3 Non-Goals: 不進 scoring formula | 所有 task | scoring_formula.py 0 異動 |
| §3 Non-Goals: 不動 merge_game_data | 所有 task | merge_game_data.py 0 異動 |
| §3 Non-Goals: RP / opener no_fire | Task 7 | `test_tto3_penalty_small_sample_below_30_bf` 涵蓋 |
| §5.1 Plan B helper（_pa_outcome_aggregates） | Task 2 | 4 個單元測試 |
| §5.1 Plan B helper（_compute_tto_from_statcast） | Task 3 | 4 個單元測試 |
| §5.3 spike outcome | Task 1 | DONE — Plan A dead, Plan B viable |
| §5.4 fallback 矩陣 | Task 4 | 5 個 test 涵蓋 6 條矩陣 row |
| §5.5 PA outcome 映射表 | Task 2 | `test_pa_outcome_aggregates_basic_mix` / `_handles_sf_and_hbp` |
| §6.1 signal contract | Task 7 | `_signal_contract(s)` helper assertion |
| §7.1 dossier helper（n/a / career suffix / small_sample / 缺 key） | Task 9 | 5 個 test |
| §9 Tests 列表 | Tasks 2 / 3 / 4 / 7 / 8 / 9 | ~25 新測試（spec 估 +18，含 dossier integration） |

