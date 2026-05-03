# TTO3 Penalty Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mlb-game-analyzer` skill 既有 8 個 derived signals 之上新增第 9 個 `tto3_penalty`，量化先發投手第三輪面對打者的 OPS 衰退幅度，dossier `## 投手對決` table 加 visible row 並走 `## 🎯 訊號摘要` + `### 額外信號` 的標準 surface。

**Architecture:** 沿用 `fetch_platoon_splits` 的 MLB Stats API `statSplits` 路徑加 `fetch_tto_splits`（season + career fallback）。新 signal 是 pure function 落在 `signals_lib.py` 與既有 8 條 contract 一致；`_HALF_LIFE_BY_NAME` 增第 9 條 `structural`。`compute_all_signals` per-pitcher loop 加一行；`signals_for_bundle` cache 自動覆蓋 dossier / summary 兩個 surface。**不**動 `merge_game_data.py`、`scoring_formula.py`、Flag 體系。

**Tech Stack:** Python 3.11+、`requests`、`pytest` + `monkeypatch` + `unittest.mock.MagicMock`（既有）、MLB Stats API `statSplits` / `careerStatSplits`。

**Spec reference:** `docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md`

---

## File Structure

**修改**：
- `scripts/pitcher_stats.py` — 新 `fetch_tto_splits()` + 3 個 helpers（`_fetch_tto_one`、`_classify_tto_bucket`、`_parse_ops_with_fallback`）；main 路徑接入 + JSON 寫入
- `scripts/signals_lib.py` — 新 `signal_tto3_penalty()`；`_HALF_LIFE_BY_NAME` 加第 9 條；`compute_all_signals` per-pitcher loop 加一行
- `scripts/dossier_renderer.py` — 新 `_render_tto_splits_cell()` helper；`## 投手對決` table caller 加 row
- `reference/matchup-factors.md` — §Signals 加 §9 條目；半衰期表 structural 列加 `tto3_penalty`
- `CHANGELOG.md` — 移除 line 50 過時條目（`wRC+ / Stuff+ — FanGraphs API non-free，不引入`）；最頂端加新版區塊
- `scripts/tests/test_pitcher_stats.py` — append fetch_tto_splits 系列測試
- `scripts/tests/test_signals_lib.py` — append signal_tto3_penalty 系列測試
- `scripts/tests/test_dossier_renderer.py` — append TTO row 系列測試

**不動**：`scoring_formula.py`、`merge_game_data.py`、`flags-checklist.md`、其他 signal 邏輯、`prepare_game.py`、`fetch_game_data.py`、`roster_checker.py`

**現有測試 baseline**：439 tests（`git log --oneline | head -1` 應為 `babae1b`）

**目標**：439 → ~454 tests（+15）

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

## Task 1: Spike — 驗證 MLB API TTO sitCode 字串

**Files:** （exploratory，不 commit）

這一步**不**走 TDD，是 5 分鐘人工驗證，目的是把 spec §5.3 提到的「sitCode 三組候選哪一組通」鎖定下來。三組都失敗時 STOP。

- [ ] **Step 1: 跑 spike script 對三組候選輪流測試**

```python
# 直接在 python REPL 或 scripts/_spike_tto.py 執行
import requests

PID = 669373  # Tarik Skubal (DET, 2025 完整賽季先發)
YEAR = 2025
BASE = "https://statsapi.mlb.com/api/v1"

for sitcode in ("ot1,ot2,ot3", "1,2,3", "1f,2f,3f"):
    r = requests.get(
        f"{BASE}/people/{PID}/stats",
        params={
            "stats": "statSplits",
            "group": "pitching",
            "season": YEAR,
            "sitCodes": sitcode,
        },
        timeout=10,
    )
    print("=" * 50)
    print(f"sitCodes={sitcode!r} → status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                desc = split.get("split", {}).get("description")
                bf = split.get("stat", {}).get("battersFaced")
                ops = split.get("stat", {}).get("ops")
                print(f"  {desc!r}  bf={bf}  ops={ops}")
```

- [ ] **Step 2: 記錄通過的 sitCode 字串 + description 樣式**

把通過組的 console output 貼到 task #1 完成 message。判斷 PASS：
- HTTP 200
- 至少 3 個 splits（description 含 "1st" / "2nd" / "3rd" 或同義字）
- 每個 split 有 `battersFaced` + `ops`（或 `obp` + `slg`）

預期通過順序：`ot1,ot2,ot3` 最可能；若不通試 `1,2,3` 再試 `1f,2f,3f`。

- [ ] **Step 3: 三組都失敗 → STOP**

如果三組全部 status != 200 或回的 splits 不含 1st/2nd/3rd 桶 → 改走 spec §12 提到的 Plan B（pybaseball Statcast career-only）。立刻 escalate 給 user，不要硬猜 sitCode。

- [ ] **Step 4: 把通過的 sitCode 字串寫進 spec 與 plan**

`docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md` §5.1 與本 plan Task 4 的程式碼把 `"ot1,ot2,ot3"` 替換為實測通過字串，並 commit：

```bash
git add docs/superpowers/specs/2026-05-03-tto3-penalty-signal-design.md docs/superpowers/plans/2026-05-03-tto3-penalty-impl.md
git commit -m "docs(spec): TTO3 — lock sitCodes to <STR> after spike verification"
```

如果通過字串就是 `ot1,ot2,ot3`（最可能），則 spec / plan 不需修改 → 跳過 Step 4 commit。

---

## Task 2: `_classify_tto_bucket` helper

**Files:**
- Modify: `scripts/pitcher_stats.py`（在 `fetch_platoon_splits` 之後新增 helper）
- Test: `scripts/tests/test_pitcher_stats.py`（appended）

純函式：把 split description 對應到 `"tto1"` / `"tto2"` / `"tto3"` / `None`。

- [ ] **Step 1: 寫 5 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py` 結尾：

```python
# ---------------------------------------------------------------------------
# TTO splits helpers
# ---------------------------------------------------------------------------

def test_classify_tto_bucket_first_lowercase():
    from pitcher_stats import _classify_tto_bucket
    assert _classify_tto_bucket("1st pa in g as p") == "tto1"


def test_classify_tto_bucket_first_word():
    from pitcher_stats import _classify_tto_bucket
    assert _classify_tto_bucket("first time facing") == "tto1"


def test_classify_tto_bucket_second_and_third():
    from pitcher_stats import _classify_tto_bucket
    assert _classify_tto_bucket("2nd time through order") == "tto2"
    assert _classify_tto_bucket("3rd pa in g as p") == "tto3"
    assert _classify_tto_bucket("second pa") == "tto2"
    assert _classify_tto_bucket("third pa") == "tto3"


def test_classify_tto_bucket_unknown_returns_none():
    from pitcher_stats import _classify_tto_bucket
    assert _classify_tto_bucket("vs left-handed batters") is None
    assert _classify_tto_bucket("4th time") is None  # TTO4+ 不收
    assert _classify_tto_bucket("") is None


def test_classify_tto_bucket_handles_uppercase():
    """description 大小寫混雜應正常解析（caller lower-case 化前的保險）。"""
    from pitcher_stats import _classify_tto_bucket
    # caller 已 .lower()，但函式自己也保險：直接傳大寫應走 None
    # 此 test 確認契約是「caller 負責 lower()」
    assert _classify_tto_bucket("1ST PA IN G AS P") is None
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py::test_classify_tto_bucket_first_lowercase -v
```

預期：`ImportError` 或 `AttributeError: module 'pitcher_stats' has no attribute '_classify_tto_bucket'`

- [ ] **Step 3: 實作 `_classify_tto_bucket`**

Append 到 `scripts/pitcher_stats.py` `fetch_platoon_splits` function 結束之後（line 522 之後）：

```python
def _classify_tto_bucket(desc: str) -> str | None:
    """把 split description 對應到 tto1 / tto2 / tto3。

    Caller 必須先 .lower() description；本函式只匹配小寫 1st/first/2nd/second/3rd/third。
    """
    if "1st" in desc or "first" in desc:
        return "tto1"
    if "2nd" in desc or "second" in desc:
        return "tto2"
    if "3rd" in desc or "third" in desc:
        return "tto3"
    return None
```

- [ ] **Step 4: 跑測試確認全 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k classify_tto_bucket -v
```

預期：5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): _classify_tto_bucket helper for TTO splits parser"
```

---

## Task 3: `_parse_ops_with_fallback` helper

**Files:**
- Modify: `scripts/pitcher_stats.py`
- Test: `scripts/tests/test_pitcher_stats.py`

純函式：MLB API `stat` dict OPS 解析；缺 `ops` 時回 `obp + slg`；都缺回 `None`。

- [ ] **Step 1: 寫 4 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py`：

```python
def test_parse_ops_with_fallback_uses_ops_when_present():
    from pitcher_stats import _parse_ops_with_fallback
    stat = {"ops": "0.745", "obp": "0.350", "slg": "0.420"}
    assert _parse_ops_with_fallback(stat) == 0.745


def test_parse_ops_with_fallback_uses_obp_plus_slg_when_ops_missing():
    from pitcher_stats import _parse_ops_with_fallback
    stat = {"obp": "0.350", "slg": "0.420"}
    result = _parse_ops_with_fallback(stat)
    assert result is not None
    assert abs(result - 0.770) < 1e-6


def test_parse_ops_with_fallback_returns_none_when_all_missing():
    from pitcher_stats import _parse_ops_with_fallback
    assert _parse_ops_with_fallback({}) is None
    assert _parse_ops_with_fallback({"avg": "0.250"}) is None


def test_parse_ops_with_fallback_handles_invalid_strings():
    from pitcher_stats import _parse_ops_with_fallback
    assert _parse_ops_with_fallback({"ops": "N/A"}) is None
    assert _parse_ops_with_fallback({"obp": "N/A", "slg": "0.420"}) is None
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k parse_ops_with_fallback -v
```

預期：4 errors（`AttributeError: module 'pitcher_stats' has no attribute '_parse_ops_with_fallback'`）

- [ ] **Step 3: 實作 `_parse_ops_with_fallback`**

Append 到 `scripts/pitcher_stats.py` 緊接 `_classify_tto_bucket` 之後：

```python
def _parse_ops_with_fallback(stat: dict) -> float | None:
    """OPS 優先；缺 → OBP+SLG fallback（mirror signal_reverse_platoon 第 257-262 行）。

    對 "N/A" / 非數值字串 graceful return None。
    """
    try:
        return float(stat["ops"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return float(stat["obp"]) + float(stat["slg"])
    except (KeyError, TypeError, ValueError):
        return None
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k parse_ops_with_fallback -v
```

預期：4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): _parse_ops_with_fallback helper (ops or obp+slg)"
```

---

## Task 4: `_fetch_tto_one` — 單次 statSplits / careerStatSplits HTTP 呼叫

**Files:**
- Modify: `scripts/pitcher_stats.py`
- Test: `scripts/tests/test_pitcher_stats.py`

包一次 MLB API 呼叫（season 或 career）+ 解析 → `{"tto1": {...}, "tto2": {...}, "tto3": {...}}` dict 或 `{"error": "..."}`。

> **重要**：本 task 程式碼裡的 `sitCodes="ot1,ot2,ot3"` 字串需與 Task 1 spike 結果一致。若 spike 通過字串不同，把所有出現處（`_fetch_tto_one`、可能還包含未來的 fixture）替換掉。

- [ ] **Step 1: 寫 4 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py`：

```python
# (Reuse _mock_requests_get pattern from test_lineup_analyzer.py)
import json as _json_mod
from unittest.mock import MagicMock as _MM


def _make_tto_resp(splits: list[dict]) -> _MM:
    """Build a MagicMock requests.get function returning {stats:[{splits:...}]}."""
    payload = {"stats": [{"splits": splits}]}
    resp = _MM()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return _MM(return_value=resp)


def test_fetch_tto_one_parses_all_three_buckets(monkeypatch):
    splits = [
        {"split": {"description": "1st PA in G as P"},
         "stat": {"battersFaced": 320, "ops": "0.700", "strikeOuts": 90, "baseOnBalls": 22}},
        {"split": {"description": "2nd PA in G as P"},
         "stat": {"battersFaced": 290, "ops": "0.740", "strikeOuts": 78, "baseOnBalls": 22}},
        {"split": {"description": "3rd PA in G as P"},
         "stat": {"battersFaced": 180, "ops": "0.810", "strikeOuts": 41, "baseOnBalls": 14}},
    ]
    monkeypatch.setattr("pitcher_stats.requests.get", _make_tto_resp(splits))

    from pitcher_stats import _fetch_tto_one
    result = _fetch_tto_one("statSplits", 669373, 2025)
    assert "error" not in result
    assert result["tto1"]["ops"] == 0.700
    assert result["tto1"]["bf"] == 320
    assert result["tto1"]["k_pct"] == round(90 / 320 * 100, 1)
    assert result["tto3"]["ops"] == 0.810
    assert result["tto3"]["bf"] == 180


def test_fetch_tto_one_uses_obp_slg_when_ops_missing(monkeypatch):
    splits = [
        {"split": {"description": "1st PA"},
         "stat": {"battersFaced": 100, "obp": "0.300", "slg": "0.400", "strikeOuts": 25, "baseOnBalls": 8}},
        {"split": {"description": "2nd PA"},
         "stat": {"battersFaced": 90, "obp": "0.310", "slg": "0.420", "strikeOuts": 22, "baseOnBalls": 7}},
        {"split": {"description": "3rd PA"},
         "stat": {"battersFaced": 60, "obp": "0.340", "slg": "0.460", "strikeOuts": 13, "baseOnBalls": 5}},
    ]
    monkeypatch.setattr("pitcher_stats.requests.get", _make_tto_resp(splits))

    from pitcher_stats import _fetch_tto_one
    result = _fetch_tto_one("statSplits", 669373, 2025)
    assert abs(result["tto1"]["ops"] - 0.700) < 1e-6
    assert abs(result["tto3"]["ops"] - 0.800) < 1e-6


def test_fetch_tto_one_skips_unknown_buckets(monkeypatch):
    splits = [
        {"split": {"description": "1st PA"},
         "stat": {"battersFaced": 100, "ops": "0.700", "strikeOuts": 25, "baseOnBalls": 8}},
        {"split": {"description": "vs Left"},  # 應被 _classify_tto_bucket 過濾
         "stat": {"battersFaced": 50, "ops": "0.650", "strikeOuts": 12, "baseOnBalls": 5}},
        {"split": {"description": "3rd PA"},
         "stat": {"battersFaced": 40, "ops": "0.820", "strikeOuts": 9, "baseOnBalls": 3}},
    ]
    monkeypatch.setattr("pitcher_stats.requests.get", _make_tto_resp(splits))

    from pitcher_stats import _fetch_tto_one
    result = _fetch_tto_one("statSplits", 669373, 2025)
    assert "tto1" in result and "tto3" in result
    assert "tto2" not in result  # 沒回傳 2nd → 缺


def test_fetch_tto_one_returns_error_on_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("network down")
    monkeypatch.setattr("pitcher_stats.requests.get", _raise)

    from pitcher_stats import _fetch_tto_one
    result = _fetch_tto_one("statSplits", 669373, 2025)
    assert "error" in result
    assert "statSplits" in result["error"]
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k fetch_tto_one -v
```

預期：4 errors（`_fetch_tto_one` not defined）

- [ ] **Step 3: 實作 `_fetch_tto_one`**

Append 到 `scripts/pitcher_stats.py` 緊接 `_parse_ops_with_fallback` 之後：

```python
def _fetch_tto_one(stats_kind: str, mlbam_id: int, year: int) -> dict:
    """單次 MLB API call 取 TTO 三桶。

    stats_kind: "statSplits"（season 限定）或 "careerStatSplits"。
    season 路徑帶 ?season=YEAR；career 路徑不帶。
    """
    try:
        params = {
            "stats": stats_kind,
            "group": "pitching",
            "sitCodes": "ot1,ot2,ot3",  # Task 1 spike 後若需替換，全檔搜替
        }
        if stats_kind == "statSplits":
            params["season"] = year
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        result: dict = {}
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                desc = split.get("split", {}).get("description", "").lower()
                key = _classify_tto_bucket(desc)
                if key is None:
                    continue
                s = split.get("stat", {})
                bf = int(s.get("battersFaced", 0))
                k = int(s.get("strikeOuts", 0))
                bb = int(s.get("baseOnBalls", 0))
                ops = _parse_ops_with_fallback(s)
                result[key] = {
                    "ops": ops,
                    "k_pct": round(k / bf * 100, 1) if bf > 0 else 0.0,
                    "bb_pct": round(bb / bf * 100, 1) if bf > 0 else 0.0,
                    "bf": bf,
                }
        return result if result else {"error": f"No TTO split data ({stats_kind})"}
    except Exception as e:
        return {"error": f"{stats_kind} fetch failed: {e}"}
```

- [ ] **Step 4: 跑測試確認 pass**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k fetch_tto_one -v
```

預期：4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): _fetch_tto_one wraps MLB API statSplits/careerStatSplits"
```

---

## Task 5: `fetch_tto_splits` — season → career fallback orchestrator

**Files:**
- Modify: `scripts/pitcher_stats.py`
- Test: `scripts/tests/test_pitcher_stats.py`

把 `_fetch_tto_one` 包成 fallback orchestrator：season 優先；TTO3 BF < 30 → career；都不夠 → 回 season（caller 走 small_sample）；都失敗 → `{"error": ...}`。

- [ ] **Step 1: 寫 5 個 failing tests**

Append 到 `scripts/tests/test_pitcher_stats.py`：

```python
def test_fetch_tto_splits_season_full(monkeypatch):
    """Season tto3.bf ≥ 30 → 用 season，不打 career。"""
    splits = [
        {"split": {"description": "1st PA"}, "stat": {"battersFaced": 320, "ops": "0.700", "strikeOuts": 90, "baseOnBalls": 22}},
        {"split": {"description": "2nd PA"}, "stat": {"battersFaced": 290, "ops": "0.740", "strikeOuts": 78, "baseOnBalls": 22}},
        {"split": {"description": "3rd PA"}, "stat": {"battersFaced": 180, "ops": "0.810", "strikeOuts": 41, "baseOnBalls": 14}},
    ]
    call_count = {"n": 0}

    def _get(*args, **kwargs):
        call_count["n"] += 1
        resp = _MM()
        resp.json.return_value = {"stats": [{"splits": splits}]}
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr("pitcher_stats.requests.get", _get)

    from pitcher_stats import fetch_tto_splits
    result = fetch_tto_splits(669373, 2025)
    assert result["source"] == "season"
    assert result["tto3"]["bf"] == 180
    assert call_count["n"] == 1  # career 沒被打


def test_fetch_tto_splits_falls_back_to_career(monkeypatch):
    """Season tto3.bf < 30 → 改 career；career 充足 → source=career."""
    season_splits = [
        {"split": {"description": "1st PA"}, "stat": {"battersFaced": 50, "ops": "0.700", "strikeOuts": 12, "baseOnBalls": 4}},
        {"split": {"description": "2nd PA"}, "stat": {"battersFaced": 40, "ops": "0.740", "strikeOuts": 9, "baseOnBalls": 3}},
        {"split": {"description": "3rd PA"}, "stat": {"battersFaced": 18, "ops": "0.810", "strikeOuts": 4, "baseOnBalls": 2}},
    ]
    career_splits = [
        {"split": {"description": "1st PA"}, "stat": {"battersFaced": 1500, "ops": "0.680", "strikeOuts": 380, "baseOnBalls": 100}},
        {"split": {"description": "2nd PA"}, "stat": {"battersFaced": 1300, "ops": "0.715", "strikeOuts": 320, "baseOnBalls": 95}},
        {"split": {"description": "3rd PA"}, "stat": {"battersFaced": 800, "ops": "0.755", "strikeOuts": 175, "baseOnBalls": 65}},
    ]
    calls = {"n": 0}

    def _get(*args, **kwargs):
        calls["n"] += 1
        params = kwargs.get("params") or {}
        chosen = season_splits if params.get("stats") == "statSplits" else career_splits
        resp = _MM()
        resp.json.return_value = {"stats": [{"splits": chosen}]}
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr("pitcher_stats.requests.get", _get)

    from pitcher_stats import fetch_tto_splits
    result = fetch_tto_splits(669373, 2025)
    assert result["source"] == "career"
    assert result["tto3"]["bf"] == 800
    assert calls["n"] == 2  # season + career


def test_fetch_tto_splits_career_also_thin(monkeypatch):
    """Season + career 都 thin → 回 season（caller 走 small_sample no_fire）。"""
    thin = [
        {"split": {"description": "1st PA"}, "stat": {"battersFaced": 30, "ops": "0.700", "strikeOuts": 8, "baseOnBalls": 2}},
        {"split": {"description": "2nd PA"}, "stat": {"battersFaced": 25, "ops": "0.740", "strikeOuts": 6, "baseOnBalls": 2}},
        {"split": {"description": "3rd PA"}, "stat": {"battersFaced": 15, "ops": "0.810", "strikeOuts": 4, "baseOnBalls": 1}},
    ]

    def _get(*args, **kwargs):
        resp = _MM()
        resp.json.return_value = {"stats": [{"splits": thin}]}
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr("pitcher_stats.requests.get", _get)

    from pitcher_stats import fetch_tto_splits
    result = fetch_tto_splits(669373, 2025)
    assert result["source"] == "season"
    assert result["tto3"]["bf"] == 15  # < 30，caller 處理


def test_fetch_tto_splits_season_error_career_ok(monkeypatch):
    """Season 失敗 → career 補上。"""
    career_splits = [
        {"split": {"description": "1st PA"}, "stat": {"battersFaced": 1500, "ops": "0.680", "strikeOuts": 380, "baseOnBalls": 100}},
        {"split": {"description": "2nd PA"}, "stat": {"battersFaced": 1300, "ops": "0.715", "strikeOuts": 320, "baseOnBalls": 95}},
        {"split": {"description": "3rd PA"}, "stat": {"battersFaced": 800, "ops": "0.755", "strikeOuts": 175, "baseOnBalls": 65}},
    ]
    calls = {"n": 0}

    def _get(*args, **kwargs):
        calls["n"] += 1
        params = kwargs.get("params") or {}
        if params.get("stats") == "statSplits":
            raise RuntimeError("season API 5xx")
        resp = _MM()
        resp.json.return_value = {"stats": [{"splits": career_splits}]}
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr("pitcher_stats.requests.get", _get)

    from pitcher_stats import fetch_tto_splits
    result = fetch_tto_splits(669373, 2025)
    assert result["source"] == "career"
    assert calls["n"] == 2


def test_fetch_tto_splits_both_fail_returns_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("network down")
    monkeypatch.setattr("pitcher_stats.requests.get", _raise)

    from pitcher_stats import fetch_tto_splits
    result = fetch_tto_splits(669373, 2025)
    assert "error" in result
```

- [ ] **Step 2: 跑測試確認 fail**

```bash
cd scripts
python -m pytest tests/test_pitcher_stats.py -k fetch_tto_splits -v
```

預期：5 errors（`fetch_tto_splits` not defined）

- [ ] **Step 3: 實作 `fetch_tto_splits` + inline `_has_sufficient_tto3`**

Append 到 `scripts/pitcher_stats.py` 緊接 `_fetch_tto_one` 之後：

```python
_TTO_MIN_BF = 30  # tto3 bucket 最小 BF；不足走 career fallback


def _has_sufficient_tto3(data: dict, min_bf: int = _TTO_MIN_BF) -> bool:
    """data 裡 tto3.bf 是否 ≥ min_bf。error / 缺 tto3 → False。"""
    if "error" in data:
        return False
    tto3 = data.get("tto3") or {}
    return (tto3.get("bf") or 0) >= min_bf


def fetch_tto_splits(mlbam_id: int, year: int) -> dict:
    """C2.5：取得投手 Times-Through-Order Splits（TTO1 / TTO2 / TTO3）。

    Season 優先；TTO3 BF < 30 → silent fallback careerStatSplits。
    回傳：
      {
        "source": "season" | "career",
        "tto1": {...}, "tto2": {...}, "tto3": {...},
      }
      或 {"error": "..."} 兩條路徑都失敗時。

    Caller (signal_tto3_penalty) 看 tto3.bf 自行判斷 small_sample。
    """
    season_data = _fetch_tto_one("statSplits", mlbam_id, year)
    if _has_sufficient_tto3(season_data):
        season_data["source"] = "season"
        return season_data

    career_data = _fetch_tto_one("careerStatSplits", mlbam_id, year)
    if _has_sufficient_tto3(career_data):
        career_data["source"] = "career"
        return career_data

    # season 不足，且 career 也不足或失敗
    if "error" not in season_data:
        season_data["source"] = "season"
        return season_data  # caller 走 small_sample no_fire
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

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "feat(pitcher): fetch_tto_splits orchestrator with career fallback"
```

---

## Task 6: 把 `fetch_tto_splits` 接進 pitcher_stats main 路徑

**Files:**
- Modify: `scripts/pitcher_stats.py`（main 路徑 + JSON 寫入）

緊接現有 `platoon_splits = fetch_platoon_splits(...)` 加一行 `tto_splits = fetch_tto_splits(...)`，並在輸出 dict 加 `tto_splits` key。**沒新測試**——現有 main 路徑沒有單元測試（屬於 integration），靠 Task 12 smoke test 驗證。

- [ ] **Step 1: Read current main 路徑 line 887 附近**

```bash
cd scripts
python -c "with open('pitcher_stats.py') as f: lines = f.readlines(); [print(i+1, l, end='') for i,l in enumerate(lines[880:930])]"
```

確認 `platoon_splits = fetch_platoon_splits(pitcher_id, args.year)` 大約在第 887 行；輸出 dict 含 `"platoon_splits": platoon_splits` 大約在第 924 行。

- [ ] **Step 2: 加 fetch 呼叫**

`scripts/pitcher_stats.py` 第 887 行（`platoon_splits = fetch_platoon_splits(...)` 那行）後，append 一行：

```python
    platoon_splits = fetch_platoon_splits(pitcher_id, args.year)
    tto_splits = fetch_tto_splits(pitcher_id, args.year)
```

- [ ] **Step 3: 加進輸出 dict**

`scripts/pitcher_stats.py` 第 924 行（`"platoon_splits": platoon_splits,` 那行）後，append 一行：

```python
        "platoon_splits": platoon_splits,
        "tto_splits": tto_splits,
```

- [ ] **Step 4: 跑全部既有測試確認沒撞到**

```bash
cd scripts
python -m pytest tests/ -v --tb=short
```

預期：所有既有測試（439 + 9 個 Task 2-5 新加 = 448）全 pass。

- [ ] **Step 5: Commit**

```bash
git add scripts/pitcher_stats.py
git commit -m "feat(pitcher): wire fetch_tto_splits into main pitcher_stats output"
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

- [ ] **Step 2: 抓出 Task 2-10 的 commit short hashes**

```bash
cd C:/Users/Loger/.claude/skills/mlb-game-analyzer
git log --oneline -12
```

從輸出抓出 9 個 commit short hash（Task 2 到 Task 10 各一），以及 Task 1（如果有 spec/plan 替換 commit）。記錄為一份 mapping，下一步要替換進 CHANGELOG 內文。

- [ ] **Step 3: 在最頂端加新版區塊**

在現有 `## 2026-05-03 — Path B refactor` 區塊之上插入下面內容；把 `<HASHN>` 佔位符替換為 Step 2 抓到的真實 short hash：

```markdown
## 2026-05-04 — TTO3 penalty signal（signal #9）

第 9 個 derived signal，pitcher-side per-game。先發投手第三輪面對打者 OPS
衰退幅度，覆蓋 PR-3 後 line 48「第二批 signals」第一項。

- **commit <HASH2>** `feat(pitcher)`: `_classify_tto_bucket` helper
- **commit <HASH3>** `feat(pitcher)`: `_parse_ops_with_fallback` helper
- **commit <HASH4>** `feat(pitcher)`: `_fetch_tto_one` MLB API wrapper
- **commit <HASH5>** `feat(pitcher)`: `fetch_tto_splits` orchestrator with career fallback
- **commit <HASH6>** `feat(pitcher)`: wire `fetch_tto_splits` into main output
- **commit <HASH7>** `feat(signals)`: `signal_tto3_penalty` (#9) + half_life=structural
- **commit <HASH8>** `feat(signals)`: wire tto3_penalty into compute_all_signals
- **commit <HASH9>** `feat(dossier)`: 投手對決 table 加 TTO splits visible row
- **commit <HASH10>** `docs(reference)`: matchup-factors §Signals §9 + 半衰期表

### 紀律保留

- ✅ 信號**不入 scoring formula**（一致 §3 / §8）
- ✅ 既有 8 signals 行為零變動（compute_all_signals 只追加一行）
- ✅ 4 月小樣本 season → career silent fallback，BF < 30 統一 small_sample no_fire
- ✅ Dossier TTO row 無條件顯示（mirror vs LHB / vs RHB pattern）
- ✅ `merge_game_data.py` / `prepare_game.py` / `scoring_formula.py` / Flag 體系全部不動

### Out of scope（下批）

- TTO4+ penalty（樣本太稀）
- Reliever inheritance penalty
- 動態調整觸發閾值（按 tier 別）— 留至 backtest 階段
- 休息天數 / 上一場用球數（CHANGELOG line 48 第二批 signals 中的另兩項）
```

> 註：Task 1 spike 通常**不**產出 commit（spec/plan 都已寫死 `ot1,ot2,ot3`）。若 spike 結果需要替換 sitCode 字串並 commit spec/plan，把該 commit 也加進 CHANGELOG（例如 `**commit <HASH1>** `docs(spec)`: lock sitCodes ...`）。

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

預期：~454 tests 全 pass（439 baseline + ~15 新增）。

如果都 pass，task 完成。Task 11 CHANGELOG 區塊裡的 `<HASHN>` 佔位符應該已經在 Task 11 Step 2-3 替換成真實 hash（在 Task 11 commit 前完成），不需事後 amend。

---

## Spec coverage 自我驗證表

| Spec 段落 | 對應 task | 驗證點 |
|---|---|---|
| §2 Goals 1: signal_tto3_penalty 落 signals_lib.py | Task 7 | 單元測試 9 個 |
| §2 Goals 2: fetch_tto_splits 沿用 statSplits | Task 4-5 | 單元測試 9 個 |
| §2 Goals 3: 4 月 fallback career, heuristic | Task 5 + Task 7 | `test_fetch_tto_splits_falls_back_to_career` + `test_tto3_penalty_career_source_marks_heuristic` |
| §2 Goals 4: dossier visible row | Task 9 | 單元測試 + integration test |
| §2 Goals 5: dossier 訊號摘要 + summary 額外信號 | Task 8（compute_all_signals 接入後 cache 自動帶） | smoke test §3 / §5 |
| §2 Goals 6: matchup-factors §9 + 半衰期表 | Task 10 | docs review |
| §2 Goals 7: CHANGELOG line 50 清理 | Task 11 | docs review |
| §3 Non-Goals: 不進 scoring formula | 所有 task | scoring_formula.py 0 異動 |
| §3 Non-Goals: 不動 merge_game_data | 所有 task | merge_game_data.py 0 異動 |
| §3 Non-Goals: RP / opener no_fire | Task 7 | `test_tto3_penalty_small_sample_below_30_bf` 涵蓋 |
| §5.3 sitCode spike | Task 1 | 5 分鐘人工驗證 |
| §5.4 fallback 矩陣 | Task 5 | 5 個 test 涵蓋 6 條矩陣 row |
| §6.1 signal contract | Task 7 | `_signal_contract(s)` helper assertion |
| §7.1 dossier helper（n/a / career suffix / small_sample / 缺 key） | Task 9 | 5 個 test |
| §9 Tests 列表 | Task 2-9 | 約 24 新測試（spec 估 14，實際多） |

