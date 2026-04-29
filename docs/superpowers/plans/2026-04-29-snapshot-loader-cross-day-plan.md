# Snapshot Loader Cross-Day Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修復 `load_snapshots_for_et_date` 用檔名過濾日期,導致跨日 snapshot 在後一日分析時被忽略的 silent data loss bug。

**Architecture:** Loader 拿掉檔名日期過濾,讀目錄內所有 `*-ET.json`;日期判斷完全交給下游 `collect_game_timeline` 既有的 `g["game_date_et"]` 場次層過濾。

**Tech Stack:** Python 3 / pytest / 原生 stdlib(json、pathlib),無新增依賴。

**Spec:** `docs/superpowers/specs/2026-04-29-snapshot-loader-cross-day-design.md`

---

## File Structure

| 路徑 | 動作 | 職責 |
|---|---|---|
| `odds/tests/fixtures/2026-04-28_21-00-ET.json` | 新建 | 跨日 fixture:1 場 4/28 + 1 場 4/29 完整 Pinnacle 盤口 |
| `odds/tests/test_snapshot_loader.py` | 修改 | 新增 cross-day 測試 + 更新 3 個既有斷言對應新契約 |
| `odds/lib/snapshot_loader.py` | 修改 | `load_snapshots_for_et_date` glob pattern + docstring |

---

## Task 1: 新增 cross-day fixture

**Files:**
- Create: `odds/tests/fixtures/2026-04-28_21-00-ET.json`

- [ ] **Step 1: 寫 fixture 檔**

格式對齊現有 `odds/tests/fixtures/2026-04-27_00-00-ET.json`。snapshot 抓取時間設為 ET 4/28 21:00,內含 1 場 4/28 + 1 場 4/29(各自完整 Pinnacle ML/OU/RL):

```json
{
  "snapshot_time_utc": "2026-04-29T01:00:00.000000+00:00",
  "snapshot_time_et": "2026-04-28 21:00 ET",
  "credits_remaining": "275",
  "credits_used": "225",
  "game_count": 2,
  "games": [
    {
      "game": "Houston Astros @ Baltimore Orioles",
      "away_team": "Houston Astros",
      "home_team": "Baltimore Orioles",
      "commence_utc": "2026-04-28T22:38:00Z",
      "commence_et": "2026-04-28 18:38 ET",
      "game_date_et": "2026-04-28",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Baltimore Orioles": { "odds": 1.05, "implied_pct": 95.2 },
            "Houston Astros":    { "odds": 12.36, "implied_pct": 8.1 }
          },
          "ou": {
            "Over":  { "odds": 1.91, "point": 8.5, "implied_pct": 52.4 },
            "Under": { "odds": 1.95, "point": 8.5, "implied_pct": 51.3 }
          },
          "rl": {
            "Baltimore Orioles": { "odds": 2.55, "point": -1.5, "implied_pct": 39.2 },
            "Houston Astros":    { "odds": 1.58, "point":  1.5, "implied_pct": 63.3 }
          }
        }
      }
    },
    {
      "game": "Tampa Bay Rays @ Cleveland Guardians",
      "away_team": "Tampa Bay Rays",
      "home_team": "Cleveland Guardians",
      "commence_utc": "2026-04-29T17:11:00Z",
      "commence_et": "2026-04-29 13:11 ET",
      "game_date_et": "2026-04-29",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Cleveland Guardians": { "odds": 2.07, "implied_pct": 48.3 },
            "Tampa Bay Rays":      { "odds": 1.85, "implied_pct": 54.1 }
          },
          "ou": {
            "Over":  { "odds": 1.91, "point": 6.5, "implied_pct": 52.4 },
            "Under": { "odds": 1.95, "point": 6.5, "implied_pct": 51.3 }
          },
          "rl": {
            "Cleveland Guardians": { "odds": 2.55, "point": -1.5, "implied_pct": 39.2 },
            "Tampa Bay Rays":      { "odds": 1.58, "point":  1.5, "implied_pct": 63.3 }
          }
        }
      }
    }
  ]
}
```

- [ ] **Step 2: 驗證 fixture 結構正確(JSON 可解析、欄位齊全)**

Run:
```
python -c "import json; d=json.load(open('odds/tests/fixtures/2026-04-28_21-00-ET.json',encoding='utf-8')); dates=sorted({g['game_date_et'] for g in d['games']}); print('dates:', dates, 'count:', len(d['games']))"
```

Expected output:
```
dates: ['2026-04-28', '2026-04-29'] count: 2
```

- [ ] **Step 3: Commit fixture**

```
git add odds/tests/fixtures/2026-04-28_21-00-ET.json
git commit -m "test(odds): add cross-day fixture (4/28 snapshot containing 4/29 game)"
```

---

## Task 2: 寫失敗測試 + 更新既有斷言(RED)

**Files:**
- Modify: `odds/tests/test_snapshot_loader.py:22-41`(更新 3 個既有測試)
- Modify: `odds/tests/test_snapshot_loader.py`(新增 1 個 cross-day 測試)

- [ ] **Step 1: 更新 `test_loads_only_matching_et_date`**

把舊斷言「filename 為 04-27 的兩份」改為「目錄下全部 4 份(04-26 一份 + 04-27 兩份 + 04-28 一份)」。

舊版(`odds/tests/test_snapshot_loader.py:22-28`):
```python
def test_loads_only_matching_et_date():
    """Fixtures 含 04-27 兩份 + 04-26 一份;查 04-27 應只回兩份,按時間排序。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    assert len(snapshots) == 2
    # 應按 snapshot_time_utc 由舊到新
    assert snapshots[0].snapshot_time_et.hour == 0
    assert snapshots[1].snapshot_time_et.hour == 4
```

改為:
```python
def test_loads_all_snapshots_in_dir():
    """Loader 已不依賴檔名日期;呼叫應回目錄下所有合法 snapshot,按 utc 時間排序。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    assert len(snapshots) == 4
    # 應按 snapshot_time_utc 由舊到新:04-26 20:00 → 04-27 00:00 → 04-27 04:00 → 04-28 21:00
    times = [s.snapshot_time_utc for s in snapshots]
    assert times == sorted(times)
```

- [ ] **Step 2: 更新 `test_loads_other_date`**

舊版(`odds/tests/test_snapshot_loader.py:31-35`):
```python
def test_loads_other_date():
    """查 04-26 應只回那一份。"""
    snapshots = load_snapshots_for_et_date("2026-04-26", FIXTURES)
    assert len(snapshots) == 1
    assert snapshots[0].snapshot_time_et.hour == 20
```

改為:
```python
def test_load_returns_full_directory_regardless_of_date():
    """無論 et_date 為何,loader 都應回目錄下全部合法 snapshot(契約已改:過濾交給 collect_game_timeline)。"""
    a = load_snapshots_for_et_date("2026-04-26", FIXTURES)
    b = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    c = load_snapshots_for_et_date("2026-04-29", FIXTURES)
    assert len(a) == len(b) == len(c) == 4
```

- [ ] **Step 3: 改寫 `test_loads_missing_date_returns_empty`**

舊版(`odds/tests/test_snapshot_loader.py:38-41`)斷言「不存在日期 → 空 list」。新契約下 loader 不過濾日期,改為驗證「下游 collect_game_timeline 對不存在日期回空 dict」。

舊版:
```python
def test_loads_missing_date_returns_empty():
    """不存在的日期 → 空 list,不 crash。"""
    snapshots = load_snapshots_for_et_date("2099-01-01", FIXTURES)
    assert snapshots == []
```

改為:
```python
def test_collect_timeline_missing_date_returns_empty():
    """不存在的 game_date_et → collect_game_timeline 回空 dict(loader 不再過濾日期,改為 timeline 層級空結果)。"""
    snapshots = load_snapshots_for_et_date("2099-01-01", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2099-01-01")
    assert timelines == {}
```

- [ ] **Step 4: 新增 cross-day 測試**

新增於 `odds/tests/test_snapshot_loader.py` 的 `# ── collect_game_timeline ──` 區塊內(可放在 `test_collect_timeline_filters_by_game_date` 之後):

```python
def test_cross_day_snapshot_contributes_to_later_date():
    """檔名為 04-28 的 snapshot 內含 04-29 場次,查 04-29 應能取到那場 timeline。

    這是這次 bug 的核心修復:loader 不再依賴檔名前綴,
    snapshot 內任何 game_date_et=2026-04-29 的場都應透過 collect_game_timeline 浮現。
    """
    snapshots = load_snapshots_for_et_date("2026-04-29", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-29")
    # fixture 04-28_21-00-ET.json 內有一場 TBR @ CLE on 2026-04-29
    keys = list(timelines.keys())
    assert any(k[0] == "Tampa Bay Rays" and k[1] == "Cleveland Guardians" for k in keys), \
        f"4/29 TBR@CLE timeline 應出現在 cross-day snapshot 讀取結果中,實際 keys={keys}"
    # 同時驗證 04-28 的場不會誤入 04-29 timeline
    assert not any(k[0] == "Houston Astros" for k in keys), \
        "4/28 HOU@BAL 不該出現在 4/29 timeline"
```

- [ ] **Step 5: 跑測試,確認 RED**

Run:
```
pytest odds/tests/test_snapshot_loader.py -v
```

Expected:
- `test_loads_all_snapshots_in_dir` FAIL(舊 impl 過濾後只回 2 份,期望 4 份)
- `test_load_returns_full_directory_regardless_of_date` FAIL(同上)
- `test_collect_timeline_missing_date_returns_empty` PASS(此測試在新舊 impl 下都通過——不存在日期下 loader 回空 list,timeline 也是空 dict)
- `test_cross_day_snapshot_contributes_to_later_date` FAIL(舊 impl glob `2026-04-29_*-ET.json` 找不到 4/28 檔案)
- 其餘原有測試仍 PASS(`test_collect_timeline_groups_by_game_key` 等不受 loader 行為改變影響——因為它們呼叫 04-27 後再走 collect_game_timeline,舊 impl 回 2 份 × game_date_et 過濾仍正確;新 impl 會回 4 份但 04-26 / 04-28 / 04-29 場次會被 collect_game_timeline 過掉,結果仍為 3 場 04-27)

不 commit(RED state)。

---

## Task 3: 實作 loader 變更(GREEN + commit)

**Files:**
- Modify: `odds/lib/snapshot_loader.py:46-71`

- [ ] **Step 1: 改 `load_snapshots_for_et_date`**

修改 `odds/lib/snapshot_loader.py:46-71`:

舊版:
```python
def load_snapshots_for_et_date(et_date: str, snapshot_dir) -> list[Snapshot]:
    """讀 snapshot_dir 下所有檔名以 <et_date>_*-ET.json 開頭的快照,按 snapshot_time_utc 排序。"""
    p = Path(snapshot_dir)
    if not p.exists():
        return []
    out: list[Snapshot] = []
    for f in p.glob(f"{et_date}_*-ET.json"):
        try:
            ...
```

改為:
```python
def load_snapshots_for_et_date(et_date: str, snapshot_dir) -> list[Snapshot]:
    """讀 snapshot_dir 下所有 *-ET.json 快照,按 snapshot_time_utc 排序。

    註:函式名保留 et_date 參數,但本實作不再依檔名前綴過濾日期——避免「跨日 snapshot
    在後一日分析時被忽略」的 silent data loss(例如 ET 4/28 21:00 的 snapshot 內含 4/29 開盤
    的場次)。日期過濾完全交給下游 `collect_game_timeline` 的 `g["game_date_et"]` 比對。
    """
    p = Path(snapshot_dir)
    if not p.exists():
        return []
    out: list[Snapshot] = []
    for f in p.glob("*-ET.json"):
        try:
            ...
```

(`et_date` 參數保留但未消費——避免擴大呼叫端 diff;`try/except` 區塊與後續 sort 邏輯不動。)

- [ ] **Step 2: 跑 snapshot_loader 測試,確認全 GREEN**

Run:
```
pytest odds/tests/test_snapshot_loader.py -v
```

Expected: 全部測試通過(包括 Task 2 新增/更新的 4 個)。

- [ ] **Step 3: 跑完整 odds 測試套件,確認沒打到其他模組**

Run:
```
pytest odds/tests/ -v
```

Expected: `test_movement.py` 與 `test_md_renderer.py` 也都通過。

- [ ] **Step 4: 端到端驗證——對真實 snapshot 跑分析**

Run:
```
python odds/analyze_smart_money.py --date 2026-04-29
```

Expected:
- 不再出現 `INFO 2026-04-29 無 snapshot;寫入空白 md`
- 應出現 `OK 寫入 .../odds/reports/2026-04-29.md | major X / significant X / watch X / quiet 14`(quiet 12 + 兩場 Pinnacle 未開盤的 Rockies@Reds、Nationals@Mets 會在 `collect_game_timeline` 場次層被略過,實際數字以執行結果為準)

- [ ] **Step 5: Commit 實作 + 測試變更**

```
git add odds/lib/snapshot_loader.py odds/tests/test_snapshot_loader.py
git commit -m "fix(odds): snapshot loader 改為按內容過濾 ET 日期

修復跨日 snapshot 被 loader 忽略導致 line movement timeline
丟失最早開盤錨點的 silent data loss bug。

loader 拿掉檔名日期過濾(glob '<date>_*-ET.json' → '*-ET.json'),
日期判斷完全交給下游 collect_game_timeline 的 game_date_et 場次層比對。
保留 et_date 參數簽章避免擴大呼叫端 diff。

更新既有 3 個測試斷言對應新契約,新增 1 個 cross-day 整合測試。"
```

---

## Self-Review

**Spec coverage:**
- Spec「變更點 / `load_snapshots_for_et_date`」→ Task 3 Step 1 ✓
- Spec「變更點 / 測試斷言」→ Task 2 Steps 1-3 ✓
- Spec「新增 fixture」→ Task 1 ✓
- Spec「新增測試 / test_cross_day_snapshot_loaded_for_later_date」→ Task 2 Step 4(實際命名為 `test_cross_day_snapshot_contributes_to_later_date`,語意對齊 spec) ✓
- Spec「測試命令 `pytest odds/tests/test_snapshot_loader.py -v`」→ Task 3 Step 2 ✓
- Spec「不做的事」→ 計畫中無相反動作 ✓

**Placeholder scan:** 無 TBD/TODO/「補上 X」之類占位文字,所有 code block 都有完整內容。

**Type consistency:** `load_snapshots_for_et_date` 簽章不變、`collect_game_timeline` 不動、`Snapshot` dataclass 不動。新測試使用的 `keys[0]`(away)、`keys[1]`(home)對齊 `GameKey = tuple[str, str, str]` 在 `snapshot_loader.py:19` 的定義。
