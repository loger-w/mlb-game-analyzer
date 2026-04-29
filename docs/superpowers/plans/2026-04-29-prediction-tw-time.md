# 比賽預測統一使用台灣時間 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 mlb-game-analyzer 預測流程的「user-facing 介面 + 最終報告」切到 TW 時區（folder name + `prediction_summary.md` 開打時間）；同時刪除已棄置的 mlb-post-game-review skill 在本 skill 內的殘留代碼與 cross-ref。

**Architecture:** `prepare_game.py --date` 改 TW 語意，內部換算 `et_date = tw_date − 1 day` 給 MLB schedule API；folder 直接用 user-input TW date；`predict.py` 的 `_extract_game_date_et` rename 為 `_extract_game_date_tw`，fallback 改用「ET astimezone → date + 1」算法（處理 early ET day game edge case）；`format_prediction_summary_md` 加開打時間 meta 行（TW 為主、ET 副欄）。中間層腳本（dossier / merged / 各 *_summary）保持 ET 語意不動。

**Tech Stack:** Python 3.x, pytest, datetime/timezone std lib, subprocess. 無新依賴。

---

## File Structure

| 檔案 | 變更類型 | 主要責任 |
|------|---------|---------|
| `scripts/predict.py` | Modify | `_extract_game_date_tw` rename + ET+1 fallback；新增 `_TW_TZ`；`format_prediction_summary_md` 加開打時間 meta 行；移除 `record["actual_*"] / ["verified"]` |
| `scripts/prepare_game.py` | Modify | `--date` help text 改 TW 語意；新增 `_tw_to_et` helper；`step_a` 呼叫處加 TW→ET 轉換 |
| `scripts/diagnose_ou_total_error.py` | **Delete** | post-game-review 範疇，依賴已不存在的 `predictions.jsonl` |
| `scripts/tests/test_predict.py` | Modify | 新增 4 個 TW 相關測試 |
| `scripts/tests/test_prepare_game.py` | Modify | 新增 `_tw_to_et` 單元測試 |
| `scripts/tests/test_prepare_game_steps.py` | Modify | 既有 omnibus test 補 fetch_game_data 收到 ET = TW-1 的 assertion |
| `SKILL.md` | Modify | line 20 移除「賽後回顧（轉 mlb-post-game-review）」 |
| `reference/prediction.md` | Modify | line 188-192 「預測紀錄存放位置」區塊精簡為只剩 per-game prediction.json |

**檔案責任邊界：**
- `predict.py` 同時是 record dict + summary 渲染 + 公式預測，本次改動集中在前兩個責任；不重構公式邏輯
- `prepare_game.py` 是 thin orchestrator，只在 `step_a` 呼叫 `fetch_game_data.py` 處需要 TW→ET 轉換（`step_b/c/d` 不吃 date）
- 中間層腳本（`fetch_game_data.py` / `merge_game_data.py` / `dossier_renderer.py`）的 ET 顯示完全不動

---

## Task 順序與依賴

| Task | 對應 Spec §8 commit | 依賴 |
|------|------|------|
| Task 1 | Commit 1（predict.py TW + summary） | 無 |
| Task 2 | Commit 2（移除 actual_*/verified） | Task 1 完成（同檔案） |
| Task 3 | Commit 3（prepare_game.py TW） | 無（與 Task 1/2 平行可，但建議順序執行） |
| Task 4 | Commit 4（刪 diagnose + 文件清理） | 無 |

每個 Task 獨立可 commit；若 subagent-driven 執行，可用 git worktree 平行（但建議順序執行避免 merge 衝突，特別是 Task 1 / 2 同改 predict.py）。

---

## Task 1: predict.py — record date + summary 開打時間改 TW

**Files:**
- Modify: `scripts/predict.py:863` (新增 `_TW_TZ`), `:867-883` (rename + fallback), `:741-758` (summary 加 meta), `:1015` (record_date fallback), `:1131` (record date 欄位)
- Test: `scripts/tests/test_predict.py` (新增 4 cases)

### Step 1.1: 寫第一個失敗測試（_extract_game_date_tw fallback ET+1）

- [ ] **Step 1.1.1: 在 `scripts/tests/test_predict.py` 末尾新增測試**

```python
# ============================================================================
# TW 時區切換測試（spec 2026-04-29）
# ============================================================================

def test_extract_game_date_tw_fallback_night_game():
    """夜場 ET 4/29 21:11 → UTC 2026-04-30T01:11Z → ET_date=4/29 → folder=4/30."""
    from predict import _extract_game_date_tw
    args = type("Args", (), {"game_data": None})()
    meta = {"game_date": "2026-04-30T01:11:00Z"}
    assert _extract_game_date_tw(args, meta) == "2026-04-30"


def test_extract_game_date_tw_fallback_early_day_game():
    """早場 ET 4/29 11:00 → UTC 2026-04-29T15:00Z → astimezone(TW) 是 4/29 23:00，
    但 user 規則 folder = ET_date + 1 = 4/30（不是直接 UTC→TW）。"""
    from predict import _extract_game_date_tw
    args = type("Args", (), {"game_data": None})()
    meta = {"game_date": "2026-04-29T15:00:00Z"}
    assert _extract_game_date_tw(args, meta) == "2026-04-30"


def test_extract_game_date_tw_path_based():
    """path 含 analysis-data/2026-04-30/... 直接拿 path segment。"""
    from predict import _extract_game_date_tw
    args = type("Args", (), {"game_data": "analysis-data/2026-04-30/TB@CLE/merged.json"})()
    meta = {"game_date": "2026-04-30T01:11:00Z"}
    assert _extract_game_date_tw(args, meta) == "2026-04-30"
```

- [ ] **Step 1.1.2: 跑測試確認失敗**

Run: `pytest scripts/tests/test_predict.py::test_extract_game_date_tw_fallback_night_game scripts/tests/test_predict.py::test_extract_game_date_tw_fallback_early_day_game scripts/tests/test_predict.py::test_extract_game_date_tw_path_based -v`

Expected: 3 個 FAIL，錯誤訊息類似 `ImportError: cannot import name '_extract_game_date_tw' from 'predict'`

### Step 1.2: 實作 `_extract_game_date_tw`

- [ ] **Step 1.2.1: 在 `scripts/predict.py:863` 之後新增 `_TW_TZ`**

找到原本 `_ET_TZ = timezone(timedelta(hours=-4))`（line 863），下方加一行：

```python
_ET_TZ = timezone(timedelta(hours=-4))
_TW_TZ = timezone(timedelta(hours=+8))
_ANALYSIS_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
```

- [ ] **Step 1.2.2: rename `_extract_game_date_et` → `_extract_game_date_tw` 並改 fallback 算法**

把 `scripts/predict.py:867-883` 整段改寫：

```python
def _extract_game_date_tw(args, meta: dict) -> str | None:
    """Derive game_date_tw from --game-data analysis-data path segment, or UTC→ET+1 fallback.

    User 規則（spec 2026-04-29）：folder = US gameday slate 用 TW notation = ET_date + 1。
    early ET day game（ET 4/29 11:00 = TW 4/29 23:00）也歸 ET_date+1（4/30），
    所以 fallback 用 `astimezone(ET).date() + 1` 而非直接 `astimezone(TW)`。
    """
    game_date_tw = None
    if getattr(args, "game_data", None):
        for part in os.path.normpath(args.game_data).split(os.sep):
            if _ANALYSIS_DATE_RE.match(part):
                game_date_tw = part
                break
    if not game_date_tw:
        game_date_iso = meta.get("game_date") or ""
        if game_date_iso:
            try:
                utc_dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
                et_date = utc_dt.astimezone(_ET_TZ).date()
                game_date_tw = (et_date + timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return game_date_tw
```

- [ ] **Step 1.2.3: 更新 line 1015 的呼叫處**

把 `scripts/predict.py:1015`：

```python
        record_date = _extract_game_date_et(args, meta) or _dt.now().strftime("%Y-%m-%d")
```

改成：

```python
        record_date = _extract_game_date_tw(args, meta) or _dt.now(_TW_TZ).strftime("%Y-%m-%d")
```

- [ ] **Step 1.2.4: 跑 3 個測試確認通過**

Run: `pytest scripts/tests/test_predict.py::test_extract_game_date_tw_fallback_night_game scripts/tests/test_predict.py::test_extract_game_date_tw_fallback_early_day_game scripts/tests/test_predict.py::test_extract_game_date_tw_path_based -v`

Expected: 3 個 PASS

- [ ] **Step 1.2.5: 跑全 test_predict.py 確認沒打到既有測試**

Run: `pytest scripts/tests/test_predict.py -v`

Expected: 全部 PASS（既有測試不應斷在 rename，因為它們不直接測 `_extract_game_date_et`）

### Step 1.3: 寫開打時間 meta 行測試

- [ ] **Step 1.3.1: 在 `scripts/tests/test_predict.py` 上一組測試後新增**

```python
def test_prediction_summary_includes_open_time_meta():
    """summary md 應在 H1 後加開打時間 meta 行（TW 為主、ET 副欄）。

    UTC 2026-04-30T01:11:00Z → TW 2026-04-30 09:11 / ET 04-29 21:11
    """
    from predict import format_prediction_summary_md
    record = _make_minimal_record(
        date="2026-04-30",
        game_time="2026-04-30T01:11:00Z",
    )
    md = format_prediction_summary_md(record, {"signals": [], "total_run_adjustment": 0.0}, [])
    assert "**開打時間**: 2026-04-30 09:11 TW（ET 04-29 21:11）" in md
    # meta 行緊接在 H1 之後（H1 + 空行 + meta 行 + 空行 + ## TL;DR）
    away_abbr_kc = "LAA"  # _make_minimal_record default away = Los Angeles Angels
    home_abbr_kc = "KC"   # default home = Kansas City Royals
    assert f"# Prediction Summary — {away_abbr_kc} @ {home_abbr_kc} (2026-04-30)\n\n**開打時間**:" in md


def test_prediction_summary_open_time_fallback_when_missing():
    """record 缺 game_time → 顯示「未知」。"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(date="2026-04-30")
    # 確認 fixture 沒帶 game_time（_make_minimal_record default 沒這欄）
    assert "game_time" not in record
    md = format_prediction_summary_md(record, {"signals": [], "total_run_adjustment": 0.0}, [])
    assert "**開打時間**: 未知" in md
```

- [ ] **Step 1.3.2: 跑測試確認失敗**

Run: `pytest scripts/tests/test_predict.py::test_prediction_summary_includes_open_time_meta scripts/tests/test_predict.py::test_prediction_summary_open_time_fallback_when_missing -v`

Expected: 2 個 FAIL — `assert "**開打時間**:" in md` 找不到（因為還沒加 meta 行）

### Step 1.4: 實作開打時間 meta

- [ ] **Step 1.4.1: 修改 `scripts/predict.py:741` 區塊**

找到 `format_prediction_summary_md` 內這段（line 741-758 附近）：

```python
    lines = [
        f"# Prediction Summary — {away_abbr} @ {home_abbr} ({date})",
        "",
        "## TL;DR",
```

改成：

```python
    # 開打時間 meta（spec 2026-04-29）：TW 為主、ET 副欄
    game_time_iso = record.get("game_time")
    if game_time_iso:
        try:
            utc_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
            tw_label = utc_dt.astimezone(_TW_TZ).strftime("%Y-%m-%d %H:%M TW")
            et_label = utc_dt.astimezone(_ET_TZ).strftime("%m-%d %H:%M")
            time_label = f"{tw_label}（ET {et_label}）"
        except ValueError:
            time_label = "未知"
    else:
        time_label = "未知"

    lines = [
        f"# Prediction Summary — {away_abbr} @ {home_abbr} ({date})",
        "",
        f"**開打時間**: {time_label}",
        "",
        "## TL;DR",
```

- [ ] **Step 1.4.2: 跑開打時間測試確認通過**

Run: `pytest scripts/tests/test_predict.py::test_prediction_summary_includes_open_time_meta scripts/tests/test_predict.py::test_prediction_summary_open_time_fallback_when_missing -v`

Expected: 2 個 PASS

- [ ] **Step 1.4.3: 跑全 test_predict.py 確保沒回歸**

Run: `pytest scripts/tests/test_predict.py -v`

Expected: 全部 PASS

### Step 1.5: 確認 line 1131 `record["date"]` 欄位（無需改 code）

- [ ] **Step 1.5.1: 確認 `record["date"]` 欄位現在是 TW**

`scripts/predict.py:1131` 已是 `"date": record_date,`。在 1.2.3 改完 `record_date` 來源後，這欄位自動變 TW（從 path 撈或 fallback ET+1）。**無需修改 code。**

### Step 1.6: Commit Task 1

- [ ] **Step 1.6.1: 確認 git status 只動到本 Task 範圍檔案**

Run: `git status --short`

Expected: 只看到
- `M scripts/predict.py`
- `M scripts/tests/test_predict.py`

- [ ] **Step 1.6.2: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(predict): record date + summary 開打時間改 TW

- _extract_game_date_et 改名 _extract_game_date_tw，fallback 算 ET_date+1
- 新增 _TW_TZ 常數；_dt.now() fallback 加上 _TW_TZ 明示
- format_prediction_summary_md 在 H1 之後加 **開打時間** meta 行
  格式：YYYY-MM-DD HH:MM TW（ET MM-DD HH:MM），缺 game_time 顯示「未知」
- 4 新測試：path-based / fallback night-game / fallback early-day-game / open-time meta + missing fallback

對應 spec docs/superpowers/specs/2026-04-29-prediction-tw-time-design.md §3-4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: predict.py — 移除 actual_* / verified 五個欄位

**Files:**
- Modify: `scripts/predict.py:1170-1174` (移除 5 lines)

### Step 2.1: 確認沒有測試引用這些欄位

- [ ] **Step 2.1.1: grep 確認**

Run: `grep -n "actual_winner\|actual_home_score\|actual_away_score\|actual_total\|verified" scripts/tests/test_predict.py`

Expected: 無任何 match（已預先確認；若有 match，需先評估該測試是否要刪）

### Step 2.2: 移除欄位

- [ ] **Step 2.2.1: 修改 `scripts/predict.py:1170-1174`**

找到 record dict 結尾這五行：

```python
            "actual_winner": None,
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_total": None,
            "verified": False,
        }
```

改成（直接刪 5 lines）：

```python
        }
```

- [ ] **Step 2.2.2: 跑全 test_predict.py 確認沒打到既有測試**

Run: `pytest scripts/tests/test_predict.py -v`

Expected: 全部 PASS

### Step 2.3: Commit Task 2

- [ ] **Step 2.3.1: Commit**

```bash
git add scripts/predict.py
git commit -m "$(cat <<'EOF'
chore(predict): record dict 移除 actual_*/verified 5 欄位

post-game-review skill 已棄置；這 5 個欄位是該 skill 的回填目標，
本 skill 內無讀取端，刪除避免冗餘 schema。

對應 spec docs/superpowers/specs/2026-04-29-prediction-tw-time-design.md §3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: prepare_game.py — --date TW 語意 + 內部 TW→ET

**Files:**
- Modify: `scripts/prepare_game.py:38-45` (--date help text), `:33-35` 後 (新增 `_tw_to_et`), `:451-455` (main step_a 呼叫處)
- Test: `scripts/tests/test_prepare_game.py` (新增 `_tw_to_et` 單元測試), `scripts/tests/test_prepare_game_steps.py` (補 ET 轉換 assertion)

### Step 3.1: 寫 `_tw_to_et` helper 測試

- [ ] **Step 3.1.1: 在 `scripts/tests/test_prepare_game.py` 末尾新增**

```python
def test_tw_to_et_converts_minus_one_day():
    """spec 2026-04-29 §2: et_date = tw_date − 1 day."""
    from prepare_game import _tw_to_et
    assert _tw_to_et("2026-04-30") == "2026-04-29"
    assert _tw_to_et("2026-05-01") == "2026-04-30"
    # 跨月
    assert _tw_to_et("2026-05-01") == "2026-04-30"
    # 跨年
    assert _tw_to_et("2027-01-01") == "2026-12-31"
```

- [ ] **Step 3.1.2: 跑測試確認失敗**

Run: `pytest scripts/tests/test_prepare_game.py::test_tw_to_et_converts_minus_one_day -v`

Expected: FAIL — `ImportError: cannot import name '_tw_to_et'`

### Step 3.2: 實作 `_tw_to_et`

- [ ] **Step 3.2.1: 在 `scripts/prepare_game.py:35` 之後新增 helper**

找到 `PYTHON = sys.executable`（line 34），下方加：

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _tw_to_et(tw_date: str) -> str:
    """TW date → ET date for MLB schedule API（spec 2026-04-29 §2）。

    規則：et_date = tw_date − 1 day（MLB 球季 EDT vs TW 永遠差 12 小時）。
    """
    from datetime import datetime, timedelta
    d = datetime.strptime(tw_date, "%Y-%m-%d").date()
    return (d - timedelta(days=1)).strftime("%Y-%m-%d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
```

- [ ] **Step 3.2.2: 跑測試確認通過**

Run: `pytest scripts/tests/test_prepare_game.py::test_tw_to_et_converts_minus_one_day -v`

Expected: PASS

### Step 3.3: 改 `--date` help text + main 呼叫處

- [ ] **Step 3.3.1: 修改 `scripts/prepare_game.py:39`**

把：

```python
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
```

改成：

```python
    parser.add_argument("--date", required=True,
                        help="YYYY-MM-DD（TW 開打日；內部換算 ET = TW − 1 day 給 MLB API）")
```

- [ ] **Step 3.3.2: 修改 `scripts/prepare_game.py:451-455` 的 step_a 呼叫**

找到 main() 內這段：

```python
    # Step A — sequential; must complete first
    ids = step_a(
        date=args.date, team_abbr=args.away, output_dir=output_dir,
        home_abbr=args.home, away_abbr=args.away, game_suffix=args.game_suffix,
    )
```

改成：

```python
    # Step A — sequential; must complete first
    # spec 2026-04-29: --date 是 TW 語意；step_a 內呼 fetch_game_data.py 給 MLB API 要 ET
    ids = step_a(
        date=_tw_to_et(args.date), team_abbr=args.away, output_dir=output_dir,
        home_abbr=args.home, away_abbr=args.away, game_suffix=args.game_suffix,
    )
```

注意：`compute_output_dir(date=args.date, ...)` 上方那行**不動**（folder 用 TW = `args.date` 直接）。

### Step 3.4: 補 omnibus test ET 轉換 assertion

- [ ] **Step 3.4.1: 修改 `scripts/tests/test_prepare_game_steps.py:625-660` 區塊**

找到 omnibus test（`test_main_runs_all_steps_in_order` 或類似名稱）內 `step_order` 的收集邏輯（line 625-635 附近）。需要新增收集每個 subprocess 呼叫的 `--date` 參數，並在最後 assert：

在 `fake_run` 內 `step_order.append(str(script))` 後，加一個收集 `--date` 的 dict：

```python
    step_order = []
    date_args = {}  # 新增：script_basename → --date arg

    def fake_run(*a, **k):
        cmd = a[0] if a else k.get("args", [])
        script = next((a for a in cmd if ".py" in str(a)), "")
        step_order.append(str(script))

        # 新增：抓 --date 後一個值
        for i, arg in enumerate(cmd):
            if str(arg) == "--date" and i + 1 < len(cmd):
                date_args[Path(str(script)).name] = str(cmd[i + 1])

        # ...（原本 -o 處理邏輯不變）
```

然後在最末尾既有 assert 後追加：

```python
    # spec 2026-04-29: fetch_game_data.py 應收到 ET = TW - 1
    # main 帶 --date 2026-04-28（TW），fetch_game_data 應收 2026-04-27
    assert date_args.get("fetch_game_data.py") == "2026-04-27", (
        f"fetch_game_data.py 應收 ET 2026-04-27（TW 4/28 - 1），實際 {date_args.get('fetch_game_data.py')}"
    )
```

- [ ] **Step 3.4.2: 跑 omnibus test 確認 PASS**

Run: `pytest scripts/tests/test_prepare_game_steps.py -v`

Expected: 全部 PASS（包含新 assertion）

- [ ] **Step 3.4.3: 跑全 prepare_game 測試**

Run: `pytest scripts/tests/test_prepare_game.py scripts/tests/test_prepare_game_steps.py -v`

Expected: 全部 PASS

### Step 3.5: Commit Task 3

- [ ] **Step 3.5.1: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game.py scripts/tests/test_prepare_game_steps.py
git commit -m "$(cat <<'EOF'
feat(prepare_game): --date 改 TW 語意 + 內部換算 ET

- 新增 _tw_to_et helper（TW → ET = TW - 1 day）
- --date help text 明示 TW 語意
- main step_a 呼叫處用 _tw_to_et(args.date) 給 fetch_game_data.py
- compute_output_dir 仍用 args.date（TW），folder 直接 TW 命名
- step_b/c/d 不吃 date 不需轉換
- 新增 _tw_to_et 單元測試 + omnibus test 補 ET 轉換 assertion

對應 spec docs/superpowers/specs/2026-04-29-prediction-tw-time-design.md §3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 刪 diagnose_ou_total_error.py + 文件 cross-ref 清理

**Files:**
- Delete: `scripts/diagnose_ou_total_error.py`
- Modify: `SKILL.md:20`
- Modify: `reference/prediction.md:188-192`

### Step 4.1: 刪 diagnose script

- [ ] **Step 4.1.1: 刪檔**

Run: `rm scripts/diagnose_ou_total_error.py`

- [ ] **Step 4.1.2: 確認無 test 殘留**

Run: `grep -rn "diagnose_ou_total_error\|diagnose_ou" scripts/tests/`

Expected: 無 match（預先確認過 scripts/tests/ 下無 test_diagnose 檔）

- [ ] **Step 4.1.3: 跑全 test 確保沒打到 import**

Run: `pytest scripts/tests/ -v`

Expected: 全部 PASS

### Step 4.2: 清 SKILL.md cross-ref

- [ ] **Step 4.2.1: 修改 `SKILL.md:20`**

把：

```markdown
**不適用**：整季預測 / 球員個人比較 / 賽後回顧（轉 `mlb-post-game-review`）/ 歷史統計查詢。
```

改成：

```markdown
**不適用**：整季預測 / 球員個人比較 / 賽後回顧 / 歷史統計查詢。
```

### Step 4.3: 清 reference/prediction.md cross-ref

- [ ] **Step 4.3.1: 修改 `reference/prediction.md:188-192`**

把：

```markdown
## 預測紀錄存放位置

- **Per-game（真相來源）**：`analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}/prediction.json`，由 `predict.py --save` 產生。
- **Per-date summary**：`analysis-data/{YYYY-MM-DD}/predictions.jsonl`，由 `mlb-post-game-review` skill 重建。
- **賽後回填** `actual_*` / `verified=true` 由 `mlb-post-game-review` skill 處理。
```

改成：

```markdown
## 預測紀錄存放位置

- **Per-game（真相來源）**：`analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}/prediction.json`，由 `predict.py --save` 產生。`{YYYY-MM-DD}` 為 TW 開打日（spec 2026-04-29）。
```

### Step 4.4: 確認沒漏網 cross-ref

- [ ] **Step 4.4.1: grep 確認 active docs 已清乾淨**

Run: `grep -n "mlb-post-game-review\|post-game-review\|predictions.jsonl\|actual_winner\|verified=true" SKILL.md reference/`

Expected: 無 match（歷史 plan/spec docs 內的 reference 不算 active，不需清）

### Step 4.5: Commit Task 4

- [ ] **Step 4.5.1: Commit**

```bash
git add -u SKILL.md reference/prediction.md scripts/diagnose_ou_total_error.py
git commit -m "$(cat <<'EOF'
chore: 移除 mlb-post-game-review skill 在本 skill 內的殘留

- 刪 scripts/diagnose_ou_total_error.py（依賴 predictions.jsonl，post-game 範疇）
- SKILL.md 移除「賽後回顧（轉 mlb-post-game-review）」cross-ref
- reference/prediction.md 「預測紀錄存放位置」精簡為只剩 per-game prediction.json
  並標註 folder date 是 TW 語意

歷史 plan/spec docs 內的 cross-ref 屬凍結紀錄，不動。

對應 spec docs/superpowers/specs/2026-04-29-prediction-tw-time-design.md §3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

- [ ] **Step F.1: 跑全 test 確保 4 個 commit 後沒回歸**

Run: `pytest scripts/tests/ -v`

Expected: 全部 PASS（既有 + 新增 6 個 TW 相關測試）

- [ ] **Step F.2: git log 確認 4 個 atomic commit**

Run: `git log --oneline -5`

Expected: 看到（從新到舊）：
```
xxxxxxx chore: 移除 mlb-post-game-review skill 在本 skill 內的殘留
xxxxxxx feat(prepare_game): --date 改 TW 語意 + 內部換算 ET
xxxxxxx chore(predict): record dict 移除 actual_*/verified 5 欄位
xxxxxxx feat(predict): record date + summary 開打時間改 TW
80c92b5 docs(spec): 開打時間掛 ET 副欄
dad41ef docs(spec): 預測流程 TW 時區切換設計
```

- [ ] **Step F.3: 手動 smoke test（選做）**

如果想實測（會打 MLB API + 寫 analysis-data/）：

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
$PYTHON scripts/prepare_game.py --date 2026-04-30 --away TB --home CLE
```

Expected:
- 終端輸出 `[A] game_data ✓` etc
- 產生 `analysis-data/2026-04-30/TB@CLE/`（folder = TW 4/30）
- 內部 fetch_game_data 應該成功（因為 ET 4/29 有 TB@CLE 的場次）

---

## 後續延伸（不在本 plan 範圍）

- 中間層 `dossier.md` / `merged_summary.md` / `*_summary.md` 是否要 TW 化 — spec §6 已聲明「不做」；若未來發現 AI 在 Phase 3 分析時被 ET 標記混淆，可起 follow-up spec
- prediction.json 是否要新增 `game_time_tw` / `game_date_tw` 派生欄位 — 目前不需要（summary 即時算就夠）
- 既有 `analysis-data/2026-04-28/` 等 ET 命名歷史資料夾是否 retroactive rename — spec §7 已聲明「不做」
