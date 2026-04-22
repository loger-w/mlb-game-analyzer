# MLB Skill Plan B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 實作 Plan B 三層防線重構：第 1 層 python hard-exit 堵 W 類繞過 + Y 類 cumulative 新缺口；第 2 層 markdown anchor（phase3_summary.md section grep）+ merged.json 擴欄；第 3 層 workflow.md TaskCreate 樣板。同步 SKILL.md 瘦身剩餘部分（刪 Rationalizations + Red Flags，去重到新 `flags-checklist.md`）。

**Architecture:** 7 個 phase 順序執行（非並行，確保 commit history 乾淨）。Phase 1 markdown；Phase 2-3 predict.py code（TDD）；Phase 4 merge_game_data + lineup_analyzer 擴欄；Phase 5 predict.py V 類 anchors（pitcher_triggers_yoy / phase3_summary grep）；Phase 6 workflow.md + cumulative.md；Phase 7 驗收。每個 task 獨立 commit。

**Tech Stack:** Python 3, pytest, argparse, regex (re), pathlib.Path, MLB 既有腳本（pitcher_stats.py / lineup_analyzer.py / merge_game_data.py / predict.py）

---

## 前置閱讀（必讀）

Engineer 開始前請依序讀：

1. `docs/specs/2026-04-22-mlb-skill-plan-b-design.md`（spec 本體，特別是 §4 Design Decisions 和 §6 測試策略）
2. `analysis-logs/cumulative.md` #1 / #3 / #4 / #8 / #9 / #10（Plan B 要解的 cumulative 問題）
3. `scripts/predict.py` L25-36（TEAM_ABBREV）、L67-79（should_force_ml_pass pattern）、L269-382（apply_rl_guardrail）、L735-784（argparse）、L883-1068（if args.save: 主流程）
4. `scripts/merge_game_data.py` L59-92（extract_pitcher_features / extract_lineup_features 現況）、L189-273（main）
5. `scripts/lineup_analyzer.py` L435-466（last7_ops 計算 + output dict）
6. `scripts/tests/test_predict_snapshot.py` L185-198（`_make_args` helper）、L504-699（現有 RL-1b tests）、L700-751（現有 should_force_ml_pass tests）

**語言規範**：code comments 保留原語言；commit messages 用繁體中文（跟現有 git log 一致）。

**commit 規範**：
- 用 explicit paths：例 `git add scripts/predict.py scripts/tests/test_predict_snapshot.py`
- 禁用 `git add -A` / `git add .`（memory: feedback_never_commit_sensitive_files）
- 每個 task 結束 commit 一次

---

## File Structure

### Files to CREATE

```
reference/flags-checklist.md                 (~60 LOC, 13 條去重 flag)
```

### Files to MODIFY

```
SKILL.md                                     (162 → ~110, -52)
reference/pitfalls.md                        (55 → ~40, -15)
reference/workflow.md                        (292 → ~330, +38)
reference/prediction.md                      (345 → ~350, +5 一行參數表注記)
scripts/predict.py                           (1072 → ~1140, +68)
scripts/merge_game_data.py                   (273 → ~310, +37)
scripts/lineup_analyzer.py                   (+3 加 last7_babip)
scripts/tests/test_predict_snapshot.py       (+ ~14 新 test)
analysis-logs/cumulative.md                  (更新 #8/#9/#10 狀態)
```

---

## Phase 1: Markdown 去重 + SKILL.md 瘦身

> **Dependencies:** 無。可最先執行。

### Task 1.1: 新寫 `reference/flags-checklist.md`

**Files:**
- Create: `reference/flags-checklist.md`

- [ ] **Step 1: 寫完整內容**

```markdown
# 旗標清單（Flags Checklist）

> 13 條分析紀律硬規則。任一條觸發 = 停下來，回到對應 Phase 閘門。

## 紀律來源

合併自舊 `pitfalls.md` Common Mistakes（12 條）+ SKILL.md Rationalizations / Red Flags（去重後加 1 條「語言」）。每條對應一個可立即停手的違規情境。

---

## 旗標

### 1. 用訓練資料/記憶代替腳本 API 輸出
所有核心數據（投手 ERA/xERA/IP、打者 wOBA/xwOBA、牛棚 ERA、BABIP）必須來自 `pitcher_stats.py` / `lineup_analyzer.py` / `fetch_game_data.py` 輸出。記憶與訓練資料不是來源。腳本失敗 → 回報錯誤等使用者指示，禁止靜默改走 WebSearch 或記憶。

### 2. BvP 樣本 <15 PA 硬推結論
PA < 15 是雜訊。標註「BvP 樣本不足」，不得引用趨勢。

### 3. Hot/Cold 判定未檢查 BABIP
近 7 天 BABIP 極端值（≤ .260 或 ≥ .370）預期回歸聯盟平均 ~.300。未檢查 = Hot/Cold 判定無效。

### 4. 牛棚傷兵只修 O/U 未修 ML
牛棚雙向閘門：偵測核心（Closer / Primary Setup / High-leverage）IL → 同時寫入 O/U 修正（+run）**和** ML 修正（-%）。只修一側 = Phase 3 未完成。

### 5. 同場推對立方向
D3 硬性規則：ML 推 A 隊 + A 隊受讓被禁。訊號方向 = 勝率強度，不是「兩邊下注對沖」。

### 6. 不寫 `phase3_summary.md` 就進 Phase 4
Phase 3 結論必須寫入 `$GAME_DIR/phase3_summary.md`。對話壓縮後結論會遺失 → Phase 4 就用記憶預測。

### 7. 跳過 Roster 檢查
Phase 2 Step 1 是阻塞閘門。IL 遺漏 → Phase 3 牛棚傷兵基礎就錯。

### 8. Agent 子代理跑 WebSearch / WebFetch
子代理無 WebSearch / WebFetch 權限，輸出是幻想。必須在主對話執行。平行可跑 BrowserFetch 以外的純計算腳本。

### 9. 省 `--game-data` / 腦補路徑
唯一合法命令：`predict.py --game-data analysis-data/<date>/<AWAY>@<HOME>/merged.json --save`。`prediction.json` 自動落在同層。省略或腦補 = predict.py 報錯或寫錯位置。

### 10. shell redirect `>` 取代 `--output` / `-o`
所有腳本必須用 `--output` / `-o`。`>` 會吃掉腳本 stderr + 破壞 pybaseball 的互動訊息輸出。

### 11. WebSearch 失敗繼續分析
WebSearch 失敗 → 向使用者回報錯誤，等待指示。不得「差不多就好」續推。

### 12. 中文對話用英文輸出
使用者中文 → 輸出必須繁體中文。搜尋可先用英文；**報告輸出照用戶語言**。

### 13. ERA-xERA 落差 ≥ 1.5 僅寫「風險提示」
可驗證的現象不得掛成條件性風險。觸發 `|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 低於 prior year ≥ 1.0` → 必須補跑 `pitcher_stats.py --year {YYYY-1} -o $GAME_DIR/{side}_pitcher_{YYYY-1}.json` 並執行 YoY Statcast 對比。閘門在 `workflow.md` Phase 2 Step 2；方法見 `matchup-factors.md#yoy-statcast-驗證`。

---

## 使用方式

每條規則均可透過 Phase 閘門自檢。完整 Phase 順序見 `workflow.md`；觸發時的補救動作見對應 Phase section。
```

- [ ] **Step 2: 確認行數**

Run: `wc -l reference/flags-checklist.md`
Expected: ~55-65 行（視空白行差異）

- [ ] **Step 3: Commit**

```bash
git add reference/flags-checklist.md
git commit -m "docs(mlb-skill): 新增 flags-checklist.md 13 條去重 flag（Plan B Task 1.1）"
```

---

### Task 1.2: `reference/pitfalls.md` 刪 Common Mistakes

**Files:**
- Modify: `reference/pitfalls.md`

- [ ] **Step 1: 讀現況**

Run: `wc -l reference/pitfalls.md` → 55 行

- [ ] **Step 2: Edit — 刪 Common Mistakes 節**

刪除 L7-L23（從 `## Common Mistakes（常見錯誤）` 起，到表格結尾，包含空白行直到 `## Edge Cases（邊界條件）` 前）。
保留：header（L1-6）+ `## Edge Cases` + `## 具體修正係數備忘`。

在 header 區塊（L1-6）加一行 note 指向新檔：

```markdown
# 常見錯誤與邊界條件

> 本檔專收 Edge Cases（非紀律違規的特殊情境）+ 具體修正係數速查。
> 13 條紀律違規 Flag → `flags-checklist.md`。
```

- [ ] **Step 3: 驗證**

Run: `wc -l reference/pitfalls.md`
Expected: ~38-42 行

Run: `grep -c "^## Common Mistakes" reference/pitfalls.md`
Expected: 0

- [ ] **Step 4: Commit**

```bash
git add reference/pitfalls.md
git commit -m "docs(mlb-skill): pitfalls.md 刪 Common Mistakes 節（Plan B Task 1.2）"
```

---

### Task 1.3: `SKILL.md` 刪 Rationalizations + Red Flags + 更新 link

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: 讀現況**

Run: `wc -l SKILL.md` → 162 行

- [ ] **Step 2: Edit — 刪 §Rationalizations（L55-L71）**

刪除從 `## Rationalizations — 藉口 vs 現實` 開頭，到該節結尾的空白行前（大約 17 行）。

- [ ] **Step 3: Edit — 刪 §Red Flags（L75-L91）**

刪除從 `## Red Flags — 停下來，回到流程` 開頭，到該節結尾（大約 20 行）。

- [ ] **Step 4: Edit — 更新 §Common Pitfalls 的 link target**

找到 L137-150 範圍內 Common Pitfalls 章節尾端的連結行（原為 `→ 完整清單（13 項 Common Mistakes + 12 項 Edge Cases）：`reference/pitfalls.md``），改為：

```markdown
→ 完整紀律 flag（13 條）：`reference/flags-checklist.md`
→ Edge Cases + 修正係數：`reference/pitfalls.md`
```

- [ ] **Step 5: 驗證**

```bash
wc -l SKILL.md                              # Expected: ~110-118
grep -c "Rationalizations\|Red Flags" SKILL.md  # Expected: 0
grep -c "flags-checklist.md" SKILL.md       # Expected: ≥1
```

- [ ] **Step 6: Commit**

```bash
git add SKILL.md
git commit -m "docs(mlb-skill): SKILL.md 刪 Rationalizations + Red Flags + 換 link（Plan B Task 1.3）"
```

---

## Phase 2: predict.py W 類 code 下沉（TDD）

> **Dependencies:** Phase 1（無強依賴，但建議順序執行）。

### Task 2.1: 廢除 `--run-line-rec` / `--run-line-stars`（W1）

**Files:**
- Modify: `scripts/predict.py` (argparse + apply_rl_guardrail + args.save 呼叫處)
- Modify: `scripts/tests/test_predict_snapshot.py` (更新 RL-1b tests)

- [ ] **Step 1: 寫新 test — 驗證 argparse 不再接受 --run-line-rec**

在 `scripts/tests/test_predict_snapshot.py` 末尾加：

```python
def test_w1_run_line_rec_arg_removed():
    """W1: --run-line-rec / --run-line-stars 已廢除，argparse 應 reject。"""
    import subprocess
    predict_py = os.path.join(os.path.dirname(__file__), "..", "predict.py")
    result = subprocess.run(
        [sys.executable, predict_py, "--test", "--run-line-rec", "NYY"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--run-line-rec" in result.stderr


def test_w1_run_line_stars_arg_removed():
    import subprocess
    predict_py = os.path.join(os.path.dirname(__file__), "..", "predict.py")
    result = subprocess.run(
        [sys.executable, predict_py, "--test", "--run-line-stars", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "--run-line-stars" in result.stderr
```

- [ ] **Step 2: Run tests — 預期 FAIL（CLI args 仍存在）**

Run: `pytest scripts/tests/test_predict_snapshot.py::test_w1_run_line_rec_arg_removed scripts/tests/test_predict_snapshot.py::test_w1_run_line_stars_arg_removed -v`
Expected: FAIL（returncode 目前應該是 0 or stderr 不含 "unrecognized"）

- [ ] **Step 3: Edit predict.py — 刪 argparse**

刪除 `scripts/predict.py` L749 和 L752：

```python
    parser.add_argument("--run-line-rec", help="Run line recommendation (team abbr or PASS)")  # L749 刪
    parser.add_argument("--run-line-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="Run line star rating")  # L752 刪
```

保留 L750 的 `--run-line`（run line 盤口值，非推薦 rec）。

- [ ] **Step 4: Edit predict.py — 改 apply_rl_guardrail 簽名**

修改 `apply_rl_guardrail` 函數（L269-382）：
1. 從簽名移除 `user_rl_rec: str | None,` 和 `user_rl_stars: int | None,` 兩行（約 L273-274）
2. 函數 body 內的 `user_rl_rec` / `user_rl_stars` 引用全部刪除 / 簡化

具體改動：

**簽名**（L269-280 → 新版）：
```python
def apply_rl_guardrail(
    *,
    adj_home: float,
    adj_away: float,
    trend_tags: list[str],
    predicted_winner: str,
    home_team: str,
    away_team: str,
    kelly_rl_available: bool = False,
) -> tuple[str, int | None, dict]:
```

**函數頂部初始化**（原 L322-324 `final_rl_rec = user_rl_rec if ...`）→ 改為：
```python
    final_rl_rec = "PASS"
    final_rl_stars = None
    rl_override = _inactive_rl_override()
```

**RL-1b gate**（原 L331 `if user_rl_rec in (None, "PASS") and diff >= RL_DIFF_MIN:`）→ 改為：
```python
    override_path = None
    if diff >= RL_DIFF_MIN:
        if diff >= RL_DIFF_BIG:
            override_path = "big-diff"
        elif strong_rl:
            override_path = "mid-diff+strong-tag"
```

**RL-2 sanity gate**（原 L377-380 `if final_rl_rec != "PASS" and final_rl_stars is None:`）→ 刪除整個 block（因為沒有 user input 路徑後，stars 永遠來自 override，不會 None 配非 PASS）。

**docstring** 更新（原 L281-319）：刪除 `user_rl_rec` / `user_rl_stars` 相關說明，改為：
- "Rules" 改為只說 RL-1b（auto override）；RL-1 和 RL-2 hard-gate 都已刪
- "Args" 刪 user_rl_rec / user_rl_stars 兩行
- 加一段 "Removed in Plan B (2026-04-22)": `--run-line-rec` / `--run-line-stars` CLI args 廢除；RL 全走 auto override（cumulative #10 W1 消除）。

- [ ] **Step 5: Edit predict.py — 改 if args.save: 呼叫處**

修改 `scripts/predict.py` L977-987 的 `apply_rl_guardrail` 呼叫，刪除 `user_rl_rec` / `user_rl_stars` 兩行：

```python
        final_rl_rec, final_rl_stars, rl_override = apply_rl_guardrail(
            adj_home=adj_home,
            adj_away=adj_away,
            trend_tags=trend_tags,
            predicted_winner=result["final"]["recommended_winner"],
            home_team=home_team,
            away_team=away_team,
            kelly_rl_available=False,
        )
```

- [ ] **Step 6: Edit tests — 更新現有 RL-1b tests**

在 `scripts/tests/test_predict_snapshot.py` 更新以下測試（L512-697 範圍），刪除 `user_rl_rec=` 和 `user_rl_stars=` kwargs：

受影響的 tests（從現有報告 L504-699）：
- `test_rl1b_mid_diff_strong_tag_1star` — 刪 `user_rl_rec=None, user_rl_stars=None`
- `test_rl1b_big_diff_no_tag_2star` — 同
- `test_rl1b_mid_diff_strong_tag_just_over_star_boundary` — 同
- `test_rl1b_diff_below_min_not_triggered` — 同
- `test_rl1b_mid_diff_without_strong_tag_not_triggered` — 同
- `test_rl1b_defensive_direction_mismatch_still_triggers` — 同
- `test_rl1b_not_gated_by_confidence_high_case` — 同
- `test_rl1b_not_gated_by_confidence_medium_case` — 同

**刪除**（功能不再存在）：
- `test_rl1b_high_confidence_user_supplied_abbr_respected` — 整個刪除（user abbr 路徑廢除）
- `test_rl1b_high_confidence_no_user_rec_auto_triggers` — 等同於 `test_rl1b_big_diff_no_tag_2star`，合併或刪
- `test_rl1b_respects_user_supplied_rec` — 整個刪除
- `test_rl1b_user_pass_treated_as_unspecified` — 整個刪除（user input 全無，PASS 是預設）

- [ ] **Step 7: Run tests — 預期全綠**

```bash
pytest scripts/tests/test_predict_snapshot.py -v
```
Expected: 所有 PASS；test_w1_* PASS；test_rl1b_* 更新後 PASS；delete 掉的 tests 消失。

- [ ] **Step 8: Verify grep**

```bash
grep -n "run_line_rec\|run_line_stars\|user_rl_rec\|user_rl_stars" scripts/predict.py
```
Expected: 無匹配（或只剩 `final_rl_rec` / `final_rl_stars` 這些不同名變數）

- [ ] **Step 9: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "refactor(mlb-skill): 廢除 --run-line-rec / --run-line-stars，RL 全走 auto override（Plan B Task 2.1 / W1）"
```

---

### Task 2.2: `--ml-rec` schema validation（W2）

**Files:**
- Modify: `scripts/predict.py` (加 helper + 呼叫)
- Modify: `scripts/tests/test_predict_snapshot.py` (新 tests)

- [ ] **Step 1: 寫 tests**

在 `scripts/tests/test_predict_snapshot.py` 末尾加：

```python
# ============================================================
# W2: --ml-rec schema validation
# ============================================================

def test_w2_ml_rec_accepts_valid_abbr():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("NYY", set(TEAM_ABBREV.values()))  # should not raise


def test_w2_ml_rec_accepts_pass():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("PASS", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_accepts_none():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec(None, set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_home():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("HOME", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_away():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("AWAY", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_bogus():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("ZZZ", set(TEAM_ABBREV.values()))
```

- [ ] **Step 2: Run tests — 預期 FAIL**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_w2 -v`
Expected: FAIL with `ImportError: cannot import name 'validate_ml_rec'`

- [ ] **Step 3: 新增 helper 到 predict.py**

在 `scripts/predict.py` `should_force_ml_pass` 之後（約 L80 後，`_SNAPSHOT_FILENAME_RE` 之前），加新函數：

```python
def validate_ml_rec(ml_rec: str | None, team_abbrevs: set[str]) -> None:
    """W2（Plan B 2026-04-22 §4.2）：`--ml-rec` 必須是 team abbr / PASS / None。

    舊 bug：傳 "HOME"/"AWAY" 字面值會寫進 predictions.jsonl，導致
    `review_stats.is_home_team` 查表失敗 → 反向判 WIN/LOSS（cumulative #9）。
    """
    valid = team_abbrevs | {"PASS"} | {None}
    if ml_rec not in valid:
        sorted_abbrs = sorted(team_abbrevs)
        sys.exit(
            f"⛔ --ml-rec 必須是 team abbr（如 NYY）或 PASS，收到 {ml_rec!r}\n"
            f"  合法值: {sorted_abbrs + ['PASS']}"
        )
```

- [ ] **Step 4: 在 main() 呼叫**

在 `scripts/predict.py` `main()` 的 `args = parser.parse_args()`（L784）之後加：

```python
    # W2: ml_rec schema validation（Plan B 2026-04-22 §4.2）
    validate_ml_rec(args.ml_rec, set(TEAM_ABBREV.values()))
```

- [ ] **Step 5: Run tests — 預期全 PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_w2 -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): --ml-rec schema 驗證（Plan B Task 2.2 / W2）"
```

---

### Task 2.3: `--game-data` 路徑 regex（W4）

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# W4: --game-data path regex validation
# ============================================================

def test_w4_game_data_valid_unix_path():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/merged.json")
    validate_game_data_path("/abs/path/to/analysis-data/2026-04-23/NYY@BOS/merged.json")


def test_w4_game_data_valid_windows_path():
    from predict import validate_game_data_path
    validate_game_data_path(r"C:\projects\analysis-data\2026-04-23\NYY@BOS\merged.json")


def test_w4_game_data_valid_doubleheader():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G1/merged.json")
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G2/merged.json")


def test_w4_game_data_rejects_missing_date():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/NYY@BOS/merged.json")


def test_w4_game_data_rejects_wrong_filename():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/foo.json")


def test_w4_game_data_rejects_bogus():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("/tmp/foo.json")
```

- [ ] **Step 2: Run — 預期 FAIL（ImportError）**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_w4 -v`

- [ ] **Step 3: 新增 helper + pattern 到 predict.py**

在 `TEAM_ABBREV` 定義後（約 L36 之後）加：

```python
# W4: --game-data 路徑規範（Plan B 2026-04-22 §4.2）
GAME_DATA_PATTERN = re.compile(
    r"analysis-data[/\\]\d{4}-\d{2}-\d{2}[/\\][A-Z]{2,3}@[A-Z]{2,3}(-G[12])?[/\\]merged\.json$"
)
```

在 `validate_ml_rec` 之後加：

```python
def validate_game_data_path(path: str) -> None:
    """W4（Plan B §4.2）：`--game-data` 路徑必須是 analysis-data/{date}/{AWAY}@{HOME}[-G1|G2]/merged.json。

    支援 Windows 反斜線 + absolute / relative path。腦補路徑（如 /tmp/foo.json）直接 reject。
    """
    normalized = path.replace("\\", "/")
    if not GAME_DATA_PATTERN.search(normalized):
        sys.exit(
            f"⛔ --game-data 路徑不符規範: {path}\n"
            f"  合法格式: analysis-data/YYYY-MM-DD/AWAY@HOME[-G1|-G2]/merged.json"
        )
```

- [ ] **Step 4: 在 main() 呼叫**

在 `main()` 的 `if not args.game_data: parser.error(...)`（L790-791）之後、`with open(args.game_data) ...`（L793）之前加：

```python
    # W4: game_data path regex（Plan B 2026-04-22 §4.2）
    validate_game_data_path(args.game_data)
```

- [ ] **Step 5: Run — 預期全 PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_w4 -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): --game-data 路徑 regex 驗證（Plan B Task 2.3 / W4）"
```

---

### Task 2.4: `signal_adjustments` allowlist 警告（W3）

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# W3: signal_adjustments allowlist warning
# ============================================================

def test_w3_signal_adjustments_known_prefix_silent(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bullpen_il_home": 0.3, "weather_mild_hr": 0.1})
    captured = capsys.readouterr()
    assert "unknown signal key" not in captured.err.lower()


def test_w3_signal_adjustments_unknown_warns(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"zzz_totally_bogus_xyz": 0.5})
    captured = capsys.readouterr()
    assert "zzz_totally_bogus_xyz" in captured.err
    assert "⚠️" in captured.err or "unknown" in captured.err.lower()


def test_w3_signal_adjustments_mixed(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({
        "bullpen_il_home": 0.3,
        "zzz_bogus": 0.2,
        "park_factor_adj": 0.1,
    })
    captured = capsys.readouterr()
    assert "zzz_bogus" in captured.err
    assert "bullpen_il_home" not in captured.err
    assert "park_factor_adj" not in captured.err


def test_w3_signal_adjustments_empty_or_none(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys(None)
    warn_unknown_signal_keys({})
    captured = capsys.readouterr()
    assert captured.err == ""


def test_w3_never_exits(capsys):
    """W3: 即便全部未知，也只警告不 exit。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bogus1": 1, "bogus2": 2, "bogus3": 3})  # should not raise
```

- [ ] **Step 2: Run — 預期 FAIL**

- [ ] **Step 3: 新增 allowlist + helper 到 predict.py**

在 `GAME_DATA_PATTERN` 後加：

```python
# W3: signal_adjustments 合法 key prefix（Plan B 2026-04-22 §4.2）
# 設計：prefix + pattern，非靜態完整 key（pitcher 名每週變動）
SIGNAL_KEYS_PREFIXES = frozenset({
    "bullpen_", "weather_", "cold_", "babip_", "park_",
    "coors_", "lineup_", "both_", "env_", "home_", "away_",
    "wind_", "sp_", "kelly_",
})
# 允許的 MLB team abbr prefix（lowercase，配合現有命名慣例）
SIGNAL_TEAM_PREFIXES = frozenset(
    abbr.lower() + "_" for abbr in TEAM_ABBREV.values()
)
# pitcher / player 個人化 signal 的後綴模式
SIGNAL_PITCHER_SUFFIX_RE = re.compile(
    r"_(velo|xera|era|bb|k|hr|hrluck|luck|new|role|small|ev|partial|gap|"
    r"regression|decline|collapse|wildness|drop|arsenal|il|out|variance|"
    r"platoon|disadvantage|reversion|reversal|warmth|partial|vs)(_|$)"
)


def _is_known_signal_key(key: str) -> bool:
    """W3: 判斷 signal key 是否在 allowlist 範圍。"""
    for prefix in SIGNAL_KEYS_PREFIXES:
        if key.startswith(prefix):
            return True
    for prefix in SIGNAL_TEAM_PREFIXES:
        if key.startswith(prefix):
            return True
    if SIGNAL_PITCHER_SUFFIX_RE.search(key):
        return True
    return False


def warn_unknown_signal_keys(signals: dict | None) -> None:
    """W3（Plan B §4.2）：未知 signal_adjustments key → stderr warning，不 exit。

    原則：允許新 signal 擴充（pitcher 名每週變動），但 typo / 幻想 key 留記錄。
    """
    if not signals:
        return
    unknown = [k for k in signals if not _is_known_signal_key(k)]
    if unknown:
        print(
            f"⚠️ unknown signal key(s): {sorted(unknown)}\n"
            f"  若為新 signal 請更新 SIGNAL_KEYS_PREFIXES；若為 typo 請修正",
            file=sys.stderr,
        )
```

- [ ] **Step 4: 在 main() 呼叫**

在 `main()` 的 argparse 之後、`validate_ml_rec` 之後加：

```python
    # W3: signal_adjustments allowlist warn（Plan B 2026-04-22 §4.2）
    warn_unknown_signal_keys(args.signal_adjustments)
```

- [ ] **Step 5: Run — 預期全 PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_w3 -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): signal_adjustments 未知 key 警告（Plan B Task 2.4 / W3）"
```

---

## Phase 3: predict.py Y 類 code 下沉（TDD）

> **Dependencies:** Phase 2（共同修改 predict.py）。

### Task 3.1: Y2 — xgb-predicted 矛盾 force PASS

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# Y2: xgb_home_lean vs predicted_winner divergent force PASS
# ============================================================

def test_y2_xgb_diverges_returns_true():
    """xgb 61% HOME but predicted_winner AWAY → True。"""
    from predict import check_xgb_divergent
    ml_pred = {"home_win_pct": 61.0}
    assert check_xgb_divergent(ml_pred, "AWAY") is True


def test_y2_xgb_aligned_home_returns_false():
    from predict import check_xgb_divergent
    ml_pred = {"home_win_pct": 58.0}
    assert check_xgb_divergent(ml_pred, "HOME") is False


def test_y2_xgb_aligned_away_returns_false():
    from predict import check_xgb_divergent
    ml_pred = {"home_win_pct": 42.0}
    assert check_xgb_divergent(ml_pred, "AWAY") is False


def test_y2_ml_pred_none_returns_false():
    from predict import check_xgb_divergent
    assert check_xgb_divergent(None, "HOME") is False


def test_y2_boundary_50_is_away_lean():
    """home_win_pct == 50.0 → AWAY lean（> 50 才算 HOME）。"""
    from predict import check_xgb_divergent
    ml_pred = {"home_win_pct": 50.0}
    assert check_xgb_divergent(ml_pred, "HOME") is True
    assert check_xgb_divergent(ml_pred, "AWAY") is False
```

- [ ] **Step 2: Run — 預期 FAIL**

- [ ] **Step 3: 新增 helper**

在 `scripts/predict.py` `should_force_ml_pass` 後（約 L79 後）加：

```python
def check_xgb_divergent(ml_pred: dict | None, predicted_winner: str) -> bool:
    """Y2（Plan B 2026-04-22 §4.4）：XGBoost 勝率方向 vs predicted_winner 矛盾。

    情境：signal_adjustments 翻轉 xgb 方向（如 xgb 61% HOME 但信號調整後 adj_away > adj_home
    → predicted_winner = AWAY）。此時 xgb 與最終推薦方向不一致 = 高不確定性，強制 PASS。

    與 D1 α 互不重疊：D1 比 ml_lean vs formula_lean（兩模型分歧）；Y2 比 xgb vs 最終推薦
    （signal 翻轉 xgb）。三者（D1、Y2、A6 user vs model）可獨立或同時觸發。
    """
    if not ml_pred:
        return False
    xgb_home_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
    return xgb_home_lean != predicted_winner
```

- [ ] **Step 4: 整合到 if args.save: 區塊**

修改 `scripts/predict.py` L897-902（在 D1 α block 後加 Y2 block），成為：

```python
        # α 實作：D1 方向分歧 → 強制 PASS（不依賴 cross_validation 字串）
        if should_force_ml_pass(ml_pred, formula_pred):
            ml_stars_cap = 0
            force_ml_pass = True
            cap_reasons.append("ml/formula 方向分歧 強制 PASS（α 實作）")

        # Y2: xgb_home_lean vs predicted_winner 矛盾（signal 翻轉 xgb）→ 強制 PASS
        # cumulative #8 觀察：連 2 天 3 次，升級 force PASS（spec §4.4）
        y2_triggered = False
        if check_xgb_divergent(ml_pred, result["final"]["recommended_winner"]):
            ml_stars_cap = 0
            force_ml_pass = True
            y2_triggered = True
            xgb_home_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
            pw = result["final"]["recommended_winner"]
            cap_reasons.append(
                f"xgb_home_lean={xgb_home_lean} vs predicted_winner={pw} 方向矛盾 強制 PASS（Y2）"
            )
```

然後在 `all_tags` 建構後（L942 之後）加 audit tag：

```python
        # Y2 audit tag
        if y2_triggered and "xgb-predicted-divergent" not in all_tags:
            all_tags.append("xgb-predicted-divergent")
```

- [ ] **Step 5: Run — 預期全 PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_y2 -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): Y2 xgb-predicted 方向矛盾 force PASS（Plan B Task 3.1）"
```

---

### Task 3.2: Y-new-2 — 近身戰（|diff| < 0.5）上限 1 星

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# Y-new-2: close game (|adj_diff| < 0.5) cap to 1
# ============================================================

def test_ynew2_close_game_caps_to_1():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 4.5, current_cap=5)
    assert new_cap == 1
    assert reason is not None
    assert "近身戰" in reason


def test_ynew2_wide_game_no_cap():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 6.8, current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew2_respects_tighter_existing_cap():
    """若 current_cap 已經更緊（如 0），不回升。"""
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 4.5, current_cap=0)
    assert new_cap == 0


def test_ynew2_boundary_0_5_not_triggered():
    """|diff| == 0.5 正好不觸發（strictly less than）。"""
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.0, 4.5, current_cap=5)
    assert new_cap == 5
    assert reason is None
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 新增 helper**

在 `check_xgb_divergent` 後加：

```python
def apply_close_game_cap(
    adj_home: float, adj_away: float, current_cap: int
) -> tuple[int, str | None]:
    """Y-new-2（Plan B §4.4）：調整後比分差 < 0.5 → 上限 1 星（SD ≈ 4.5，噪音範圍）。

    cumulative #3 連 4 天觸發，規則下沉到 code 層。
    """
    diff = abs(adj_home - adj_away)
    if diff < 0.5:
        reason = f"近身戰 |adj 比分差|={diff:.2f} < 0.5 上限 1（Y-new-2）"
        return min(current_cap, 1), reason
    return current_cap, None
```

- [ ] **Step 4: 整合到 if args.save:**

在 `scripts/predict.py` F2 direction override block 後（L921 之後），加：

```python
        # Y-new-2: 近身戰（|adj 比分差| < 0.5）上限 1 星（cumulative #3）
        ml_stars_cap, y_new_2_reason = apply_close_game_cap(adj_home, adj_away, ml_stars_cap)
        if y_new_2_reason:
            cap_reasons.append(y_new_2_reason)
```

- [ ] **Step 5: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_ynew2 -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): Y-new-2 近身戰上限 1 星（Plan B Task 3.2）"
```

---

### Task 3.3: Y-new-3 — divergent user tag 上限 2 星

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# Y-new-3: user-supplied 'divergent' tag caps to 2
# ============================================================

def test_ynew3_divergent_caps_to_2():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["divergent"], current_cap=5)
    assert new_cap == 2
    assert "divergent" in reason


def test_ynew3_no_divergent_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["early-season", "weather"], current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew3_mixed_tags_with_divergent():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["early-season", "divergent", "bullpen-il"], current_cap=5)
    assert new_cap == 2


def test_ynew3_respects_tighter_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, _ = apply_divergent_user_tag_cap(["divergent"], current_cap=1)
    assert new_cap == 1


def test_ynew3_empty_user_tags_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap([], current_cap=5)
    assert new_cap == 5
    assert reason is None
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 新增 helper**

在 `apply_close_game_cap` 後加：

```python
def apply_divergent_user_tag_cap(
    user_tags: list[str], current_cap: int
) -> tuple[int, str | None]:
    """Y-new-3（Plan B §4.4）：user-supplied 'divergent' tag → 上限 2 星。

    'divergent' 是 Phase 3 Claude 手動加的 user tag（非 compute_trend_tags 自動產生），
    代表基本面判讀與模型輸出方向不一致。cumulative #4 顯示推薦場 0W-4L。
    """
    if "divergent" in user_tags:
        return min(current_cap, 2), "'divergent' tag 上限 2（Y-new-3）"
    return current_cap, None
```

- [ ] **Step 4: 整合到 if args.save:**

在 Y-new-2 block 後加：

```python
        # Y-new-3: user-supplied 'divergent' tag → 上限 2 星（cumulative #4）
        user_tags_raw = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
        ml_stars_cap, y_new_3_reason = apply_divergent_user_tag_cap(user_tags_raw, ml_stars_cap)
        if y_new_3_reason:
            cap_reasons.append(y_new_3_reason)
```

**重要**：此處 `user_tags_raw` 是 Y-new-3 本地變數。下游 L939 `user_tags = [t.strip() for t in args.tags.split(",")] if args.tags else []` 保留不動（不同變數名，避免衝突）。

- [ ] **Step 5: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_ynew3 -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): Y-new-3 divergent tag 上限 2 星（Plan B Task 3.3）"
```

---

### Task 3.4: Y-new-1 — 主場 2★ audit tag

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# Y-new-1: home 2-star audit tag (observation only)
# ============================================================

def test_ynew1_home_2star_triggers_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "NYY") is True


def test_ynew1_away_2star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("AWAY", 2, "BOS") is False


def test_ynew1_home_3star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 3, "NYY") is False


def test_ynew1_home_2star_but_pass_no_tag():
    """final_ml_rec == PASS 時 tag 無意義（推薦已消）。"""
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "PASS") is False


def test_ynew1_home_2star_none_stars_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", None, "NYY") is False
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 新增 helper**

在 `apply_divergent_user_tag_cap` 後加：

```python
def should_add_home_2star_tag(
    predicted_winner: str, final_ml_stars: int | None, final_ml_rec: str | None
) -> bool:
    """Y-new-1（Plan B §4.4）：主場 2 星推薦 audit tag（cumulative #1 連 4 天觸發）。

    規則尚在觀察期（條件難定義，先 audit-only），tag `home-2star-risk` 供
    post-game review 分桶；不 cap / 不 force PASS。
    """
    return (
        predicted_winner == "HOME"
        and final_ml_stars == 2
        and final_ml_rec is not None
        and final_ml_rec != "PASS"
    )
```

- [ ] **Step 4: 整合到 if args.save:**

在 `all_tags` 建構後（L942，Y2 audit tag 之前或之後）加：

```python
        # Y-new-1: 主場 2 星 audit tag（cumulative #1，觀察期先加 tag 不 cap）
        if should_add_home_2star_tag(
            result["final"]["recommended_winner"], final_ml_stars, final_ml_rec
        ) and "home-2star-risk" not in all_tags:
            all_tags.append("home-2star-risk")
```

**順序**：與 Y2 audit tag 同一區塊（L942 之後），兩個 tag 都 append 到 `all_tags`。

- [ ] **Step 5: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_ynew1 -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): Y-new-1 主場 2 星 audit tag（Plan B Task 3.4）"
```

---

## Phase 4: merge_game_data.py + lineup_analyzer.py 擴欄（TDD）

> **Dependencies:** Phase 2-3 完成（確保 predict.py 基礎清理完）。

### Task 4.1: `lineup_analyzer.py` 加 `last7_babip`

**Files:**
- Modify: `scripts/lineup_analyzer.py`
- Modify: `scripts/tests/test_lineup_analyzer.py`（若無則 create）

- [ ] **Step 1: 確認 test file 存在**

Run: `ls scripts/tests/test_lineup_analyzer.py`
If missing: create new empty test file with pytest + fixture imports

- [ ] **Step 2: 寫 test**

```python
# scripts/tests/test_lineup_analyzer.py（片段）

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_last7_babip_averaged_from_core_lineup():
    """last7_babip = 核心打線近 7 天 BABIP 平均。"""
    # 測試不直接跑 pybaseball；直接測純計算函數
    # 如果 lineup_analyzer.py 沒有可獨立測試的函數，這個 test 改為 smoke
    import lineup_analyzer as la

    # 模擬 core_lineup（各含 last_7.babip）
    core_lineup = [
        {"last_7": {"babip": "0.320"}},
        {"last_7": {"babip": "0.280"}},
        {"last_7": {"babip": "0.340"}},
        {"last_7": {"babip": None}},  # 應被忽略
        {"last_7": {}},  # 缺 key 也忽略
    ]
    # 預期 avg = (0.320 + 0.280 + 0.340) / 3 = 0.313
    # 驗證策略：提煉一個 pure helper function `compute_last7_babip(core_lineup) -> float | None`
    # 然後測試這個 helper。
    from lineup_analyzer import compute_last7_babip
    result = compute_last7_babip(core_lineup)
    assert result is not None
    assert abs(result - 0.313) < 0.002


def test_last7_babip_empty_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([]) is None


def test_last7_babip_all_missing_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([{"last_7": {}}, {}]) is None
```

- [ ] **Step 3: Run — FAIL (ImportError)**

Run: `pytest scripts/tests/test_lineup_analyzer.py -v`

- [ ] **Step 4: 新增 `compute_last7_babip` helper + 整合**

在 `scripts/lineup_analyzer.py` 加 helper（約 L50-100 範圍，其他 helper 附近）：

```python
def compute_last7_babip(core_lineup: list[dict]) -> float | None:
    """近 7 天 BABIP 平均（Plan B 2026-04-22 §4.6）。

    從 core_lineup 每個打者的 `last_7.babip` 取值；非數值或缺失則忽略。
    空結果回 None（而非 0.0 — 差別是「沒數據」vs「平均 0」）。
    """
    values = []
    for b in core_lineup:
        val = (b.get("last_7") or {}).get("babip")
        if val is None:
            continue
        try:
            values.append(float(val))
        except (ValueError, TypeError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)
```

修改 output dict（約 L453-466），加一行：

```python
    return {
        "team": team,
        "team_id": team_id,
        "tier": tier,
        "avg_ops": round(avg_ops, 3),
        "avg_xwoba": round(avg_xwoba, 3) if avg_xwoba else None,
        "avg_babip": round(avg_babip, 3),
        "avg_k_pct": round(avg_k_pct, 1),
        "avg_bb_pct": round(avg_bb_pct, 1),
        "over_under_lean": over_under_lean,
        "recent_heat": recent_heat,
        "last7_babip": compute_last7_babip(core_lineup),  # Plan B §4.6（B10 BABIP 觸發用）
        "chain": chain,
        "lineup": core_lineup,
    }
```

- [ ] **Step 5: Run — PASS**

Run: `pytest scripts/tests/test_lineup_analyzer.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/lineup_analyzer.py scripts/tests/test_lineup_analyzer.py
git commit -m "feat(mlb-skill): lineup_analyzer 加 last7_babip 輸出（Plan B Task 4.1）"
```

---

### Task 4.2: `merge_game_data.py` 加 nested pitcher / lineup dict

**Files:**
- Modify: `scripts/merge_game_data.py`
- Create: `scripts/tests/test_merge_game_data.py`（若無則新建）

- [ ] **Step 1: 寫 tests**

```python
# scripts/tests/test_merge_game_data.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_nested_pitcher_from_pitcher_stats_output():
    """extract_pitcher_nested 從 pitcher_stats.py output 取 era/xera/ip + delta。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50, "xera": 2.80, "ip": 45.2, "fip": 3.10},
    }
    result = extract_pitcher_nested(pitcher_data, prior_pitcher_data=None, prefix="home")
    assert "home_pitcher" in result
    p = result["home_pitcher"]
    assert p["era"] == 3.50
    assert p["xera"] == 2.80
    assert p["ip"] == 45.2
    assert abs(p["era_xera_delta"] - 0.70) < 0.001
    assert p["prior_year"]["era"] is None


def test_nested_pitcher_with_prior_year():
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {"season": {"era": 3.50, "xera": 2.80, "ip": 45.2}}
    prior = {"season": {"era": 4.20, "xera": 3.90, "ip": 180.0}}
    result = extract_pitcher_nested(pitcher_data, prior, prefix="home")
    assert result["home_pitcher"]["prior_year"]["era"] == 4.20


def test_nested_pitcher_missing_fields_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {}}, None, prefix="home")
    assert result["home_pitcher"]["era"] is None
    assert result["home_pitcher"]["xera"] is None
    assert result["home_pitcher"]["ip"] is None
    assert result["home_pitcher"]["era_xera_delta"] is None


def test_nested_pitcher_season_error_treats_as_empty():
    """pitcher_stats.py 回 {season: {error: ...}} 時視為缺失（現有行為）。"""
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {"error": "lookup_failed"}}, None, prefix="away")
    assert result["away_pitcher"]["era"] is None


def test_nested_lineup_from_lineup_analyzer_output():
    from merge_game_data import extract_lineup_nested
    lineup_data = {"last7_babip": 0.320, "avg_babip": 0.290}
    result = extract_lineup_nested(lineup_data, prefix="home")
    assert result["home_lineup"]["recent_babip"] == 0.320


def test_nested_lineup_missing_last7_babip_returns_none():
    from merge_game_data import extract_lineup_nested
    result = extract_lineup_nested({"avg_babip": 0.290}, prefix="home")
    assert result["home_lineup"]["recent_babip"] is None
```

- [ ] **Step 2: Run — FAIL (ImportError)**

- [ ] **Step 3: 新增兩個 helper 到 `merge_game_data.py`**

在現有 `extract_pitcher_features` 後（L76）加：

```python
def extract_pitcher_nested(
    pitcher_data: dict,
    prior_pitcher_data: dict | None,
    prefix: str,
) -> dict:
    """Plan B §4.6：產 nested `{prefix}_pitcher` dict，包含 era/xera/ip + prior_year.era。

    給 predict.py 的 pitcher_triggers_yoy 讀（B7 YoY 觸發判斷）。
    與現有 `extract_pitcher_features` 共存（不動 flat keys，確保 review_stats 等 backward-compat）。
    """
    season = pitcher_data.get("season", {}) if pitcher_data else {}
    if "error" in season:
        season = {}
    prior_season = (prior_pitcher_data or {}).get("season", {}) if prior_pitcher_data else {}
    if "error" in prior_season:
        prior_season = {}

    era = season.get("era")
    xera = season.get("xera")
    ip = season.get("ip")
    delta = None
    if era is not None and xera is not None:
        delta = round(abs(era - xera), 3)

    return {
        f"{prefix}_pitcher": {
            "era": era,
            "xera": xera,
            "ip": ip,
            "era_xera_delta": delta,
            "prior_year": {"era": prior_season.get("era")},
        }
    }


def extract_lineup_nested(lineup_data: dict, prefix: str) -> dict:
    """Plan B §4.6：產 nested `{prefix}_lineup` dict，包含 recent_babip。

    給 predict.py 的 lineup_triggers_babip 讀（B10 BABIP 回歸觸發判斷）。
    """
    recent = lineup_data.get("last7_babip") if lineup_data else None
    return {
        f"{prefix}_lineup": {
            "recent_babip": recent,
        }
    }
```

- [ ] **Step 4: 整合到 main() 的 merge output 構造**

修改 `scripts/merge_game_data.py` L250-259（`merged = {}` 區塊）為：

```python
    merged = {}
    merged.update(extract_game_features(game_data))
    merged.update(extract_pitcher_features(home_pitcher_data, "home"))
    merged.update(extract_pitcher_features(away_pitcher_data, "away"))
    merged.update(extract_lineup_features(load_json(args.home_lineup), "home"))
    merged.update(extract_lineup_features(load_json(args.away_lineup), "away"))
    # Plan B §4.6: 新增 nested dict（B7 YoY / B10 BABIP 觸發判斷用），flat keys 保留不動
    merged.update(extract_pitcher_nested(
        home_pitcher_data, home_pitcher_prior_data, "home"
    ))
    merged.update(extract_pitcher_nested(
        away_pitcher_data, away_pitcher_prior_data, "away"
    ))
    merged.update(extract_lineup_nested(load_json(args.home_lineup), "home"))
    merged.update(extract_lineup_nested(load_json(args.away_lineup), "away"))
    merged["home_bullpen_era"] = home_bp_era
    merged["away_bullpen_era"] = away_bp_era
    merged["park_factor"] = park_factor
    merged.update(extract_meta(game_data, home_pitcher_data, away_pitcher_data))
```

**注意**：`home_pitcher_prior_data` / `away_pitcher_prior_data` 變數在 Task 4.3 定義。這個 task 先寫出呼叫方式，但在 Task 4.3 完成前，這裡暫時用 `None`：

```python
    merged.update(extract_pitcher_nested(home_pitcher_data, None, "home"))
    merged.update(extract_pitcher_nested(away_pitcher_data, None, "away"))
```

Task 4.3 會把 `None` 改為 `home_pitcher_prior_data` / `away_pitcher_prior_data`。

- [ ] **Step 5: Run — PASS**

Run: `pytest scripts/tests/test_merge_game_data.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/merge_game_data.py scripts/tests/test_merge_game_data.py
git commit -m "feat(mlb-skill): merge_game_data 加 nested pitcher / lineup dict（Plan B Task 4.2）"
```

---

### Task 4.3: `merge_game_data.py` 加 `--home-pitcher-prior` / `--away-pitcher-prior`

**Files:**
- Modify: `scripts/merge_game_data.py`
- Modify: `scripts/tests/test_merge_game_data.py`

- [ ] **Step 1: 寫 test**

```python
def test_prior_year_pitcher_file_loaded(tmp_path):
    """merge_game_data.py --home-pitcher-prior {path}.json 能讀 prior year era。"""
    import subprocess, json
    # 建 fixture
    home_prior = tmp_path / "home_pitcher_2025.json"
    home_prior.write_text(json.dumps({"season": {"era": 4.20, "xera": 3.90, "ip": 180.0}}))

    home_pitcher = tmp_path / "home_pitcher.json"
    home_pitcher.write_text(json.dumps({"season": {"era": 3.50, "xera": 2.80, "ip": 45.2}}))
    away_pitcher = tmp_path / "away_pitcher.json"
    away_pitcher.write_text(json.dumps({"season": {"era": 3.80, "xera": 3.50, "ip": 50.0}}))
    home_lineup = tmp_path / "home_lineup.json"
    home_lineup.write_text(json.dumps({"last7_babip": 0.320, "avg_babip": 0.290}))
    away_lineup = tmp_path / "away_lineup.json"
    away_lineup.write_text(json.dumps({"last7_babip": 0.280, "avg_babip": 0.300}))
    game = tmp_path / "game.json"
    game.write_text(json.dumps({
        "home_team": {"name": "New York Yankees"},
        "away_team": {"name": "Boston Red Sox"},
        "game_pk": 999,
        "game_date": "2026-04-23T23:05:00Z",
        "venue": {"name": "Yankee Stadium"},
        "home_recent_10": {"rs": 4.8, "ra": 4.1},
        "away_recent_10": {"rs": 5.2, "ra": 3.9},
        "home_recent_30": {"rs": 4.9, "ra": 4.2},
        "away_recent_30": {"rs": 5.1, "ra": 4.0},
        "home_season": {"rs": 4.8, "ra": 4.2, "games": 20},
        "away_season": {"rs": 5.0, "ra": 4.0, "games": 20},
    }))

    out = tmp_path / "merged.json"
    merge_py = os.path.join(os.path.dirname(__file__), "..", "merge_game_data.py")
    result = subprocess.run(
        [sys.executable, merge_py,
         "--game", str(game),
         "--home-pitcher", str(home_pitcher),
         "--away-pitcher", str(away_pitcher),
         "--home-lineup", str(home_lineup),
         "--away-lineup", str(away_lineup),
         "--home-pitcher-prior", str(home_prior),
         "--home-bullpen-era", "4.0",
         "--away-bullpen-era", "4.0",
         "--park-factor", "100",
         "-o", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    merged = json.loads(out.read_text())
    assert merged["home_pitcher"]["prior_year"]["era"] == 4.20
    assert merged["away_pitcher"]["prior_year"]["era"] is None
```

- [ ] **Step 2: Run — FAIL（CLI args 還沒加）**

- [ ] **Step 3: 加 CLI args + load**

在 `scripts/merge_game_data.py` `main()` argparse 區塊加（在 `--away-bullpen-era` 之後）：

```python
    parser.add_argument("--home-pitcher-prior", default=None,
                        help="Optional prior-year home pitcher stats JSON (for B7 YoY)")
    parser.add_argument("--away-pitcher-prior", default=None,
                        help="Optional prior-year away pitcher stats JSON (for B7 YoY)")
```

在 main() 內 load pitcher data 之後（load_json 呼叫區塊）加：

```python
    home_pitcher_prior_data = load_json(args.home_pitcher_prior) if args.home_pitcher_prior else None
    away_pitcher_prior_data = load_json(args.away_pitcher_prior) if args.away_pitcher_prior else None
```

**注意**：如果 `load_json` 只接受 required path，遇 None 會 error — 這裡要用 inline check 或讓 `load_json` tolerate None。檢查現有實作並酌情調整。

修改 Task 4.2 留的 `None` 呼叫為實變數：

```python
    merged.update(extract_pitcher_nested(
        home_pitcher_data, home_pitcher_prior_data, "home"
    ))
    merged.update(extract_pitcher_nested(
        away_pitcher_data, away_pitcher_prior_data, "away"
    ))
```

- [ ] **Step 4: Run — PASS**

Run: `pytest scripts/tests/test_merge_game_data.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_game_data.py scripts/tests/test_merge_game_data.py
git commit -m "feat(mlb-skill): merge_game_data 加 --home-pitcher-prior / --away-pitcher-prior（Plan B Task 4.3）"
```

---

## Phase 5: predict.py V 類 anchor（TDD）

> **Dependencies:** Phase 4（需 merged.json 有 nested pitcher / lineup keys）。

### Task 5.1: `pitcher_triggers_yoy` / `lineup_triggers_babip` helpers

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# B7 YoY trigger helper
# ============================================================

def test_pitcher_yoy_triggered_by_era_xera_gap():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.50, "xera": 1.88, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_triggered_by_small_ip_era_drop():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": 3.80}}) is True


def test_pitcher_yoy_not_triggered_normal():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.80, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 3.90}}) is False


def test_pitcher_yoy_boundary_1_5_triggers():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_boundary_1_49_no_trigger():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.51, "ip": 45.0, "prior_year": {"era": 4.00}}) is False


def test_pitcher_yoy_none_tolerant():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({}) is False
    assert pitcher_triggers_yoy({"era": None}) is False


def test_pitcher_yoy_no_prior_year_small_ip_not_triggered():
    """小 IP 但無 prior year 比較對象 → 不觸發 IP 路徑（era gap 路徑獨立）。"""
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": None}}) is False


# ============================================================
# B10 BABIP trigger helper
# ============================================================

def test_lineup_babip_low_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.250}) is True


def test_lineup_babip_high_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.380}) is True


def test_lineup_babip_normal_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.300}) is False


def test_lineup_babip_boundary_260_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.260}) is True


def test_lineup_babip_boundary_370_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.370}) is True


def test_lineup_babip_none_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": None}) is False
    assert lineup_triggers_babip({}) is False
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 新增 helpers 到 predict.py**

在 `should_add_home_2star_tag` 之後（約 L120 區域）加：

```python
def pitcher_triggers_yoy(pitcher: dict | None) -> bool:
    """B7 YoY 補跑觸發條件（Plan B §4.3）。

    True 當：
      - |ERA − xERA| ≥ 1.5 （本季數據內部 divergence）
      - OR IP < 30 且 ERA 比 prior_year.era 低 ≥ 1.0（小樣本劇烈 yoy 改善 → 需驗證持續性）

    None-tolerant：無 pitcher data 或關鍵欄缺失 → False（不誤觸發）。
    """
    if not pitcher:
        return False
    era = pitcher.get("era")
    xera = pitcher.get("xera")
    ip = pitcher.get("ip")
    prior_era = (pitcher.get("prior_year") or {}).get("era")
    if era is not None and xera is not None and abs(era - xera) >= 1.5:
        return True
    if (ip is not None and ip < 30
            and era is not None and prior_era is not None
            and era < prior_era - 1.0):
        return True
    return False


def lineup_triggers_babip(lineup: dict | None) -> bool:
    """B10 BABIP 回歸觸發條件（Plan B §4.5）。

    True 當 recent_babip ≤ .260 or ≥ .370（聯盟平均 ~.300，此範圍回歸確定性高）。
    """
    if not lineup:
        return False
    rb = lineup.get("recent_babip")
    if rb is None:
        return False
    return rb <= 0.260 or rb >= 0.370
```

- [ ] **Step 4: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k "pitcher_yoy or lineup_babip" -v`
Expected: 13 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): pitcher_triggers_yoy / lineup_triggers_babip helpers（Plan B Task 5.1）"
```

---

### Task 5.2: YoY prior year 檔存在性檢查 + `--skip-yoy-check` flag

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# B7: YoY prior year file existence check
# ============================================================

def test_yoy_check_triggered_missing_file_exits(tmp_path, monkeypatch):
    """觸發 YoY 但缺 prior year file → SystemExit with hint。"""
    import subprocess, json
    game_dir = tmp_path / "2026-04-23" / "NYY@BOS"
    game_dir.mkdir(parents=True)
    merged = game_dir / "merged.json"
    merged.write_text(json.dumps({
        "_meta": {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
                  "home_sp": "Max Fried", "away_sp": "Garrett Crochet",
                  "game_date": "2026-04-23T23:05:00Z", "venue": "Yankee Stadium"},
        "home_starter_fip": 3.50, "home_starter_k_bb": 18.0, "home_starter_whip": 1.20,
        "away_starter_fip": 3.80, "away_starter_k_bb": 22.0, "away_starter_whip": 1.15,
        "home_batting_xwoba": 0.330, "home_batting_ops": 0.780, "home_batting_k_pct": 20.0,
        "away_batting_xwoba": 0.340, "away_batting_ops": 0.800, "away_batting_k_pct": 21.0,
        "home_pitcher": {"era": 5.00, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 4.00}},
        "away_pitcher": {"era": 3.50, "xera": 3.40, "ip": 45.0, "prior_year": {"era": 3.60}},
        "home_lineup": {"recent_babip": 0.300},
        "away_lineup": {"recent_babip": 0.300},
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0, "park_factor": 100,
        "home_season_games": 22, "away_season_games": 22,
    }))
    predict_py = os.path.join(os.path.dirname(__file__), "..", "predict.py")
    result = subprocess.run(
        [sys.executable, predict_py, "--game-data", str(merged), "--save"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "B7" in result.stderr or "YoY" in result.stderr
    assert "pitcher_stats.py" in result.stderr


def test_yoy_check_skip_flag_bypasses(tmp_path):
    # 同 fixture，但加 --skip-yoy-check → 不因 YoY 退出；
    # 可能因 phase3_summary 缺退出（Task 5.3）— 此處只驗 YoY 被 bypass
    ...  # 實作時完整展開
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 加 `--skip-yoy-check` argparse**

在 `scripts/predict.py` argparse 區塊（main() 內）加：

```python
    parser.add_argument("--skip-yoy-check", action="store_true",
                        help="Bypass B7 YoY prior year file existence check (edge case / testing)")
    parser.add_argument("--skip-phase3-check", action="store_true",
                        help="Bypass phase3_summary.md section check (edge case / testing)")
```

- [ ] **Step 4: 加 imports**

在 `scripts/predict.py` imports 加：

```python
from pathlib import Path
```

- [ ] **Step 5: 加 YoY check block**

在 `if args.save:` 的 `original_ml_stars = args.ml_stars`（L892）之前加：

```python
        # B7 YoY prior year 檔存在性檢查（Plan B §4.3）
        if not args.skip_yoy_check:
            game_dir = Path(args.game_data).parent
            for side in ("home", "away"):
                side_pitcher = data.get(f"{side}_pitcher", {})
                if pitcher_triggers_yoy(side_pitcher):
                    # 從 _meta.game_date 推 YYYY-1
                    game_date_iso = data.get("_meta", {}).get("game_date", "")
                    current_year = int(game_date_iso[:4]) if game_date_iso[:4].isdigit() else _dt.now().year
                    prior_year = current_year - 1
                    prior_file = game_dir / f"{side}_pitcher_{prior_year}.json"
                    if not prior_file.exists():
                        pitcher_name = data.get("_meta", {}).get(f"{side}_sp", "UNKNOWN")
                        era = side_pitcher.get("era")
                        xera = side_pitcher.get("xera")
                        delta = abs(era - xera) if (era is not None and xera is not None) else "N/A"
                        sys.exit(
                            f"⛔ B7 YoY 紀律：{pitcher_name} "
                            f"|ERA-xERA|={delta} 或 IP<30 yoy drop 觸發，但缺 prior year data。\n"
                            f"請先跑：\n"
                            f"  pitcher_stats.py --name \"{pitcher_name}\" --year {prior_year} "
                            f"-o {prior_file}\n"
                            f"再重跑 predict.py；或加 --skip-yoy-check 跳過（測試用）。"
                        )
```

- [ ] **Step 6: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k test_yoy_check -v`

- [ ] **Step 7: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): B7 YoY prior year 檔存在性硬擋（Plan B Task 5.2）"
```

---

### Task 5.3: phase3_summary.md grep check + `--skip-phase3-check` flag

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 tests**

```python
# ============================================================
# Phase3 summary grep check
# ============================================================

def test_phase3_summary_missing_file_exits(tmp_path):
    """phase3_summary.md 不存在 → SystemExit（無論 trigger）。"""
    import subprocess, json
    game_dir = tmp_path / "2026-04-23" / "NYY@BOS"
    game_dir.mkdir(parents=True)
    merged = game_dir / "merged.json"
    merged.write_text(_minimal_merged_json_no_triggers())  # helper 產正常觸發數據

    predict_py = os.path.join(os.path.dirname(__file__), "..", "predict.py")
    result = subprocess.run(
        [sys.executable, predict_py, "--game-data", str(merged), "--save"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "phase3_summary" in result.stderr


def test_phase3_summary_yoy_trigger_missing_section_exits(tmp_path):
    """觸發 B7 YoY + phase3_summary.md 缺 `## YoY 對比結論` → SystemExit。"""
    # 建 prior year file（讓 Task 5.2 的 YoY 檔檢查通過）
    # phase3_summary.md 只有基本 section，無 YoY section
    ...  # 完整實作見 test body


def test_phase3_summary_babip_trigger_missing_section_exits(tmp_path):
    ...


def test_phase3_summary_bullpen_signal_missing_section_exits(tmp_path):
    ...


def test_phase3_summary_skip_flag_bypasses(tmp_path):
    """--skip-phase3-check 即便缺 section 也放行。"""
    ...


def test_phase3_summary_all_required_sections_present_pass(tmp_path):
    ...
```

（完整 test body 在執行時補；skeleton 先寫完）

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: 加 phase3_summary grep block**

在 `if args.save:` 的 B7 YoY check block 之後、`ml_stars_cap` 初始化之前，加：

```python
        # Plan B §4.5: phase3_summary.md section header 檢查（第 2 層防線）
        if not args.skip_phase3_check:
            game_dir = Path(args.game_data).parent
            phase3_path = game_dir / "phase3_summary.md"
            if not phase3_path.exists():
                sys.exit(
                    f"⛔ {phase3_path} 不存在 — Phase 3 結論未存檔\n"
                    f"  請先寫入分析結論（見 reference/workflow.md#3.5）；"
                    f"或加 --skip-phase3-check 跳過（測試用）。"
                )
            content = phase3_path.read_text(encoding="utf-8")

            required_sections = []
            if (pitcher_triggers_yoy(data.get("home_pitcher"))
                    or pitcher_triggers_yoy(data.get("away_pitcher"))):
                required_sections.append("## YoY 對比結論")
            if (lineup_triggers_babip(data.get("home_lineup"))
                    or lineup_triggers_babip(data.get("away_lineup"))):
                required_sections.append("## BABIP 回歸判定")
            sig_adj = args.signal_adjustments or {}
            if "bullpen_il_home" in sig_adj or "bullpen_il_away" in sig_adj:
                required_sections.append("## 牛棚雙向修正值")

            missing = [s for s in required_sections
                       if not re.search(rf"^{re.escape(s)}\b", content, re.M)]
            if missing:
                sys.exit(
                    f"⛔ phase3_summary.md 缺必要 section（Plan B §4.5）:\n"
                    f"  {missing}\n"
                    f"  請先補上；或加 --skip-phase3-check 跳過（測試用）。"
                )
```

- [ ] **Step 4: Run — PASS**

Run: `pytest scripts/tests/test_predict_snapshot.py -k phase3_summary -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): phase3_summary.md 必要 section 硬擋（Plan B Task 5.3）"
```

---

## Phase 6: workflow.md + prediction.md + cumulative.md

> **Dependencies:** Phase 2-5 完成（確保 CLI args / schema 已反映在 code）。

### Task 6.1: workflow.md Phase 2 Step 2 B7 TaskCreate + 刪除 --run-line-* 引用

**Files:**
- Modify: `reference/workflow.md`

- [ ] **Step 1: Edit — Phase 2 Step 2 加 TaskCreate block**

在 L136（Step 2 閘門最後一項 `ERA vs xERA 落差閘門` 那個 bullet 下）加：

```markdown

**B7 同步補跑 TaskCreate（forcing function，Plan B §4.7）：**

```
subject: 補跑 {side} YoY 對比（{pitcher_name}）
description: 對比 5 項 Statcast 指標（avg_velo / pitch_types / whiff_pct / hard_hit_pct / xera）；結論寫入 phase3_summary.md §YoY 對比結論
```

此 task 必須 complete 才能進 Phase 3.5。
```

- [ ] **Step 2: Edit — 刪除 L225 / L231-232 的 --run-line-rec / --run-line-stars 參數表引用**

在 Phase 4 參數表（L215-240 範圍），刪除以下行（原本在 L225 和 L231-232）：
- `| `--run-line-rec` | 可選 | 隊伍縮寫或 PASS ... |`
- `| `--run-line-stars` | 可選 | 0-5 |`

**同時加 note**：在 Phase 4 參數表上方加一行：

```markdown
> **RL 推薦**：無 `--run-line-rec` / `--run-line-stars`（Plan B 2026-04-22 廢除）。RL 全走 `predict.py` 自動 override（RL-1b gate：|adj 比分差| ≥ 1.5 + strong tag / big diff）。
```

- [ ] **Step 3: Verify**

```bash
grep -n "run-line-rec\|run-line-stars\|run_line_rec\|run_line_stars" reference/workflow.md
```
Expected: 無匹配（或只剩 `--run-line` 盤口值參數，不同於 --run-line-rec）

- [ ] **Step 4: Commit**

```bash
git add reference/workflow.md
git commit -m "docs(mlb-skill): workflow.md Phase 2 Step 2 加 B7 TaskCreate + 刪 run-line-* 參數（Plan B Task 6.1）"
```

---

### Task 6.2: workflow.md Phase 3.2 + 3.4 TaskCreate

**Files:**
- Modify: `reference/workflow.md`

- [ ] **Step 1: Edit — Phase 3.2 牛棚加 B9 TaskCreate**

在現有 L180-181 (3.2 牛棚 row) 之後，加一個獨立 block：

```markdown

**B9 牛棚雙向閘門擴充（Plan B §4.7，第 3 層 TaskCreate forcing function）：**

⛔ 偵測核心（Closer / Primary Setup / High-leverage）IL 任一人時，立即 TaskCreate：

```
subject: 牛棚雙向修正值（核心 {N} 人 IL）
description: 同時計算 ML 修正（-%）+ OU 修正（+run）；寫入 phase3_summary.md §牛棚雙向修正值；呼叫 predict.py 時 --signal-adjustments 含 bullpen_il_{side}
```

此 task 必須 complete 才能進 Phase 3.5。
```

- [ ] **Step 2: Edit — Phase 3.4 BABIP 加 B10 TaskCreate**

在 L182-183 （3.4 近期狀態）之後：

```markdown

**B10 BABIP 回歸閘門擴充（Plan B §4.7）：**

⛔ 偵測任一打線近 7 天 BABIP ≤ .260 或 ≥ .370 時，立即 TaskCreate：

```
subject: BABIP 回歸判定（{team} 近 7 天 {value}）
description: 回歸 ~.300 後判定 Hot/Cold 是否調整；結論寫入 phase3_summary.md §BABIP 回歸判定
```

此 task 必須 complete 才能進 Phase 3.5。
```

- [ ] **Step 3: Commit**

```bash
git add reference/workflow.md
git commit -m "docs(mlb-skill): workflow.md Phase 3.2 / 3.4 加 B9 B10 TaskCreate 樣板（Plan B Task 6.2）"
```

---

### Task 6.3: workflow.md Phase 轉換 TaskList 檢查

**Files:**
- Modify: `reference/workflow.md`

- [ ] **Step 1: Edit — Phase 2→3 轉換檢查**

在 Phase 2 末尾（約 L170，Phase 3 章節前）加：

```markdown

### Phase 2 → Phase 3 轉換檢查（Plan B §4.7）

⛔ 進入 Phase 3 前必須：

1. **TaskList 檢查**：前 Phase 產生的 V 類 tasks（B7 YoY 補跑）全部 complete
2. 有 pending task 不得進 Phase 3
```

- [ ] **Step 2: Edit — Phase 3→4 轉換檢查（Phase 3.5 前）**

在 L184（`### 3.5 分析結論存檔` 之前）加：

```markdown

### Phase 3 → Phase 3.5 轉換檢查（Plan B §4.7）

⛔ 進入 Phase 3.5（phase3_summary.md 存檔）前必須：

1. **TaskList 檢查**：本 Phase 產生的 V 類 tasks（B9 牛棚雙向、B10 BABIP 回歸）全部 complete
2. 有 pending task 不得進 Phase 3.5

> Phase 4（predict.py --save）會透過 phase3_summary.md grep 硬擋缺 section 的情況（第 2 層 code 防線，Plan B §4.5）。
```

- [ ] **Step 3: Commit**

```bash
git add reference/workflow.md
git commit -m "docs(mlb-skill): workflow.md 加 Phase 2→3 / 3→3.5 TaskList 檢查（Plan B Task 6.3）"
```

---

### Task 6.4: cumulative.md #8 / #9 / #10 狀態更新

**Files:**
- Modify: `analysis-logs/cumulative.md`

- [ ] **Step 1: Edit — 更新問題狀態**

找到 #8 行，把 `假設` / `待確認` 改為 `已修復（Plan B Phase 3.1 Y2）`：

```markdown
| 8 | xgb_raw 與 predicted_winner 內部矛盾 | **已修復（Plan B 3.1 Y2）** | 2026-04-20 | 3 | 3 | ... 已改 force PASS + audit tag `xgb-predicted-divergent`。|
```

#9 行改為 `已修復（Plan B Phase 2.2 W2）`：

```markdown
| 9 | **新：`ml_rec` 存字面值 `HOME`/`AWAY` 導致 `judge_ml` 誤判** | **已修復（Plan B 2.2 W2）** | 2026-04-21 | 1 | 1 | ... predict.py 加 validate_ml_rec schema，寫入前 reject。|
```

#10 行改為 `已修復（Plan B Phase 2.1 W1）`：

```markdown
| 10 | **新：人工 `--run-line-rec` 完全繞過 RL-1b gate 門檻** | **已修復（Plan B 2.1 W1）** | 2026-04-21 | 1 | 3 | ... 廢除 `--run-line-rec` / `--run-line-stars` CLI；RL 全走 auto override。|
```

- [ ] **Step 2: Commit**

```bash
git add analysis-logs/cumulative.md
git commit -m "docs(mlb-skill): cumulative.md #8 #9 #10 狀態更新為已修復（Plan B Task 6.4）"
```

---

## Phase 7: 驗收

### Task 7.1: 全 pytest + grep + LOC + 手動 integration

**Files:** (no modifications; verification only)

- [ ] **Step 1: 跑全 pytest**

```bash
cd scripts/tests && python -m pytest -v
```
Expected: 全部 PASS（含既有 36 + 新增 ~40 個 Plan B test）

- [ ] **Step 2: grep acceptance**

```bash
grep -n "run_line_rec\|run_line_stars" scripts/predict.py
# Expected: no matches
grep -n "Rationalizations\|Red Flags" SKILL.md
# Expected: no matches
```

- [ ] **Step 3: LOC targets**

```bash
wc -l SKILL.md reference/pitfalls.md reference/flags-checklist.md reference/workflow.md scripts/predict.py scripts/merge_game_data.py
```

Expected approx:
- `SKILL.md` ≤ 115
- `reference/pitfalls.md` ≤ 42
- `reference/flags-checklist.md` ~55-65
- `reference/workflow.md` 320-340
- `scripts/predict.py` 1130-1150
- `scripts/merge_game_data.py` 305-315

- [ ] **Step 4: 手動 integration — 挑一場 2026-04-23 比賽**

1. `python scripts/fetch_game_data.py --date 2026-04-23 --team NYY -o $GAME_DIR/game.json`
2. `python scripts/pitcher_stats.py --name "<home_sp>" --year 2026 -o $GAME_DIR/home_pitcher.json`
3. 同為 away / lineup / bullpen
4. 若 home_pitcher 觸發 YoY → 再跑一次 `--year 2025 -o $GAME_DIR/home_pitcher_2025.json`
5. `python scripts/merge_game_data.py --game $GAME_DIR/game.json --home-pitcher $GAME_DIR/home_pitcher.json ... --home-pitcher-prior $GAME_DIR/home_pitcher_2025.json -o $GAME_DIR/merged.json`
6. 寫 `$GAME_DIR/phase3_summary.md`（確保有所有觸發的 section）
7. `python scripts/predict.py --game-data $GAME_DIR/merged.json --save`
8. 驗證：
   - Valid `--ml-rec NYY` 成功；`--ml-rec HOME` 退出 ⛔
   - `--game-data /tmp/foo.json` 退出 ⛔
   - `--signal-adjustments '{"bogus_key":1}'` 印警告，繼續
   - 刪除 `home_pitcher_2025.json` 後重跑 → 退出 ⛔ 並印具體命令
   - 刪除 phase3_summary.md `## YoY 對比結論` section → 退出 ⛔
   - 加 `--skip-phase3-check` → 放行，存 prediction.json

- [ ] **Step 5: 回測相容性**

```bash
python scripts/review_stats.py --date 2026-04-20
# Expected: 不 crash；output 正常（舊 predictions.jsonl 無新欄位 tolerate）
```

- [ ] **Step 6: 最終 commit（若有需要）**

如果手動 integration 發現小 bug 需修正：
```bash
git add <修正的檔案>
git commit -m "fix(mlb-skill): Plan B 手動驗收發現的 <具體問題>（Plan B Task 7.1 fix）"
```

如果全綠無需修正：無新 commit，結束 Phase 7。

---

## 驗收準則（spec §6.3）

- [ ] 所有 pytest 綠
- [ ] `SKILL.md` ≤ 115 行
- [ ] `pitfalls.md` ≤ 42 行
- [ ] `flags-checklist.md` 13 條 + `pitfalls.md` Edge Cases 搬家完成
- [ ] `grep "run_line_rec\|run_line_stars" scripts/predict.py` 無結果
- [ ] `grep "Rationalizations\|Red Flags" SKILL.md` 無結果
- [ ] 手動跑一場新資料，所有新 guardrail 正確觸發
- [ ] `analysis-logs/cumulative.md` 狀態更新（#8/#9/#10）

---

## 風險與相容性（spec §8）

### R1. W1 廢除 `--run-line-rec` 後 RL 手動推薦能力消失
- 監控：5/1-5/15 RL 戰績若 <20%，回頭評估更寬鬆 RL-1b 門檻而非回復 CLI
- user 2026-04-22 明確接受此 trade-off

### R2. Y2 xgb-predicted force PASS 可能過殺
- 回測 4/21 NYY@BOS 會從 WIN 變 PASS；前 3 筆 cumulative 中 1 筆 WIN（33%）
- user 明確選 force PASS 而非 audit-only

### R3. phase3_summary.md hard exit 可能阻擋測試
- 保留 `--skip-phase3-check` flag

### R4. merged.json backward-compat
- 新 nested keys 加在既有 flat keys 之外；predict.py 讀 nested 時用 `.get()` tolerate；review_stats.py 繼續讀 flat。舊 merged.json 無 nested → Plan B triggers 全 False（不誤觸發）。

### R5. TaskCreate forcing function 依賴 Claude 自律
- `superpowers:using-superpowers` skill 強制 TaskList；日後遺忘再加 `blocks/blockedBy`。

---

## References

- Spec: `docs/specs/2026-04-22-mlb-skill-plan-b-design.md`
- Rule inventory: `docs/specs/2026-04-22-mlb-skill-rule-inventory.md`
- 瘦身 spec（Phase 0）: `docs/specs/2026-04-22-mlb-skill-slimming-design.md`
- Cumulative: `analysis-logs/cumulative.md`
- RL relaxation: `docs/specs/2026-04-20-rl-threshold-relaxation.md`
- RL symmetrization: `docs/superpowers/plans/2026-04-21-rl-symmetrization.md`
