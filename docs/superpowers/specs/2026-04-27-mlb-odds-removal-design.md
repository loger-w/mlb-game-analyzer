# MLB Game Analyzer — 盤口與 Kelly 系統移除 Design Spec

| 欄位 | 內容 |
|---|---|
| Date | 2026-04-27 |
| Branch | `refactor/skill-slimming` |
| Status | Approved |
| Supersedes | `2026-04-26-mlb-skill-slimming-design.md` D9（保留 fetch_odds 的決議反向） |
| 預估時程 | 2.5-3.3 hr，3 commits |
| 上游 brainstorm | 本檔由 superpowers:brainstorming 流程產出 |

---

## 1. Background & Motivation

mlb-game-analyzer skill 內含一套以 Pinnacle snapshot 為核心的盤口分析與 Quarter-Kelly 注碼計算系統，包括：

- `scripts/odds_analyzer.py`（661 行）— 盤口分析主邏輯
- `scripts/fetch_odds.py`（201 行）— Pinnacle snapshot 抓取器
- `odds_snapshots/` 目錄 — 34 個 historical snapshot 檔案
- `scripts/predict.py` 內 ~480 行 Kelly block 計算邏輯（含 `compute_kelly_block` / `load_closest_snapshot` / `resolve_pinnacle_odds`）
- `reference/odds-format.md`（62 行）— 盤口輸入格式
- `reference/prediction.md` Kelly Sizing & Unit Output 章節（99 行）
- `reference/output-format.md` 建議注碼表（6 行）
- 對應測試：`test_kelly.py`、`test_odds_analyzer_extended.py`、`test_predict_snapshot.py` 內 14 個 snapshot/Kelly 測試

**重構動機**：

使用者決定未來在本 skill **之外**獨立建構 Pinnacle 盤口時間序列追蹤系統做趨勢分析，且**不再灌入** `predict.py`。本 skill 退化為純預測工具，由星級（1-5 ⭐）作為唯一下注信心表達。

關鍵理由：
1. Kelly 在沒有 closing line 驗證下意義有限
2. 星級護欄已對齊紀律 D1-D5，足以表達信心
3. Pinnacle trend tracking 的資料結構與本 skill 的單場 prediction.json 不同步，獨立系統更乾淨
4. 移除 ~1500 行程式 + 測試 + 文件，大幅減少 maintain 負擔

此決議 **supersede** 既存 `2026-04-26-mlb-skill-slimming-design.md` 中的 D9（「保留 fetch_odds + 盤口追蹤系統留待後續另開設計」），但不修改該 spec 結構，僅在 D9 後加 superseded 註記。

---

## 2. 目標 / 非目標

### 目標

- 從本 skill 完全移除盤口相關的 scripts、測試、資料目錄、文件
- 從 `predict.py` 移除 Kelly block 計算與 snapshot 自動撈取邏輯，及對應 CLI args
- 報告輸出格式去除 Kelly 注碼建議表，星級（1-5 ⭐）作為唯一下注信心表達
- 既有的 slimming-spec D9 決議標記 superseded，不修改其結構
- 每個 commit 結束時 pytest 全綠 + predict.py smoke test 通過

### 非目標

- ❌ 修改既有 prediction.json 歷史檔（含 `kelly_block` 欄位的舊檔保留現狀）
- ❌ 修改非 Kelly 部分的 predict.py 邏輯（公式、星級、紀律 D1-D5、guardrails 不動）
- ❌ 重構 mlb-post-game-review skill（在另一台電腦）
- ❌ 新建外部 Pinnacle 趨勢追蹤系統（本 spec 範圍外，未來獨立 design）

---

## 3. 決議摘要

| # | 議題 | 決議 |
|---|---|---|
| **Q1** | Kelly 處理 | **整個刪除**。Kelly 邏輯 + CLI args + prediction.md 章節 + 測試全清。星級替代下注信心表達。 |
| **Q2** | fetch_odds.py + odds_snapshots/ | **整個刪除**。未來盤口記錄系統完全在本 skill 之外。 |
| **Q3** | 與 slimming spec 關係 | **獨立新建** design + plan。slimming spec D9 後加 superseded 註記，不改其結構。 |
| **Q4** | test_predict_snapshot.py | **rename → test_predict.py**，刪 14 個 snapshot/Kelly 測試（~412 行），保留 62 個非 odds 測試（~700 行）。檔名才符實際內容。 |
| **Q5** | Commit 切片 | **3 commits**：(1) predict.py 內部清理 → (2) 孤立 scripts/data 刪除 + test rename → (3) 文件 + spec。每個 commit 結束 pytest 全綠 + smoke test。修正 ordering：predict.py 先脫鉤再刪 odds_analyzer，避免中間 ImportError。 |

---

## 4. 三 Commit 執行計畫

### C1：predict.py 內部清理

**目標**：predict.py 不再依賴 odds_analyzer / fetch_odds / odds_snapshots，但這些檔案此 commit 還在 disk 上。

**動作**：
- 刪 `_SNAPSHOT_FILENAME_RE` + `load_closest_snapshot()` + `resolve_pinnacle_odds()` + `_NAME_TO_ABBREV`
- 刪 `compute_kelly_block()` 完整函式（含內部 `from odds_analyzer import (...)`）
- 刪 argparse CLI args：`--kelly-divisor` / `--kelly-cap` / `--unit-size` / `--no-auto-odds` / `--ml-odds-*-dec` / `--ou-odds-*-dec` / `--rl-odds-*-dec` / `--game-index`
- 刪 main() 內 Kelly 呼叫流程 + `record['kelly']` 寫入
- 刪 `kelly_available` 欄位（rl_override dict）+ `kelly_rl_available` 參數（apply_rl_guardrail signature）
- 刪 `import glob`（僅 load_closest_snapshot 使用）
- 刪 SIGNAL_KEYS_PREFIXES 內 `"kelly_"` prefix
- 同步刪 test_predict_snapshot.py 內 14 個 snapshot/Kelly 測試（不然測試會 import 已不存在的 symbol 失敗）

**完成條件**：
- pytest 全綠
- predict.py smoke test 通過（不會因為缺 odds 而失敗）
- 新產出 prediction.json 不含 `kelly` 欄位
- `grep -rn "kelly\|odds_analyzer\|load_closest_snapshot\|resolve_pinnacle_odds" scripts/predict.py` 無 hits

### C2：孤立 scripts、資料、fixtures 刪除 + test rename

**目標**：移除 C1 後不再被使用的所有檔案，重命名 test 檔反映實際內容。

**動作**：
- `git rm scripts/odds_analyzer.py` + `scripts/fetch_odds.py`
- `git rm scripts/tests/test_kelly.py` + `test_odds_analyzer_extended.py`
- `git rm scripts/tests/fixtures/sample_snapshot*.json` + `sample_merged.json`
- `git add odds_snapshots/`（已 staged D 的 34 檔確認刪除）
- `git mv scripts/tests/test_predict_snapshot.py scripts/tests/test_predict.py`

**完成條件**：
- pytest 全綠（62 測試從 `test_predict.py` 跑）
- predict.py smoke test 通過
- `git ls-files | grep -E "odds_analyzer|fetch_odds|test_kelly|...|odds_snapshots/"` 0 hits
- `python -c "import scripts.predict"` 無 ImportError

### C3：文件 + spec 更新

**目標**：文件對齊新狀態，新建本次 refactor 的獨立 design + plan，標記原 slimming spec D9 superseded。

**動作**：
- `git rm reference/odds-format.md`
- `reference/workflow.md`：刪 Phase 2 Step 3b 盤口賠率列 + odds_analyzer 命令 / 刪 Phase 4.0 自動 Odds 查詢段落 / Phase 4.3 標題改「推薦結果」
- `reference/prediction.md`：把 P(margin ≥ 2 | win) 表搬至 Run Line 章節（保留作 RL 計算用）/ 刪整章 Kelly Sizing & Unit Output
- `reference/output-format.md`：刪建議注碼 Quarter-Kelly 表 / TL;DR 「💰 盤口速查」改「📊 推薦速查」 / 第 10 段標題改「推薦結果」
- `SKILL.md`：description 移除「betting lines」/ L10「投打、牛棚、環境、盤口四層」改「投打、牛棚、環境三層」/ L16「盤口推薦」改「推薦方向」
- `2026-04-26-mlb-skill-slimming-design.md` D9 + 後續工作清單第 1 項加 superseded 註記
- 新建 `2026-04-27-mlb-odds-removal-design.md`（本檔）+ 對應 plan

**完成條件**：
- pytest 全綠（無回歸）
- predict.py smoke test 通過
- `grep -rn "kelly\|Kelly\|odds_analyzer\|fetch_odds\|odds_snapshots\|--ml-odds\|--ou-odds\|--rl-odds\|--no-auto-odds\|--kelly-" SKILL.md reference/ scripts/` 無 hits（除歷史 prediction.json）
- `grep "盤口" reference/ SKILL.md` 剩餘提及僅限紀律性提示
- D9 superseded 註記指向正確的新 spec 路徑

---

## 5. 風險與緩解

| 風險 | 緩解 |
|---|---|
| `compute_kelly_block` 內部 `from odds_analyzer import` 是 lazy import，C1 刪程式碼後若漏掉某處 `kelly` 欄位仍試圖讀 → ImportError | C1 完成條件含 grep 與 smoke test 雙重驗證；`import scripts.predict` 無 error |
| P(margin ≥ 2 \| win) 表在 Kelly 章節內，整章刪除會誤刪此表 | C3 動作 1 明列「先搬表，後刪章」順序；驗證 RL 章節整合後可獨立讀懂 |
| 既有 prediction.json 歷史檔含 `kelly` 欄位，下游 mlb-post-game-review 可能讀取 | Non-Goal 已說明歷史檔不修改；mlb-post-game-review 在另一台電腦，若依賴需另行處理 |
| RL 推薦邏輯依賴 P(margin) 表，若搬到 RL 章節後表內容讀法改變 | 表搬移時內容 100% 一致複製，僅改位置；RL 計算邏輯（predict.py 內 `apply_rl_guardrail`）不動 |
| Slimming spec D9 加 superseded 註記後，後續 reader 可能誤解 spec 仍在執行 | 註記明確指向新 design 路徑；新 design 的 Background 開頭也聲明 supersedes |

---

## 6. 後續工作（非本 spec 範圍）

1. **外部盤口趨勢追蹤系統**：完全獨立於本 skill 設計。可能涉及 smart money 流向追蹤、開盤到收盤的盤口移動、Pinnacle 多時間點 snapshot；資料結構與本 skill 解耦。
2. **mlb-post-game-review skill 整理**（在另一台電腦）：若該 skill 仍依賴歷史 prediction.json 的 `kelly` 欄位，需獨立處理 backwards compat。

---

**End of spec.**
