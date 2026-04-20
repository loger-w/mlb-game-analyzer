# Skill Path Routing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 MLB skill 把 `prediction.json` 寫到空 `games/` 資料夾的誘餌，改由 baseline test 驗證問題是否仍在，再以最小侵入方式（整入既有 Red Flags / Rationalizations / Phase 4）加強 SKILL.md 的路徑約束。

**Architecture:** TDD 四步走：(0) RED — baseline subagent 驗證問題仍存在；(1) GREEN — 僅在 Red Flags + Rationalizations 各加一項（不開新 section）；(2) 確認誘餌目錄 `games/` 已不存在；(3) Phase 4 補一句寫入前落點檢查；(4) 逐檔 stage commit。原 spec 的 20 行獨立 contract section 被壓縮為 2 行既有結構延伸 + 1 行 Phase 4 guard，利用模型已學會的自檢鉤子而非新讀硬背。

**Tech Stack:** Markdown 編輯、subagent baseline testing、bash 驗證、git 逐檔 stage commit。

**Pre-work 觀察結論（寫 plan 時已驗證）：**
- `games/` 目錄目前**已不存在**於 `C:\Users\Loger\.claude\skills\mlb-game-analyzer\`（spec 撰寫 2026-04-19 23:02 後已自行清掉）
- 全 repo 對 `games/` 的引用僅出現在 spec 本身 — 無 script / reference 引用
- 因此**問題是否仍存在未知**：誘餌已消失，模型可能已不會再寫錯；本 plan 先用 Task 0 baseline 釐清

**相對 spec 原版的關鍵偏離（審閱要點）：**
| Spec 原版 | 本 plan 版 | 理由 |
|---|---|---|
| 新增 20 行「資料儲存契約」獨立 section | Red Flags +1 行、Rationalizations +1 列 | 既有結構是模型自檢鉤子；獨立 section 僅讀到才生效 |
| 「`games/` 此資料夾不存在」 | 「不論是否存在均禁止寫入」 | 避免未來意外建目錄後此 claim 變謊言 |
| 直接 GREEN 動手改 | Task 0 先做 baseline 確認 | 遵守 writing-skills Iron Law — 無 failing test 不得寫 skill |
| Phase 4 加 guard | 保留（這點有價值） | Phase 4 = 實際寫入時點，guard 對應具體動作 |

---

### Task 0: Baseline Test — 確認問題在當前 SKILL.md 下仍會重現

**Files:**
- None (read-only subagent dispatch)

**Rationale:** `games/` 誘餌目錄已被清掉。若模型在沒有本 plan 的修改下已經寫對路徑，則整個 plan 後續 tasks 的邊際價值大幅下降，甚至**可能不該 merge**。必須先跑 RED，記錄實際行為。

- [ ] **Step 1: 派發 baseline subagent**

使用 Agent 工具（subagent_type=general-purpose，model=opus 或 omitted），prompt 大意：

```
你是 MLB 比賽分析 agent。你的 skill 目錄位於：
C:\Users\Loger\.claude\skills\mlb-game-analyzer\

任務：對一個虛構的 2026-04-20 Yankees @ Dodgers 比賽，
規劃完整的 predict.py 執行命令（不要真的跑，只輸出你會下的命令）。

限制：
- 只讀 SKILL.md，不讀 reference/ 下的任何檔案（模擬 reference 未載入的實況）
- 輸出：(a) 你會下的完整 predict.py 命令，(b) 你預期 prediction.json 會被寫到哪個路徑

報告限 80 字。
```

- [ ] **Step 2: 分類 baseline 結果**

根據 subagent 回傳判斷：

| 模型選擇 | 判定 | 下一步 |
|---|---|---|
| `--game-data analysis-data/2026-04-20/NYY@LAD/merged.json` | ✅ 路徑正確 | **本 plan 僅執行 Task 2（驗證 games/ 消失）+ Task 4（commit spec/plan 入庫）**；Task 1 / 3 跳過或降級為選擇性加固 |
| 寫到 `games/` 或 skill 根目錄或其他位置 | ❌ 仍會犯錯 | 記錄 rationalization（模型自述原因）→ 正常進入 Task 1 |
| 模型拒答或要求更多資訊 | ⚠ 資訊不足 | 再派一次，prompt 加「你**必須**給出具體路徑」 |

- [ ] **Step 3: 紀錄 baseline 結果到 plan 底部**

在本 plan 檔末尾追加一段 `## Baseline Result (Task 0)`，至少記：
- subagent 輸出的命令字串
- 判定結果（正確 / 錯誤 / 資訊不足）
- 若錯誤：模型給出的理由原文

此紀錄作為後續是否 merge 的依據。

- [ ] **Step 4: 與使用者確認下一步**

回報 baseline 結果給使用者，確認：
- 正確 → 是否同意只跑 Task 2 + Task 4
- 錯誤 → 繼續 Task 1–4
- 由使用者決定，不自動推進

---

### Task 1: 整合進 SKILL.md 既有結構（GREEN）

**Files:**
- Modify: `C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md`
  - Rationalizations 表（約 line 60-69）增一列
  - Red Flags 清單（約 line 76-87）增一項

**Rationale:** 這兩個既有結構是模型學會**主動自檢**的鉤子，命中率高於獨立 section。總增量 2 行，遠小於 spec 原版 20 行。

**Gate:** 僅在 Task 0 判定「仍會犯錯」時執行。若 baseline 已正確則跳過。

- [ ] **Step 1: 讀取 SKILL.md 確認 Rationalizations 表結構**

Run: `Read SKILL.md offset 55 limit 20`
Expected: 確認 line 60-68 為 Rationalizations 表格，line 69 為空行，line 70 為 `**以上任何一項 = 停下來...**`。

- [ ] **Step 2: 在 Rationalizations 表最後一列後插入新列**

使用 Edit 工具。基於實際檔案內容，`old_string` 為表格最後一列 + 其下的粗體結語（確保唯一性）：

```
| 「Agent 子代理平行跑 WebSearch 比較快。」 | 子代理沒 WebSearch 權限，輸出是幻想。主對話平行跑才對。 |

**以上任何一項 = 停下來，回到對應 Phase 的閘門檢查清單。**
```

`new_string` 為：

```
| 「Agent 子代理平行跑 WebSearch 比較快。」 | 子代理沒 WebSearch 權限，輸出是幻想。主對話平行跑才對。 |
| 「路徑太長，先寫 `games/` 或 skill 根目錄之後再搬。」 | 搬動 = 多一步錯誤點。第一筆就寫對 `analysis-data/<date>/<AWAY>@<HOME>/`，這是唯一合法位置。 |

**以上任何一項 = 停下來，回到對應 Phase 的閘門檢查清單。**
```

- [ ] **Step 3: 在 Red Flags 清單最後一項後插入新項**

`old_string` 為 Red Flags 清單最後兩項 + 結語（注意使用 Read 結果的實際字元）：

```
- 用 shell redirect `>` 因為「比較快」
- 用 Agent 子代理跑 WebSearch

**以上任何一項 = 停下來，回到對應的 Phase 閘門。**
```

`new_string` 為：

```
- 用 shell redirect `>` 因為「比較快」
- 用 Agent 子代理跑 WebSearch
- 把 `prediction.json` 寫到 `analysis-data/<date>/<AWAY>@<HOME>/` 以外的任何位置（不論 `games/` 是否存在，均禁止寫入）

**以上任何一項 = 停下來，回到對應的 Phase 閘門。**
```

- [ ] **Step 4: 驗證插入**

Run:
```bash
grep -n "第一筆就寫對" "C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md"
grep -n "不論 \`games/\` 是否存在" "C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md"
```

Expected: 兩條 grep 各回一行，分別落在 Rationalizations 表格區間與 Red Flags 區間。

---

### Task 2: 驗證 games/ 目錄不存在

**Files:**
- Verify: `C:\Users\Loger\.claude\skills\mlb-game-analyzer\games\`（應不存在）

**Rationale:** Pre-work 已確認目錄不在，本 task 僅做最終 locking。

- [ ] **Step 1: 驗證目錄不存在**

Run:
```bash
ls -la "C:\Users\Loger\.claude\skills\mlb-game-analyzer\games" 2>&1 || echo "GAMES_DIR_ABSENT"
```

Expected: `No such file or directory` 或 `GAMES_DIR_ABSENT`。

- [ ] **Step 2: 驗證無其他活躍 `games/` 引用**

使用 Grep 工具：
- pattern: `games/`
- path: `C:\Users\Loger\.claude\skills\mlb-game-analyzer`
- output_mode: `files_with_matches`

Expected: 僅 spec、本 plan、SKILL.md（Red Flags 條目內）；**不得**命中 `scripts/` 或 `reference/`。

- [ ] **Step 3: 意外存在時刪除（僅條件執行）**

Condition: 僅當 Step 1 列出檔案才跑。

```bash
ls "C:\Users\Loger\.claude\skills\mlb-game-analyzer\games"
# 有檔 → 先 mv 到 archive/
# 空 → rmdir
rmdir "C:\Users\Loger\.claude\skills\mlb-game-analyzer\games"
```

---

### Task 3: Phase 4 加寫入前落點 guard

**Files:**
- Modify: `C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md`（Phase 4 段，約 line 123-129）

**Rationale:** Phase 4 是 `prediction.json` 實際寫入時點。在具體動作旁放 guard，與 Task 1 的心智鉤子（自檢 rationalization）形成雙層保險。

**Gate:** 建議不論 Task 0 判定為何皆執行（guard 本身只加 1 行、風險極低、對 Phase 4 讀者直接有用）。

- [ ] **Step 1: 讀取 Phase 4 確認插入點**

Run: `Read SKILL.md offset 121 limit 12`
Expected: 確認 Phase 4 段結尾為 `→ 詳細（...）：reference/workflow.md#phase-4預測輸出`。

- [ ] **Step 2: 插入 guard 句**

`old_string`（取足上下文保證唯一）：

```
- **賽後彙總 / 回填**：轉交 `mlb-post-game-review`

→ 詳細（`--save` 參數表、輸出前驗證清單、輸出格式）：`reference/workflow.md#phase-4預測輸出`
```

`new_string`：

```
- **賽後彙總 / 回填**：轉交 `mlb-post-game-review`

> ⚠️ 寫 `prediction.json` 前確認 `--game-data` 指向 `analysis-data/<date>/<AWAY>@<HOME>/merged.json`；不對就停下來重新定位，不得自建替代目錄。

→ 詳細（`--save` 參數表、輸出前驗證清單、輸出格式）：`reference/workflow.md#phase-4預測輸出`
```

- [ ] **Step 3: 驗證**

```bash
grep -n "寫 \`prediction.json\` 前確認" "C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md"
```

Expected: 回一行，位於 Phase 4 section 之內。

---

### Task 4: 最終驗證 + 逐檔 commit

**Files:**
- Stage: `SKILL.md`（若改）、`docs/specs/2026-04-20-skill-path-routing.md`、`docs/superpowers/plans/2026-04-20-skill-path-routing.md`
- **NEVER `git add -A`** — 逐檔 stage（遵守 never-commit-sensitive-files 約束）
- **不** stage `scripts/pitcher_stats.py`（pre-existing unstaged，不屬本 task）

- [ ] **Step 1: 檢查 git status**

```bash
git status
```

Expected:
- `SKILL.md`（如 Task 1/3 有執行）modified
- 兩個 docs 檔案 untracked 或 modified
- `scripts/pitcher_stats.py` modified（忽略）

- [ ] **Step 2: 檢查 diff**

```bash
git diff SKILL.md
```

Expected: 僅預期的 2-3 處變更（Task 1 表格 +1 列、Red Flags +1 項；Task 3 Phase 4 +1 行）。**不得**有意外修改。

- [ ] **Step 3: 完整 repo 驗證**

使用 Grep 工具：
- pattern: `games/`
- path: `C:\Users\Loger\.claude\skills\mlb-game-analyzer\scripts`
- output_mode: `files_with_matches`

重複對 `reference` 路徑再跑一次。

Expected: 兩次皆無 match。

- [ ] **Step 4: 逐檔 stage**

```bash
git add SKILL.md
git add docs/specs/2026-04-20-skill-path-routing.md
git add docs/superpowers/plans/2026-04-20-skill-path-routing.md
git status
```

Expected: 三檔 staged；`scripts/pitcher_stats.py` 仍 unstaged。

- [ ] **Step 5: Commit**

Commit message 依 Task 0 結果選一版：

**(a) Task 0 判定「仍會犯錯」且執行了 Task 1+3：**

```bash
git commit -m "$(cat <<'EOF'
docs(mlb-skill): harden path-routing via Red Flags + Rationalizations

Baseline subagent 在只讀 SKILL.md 時仍會寫錯 prediction.json 路徑。
將約束整入既有 Red Flags / Rationalizations 兩個自檢鉤子各 +1 項，
加上 Phase 4 寫入前落點 guard 一行；不採獨立 section（省 token、
利用模型已學會的自檢習慣）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**(b) Task 0 判定「已寫對」，僅執行 Task 3：**

```bash
git commit -m "$(cat <<'EOF'
docs(mlb-skill): add Phase 4 write-location guard + route-fix spec/plan

Baseline 確認當前 SKILL.md 已足以引導模型寫對 analysis-data/ 路徑
（誘餌 games/ 目錄已清除即解）。保留 Phase 4 一行 guard 作雙層保險，
spec/plan 入庫供後續回溯。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Post-commit 確認**

```bash
git log --oneline -1
git status
```

Expected: 新 commit 可見；working tree 僅剩 `scripts/pitcher_stats.py` unstaged。

---

## Self-Review 結果

1. **Spec coverage:**
   - Spec Step 1（路徑契約）→ Task 1（改以整入既有結構取代獨立 section；**偏離需審核確認**）
   - Spec Step 2（刪 games/）→ Task 2（退化為驗證）
   - Spec Step 3（Phase 4 guard）→ Task 3（保留）
   - 新增 Task 0 baseline → 遵守 writing-skills Iron Law
   - 新增 Task 4 收尾 commit

2. **Placeholder scan:** 所有 step 含具體指令、程式碼、grep 字串、預期輸出；無 TBD。

3. **Type consistency:** N/A（無型別）。但 `<date>`、`<AWAY>@<HOME>` 在 Task 1/3 統一使用同格式。

4. **決策需使用者確認之處：**
   - 是否同意以「Red Flags + Rationalizations 各 +1」取代 spec 原定的「獨立 20 行 contract section」
   - Task 0 baseline 結果為「已寫對」時，是否同意僅 merge Phase 4 guard + 入庫文件、不動 Red Flags / Rationalizations

5. **Verification:** Spec「前置檢查」→ Task 2 Step 1-2 + Task 4 Step 3；Spec「功能驗證」需實際比賽時驗收，不在本 plan 範圍。

---

## Baseline Result (Task 0)

**執行時間：** 2026-04-20（本 session）
**Subagent 設定：** general-purpose, opus（預設），限制僅讀 SKILL.md、禁 reference/scripts

**Subagent 輸出：**
```
命令: python scripts/predict.py --save
落點: $GAME_DIR/prediction.json (依 Phase 4 慣例)
依據: SKILL.md L125「執行：predict.py --save → assemble_analysis.py --validate → upload_prediction.py」；
      L37 Phase 4 輸出 prediction.json；詳細參數表在 reference/workflow.md (未載入)，故僅用 --save。
```

**判定：❌ 仍會犯錯**

**失敗分析：**
1. **缺 `--game-data` 參數** — 當前 SKILL.md 僅寫 `predict.py --save`，沒提 `--game-data`。模型依此命令實跑 `predict.py` 會報錯。
2. **`$GAME_DIR` 是幻想** — SKILL.md 全文無 `$GAME_DIR` 定義；變數只存在於 reference/workflow.md:20。Subagent 在被禁讀 reference 的情況下仍引用此變數 = 從訓練資料或記憶腦補。
3. **錯誤 recovery 路徑** — 當命令執行失敗，模型會臨場自救，此時若看到根目錄曾有 `games/` 資料夾（雖已刪，但若某次意外建回來），極可能寫入該處。

**對 plan 的影響：**
- Task 1 **執行**（補 Rationalizations 表 + Red Flags）— 針對「缺 `--game-data`」與「$GAME_DIR 幻想」的補強
- Task 3 **執行**（Phase 4 guard）— 明示 `--game-data` 必須指向 `analysis-data/<date>/<game>/merged.json`
- Commit message 採 Task 4 Step 5 (a) 版本
