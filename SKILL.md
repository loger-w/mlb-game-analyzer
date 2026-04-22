---
name: mlb-game-analyzer
description: Use when the user asks about MLB game predictions, matchup analysis, betting lines, score predictions, pitcher duels, or "who will win" questions for any specific MLB game — including queries like "analyze today's Yankees game" or "Dodgers vs Padres"
---

# MLB Game Analyzer — 單場對決分析與比分預測

## Overview

系統化的 MLB 單場對決分析流程 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，經過投打、牛棚、環境、盤口四層修正後，輸出勝率與比分預測。

---

## When to Use

- 使用者詢問特定 MLB 比賽的勝負預測或分析
- 使用者提供對戰組合（如「洋基打道奇」）想知道誰會贏
- 使用者需要盤口推薦（ML / O/U / Run Line）
- 使用者詢問先發投手對決分析
- 使用者想了解特定比賽的進階數據分析

**不適用：**
- 整季預測或球隊排名預測（非單場分析）
- 球員個人表現比較（無特定比賽）
- 賽後回顧（使用 `mlb-post-game-review`）
- 歷史對戰統計查詢（無即時比賽）

---

## Quick Reference

| Phase | 關鍵活動 | 閘門條件 | 輸出 |
|-------|---------|---------|------|
| 1. 資料收集 | `fetch_game_data.py` | 比賽存在且為例行賽 | `game_data.json` |
| 2. 投打驗證 | roster + pitcher + lineup + merge | Step 1 roster 閘門 + Step 2 role_change 檢查 | `merged.json` |
| 3. 綜合分析 | 投打對決 + 牛棚 + 條件修正 | BvP PA≥15 + 牛棚雙向閘門 + BABIP 回歸 | `phase3_summary.md` |
| 4. 預測輸出 | `predict.py` + 紀律閘門 D1-D5 | 同場無對立 + 星級護欄 + 一致性檢查 | `prediction.json` + 報告 |

> 詳細執行命令、參數、逐項 checklist：`reference/workflow.md`

---

## The Iron Law

```
NO PREDICTION OUTPUT WITHOUT ALL PHASE GATES PASSED IN SEQUENCE
```

Phase 1 → Phase 2 → Phase 3 → Phase 4，每個 Phase 的閘門未通過就不能進入下一個。

**違反規則的字面意義就是違反規則的精神。**

---

## Rationalizations — 藉口 vs 現實

下表把分析壓力下最常出現的內心獨白配上它的反駁。**Red Flags 抓行為，此表抓想法**。

| 心裡想的 | 實際上 |
|---------|--------|
| 「腳本失敗，但我記得這個投手數據大概…」 | 記憶與訓練資料不是來源。回報錯誤等使用者指示，禁止靜默改走 WebSearch 或記憶。 |
| 「BvP 12 PA 趨勢看起來很明顯，可以引用。」 | PA<15 是雜訊，不是訊號。硬推結論 = 無根據推薦。 |
| 「近 7 天 Hot 很明顯，BABIP 就不用查了吧。」 | 近 7 天 .400 BABIP 回歸後跟 .280 差不多。未檢查 = Hot/Cold 判定無效。 |
| 「牛棚核心 IL 了，O/U 修一下就好。」 | 牛棚雙向閘門：同隊 ML 也必須下修。只修一側 = Phase 3 未完成。 |
| 「同場 ML 推 A，A 的受讓也推，兩邊都有理由。」 | D3 硬規則互斥。訊號方向 = 勝率強度，不是「兩邊下注對沖」。 |
| 「phase3_summary 就是 checklist，腦袋記得就好。」 | 對話壓縮後結論就沒了。Phase 4 沒 anchor = 用記憶預測。 |
| 「Roster 看起來沒變動，跳過 Step 1 直接分析。」 | Step 1 是阻塞閘門。IL 遺漏 → Phase 3 牛棚傷兵基礎就錯。 |
| 「Agent 子代理平行跑 WebSearch 比較快。」 | 子代理沒 WebSearch 權限，輸出是幻想。主對話平行跑才對。 |
| 「SKILL.md 只寫 `predict.py --save`，`--game-data` 或 `$GAME_DIR` 就用記憶中的吧。」 | `$GAME_DIR` 是 reference/workflow.md 的 shell 變數，未載入就等同未定義。唯一合法命令是 `predict.py --game-data analysis-data/<date>/<AWAY>@<HOME>/merged.json --save`，`prediction.json` 自動落在同層。省略或腦補 = predict.py 報錯或寫錯位置。 |

**以上任何一項 = 停下來，回到對應 Phase 的閘門檢查清單。**

---

## Red Flags — 停下來，回到流程

如果你發現自己正在：
- **使用訓練資料或記憶的投手/球員數據**（所有核心數據必須來自腳本 API 輸出，禁止臆測或幻想）
- **在用戶說中文的對話裡用英文輸出報告**（搜尋優先英文，輸出照用戶語言；中文 → 繁體中文）
- 用記憶中的數據代替腳本輸出
- WebSearch 失敗後「差不多就好」繼續分析
- 跳過 Roster 檢查因為「應該沒問題」
- 牛棚傷兵只修了 O/U 就急著往下走
- BvP 樣本不足卻引用結論因為「看起來有趨勢」
- 沒寫 `phase3_summary.md` 就開始 Phase 4
- 同場推了對立方向因為「兩邊都有理由」
- 用 shell redirect `>` 因為「比較快」
- 用 Agent 子代理跑 WebSearch
- 下 `predict.py` 時省略 `--game-data`，或把 `prediction.json` 寫到 `analysis-data/<date>/<AWAY>@<HOME>/` 以外位置（不論 `games/` 是否存在，均禁止寫入）

**以上任何一項 = 停下來，回到對應的 Phase 閘門。**

---

## 初始化

**每次對話開始時執行一次**：Python 指令偵測、`$GAME_DIR` 設定、`scripts/**/*.py` Glob 偵測。

- 腳本偵測成功 → 切換 **🐍 腳本模式**（WebSearch 僅限傷兵快訊一類例外；傷兵優先用 API 40 人 + IL 名單）
- 腳本偵測失敗 → **禁止自動改用 WebSearch**，先詢問使用者腳本路徑

> 完整初始化步驟（含 bash 指令、模式切換規範）：`reference/workflow.md`「初始化」章節

---

## Phase 2：投打驗證與資料擴充

- **Step 1（🔒 阻塞）**：先發必須在 active roster + IL 已記錄，未通過不得進 Step 2
- **Step 2 閘門**：`role_change` 處理；`|ERA−xERA| ≥ 1.5` 或 `IP<30 且 ERA 比 prior year 低 ≥1.0` → 必須補跑 `pitcher_stats.py --year {YYYY-1}` 做 YoY Statcast 對比

→ 詳細：`reference/workflow.md#phase-2投打驗證與資料擴充`

---

## Phase 3：綜合分析

依序執行投打對決 → 牛棚 → 條件修正 → 近期狀態，寫入 `phase3_summary.md`。

⛔ **MUST NOT**：`phase3_summary.md` 不得含 ML / O/U / Run Line 星級，不得含「初步盤口推薦」— 盤口推薦的 single source of truth 是 Phase 4 `prediction.json`。

→ 詳細（含 BvP/牛棚/BABIP 閘門）：`reference/workflow.md#phase-3綜合分析`

---

## Phase 4：預測輸出

- **執行**：`predict.py --save`（自動寫 `prediction.json` 到 `$GAME_DIR`）
- **紀律 D1-D5**：D1 模型覆蓋（α 實作：ml_lean vs formula_lean）/ D2 信號修正 / D3 同場互斥 / D5 比分一致性 — 由 `predict.py` guardrail 自動執行，完整條文見 `reference/prediction.md`「分析紀律」
- **賽後彙總 / 回填**：轉交 `mlb-post-game-review`

> ⚠️ 寫 `prediction.json` 前確認 `--game-data` 指向 `analysis-data/<date>/<AWAY>@<HOME>/merged.json`；不對就停下來重新定位，不得自建替代目錄。

→ 詳細（`--save` 參數表、輸出前驗證清單、輸出格式）：`reference/workflow.md#phase-4預測輸出`

---

## Common Pitfalls & Edge Cases

最高優先 3 項技術漏洞（與 Red Flags 不重疊；完整清單見 `reference/pitfalls.md`）：

1. **Hot/Cold 判定未查 BABIP**
   近 7 天 BABIP 極端值（≤ .260 或 ≥ .370）預期回歸 ~.300，未檢查 = Hot/Cold 判定無效。

2. **ERA vs xERA 落差 ≥ 1.5 僅寫成「風險提示」**
   可驗證的現象不得掛成條件性風險。必須補跑 `pitcher_stats.py --year {YYYY-1}` + YoY Statcast 對比。

3. **Phase 3 summary 寫入「初步盤口推薦」或星級**
   盤口推薦 single source = Phase 4 `prediction.json`。Summary 只放基本面，避免 stale。

→ 完整清單（13 項 Common Mistakes + 12 項 Edge Cases）：`reference/pitfalls.md`

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據

---

