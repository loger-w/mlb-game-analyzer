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

## 初始化

**每次對話開始時執行一次**：Python 指令偵測、`$GAME_DIR` 設定、`scripts/**/*.py` Glob 偵測。

- 腳本偵測成功 → 切換 **🐍 腳本模式**（WebSearch 僅限傷兵快訊一類例外；傷兵優先用 API 40 人 + IL 名單）
- 腳本偵測失敗 → **禁止自動改用 WebSearch**，先詢問使用者腳本路徑

> 完整初始化步驟（含 bash 指令、模式切換規範）：`reference/workflow.md`「初始化」章節

---

## Phase 2：投打驗證與資料擴充

- **Step 1（🔒 阻塞）**：先發必須在 active roster + IL 已記錄，未通過不得進 Step 2
- **Step 2 閘門**：`role_change` 處理；ERA-xERA / IP 落差閘門 → 詳見 `reference/workflow.md` §Phase 2 Step 2

→ 詳細：`reference/workflow.md#phase-2投打驗證與資料擴充`

---

## Phase 3：綜合分析

依序執行投打對決 → 牛棚 → 條件修正 → 近期狀態，寫入 `phase3_summary.md`。

⛔ **MUST NOT**：`phase3_summary.md` 不得含 ML / O/U / Run Line 星級，不得含「初步盤口推薦」— 盤口推薦的 single source of truth 是 Phase 4 `prediction.json`。

→ 詳細（含 BvP/牛棚/BABIP 閘門）：`reference/workflow.md#phase-3綜合分析`

---

## Phase 4：預測輸出

- **執行**：`predict.py --save`（自動寫 `prediction.json` 到 `$GAME_DIR`）
- **紀律 D1-D5**：D1 formula 方向 / D2 信號修正 / D3 同場互斥 / D5 比分一致性 — 由 `predict.py` guardrail 自動執行，完整條文見 `reference/prediction.md`「分析紀律」
- **賽後彙總 / 回填**：轉交 `mlb-post-game-review`

> ⚠️ 寫 `prediction.json` 前確認 `--game-data` 指向 `analysis-data/<date>/<AWAY>@<HOME>/merged.json`；不對就停下來重新定位，不得自建替代目錄。

→ 詳細（`--save` 參數表、輸出前驗證清單、輸出格式）：`reference/workflow.md#phase-4預測輸出`

---

## Common Pitfalls

紀律違規 13 條 + 觸發處理：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、Doubleheader、TJ 復出等）：見 `reference/matchup-factors.md` 與 `prediction.md`。

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協

---

