---
name: mlb-game-analyzer
description: Use when the user asks about MLB game predictions, matchup analysis, score predictions, pitcher duels, or "who will win" questions for any specific MLB game — including queries like "analyze today's Yankees game" or "Dodgers vs Padres"
---

# MLB Game Analyzer — 單場對決分析與比分預測

## Overview

系統化的 MLB 單場對決分析流程 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，經過投打、牛棚、環境三層修正後，輸出勝率與比分預測。

---

## When to Use

特定 MLB 比賽的勝負預測 / 對戰組合分析 / 推薦方向（ML / O/U / Run Line）/ 先發投手對決 / 進階數據解讀。

**不適用**：整季預測 / 球員個人比較 / 賽後回顧（轉 `mlb-post-game-review`）/ 歷史統計查詢。

---

## Quick Reference

| Phase | 主要產出 |
|-------|---------|
| 1. 資料收集 | `game_data.json` + `game_data_summary.md`（`fetch_game_data.py`，例行賽） |
| 2. 投打驗證 | `merged.json` + `merged_summary.md`（roster + pitcher + lineup + merge；Step 1 roster + Step 2 role_change 閘門） |
|  | 各腳本同時產出 `*_summary.md`（含 🚨 Trigger section：Flag 13 / Flag 3 自動偵測） |
| 3. 綜合分析 | `phase3_summary.md`（投打 / 牛棚 / 條件修正；BvP PA≥15、牛棚雙向、BABIP 回歸閘門） |
| 4. 預測輸出 | `prediction.json` + `prediction_summary.md`（`predict.py`；紀律 D1-D5 自動執行） |

> 命令、參數、checklist：`reference/workflow.md`

---

## The Iron Law

```
NO PREDICTION OUTPUT WITHOUT ALL PHASE GATES PASSED IN SEQUENCE
```

Phase 1 → Phase 2 → Phase 3 → Phase 4，閘門未通過不得進下一階段。

---

## 初始化

對話開始執行一次：偵測 Python 指令、設定 `$GAME_DIR`、Glob `scripts/**/*.py`。腳本偵測成功進 **🐍 腳本模式**（核心數據禁 WebSearch，傷兵快訊例外）；偵測失敗詢問使用者，禁止改走 WebSearch。

> 細節：`reference/workflow.md`「初始化」章節

---

## Phase 重點

- **Phase 2**：Step 1 roster（🔒 阻塞）→ Step 2 投手 / 打線 + role_change + ERA-xERA 落差閘門
- **Phase 3**：投打對決 → 牛棚 → 條件修正 → 近期狀態，寫入 `phase3_summary.md`
  ⛔ summary 不得含星級或盤口推薦（single source of truth = Phase 4 `prediction.json`）
- **Phase 4**：`predict.py --save`；紀律 D1-D5 由 guardrail 自動執行；賽後回填轉 `mlb-post-game-review`
  ⚠️ `--game-data` 必須指向 `analysis-data/<date>/<AWAY>@<HOME>/merged.json`，不得自建替代目錄

→ 詳細執行步驟、閘門檢查、CLI 參數：`reference/workflow.md`；紀律完整條文：`reference/prediction.md`

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

