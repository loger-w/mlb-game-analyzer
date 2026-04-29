---
name: mlb-game-analyzer
description: Use when the user asks about MLB game predictions, matchup analysis, score predictions, pitcher duels, or "who will win" questions for any specific MLB game — including queries like "analyze today's Yankees game" or "Dodgers vs Padres"
---

# MLB Game Analyzer — 單場對決分析與比分預測

## Overview

系統化的 MLB 單場對決分析流程 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，經過投打、牛棚、環境三層修正後，輸出勝率與比分預測。

**Phase 1+2 已整合為單一 `prepare_game.py`**，AI 唯一需要 Read 的整合檔為 `dossier.md` + `phase3_skeleton.md`。

---

## When to Use

特定 MLB 比賽的勝負預測 / 對戰組合分析 / 推薦方向（ML / O/U / Run Line）/ 先發投手對決 / 進階數據解讀。

**不適用**：整季預測 / 球員個人比較 / 賽後回顧（轉 `mlb-post-game-review`）/ 歷史統計查詢。

---

## Quick Reference

| Phase | 主要產出 | 工具 |
|-------|---------|------|
| 1+2. 資料收集 | `merged.json` + `dossier.md` + `phase3_skeleton.md` | `prepare_game.py` |
| 3. 綜合分析 | `phase3_summary.md`（在 skeleton 上補結論） | AI 編輯 |
| 4. 預測輸出 | `prediction.json` + `prediction_summary.md` | `predict.py --save` |

---

## The Iron Law

```
NO PREDICTION OUTPUT WITHOUT ALL PHASE GATES PASSED IN SEQUENCE
```

Phase 1+2 → Phase 3 → Phase 4，閘門未通過不得進下一階段。

---

## 初始化（每次對話一次）

### Python 指令偵測

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
```

### 輸出目錄規範

```bash
GAME_DIR=analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}
# Doubleheader：{AWAY}@{HOME}-G1 / -G2
mkdir -p $GAME_DIR
```

### 模式切換規範（🐍 腳本模式）

- ⛔ 禁止 WebFetch / WebSearch 收集核心數據
- ✅ 唯一例外：當日傷兵快訊（API 40 人名單 + IL 名單為主，WebSearch 補充）
- ⛔ 腳本失敗 → 向使用者回報，禁止靜默改走 WebSearch
- ⛔ 所有腳本輸出必須用 `--output / -o`，禁止 shell redirect `>`
- ⛔ 隊伍縮寫一律用英文縮寫（KC / LAA / NYY），純數字 team_id 已被各腳本拒絕

### 資料來源優先順序

API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

---

## Phase 1+2：資料收集（單一命令）

```bash
$PYTHON scripts/prepare_game.py --date {YYYY-MM-DD} --away {AWAY} --home {HOME}
# Doubleheader：加 --game-suffix G1 / G2
```

**閘門（自動執行）**：exit 0 = 全 phase 通過；非 0 = 各種 hard error（exit 2-7，見 prepare_game.py --help）。

**後續動作**：
1. Read `$GAME_DIR/dossier.md`（單一檔，~250 行）
2. Read `$GAME_DIR/phase3_skeleton.md` 與 `reference/matchup-factors.md` / `reference/prediction.md`
3. 在 phase3_skeleton.md 補結論段落，存檔為 `phase3_summary.md`
4. 進入 Phase 4

ℹ️ 如需深入查驗某球員 / 投手細節，可主動 Read 同目錄下個別 `*_summary.md`（drill-down）。

---

## Phase 3：綜合分析

> ⛔ **分析前**：Read `reference/matchup-factors.md`（投手 Tier、打線評級、牛棚傷兵修正、條件修正值）

### 3.1-3.4 順序執行

| 步驟 | 分析內容 | 參考 |
|------|---------|------|
| 3.1 投打對決 | 投手 Tier + 打線評級 + Platoon + 球種 | `matchup-factors.md` |
| 3.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵修正（雙向：O/U + ML） | `matchup-factors.md` |
| 3.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場 | `matchup-factors.md` |
| 3.4 風險提示 | dossier 已標的 ⚠️（Flag 13 / Flag 3）AI 敘事判讀 | `flags-checklist.md` |

⛔ BvP 樣本 PA ≥ 15 才可引用（`flags-checklist.md` Flag 2）

### 3.5 phase3_summary.md 存檔

⛔ Phase 3 完成、Phase 4 開始前，必須將 phase3_skeleton.md 的填空全部完成、另存為 `$GAME_DIR/phase3_summary.md`。

**MUST contain**：投手 Tier 判斷、打線評級、牛棚雙向修正值、風險提示判讀、條件修正、修正後預期得分、整體判斷。

⛔ **MUST NOT contain**：星級 / 明確盤口推薦（這些是 Phase 4 專屬）。

---

## Phase 4：預測輸出

> ⛔ **預測前**：Read `$GAME_DIR/phase3_summary.md` + `reference/prediction.md`（公式、信號表、星級門檻、紀律 D1-D5）

### 4.0 執行預測腳本

```bash
$PYTHON scripts/predict.py --game-data $GAME_DIR/merged.json --save [參數]
```

**`--save` 必填參數**：

| 參數 | 必填 | 說明 |
|------|------|------|
| `--ou-line` | 是 | 大小分線（如 9.5） |
| `--ou-rec` | 是 | OVER / UNDER / PASS |
| `--ou-stars` | OVER/UNDER 時必填 | 0-5（缺則 hard exit 6） |
| `--ml-rec` | 是 | 隊伍縮寫或 PASS |
| `--ml-stars` | 是 | 0-5 |
| `--adjusted-home` | 建議 | 分析後調整的主隊得分 |
| `--adjusted-away` | 建議 | 分析後調整的客隊得分 |
| `--signal-adjustments` | 建議 | JSON 格式，如 `'{"puk_il":0.3}'` |
| `--tags` | 建議 | 逗號分隔，如 `divergent,early-season` |
| `--temperature` / `--wind-mph` / `--wind-direction` / `--umpire` / `--umpire-ou-rate` | 若有 | 環境補充 |

> RL 推薦走 `predict.py` auto override（無 `--run-line-rec` / `--run-line-stars` CLI args）。

### 4.1-4.6 紀律 / 護欄 / 輸出

- PASS 門檻 + 星級護欄 → `prediction.md` PASS 章節
- D1-D5 紀律自動執行 → `prediction.md` 分析紀律
- predict.py --save 自動寫入 `$GAME_DIR/prediction.json` + `prediction_summary.md`

### 4.7 輸出前驗證

✅ Read `$GAME_DIR/prediction_summary.md`，逐項確認：

- [ ] D1 / D2 紀律通過？
- [ ] D3 同場無對立推薦？
- [ ] D5 比分與盤口一致性？
- [ ] 牛棚傷兵雙向反映（O/U + ML）？
- [ ] 星級護欄降級警告已確認？

### 4.8 輸出格式

完整 TL;DR + Section 8-10 模板已內化於 `prediction_summary.md`，AI 直接複製貼上。Section 1-7（基本面）由 AI 從 `dossier.md` / `phase3_summary.md` 補充。

---

## Common Pitfalls

紀律違規 12 條：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、Doubleheader、TJ 復出等）：見 `reference/matchup-factors.md`。

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協
