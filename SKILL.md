---
name: mlb-game-analyzer
description: Use when the user asks for MLB single-game matchup analysis — pitcher / lineup / bullpen breakdown, pre-game data dossiers, BABIP / ERA-xERA risk reads.
---

# MLB Game Analyzer — 單場對決數據分析

## Overview

系統化的 MLB 單場對決資料分析 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，以投打、牛棚、條件三層因子產出綜合分析報告。

---

## When to Use

特定 MLB 比賽的數據分析 / 對戰組合解讀 / 先發投手對決 / 進階數據（xwOBA, FIP, Statcast）讀法。

**不適用**：整季預測、球員個人比較、賽後回顧、歷史統計查詢。

---

## Quick Reference

| 步驟 | 主要產出 | 工具 |
|------|---------|------|
| 1. 資料收集 | `merged.json` + `dossier.md` + `summary.md`（含 AI 填空 placeholder） | `prepare_game.py` |
| 2. 綜合分析 | 在 `summary.md` 補完所有 placeholder | AI 編輯 |

> Doubleheader：產出檔名帶 suffix → `dossier-G1.md` / `summary-G1.md` / `dossier-G2.md` / `summary-G2.md`。

---

## 初始化

### Python 指令偵測

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
```

### 日期語意

`--date` 與 folder 一律用 **ET 開打日**（與 MLB Stats API `officialDate` 對齊）。腳本內部不做時區換算。

### 輸出目錄規範

```bash
GAME_DIR=analysis-data/{ET-YYYY-MM-DD}/{AWAY}@{HOME}
# Doubleheader：{AWAY}@{HOME}-G1 / -G2
mkdir -p $GAME_DIR
```

### 工具使用規範

- ⛔ 禁止 WebFetch / WebSearch 收集核心數據
- ✅ 唯一例外：當日傷兵快訊（API 40 人名單 + IL 名單為主，WebSearch 補充）
- ⛔ 腳本失敗 → 向使用者回報，禁止靜默改走 WebSearch
- ⛔ 所有腳本輸出必須用 `--output / -o`，禁止 shell redirect `>`
- ⛔ 隊伍縮寫一律用英文縮寫（KC / LAA / NYY），純數字 team_id 已被各腳本拒絕

### 資料來源優先順序

API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

---

## 步驟 1：資料收集

```bash
$PYTHON scripts/prepare_game.py --date {ET-YYYY-MM-DD} --away {AWAY} --home {HOME}
# Doubleheader：加 --game-suffix G1 / G2
```

失敗 exit code 1/2/3/4/5/7（`1` = 子腳本找不到；其餘見 `prepare_game.py --help`）。

**後續動作**：
1. Read `$GAME_DIR/dossier.md`
2. Read `$GAME_DIR/summary.md` 與 `reference/matchup-factors.md`
3. 進入步驟 2：在 summary.md 上補完所有 `<!-- AI 補 -->` placeholder（直接在這個檔上改，不需另存）

ℹ️ 如需深入查驗某球員 / 投手細節，可主動 Read 同目錄下個別 drill-down 檔：
`away_pitcher_summary.md` / `home_pitcher_summary.md` / `away_lineup_summary.md` / `home_lineup_summary.md` / `away_roster_summary.md` / `home_roster_summary.md` / `game_data_summary.md` / `merged_summary.md`

---

## 步驟 2：綜合分析

> 假設 `reference/matchup-factors.md` 已於步驟 1 後續動作 #2 讀過，本段不再重複叫讀；遺漏時補讀。

### 2.1-2.4 順序執行

| 子步驟 | 分析內容 | 參考 |
|------|---------|------|
| 2.1 投打對決 | 投手 Tier + 打線評級 + Platoon + 球種 | `matchup-factors.md` |
| 2.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵影響度 | `matchup-factors.md` |
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場 | `matchup-factors.md` |
| 2.4 風險提示 | dossier 已標的 ⚠️（Flag 8 / Flag 3）AI 敘事判讀 | `flags-checklist.md` |

⛔ BvP 樣本 PA ≥ 15 才可引用（`flags-checklist.md` Flag 2）

### 2.5 完工條件

`$GAME_DIR/summary.md` 內所有 `<!-- AI 補 -->` placeholder 都已補完即為最終輸出。

**MUST contain**：投手 Tier 判斷、打線評級、牛棚影響判讀、風險提示判讀、條件修正、修正後預期得分、整體判斷（方向 / 總分 / 信心 / 風險 1-4 點）。

ℹ️ 重跑 `prepare_game.py` 預設不會覆蓋已編輯的 summary.md（偵測 placeholder 是否還在）；要強制重產用 `--force`。

---

## Common Pitfalls

紀律違規條目：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、Doubleheader、TJ 復出等）：見 `reference/matchup-factors.md`。

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協
- Score override 政策：嚴重 small_sample / era_xera gap 觸發時，依 `reference/flags-checklist.md` §8 走嚴格 formula 預測等實際結果比對，不主動 override。使用者明確要求 override 時才走 override 路徑並記錄理由
