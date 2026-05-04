---
name: mlb-game-analyzer
description: Use when the user asks for MLB single-game matchup analysis — pitcher / lineup / bullpen breakdown, pre-game data dossiers, BABIP / ERA-xERA risk reads — or interpreting odds-report movement (ML / RL / O-U, no-vig pp delta, key-number flags) for that game.
---

# MLB Game Analyzer — 單場對決數據分析

## Overview

系統化的 MLB 單場對決資料分析 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，以投打、牛棚、條件三層因子產出綜合分析報告。

---

## When to Use

特定 MLB 比賽的數據分析 / 對戰組合解讀 / 先發投手對決 / 進階數據（xwOBA, FIP, Statcast）讀法。

**不適用**：整季預測、球員個人比較、歷史統計查詢、賽後回顧（final state 不分析，見 §場景路由 Step 4）、多場 batch 分析（單場 only）。

---

## 場景路由 (Routing)

> **每次請求都要先做這一段**。不該重跑已完成的基本面，不該越權用 odds 補基本面，不該越權用基本面跑盤口。

### Step 0：解析 intent

| Keywords | Intent |
|---|---|
| 分析 / 預測 / 看一下 / 先發 / 打線 / 牛棚 + 隊名 | `fundamentals_only` |
| 盤口 / ML / RL / O-U / O/U / 下哪邊 / 贏面 / value / line | `odds_only` |
| 兩種都有 | `both` |
| 都沒明確 | `ambiguous` → 預設 `fundamentals_only` |

### Step 1：確定 GAME_DIR

- date 沒給 → ET 今日
- doubleheader 多場 → 強制問 G1/G2
- matchup 多場（連戰 / 系列賽 / 「分析三連戰」）→ reject「目前只能分析單場比賽」，要求 user 指定單一場

### Step 2：平行 state probe（無腳本呼叫）

```
basic_state :=
  none     ← analysis-data/{date}/{matchup}/summary.md 不存在
  partial  ← summary.md 存在但內含 "<!-- AI 補"
  complete ← summary.md 存在且無 placeholder

odds_state :=
  no_report       ← odds/reports/{date}.md 不存在
  report_no_match ← report 存在但無此 matchup 段
  has_match       ← report 含此 matchup
```

順手讀 `game_data.json.gameState`（preview / live / final），用於後續 idempotence 判斷。

### Step 3：Routing 矩陣

| intent | basic_state | odds_state | 動作 |
|--------|-------------|------------|------|
| `fundamentals_only` | none | — | 跑步驟 1 + 步驟 2 |
| `fundamentals_only` | partial | — | **跳過 prepare_game**，直接補 summary.md 剩餘 placeholder |
| `fundamentals_only` | complete | — | Read summary.md 直接呈現，標 mtime；不主動 refresh |
| `odds_only` | * | no_report | 停步：「`odds/reports/{date}.md` 不存在」+ 建議等下個 snapshot |
| `odds_only` | * | report_no_match | 停步：「report 中無 {matchup}」+ 列該日含哪些場次 |
| `odds_only` | none / partial | has_match | **步驟 3：純讀 odds report**（不主動跑 fundamentals） |
| `odds_only` | complete | has_match | 步驟 3 + **被動引用既有 summary 當 fair-value 錨**（仍不重跑） |
| `both` | * | * | fundamentals_only 路徑 → odds_only 路徑 → 尾段 paired analysis |
| `ambiguous` | * | * | 預設 `fundamentals_only`；footer 提示「想看盤口可額外要求」 |

### Step 4：Idempotence 標註

- 重用既有 summary 時必須輸出明說「summary.md 已於 {mtime} 完成，重用」
- gameState = `live` → 告知並降級（live data ≠ pre-game data）
- gameState = `final` → 停步：本 skill 不提供賽後分析（user 想看結果可手動讀 boxscore）

### Force / refresh override

關鍵字 → `prepare_game.py --force`（覆蓋既有 summary）：
- "refresh"、"重跑"、"force"、"最新打線"、"再跑一次"

關鍵字 → 主動拉新 odds + 自動分析（user 觸發，每次 fetch ~12 credits）：
- "拉盤口"、"拉新 odds"、"最新 odds"、"refresh odds"、"odds 重抓"
- 流程：`python odds/fetch_odds.py` → `python odds/analyze_smart_money.py --date {ET-YYYY-MM-DD}` → 接 Step 3 odds_only 路徑主動分析該 matchup（不需 user 再追問）

---

## Quick Reference

| 步驟 | 主要產出 | 工具 | 觸發路徑 |
|------|---------|------|---------|
| 0. 場景路由 | intent + state 判斷 | 無（純 state probe） | 每次必跑 |
| 1. 資料收集 | `merged.json` + `dossier.md` + `summary.md`（含 AI 填空 placeholder）<br>**自動偵測**：official lineup（公布後）/ 天氣（公布後） | `prepare_game.py` | `fundamentals_only` / `both` 且 `basic_state ≠ complete` |
| 2. 綜合分析 | 在 `summary.md` 補完所有 placeholder | AI 編輯 | 同上 |
| 3. 盤口分析 | odds report 解讀（可選 paired with summary） | Read `odds/reports/{date}.md` | `odds_only` / `both` 且 `odds_state = has_match` |

---

## 初始化

### Python 指令偵測

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
```

### 日期語意

`--date` 與 folder 一律用 **ET 開打日**（與 MLB Stats API `officialDate` 對齊）。腳本內部不做時區換算。

### 輸出目錄 / 檔名規範

```bash
GAME_DIR=analysis-data/{ET-YYYY-MM-DD}/{AWAY}@{HOME}
# Doubleheader: -G1 / -G2 後綴
```

### 工具使用規範

- 核心紀律見 `reference/flags-checklist.md`（資料來源 / 輸出規範 / WebSearch 邊界 — Flag 1 / 4 / 5）
- 隊伍縮寫一律用英文（KC / LAA / NYY），純數字 team_id 已被各腳本拒絕

### 資料來源優先順序

API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

### 條件式資料（公布後才有）

| 資料 | 來源 | 缺資料行為 |
|------|------|-----------|
| 公布打線（battingOrder） | feed/live | fallback 至 PA proxy（lineup_source = "projected"） |
| 天氣（condition / temp / wind） | feed/live `gameData.weather` | summary 標「未公布（跳過天氣分析）」 |

**公布時機**：打線通常開賽前 2–4 小時、天氣前 1 小時 ~ 開賽後填齊。
**重跑取最新**：見 §場景路由 §Force / refresh override。

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
3. 進入步驟 2：在 summary.md 上補完所有 `<!-- AI 補 -->` placeholder

ℹ️ **drill-down 只在以下情境 Read**（dossier 已涵蓋核心欄位，預設不必看）：
- 要查 GB% / xBA / csw% / EV95% / 完整 pitch mix / Pitch Arsenal RV/100 → `<side>_pitcher_summary.md`
- 要看 9 人完整 table / Last 7 per-player / Platoon per-player / BvP table → `<side>_lineup_summary.md`
- 要驗 IL list 細節（status / position） → `<side>_roster_summary.md`
- 其他 (`game_data_summary.md` / `merged_summary.md`) 為 debug 用，正常分析不需 Read

ℹ️ **打線來源 / 天氣**：dossier 與 summary 都會標記。official 與 projected 分析架構相同，差異僅在 9 人組成是真實打序還是 PA 近似（見 `matchup-factors.md` §打線分析）。

---

## 步驟 2：綜合分析

> 假設 `reference/matchup-factors.md` 已於步驟 1 後續動作 #2 讀過，本段不再重複叫讀；遺漏時補讀。

### 2.1-2.4 順序執行

| 子步驟 | 分析內容 | 參考 |
|------|---------|------|
| 2.1 投打對決 | 投手 Tier + 打線評級 + Platoon + 球種 | `matchup-factors.md` |
| 2.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵影響度 | `matchup-factors.md` |
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場/**天氣** | `matchup-factors.md` §天氣修正 |
| 2.4 風險提示 | dossier 已標的 ⚠️（Flag 8 / Flag 3）AI 敘事判讀 | `flags-checklist.md` |

⛔ BvP 樣本 PA ≥ 15 才可引用（`flags-checklist.md` Flag 2）

### 2.5 完工條件

`$GAME_DIR/summary.md` 內所有 `<!-- AI 補 -->` placeholder 都已補完即為最終輸出。

**MUST contain**：投手 Tier 判斷、打線評級、牛棚影響判讀、風險提示判讀、條件修正、修正後預期得分、整體判斷（方向 / 總分 / 信心 (%) / 風險 1-4 點）。

---

## 步驟 3：盤口分析

> **觸發**：intent ∈ {`odds_only`, `both`} 且 `odds_state = has_match`。
> 假設 `odds/reports/{date}.md` 由 odds 模組 cron 產出（Pinnacle / The Odds API）。

### 3.1 找出本場條目

`odds/reports/{date}.md` 結構：tier 分組（🔥 Major ≥ 5pp / 🟡 Significant ≥ 3pp / 🔵 Watch ≥ 1pp / ⚪ Quiet < 1pp）→ Anchor Notes → 解讀說明。

用 Grep tool 搜 pattern `{Away} @ {Home}` on `odds/reports/{date}.md`，讀取：
- `direction_label`（→ TEAM ±Xpp，no-vig latest vs anchor 差）
- 時間軸 table（snapshots × ML / RL / Over / Under）
- Flags（位移 + 薄盤 + key number 跨越）

### 3.2 解讀架構

| 維度 | 判讀 |
|------|------|
| Tier | 🔥/🟡 = strong move 必看；🔵 watch；⚪ noise（不必引用） |
| 方向 | direction_label 顯示 market 偏向；對照 anchor 看 sharp 還是穩定 |
| 薄盤 | latest 距開球 < 4h → 訊號可能被晚場閉盤動作污染，可信度降一檔 |
| Key number | Total 跨 7 / 9 / 11 標 ⚠️ — 1.0 run 跨 key 比 0.5 跨非 key 重要 |
| 雙邊 vs 單邊 | no-vig pp-delta 區分莊家整體 vig 調整（雙邊同向）vs 真 sharp money（單邊位移） |

> 解讀說明的完整定義在 `odds/reports/{date}.md` 文末「## 解讀說明 (給 AI)」段。

### 3.3 Paired analysis（only if `basic_state = complete`）

- 比對 summary.md 的 direction（基本面）vs odds report 的 direction（market）
- 同向 + market move 強 → confluence（雙重支持）
- 反向 → fundamental disagreement，AI 必須解釋 gap
- 計算 fundamental fair vs market price gap（如果 summary 有 adjusted runs，可粗算 implied ML 對照）

### 3.4 完工條件 / 紀律

✅ MUST contain：tier 引用、direction、薄盤 flag（若有）、paired lean（若 basic complete）
⛔ MUST NOT contain：
- 自行補資料 / 無中生有 fair odds
- 「下哪邊」明確指令（給 lean + 信心，user 自行決策）
- Path A 風格的數字硬推（「市場 +EV 6.4%」之類無錨估計）

---

## Common Pitfalls

紀律違規條目：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、TJ 復出等）：見 `reference/matchup-factors.md`。

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協
- Score override 政策：嚴重 small_sample / era_xera gap 觸發時，依 `reference/flags-checklist.md` §8 讓 formula 當 sanity rail 對比實際結果，不主動 override。使用者明確要求 override 時才走 override 路徑並記錄理由
