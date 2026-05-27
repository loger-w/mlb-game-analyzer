---
name: mlb-game-analyzer
description: Use when the user asks for MLB single-game matchup analysis — pitcher / lineup / bullpen breakdown, pre-game data dossiers, BABIP / ERA-xERA risk reads — or interpreting odds-report movement (ML / RL / O-U, no-vig pp delta, key-number flags) for that game.
---

# MLB Game Analyzer — 單場對決數據分析

## Overview

系統化的 MLB 單場對決資料分析 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，以投打、牛棚、條件三層因子產出綜合分析報告。

執行步驟細節按 intent 拆到 `reference/workflow-*.md`：
- **fundamentals**（步驟 1+2）→ `reference/workflow-fundamentals.md`
- **odds**（步驟 3）→ `reference/workflow-odds.md`

---

## When to Use

特定 MLB 比賽的數據分析 / 對戰組合解讀 / 先發投手對決 / 進階數據（xwOBA, FIP, Statcast）讀法。

**不適用**：整季預測、球員個人比較、歷史統計查詢、賽後回顧（final state 不分析，見 §場景路由 Step 4）、多場 batch 分析（單場 only）。

---

## 場景路由 (Routing)

### Step 0a：建立 ET_NOW（**必跑、1 次 tool call、不可省略**）

⚠️ **不得信賴 system 注入的 `currentDate`**（可能 stale / 算錯時區 / 跨日 boundary 漂移）。所有時序判斷與相對日期解析以下指令輸出為單一事實來源：

```bash
$PYTHON -c "from datetime import datetime; from zoneinfo import ZoneInfo; n=datetime.now(ZoneInfo('America/New_York')); print(n.strftime('%Y-%m-%d %H:%M %Z'))"
```

把輸出記為 `ET_NOW`（精度到分）。

**用途**：
- 解析「今天 / 今晚 / 明天 / 昨天」等相對日期 → 用 `ET_NOW.date()`
- 當 user 給絕對日期（例「5/6」）且 system context 顯示不同日期時，**以 ET_NOW 為準**，不要自行推論「user 說的是過去 / 未來」
- 推導 `gameState`（見 Step 2）

**矛盾偵測**：若 user 給的日期比 `ET_NOW.date()` 早 ≥ 1 天 → 多半是賽後查詢（Step 4 處理）；晚 ≤ 7 天 → preview；其他要回頭問 user 確認（避免 typo）。

### Step 0b：解析 intent

| Keywords | Intent |
|---|---|
| 分析 / 預測 / 看一下 / 先發 / 打線 / 牛棚 + 隊名 | `fundamentals_only` |
| 盤口 / ML / RL / O-U / O/U / 下哪邊 / 贏面 / value / line | `odds_only` |
| 兩種都有 | `both` |
| 都沒明確 | `ambiguous` → 預設 `fundamentals_only` |

### Step 1：確定 GAME_DIR

- date 沒給 → 用 `ET_NOW.date()`（**不要用 system `currentDate`**）
- 相對日期（今天 / 今晚 / 明天 / 昨天）→ 用 `ET_NOW.date()` ± 偏移
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

**gameState 推導**（兩條路擇一，**不得跳過**）：
- `basic_state ≠ none` → 讀 `game_data.json.gameState`（authoritative：preview / live / final）
- `basic_state = none` → 比對 `ET_NOW`（Step 0a）vs **比賽開球時間**：
  - 開球時間來源優先序：odds report 段標題「開球 YYYY-MM-DD HH:MM ET」 > MLB Stats API
  - `ET_NOW < 開球` → `preview`
  - `開球 ≤ ET_NOW < 開球 + 4h` → `live`
  - `ET_NOW ≥ 開球 + 4h` → 推測 `final`（仍可跑 prepare_game 驗證；該腳本會回傳真實 gameState）

⚠️ **禁止**：在沒有 ET_NOW 與開球時間比對下，僅依 user 給的「日期 vs system currentDate」來推論 gameState。這是常見故障模式（user 說「5/6」+ stale currentDate 顯示「5/7」→ 錯誤推論為 final）。

### Step 3：Routing 矩陣

| intent | basic_state | odds_state | 動作 | 載入 workflow |
|--------|-------------|------------|------|--------------|
| `fundamentals_only` | none | — | 跑步驟 1 + 步驟 2 | `reference/workflow-fundamentals.md` |
| `fundamentals_only` | partial | — | **跳過 prepare_game**，直接補 summary.md 剩餘 placeholder | `reference/workflow-fundamentals.md`（§步驟 2） |
| `fundamentals_only` | complete | — | Read summary.md 直接呈現，標 mtime；不主動 refresh | （無 workflow，直接讀 summary.md） |
| `odds_only` | * | no_report | 停步：「`odds/reports/{date}.md` 不存在」+ 建議等下個 snapshot | （無 workflow） |
| `odds_only` | * | report_no_match | 停步：「report 中無 {matchup}」+ 列該日含哪些場次 | （無 workflow） |
| `odds_only` | none / partial | has_match | **步驟 3：純讀 odds report**（不主動跑 fundamentals） | `reference/workflow-odds.md` |
| `odds_only` | complete | has_match | 步驟 3 + **被動引用既有 summary 當 fair-value 錨**（仍不重跑） | `reference/workflow-odds.md` |
| `both` | * | * | fundamentals_only 路徑 → odds_only 路徑 → 尾段 paired analysis | 兩份都載入 |
| `ambiguous` | * | * | 預設 `fundamentals_only`；footer 提示「想看盤口可額外要求」 | `reference/workflow-fundamentals.md` |

### Step 4：Idempotence 標註

> 前置要求：`gameState` 必須來自 Step 2 的兩條路徑之一（讀 game_data.json 或 ET_NOW vs 開球時間比對）。**不得從 system currentDate 推論。**

- 重用既有 summary 時必須輸出明說「summary.md 已於 {mtime} 完成，重用」
- gameState = `live` → 告知並降級（live data ≠ pre-game data）
- gameState = `final` → 停步：本 skill 不提供賽後分析（user 想看結果可手動讀 boxscore）

**故障模式自檢**：若你想宣告 `gameState = final`，先回頭確認 ET_NOW 已建立（Step 0a 的 tool call 跑過）。若 user 給的日期與 system 注入的 currentDate 不一致 → 信 ET_NOW、不信 currentDate。

### Force / refresh override

關鍵字 → `prepare_game.py --force`（覆蓋既有 summary）：
- "refresh"、"重跑"、"force"、"最新打線"、"再跑一次"

關鍵字 → 主動拉新 odds + 自動分析：
- "拉盤口"、"拉新 odds"、"最新 odds"、"refresh odds"、"odds 重抓"
- 流程：`python odds/fetch_odds.py` → `python odds/analyze_smart_money.py --date {ET-YYYY-MM-DD}` → 接 Step 3 odds_only 路徑主動分析該 matchup

---

## Quick Reference

| 階段 | 主要產出 | 工具 / Workflow |
|------|---------|-----------------|
| 0. 場景路由 | ET_NOW + intent + state 判斷 | 1 次 Python tool call (ET_NOW) + state probe |
| 1+2. 基本面 | `merged.json` + `dossier.md` + `summary.md`（AI 補完） | `prepare_game.py` + `reference/workflow-fundamentals.md` |
| 3. 盤口 | odds report 解讀（可選 paired with summary） | `reference/workflow-odds.md` |

---

## 📊 回測校準提示（讀報告時參考）

基於 2026-05 回測 n=114（純基本面 vs Pinnacle 12:00 ET no-vig vs 實際結果）。

### 信心檔位 → 真實命中率
| 信心 | 真實命中率 | n | 建議 |
|---|---|---|---|
| HIGH | **80%** | 20 | 唯一強訊號，可下注參考 |
| MEDIUM | 51% | 70 | ~coin flip，只看不碰 |
| LOW | 48% | 21 | ~coin flip，只看不碰 |

註：MEDIUM 桶門檻已於 5 月校準後收緊（須 adjusted total gap ≥ 0.8 run 且無反向信號），未來 MEDIUM 場應減少但命中率應拉高；新 baseline 待 6 月驗證。

### 內部已自動校正（formula 已 bake-in，不用手動調）
- HOME +0.3 run（fix HOME 預測不足，原本 HOME 命中 62% vs AWAY 52%）
- Total −1.0 run（fix over-estimate，原本 bias +0.96）

### 觀察中（樣本不足下定論）
- skill 反市場時命中 47%（n=38）— 跟 Pinnacle 收盤線意見不合且非 HIGH 信心時，建議使用者 fade skill；待 6 月底回測驗證後決定是否寫成 hard rule

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
- 隊伍縮寫一律用英文（KC / LAA / NYY）

### 資料來源優先順序

API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

---

## 語氣與風格

- Score override 政策：嚴重 small_sample / era_xera gap 觸發時，依 `reference/flags-checklist.md` §8 讓 formula 當 sanity rail；使用者明確要求才走 override 並記錄理由
