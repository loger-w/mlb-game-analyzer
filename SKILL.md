---
name: mlb-game-analyzer
description: Use when the user asks about MLB game predictions, matchup analysis, betting lines, score predictions, pitcher duels, or "who will win" questions for any specific MLB game — including queries like "analyze today's Yankees game" or "Dodgers vs Padres"
---

# MLB Game Analyzer — 單場對決分析與比分預測

分析 MLB 單場對決，使用進階數據（Statcast / FanGraphs）、投打分析、場地天氣、
牛棚消耗、傷兵評估、賽季階段修正，輸出勝率與比分預測。

> 所有搜尋優先英文。若用戶使用中文，以**繁體中文**輸出。
> 請一定要使用真實存在的資料，切記不要臆測或幻想，也不要使用過舊的訓練資料集。

---

## 分析紀律（全域適用）

> 違反任何一條紀律 = 分析失敗，必須重新執行。

### 紀律 1：模型覆蓋紀律

ML + Log5 方向一致時（CONSISTENT），**不得因軟性因素翻轉勝方**（Platoon 劣勢、連勝動能、H2H 等）。

- 可調整：勝率幅度 ±5%、信心降級、星級降級
- 可覆蓋：DIVERGENT、模型未計入的重大因素（先發臨時更換等）、用戶明確要求
- **原則**：模型方向 > 直覺。軟性因素影響幅度，不影響方向。

### 紀律 2：分析獨立性

使用者質疑時，**不得直接翻轉結論**，必須先用 ML + 投手分析獨立驗證。
> Why：直接翻轉會導致推薦方向與數據矛盾。

1. 質疑 → 獨立驗證 → 驗證後才決定是否修改
2. 驗證發現使用者正確 → 修改並解釋驗證過程
3. 驗證發現原始正確 → 堅持並向使用者解釋
4. 存在歧義 → 列出不同解讀的結果

| 紅旗念頭 | 正確做法 |
|---------|---------|
| 「使用者說的應該對」 | 用 ML 獨立驗證 |
| 「我先改了再說」 | 先驗證，再改 |
| 「使用者不高興了」 | 分析師的價值在於獨立判斷 |

### 紀律 3：信號修正紀律

信號因子必須量化為 **Run Value 修正值**，不得獨立給 O/U 方向。

- 修正後總分 > O/U line → Over
- 修正後總分 < O/U line → Under
- 差距 < 1.5 run → 不推薦（SD ≈ 4.5 run）
- **不允許「信號說 Over 但比分說 Under」的矛盾。**

### 紀律 0：腳本執行紀律（🐍 模式強制適用）

> ⛔ **Python 腳本模式下，禁止使用 WebFetch / WebSearch 收集比賽核心數據。**
> 違反 = 分析失敗，必須從 Phase 1 腳本重新執行。

**允許使用 WebSearch 的唯一例外（僅限以下三類）：**
1. 天氣預報（溫度、風速、降雨機率）
2. 主審分配（umpire assignment）
3. 當日最新傷兵快訊（IL 異動）

**⛔ 每個 Phase 的腳本執行 Checkpoint：**
- Phase 1：必須先執行 `fetch_game_data.py`，確認有 JSON 輸出後才能繼續
- Phase 2：必須先執行 `pitcher_stats.py`（×2）和 `lineup_analyzer.py`（×2）
- Phase 4：必須先執行 `merge_game_data.py` → `predict.py`，取得模型輸出
- 若任何腳本失敗 → 向使用者回報錯誤，**禁止靜默改用 WebSearch**

| 紅旗念頭 | 正確做法 |
|---------|---------|
| 「直接 WebSearch 比較快」 | ⛔ 禁止，先跑腳本 |
| 「腳本可能沒裝好」 | 跑看看，失敗再回報，不得預設失敗 |
| 「WebFetch 一樣可以拿到資料」 | ⛔ 禁止，腳本有 XGBoost 模型，WebFetch 沒有 |
| 「先查一下資料再決定」 | ⛔ 禁止，查資料就是腳本的職責 |

---

## 初始化：腳本偵測

腳本固定在**本 SKILL.md 同層**的 `scripts/` 資料夾內。依序嘗試以下方式，**任一成功即切換 🐍 模式，不得在第一步失敗後直接放棄**：

**步驟 1：Glob 遞迴搜尋（從 skill 根目錄）**
```
Glob(pattern="**/*.py")  ← 從 SKILL.md 所在目錄執行
```

**步驟 2：若步驟 1 回傳空結果，改用 Bash 搭配 ~ 確認**
```bash
ls ~/.claude/skills/mlb-game-analyzer/scripts/
```

**步驟 3：若前兩步皆失敗，使用 Bash find 廣域搜尋**
```bash
find ~/.claude/skills -name "fetch_game_data.py" 2>/dev/null
```

⛔ **三步皆失敗時，禁止自動切換 📡 模式，必須先詢問使用者：**
> 「找不到分析腳本，請確認 `scripts/` 目錄位置，或告知腳本實際路徑。」

| 條件 | 模式 |
|------|------|
| 任一步驟找到 `fetch_game_data.py` | 🐍 Python 腳本模式 |
| 三步皆無結果 | 詢問使用者，**禁止自動切換 📡 模式** |

🐍 模式下的主要指令：

> ⚙️ **Python 指令偵測（每次對話開始時執行一次）：**
> ```bash
> PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
> ```
> 後續所有腳本呼叫皆使用 `$PYTHON`。Windows 通常只有 `python`；macOS/Linux 通常有 `python3`。

- `$PYTHON scripts/fetch_game_data.py --date {date} --team {team}` → Phase 1
- `$PYTHON scripts/pitcher_stats.py --name "{pitcher}" --year {year}` → Phase 2 A/B
- `$PYTHON scripts/lineup_analyzer.py --team {team} --year {year}` → Phase 2 D
- `$PYTHON scripts/predict.py --game-data merged.json --save [分析後參數]` → Phase 4
- 盤口需手動輸入 → `$PYTHON scripts/odds_analyzer.py --home-ml {ml} ...`

**predict.py --save 分析後參數（Phase 4 完成後必須傳入）**：

| 參數 | 必填 | 說明 |
|------|------|------|
| `--adjusted-home` | 建議 | 分析後調整的主隊得分 |
| `--adjusted-away` | 建議 | 分析後調整的客隊得分 |
| `--ou-line` | 是 | 有效大小分線（四分球取中位，如 9.75）|
| `--ou-rec` | 是 | OVER / UNDER / PASS |
| `--ml-rec` | 是 | 隊伍縮寫或 PASS |
| `--ml-stars` | 是 | 0-5 |
| `--run-line-rec` | 是 | 隊伍縮寫或 PASS |
| `--signal-adjustments` | 建議 | JSON 格式，如 `'{"puk_il":0.3}'` |
| `--tags` | 建議 | 逗號分隔，如 `divergent,early-season` |
| `--temperature` | 若有 | 氣溫 °F |
| `--wind-mph` | 若有 | 風速 mph |
| `--wind-direction` | 若有 | 風向 |
| `--umpire` | 若有 | 主審姓名 |
| `--umpire-ou-rate` | 若有 | 主審 Over% |

> 投手/打線腳本使用 MLB Stats API + Statcast（FanGraphs legacy API 已被封鎖）。

---

## Phase 1: 資料收集

> ⛔ **執行前必須 Read** `reference/teams-and-api.md`（隊名對照 + API 端點）

### 1.1 日期解析

將使用者輸入轉為 `YYYY-MM-DD`。
> ⚠️ MLB 賽程使用**美國時間**。亞洲時區使用者需確認。

### 1.2 API 呼叫（平行發送）

同時發送：
- **請求 A**：當日賽程 + 先發投手（schedule + hydrate=probablePitcher）
- **請求 B & C**：雙方近 10 場戰績（schedule + hydrate=linescore）

> 僅使用 `gameType = "R"` 例行賽，排除春訓。端點細節見 `reference/teams-and-api.md`。

### 1.3 Pythagorean Win% + 系列賽前場驗證

- 從近 10 場計算 Pythagorean Win%（公式見 `reference/teams-and-api.md`）
- 系列賽第 2+ 場 → 用 API 拉取前場實際比分（嚴禁用 WebSearch 摘要）

### 1.4 輸出確認

```
📅 {日期} — {客隊} @ {主隊}（{球場}）
⚾ 先發：{客隊投手} vs {主隊投手}
🕐 {比賽時間} | 狀態：{Preview/Live/Final}
📊 {主隊} 近 10 場：{W}-{L}（RS/G {X} | RA/G {Y}）（Pyth {Z}%）
📊 {客隊} 近 10 場：{W}-{L}（RS/G {X} | RA/G {Y}）（Pyth {Z}%）
```

- Doubleheader → 列出所有場次供選擇
- 無比賽 → 建議查前後日期
- 先發 TBD → Phase 2 搜尋確認
- Final → 詢問是否賽後分析

---

## Phase 2: 平行搜尋

> ⛔ **執行前必須 Read** `reference/matchup-factors.md`（投手分級 + 修正因子）
>
> ⚠️ **嚴禁使用 Agent 子代理執行搜尋**（子代理無法存取 WebSearch/WebFetch）。
> 必須在主對話中直接平行呼叫多個 WebSearch。

所有任務互相獨立，**單一訊息內平行發送 WebSearch**：

| 任務 | 內容 | 核心指標 |
|------|------|---------|
| A & B | 雙方先發投手進階數據 | ERA/FIP/xERA/K-BB%/Hard Hit%/球種 |
| C | 傷兵名單（雙方） | IL / DTD / 牛棚核心可用性 |
| D | 打線分析（雙方） | xwOBA/OPS/Platoon splits/BvP |
| E | 球場 & 天氣 | Park Factor / 溫度 / 風向 |
| F | 盤口賠率 | ML / Run Line / O/U + 讓分方向驗證 |
| G | 主審傾向（best effort） | O/U 紀錄 / K Boost |

> 指標定義、投手分級、牛棚累計、PF/天氣、傷病/TJ/角色轉換/年齡等細節見 `reference/matchup-factors.md`

---

## Phase 3: 綜合分析（順序執行）

> 後續修正依賴前面的基礎判斷，必須順序執行。

### 3.1 投打對決
- 先發投手 vs 對方打線 Platoon 優劣勢
- BvP 歷史（≥ 15 PA 才有參考價值）
- 球種對決

### 3.2 牛棚分析
- 整體品質 + 關鍵角色可用性 + 近 3 天消耗
- **牛棚傷兵累計效應**（1/2/3+ 名核心 → 非線性放大）
- 牛棚傷兵必須**同時反映在 ML 和 O/U**
- 替補品質反向檢查
- 細節見 `reference/matchup-factors.md`

### 3.3 條件修正（僅符合條件時觸發）
- **開季膨脹**：依投手先發場次遞減（×1.15 → ×1.00），見 `reference/prediction.md`
- **傷病/TJ/角色轉換/年齡退化**：見 `reference/matchup-factors.md`
- **賽季階段修正**：見 `reference/matchup-factors.md`
- **球場 & 天氣 & 主審**：見 `reference/matchup-factors.md`

### 3.4 近期狀態 & H2H
- 近 10 場得失分差、連勝/敗、主客場分拆
- BABIP 回歸檢查（見 `reference/matchup-factors.md`）
- H2H 歷史對戰

---

## Phase 4: 預測輸出

> ⛔ **執行前必須 Read** `reference/prediction.md`（公式 + 信號修正表 + 星級）

### 4.0 PASS 規則（任何推薦低於門檻 = 不推薦）

| 盤口類型 | PASS 門檻 |
|---------|----------|
| O/U | 修正後總分 vs line 差距 < 1.5 run |
| ML | 真實勝率 vs 隱含勝率差距 < 5% |
| Run Line -1.5 | P(cover) < breakeven |
| 受讓盤 +1.5 | ML 信心 < MEDIUM 時不推 |

**開季限定（前 2 週）**：Under 上限 ⭐⭐⭐ / 受讓上限 ⭐⭐⭐ / Run Line 上限 ⭐⭐

> ⚠️ **勝率必須用 `predict.py` 的 `ml_prediction.home_win_pct`（XGBoost 模型）。**
> **比分使用 `formula_prediction`**（total_model 訓練資料有結構性缺陷，比分不可靠）。
> 手動估算只能作為輔助驗算，不得作為最終數字。

### 4.1 比分計算
使用期望得分公式（打線 xwOBA × 投手 ERA × PF），套用 Run Value 修正

### 4.2 信號修正
觸發的環境信號轉為 run 加減 → 修正後總分

### 4.3 盤口推薦
- O/U：修正後總分 vs Line（差距制）
- ML：真實勝率 vs 隱含勝率
- Run Line -1.5：P(cover) 計算（見 `reference/prediction.md`）
- Spread：**讓分方向交叉驗證**（必須執行）

### 4.4 硬性規則（Phase 4 閘門）

**禁止同場對立方向推薦**：
- ❌ A 隊 ML + B 隊 +1.5（矛盾：認為 A 贏但 B 輸不多）
- ML ≥ 60% + 信心高 → 可推 ML，**不推對方受讓**
- ML 55-60% → 二選一，不可同時推
- ML < 55% → 不推 ML，可考慮受讓或 PASS

**受讓盤偏見防護（開季前 2 週）**：
- ML ≥ -250 大熱門：受讓最多 ⭐⭐
- 「投手差距 = 接近比賽」假設在開季無效
- 首選不推讓分盤，次選推讓分方（非受讓）

**比分與盤口一致性**：
- 修正後總分 ≤ O/U line → 不得推 Over
- 修正後總分 ≥ O/U line → 不得推 Under

### 4.5 比賽敘事
根據量化觸發條件選擇劇本（投手戰 / 打線互爆 / 單方碾壓 / 牛棚崩盤 / 硬幣翻轉 / 開季混沌等）

### 4.6 預測紀錄
結果存入 `scripts/predictions.jsonl`，賽後回填實際比分。

### 4.7 輸出前驗證

> ⛔ **輸出前必須逐項檢查：**
- [ ] 紀律 1：ML + Log5 一致時，未因軟性因素翻轉勝方？
- [ ] 紀律 3：所有信號已轉為 run 修正，無獨立 O/U 方向？
- [ ] 讓分方向：已用 ML + 投手分析交叉驗證？
- [ ] 同場對立：未同時推 A 隊 ML + B 隊受讓？
- [ ] 比分一致：修正後總分方向與 O/U 推薦一致？

### 4.8 輸出格式

**TL;DR（最前面）**：
```
🎯 TL;DR
最可能比分：{主隊} {比分} - {比分} {客隊}（{勝方} 勝，勝率 {X}%）
📖 比賽走勢：{2-3 句敘事}
得分區間：{主隊} {低}-{高} / {客隊} {低}-{高}（預測總分 {低}-{高}）

💰 盤口速查：
| 盤口 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | ✅/⚠️ ... | ⭐⭐⭐⭐ | ... |
| O/U | ✅/⚠️ ... | ⭐⭐⭐ | 基礎 {X} + 修正 {+Y} = {Z} vs Line {L} |
| Run Line | ✅/⚠️/PASS | ⭐⭐ | ... |
```

**完整報告（TL;DR 之後）**：
1. 比賽資訊（日期、球場、先發投手）
2. 雙方近 10 場戰績（含 Pythagorean W%）
3. 先發投手對決（等級、進階數據、休息天數、用球數）
4. 打線分析（評級、熱度、串聯、投打對決）
5. 牛棚分析（品質、可用性、累計傷兵）
6. 球場 & 天氣（PF、風向）
7. 條件修正摘要（開季/傷病/年齡/賽季階段）
8. 勝率預測（含信心區間）
9. 比分預測（得分區間 + 情境機率分佈）
10. 盤口建議（含讓分方向確認 + 一致性檢查）

---

## 腳本執行順序與資料流向（🐍 模式）

### 完整執行流程

```bash
# 偵測 Python 指令（每次對話第一次執行腳本前先跑這行）
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)

# Phase 1：比賽基本資料
$PYTHON scripts/fetch_game_data.py --date YYYY-MM-DD --team [隊名] > game_data.json

# Phase 2A/B：雙方先發投手（可平行執行）
$PYTHON scripts/pitcher_stats.py --name "[主隊投手]" --year YYYY > home_pitcher.json
$PYTHON scripts/pitcher_stats.py --name "[客隊投手]" --year YYYY > away_pitcher.json

# Phase 2D：雙方打線（可平行執行）
$PYTHON scripts/lineup_analyzer.py --team [主隊] --year YYYY > home_lineup.json
$PYTHON scripts/lineup_analyzer.py --team [客隊] --year YYYY > away_lineup.json

# Phase 2 補充（WebSearch 例外）：天氣、主審、傷兵
# → 手動取得 bullpen ERA、park factor

# Merge：合併為 predict.py 所需格式
$PYTHON scripts/merge_game_data.py \
  --game game_data.json \
  --home-pitcher home_pitcher.json \
  --away-pitcher away_pitcher.json \
  --home-lineup home_lineup.json \
  --away-lineup away_lineup.json \
  --home-bullpen-era [數值] \
  --away-bullpen-era [數值] \
  --park-factor [數值] > merged.json

# Phase 4：預測（分析後手動傳入推薦結果）
$PYTHON scripts/predict.py --game-data merged.json --save \
  --adjusted-home [主隊調整得分] \
  --adjusted-away [客隊調整得分] \
  --ou-line [有效大小分線] \
  --ou-rec [OVER/UNDER/PASS] \
  --ml-rec [隊伍縮寫或PASS] \
  --ml-stars [0-5] \
  --run-line-rec [PASS或隊伍縮寫] \
  --signal-adjustments '{"信號名": 修正值}' \
  --tags "標籤1,標籤2" \
  --temperature [°F] \
  --wind-mph [mph]

# 盤口分析
$PYTHON scripts/odds_analyzer.py \
  --hk-home [主隊HK賠率] --hk-away [客隊HK賠率] \
  --total [大小分線] \
  --model-win-pct [predict.py 輸出的 home_win_pct / 100] \
  --predicted-home [主隊預測得分] \
  --predicted-away [客隊預測得分] \
  --quarter-handicap \
  --low-line [低線] --high-line [高線] \
  --handicap-giving [home/away] \
  --handicap-odds-hk [讓分賠率]
```

### merged.json 格式（predict.py FEATURE_COLS + _meta）

`merge_game_data.py` 自動生成，包含兩個區塊：

```json
{
  "home_starter_fip": 4.15,
  "home_starter_k_bb": 3.5,
  "home_starter_whip": 0.92,
  "away_starter_fip": 5.03,
  "away_starter_k_bb": 6.0,
  "away_starter_whip": 1.28,
  "home_batting_xwoba": 0.310,
  "home_batting_ops": 0.700,
  "home_batting_k_pct": 22.0,
  "away_batting_xwoba": 0.325,
  "away_batting_ops": 0.730,
  "away_batting_k_pct": 21.0,
  "home_bullpen_era": 4.0,
  "away_bullpen_era": 0.51,
  "home_recent_rs": 4.1,
  "home_recent_ra": 5.0,
  "away_recent_rs": 5.7,
  "away_recent_ra": 4.7,
  "park_factor": 102,
  "_meta": {
    "home_team": "Philadelphia Phillies",
    "away_team": "Arizona Diamondbacks",
    "home_sp": "Jesús Luzardo",
    "away_sp": "Michael Soroka",
    "home_sp_starts": 2,
    "away_sp_starts": 2,
    "venue": "Citizens Bank Park",
    "game_pk": 823482,
    "game_date": "2026-04-10T22:40:00Z"
  }
}
```

> `_meta` 由 `merge_game_data.py` 從 `game_data.json` 和 pitcher JSON 自動提取，**無需手動填寫**。`predict.py --save` 讀取 `_meta` 自動帶入 predictions.jsonl。

---

## 盤口輸入格式

### 使用者標準格式

使用者會以下列格式傳入盤口資訊，直接解析即可：

```
[客隊]打[主隊]([讓分符號]) 賠率 [HK]
大分 [總分線] 賠率 [HK]
小分 [總分線] 賠率 [HK]
客獨贏 賠率 [HK]
主獨贏 賠率 [HK]
```

### 讓分符號解讀規則（亞洲四分球 Quarter Handicap）

讓分符號附在哪支隊後面，那支隊就是讓分方（熱門）。
注金自動拆成兩半，各押一條線：

| 符號 | 拆分兩線 | 讓分方贏 1 分 | 讓分方贏 2+ 分 | 受讓方贏 |
|------|---------|-------------|--------------|---------|
| (1+50) | -0.5 / -1.0 | 贏半 | 全贏 | 全輸 |
| (1-50) | -1.0 / -1.5 | 賠半 | 贏半 | 全贏（受讓方視角）|

完整情境表（以押讓分方 $100 為例，賠率 HK 0.97）：

**(1+50) = -0.5 / -1.0：**
| 結果 | 押讓分方 | 押受讓方 |
|------|---------|---------|
| 讓分方贏 2+ 分 | +97（全贏）| -100（全輸）|
| 讓分方贏 1 分 | +48.5（贏半）| -50（賠半）|
| 受讓方贏 | -100（全輸）| +97（全贏）|

**(1-50) = -1.0 / -1.5：**
| 結果 | 押讓分方 | 押受讓方 |
|------|---------|---------|
| 讓分方贏 2+ 分 | +97（全贏）| -100（全輸）|
| 讓分方贏 1 分 | -50（賠半）| +48.5（贏半）|
| 受讓方贏 | -100（全輸）| +97（全贏）|

### HK 賠率格式

所有賠率 = HK 格式（獲利倍率）：
- 押 $100，獲利 = $100 × HK 賠率
- 本金 $100 另外退回（不含在賠率中）
- 轉換公式：HK ≥ 1.0 → American = HK × 100（如 1.24 → +124）
- 轉換公式：HK < 1.0 → American = -100 / HK（如 0.65 → -154）

### 讓分盤 vs 獨贏盤定價差異

兩者為獨立市場，定價不同屬正常。
分析時以**獨贏盤隱含勝率為主要參考**，讓分盤作為交叉驗證。

---

## Edge Cases

| 情境 | 處理 |
|------|------|
| 先發臨時更換 | 產生備案分析 |
| Doubleheader | 牛棚消耗累積 |
| Opener 策略 | 調整分析框架 |
| Coors Field | 4 月 PF=112（非 128），5 月後恢復 |
| 跨聯盟比賽 | BvP 較少，增加不確定性 |
| Innings Cap | 搜尋確認 |
| 交易截止前後 | 搜尋最新交易 |
| 九月擴編 | 搜尋確認今日陣容 |
| 季後賽 | 得分壓縮 ×0.84-0.86 |
| 二次 TJ | 65% RTP，42% 能投 10+ 場 |
| 亞洲盤口格式歧義 | 必須用 ML + 投手分析驗證 |
| 使用者質疑結果 | 紀律 2：獨立驗證後才決定 |
| 本季樣本不足 | 優先用投影系統 |
| BABIP 回歸 | 極端值預期回歸 ~.300 |
| 信號修正 vs O/U 差距 < 1.5 | 不推薦 |

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性，MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
