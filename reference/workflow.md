# 執行流程 SOP（Phase 1-4）

> 本檔是 `SKILL.md` 的完整執行細節。SKILL.md 僅保留骨架與閘門名稱，bash 命令、參數表、逐項 checklist 皆在此查閱。

---

## 初始化（每次對話開始時執行一次）

### Python 指令偵測

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
```

後續所有腳本呼叫皆使用 `$PYTHON`。Windows 通常只有 `python`；macOS/Linux 通常有 `python3`。

### 輸出目錄規範（每場比賽開始分析前設定）

```bash
GAME_DIR=analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}
mkdir -p $GAME_DIR
```

- 所有腳本輸出統一存放至 `$GAME_DIR/`，例如 `analysis-data/2026-04-13/CHC@PHI/`
- Doubleheader 加後綴：`{AWAY}@{HOME}-G1`、`{AWAY}@{HOME}-G2`
- 隊名使用縮寫（PHI、CHC、NYY 等），對照表見 `reference/teams-and-api.md`

### 腳本偵測（Glob 遞迴搜尋）

```
Glob(pattern="**/*.py")  ← 從 SKILL.md 所在目錄執行
```

⛔ 失敗時禁止自動使用 WebSearch，必須先詢問使用者：
> 「找不到分析腳本，請確認 `scripts/` 目錄位置，或告知腳本實際路徑。」

### 模式切換規範

**🐍 腳本模式（腳本偵測成功即啟用）**

- ⛔ 禁止使用 WebFetch / WebSearch 收集比賽**核心數據**
- ✅ 允許使用 WebSearch 的**唯一例外**（僅限一類）：
  1. 當日最新傷兵快訊（IL 異動）— 優先用 API 40 人名單 + IL 名單；WebSearch 僅作補充查詢
- ⛔ 任何腳本失敗 → 向使用者回報錯誤，禁止靜默改用 WebSearch
- ⛔ 所有腳本輸出必須使用 `--output / -o` 參數存檔，禁止使用 shell redirect `>`

---

## Phase 1：資料收集

### 1.1 日期解析

將使用者輸入轉為 `YYYY-MM-DD`。
> ⚠️ MLB 賽程使用**美國時間**。

### 1.2 執行資料擷取腳本

```bash
$PYTHON scripts/fetch_game_data.py --date {YYYY-MM-DD} --team {team} -o $GAME_DIR/game_data.json
```

腳本自動完成以下 API 呼叫：
- **請求 A**：當日賽程 + 先發投手（schedule + hydrate=probablePitcher）
- **請求 B & C**：雙方多窗口戰績（近 10 場 + 近 30 場 + 本季全部）（schedule + hydrate=linescore）
- **系列賽前場驗證**：自動偵測同系列賽前場並拉取實際比分

> 僅使用 `gameType = "R"` 例行賽，排除春訓。

> 腳本同時輸出 `game_data_summary.md` 至同目錄（~30-50 行 markdown，含戰績 / 趨勢 / 當前系列賽 / Streak 脈絡）。

### 1.3 Pythagorean Win%

- 從近 10 場計算 Pythagorean Win%（腳本 predict.py 內建 Pythagenport 公式）

### 1.4 輸出確認

✅ Read `$GAME_DIR/game_data_summary.md`，依其內容填入下方輸出模板。

ℹ️ 一般情況下無需 Read `game_data.json`；僅在 summary 缺漏 / 使用者明確要求查驗 / 除錯時 Read 完整 JSON。

```
📅 {日期} — {客隊} @ {主隊}（{球場}）
⚾ 先發：{客隊投手} vs {主隊投手}
🕐 {比賽時間} | 狀態：{Preview/Live/Final}
📊 {主隊} 近 10 場：{W}-{L}（RS/G {X} | RA/G {Y}）
📊 {主隊} 近 30 場：{W}-{L}（RS/G {X} | RA/G {Y}）（Pyth {Z}%）
📊 {主隊} 本季：{W}-{L}（RS/G {X} | RA/G {Y}）（{N} 場）
📈 {主隊} 趨勢：{↑上升 / →持平 / ↓下滑}
📊 {客隊} 近 10 場：{W}-{L}（RS/G {X} | RA/G {Y}）
📊 {客隊} 近 30 場：{W}-{L}（RS/G {X} | RA/G {Y}）（Pyth {Z}%）
📊 {客隊} 本季：{W}-{L}（RS/G {X} | RA/G {Y}）（{N} 場）
📈 {客隊} 趨勢：{↑上升 / →持平 / ↓下滑}
```

### 1.5 Phase 1 閘門

- [ ] `game_data.json` 已輸出
- [ ] `game_data_summary.md` 已輸出
- [ ] `gameType == "R"`（例行賽）
- [ ] Doubleheader → 列出所有場次供使用者選擇
- [ ] 無比賽 → 建議查前後日期
- [ ] 先發 TBD → 進 Phase 2 時透過 WebSearch 例外確認
- [ ] Final → 詢問使用者是否要改用 `mlb-post-game-review`

---

## Phase 2：投打驗證與資料擴充

> ⚠️ **嚴禁使用 Agent 子代理執行 WebSearch**（子代理無法存取 WebSearch/WebFetch）。
> 必須在主對話中直接平行呼叫多個 WebSearch。

### Step 1（🔒 阻塞）：Roster 檢查

⛔ **必須完成 Step 1 並通過閘門後，才能執行 Step 2。**

```bash
$PYTHON scripts/roster_checker.py --team {主隊teamId} --season {year} -o $GAME_DIR/home_roster.json
$PYTHON scripts/roster_checker.py --team {客隊teamId} --season {year} -o $GAME_DIR/away_roster.json
```

**Step 1 閘門（逐項確認，未通過不得繼續）：**
- [ ] 雙方 roster JSON 已輸出
- [ ] 主隊先發投手在 active roster？→ 否 = 暫停 Skill 並告知使用者
- [ ] 客隊先發投手在 active roster？→ 否 = 暫停 Skill 並告知使用者
- [ ] IL 名單已記錄 → 作為 Phase 3 牛棚/傷兵分析基礎

### Step 2（可平行，需 Step 1 閘門通過）：投手 + 打線

```bash
# 先發投手（可平行）
$PYTHON scripts/pitcher_stats.py --name "{主隊投手}" --year YYYY -o $GAME_DIR/home_pitcher.json
$PYTHON scripts/pitcher_stats.py --name "{客隊投手}" --year YYYY -o $GAME_DIR/away_pitcher.json

# 打線（可平行）
$PYTHON scripts/lineup_analyzer.py --team {主隊} --year YYYY --opposing-pitcher-id {客隊投手ID} -o $GAME_DIR/home_lineup.json
$PYTHON scripts/lineup_analyzer.py --team {客隊} --year YYYY --opposing-pitcher-id {主隊投手ID} -o $GAME_DIR/away_lineup.json
```

**Step 2 閘門（腳本輸出後逐項確認）：**
- [ ] 投手有 `role_change` 標記？→ 是 = ⛔ **僅用先發場次數據，牛棚期 ERA/FIP 不可用於先發評估**
- [ ] 打線數據僅含 active roster 球員？（比對 Step 1 roster）
- [ ] **ERA vs xERA 落差閘門**（觸發條件 / 處理見 `flags-checklist.md` §13）：補跑 `pitcher_stats.py --name "..." --year {YYYY-1} -o $GAME_DIR/{side}_pitcher_{YYYY-1}.json`，YoY Statcast 對比方法見 `matchup-factors.md#yoy-statcast-驗證`。未完成不得進 Phase 3；不得以「風險提示」代替驗證。

**B7 TaskCreate 樣板（Plan B 2026-04-22 §4.7，第 3 層 forcing function）：**

觸發 YoY 時，同步 TaskCreate 追蹤補跑進度：

```
subject: 補跑 {side} YoY 對比（{pitcher_name}）
description: 對比 5 項 Statcast 指標（avg_velo / pitch_types / whiff_pct / hard_hit_pct / xera）；結論寫入 phase3_summary.md §YoY 對比結論
```

此 task 必須 complete 才能進 Phase 3.5（summary 存檔）。predict.py --save 會硬擋 prior year file 缺失（§4.3）。

### Step 3（可平行）：合併數據 + 環境補充

> Step 2 閘門通過後，以下兩項**同時執行**：

#### 3a. 合併數據（腳本）

```bash
$PYTHON scripts/merge_game_data.py \
  --game $GAME_DIR/game_data.json \
  --home-pitcher $GAME_DIR/home_pitcher.json \
  --away-pitcher $GAME_DIR/away_pitcher.json \
  --home-lineup $GAME_DIR/home_lineup.json \
  --away-lineup $GAME_DIR/away_lineup.json \
  -o $GAME_DIR/merged.json
```

- 自動取得牛棚 ERA + Park Factor
- 輸出 `merged.json`，作為 Phase 4 `predict.py` 的輸入

#### 3b. 環境補充

| 任務 | 核心指標 |
|------|---------|
| 傷兵名單 | 以 Step 1 的 40 人名單 + IL 名單為主（API 抓取） |
| 球場 | Park Factor（查 `matchup-factors.md` §Park Factor；未來接大數據 md 檔） |
| 盤口賠率 | ML / Run Line / O/U + 讓分方向驗證 |

盤口分析（使用者提供盤口數據後執行，輸入格式見 `reference/odds-format.md`）：

```bash
$PYTHON scripts/odds_analyzer.py --hk-home {hk} --hk-away {hk} ... -o $GAME_DIR/odds_analysis.json
```

---

## Phase 2 → Phase 3 轉換檢查（Plan B 2026-04-22 §4.7）

⛔ 進入 Phase 3 前必須：

1. **TaskList 檢查**：Phase 2 產生的 V 類 tasks（B7 YoY 補跑）全部 complete
2. 有 pending task 不得進 Phase 3

---

## Phase 3：綜合分析（順序執行）

> ⛔ **分析前查表**：Read `reference/matchup-factors.md`（投手分級、打線評級、牛棚傷兵修正、條件修正值）
> 後續修正依賴前面的基礎判斷，必須順序執行。

| 步驟 | 分析內容 | 閘門 | 參考 |
|------|---------|------|------|
| 3.1 投打對決 | 投手分級 + 打線評級 + Platoon + 球種 | ⛔ BvP：PA>=15 才可引用 | `matchup-factors.md` |
| 3.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵修正 | ⛔ 雙向閘門：O/U 和 ML 修正值皆填 | `matchup-factors.md` |
| 3.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場 | 僅符合條件時觸發 | `matchup-factors.md` + `prediction.md` |
| 3.4 近期狀態 | 多窗口趨勢 + H2H + 連勝敗 | ⛔ BABIP 回歸閘門：Hot/Cold 前必檢查 | `matchup-factors.md` |

**B9 牛棚雙向閘門擴充（Plan B 2026-04-22 §4.7，第 3 層 TaskCreate forcing function）：**

⛔ 偵測核心（Closer / Primary Setup / High-leverage）IL 任一人時，立即 TaskCreate：

```
subject: 牛棚雙向修正值（核心 {N} 人 IL）
description: 同時計算 ML 修正（-%）+ OU 修正（+run）；寫入 phase3_summary.md §牛棚雙向修正值；呼叫 predict.py 時 --signal-adjustments 含 bullpen_il_{side}
```

此 task 必須 complete 才能進 Phase 3.5。predict.py --save 會硬擋缺 section（§4.5）。

**B10 BABIP 回歸閘門擴充（Plan B 2026-04-22 §4.7）：**

⛔ 偵測任一打線近 7 天 BABIP 極端值時（閾值見 `matchup-factors.md` §BABIP 回歸檢查），立即 TaskCreate：

```
subject: BABIP 回歸判定（{team} 近 7 天 {value}）
description: 回歸 ~.300 後判定 Hot/Cold 是否調整；結論寫入 phase3_summary.md §BABIP 回歸判定
```

此 task 必須 complete 才能進 Phase 3.5。

### Phase 3 → Phase 3.5 轉換檢查（Plan B 2026-04-22 §4.7）

⛔ 進入 Phase 3.5（phase3_summary.md 存檔）前必須：

1. **TaskList 檢查**：本 Phase 產生的 V 類 tasks（B9 牛棚雙向、B10 BABIP 回歸）全部 complete
2. 有 pending task 不得進 Phase 3.5

> Phase 4（predict.py --save）會透過 phase3_summary.md grep 硬擋缺 section 的情況（第 2 層 code 防線，Plan B §4.5）。

### 3.5 分析結論存檔（phase3_summary.md）

⛔ **Phase 3 完成後、Phase 4 開始前，必須將分析結論寫入 `$GAME_DIR/phase3_summary.md`。**

**MUST contain（基本面分析結論）**：
- 雙方先發投手分級與關鍵數據
- 打線評級與熱度判定
- 牛棚傷兵修正值（O/U +run、ML -%）
- 條件修正摘要（觸發了哪些信號、各自的 Run Value）
- 修正後預期得分（主隊 / 客隊 / 總分）
- 整體判斷（方向傾向 + 信心程度 + 值得注意的風險）

⛔ **MUST NOT contain（盤口推薦專屬 Phase 4）**：
- ML / O/U / Run Line 星級（`⭐⭐⭐` 等）
- 明確盤口格式（`ML XXX ⭐⭐`、`OVER ⭐⭐⭐` 等）
- 「初步盤口推薦」「盤口初判」等任何預判段落

**原則**：盤口推薦的 single source of truth = Phase 4 產生的 `prediction.json`。Phase 3 summary 是基本面快照，不得包含任何需要 `predict.py` 模型輸出才能確定的結論。

「整體判斷」可以表達方向性（例如「基本面偏 HOME，投手差 2 檔」），但不得給出具體盤口或星級。

> 此檔案確保 Phase 4 執行時，分析結論不會因對話壓縮而遺失；但推薦必須在 Phase 4 產生，避免 stale。

---

## Phase 4：預測輸出

> ⛔ **預測前載入**：
> - Read `$GAME_DIR/phase3_summary.md`（Phase 3 分析結論）
> - Read `reference/prediction.md`（公式、信號表、星級門檻、紀律規則 D1-D5）

### 4.0 執行預測腳本

```bash
$PYTHON scripts/predict.py --game-data $GAME_DIR/merged.json --save [分析後參數]
```

**`--save` 分析後參數（Phase 3 完成後必須傳入）**：

| 參數 | 必填 | 說明 |
|------|------|------|
| `--adjusted-home` | 建議 | 分析後調整的主隊得分 |
| `--adjusted-away` | 建議 | 分析後調整的客隊得分 |
| `--ou-line` | 是 | 有效大小分線（四分球取中位，如 9.5） |
| `--ou-rec` | 是 | OVER / UNDER / PASS |
| `--ml-rec` | 是 | 隊伍縮寫或 PASS（Plan B 2026-04-22 W2：`HOME` / `AWAY` 字面值會被 reject） |
| `--ml-stars` | 是 | 0-5 |
| `--signal-adjustments` | 建議 | JSON 格式，如 `'{"puk_il":0.3}'`（未知 key 會 stderr warning） |
| `--tags` | 建議 | 逗號分隔，如 `divergent,early-season` |
| `--temperature` | 若有 | 氣溫 °F |
| `--wind-mph` | 若有 | 風速 mph |
| `--wind-direction` | 若有 | 風向 |
| `--umpire` | 若有 | 主審姓名 |
| `--umpire-ou-rate` | 若有 | 主審 Over% |

> **RL 推薦（Plan B 2026-04-22 W1）**：無 `--run-line-rec` / `--run-line-stars` CLI args，已廢除。RL 全走 `predict.py` auto override（RL-1b gate：|adj 比分差| ≥ 1.5 + strong tag 或 big-diff ≥ 2.2）。

> **自動 Odds 查詢**：`predict.py --save` 會自動從 `odds_snapshots/` 撈推薦時間最近的 Pinnacle snapshot 作為 Kelly 計算來源（Kelly 區塊詳見 `reference/prediction.md` Kelly Sizing 章節）。若需手動覆寫，加 `--ml-odds-home-dec 1.83` / `--ou-odds-over-dec 1.91` / `--rl-odds-home-dec 1.56` 等 args。Doubleheader 需指定 `--game-index 1` 或 `2`。

> ⚠️ **勝率與比分皆用 predict.py 的 `formula_prediction`**（XGBoost 路徑於 2026-04 重構移除）。手動估算只能作為輔助驗算。

### 4.1 PASS 規則與星級護欄

- PASS 門檻、星級護欄規則 → 見 `prediction.md`「PASS 門檻 + 星級護欄」章節
- predict.py 自動執行星級護欄，確認輸出中的降級警告

### 4.2 比分與信號修正

- 比分公式 + Run Value 信號修正 → 見 `prediction.md`「比分預測方法 + 信號 → Run Value 修正表」

### 4.3 盤口推薦

- ⛔ **紀律閘門 D1/D2** → 見 `prediction.md`「分析紀律」
- O/U、ML、Run Line、讓分方向交叉驗證 → 見 `prediction.md`「讓分方向交叉驗證」

### 4.4 硬性規則（Phase 4 閘門）

- ⛔ **D3：禁止同場對立方向推薦** → 見 `prediction.md` D3
- ⛔ **D5：比分與盤口一致性** → 見 `prediction.md` D5

### 4.5 比賽敘事

- 根據量化觸發條件選擇劇本 → 見 `prediction.md`「比賽敘事觸發條件」

### 4.6 寫入 prediction.json

⛔ **predict.py --save 成功後，prediction.json 自動落在 `$GAME_DIR/prediction.json`**（單數 JSON、per-game 真相來源）。

```bash
# predict.py --save 已完成 prediction.json 寫入；無需額外動作
```

> **當日彙總與賽後回填**請交由 `mlb-post-game-review` skill 處理，不屬於本 skill 範圍。

### 4.7 輸出前驗證

⛔ **輸出前必須逐項檢查：**

- [ ] D1 / D2 紀律通過？
- [ ] D3 同場無對立推薦？
- [ ] D5 比分與盤口一致性？
- [ ] 讓分方向已交叉驗證？
- [ ] 牛棚傷兵雙向反映（O/U + ML）？
- [ ] 星級護欄降級警告已確認？
- [ ] Roster 一致？

### 4.8 輸出格式

完整模板見 `reference/output-format.md`（TL;DR + 10 段完整報告）。
