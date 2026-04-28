# Phase 1 `game_data_summary.md` 設計

**日期**: 2026-04-26
**Skill**: mlb-game-analyzer
**範圍**: Phase 1 資料收集流程的 context 精簡

---

## 1. 背景與目標

### 問題

Phase 1 執行 `fetch_game_data.py` 後產生的 `game_data.json` 約 1150 行，主要篇幅來自雙方 近10 / 近30 / 本季 三個區間的逐場比分陣列。

實證觀察：
- Phase 1 SOP 輸出模板只用聚合欄位（`record / rs_per_game / ra_per_game / run_diff / streak`）
- 下游腳本（已 grep 確認）：`merge_game_data.py` 與 `predict.py` 只讀聚合欄位，**完全不消費 `games` 陣列**
- 但 Claude 為了從 JSON 抓聚合欄位，會 Read 完整 1150 行 → 浪費 token、稀釋 context、提高對話壓縮機率

### 目標

Phase 1 階段，Claude 只 Read 一個 30-50 行的 markdown summary，**完整 JSON 保留不動**（人類除錯可讀性 + 下游腳本相容性）。

### 範圍邊界

| 範圍內 | 範圍外 |
|--------|--------|
| 改 `scripts/fetch_game_data.py` 額外輸出 summary | 修改 `merge_game_data.py` / `predict.py` |
| 改 `SKILL.md` / `reference/workflow.md` Phase 1 SOP | 回填歷史 `analysis-data/` 目錄 |
| 新增 `scripts/tests/test_fetch_game_data.py` 純函式測試 | 加入 SoS / 強隊勝率 / 季 H2H 累計 |

---

## 2. Summary 檔案規格

### 路徑與命名

- 與 `game_data.json` 同目錄
- 固定檔名 `game_data_summary.md`
- 若 `--output` 未指定（純 stdout 模式），不產生 summary（與舊行為一致）

### 內容結構（範例：LAA @ KC, 2026-04-26）

```markdown
# Game Data Summary — LAA @ KC (2026-04-26)

## 比賽資訊
- 日期 (ET): 2026-04-26
- 開賽: ET 18:20
- 球場: Kauffman Stadium
- 狀態: Preview
- 先發: Reid Detmers (LAA) vs Seth Lugo (KC)

## 戰績摘要

| 區間 | KC（主） | LAA（客） |
|------|---------|----------|
| 近 10 場 | 3-7  (RS 5.10 / RA 6.00 / diff −9  / streak +2) | 3-7  (RS 4.00 / RA 4.50 / diff −5  / streak −3) |
| 近 30 場 | 10-18 (RS 3.79 / RA 4.54 / diff −21)             | 12-16 (RS 4.64 / RA 4.79 / diff −4)             |
| 本季    | 10-18 (28 場)                                    | 12-16 (28 場)                                    |

## 趨勢（近 10 vs 近 30）
- KC: 攻↑ (RS 5.10 vs 3.79，+1.31) | 守↓ (RA 6.00 vs 4.54，+1.46)
- LAA: 攻↓ (RS 4.00 vs 4.64，−0.64) | 守→ (RA 4.50 vs 4.79，−0.29)

> 規則：|Δ| ≥ 0.5 才標箭頭。攻↑ = RS 上升；守↓ = RA 上升（防守變差）。

## 當前系列賽 (LAA @ KC)
- G1 (04-24): KC 6-3 LAA → KC 勝
- G2 (04-25): KC 12-1 LAA → KC 勝
- G3 (04-26): 本場
- 系列累計: **KC 2-0 LAA**

## Streak 脈絡
- KC +2: 連勝對手 → LAA (04-24), LAA (04-25)
- LAA −3: 連敗對手 → TOR (04-22), KC (04-24), KC (04-25)
```

### Section 對應的資料來源

| Section | 來源欄位 | 計算 |
|---------|---------|------|
| 比賽資訊 | `game.*` | 直接帶 |
| 戰績摘要 | 6 個聚合區塊（home/away × recent/recent_30/season）的 `record / rs_per_game / ra_per_game / run_diff / streak` + `home_season_games_count` / `away_season_games_count` | 直接帶 |
| 趨勢 | `recent.rs_per_game − recent_30.rs_per_game`、`recent.ra_per_game − recent_30.ra_per_game` | `compute_trend_arrows` |
| 當前系列賽 | `home_recent.games`（從 index 0 往後連續同對手） | `detect_current_series` |
| Streak 脈絡 | `home_recent.games[:abs(streak)]` 的 opponent 列表 | `format_streak_context` |

### 球隊縮寫規則

- 反向查 `TEAM_MAP`（team_id → 縮寫，例如 118 → KC）
- 對應 `TEAM_MAP` 中的英文縮寫優先；找不到時用 team name 前 3 字大寫 fallback

### 趨勢箭頭規則

- 計算 `Δ_off = rs_per_game(近10) − rs_per_game(近30)`
- 計算 `Δ_def = ra_per_game(近10) − ra_per_game(近30)`
- 攻擊箭頭：`Δ_off ≥ +0.5` → 攻↑，`Δ_off ≤ −0.5` → 攻↓，否則 攻→
- 守備箭頭：`Δ_def ≥ +0.5` → 守↓（失分上升 = 防守變差），`Δ_def ≤ −0.5` → 守↑（失分下降 = 防守變好），否則 守→
- 數值顯示保留小數第二位

### 系列賽偵測規則

- 從 `home_recent.games[0]` 開始向後掃描
- 遇到 opponent 等於當前對手球隊名 → 加入系列賽
- 遇到不同對手 → 停止
- 收集到的場次按日期升序排列，編號 G1, G2, ...
- 本場（未開始）編號 = 已收集場次數 + 1

### Streak 脈絡規則

- 從 `home_recent.games[0]` 取 `abs(streak)` 場
- 列出每場對手 + 日期（MM-DD）
- 連勝 → `"連勝對手 → ..."`、連敗 → `"連敗對手 → ..."`
- 對手球隊名顯示用縮寫（同球隊縮寫規則）

---

## 3. 邊界條件處理（混合模式）

### Hard sections（必須出現，缺值降級顯示）

- 比賽資訊
- 戰績摘要表
- 趨勢

缺值時使用 `—` 取代數值，**不**寫「資料不足」字樣。

### Soft sections（不適用就整段省略）

- 當前系列賽
- Streak 脈絡

### 逐個邊界處理

| 情境 | 處理 |
|------|------|
| `probable_pitcher == "TBD"` | 比賽資訊照常顯示 `先發: TBD vs Seth Lugo` |
| 戰績區塊欄位為 `None` | 顯示 `—`，row 保留 |
| 雙方近 10 場場數 < 10 | 戰績表用實際場數計算；不額外提示（已隱含在「本季 X-Y (N 場)」） |
| `series_prev` 為 `None` 且 `home_recent.games[0]` 對手不是當前對手 | **本系列首戰** — section 顯示：<br>`G1 (MM-DD): 本場`<br>`系列累計: 本系列首戰，無前場` |
| Doubleheader（同日對同對手 2 場） | 系列賽 section 各列一行，標 `(DH-1) / (DH-2)`；G 編號連續遞增 |
| `streak == 0` | Streak 脈絡 section 整段省略 |
| `home_recent.games` 為空 | 戰績表 row 顯示 `0-0 (—)`；趨勢、系列賽、Streak 脈絡 section 全部省略 |
| 雙方在系列賽中段移地（互換主客） | 系列偵測仍運作（按 opponent 名）；不特別標主客變化 |

### Fail-fast 條件

直接 raise（不寫 summary、不靜默降級）：
- `result["game"]` 缺失
- 無法解析 `home.team_id` 或 `away.team_id`

---

## 4. 實作項目

### 4.1 `scripts/fetch_game_data.py` 變更

新增純函式：

| 函式 | 簽章 | 職責 |
|------|------|------|
| `team_abbr(team_id, team_name)` | `(int \| None, str) → str` | 優先 TEAM_MAP[team_id] 反查；team_id 為 None 時用 FULL_NAMES[team_name.lower()] 反查；都失敗 fallback 用 team_name 前 3 字大寫 |
| `compute_trend_arrows(rs10, ra10, rs30, ra30)` | `(float×4) → dict` | 返回 `{off_arrow, def_arrow, off_delta, def_delta}` |
| `detect_current_series(games, current_opp_team_name, current_game_date)` | `(list, str, str) → list[dict]` | 從 `games[0]` 連續同對手收集，含 DH 標記 |
| `format_streak_context(games, streak)` | `(list, int) → str \| None` | streak=0 或 games 空時返回 None；內部呼叫 `team_abbr(None, opponent_name)` |
| `format_summary_md(result_dict)` | `(dict) → str` | 組合所有 sections，套用混合模式邊界規則 |

修改 `main()`：
- 寫完 JSON 後額外寫 `Path(args.output).parent / "game_data_summary.md"`
- stderr 輸出：`Saved summary to <path>`
- `--output` 未指定時不產生 summary（保持舊 stdout 行為）

### 4.2 `scripts/tests/test_fetch_game_data.py`（新增）

測試類別與案例：

| Test class | Test cases |
|-----------|-----------|
| `TestComputeTrendArrows` | Δ=+0.5 邊界、Δ=+0.49 → →、Δ=−0.5 邊界、雙 ↑、雙 ↓、混合方向、零差距 |
| `TestDetectCurrentSeries` | G1 首戰（games[0] 對手不同）、G3（前 2 場同對手）、Doubleheader（同日同對手 2 場）、空 games |
| `TestFormatStreakContext` | streak=+2、streak=−3、streak=+1、streak=0（→ None）、空 games（→ None） |
| `TestTeamAbbr` | 已知 team_id（118 → KC）、未知 team_id fallback、team_name 為英文全名、team_name 為中文 |

### 4.3 `SKILL.md` / `reference/workflow.md`

`workflow.md` 變更（建議性，無 ⛔）：

- Phase 1.2 後加：「腳本同時輸出 `game_data_summary.md` 至同目錄」
- Phase 1.4 開頭加：「✅ Read `$GAME_DIR/game_data_summary.md`，依其內容填入下方輸出模板」
- Phase 1.4 後加：「ℹ️ 一般情況下無需 Read `game_data.json`；僅在 summary 缺漏 / 使用者明確要求查驗 / 除錯時 Read 完整 JSON。」
- Phase 1.5 閘門加一條：`[ ] game_data_summary.md 已輸出`

`SKILL.md` 變更：
- Quick Reference 表 Phase 1 行的「主要產出」由 `game_data.json` 改為 `game_data.json + game_data_summary.md`

### 4.4 不動的部分

- `merge_game_data.py` / `predict.py` / 其他 reference 檔
- 歷史 `analysis-data/` 目錄

---

## 5. 預期收益

| 指標 | 改動前 | 改動後 |
|------|-------|-------|
| Phase 1 Claude Read 行數 | ~1150 行 JSON | ~30-50 行 markdown |
| Token 占用 | 高 | 降幅約 95%+ |
| 下游腳本相容性 | — | 完全相容（JSON 結構不變） |
| 人類除錯可讀性 | JSON 完整保留 | JSON 完整保留 + summary 補強 |

---

## 6. 後續

實作計畫由 `writing-plans` skill 產出於 `docs/superpowers/plans/`。
