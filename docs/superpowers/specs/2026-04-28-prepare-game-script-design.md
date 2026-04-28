# `prepare_game.py` 一鍵 Phase 1+2 整合 + Skill 瘦身設計

**日期**: 2026-04-28（修訂 2026-04-29）
**Skill**: mlb-game-analyzer
**範圍**: Phase 1+2 全流程整合、reference 大砍、移除 luck-based 自動回測

---

## 1. 背景

### 觀察到的痛點（2026-04-28 TB@CLE 實測 + 流程重檢）

| # | 痛點 | 來源 | 影響 |
|---|------|------|------|
| **P1** | AI 連續 Read 8 份 `*_summary.md`（game / 2×roster / 2×pitcher / 2×lineup / merged） | Phase 1+2 SOP 要求逐個確認閘門 | 約 750 行 / 60-70% phase 1+2 token |
| **P2** | Pitcher ID 取得繞一圈：`fetch_game_data.py` 只存名字 → 必須先跑 `pitcher_stats.py` 取 `mlbam_id` → 再 `lineup_analyzer.py --opposing-pitcher-id` | `fetch_game_data.py` 沒存 `probable_pitcher_id` | 多一個 sequential hop |
| **P3** | `Nick Martinez`（純 ASCII）在 `playerid_lookup` strict match 失敗，必須改 `Nick Martínez` | `pitcher_stats.py` 沒 diacritic fallback | AI debug 一輪 |
| **P4** | `phase3_summary.md` H2 不符 `predict.py` grep（`### YoY 對比結論` 不被認） | predict.py grep 規則只認 H2 開頭 | AI 重 Edit + 重跑 predict |
| **P5** | `predict.py --save` 漏帶 `--ou-stars` 自動降 OU 為 PASS（silent fallback） | CLI args 設計上 stars 為 optional | AI 重跑 |
| **P6** | B7 YoY / B10 BABIP TaskCreate 兩個 forcing function 已被 `*_summary.md` 自動偵測取代，TaskCreate 變純儀式 | Plan B 2026-04-22 §4.7 第 3 層 forcing function | 多 2 次 TaskCreate + TaskUpdate |
| **P7** | `reference/` 結構臃腫（5 檔 1031 行），包含大量「AI 不該讀」的 SOP 細節（Phase 1+2 命令、API 端點、隊名表） | 流程演化累積 | AI 每場讀大量無關內容 |
| **P8** | Flag 13（\|ERA-xERA\| ≥ 1.5）/ Flag 3（last7 BABIP 極端）自動觸發補跑 + 自動 ±run value 本身是 luck-based bet — 投手可能整季都運氣好，「期待回歸」也是賭運氣 | 流程演化累積 | 偽科學的修正、預測信心虛胖 |

### 目標

1. **Phase 1+2 整合為一支 `prepare_game.py`** — AI 一條命令啟動，全程不需平行 / 序列協調
2. **AI 唯一需要 Read 的整合檔：`dossier.md`**（取代 8 份 summary）
3. **Phase 3 寫作由 `phase3_skeleton.md` 預先框定 H2 + 預填表格** — 消除 P4 grep 衝突、P6 的 TaskCreate 儀式
4. **修掉上游 P2 / P3 兩個 gotcha**
5. **`reference/` 大砍**：5 檔 1031 行 → 3 檔 ~310 行（含改寫的 SKILL.md ~150 行）
6. **移除 luck-based 自動回測**：刪 Step C-prior、Flag 3/13 改為「標註風險、AI 敘事判斷、不自動調整 run value」

### 目標達成後的新 SOP

```
1. python scripts/prepare_game.py --date YYYY-MM-DD --away XXX --home YYY
   → 產出 merged.json / dossier.md / phase3_skeleton.md
2. AI Read dossier.md（一份，~200 行）
3. AI 補 phase3_skeleton.md 結論段落 → 存檔為 phase3_summary.md
4. AI 跑 predict.py --save
5. AI Read prediction_summary.md → 輸出最終報告
```

每場 token 消耗預估：**舊 ~750 行 → 新 ~200 行（dossier 200 + skeleton 30 - workflow.md 不再讀）**

---

## 2. 範圍邊界

| In Scope | Out of Scope |
|---|---|
| 新增 `scripts/prepare_game.py` | 改 `predict.py` 的 D1-D5 紀律 / formula_prediction 公式 |
| 改 `scripts/fetch_game_data.py` 加 `probable_pitcher_id` | 改 `prediction.json` schema |
| 改 `scripts/pitcher_stats.py` 加 diacritic fallback | 改 `merged.json` 給 predict.py 機讀的 schema |
| 改 `scripts/predict.py`：`--ou-stars` 必填化 + 移除 H2 grep guard | 改 `mlb-post-game-review` skill |
| 新增 `dossier.md` schema | 改 `*_summary.md` 內容（仍由各腳本產生） |
| 新增 `phase3_skeleton.md` schema | |
| 刪除 `reference/teams-and-api.md`（已存在於 `_team_resolver.py`） | |
| 刪除 `reference/workflow.md`（併入 SKILL.md） | |
| 改寫 `reference/flags-checklist.md`（移 Flag 7、改寫 Flag 3/13） | |
| 改寫 `reference/matchup-factors.md`（刪 YoY 整段、BABIP 改風險標註） | |
| 改寫 `reference/prediction.md`（刪比賽敘事、刪 JSON schema） | |
| 改寫 `SKILL.md`（合併 workflow 內容） | |
| 對應 unit tests | |

**重要決策**：
- `*_summary.md` **不刪除** — 仍由各腳本產生，作 drill-down / debug
- `dossier.md` 是「正常路徑」入口，AI 預設只讀這個
- 不再做 Step C-prior（YoY 補跑）— 即使 Flag 13 觸發也只標 ⚠️ 風險，AI 在敘事處理

---

## 3. `prepare_game.py` 規格

### 3.1 CLI 介面

```
python scripts/prepare_game.py \
  --date YYYY-MM-DD \
  --away XXX \
  --home YYY \
  [--output-dir analysis-data/{date}/{away}@{home}]   # 預設依規範組
  [--season YYYY]                                      # 預設 = year of date
  [--game-suffix G1|G2]                                # Doubleheader 用
  [--force]                                            # 覆蓋已存在的輸出檔
```

**錯誤處理規範**：

| Exit code | 條件 |
|---|---|
| 0 | 全部成功 |
| 2 | `gameType ≠ "R"`（春訓 / 季後賽不支援） |
| 3 | 雙隊未對戰 |
| 4 | Doubleheader 但未指定 `--game-suffix` |
| 5 | 先發不在 active roster |
| 7 | API 失敗（網路 / 5xx 持續）— exit 6 預留給 `predict.py` 的 `--ou-stars` 必填錯誤（見 §6.3a），避免 cross-script 混淆 |

任何子步驟失敗 → 立即 stderr 輸出錯誤 + 非 0 exit code；**不靜默 fallback**。

### 3.2 內部執行順序（單一進程，全自動）

```
[Step A] fetch_game_data.py 邏輯內聯
  → 寫 game_data.json + game_data_summary.md
  → 取出 home_probable_pitcher_id / away_probable_pitcher_id（見 §6.1）

[Step B] roster_checker.py 邏輯內聯（雙隊平行）
  → home_roster.json / away_roster.json + summaries
  → 自動帶 --expected-starter（從 Step A 取得名字）
  → 若有「先發不在 active」→ 立即 exit 5 + 訊息

[Step C] pitcher_stats.py 邏輯內聯（雙隊平行）
  → 直接用 Step A 的 mlbam_id（不再走 name lookup）— 解決 P3
  → home_pitcher.json / away_pitcher.json + summaries
  → 偵測 Flag 13 → **僅 stderr ⚠️，不補跑 prior year**

[Step D] lineup_analyzer.py 邏輯內聯（雙隊平行）
  → 直接用 Step A 的 mlbam_id 作為 --opposing-pitcher-id — 解決 P2
  → home_lineup.json / away_lineup.json + summaries
  → 偵測 Flag 3 → **僅 stderr ⚠️，不下修 run value**

[Step E] merge_game_data.py 邏輯內聯
  → merged.json + merged_summary.md（不變，仍給 predict.py 機讀）

[Step F] dossier.md 渲染（見 §4）
[Step G] phase3_skeleton.md 渲染（見 §5）
```

**移除 Step C-prior**（YoY 補跑）— 不再做 luck-based 自動回測。

**平行化**：Step B / C / D 內各自雙隊平行；Step 之間因相依關係（D 需 A 的 ID，E 需 A+C+D）必須序列。

### 3.3 輸出

```
analysis-data/{date}/{away}@{home}/
├── game_data.json
├── game_data_summary.md
├── home_roster.json
├── home_roster_summary.md
├── away_roster.json
├── away_roster_summary.md
├── home_pitcher.json
├── home_pitcher_summary.md
├── away_pitcher.json
├── away_pitcher_summary.md
├── home_lineup.json
├── home_lineup_summary.md
├── away_lineup.json
├── away_lineup_summary.md
├── merged.json
├── merged_summary.md
├── dossier.md                     ★ 新增（AI 主要入口）
└── phase3_skeleton.md             ★ 新增（AI 補結論用）
```

共 18 個檔案（原 spec 22 個 - 4 個 `*_pitcher_{YYYY-1}.*` prior year 檔）。

Doubleheader：`dossier.md` / `phase3_skeleton.md` 跟 `--game-suffix` 走，分別命名為 `dossier-G1.md` / `dossier-G2.md` / `phase3_skeleton-G1.md` / `phase3_skeleton-G2.md`。其他檔案沿用 `--output-dir` 指定的目錄結構（如 `analysis-data/{date}/{a}@{h}-G1/`）。

### 3.4 stdout 規範

```
[A] game_data        ✓
[B] rosters          ✓ (home 26P/13B IL=3 / away 26P/13B IL=9)
[C] pitchers         ✓
[D] lineups          ✓
[E] merge            ✓
[F] dossier.md       → analysis-data/2026-04-28/TB@CLE/dossier.md
[G] phase3_skeleton  → analysis-data/2026-04-28/TB@CLE/phase3_skeleton.md

⚠️  Risk Notes (AI 在 phase3_skeleton 風險提示段處理):
  - away pitcher Flag 13 (era_xera_delta=-2.54)
  - away lineup Flag 3 (last7 BABIP=0.241)
```

無 Flag 觸發時：`Risk Notes:` 段顯示 `（無）`。

---

## 4. `dossier.md` 結構

### 4.1 設計原則

- **AI 一次 Read 完即拿到 Phase 1+2 全部關鍵資訊**，不需再 Read 任何 phase 1/2 summary
- **長度上限：250 行**（含 markdown 表格）
- **Drill-down 規則**：dossier 末尾統一列「File 索引」，不再 inline `> Drill-down:` 註腳

### 4.2 章節結構（強制順序）

```markdown
# Game Dossier — {AWAY} @ {HOME} ({YYYY-MM-DD})

## 比賽資訊
（從 game_data_summary.md 拷貝：日期、開球時間、球場、狀態、先發名字 + ID + GS 數）

## 戰績速查
| 區間 | HOME (RS/RA/diff/streak) | AWAY (RS/RA/diff/streak) |
| 近 10 | ... | ... |
| 近 30 | ... | ... |
| 本季 | ... | ... |
| 趨勢 | (攻↑/守↓ 等箭頭) | (...) |

## 系列脈絡
（拷貝 game_data_summary.md「系列賽」+「Streak 脈絡」section）

## 投手對決
| | HOME ({pitcher}) | AWAY ({pitcher}) |
| Tier (script) | 🟢 Back-end | 🟠 Strong Ace |
| Hand / Age | RHP / 27 ⚡ | RHP / 35 📉📉 |
| ERA / xERA | 4.45 / 4.64 | 2.10 / 4.64 |
| FIP / xFIP | 4.62 / 3.85 | 3.87 / 4.34 |
| K-BB% / WHIP | 11.4 / 1.45 | 9.8 / 1.10 |
| velo (avg/max) | 87.3 / 96.1 | 86.8 / 94.4 |
| whiff% / hard_hit% / barrel% | 12.1 / 34.5 / 8.5 | 7.1 / 25.8 / 8.5 |
| 主球種 (top3) | FF 27.8 / FC 27.5 / CH 17.9 | SI 31.3 / CH 27.1 / FC 18.8 |
| vs LHB (slash) | .238/.314/.492 (70 BF) | .239/.311/.388 (74 BF) |
| vs RHB (slash) | .316/.361/.386 (61 BF) | .196/.208/.283 (48 BF) |
| 近 3 場 ER/IP | 5/13.7 | 4/16.7 |
| ⚠️ 風險提示 | — | era_xera_delta=-2.54（Flag 13）|

## 打線
| | HOME | AWAY |
| Tier (script) | 🟡 Average | 🟢 Weak |
| Heat (script) | ⚖️ Normal | ⚖️ Normal |
| xwOBA / OPS | 0.331 / .707 | 0.304 / .711 |
| K% / BB% | 20.3 / 10.7 | 17.8 / 7.8 |
| chain OBP top3 / SLG mid | .329 / .452 | .368 / .299 |
| last7 BABIP | 0.284 | 0.241 |
| ⚠️ 風險提示 | — | last7 BABIP=0.241（Flag 3）|

### Top 5 vs 對方先發手感（PA 排序，PA ≥ 30，IL'd 排除）
**HOME vs {AWAY pitcher}（RHP）**:
| # | Name | season OPS | vs RHP OPS | last7 OPS | last7 BABIP | EV95% | Barrel% |
| 1 | José Ramírez | .804 | .703 | .663 | .273 | 45.7 | 10.9 |
| ...（最多 5 人，候選人 < 5 就少列）

> last7 OPS top1（不在 PA top 5 內）：Schneemann (last7 OPS 1.164)
> 若 top1 已在表內或無候選，省略此行

**AWAY vs {HOME pitcher}（RHP）**: 同格式

## 牛棚 / Park
| | HOME | AWAY |
| Bullpen ERA | 4.57 | 5.18 |
| 投手 IL 數 | 2 | 8 |
| 核心 IL（按角色判斷） | 1 (Walters HL setup) | 2+ (Cleavinger HL setup, Boyle, ...) |
| Park Factor (runs) | 101 | — |
| Park 備註 | Progressive Field 2024 LHB HR +16% | — |

## ⚠️ 風險提示摘要（AI 在 phase3_skeleton 風險提示段處理）
- AWAY 投手：era_xera_delta=-2.54（Flag 13）— 運氣或結構性？AI 判斷，**不自動補跑 YoY、不自動下修預測**
- AWAY 打線：last7 BABIP=0.241（Flag 3）— 可能回歸或可能持續？AI 判斷，**不自動 ±run value**

（無 Flag 觸發時：「無風險提示」單行）

## File 索引
- merged.json (Phase 4 機讀): `analysis-data/{date}/{a}@{h}/merged.json`
- phase3 寫作: `analysis-data/{date}/{a}@{h}/phase3_skeleton.md`
- 個別 detail summary（drill-down / debug）: 同目錄下 `<basename>_summary.md`
```

### 4.3 渲染規則

- 數值四捨五入規則跟原 summary 一致（ERA/FIP 2 位小數、velo 1 位、% 1 位）
- ⚠️ emoji 只在 Flag 真的觸發時顯示，避免 alert fatigue
- Tier 顯示**腳本判斷值** — AI 仍可在 phase3_skeleton.md 覆寫，但 dossier 不做覆寫
- Top 5 候選池：`active && PA ≥ 30 && !IL`；不夠 5 人就照數量輸出
- last7 OPS top1 註腳：候選池內取 last7 OPS 最高者；若已在 PA top 5 或候選池為空，省略

---

## 5. `phase3_skeleton.md` 結構

### 5.1 設計原則

- 預先寫好所有 H2 必須 section（不再倚賴 `predict.py` grep guard）— 解決 P4
- 預填可機械算的數值表（牛棚 IL、Park Factor、base 比分）
- AI 只需補「結論段落」與「整體判斷」
- 對比舊流程：AI 從零寫 ~150 行 → 改為填空 ~30 行

### 5.2 範本

```markdown
# Phase 3 Summary — {AWAY} @ {HOME} ({date})

## 投手對決

### {HOME pitcher} (HOME, {hand}, {age} {age_emoji})
- **Tier 覆寫**：<!-- AI 補：覆寫 + 理由 / 或「沿用腳本 {tier}」 -->
- 真實水平判斷：<!-- AI 補：基於 ERA/xERA/FIP/Statcast/年齡綜合 -->
- 對手打線威脅：<!-- AI 補 -->

### {AWAY pitcher} (AWAY, {hand}, {age} {age_emoji})
（同上）

## 打線評級

### HOME — {tier} / {heat}
- **Tier 覆寫**：<!-- AI 補：覆寫 + 理由 / 或「沿用腳本」 -->
（AI 摘要 + 補主威脅 / 黑洞 list）

### AWAY — {tier} / {heat}
（同上）

## 牛棚

| | HOME | AWAY |
| ERA / IL 數 / 核心 IL 估計 | ... | ... |

### 牛棚雙向修正值
- HOME 牛棚：對手 +{?} run | HOME ML {?}%
- AWAY 牛棚：對手 +{?} run | AWAY ML {?}%
<!-- AI 補：填入修正值，依 matchup-factors.md §牛棚傷兵累計效應 -->

## 風險提示

<!-- prepare_game.py 預填以下偵測到的 Flag，AI 在每條下方補敘事段落 -->
- ⚠️ AWAY 投手 Flag 13 (era_xera_delta=-2.54)：
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.241)：
  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->

（無 Flag 觸發時：「無風險提示」單行）

## 條件修正

- Park Factor: {pf} → +{(pf-100)*0.05} run
- 雙方先發 tier: <!-- AI 補：是否觸發 -1.0 / -0.5 投手戰 -->
- 其他（doubleheader / platoon / 休息日 / 天氣）: <!-- AI 補 -->

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
| HOME | {從 predict.py dry-run 取得} | <!-- AI 補 --> | <!-- AI 補 --> |
| AWAY | ... | ... | ... |
| Total | ... | ... | ... |

## 整體判斷

- **方向（基本面）**：<!-- AI 補 -->
- **總分（基本面）**：<!-- AI 補 -->
- **信心**：<!-- AI 補 LOW/MEDIUM/HIGH -->
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain（仍由 SKILL.md SOP 強制）：星級、明確盤口推薦
```

### 5.3 「永遠存在的 H2」清單

7 個 H2 全部永遠存在：
- `## 投手對決`
- `## 打線評級`
- `## 牛棚`
- `## 風險提示`（無 Flag 時內文「無風險提示」）
- `## 條件修正`
- `## 修正後預期得分`
- `## 整體判斷`

刪除原 spec 的 `## YoY 對比結論` / `## BABIP 回歸判定` 兩個 H2 — luck-based 邏輯整段移除。

---

## 6. 上游腳本配套修改

### 6.1 `fetch_game_data.py` — 加 probable_pitcher_id（解 P2）

現況：
```python
"home": {"team": "...", "team_id": 114, "probable_pitcher": "Tanner Bibee"}
```

改為：
```python
"home": {"team": "...", "team_id": 114, "probable_pitcher": "Tanner Bibee", "probable_pitcher_id": 676440}
```

ID 來自 schedule API 的 `hydrate=probablePitcher` response（`teams.{home,away}.probablePitcher.id`）— 已在現有 API 呼叫範圍內，只是沒存。

`game_data_summary.md` 同步在「先發」行加 ID：`Nick Martínez (TB, 607259) vs Tanner Bibee (CLE, 676440)`。

### 6.2 `pitcher_stats.py` — Diacritic fallback（解 P3）

現況：`--name "Nick Martinez"` → strict match → 找不到 → exit 1。

改為：
1. 第一輪 strict match
2. 失敗 → 自動跑 `playerid_lookup(last, first, fuzzy=True)` 取最高相似度結果
3. 若 fuzzy 結果 `mlb_played_last` 為當前年或前一年 → 接受並 stderr warning：`⚠️ name "{input}" matched fuzzy → "{matched_name}" (mlbam={id})`
4. 否則 → exit 1 同現況

**單元測試需 cover**：
- ASCII → Unicode 對應（`Nick Martinez` → `Nick Martínez`）
- 多重相似名字（取最高相似 + 年份過濾）
- 完全不存在的名字仍 exit 1

### 6.3 `predict.py` — 兩項修改

#### 6.3a `--ou-stars` 必填化（解 P5）

現況：`--ou-rec OVER` 但漏 `--ou-stars` → 自動降為 PASS + stderr 警告。

改為：當 `--ou-rec` 不是 `PASS` 時，`--ou-stars` 必填；缺則 hard error exit 6 + 訊息：`--ou-rec=OVER/UNDER 必須同時提供 --ou-stars (0-5)`。

合法 PASS（PASS 時 stars 預設 0）不受影響。

#### 6.3b 移除 H2 grep guard

原 `predict.py` 會 grep `phase3_summary.md` 內的 H2 sections（`## YoY 對比結論` / `## BABIP 回歸判定` 等）來確認結構完整。新邏輯：

- **不再 grep 個別 H2 內容**
- **保留**：`phase3_summary.md` 必須存在（檔案層 reject）
- **保留**：星級 / 盤口字串硬擋（Phase 3 spec 的 MUST NOT contain）

理由：H2 大砍 + skeleton 預填完整骨架，AI 漏寫個別 section 的可能性接近零；維護 grep 規則的成本超過效益。

---

## 7. Reference 瘦身與 SOP 變更

### 7.1 `SKILL.md`（83 → ~150 行）

合併 `workflow.md` 內容後新增章節：

- **初始化**：Python 偵測、`$GAME_DIR` 設定、模式切換規範（核心數據禁 WebSearch、shell redirect 禁用、隊伍縮寫格式）
- **Phase 1+2 SOP**：1 行 `prepare_game.py` 命令 + Read dossier.md 流程
- **Phase 3 SOP**：補 `phase3_skeleton.md` → 存 `phase3_summary.md` 流程；MUST contain / MUST NOT contain 清單
- **Phase 4 SOP**：`predict.py --save` CLI 參數表、4.7 輸出前驗證 checklist
- **資料來源優先順序**：1 行紀律「API > 官網 > 第三方」（從 teams-and-api.md 搬）

「Quick Reference」表格收斂為：

| Phase | 主要產出 |
|-------|---------|
| 1+2. 資料收集 | `merged.json` + `dossier.md` + `phase3_skeleton.md`（`prepare_game.py`） |
| 3. 綜合分析 | `phase3_summary.md`（在 skeleton 上補結論） |
| 4. 預測輸出 | `prediction.json` + `prediction_summary.md`（`predict.py`） |

### 7.2 `reference/workflow.md` — 整檔刪除

內容全部併入 `SKILL.md`（見 §7.1）。

### 7.3 `reference/teams-and-api.md` — 整檔刪除

- 隊名表已在 `scripts/_team_resolver.py`（`TEAM_MAP` / `FULL_NAMES`）
- API 端點已在各腳本實作中（`fetch_game_data.py` 等）
- Pythagorean Win% 已在 `predict.py` 內建
- 「資料來源優先順序」1 行紀律 → 移到 `SKILL.md`「初始化」段

### 7.4 `reference/flags-checklist.md`（65 → ~45 行）

- **刪除 Flag 7**（Roster 跳過）— `prepare_game.py` 整合後 Step B 失敗會 exit 5，AI 無從跳過
- **改寫 Flag 13**（\|ERA-xERA\| ≥ 1.5）：原「補跑 prior year 做 YoY 5 指標對比」 → 新「腳本標 ⚠️ 風險提示，AI 在 phase3_skeleton.md 風險提示段判斷，不自動補跑、不自動下修」
- **改寫 Flag 3**（last7 BABIP 極端）：原「不扣 Cold/加 Hot run value」 → 新「腳本標 ⚠️ 風險提示，AI 在 phase3_skeleton.md 風險提示段判斷，不自動 ±run value」
- 其他 11 條（Flag 1/2/4/5/6/8/9/10/11/12）KEEP 原樣

### 7.5 `reference/matchup-factors.md`（182 → ~100 行）

- **刪除 §YoY Statcast 驗證 整段**（27-46 行，含 5 指標表 + 三條判定規則）— 不再做 YoY 補跑
- **改寫 §BABIP 回歸檢查**（72-77 行）：刪「不扣 Cold/加 Hot run value」邏輯，改為「dossier 標 ⚠️ 風險提示，AI 敘事提及，不動 run value」
- **刪除 §影響分析的賽制規則**（180-182 行：DH / Pitch Clock / 三打者規則）— 對單場修正幾乎無影響
- 其他 KEEP / SLIM：投手核心指標、Tier 表、牛棚累計效應、傷兵過濾、TJ 復出、年齡退化、Park Factor

### 7.6 `reference/prediction.md`（249 → ~170 行）

- **刪除 §比賽敘事觸發條件 整段**（119-127 行）— 前端只顯示推薦，不顯示比賽敘事
- **刪除 §預測紀錄格式 JSON schema**（210-246 行）— AI 看 `prediction_summary.md` 不看 JSON；要 debug 直接看 `predict.py` 源碼
- **SLIM §預測紀錄存放位置**（保留 1 段話：per-game / per-date 兩層）
- 其他 KEEP：比分公式、信號 Run Value 表、O/U/ML/RL 星級、讓分交叉驗證、D1-D5 紀律、PASS 門檻 + 護欄

### 7.7 廢除 B7 / B9 / B10 TaskCreate（forcing function）

`workflow.md` 整檔刪除 → 內含 B7（YoY 補跑）/ B9（牛棚雙向）/ B10（BABIP 回歸）TaskCreate 模板自然消失。

新 `SKILL.md` 不再包含任何 TaskCreate forcing function。

理由：腳本 + skeleton 已預填完整骨架，AI 在 skeleton 內漏寫的可能性接近零；TaskCreate 的「forcing function」失去意義。

---

## 8. AI 必須保留的判斷

| 環節 | 為何保留 |
|---|---|
| **Tier 覆寫** | 例：Martinez 腳本 🟠（基於 ERA 2.10）但 xERA 4.64 → AI 必須降為 🟢。需 xERA / 年齡 / velo / Statcast 綜合判讀 |
| **signal_adjustments 的具體大小** | 例：`martinez_era_xera_gap` +0.3 還 +0.6，是 sanity check 的核心 |
| **Flag 13 風險提示的判斷** | 投手 ERA-xERA gap 是運氣或結構性退化？AI 在敘事處理（**不自動補跑 YoY**） |
| **Flag 3 風險提示的判斷** | 打線 last7 BABIP 偏離是回歸還是持續？AI 在敘事處理（**不自動 ±run value**） |
| **BvP 個人 vs 團隊判定** | 球員層面偏離不對應團隊整體 |
| **ML / OU / RL 推薦方向 + 星級** | D1-D5 紀律的最後守門 |
| **市場 vs 模型分歧的判讀** | 純判斷題，AI 的核心 alpha |
| **最終敘事與報告** | 風格、解釋深度 |

刪除原 spec 的「YoY 結構性 vs 樣本噪音分類」「BABIP 回歸個人 vs 團隊判定」 — 這兩條都隱含了「期待回歸後 ±run value」的 luck-based 邏輯，已不在判斷流程內。

---

## 9. 驗收條件

### 功能性

- [ ] `prepare_game.py` 對 2026-04-28 TB@CLE 跑通，產出 18 個檔案
- [ ] `dossier.md` 行數 ≤ 250
- [ ] `phase3_skeleton.md` 行數 ≤ 50；含 7 個 H2（包括 `## 風險提示`）
- [ ] `fetch_game_data.py` 輸出含 `probable_pitcher_id`（home + away）
- [ ] `pitcher_stats.py --name "Nick Martinez"`（無重音）能 fuzzy match 到 `Nick Martínez` 並產出資料
- [ ] `predict.py --ou-rec OVER` 漏 `--ou-stars` 時 hard error exit 6
- [ ] `predict.py` 不再 grep H2 sections（仍要求 `phase3_summary.md` 存在）
- [ ] `reference/teams-and-api.md` 已刪
- [ ] `reference/workflow.md` 已刪
- [ ] `reference/flags-checklist.md`：剩 11 條（刪 Flag 7），Flag 3/13 改寫為「標註 + AI 判斷」
- [ ] `reference/matchup-factors.md`：YoY Statcast 驗證 / 賽制規則 已刪；BABIP 改為風險標註
- [ ] `reference/prediction.md`：比賽敘事觸發 / JSON schema 已刪
- [ ] `SKILL.md`：合併 workflow 內容，總行數 ≤ 200
- [ ] 所有腳本 / reference 修改通過 unit tests

### 回歸測試

**跳過 4 場歷史比賽回歸**（使用者決定整套實作完後重跑當日比賽自行驗證）。

### Token 量化

實測 2026-04-28 TB@CLE：
- 舊流程 Phase 1+2 AI 端 Read 字數
- 新流程 Phase 1+2 AI 端 Read 字數
- **目標：減少 ≥ 60%**（原 spec 50%，此版瘦身更深）

---

## 10. 開發順序

| 階段 | 任務 | 預估 |
|---|---|---|
| 0 | **Reference 瘦身**：刪 teams-and-api.md / workflow.md，改寫 flags-checklist.md / matchup-factors.md / prediction.md，重寫 SKILL.md | 1d |
| 1 | `fetch_game_data.py` 加 `probable_pitcher_id` + test | 0.5d |
| 2 | `pitcher_stats.py` diacritic fallback + test | 0.5d |
| 3 | `predict.py` `--ou-stars` 必填化 + 移除 H2 grep guard + test | 0.5d |
| 4 | `prepare_game.py` Step A-E 整合（無新檔案）+ test | 1d |
| 5 | `dossier.md` 渲染器（含 ⚠️ 風險提示）+ test | 1d |
| 6 | `phase3_skeleton.md` 渲染器（含 `## 風險提示` H2）+ test | 0.5d |
| 7 | 2026-04-28 TB@CLE 實測 + token 量化 | 0.25d |

**總計：約 5.25 天**

---

## 11. 決議紀錄（原開放問題）

全部結案：

| # | 議題 | 決議 |
|---|---|---|
| Q1 | dossier.md Top 5 排序 | PA 排序 + last7 OPS top1 註腳；PA ≥ 30 floor；IL'd 排除 |
| Q2 | Tier 覆寫紀錄位置 | 不入 prediction.json；phase3_skeleton.md 預留 `**Tier 覆寫**:` slot |
| Q3 | Doubleheader dossier | 分檔（`dossier-G1.md` / `dossier-G2.md`，phase3_skeleton 同例） |
| Q4 | `*_summary.md` 是否仍產生 | 預設產生；不加 `--skip-detail-summaries` flag（YAGNI） |
| Q5 | Step C-prior 處理 | **整段刪除** — 不再做 YoY 補跑；Flag 13 改為風險提示，AI 敘事判斷 |

額外決議（2026-04-29 brainstorm 擴充）：

| # | 議題 | 決議 |
|---|---|---|
| E1 | reference 瘦身範圍 | 刪 teams-and-api.md / workflow.md；改寫 flags-checklist / matchup-factors / prediction |
| E2 | teams-and-api 處置 | 整檔刪 — 內容已在 `_team_resolver.py` 與各腳本內 |
| E3 | prediction.md 比賽敘事段落 | 刪 — 前端只顯示推薦，不顯示比賽敘事 |
| E4 | matchup-factors / flags-checklist luck-based 邏輯 | 全砍 — Flag 3/13 改為風險標註，不自動補跑 / ±run value |
| E5 | workflow.md 是否保留 | 不保留，併入 SKILL.md |
| E6 | prediction.md JSON schema 移到哪 | 整段刪 — 真要 debug 看 `predict.py` 源碼 |
