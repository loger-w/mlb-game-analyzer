# CHANGELOG

## 2026-05-04 — Path B 採用 + 場景路由 + 反身性收斂

### 動機

`mlb-game-analyzer` skill 深層 review 結論：
- **epistemic 防衛已過度**（"不自動 ±run value" 在 codebase 重複 ~18 次）；
- **decision-support 投資不足**（formula 70 行、AI `+ 信號` 欄無 magnitude 錨點）。

決定採 **Path B**：formula 維持簡單 baseline 當 sanity rail，AI 用「量級錨點 + 機率信心」主導 magnitude judgment（公開 ML 模型贏不過，AI narrative 是差異化 edge）。同時加 **場景路由** 解決「fundamentals vs odds 邊界、idempotence 重跑」痛點。

### 落地（6 commits + 1 chore）

#### 場景路由（SKILL.md）
- `docs(skill)`: 加 `## 場景路由` Step 0-4（intent / state probe / routing 矩陣 / idempotence / force override）
- `docs(skill)`: 加 `## 步驟 3：盤口分析`（odds_only / both 路徑；引 `odds/reports/{date}.md`）
- `docs(skill)`: drill-down hint 從「8 個檔列舉」改成「何時 Read」清單

#### Path B 量級錨點 + 半量化信心
- `docs(reference)`: `matchup-factors.md §量級錨點` — 9 signals ±run 區間 + 累積規則（cap ±0.8 / 場）+ calibration 路徑
- `feat(summary)`: `_render_overall_section` 信心改 % 機率（取代 LOW/MED/HIGH）+ 方向 / 總分 placeholder 細化
- `feat(summary)`: `_render_expected_runs_section` caveat 連 §量級錨點

#### 反身性收斂（避免 ~18 處重複）
- `docs(reference)`: `flags-checklist.md §Signals` trim 成 4 行 pointer（與 `matchup-factors.md §Signals` 不重複）
- `refactor(dossier)`: `_FLAG8_TAIL` / `_FLAG3_TAIL` 模組常數 + `_flag8_pitcher_lines` / `_flag3_lineup_lines` helper（Single source of truth）
- `refactor(summary)`: `_FLAG8_AI_PLACEHOLDER` / `_FLAG3_AI_PLACEHOLDER` 模組常數

#### 簡化 baseline（Path B 對齊）
- `refactor(formula)`: `scoring_formula.py` 70 行 → 39 行；刪 dead `log5()` / `pythagorean_runs()` / 對應 dict keys（無 caller 依賴）；docstring 改寫對齊 Path B 「formula = sanity rail」

#### Legacy 清除
- `chore(data)`: 刪 62 個 pre-refactor 殘留檔（phase3_skeleton/summary + prediction.json/summary，跨 20 個 game dirs）

### 紀律保留

- ✅ 既有 9 signals 行為零變動（compute_all_signals 邏輯完全不動）
- ✅ Flag 3/8 stderr 顯示維持只列 flag（`_print_risk_notes` 不動）
- ✅ summary 7 個 H2 不變、line count ≤ 85 仍通過
- ✅ `predict_with_formula` return shape 保留 `home_score / away_score / total`（dossier_renderer 與 summary_renderer 既有 caller 全部 OK）
- ✅ Path A 的 sanity-check 優點以「formula 當 guardrail」保留：AI adjusted vs formula base 差距 > 1.5 run 時 AI 必須在風險段解釋

### Tests

469 → 471 passing（既有測試零修改全綠；refactor 路徑由 existing tests 覆蓋；無新增測試）。

---

## 2026-05-03 — TTO3 penalty signal（signal #9，Plan B）

第 9 個 derived signal，pitcher-side per-game。先發投手第三輪面對打者 OPS
衰退幅度，覆蓋 PR-3 後 line 48「第二批 signals」第一項。Plan A（MLB API
statSplits + sitCodes）spike 後證實 MLB API 不曝光 TTO 切面；改走 Plan B 用
pybaseball Statcast pitch-by-pitch 自行依 (game_pk, batter) 分組聚合。

- **commit 941f119** `docs(spec)`: brainstormed design — signal contract / threshold / surface
- **commit ac97719** `docs(plan)`: implementation plan（12 tasks → 後改 10）
- **commit 2d7f0ab** `docs(spec/plan)`: Plan B amendment after spike disproved Plan A
- **commit f7c84a0** `feat(pitcher)`: `_pa_outcome_aggregates` helper (PA events → OPS/K%/BB%)
- **commit 7606e11** `refactor(pitcher)`: Task 2 cleanup — type hint + drop dead defense
- **commit e8f317d** `feat(pitcher)`: `_compute_tto_from_statcast` (Plan B Statcast aggregation)
- **commit 47601f0** `feat(pitcher)`: `fetch_tto_splits` orchestrator + main 路徑接入
- **commit e551b70** `feat(signals)`: `signal_tto3_penalty` (#9) + half_life=structural
- **commit 8afa407** `feat(signals)`: wire tto3_penalty into `compute_all_signals` per-pitcher loop
- **commit dc17593** `feat(dossier)`: 投手對決 table 加 TTO splits visible row
- **commit 58aeae5** `docs(reference)`: matchup-factors §Signals §9 + 半衰期表

### 紀律保留

- ✅ 信號**不入 scoring formula**（一致 §3 / §8）
- ✅ 既有 8 signals 行為零變動（compute_all_signals 只追加一行）
- ✅ 4 月小樣本 season → 5-year career silent fallback；BF < 30 統一 small_sample no_fire
- ✅ Dossier TTO row 無條件顯示（mirror vs LHB / vs RHB pattern）
- ✅ `merge_game_data.py` / `prepare_game.py` / `scoring_formula.py` / Flag 體系全部不動

### Tests
439 → 469（+30：22 個 pitcher_stats helpers + orchestrator + 11 個 signals + 6 個 dossier，+ Task 2 cleanup tests）。

### Out of scope（下批）

- TTO4+ penalty（樣本太稀）
- Reliever inheritance penalty
- 動態調整觸發閾值（按 tier 別）— 留至 backtest 階段
- 休息天數 / 上一場用球數（line 48 第二批 signals 中的另兩項）

---

## 2026-05-03 — Path B refactor (`refactor/path-b-signals`)

把腳本貢獻從「資料 plumber 80%」抬到「指標 50%」。AI 從「合成」轉「驗證 / 判讀」。3 PRs / 17 commits / +160 tests（既有 230 全綠 → 390 total）。詳見 `docs/superpowers/specs/`（plan 在 `~/.claude/plans/quizzical-brewing-snowflake.md`）。

### PR-1: Foundation（純加性，4 commits）

- **commit 0bff2b6** `data(baseline)`: 加 `data/league_pitcher_baseline.json` + `refresh_baselines.py` 年度更新工具 + `data/README.md`
- **commit 065da85** `feat(pitcher)`: 加 `fetch_pitch_arsenal()` 用 `pybaseball.statcast_pitcher_arsenal_stats` leaderboard
- **commit 252abd1** `feat(pitcher)`: persist `arsenal: list[dict]` in JSON + `format_md` 加 `## Pitch Arsenal (RV/100)` 段
- **commit f1a1f90** `feat(merge)`: `extract_pitcher_nested` 加 `arsenal_top` top-3 pass-through

### PR-2: Tier reform + Bullpen tagging（5 commits）

- **commit 1af443d** `feat(pitcher)`: `compute_tier_v2` blended formula（xFIP 40 / K-BB% 35 / velo 15 / age 10）
- **commit 901420d** `feat(pitcher)`: `tier_gap` field surface tier_v2 vs ERA-only delta
- **commit 7ab3d49** `feat(lineup)`: `tier_vs_lhp` / `tier_vs_rhp` re-aggregation
- **commit 7b650e8** `feat(roster)`: `core_role` heuristic（Closer / Setup / High-leverage / Co-Closer / Opener）
- **commit 43df593** `feat(merge)`: `bullpen_core_il_count` from injured_list

### PR-3: signals_lib + dossier 重組 + summary placeholder rewrite（8 commits）

- **commit 447df66** `feat(signals)`: signals_lib batch 1（tier_mismatch / heat_vs_babip / platoon_advantage / strong_park）
- **commit c2605f7** `feat(signals)`: signals_lib batch 2（reverse_platoon / chain_break / pitch_mix_concentration / core_il_count）
- **commit c312f6e** `feat(signals)`: `compute_all_signals(bundle)` aggregator
- **commit 24ae2cc** `feat(dossier)`: 加 `## 🎯 訊號摘要` 段（在 `## 投手對決` 之前）
- **commit 620931d** `feat(dossier)`: 投手對決加 4 列 + 折疊既有 13 列入 `<details>`
- **commit 73f8a07** `feat(summary)`: placeholder 從「合成」改「驗證 signal X」
- **commit 867fad0** `feat(summary)`: `## 風險提示` 段尾追加 `### 額外信號`
- **commit (this)** `docs(reference)`: `flags-checklist.md` 加 §Signals + `matchup-factors.md` §Signals + CHANGELOG

### 紀律保留

- ✅ 既有 `tier`（v1 ERA-derived）完全不變，`tier_v2` 新增為平行欄位
- ✅ `_print_risk_notes` stderr 只顯示 Flag 3/8（信號不污染 CI log）
- ✅ Signals **不入 scoring formula**（一致 §3 / §8「不自動 ±run value」紀律）
- ✅ Dossier `<details>` 折疊保留所有既有 substring，向下相容
- ✅ `pitchers: list[str]` schema 維持向下相容（新增 `pitcher_ids` / `pitcher_roles` 平行）

### 年度維護任務

- **每年 2 月開季前**：跑 `python scripts/refresh_baselines.py --year {LAST_SEASON} --output scripts/data/league_pitcher_baseline.json` 並 commit。避免 silent baseline drift（季中 tier_v2 會與 dossier 歷史紀錄不一致）。

### Out of scope（PR-3 後可考慮）

- **Composite leverage score**（Path C） — 待 PR-3 上線兩週後 backtest 評估
- **休息天數 / 上一場用球數** — 第二批 signals 剩餘項（TTO3 penalty 5/3 已上線）
- **Park HR PF L/R split** — 目前 strong_park 用整體 PF；LR split 需擴充 `data/park_factors.json`
- **Backtest framework** — 與 4/23「走嚴格 formula 對比實際結果」紀律配套，但需獨立規劃
