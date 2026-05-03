# CHANGELOG

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
- **TTO3 penalty / 休息天數 / 上一場用球數** — 第二批 signals
- **Park HR PF L/R split** — 目前 strong_park 用整體 PF；LR split 需擴充 `data/park_factors.json`
- **wRC+ / Stuff+** — FanGraphs API non-free，不引入
- **Backtest framework** — 與 4/23「走嚴格 formula 對比實際結果」紀律配套，但需獨立規劃
