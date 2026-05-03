# Follow-up backlog

Path B refactor（branch `refactor/path-b-signals`，18 commits）完成後留給下一個 session 的事。優先序由上往下。

## 1. Stuff+ / Pitching+ 接進 tier_v2（高 ROI）

### 動機

審計時點名 tier_v2 缺少 stuff-quality 維度。現在 tier_v2 用 xFIP / K-BB% / velo / age — velo 只是物理特徵單點，無法捕捉「球質好不好」。Stuff+ 是 FanGraphs 的合成 stuff metric（velo + spin + movement），100 = 聯盟平均；Pitching+ = Stuff+ × Location+ 是更全面的「pitcher 真實水平」指標。

特別在 4 月小樣本期（IP < 30 → tier_v2 fallback v1 ERA-only）有用：Stuff+ 是物理特徵推算，第 1 場就能用。

### 步驟

1. **驗證 pybaseball column 可用性**（5 分鐘）
   ```python
   from pybaseball import pitching_stats
   df = pitching_stats(2025, qual=50)
   print([c for c in df.columns if "+" in c])
   # 預期：Stuff+, Location+, Pitching+
   ```
   若 column 命名不一致，看 pybaseball source 找實際欄位名稱。

2. **加 `fetch_stuff_pitching_plus(mlbam_id, year)`**：用 leaderboard pattern（同 `fetch_pitch_arsenal`），記憶體內 filter player_id。回傳 `{stuff_plus, location_plus, pitching_plus}`。

3. **重新加權 tier_v2 公式**：
   ```
   score = 30 × pct(xFIP, lower)
         + 25 × pct(K-BB%, higher)
         + 30 × pct(Stuff+, higher) | 或用 Pitching+
         + 15 × age_factor
   ```
   保留 v1 / 現 v2 / 新 v3 三種 tier？或直接覆寫 v2？傾向 **覆寫 v2**（避免 schema 增生），但 dossier 顯示 components 時加入 stuff_plus row。

4. **新增 baseline.json 欄位**：`league_pitcher_baseline.json` 的 `metrics` 加 `stuff_plus` / `pitching_plus` percentiles。`refresh_baselines.py` 同步更新。

5. **Tests**：12-15 個（pct 邊界 / fallback / score 計算 / dossier render）。

## 2. wRC+ 接進 lineup_analyzer（中 ROI）

### 動機

打線 tier 目前用 xwOBA（主）/ OPS（fallback）。wRC+ 是 park-adjusted offensive metric，100 = 聯盟平均；對跨球場比較較準（Coors 打者不會被 OPS 高估）。

### 步驟

1. **加 `fetch_team_wrc_plus(team_id, year)`**：用 `pybaseball.batting_stats(year)` leaderboard，filter team。回傳 dict `{batter_id: wrc_plus}`。

2. **lineup_analyzer 整合**：每位打者加 `wrc_plus` 欄位，team 層級加 `avg_wrc_plus`。

3. **dossier 打線表新增一列「avg wRC+」**（HOME / AWAY）。

4. **Tests**：6-8 個。

## 3. Bug 3：role tagging prior_year fallback（修正 5/02 BAL@NYY 漏抓 Bautista）

### 動機

5/02 BAL@NYY 整合驗證發現：長傷球員（整季 IL，例如 Bautista）當季 stats = 0/0/0/0/0 → `tag_role` 判 Unknown，core_il_count 漏算。手寫版識別 BAL core IL = 2（Bautista + Helsley），新 pipeline 只抓到 1。

### 步驟

1. **`fetch_pitcher_season_stats_bulk` 加 prior_year 抓取**：當季 G < 5 時 fallback 抓上季 stats。
2. **`tag_role` 加 `from_prior_year` flag**：confidence 標記 "data, prior_year"。
3. **Tests**：3-4 個（長傷 case / 新人沒上季資料 case）。

## 4. Bug 4：ERA_ONLY_SCORE_MAP linear interpolation（修正邊界跳躍）

### 動機

Bradish ERA 5.03 → score 15（Below）；ERA 4.99 → 35（Back-end）。0.04 ERA 差跳 20 分 → tier_gap +38.8 看起來誇張但其實是邊界不連續。

### 步驟

1. **`compute_era_only_score(era)`**：linear interpolation between bucket midpoints。
   ```
   ERA   2.0 → 95   (Elite 上緣)
   ERA   2.50 → 90  (Elite 中)
   ERA   3.20 → 75  (Strong 中)
   ERA   4.20 → 55  (Solid 中)
   ERA   5.00 → 35  (Back-end 中)
   ERA   6.00 → 15  (Below 中)
   ```
   兩 anchor 之間線性內插。
2. **`compute_tier_gap`** 接這個 function 取代 ERA_ONLY_SCORE_MAP table lookup。
3. **Tests 更新**：原 ERA_ONLY_SCORE_MAP tests 改測 linear interpolation behavior。

---

# Cleanup pass findings（2026-05-03 review session）

以下 9 條由本日 4-agent code review 留下，按 severity 排。詳見 `plans/glittery-stargazing-moth.md`。

## 5. dossier `_render_bullpen_park` 重複計算 IL count（HIGH）

### 動機
`dossier_renderer.py:782-849` 用 substring filter (`"pitcher" / "p"`) 重新算 IL 數，但 `merge_game_data` 已把 `{side}_core_bullpen_il_count` 寫進 bundle。重複計算且分類條件不一致（dossier 用位置字串 vs merge 用 `core_role`）。

### 步驟
1. `_render_bullpen_park` 讀 `merged.{side}_core_bullpen_il_count` 取代 substring filter。
2. 名單列表用 `core_role ∈ CORE_BULLPEN_ROLES` 篩（從 `lib_role_tagging` import）。
3. Tests：3-4 個（混合 core / non-core IL 場景）。

## 6. `fetch_pitcher_season_stats_bulk` 平行化（HIGH）

### 動機
`roster_checker.py:124-155` per-pitcher sequential `requests.get`。每隊 ~13 隻投手 × 2 隊 = 26 round-trips on critical path of step_b。

### 步驟
1. `ThreadPoolExecutor(max_workers=8)` 包 per-pid 迴圈。
2. 保留現有 `try/except` 個別失敗略過邏輯。
3. Tests：1-2 個（並行不影響輸出）。

## 7. `compute_all_signals` 重複計算（MED）

dossier + summary 各跑一次同樣的 signals。建議 `prepare_game` 算一次往下傳 `bundle["signals"]`。

## 8. `_arsenal_top3_str` 重新過濾（MED）

`dossier_renderer.py:557-568` 重新過濾 arsenal，但 `merge_game_data.extract_pitcher_nested:91` 已輸出 `arsenal_top`（pre-filtered top-3）。改讀 pre-filtered 欄位即可。

## 9. schema 命名一致 `pitcher_hand` vs `pitch_hand`（MED）

JSON 欄位是 `pitch_hand`；`signals_lib` / `lineup_analyzer` 函式參數叫 `pitcher_hand`。建議統一 `pitch_hand` 並更新 call sites。

## 10. `lib_tier_v2` unreachable 防呆（MED）

lines 83 / 100：`return 0.5  # defensive fallback` 在 clamp 之後實際走不到。改 `raise AssertionError` 並收緊 invariant 訊息。

## 11. `summary_renderer._lineup_block` lookup table（MED）

lines 75-97 三分支 `if/elif/else` on `opposing_pitcher_hand` → 改 `_HAND_TO_KEYS = {"L": ("tier_vs_lhp", "vs LHP"), "R": ("tier_vs_rhp", "vs RHP")}` lookup，扁平化分支。

## 12. `merge_game_data` 並行 fetch（LOW）

`fetch_bullpen_era × 2 + fetch_weather × 1` 序列。3 round-trips 可 `ThreadPoolExecutor(max_workers=3)` 並行（保留各自 try/except fallback）。

## 13. `summary_renderer` 同 package `try/except ImportError → lambda` 是 dead code（LOW）

142 / 146 行的 fallback lambda 在 same-package import 不會觸發，是 dead defensive code。`dossier_renderer.py:462-465 / 672-675 / 866-869 / 891-894 / 896-900` 也有重複 5 次的相同模式。建議全部刪 try/except，讓 ImportError 自然向上拋。

---

## 開頭 prompt 範本（給下一個 session）

```
Path B refactor 已完成（branch refactor/path-b-signals，18 commits / 382 tests）。
2026-05-03 cleanup pass commit 1f0d744（去重 + stale bug markers + bundle share）。
看 docs/follow-up-backlog.md 按優先序處理 Stuff+ / wRC+ / Bug 3 / Bug 4 / Cleanup 5-13。
從 #1 Stuff+ 開始 — 看 plans/glittery-stargazing-moth.md §Item 2 的設計。
```
