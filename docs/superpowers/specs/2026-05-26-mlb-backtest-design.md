# MLB Skill 回測設計 — 2026-05-26

## 1. 目的

驗證 mlb-game-analyzer skill 在 2026 年 5 月實際比賽上的預測表現，輸出可重跑的腳本 + 報告，作為 skill 後續迭代的「燃料」（哪些情境穩、哪些有 systematic bias、信心分檔是否 calibrated）。

**不在 scope**：
- odds reports Watch 軌的回測（軌 B，留另一份 spec）
- 下注 EV / ROI 模擬
- 自動 regression detection（push 到 git 就跑那種）
- 信心 mapping 第二輪實測反推 + 重跑（屬 follow-up，跑完 v1 看結果再決定要不要做）

## 2. 範圍與資料

- **時間範圍**：2026-05-01 ~ 2026-05-25（22 個有資料的日期；5/5、5/11、5/14 缺）
- **樣本**：284 場 summary.md 中，「整體判斷」段已被 AI 填寫的 **136 場**（解析時動態判定，不寫死數字）
- **預測欄位來源**：每場 `analysis-data/{date}/{matchup}/summary.md` 的「整體判斷」段
  - 方向：HOME / AWAY / 持平（從 `**方向（基本面）**：...` 行抽前綴關鍵字）
  - 總分：`**總分（基本面）**：adjusted X.X` 的數值
  - 信心：`**信心**：LOW / MEDIUM / HIGH` （三檔之一）
- **市場 baseline**：Pinnacle no-vig 收盤線
  - 來源：`odds/snapshots/{date}/` 內的 snapshot JSON
  - 「收盤線」定義：對每個 `game_pk`，取 `commence_time` 之前**最後一個** Pinnacle pre-game snapshot 的 ML / Total 線
  - 取出後跑 no-vig 換算（兩側機率歸一）
- **比賽結果**：`scripts/fetch_results.py`（從 git commit `3c1cd89` 撈回），跑全 5 月，輸出寫到 `analysis-data/{date}/{matchup}/result.json`
  - 主鍵：`game_data.json` 的 `gamePk`
  - 欄位：`{ game_pk, winner: "HOME"|"AWAY", final_score: [home, away], home_score, away_score, total, status, postponed }`

## 3. Pipeline 架構

```
[Stage 1] 補比賽結果
  git show 3c1cd89:scripts/fetch_results.py → 撈回 + 跑全 5 月
  輸出: analysis-data/{date}/{matchup}/result.json

[Stage 2] 拼接資料 (scripts/lib/load.py)
  遍歷 analysis-data/2026-05-*/{matchup}/
  剔除：「整體判斷」段為模板狀態 (含 <!-- AI 補 HOME / AWAY -->) 的場次
  輸出: pandas DataFrame (一場一列)

[Stage 3] 指標計算 (scripts/lib/metrics.py)
  方向 / 總分 / Calibration / 切片
  輸出: dict + DataFrame

[Stage 4] 失敗案例 diagnostic (scripts/lib/diagnostic.py)
  direction_miss + total MAE top 10 → 結構化清單

[Stage 5] 渲染輸出 (scripts/lib/render.py)
  → analysis-data/backtest/2026-05-report.md
  → analysis-data/backtest/2026-05-details.csv
```

**入口**：`scripts/backtest.py`，subcommand 設計：
- `backtest.py run --month 2026-05`（預設全月）
- `backtest.py run --month 2026-05 --days 2026-05-02`（單日 smoke test）
- `backtest.py run --month 2026-05 --out /tmp/test_out`（自訂輸出位置）

## 4. 指標定義

### 4.1 方向類

| 指標 | 算法 |
|---|---|
| `skill_direction_hit` | `(skill_direction == actual_winner)` 各場布林，平均得命中率 |
| `market_fav_hit` | `(home_winprob_no_vig > 0.5 → "HOME" else "AWAY") == actual_winner`，平均 |
| `skill_edge_pp` | `skill_direction_hit - market_fav_hit`（百分點） |
| `skill_aligned_with_market_hit` | filter `skill_direction == market_fav` 後算 skill 命中率 |
| `skill_against_market_hit` | filter `skill_direction != market_fav` 後算 skill 命中率（**重要**：skill 反盤時是否有 alpha） |

「持平」場處理：skill 預測「持平」時不計入方向類分母（無方向可比）。

95% CI 用 Wilson interval（樣本 < 200，比 normal approx 穩）。

### 4.2 總分類

| 指標 | 算法 |
|---|---|
| `total_mae` | `mean(|skill_total - actual_total|)` |
| `total_bias` | `mean(skill_total - actual_total)` — 正值表 skill 系統性偏高 |
| `total_ou_hit` | `sign(skill_total - pinn_total_line) == sign(actual_total - pinn_total_line)` 的平均 |

排除分母：
- `actual_total == pinn_total_line`（push，line 命中盤口）
- `skill_total == pinn_total_line`（skill 無方向）

### 4.3 Calibration

**信心 → 機率 mapping**（v1 啟動值）：

| 信心檔 | mapped p |
|---|---|
| LOW | 0.55 |
| MEDIUM | 0.62 |
| HIGH | 0.72 |

v1 直接用此 mapping；reliability table 會顯示實測命中率 vs mapped p 的落差，看 mapping 是否合理。**實測反推 + 重跑屬 follow-up**，不在 v1 spec scope。

| 指標 | 算法 |
|---|---|
| `brier_score` | `mean((p_skill - outcome)²)`，outcome ∈ {0, 1} 是 skill 方向是否命中 |
| `log_loss` | `-mean(y·log(p) + (1-y)·log(1-p))`，y = outcome |
| baseline | 相同公式，但 `p` 改用 market no-vig 機率（指 skill 方向那一邊的隱含機率） |

**Reliability table**（最關鍵單表）：

```
信心檔位 | n  | mapped p | 實際命中率 | 落差(實際-mapped) | CI
LOW     | XX | 0.55     | X.XX       | ±X.XX            | [X, X]
MEDIUM  | XX | 0.62     | X.XX       | ±X.XX            | [X, X]
HIGH    | XX | 0.72     | X.XX       | ±X.XX            | [X, X]
```

### 4.4 分組切片

每個切片重跑 `skill_direction_hit` / `total_ou_hit` / `total_mae` / `total_bias`，看 skill 在哪類情境穩：

1. **方向**：HOME / AWAY（skill 預測 HOME 的場次 vs AWAY 的場次）
2. **信心檔位**：LOW / MED / HIGH
3. **Park Factor**：高 (>102) / 中 (98-102) / 低 (<98)，PF 從 `summary.md` 的 `## 條件修正 - Park Factor: X.X` 抽
4. **Reverse platoon 訊號**：有 / 無（summary.md 含 `🟠/🔴 ... reverse platoon Δ` 字串）
5. **Chain break ≥ 0.300**：有 / 無（summary.md 含 `🔴 ... chain breaks at #` 且 OPS 落差 ≥ 0.300）
6. **牛棚 core IL ≥ 2**：有 / 無（summary.md 含 `🔴 ⏳ ... 牛棚 core IL ×2` 或 `×3`）

切片表示意：

```
切片                | n  | dir_hit | ou_hit | mae | bias
HOME 預測          | XX | 0.XX    | 0.XX   | X.X | +X.X
AWAY 預測          | XX | 0.XX    | 0.XX   | X.X | +X.X
信心 LOW           | XX | 0.XX    | 0.XX   | X.X | +X.X
信心 MED           | XX | 0.XX    | 0.XX   | X.X | +X.X
信心 HIGH          | XX | 0.XX    | 0.XX   | X.X | +X.X
PF 高              | XX | 0.XX    | 0.XX   | X.X | +X.X
...
```

切片結果也進 `details.csv` 的 boolean 欄位（has_reverse_platoon 等），人可在 Excel/pandas 自行交叉切。

## 5. 失敗案例 Diagnostic

兩條 ranking 合併：

1. **方向誤判**（限定信心 ≥ MED）：`skill_direction != actual_winner AND skill_confidence in [MED, HIGH]`，全部列出
2. **總分大失誤**：`abs(skill_total - actual_total)` 排序 top 10

合併去重，輸出表格：

| date | matchup | skill 方向 (信心) | 實際勝方 | skill total | 實際 total | 主訊號 | dossier |
|---|---|---|---|---|---|---|---|
| 5/02 | BAL@NYY | NYY (MED) | BAL | 8.5 | 5 | bullpen IL ×3 / Bradish reverse platoon | [link](../2026-05-02/BAL@NYY/dossier.md) |

「主訊號」抽取規則：summary.md「整體判斷 - 方向（基本面）」段第一句逗號／句號前的核心 phrase（regex `^\*\*方向（基本面）\*\*：(.{0,40})[，。]`），給人快速看 skill 當時的判讀依據。

## 6. 報告骨架

`analysis-data/backtest/2026-05-report.md`：

```markdown
# MLB Skill 回測 — 2026 年 5 月

_樣本：N 場（5/01–5/25, 22 天）｜ baseline: Pinnacle no-vig 收盤線 ｜ 生成於 YYYY-MM-DD_

## 資料健康度
- 輸入 summary.md：284
- 通過解析：N (剔出 parse_failed X, template Y)
- 通過 closing snapshot 匹配：N (剔出 closing_missing Z)
- 通過 result 取得：N (剔出 result_missing W)
- **有效樣本：N 場**

## TL;DR
- 方向命中率：skill X.X% vs market X.X%，edge ±X.Xpp（95% CI: ...）
- 總分 MAE：X.XX，bias ±X.XX
- Calibration：[HIGH/MED/LOW 哪一檔最 calibrated]
- 反市場時：skill 命中 X.X%（vs 50% baseline）
- 訊號最強切片：[切片名]，n=XX，dir_hit X.XX

## 1. 方向類指標
[4.1 表]

## 2. 總分類指標
[4.2 表]

## 3. 信心 Calibration
[4.3 Reliability table + 1 段 stub「<!-- 結論待人工填 -->」]

## 4. 分組切片
[4.4 切片表]

## 5. 失敗案例
[第 5 節清單]

## 6. 結論與下一步
<!-- 結論待人工填 -->
```

## 7. CSV 細節（`2026-05-details.csv`）

一場一列。完整 26 欄：

```
date, matchup, game_pk,
skill_direction, skill_total, skill_confidence, skill_prob_mapped,
market_home_winprob_no_vig, market_total_line, market_favorite, market_favorite_winprob,
actual_winner, actual_total, actual_home_score, actual_away_score,
direction_hit, ou_hit, total_abs_error, total_signed_error,
brier_score, log_loss,
park_factor, has_reverse_platoon, has_chain_break_300, has_bullpen_il_2plus,
closing_snapshot_ts, closing_missing, result_missing, parse_failed,
dossier_path
```

布林欄存 `True`/`False`，缺失欄存空字串（不存 `NaN`，避免 Excel 解析雜訊）。

## 8. 邊界處理

| 情況 | 處理 |
|---|---|
| MLB API 拿不到 final score（postponed/canceled） | `result_missing=True`，剔出指標、保留 CSV |
| `summary.md` 解析失敗 | `parse_failed=True`，剔出 + 進報告「資料健康度」section 顯示計數 |
| `commence_time` 之前無 Pinnacle pre-game snapshot | `closing_missing=True`，剔出指標、保留 CSV |
| skill 預測「持平」 | 方向類指標不計入分母 |
| Push（actual_total == line / skill_total == line） | Over/Under 命中率分母不計入 |
| **Doubleheader** | 5 月實證無同對戰目錄衝突；`game_data.json` 有 `gamePk` 主鍵 + `_games_on_date_for_team` 欄位可偵測。若未來出現衝突再迭代 |

## 9. 檔案組織

**新增**：
- `scripts/fetch_results.py`（git restore from `3c1cd89`，可能需小幅修正以對齊當前 repo 結構）
- `scripts/backtest.py`（新寫，入口 + argparse）
- `scripts/lib/load.py`（讀 analysis-data 拼成 DataFrame）
- `scripts/lib/parse_summary.py`（從 summary.md 抽方向/總分/信心 + 訊號 flags）
- `scripts/lib/closing_line.py`（找 commence 前最後 snapshot）
- `scripts/lib/metrics.py`（指標計算）
- `scripts/lib/diagnostic.py`（失敗案例選取 + 主訊號抽取）
- `scripts/lib/render.py`（Markdown 報告 + CSV 寫出）
- `scripts/tests/test_parse_summary.py`
- `scripts/tests/test_closing_line.py`
- `scripts/tests/test_metrics.py`
- `scripts/tests/test_e2e_smoke.py`
- `analysis-data/backtest/`（產出目錄）
- `analysis-data/{date}/{matchup}/result.json`（fetch_results.py 輸出）

**不動既有**：
- 既有 `analysis-data/` schema（result.json 是新檔，不改 summary.md / dossier.md / merged.json）
- 既有 `odds/` schema（只讀，不寫）
- 既有 mlb-game-analyzer skill（這是另一個 skill 的回測，獨立）

## 10. 測試

4 個檔案，fixture-driven：

1. `test_parse_summary.py` — fixture: 5/02 BAL@NYY summary → assert `(AWAY, 8.5, MEDIUM)` + flags 抽取正確
2. `test_closing_line.py` — 合成 5 個 snapshot fixture（含 in-play）→ assert 挑到正確的 pre-game 最後一個
3. `test_metrics.py` — 合成 10 場迷你 dataset → 手算 direction_hit / ou_hit / brier 期望值 → assert 函式輸出一致
4. `test_e2e_smoke.py` — 用 2026-05-02 一天真實資料跑 `backtest.py --month 2026-05 --days 2026-05-02 --out /tmp/test_out` → assert `report.md` + `details.csv` 都產生、row count > 0、必要欄位都存在

不做 mock-heavy unit test（解析 / 指標都是純函式，fixture 就夠）。

## 11. 已知限制

- **136 場樣本量不足以判定統計顯著性**：若實際 edge 是 5pp，需 ~400 場才能 95% 信心；本回測的 CI 多半會跨越 0，結論偏「偵測性」而非「驗證性」
- **「持平」場 skill direction 處理保守**：直接剔出方向類分母；若 5 月「持平」場很多會壓縮有效樣本
- **信心 mapping 是假設值**：v1 用 0.55/0.62/0.72 是粗估，reliability table 會顯示落差但 v1 不重跑（重跑屬 follow-up）
- **odds snapshot 密度不均**：早期日期 snapshot 較稀（5/01 只有兩個 snapshot），closing 取到的可能距 commence 較遠；計入 `closing_snapshot_ts` 讓人可查
- **不評估 skill 對 odds report Watch 軌的影響**：兩條軌獨立，此 spec 只涵蓋 fundamentals
