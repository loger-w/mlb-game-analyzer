# 指標診斷報告 - 2026-04-19

## 樣本範圍

- 起日：2026-04-18
- 止日：2026-04-18
- 總場次：12
- 過濾條件：`date >= 2026-04-18`
- ML 推薦場次：4
- O/U 推薦場次：2
- RL 推薦場次：0

## (A) PASS 召回率

| 盤口 | 實際推薦 W-L-P | 強制全下 W-L-P | 差距（W/L） |
|------|---------------|----------------|-------------|
| ML | 1-3-0 | 5-7-0 | +4 W / +4 L |
| O/U | 0-2-0 | 4-8-0 | +4 W / +6 L |
| RL | 0-0-0 | 8-4-0 | +8 W / +4 L |

### ML PASS 但方向正確（4 場）

| 比賽 | 預測 | 實際 | PASS 原因 |
|------|------|------|----------|
| DET @ BOS | AWAY | AWAY | LOW, INSUFFICIENT_SAMPLE |
| TEX @ SEA | HOME | HOME | LOW, INSUFFICIENT_SAMPLE |
| TOR @ ARI | HOME | HOME | LOW, INSUFFICIENT_SAMPLE, insufficient-sample, divergent |
| SD @ LAA | AWAY | AWAY | LOW, INSUFFICIENT_SAMPLE, insufficient-sample |

### O/U PASS 但方向正確（4 場）

| 比賽 | 預測 | 實際 | PASS 原因 |
|------|------|------|----------|
| KC @ NYY | 8.5 vs 8.0 | 17 | LOW, INSUFFICIENT_SAMPLE, insufficient-sample |
| NYM @ CHC | 7.7 vs 8.5 | 6 | LOW, INSUFFICIENT_SAMPLE, divergent |
| MIL @ MIA | 7.7 vs 7.85 | 7 | LOW, INSUFFICIENT_SAMPLE |
| ATL @ PHI | 7.4 vs 7.5 | 4 | LOW, INSUFFICIENT_SAMPLE, divergent, insufficient-sample |

### RL PASS 但方向正確（8 場）

| 比賽 | 預測 | 實際 | PASS 原因 |
|------|------|------|----------|
| KC @ NYY | 5.3-3.2 | 13-4 | LOW, INSUFFICIENT_SAMPLE, insufficient-sample |
| CIN @ MIN | 3.8-3.5 | 4-5 | LOW, INSUFFICIENT_SAMPLE, divergent |
| SF @ WSH | 5.1-5.0 | 6-7 | LOW, INSUFFICIENT_SAMPLE |
| TB @ PIT | 4.0-3.1 | 7-8 | LOW, INSUFFICIENT_SAMPLE, divergent, insufficient-sample |
| DET @ BOS | 2.5-6.8 | 1-4 | LOW, INSUFFICIENT_SAMPLE |
| TEX @ SEA | 3.6-3.3 | 7-3 | LOW, INSUFFICIENT_SAMPLE |
| TOR @ ARI | 6.5-5.0 | 6-2 | LOW, INSUFFICIENT_SAMPLE, insufficient-sample, divergent |
| SD @ LAA | 6.0-6.3 | 1-4 | LOW, INSUFFICIENT_SAMPLE, insufficient-sample |

## (B) 指標校準

### ML：`predicted_home_pct` 分檔命中率

| 分檔 | N | 命中 | 命中率 |
|------|---|------|--------|
| < 50% | 4 | 3 | 75.0% |
| 50–55% | 0 | 0 | insufficient |
| 55–60% | 2 | 1 | insufficient |
| 60–65% | 2 | 1 | insufficient |
| ≥ 65% | 4 | 1 | 25.0% |

### O/U：總分預測 MAE

| 指標 | N | MAE |
|------|---|-----|
| formula_total (home+away pre-signal) | 12 | 5.26 |
| adjusted_total | 12 | 4.03 |
| predicted_total | 12 | 4.03 |

方向準確率（vs ou_line）：33.3%（N=12）

### RL：margin 與 per-side

- margin MAE：2.88（N=12）
- winner 一致率：41.7%（N=12）
- home 得分 MAE：2.36
- away 得分 MAE：2.14

### XGB raw vs final pct

（`xgb_raw_home_pct` 欄位不在樣本中，跳過）

## (C) Signal 類別 ablation

納入分析場次（signal_adjustments 非空）：9

| 類別 | 觸發次數 | 涉及場次 | 平均貢獻 | 觸發 MAE (N) | 未觸發 MAE (N) | Δ |
|------|---------|---------|---------|-------------|----------------|---|
| bullpen/injury | 14 | 9 | 0.37 | 3.64 (9) | — (0) | — |
| other | 7 | 3 | -0.03 | 2.67 (3) | 4.13 (6) | -1.47 |
| park | 4 | 4 | 0.06 | 3.45 (4) | 3.80 (5) | -0.35 |
| pitcher quality | 4 | 3 | -0.28 | 4.43 (3) | 3.25 (6) | 1.18 |
| weather | 2 | 2 | -0.12 | 2.30 (2) | 4.03 (7) | -1.73 |
| recent form | 1 | 1 | 0.30 | 7.30 (1) | 3.19 (8) | 4.11 |

註：Δ > 0 代表觸發此類 signal 的場次 adjusted_total MAE 反而更差（可能負貢獻）。

## 結論摘要

- **ML PASS 門檻可能過嚴**：強制全下多贏 4 場（33% of forced）。
- **O/U PASS 門檻可能過嚴**：強制全下多贏 4 場（33% of forced）。
- **RL PASS 門檻可能過嚴**：強制全下多贏 8 場（67% of forced）。
- **`pitcher quality` 類 signal 觸發後 MAE 顯著惡化**：4.43 vs 3.25（Δ +1.18）。
