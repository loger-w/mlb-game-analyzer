# RA-defense ablation — 2026 (train Mar–Apr → test May)

訓練=468 場  測試(有盤口)=292 注場
選出 w* = 0.0

| 模型 | w | league_rg | sigma_team | RL ll | OU ll | pooled ll |
|------|------|-----------|------------|-------|-------|-----------|
| baseline | 0.0 | 4.007 | 3.693 | 0.6922 | 0.7038 | 0.6979 |
| candidate | 0.0 | 4.007 | 3.693 | 0.6922 | 0.7038 | 0.6979 |

OOS pooled 改善(baseline − candidate)= 0.0000 ± 0.0000 (1 SE)
**判決:REJECT**(接受條件:改善 > 1 SE)

離市場差距(pooled ll − market 0.6876): baseline 0.0102 → candidate 0.0102

> 北極星=差距≤0(打敗市場)。此判決僅決定 RA 是否值得進模型;baking 進 config/run_model 是另一個決定。
