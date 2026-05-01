# Prediction Summary — LAA @ CWS (2026-04-30)

**開打時間**: 2026-04-30 01:10 TW（ET 04-29 13:10）

## TL;DR
- 預測比分: **CWS 4.7 − 5.4 LAA**（AWAY 勝，勝率 62.2%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 62.2% (HOME) |
| O/U | OVER | ⭐⭐⭐ | adj_total 10.1 vs line 8.5，差距 1.6 run |
| Run Line | PASS | — | |diff|=0.7 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: CWS 3.7 / LAA 4.9（總分 8.6）
- Adjusted 比分: CWS 4.7 / LAA 5.4（總分 10.1）
- O/U gap: |adj_total 10.1 − line 8.5| = 1.6

## 勝率預測
- ⚠️ Formula 62.2% (HOME) → adjusted 比分 4.7 < 5.4 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 客隊牛棚 ERA 5.79 ≥ 5.0 | +0.50 |
| Park Factor 97.0（修正 -0.15） | -0.15 |
| 雙方打線 K% ≥ 25% | -0.30 |
| **總和** | **+0.05** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `bullpen_laa_3core_il` | +1.00 |
| `bullpen_cws_2core_il` | +0.50 |
| `park_rate_field` | -0.15 |
| **總和** | **+1.35** |

## 推薦結果
- **ML**: **PASS** — Log5 62.2% (HOME)
- **O/U**: **OVER ⭐⭐⭐** — adj_total 10.1 vs line 8.5，差距 1.6 run
- **Run Line**: **PASS** — |diff|=0.7 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 10.1 > ou_line 8.5 vs ou_rec=OVER

## 趨勢標記
- `early-season`、`kikuchi-era-fip-divergent`、`reverse-platoon`、`formula-log5-vs-score-split`、`home-hot-offense`、`away-cold-offense`、`away-bullpen-slump`
