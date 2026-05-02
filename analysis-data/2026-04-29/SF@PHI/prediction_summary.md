# Prediction Summary — SF @ PHI (2026-04-29)

**開打時間**: 2026-04-29 06:40 TW（ET 04-28 18:40）

## TL;DR
- 預測比分: **PHI 6.6 − 4.9 SF**（HOME 勝，勝率 45.0%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PHI | ⭐⭐ | Log5 45.0% (AWAY)，audit `home-2star-risk` |
| O/U | OVER | ⭐⭐⭐⭐ | adj_total 11.5 vs line 7.0，差距 4.5 run |
| Run Line | PASS | — | |diff|=1.7 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: PHI 5.9 / SF 2.8（總分 8.7）
- Adjusted 比分: PHI 6.6 / SF 4.9（總分 11.5）
- O/U gap: |adj_total 11.5 − line 7.0| = 4.5

## 勝率預測
- ⚠️ Formula 45.0% (HOME) → adjusted 比分 6.6 > 4.9 判 HOME 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| Park Factor 104.0（修正 +0.20） | +0.20 |
| **總和** | **+0.20** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `luzardo_real_quality_above_era` | -1.20 |
| `phi_bullpen_duran_il` | +0.30 |
| `sf_bullpen_foley_il` | +0.30 |
| **總和** | **-0.60** |

## 推薦結果
- **ML**: **PHI ⭐⭐** — Log5 45.0% (AWAY)，audit `home-2star-risk`
- **O/U**: **OVER ⭐⭐⭐⭐** — adj_total 11.5 vs line 7.0，差距 4.5 run
- **Run Line**: **PASS** — |diff|=1.7 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：predicted_winner=HOME(PHI) 與 ml_rec=PHI 一致
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 11.5 > ou_line 7.0 vs ou_rec=OVER

## 趨勢標記
- `early-season`、`flag13-luzardo`、`divergent-era-xera`、`log5-formula-divergent`、`away-hot-offense`、`away-pitching-hot`
