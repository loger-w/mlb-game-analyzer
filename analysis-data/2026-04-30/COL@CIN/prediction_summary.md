# Prediction Summary — COL @ CIN (2026-04-30)

**開打時間**: 2026-04-30 06:40 TW（ET 04-29 18:40）

## TL;DR
- 預測比分: **CIN 7.0 − 4.9 COL**（HOME 勝，勝率 58.2%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | CIN | ⭐⭐ | Log5 58.2% (HOME)，audit `divergent`, `home-2star-risk` |
| O/U | OVER | ⭐⭐⭐ | adj_total 11.9 vs line 9.5，差距 2.4 run |
| Run Line | PASS | — | |diff|=2.1 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: CIN 6.2 / COL 6.6（總分 12.8）
- Adjusted 比分: CIN 7.0 / COL 4.9（總分 11.9）
- O/U gap: |adj_total 11.9 − line 9.5| = 2.4

## 勝率預測
- Formula log5: **58.2% (HOME)**

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| Park Factor 104.0（修正 +0.20） | +0.20 |
| **總和** | **+0.20** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `sugano_xera_regression` | +0.50 |
| `gabp_hr_path` | +0.30 |
| `col_babip_regression` | -0.40 |
| `col_recent_slump` | -1.00 |
| `cin_bullpen_il_minor` | +0.00 |
| `col_vs_lhp` | -0.30 |
| **總和** | **-0.90** |

## 推薦結果
- **ML**: **CIN ⭐⭐** — Log5 58.2% (HOME)，audit `divergent`, `home-2star-risk`
- **O/U**: **OVER ⭐⭐⭐** — adj_total 11.9 vs line 9.5，差距 2.4 run
- **Run Line**: **PASS** — |diff|=2.1 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：predicted_winner=HOME(CIN) 與 ml_rec=CIN 一致
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 11.9 > ou_line 9.5 vs ou_rec=OVER

## 趨勢標記
- `early-season`、`flag13-both-pitchers`、`home-hot-offense`、`home-bullpen-strong`
