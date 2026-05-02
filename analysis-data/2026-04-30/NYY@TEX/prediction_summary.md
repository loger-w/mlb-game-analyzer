# Prediction Summary — NYY @ TEX (2026-04-30)

**開打時間**: 2026-04-30 02:35 TW（ET 04-29 14:35）

## TL;DR
- 預測比分: **TEX 3.6 − 6.4 NYY**（AWAY 勝，勝率 44.2%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | NYY | ⭐⭐⭐⭐ | Log5 44.2% (AWAY) |
| O/U | OVER | ⭐⭐⭐ | adj_total 10.0 vs line 8.5，差距 1.5 run |
| Run Line | NYY | ⭐⭐ | override `big-diff`，|diff|=2.8 |

---

## 比分預測
- Formula 比分: TEX 4.6 / NYY 6.4（總分 11.0）
- Adjusted 比分: TEX 3.6 / NYY 6.4（總分 10.0）
- O/U gap: |adj_total 10.0 − line 8.5| = 1.5

## 勝率預測
- Formula log5: **44.2% (HOME)**

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| Park Factor 96.0（修正 -0.20） | -0.20 |
| **總和** | **-0.20** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `platoon_lhb_vs_eovaldi` | +0.40 |
| `eovaldi_xfip_regression` | -0.40 |
| `tex_cold_streak` | -0.30 |
| `rodriguez_unknown_conservative` | -0.50 |
| `park_factor_drag` | -0.20 |
| **總和** | **-1.00** |

## 推薦結果
- **ML**: **NYY ⭐⭐⭐⭐** — Log5 44.2% (AWAY)
- **O/U**: **OVER ⭐⭐⭐** — adj_total 10.0 vs line 8.5，差距 1.5 run
- **Run Line**: **NYY ⭐⭐** — override `big-diff`，|diff|=2.8

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：predicted_winner=AWAY(NYY) 與 ml_rec=NYY 一致
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 10.0 > ou_line 8.5 vs ou_rec=OVER

## Run Line override 細節
- 路徑: `big-diff`
- |diff|: 2.80
- stars: 2
- thresholds: diff_min=1.5, diff_big=2.2, diff_star=2.0

## 趨勢標記
- `rookie-debut-pitcher`、`platoon-edge`、`xfip-divergence`、`home-bullpen-strong`、`away-hot-offense`、`away-pitching-hot`
