# Prediction Summary — HOU @ BAL (2026-04-30)

**開打時間**: 2026-04-30 06:35 TW（ET 04-29 18:35）

## TL;DR
- 預測比分: **BAL 3.1 − 6.3 HOU**（AWAY 勝，勝率 54.2%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 54.2% (HOME) |
| O/U | PASS | — | 差距 0.4 < 1.5 run |
| Run Line | HOU | ⭐⭐ | override `big-diff`，|diff|=3.2，tags=`away-bullpen-slump`, `home-pitching-slump` |

---

## 比分預測
- Formula 比分: BAL 1.4 / HOU 6.4（總分 7.8）
- Adjusted 比分: BAL 3.1 / HOU 6.3（總分 9.4）
- O/U gap: |adj_total 9.4 − line 9.0| = 0.4

## 勝率預測
- ⚠️ Formula 54.2% (HOME) → adjusted 比分 3.1 < 6.35 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 客隊牛棚 ERA 6.27 ≥ 5.0 | +0.50 |
| Park Factor 96.0（修正 -0.20） | -0.20 |
| **總和** | **+0.30** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `lambert_gs2_sample_downgrade` | +0.70 |
| `hou_bullpen_il_3plus` | +1.00 |
| `hou_lhb_platoon_vs_bassitt` | +0.30 |
| `bal_bullpen_il_1to2` | +0.30 |
| `park_factor_camden_softened` | -0.15 |
| `bal_last7_babip_regress` | +0.15 |
| `away_starter_overheat_correction` | -0.50 |
| **總和** | **+1.80** |

## 推薦結果
- **ML**: **PASS** — Log5 54.2% (HOME)
- **O/U**: **PASS** — 差距 0.4 < 1.5 run
- **Run Line**: **HOU ⭐⭐** — override `big-diff`，|diff|=3.2，tags=`away-bullpen-slump`, `home-pitching-slump`

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：ou_rec=PASS 或無 line

## Run Line override 細節
- 路徑: `big-diff`
- |diff|: 3.25
- stars: 2
- 觸發 tags: `away-bullpen-slump`, `home-pitching-slump`
- thresholds: diff_min=1.5, diff_big=2.2, diff_star=2.0

## 趨勢標記
- `early-season`、`small-sample`、`bullpen-il`
