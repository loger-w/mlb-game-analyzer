# Prediction Summary — WSH @ NYM (2026-04-30)

**開打時間**: 2026-04-30 07:10 TW（ET 04-29 19:10）

## TL;DR
- 預測比分: **NYM 2.7 − 4.8 WSH**（AWAY 勝，勝率 54.8%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 54.8% (HOME) |
| O/U | PASS | — | 差距 0.5 < 1.5 run |
| Run Line | WSH | ⭐⭐ | override `mid-diff+strong-tag`，|diff|=2.1，tags=`away-bullpen-slump` |

---

## 比分預測
- Formula 比分: NYM 2.3 / WSH 3.8（總分 6.1）
- Adjusted 比分: NYM 2.7 / WSH 4.8（總分 7.5）
- O/U gap: |adj_total 7.5 − line 7.0| = 0.5

## 勝率預測
- ⚠️ Formula 54.8% (HOME) → adjusted 比分 2.7 < 4.8 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 客隊牛棚 ERA 5.11 ≥ 5.0 | +0.50 |
| Park Factor 96.0（修正 -0.20） | -0.20 |
| **總和** | **+0.30** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `nym_bullpen_2_core_il` | +0.50 |
| `home_babip_regression` | +0.20 |
| `cavalli_vs_rhb_platoon` | -0.30 |
| `away_babip_regression` | +0.30 |
| `wood_vs_lhp_advantage` | +0.20 |
| **總和** | **+0.90** |

## 推薦結果
- **ML**: **PASS** — Log5 54.8% (HOME)
- **O/U**: **PASS** — 差距 0.5 < 1.5 run
- **Run Line**: **WSH ⭐⭐** — override `mid-diff+strong-tag`，|diff|=2.1，tags=`away-bullpen-slump`

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：ou_rec=PASS 或無 line

## Run Line override 細節
- 路徑: `mid-diff+strong-tag`
- |diff|: 2.10
- stars: 2
- 觸發 tags: `away-bullpen-slump`
- thresholds: diff_min=1.5, diff_big=2.2, diff_star=2.0

## 趨勢標記
- `platoon-edge`、`bullpen-double-weak`、`early-season-babip`、`home-pitching-hot`
