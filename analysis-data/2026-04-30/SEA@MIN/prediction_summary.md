# Prediction Summary — SEA @ MIN (2026-04-30)

**開打時間**: 2026-04-30 01:40 TW（ET 04-29 13:40）

## TL;DR
- 預測比分: **MIN 3.5 − 5.0 SEA**（AWAY 勝，勝率 50.5%）
- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 50.5% (HOME) |
| O/U | PASS | — | 差距 0.9 < 1.5 run |
| Run Line | SEA | ⭐ | override `mid-diff+strong-tag`，|diff|=1.5，tags=`home-bullpen-slump` |

---

## 比分預測
- Formula 比分: MIN 4.3 / SEA 4.5（總分 8.8）
- Adjusted 比分: MIN 3.5 / SEA 5.0（總分 8.4）
- O/U gap: |adj_total 8.4 − line 7.5| = 0.9

## 勝率預測
- ⚠️ Formula 50.5% (HOME) → adjusted 比分 3.45 < 4.95 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 主隊牛棚 ERA 5.13 ≥ 5.0 | +0.50 |
| Park Factor 106.0（修正 +0.30） | +0.30 |
| **總和** | **+0.80** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `min_bullpen_weak` | +0.40 |
| `sea_bullpen_strong` | -0.30 |
| `min_offense_cold_last10` | -0.30 |
| `bradley_xera_regression` | +0.30 |
| `park_factor_106` | +0.30 |
| `both_solid_plus_starters` | -0.50 |
| **總和** | **-0.10** |

## 推薦結果
- **ML**: **PASS** — Log5 50.5% (HOME)
- **O/U**: **PASS** — 差距 0.9 < 1.5 run
- **Run Line**: **SEA ⭐** — override `mid-diff+strong-tag`，|diff|=1.5，tags=`home-bullpen-slump`

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：ou_rec=PASS 或無 line

## Run Line override 細節
- 路徑: `mid-diff+strong-tag`
- |diff|: 1.50
- stars: 1
- 觸發 tags: `home-bullpen-slump`
- thresholds: diff_min=1.5, diff_big=2.2, diff_star=2.0

## 趨勢標記
- `formula-vs-market-divergent`、`bradley-xera-gap`、`bullpen-asym`、`bookmaker-fav-AWAY`、`early-season`、`home-cold-offense`
