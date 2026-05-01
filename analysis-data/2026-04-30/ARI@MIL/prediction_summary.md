# Prediction Summary — ARI @ MIL (2026-04-30)

**開打時間**: 2026-04-30 07:40 TW（ET 04-29 19:40）

## TL;DR
- 預測比分: **MIL 6.0 − 6.5 ARI**（AWAY 勝，勝率 58.2%）
- 比賽走勢: 雙弱投對戰（Sproat 真 Below-Avg、E-Rod 表面 ERA 2.89 但 xERA 4.94 為 contact-heavy 軟投），雙打線近期都偏熱、ARI 牛棚 ERA 5.21 + 2 名核心 IL，6 局後易失血；總分大概率往 12-13 分區間走，方向則受 Sproat 對 LHB 弱點（ARI Carroll/Vargas 平台優勢）與 MIL 主場 + 牛棚優勢拉鋸。

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 58.2% (HOME) |
| O/U | OVER | ⭐⭐⭐⭐⭐ | adj_total 12.5 vs line 8.5，差距 4.0 run |
| Run Line | PASS | — | |diff|=0.6 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: MIL 5.2 / ARI 6.3（總分 11.5）
- Adjusted 比分: MIL 6.0 / ARI 6.5（總分 12.5）
- O/U gap: |adj_total 12.5 − line 8.5| = 4.0

## 勝率預測
- ⚠️ Formula 58.2% (HOME) → adjusted 比分 5.95 < 6.55 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 雙方打線近期 Hot（場均 ≥ 5 分） | +0.50 |
| 客隊牛棚 ERA 5.21 ≥ 5.0 | +0.50 |
| Park Factor 97.0（修正 -0.15） | -0.15 |
| **總和** | **+0.85** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `away_bullpen_era_high` | +0.50 |
| `both_hot` | +0.50 |
| `park_factor` | -0.15 |
| **總和** | **+0.85** |

## 推薦結果
- **ML**: **PASS** — Log5 58.2% (HOME)
- **O/U**: **OVER ⭐⭐⭐⭐⭐** — adj_total 12.5 vs line 8.5，差距 4.0 run
- **Run Line**: **PASS** — |diff|=0.6 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 12.5 > ou_line 8.5 vs ou_rec=OVER

## 趨勢標記
- `early-season`、`bullpen-edge`、`pitcher-mismatch`、`hot-bats`、`away-hot-offense`、`away-pitching-slump`、`away-bullpen-slump`
