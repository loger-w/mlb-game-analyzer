# Prediction Summary — STL @ PIT (2026-04-30)

**開打時間**: 2026-04-30 06:40 TW（ET 04-29 18:40）

## TL;DR
- 預測比分: **PIT 6.0 − 6.3 STL**（AWAY 勝，勝率 50.6%）
- 比賽走勢: 兩位 Back-end 先發 K-BB% 都僅 ~5%，打者擊球機會多；STL 牛棚 ERA 5.18 嚴重落後 PIT 3.49，後段任一隊領先都易被反超 → 局數越往後分數越易堆疊，整體偏向高分混戰。

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 50.6% (HOME) |
| O/U | OVER | ⭐⭐⭐⭐ | adj_total 12.4 vs line 8.5，差距 3.9 run |
| Run Line | PASS | — | |diff|=0.3 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: PIT 5.5 / STL 6.3（總分 11.8）
- Adjusted 比分: PIT 6.0 / STL 6.3（總分 12.4）
- O/U gap: |adj_total 12.4 − line 8.5| = 3.9

## 勝率預測
- ⚠️ Formula 50.6% (HOME) → adjusted 比分 6.05 < 6.35 判 AWAY 勝（pct 未隨翻轉重算）

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 客隊牛棚 ERA 5.18 ≥ 5.0 | +0.50 |
| Park Factor 102.0（修正 +0.10） | +0.10 |
| **總和** | **+0.60** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `stl_weak_bullpen` | +0.50 |
| `park_pf` | +0.10 |
| **總和** | **+0.60** |

## 推薦結果
- **ML**: **PASS** — Log5 50.6% (HOME)
- **O/U**: **OVER ⭐⭐⭐⭐** — adj_total 12.4 vs line 8.5，差距 3.9 run
- **Run Line**: **PASS** — |diff|=0.3 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 12.4 > ou_line 8.5 vs ou_rec=OVER

## 趨勢標記
- `early-season`、`low-kbb-both`、`bullpen-asymmetry`、`away-bullpen-slump`
