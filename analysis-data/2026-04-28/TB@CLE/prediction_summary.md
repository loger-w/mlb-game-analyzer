# Prediction Summary — TB @ CLE (2026-04-28)

## TL;DR
- 預測比分: **CLE 5.4 − 5.2 TB**（HOME 勝，勝率 52.0%）
- 比賽走勢: 雙方 back-end 級先發、ML 4% 內均勢（硬幣翻轉），前 5 局多在 3-3 / 4-4 拉鋸；6 局後 TB 牛棚 2+ 核心 IL（Cleavinger 等）+ ERA 5.18 易被攻破，CLE 牛棚 1 名 IL（Walters）也偏弱 → **牛棚崩盤劇本上修總分**。Martínez ERA 2.10 為 5 場運氣假象（xERA 4.64），CLE 後段壓力不會比 TB 小。

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | PASS | — | Log5 52.0% (HOME) |
| O/U | OVER | ⭐⭐⭐⭐ | adj_total 10.6 vs line 7.5，差距 3.1 run |
| Run Line | PASS | — | |diff|=0.2 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: CLE 4.4 / TB 4.8（總分 9.2）
- Adjusted 比分: CLE 5.4 / TB 5.2（總分 10.6）
- O/U gap: |adj_total 10.6 − line 7.5| = 3.1

## 勝率預測
- Formula log5: **52.0% (HOME)**

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| 客隊牛棚 ERA 5.18 ≥ 5.0 | +0.50 |
| **總和** | **+0.50** |

### User-supplied signals
| Key | ±run |
|-----|------|
| `martinez_era_xera_gap` | +0.60 |
| `tb_bullpen_il_extra` | +0.30 |
| `cle_bullpen_il` | +0.30 |
| `park_factor` | +0.10 |
| **總和** | **+1.30** |

## 推薦結果
- **ML**: **PASS** — Log5 52.0% (HOME)
- **O/U**: **OVER ⭐⭐⭐⭐** — adj_total 10.6 vs line 7.5，差距 3.1 run
- **Run Line**: **PASS** — |diff|=0.2 < 1.5（RL_DIFF_MIN）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：ml_rec=PASS
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 10.6 > ou_line 7.5 vs ou_rec=OVER

## 趨勢標記
- `flag-13-era-xera`、`flag-3-babip`、`bullpen-il-both`、`away-bullpen-slump`
