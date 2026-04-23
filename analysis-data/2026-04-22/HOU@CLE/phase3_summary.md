# Phase 3 — HOU @ CLE 2026-04-22（Progressive Field）

## 先發投手對決

### CLE Tanner Bibee（RHP, 27, 🟢 Back-end Starter）
- 2026: 5GS / 24.1 IP / ERA 4.81 / FIP 4.46 / xFIP 3.60 / WHIP 1.52 / K% 21.5 / BB% 8.4 / HR/9 1.48
- 2025: 31GS / 182.1 IP / ERA 4.24 / FIP 4.14 / xFIP 3.53 / K% 21.3 / BB% 7.1
- Statcast: avg velo 87.4（加權，含 FC/FF/CH/CU/SI）/ max 96.1 / Whiff 12.7% / HardHit 35.6% / Barrel 9.3% / CSW 28.7%
- Pitch mix: FC 28.7 / FF 27.7 / CH 17.5 / CU 13.1 / SI 13.1（五球種均勻）
- Platoon（本季小樣本）: vs LHB 65 BF .207/.292/.431 / vs RHB 42 BF **.400/.429/.475**
- 去年大樣本 vs 本季：whiff 降、HR/9 升、BB% 升、WHIP 明顯惡化（1.23 → 1.52）— 本季前 5 場狀況比去年差，但無結構性球速/球種劇變。
- YoY 閘門：2026 ERA 4.81 > 2025 ERA 4.24，未觸發「IP<30 且低 ≥1.0」條件，不需補跑。

### HOU Peter Lambert（RHP, 29, ⚪ Below Average）
- 2026: 1GS / 5.0 IP / ERA 7.20（4/17 vs STL：5 IP, 7H, 4 ER, 8K, 1BB, 90 pitches）
- **2025: 無 MLB 資料**（腳本 return "No 2025 pitching stats found"）
- Statcast: avg velo 90.3 / max 97.4 / Whiff 23.3% / HardHit 37.5% / **Barrel 15.4%** / CSW 37.8%
- Pitch mix: FF 30 / CH 27.8 / SL 17.8 / FC 8.9 / SV 8.9
- 關鍵紅旗：Barrel% 15.4 接近聯盟平均兩倍（~8%），長球風險極高
- 非正規先發 — 因 HOU 輪值重創（Hunter Brown、Javier、Blanco、Wesneski、Walter 5 名 IL）頂上來

### 投手差距
**Bibee 🟢 vs Lambert ⚪ = 1 檔以上差距**，CLE 在 SP 上明顯佔優。Bibee 本季 5 場狀況不佳但底子仍高於 Lambert；Lambert 的 FIP 0.5 / xFIP 1.02 是 5 IP 極小樣本雜訊，**不可採信**。

---

## 打線評級

### CLE 🟠 Strong
- OPS .720 / xwOBA .345 / BABIP .273 / K% 19.9 / BB% 11.1
- 近 7 天 BABIP .306（中性，無回歸閘門觸發）
- 近 10 場 RS/G 4.8（高於本季 4.04）— 打線正在加溫

### HOU 🟡 Average
- OPS .814 / xwOBA .340 / BABIP .303 / K% 20.1 / BB% 11.2
- 近 7 天 BABIP .269（略低於 .270 但 > .260 下限，未觸發回歸）
- 近 10 場 RS/G 4.9（低於本季 5.4）— 打線正在降溫
- ⚠️ 打線傷兵：Peña（SS 核心）、Jake Meyers（正 CF）、Loperfido / Dezenzo / Trammell / Allen 共 6 人 IL，已部分反映在本季數據

---

## 牛棚雙向修正值（🔴 關鍵差異）

| 項目 | CLE | HOU |
|---|---|---|
| 牛棚 ERA | 5.04（中等偏差） | **5.91（極差）** |
| 核心 IL | 1 人（Andrew Walters, 非 high-leverage） | **4+ 人（Josh Hader closer 60-day / Nate Pearson / Bennett Sousa / Cody Bolton 皆 IL）** |
| 深度 | 13 人 active 完整 | 13 人 active 但被大量傷兵吞蝕 |

**IL 累計效應判定（matchup-factors §牛棚）**：
- HOU 牛棚保守算 **2 名核心 IL**（Hader Closer + Sousa LH specialist 或 Pearson setup，Bolton 視為邊緣）
- 對應修正：CLE O/U **+0.5 ~ +0.7 run**，HOU ML **-3% ~ -4%**
- CLE 側：僅 1 邊緣 IL，不計入修正

**雙向反映（避免只記 O/U 不記 ML 的錯）**：
- O/U 方向：HOU 牛棚差 → CLE 總分上修 +0.5
- ML 方向：HOU 牛棚差 → HOU 勝率下修 -3%

---

## 條件修正

| 信號 | 值 | 理由 |
|---|---|---|
| Park Factor | 98 → -0.1 總分 | Progressive Field 偏投手 |
| HOU 牛棚 2 名核心 IL | +0.5 run（加到 CLE 得分）、HOU ML -3% | 見上 |
| Bibee vs RHB 小樣本極差 | +0.2 run（加到 HOU 得分，保守） | 42 BF .400 BA 有運氣成分但值得 note |
| HOU 打線傷兵（Peña 等 6 人） | 不額外扣（已反映在本季 xwOBA .340） | 避免雙重計算 |
| Lambert replacement-level | 不額外扣（已由 ⚪ tier 反映） | 同上 |

---

## 修正後預期得分（手動估算，最終以 predict.py 為準）

基準：聯盟 R/G ≈ 4.3，xwOBA ≈ .315

- CLE 預期得分 ≈ 4.3 × (.345/.315) × (Lambert 實戰 ERA ~5.0 / 4.0) × (98/100) ≈ **5.8 run**，加 +0.5 牛棚信號 → **~6.3**
- HOU 預期得分 ≈ 4.3 × (.340/.315) × (4.81/4.0) × (98/100) ≈ **5.5 run**，加 +0.2 平台信號 → **~5.7**
- 修正後總分 ≈ **12.0 run**（vs O/U 8.5，差距 +3.5）

⚠️ 此為手動估算，高度敏感於 Lambert「實戰 ERA」的假設。predict.py 會以 Lambert 實際 ERA 7.20 或 xFIP 套入 formula，可能得到更極端的結果；實際比分模型會因小樣本有 regression。最終盤口由 Phase 4 `prediction.json` 決定。

---

## 整體判斷

- **基本面方向**：CLE（主場 + 投手差 1 檔 + 牛棚差異巨大 + 打線近期熱）
- **總分傾向**：**偏高**（HOU 牛棚瓦解 + Lambert 長球率高 + H2H 前兩場平均 12 分）
- **信心程度**：中高（投手/牛棚差異明確，但 Lambert 僅 1 GS 小樣本 + Bibee vs R 小樣本 = 兩側都有雜訊）
- **主要風險**：
  1. Lambert 的「真實水平」不可能由 5 IP 推斷；若他今天的大谷模式（whiff 23%/CSW 38%）持續 5-6 局，CLE 得分可能低於預期
  2. Bibee 本季 WHIP 1.52 加上 vs RHB 42 BF .400 — HOU 打線若點到他的 FF/FC，可能中段爆出 3+ run
  3. HOU 昨天被大勝後的反彈動能（series 4/20 9-2 贏、4/21 5-8 輸）
  4. 系列賽第 3 場，雙方牛棚前一日消耗情況未查（保守視為可用）
