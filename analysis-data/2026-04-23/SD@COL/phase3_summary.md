# Phase 3 分析結論 — SD @ COL 2026-04-23 (Coors Field, 15:10 ET)

## 先發對決

### Matt Waldron (SD, R, 29, Knuckleballer) — ⚪ Below Average

| 指標 | 2026 (3.67 IP) | 2025 (4.67 IP) | 2024 (146.7 IP) |
|------|---------------|----------------|-----------------|
| ERA | 14.73 | 7.71 | 4.91 |
| FIP | 5.28 | 8.46 | 3.97 |
| xERA | **4.77** | 11.19 | 4.08 |
| avg velo | 85.4 | 81.9 | 83.0 |
| whiff% | 6.5 | 5.8 | 8.7 |
| hard hit% | 18.5 | 29.4 | 22.2 |
| barrel% | 6.7 | 12.5 | 7.0 |
| 主球種分布 | KN 39.6 / ST 18.7 / FF 15.8 / SI 15.1 / FC 10.8 | KN 74.0（崩盤）| KN 38.2 / ST 19.8 / FF 18.7 / SI 14.6 / FC 8.7 |

### §YoY 對比結論（Waldron）
- **ERA vs xERA 落差 |14.73 − 4.77| = 9.96（> 1.5，觸發 YoY）**
- 五項指標對照 2024：velo +2.4 mph（實質改善）/ hard_hit −3.7pp（改善）/ barrel 持平 / whiff −2.2pp（樣本小）/ xERA 4.77 接近 2024 的 4.08
- 球種配比**完全回歸 2024 的 below-average 先發 pattern**（2025 極端 KN 74% 的崩盤已結束）
- 判定：**ERA 14.73 是 1 場 small-sample 噪音；真實水準 = xERA 4.77 ≈ 2024 全季等級（ERA 4.91 / FIP 3.97）**
- **採用 xERA 4.77 作為本場預測基準，非 ERA 14.73**

### Ryan Feltner (COL, R, 29) — ⚪ Below Average

| 指標 | 2026 (18 IP) | 2025 (30.3 IP) |
|------|--------------|----------------|
| ERA | 6.00 | 4.75 |
| FIP | 6.32 | 4.35 |
| xERA | **6.96** | — |
| avg velo | 89.4 | |
| whiff% | 11.6 | |
| hard hit% | 30.8 | |
| barrel% | **17.2**（高）| |

- ERA vs xERA 落差 0.96 < 1.5，**不觸發 YoY**；但 xERA 6.96 > ERA 6.00 意味 ERA 可能會往上回歸
- **Platoon：vs L .194/.242/.452 (K% 24.2) vs R .341/.413/.683 (K% 13.0, BF 46)** — **vs RHB 被打爆（.683 SLG）**
- 近 3 場 game log：3/31 TOR 3 IP 0 ER / 4/6 HOU 5.3 IP 4 ER / **4/11 vs SD 4 IP 6 ER**（剛被 SD 打爆）

### 投手評級差距
雙方皆 ⚪ Below Average，但 Waldron xERA 4.77 vs Feltner xERA **6.96**，差距 ~2.2 run → **Waldron 有 ~1 檔以上優勢**。

## 打線

### COL vs Waldron (R) — 🟡 Average
- OPS .727 / xwOBA .318 / K% 26.7 / BB% 8.3
- last7 BABIP **.309**（正常，不觸發回歸）/ recent heat ⚖️ Normal / chain OBP top3 .301 · SLG mid .385
- 關鍵打者：Moniak vs R OPS **1.142**、Goodman .884、Johnston .918
- **對 Waldron knuckleball 特殊風險**：knuckleball 在 Coors 高海拔空氣密度低，球的失轉/擾動效果顯著下降（物理依據 + MLB 歷史觀察）

### SD vs Feltner (R) — 🟡 Average
- OPS .639 / xwOBA .322 / K% 21.8 / BB% 8.8
- last7 BABIP **.272**（> .260，不觸發閘門；**但接近下緣，運氣偏低**）/ recent heat **🥶 Cold**
- chain OBP top3 .312 / SLG mid **.430**（清壘端優於 COL）
- 打線組成：6 RHB + 3 LHB（Tatis, Machado, Bogaerts, Laureano, Andujar, Castellanos 為 RHB）
- **Feltner vs RHB .683 SLG 被打爆 → SD 6 RHB 對 Feltner 明顯 platoon 優勢**
- 上場（4/11）對 Feltner 4 IP 6 ER 打爆

## §BABIP 回歸判定
- COL last7 .309：聯盟均值附近，不調整 hot/cold
- SD last7 .272：Cold 標籤看起來成立，但 BABIP .272 偏低 → **部分 Cold 是運氣，預期回歸至 ~.300**；Cold 不扣 run value

## 牛棚（雙向修正）

### §牛棚雙向修正值
- COL bullpen ERA **3.31**（中上）/ SD bullpen ERA **3.06**（優秀）
- IL 名單：COL 4P + 1DH / SD 6P + 1 3B（roster JSON fullName 欄位為 null 無法鎖定角色，**保守認定無核心 IL**）
- O/U 修正：0（無核心 IL 觸發）
- ML 修正：0

> 若實際有核心 IL 未能識別，此為潛在向下風險（雙方皆可能）。

## 條件修正（Run Value 信號）

| 信號 | Run 修正 | 影響方 |
|------|---------|-------|
| Park Factor Coors 4 月 ~112 | (112−100) × 0.05 = **+0.6 run**（每隊）| 雙方各 +0.6 |
| Feltner vs RHB 劣勢 + SD 多 RHB | +0.4 run | 加到 SD 得分 |
| Waldron knuckleball at altitude | +0.3 run（保守）| 加到 COL 得分 |
| 雙方皆 Below Average 投手 | — | 不觸發 Solid+ 下修 |
| SD Cold BABIP 未極端 | 0 | 不扣 cold run value |

## 修正後預期得分（手動估算，供驗算）

- **基礎公式**（PF=112 April Coors）：
  - E[SD]  = 4.3 × (.322/.315) × (6.96/4.00) × 1.12 ≈ **8.56**
  - E[COL] = 4.3 × (.318/.315) × (4.77/4.00) × 1.12 ≈ **5.80**
- **信號修正後**：
  - SD: 8.56 + 0.4 = **8.96**
  - COL: 5.80 + 0.3 = **6.10**
  - **Total ≈ 15.06**（vs O/U line 11 → 差距 +4.06 ⭐⭐⭐⭐⭐ 方向）

## 近期狀態 + H2H
- SD 近 10 場 8-2（但最近剛輸 4/23 早場 3-8）/ 近 30 16-8 / 本季 16-8
- COL 近 10 場 4-6 / 本季 10-15 / 近期剛斷連敗
- H2H：SD 對 COL 本季多次碰面、近期優勢明顯（4/10–4/12 在 SD 主場全 4-0、4/22 在 Coors 贏 1-0）
- SD 作客 Coors 表現：歷史上 SD 打者群 vs Coors 友好（DBK 強邊 + 陣容 R-heavy）

## 整體判斷（方向 + 信心 + 風險）

- **方向性**：
  - 總分 — **強烈偏 OVER**（Coors PF + Feltner vs RHB 劣勢 + 無 Ace 壓制；唯一 Under 風險是 SD Cold 打線延續）
  - ML — **偏 SD**（投手差 ~2.2 run xERA、SD 近 30/16-8、SD 陣容對 Feltner 劣勢+過往有壓制）
- **信心程度**：中等
  - OVER 方向：高信心（PF + 對決基本面一致）
  - ML 方向：中等（Coors 隨機性高、SD 打線冷、Waldron knuckleball 變數）
- **值得注意的風險**：
  1. Waldron knuckleball 在高海拔是雙面刃 — 可能「效果下降 → 大失分」也可能「因對手打者抓不到節奏而意外好」
  2. SD 打線 last7 BABIP .272（運氣差），若 SD 打擊真的回歸正常 → OVER 加碼
  3. 早季樣本（雙方 ~25 場），xwOBA/BABIP/ERA 皆帶高方差
  4. IL 核心角色無法從 roster names=null 鎖定（雙向潛在風險）

> ⚠️ 基本面分析結束 — 盤口推薦與星級由 Phase 4 `predict.py` 決定，此 summary 不含具體盤口。
