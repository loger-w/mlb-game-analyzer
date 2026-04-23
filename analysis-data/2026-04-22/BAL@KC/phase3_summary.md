# Phase 3 Summary — 2026-04-22 BAL @ KC (Kauffman Stadium)

## 先發投手對決

### Michael Wacha (KC, R, 34y) 🟡 Solid Starter（xERA 定性）
- 本季：ERA 1.00 / xERA **3.47** / FIP 3.25 / xFIP 3.87 / WHIP 0.78 / IP 27 / 4 GS
- Statcast：avg_velo 87.6、whiff 11.4%、csw 28.4%、hard_hit 34.7%、xwOBA .298
- Platoon：vs L .136/.215/.254（65 PA）/ vs R .167/.242/.267（33 PA）→ 雙側皆壓制，**對 L 尤其極端**
- 年齡評估：📉📉 明顯退化（球速 87.6 偏低但 pitch mix 廣）

### YoY 對比結論（觸發：|ERA − xERA| = 2.47 ≥ 1.5）
| 指標 | 2025 | 2026 | 變化 |
|---|---|---|---|
| avg_velo | 86.6 | 87.6 | +1.0 mph ✓ |
| whiff % | 10.1 | 11.4 | +1.3 ✓ |
| csw % | 25.0 | 28.4 | +3.4 ✓ |
| hard_hit % | 23.0 | 34.7 | +11.7 ⚠️ 顯著變差 |
| xERA | 4.19 | 3.47 | -0.72 ✓ |
| ERA | 3.86 | 1.00 | -2.86（運氣/樣本） |

**判定**：球速與 whiff/csw 三項一致微升（new-version 特徵），但 hard_hit% 飆升指示被打質變差。**綜合仍為 🟡 Solid Starter（xERA 3.47 = 真實水平）**，ERA 1.00 是 BABIP / LOB% 僥倖。不按 ERA 1.00 而按 xERA 3.47 估算預期得分。

### Chris Bassitt (BAL, R, 37y) ⚪ Below Average
- 本季：ERA 6.19 / xERA **5.71** / FIP 5.10 / xFIP 5.91 / WHIP 2.13 / IP 16 / 4 GS（僅 4 IP/GS）
- Statcast：avg_velo 84.3（極低）、whiff 7.5%（差）、csw 22.8%、hard_hit 17.7%（反常低）、xwOBA **.372**、xBA **.306**
- Platoon：vs L **.341/.453/.463（9 BB, 17% BB% 崩盤）** / vs R .346/.433/.346
- 年齡評估：📉📉📉 快速退化（球速從生涯 90+ 掉到 84.3）
- 樣本小（16 IP）但 FIP 5.10 + xERA 5.71 + xwOBA .372 三項一致 → ⚪ Below Average 有信心

**投手差：2 個等級**（Wacha Solid vs Bassitt Below Average）→ 觸發「單方碾壓」劇本候選

---

## 打線評級

### KC 🟢 Weak（但今天面對 ⚪ 投手）
- avg_ops .655 / avg_xwoba **.307** / avg_babip .289
- 近 7 天 BABIP .313（正常）/ recent_heat ⚖️ Normal
- 核心火力：Witt Jr.（xwOBA .387 🔴）、Garcia (.339)、Caglianone (.342)、Jensen (.330)
- 沉重包袱：Pasquantino (.264, BABIP .159) / Perez (.284) / Collins (.225)
- **LHB 分布**（4-5 人）：Pasquantino L、Caglianone L、Jensen L、Isbel L、Collins L、Witt Switch → **對 Bassitt vs L .453 OBP 極度有利**

### BAL 🟡 Average
- avg_ops .683 / avg_xwoba .324 / avg_babip .276
- 近 7 天 BABIP **.200**（Cold）/ recent_heat 🥶 Cold
- 核心火力：Ward (.367)、Jackson (.338)、Basallo (.333)、Taveras (.392)
- 被低 BABIP 拖累：Henderson (.210 BABIP)、Basallo (.175)、Mayo (.211)
- **LHB 分布**（2-3 人）：Henderson L、Basallo L、Beavers L、Taveras Switch → 面對 Wacha vs L .215 OBP **極壓制，吃虧**

---

## BABIP 回歸判定

- **BAL 近 7 天 BABIP .200 ≤ .260** → Luck-driven Cold，回歸 .300
- 判定：**不扣 Cold run value**（保持 baseline 預期得分），追加 **+0.2 run 回歸上升**
- KC .313 正常，無須調整

---

## 牛棚評估

### KC 牛棚 🔴🔴 極差
- 整體 ERA **6.35**
- 核心 IL：**Carlos Estévez（closer, 15-Day）**、Bailey Falter（15-Day, 長救援）、Stephen Kolek（15-Day）
- 其他 60-Day：Alec Marsh、James McArthur
- 評估：closer + 1-2 中繼核心全倒 ≈ 2 核心缺陣；加上 6.35 整體 ERA，屬極差

### BAL 牛棚 🟡 中等（有補位）
- 整體 ERA 3.63
- 核心 IL：**Félix Bautista（前 closer, 60-Day）**、Andrew Kittredge（setup, 15-Day）
- 關鍵補位：**Ryan Helsley**（前 STL All-Star closer）已入 active roster 接 closer → 替補品質反向，IL 衝擊大幅下修
- 其他 IL：Kyle Bradish / Tyler Wells / Eflin（輪值傷兵，不納入牛棚）

---

## 牛棚雙向修正值

| 側 | O/U 修正（對手 +run） | ML 修正 |
|---|---|---|
| KC 牛棚 IL（2 核心 + 整體 6.35） | **對手 BAL +0.6 run** | KC ML **-3%** |
| BAL 牛棚 IL（Bautista 60-Day、但 Helsley 補位） | **對手 KC +0.3 run** | BAL ML **-2%** |
| **淨差**（扣抵後） | **BAL +0.3 vs KC 0** | KC 淨 -1% |

---

## 條件修正

- **Park Factor 99（Kauffman）**：(99-100) × 0.05 = -0.05 run → 忽略
- **Bassitt vs LHB 崩盤**（4-5 KC LHB）：**KC +0.3 run** Platoon 優勢
- **Wacha vs LHB 壓制**（2-3 BAL LHB）：**BAL -0.2 run** Platoon 劣勢
- **Bassitt IP/GS = 4**（早退）：牛棚提早出場 → KC 已低水準牛棚提前用，額外 **BAL +0.2 run**
- **年齡退化**：Wacha 34y、Bassitt 37y → 當季 Statcast 已反映，不重複扣

---

## 近期狀態

- **KC**：近 10 場 **2-8**（RS 3.5 / RA 5.6）↓ 嚴重下滑；BABIP .313 正常 → 真實狀態確實差；但今天打線升級 + 投手優勢應翻轉
- **BAL**：近 10 場 4-6（RS 4.9 / RA 5.2）→ 持平；近 7 天 BABIP .200 回歸預期上升
- **系列賽**：1-1（4/21 KC 6-5 險勝）→ H2H 接近

---

## 修正後預期得分

### 基礎模型（xwOBA × xERA）
- BAL 得分基礎 = 4.5 × (.324 / .315) × (5.71 / 4.20) × 0.99 = **3.78**
- KC 得分基礎 = 4.5 × (.307 / .315) × (3.47 / 4.20) × 0.99 = **3.63**

### 信號修正疊加
| 信號 | KC | BAL |
|---|---|---|
| 基礎 | 3.63 | 3.78 |
| KC 牛棚 IL 2+ 核心 | — | +0.6 |
| BAL 牛棚 IL（Helsley 補位後） | +0.3 | — |
| Bassitt vs LHB Platoon（KC 4-5 LHB） | +0.3 | — |
| Wacha vs LHB Platoon（BAL 2-3 LHB） | — | -0.2 |
| Bassitt 早退 IP/GS=4 | — | +0.2 |
| BAL BABIP 回歸（.200 → .300） | — | +0.2 |
| **修正後** | **4.23** | **4.58** |
| **總分** | **8.81** | |

⚠️ **注意：基礎模型在 xERA 收斂後 KC-BAL 得分差縮小**（Bassitt xERA 5.71 vs Wacha xERA 3.47，差 2.24，但同時 KC 打線 xwOBA 比 BAL 低 0.017），加上 BAL 牛棚優勢、BAL BABIP 回歸，最終**比分優勢實際偏 BAL 而非 KC**。

---

## 整體判斷

- **方向傾向**：**BAL 略占上風**（比分 4.58 vs 4.23，差距 0.35）
- **信心程度**：**低（硬幣翻轉區）** — 名目上 Wacha ERA 1.00 壓制 Bassitt 6.19 看似 KC 碾壓，但 xERA 收斂 + BAL 打線基礎面略優 + KC 牛棚極差 + BAL BABIP 回歸 → 實際近均勢
- **關鍵風險**：
  1. Wacha 的 ERA 1.00 若持續（BABIP 運氣延伸）則 KC 得分優勢兌現 → KC 才是真正贏家
  2. Bassitt 小樣本變異大，單場可能突然回穩
  3. KC 打線近 10 場 RS 3.5 動能極差，xwOBA 估算可能高估即時得分
- **盤口方向（基本面）**：總分 8.81 略低於 OU 9.25 → 基本面偏 **UNDER**；ML 偏 BAL 但差距微小

---

## 市場對照（Pinnacle 2026-04-22 04:00 ET）

| 市場 | 價格 | 隱含% |
|---|---|---|
| ML BAL | 2.14 | 46.7 |
| ML KC | **1.79** | **55.9** |
| OU Over 9.0 | 1.87 | 53.5 |
| OU Under 9.0 | 2.00 | 50.0 |
| RL BAL +1.5 | 1.56 | 64.1 |
| RL KC -1.5 | 2.58 | 38.8 |

**市場傾向 KC**（Pinnacle 隱含 ML 55.9%）但本分析傾向 BAL（基本面） → 存在 **divergent** 傾向，D1 模型覆蓋紀律會要求比對 ml_lean / formula_lean。等 predict.py 輸出。
