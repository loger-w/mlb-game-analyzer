# Phase 3 Summary — BAL @ KC (2026-04-21 ET 19:40)

## 先發投手對決

| 項目 | Shane Baz (BAL, R) | Kris Bubic (KC, L) |
|------|-------------------|-------------------|
| Tier | 🟢 Back-end Starter | 🟡 Solid Starter |
| 本季 ERA / xERA | 4.91 / 4.22 | 3.97 / 4.16 |
| K-BB% | 11.4% | 18.0% |
| FIP / xFIP | 3.65 / 3.70 | 3.28 / 3.33 |
| IP / GS | 22.0 / 4 | 22.67 / 4 |
| Avg Velo | 91.2 mph | 88.3 mph |
| Hard Hit% / Barrel% | 18.1% / 8.8% | 32.3% / 13.2% |
| Whiff% / CSW% | 8.7% / 24.8% | 13.2% / 28.7% |
| Prior Year (2025) | ERA 4.87, FIP 4.17 | ERA 2.55, FIP 2.78 |
| 年齡評估 | ⚡ 巔峰期 | ⚡ 巔峰期 |
| role_change | None | None |

**YoY Statcast 閘門**：兩位皆不觸發（Baz 4.91>4.87，Bubic 3.97>2.55 均未低於 prior ≥1）。

**Platoon（關鍵）**：
- **Baz vs L**：.365/.452/.538（BF62，足量）→ **嚴重劣勢**
- Baz vs R：.206/.229/.412（BF35）
- Bubic vs L：.167/.211/.444（BF19 小樣本）
- Bubic vs R：.180/.286/.328（BF70 足量）

**球種**：Baz FF33/KC31/FC22；Bubic FF46/CH19/ST15。Bubic velo 僅 88.3 但憑 spin+ST 取勝，對 BAL 右打為主的陣容 CSW 28.7 會造成壓力。

## 打線分析

| 項目 | BAL（客） | KC（主） |
|------|-----------|---------|
| Tier | 🟢 Weak | 🟢 Weak |
| Avg OPS / xwOBA | .622 / .306 | .647 / .305 |
| BABIP | .264 | .284 |
| K% / BB% | 26.8 / 10.2 | 24.5 / 9.5 |
| recent_heat | 🥶 Cold | 🥶 Cold |
| RS/G 近 30 | 4.26 | 3.17 |

**BABIP 回歸閘門**：BAL .264（非 ≤.260 strict），KC .284 接近均值。**皆未觸發 Hot/Cold 強制調整**，原始 Cold 判定保留但打折處理。

**關鍵 Platoon 適配**：
- **KC 左打群 vs Baz (R)**：Pasquantino vs R .571（PA69）、Caglianone vs R .853（PA53）、Jensen vs R .846（PA53）、Isbel vs R .738（PA50）— 多位左打 vs RHP 樣本充足且貼近 Baz 被左打屠殺的事實，**實質加成**
- **BAL 上位 vs Bubic (L)**：Henderson vL .927、Ward vL 1.169、Jackson vL 1.277（皆 PA<25 小樣本），Alonso vL .368 平庸 — **方向性支持但樣本薄弱**，不全量採計
- Bubic vs L 小樣本，真正壓力來自 BAL 右打群；Bubic vs R 歷史優異（.180/.286/.328）

## 牛棚

| 項目 | BAL | KC |
|------|-----|-----|
| Bullpen ERA | **3.50** ✅ | **6.37** 🔴🔴 |
| Closer 狀態 | Bautista 60d IL ⛔ | Estévez 15d IL ⛔ |
| 核心 IL 人數 | 2（Bautista + Kittredge 15d）| 1（Estévez）+ 深度弱 |

**牛棚雙向閘門**：
- **KC 牛棚 ERA 6.37 是全聯盟最差級別** — 即使 Estévez 回來也難救。Bubic 平均 5-6 IP → 後段 3-4 IP 由 6.37 ERA 群體承擔 → **BAL 後段得分預期大升**
- BAL bullpen 2 核心 IL，但整體 ERA 3.50 仍健康 → 深度強，IL 影響 partially baked in
- 淨效應：**對 BAL 總分有強烈上修傾向**

**O/U 修正**：+0.7 run 給 BAL（KC 牛棚崩壞超過標準「2 核心 IL」的 +0.5 程度）；+0.3 run 給 KC（BAL 2 核 IL 但深度補回大半）
**ML 修正**：KC -4%（牛棚災難）；BAL -2%（2 核 IL 但 ERA 仍健康）

## 條件修正

| 觸發信號 | Run Value |
|----------|-----------|
| KC 牛棚災難（ERA 6.37 vs 聯盟 ~4.1）| **+0.7 run 給 BAL** |
| BAL 牛棚 2 核 IL（ERA 仍 3.50）| **+0.3 run 給 KC**（保守，已部分體現）|
| KC 打線 vs Baz 左打適配 | +0.3 run 給 KC（Platoon 局部優勢）|
| 雙方 Cold（但 BABIP 未觸發極端）| 不強制下修，打折 |
| Park Factor 99（Kauffman）| 約 0 調整（中性）|
| 開季（24/23 games）| LOW confidence，tag `early-season` |
| 天氣 / 主審 | 未取（非核心） |

## 基礎期望得分（公式）

- 聯盟基準：R/G=4.50, xwOBA=0.318, ERA=4.10
- **E[KC]** = 4.50 × (.305/.318) × (4.91/4.10) × 0.99 ≈ **5.11**
- **E[BAL]** = 4.50 × (.306/.318) × (3.97/4.10) × 0.99 ≈ **4.15**

## 修正後期望得分

- KC：5.11 − 0.3（Cold 折）+ 0.3（Platoon 適配 Baz 左打）+ 0.3（BAL 牛棚 2 核 IL 名義）≈ **5.4**
- BAL：4.15 − 0.2（Cold 折）+ 0.7（KC 牛棚災難）≈ **4.65**
- **修正後總分 ≈ 10.05** vs O/U 9.5

✏️ 考量 Baz vs L 樣本放大的隱憂（可能壓不住 KC 左打）、BAL 打線 Cold 且 BABIP .264 偏低但未極端：
- 最終比分估算：**KC 5.2, BAL 4.8（Total ≈ 10.0）**
- 預期差距：KC 領先 ~0.4 run（微幅主場優勢 + Platoon 適配，儘管 KC 牛棚糟糕被 BAL 牛棚深度抵消回 BAL 偏優）

## 整體判斷

- **方向性**：近交易盤 BAL 小熱（Pinnacle BAL 1.81 / KC 2.14，隱含 BAL 55%）
- **我的基礎面**：KC 左打適配 Baz 劣勢 + 主場 → 對 KC 略有利；但 **KC 牛棚災難** 抵消並翻轉回 BAL 微利
- **修正後預估：BAL 微幅優勢 (~52-54%)** — 與市場方向一致但幅度較小
- **總分**：修正後 ~10.0 vs line 9.5 → 差距 ~0.5（**遠低於 1.5 噪音門檻 → O/U PASS 高機率**）
- **讓分**：差距預估 <1 run → Run Line -1.5 低機率 cover，**PASS 或受讓方**

## 風險 / 不確定性

1. **開季小樣本**：雙方先發僅 4 GS，打者 PA 50-110 → LOW confidence
2. Bubic vs L 樣本僅 BF19，若 BAL 左打群打開完全未知
3. Baz vs L 雖 BF62 足量但 wOBA 是否 sustainable 仍有變異
4. KC 連敗 8 場的 negative momentum 是否影響心理（不量化，僅標註）
5. 天氣/主審未取 — 若賽前發布極端風向/高/低溫主審，預測需重估
