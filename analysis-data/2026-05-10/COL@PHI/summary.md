## 投手對決

### Cristopher Sánchez (HOME, LHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +3.6
  - **同意**：ERA 2.42 / xERA 3.10 / FIP **2.29** / xFIP 2.24 / K-BB% **22.0%** / WHIP 1.34 / velo 90.1 — FIP-base 真實 elite。+3.6 gap 表示 ERA 微微沒運氣。本場按 🔴 Elite Ace 對待。
- **vs LHB 強壓制（.146/.180/.208，50 BF）**：對 LHB 接近完美壓制；但 vs RHB **.306/.361/.424**（159 BF）顯著弱。
  - COL 中段 RHB 多（Goodman/Castro/Doyle/Karros/Tovar 多 RHB），可能踩中 Sánchez 弱點 — 但 COL vs LHP 整體弱。
- **單一球種依賴（🟠 SI 45.4%）**：SI/CH/SL 三球種，COL 後段打者可能 sit sinker。
- **對手打線威脅**：🟢 低。COL matchup tier 🟡 Average (vs LHP) — Beck vs LHP **.848** / Goodman .747 / McCarthy .857 / Moniak .572 — 攻擊點分散，深度不足。

### Tomoyuki Sugano (AWAY, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p44, K-BB% p30），gap vs ERA-only = -25.6
  - **同意 + 強調 ERA 嚴重高估真實水平**：ERA 3.41 看起來中段，但 xERA **6.20**（極差）/ FIP 4.91 / xFIP 4.21 / K-BB% **7.3%** / whiff% 8.2% / barrel% **15.3%**（極高）/ velo 88.3 — 真實水平 ⚪ Below Average（5.5+ ERA 區間）。36 歲明顯退化、ERA 失真。-25.6 gap 嚴重。
- **Flag 8 era_xera_delta=-2.79**：xERA 6.20 vs ERA 3.41 是極端 -2.79 gap，barrel% 15.3% 顯示接觸品質崩。本場按 ⚪ Below Average 對待。
- **對手打線威脅**：🔴 極高。PHI matchup tier 🟡 Average (vs RHP) — 但 Schwarber vs RHP **.966** / Harper **1.026** last7 1.490（火燙）/ Marsh .911 last7 1.040 — 中心 2-5 棒對 Sugano 是 dream matchup；HR friendly Citizens Bank 推升爆分機率。

## 打線評級

### HOME — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；前 5 棒火力齊備（Schwarber/Harper/Marsh）。
- **chain_break 信號（🟠 #3-4）**：Harper 1.026 → García .636 — 中度斷層，但 Marsh #5 補強（.911）。前 5 棒密集爆分機會大。
- **platoon advantage（🟠）**：top 5 中 4 人 vs RHP OPS > season +0.050 → 對 Sugano 是強配對。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟡 Average — 比 season tier 上修一檔（vs LHP 部分打者表現好），但 Sánchez vs LHB 接近完美壓制 → 實際下修。
- **chain_break 信號（🔴 #6-7）**：Moniak 1.022 → Karros .645 — 嚴重斷層，COL 攻擊密度極差。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.02 / 3 / **0 名核心** | 4.46 / 4 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（PHI）：ERA 4.02 中段稍弱，無核心 IL。Strahm/Alvarado 等 setup 健康，Duran (closer) 健康。對 COL 弱進攻仍 OK。
- AWAY 牛棚（COL）：ERA **4.46** 偏弱，無核心 IL 但整體深度有限。若 Sugano 早下，PHI 中段攻擊將吃 COL 中繼。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.79):
  - **嚴重結構性 + 極端 barrel%**：Sugano xERA 6.20 / barrel% 15.3% / K-BB% 7.3% 全指向 ⚪ Below Average 真實水平。本場 PHI 多強打 + 球場 HR +16% → ERA 3.41 不可持續。**不自動下修**，敘事上 PHI 失分基準應該按 6.0+ 而非 base 5.6（base 已接近真實但偏低）。

### 額外信號
- 🟠 HOME single-pitch dependent SI 45.4% — 影響輕。
- 🟠 HOME platoon advantage top 5 中 4 人對 Sugano 強配對 — 推升 PHI 攻擊。
- 🟠 HOME chain break #3-4 — 但 Marsh 補強，影響輕。
- 🔴 AWAY chain break #6-7 落差 0.377 — COL 攻擊深度極差。

## 條件修正

- Park Factor: 104.0 → +0.20 run（Citizens Bank Park runs 104 + HR **+16%**）
- 天氣：Sunny 79°F, wind 8 mph **Out To RF** — 順風右外野推 LHB pull HR（Schwarber/Harper/Marsh 全 LHB）→ 強推升 PHI HR
- 先發 tier：HOME Sánchez 🔴 Elite Ace vs AWAY Sugano 真實 ⚪ Below Average → 嚴重不對稱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.6 | 0（核心 IL 0 名） | 5.6 |
| AWAY | 2.5 | 0（核心 IL 0 名） | 2.5 |
| Total | 8.1 | 0 | 8.1 |

## 整體判斷

- **方向（基本面）**：**HOME (PHI) 強勢有利**。Sánchez Elite Ace + COL 弱進攻 + chain break #6-7 → COL 進攻面被壓制。Sugano 真實 ⚪ Below Average（xERA 6.20 / barrel% 15.3）+ PHI 強打 + Citizens Bank HR +16% + Out To RF → PHI 進攻面爆分機率高。
- **總分（基本面）**：**8.1 接近實際但偏低**，落點 7.5-10.0。Sánchez 強壓制限制 COL 上限，但 PHI 攻擊面可能爆分（base 5.6 偏低，實際 6-7 區間）→ Total 上行壓力。
- **方向信心**：~72%（HOME），結構性數據強支撐（Sugano ERA 失真嚴重 + PHI 強打配對）。
- **風險**：
  1. **Sugano ERA 3.41 vs xERA 6.20**：本場可能繼續運氣加持（5 IP 2R），但 barrel% 15.3% 表示遲早爆掉
  2. PHI Harper last7 OPS **1.490** — 火燙不可持續但本場可能延續，極端高 PHI 得分（8+）也可能
  3. Sánchez vs RHB OPS .785 是真實弱點，COL 中段 RHB 多 — 雖然 COL 整體弱但個別打者可吃
  4. Out To RF 風 + Citizens Bank LHB pull HR — Schwarber 是 single-game HR 機率高的點，可能單場 2 HR

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
