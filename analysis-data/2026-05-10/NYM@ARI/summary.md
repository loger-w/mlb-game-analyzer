## 投手對決

### Eduardo Rodriguez (HOME, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p29, K-BB% p22），gap vs ERA-only = -53.1
  - **同意 + 強調 ERA 嚴重高估真實水平**：ERA **2.50** 看起來像 🔴 Elite Ace 但 xERA **4.72** / FIP 4.39 / xFIP 4.48 / K-BB% **6.0%**（極低）/ whiff% 7.0% / velo 87.9 — 真實水平 ⚪ Below Average ~ 🟢 Back-end（4.5+ ERA 區間）。-53.1 gap 是極端嚴重運氣加持。本場按 ⚪ Below Average 對待。
- **Flag 8 era_xera_delta=-2.22**：嚴重運氣加持。本場 NYM 中段（Alvarez/Baty/Benge）對 LHP 雖然 .544-.669 季度 OPS，但 Benge last7 1.067 火燙是真實爆分點。
- **TTO3 penalty -0.155**：第三輪反向，影響輕。
- **對手打線威脅**：🟡 中等。NYM matchup tier 🟢 Weak (vs LHP) — Bichette vs LHP .640 / Semien .754 / Alvarez .669 / Benge .539 last7 1.067 — 中段火力分散但 Benge 是真實爆分點。

### Huascar Brazobán (AWAY, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 1 / 3 IP 太薄 — opener 或緊急角色）。原始 tier 🟠 Strong Ace — ERA 1.53 / xERA 2.94 / FIP 3.10 / K-BB% 13.1% / WHIP 0.96 / velo 93.2。
  - **謹慎按 🟢 Back-end ~ 🟡 Solid 對待**：3 IP 樣本太薄；但 SI/CH 球種組合不錯（48.3% SI + 41.2% CH）。本場可能 2-4 IP 後接力。
- **單一球種 SI 48.3%**：球種組合不健全（FF 僅 5.6%）— ARI 中心可能難 sit fastball。
- **TTO3 penalty career fallback**：n/a 樣本太薄。
- **對手打線威脅**：🟡 中等。ARI matchup tier 🟡 Average (vs RHP) — Carroll vs RHP .709 / Marte .618 / Arenado vs RHP .810 last7 .865 / Vargas .896 — 中心 vs RHP 平均水平，但 Arenado/Vargas 是真實爆分點。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Arenado/Vargas vs RHP 強。
- **chain_break 信號（🟠 #5-6）**：影響輕。
- **last7 BABIP .203 Flag 3 unlucky-cold**：見風險段。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 比 season tier 下修；對 Rodriguez 雖然 ERA 假象但 vs LHB 仍可吃。
- **chain_break 信號（🟠 #7-8）**：影響輕。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.45 / 6 / **3 名（Puk + Saalfrank + 1 other）** | 3.87 / 7 / **3 名（Minter + Núñez + 1 other）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（ARI）：ERA **4.45** 偏弱 + **3 核心 IL → 🔴🔴 極高**（牛棚崩盤級）→ NYM 末段攻擊極大化。
- AWAY 牛棚（NYM）：ERA 3.87 中段稍弱 + **3 核心 IL → 🔴🔴 極高**（牛棚崩盤級）→ ARI 末段攻擊極大化。雙方都崩。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.22):
  - **嚴重運氣加持**：Rodriguez ERA 2.50 vs xERA 4.72 不可持續，K-BB% 6.0% 是真實偏弱。本場按 ⚪ Below Average 對待，**不自動下修**，但敘事上 NYM 進攻面 base 4.8 仍偏低。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.203):
  - **可能反彈**：ARI last7 BABIP .203 嚴重偏低（聯盟均 .300），Carroll/Marte/Vargas EV/Barrel 都不差，回歸壓力大。本場對 Brazobán 樣本零 → 反彈 + 數據反彈雙重，ARI base 3.3 偏低。

### 額外信號
- 🟠 AWAY single-pitch SI 48.3% — 球種組合不健全。
- 🟠 雙方 chain breaks — 影響輕。
- 🔴 雙方牛棚 3 核心 IL — 雙方末段攻擊都極大化關鍵。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Chase Field 中性，HR -18%）
- 天氣：未公布（跳過天氣分析）— Chase Field 屋頂可關閉
- 先發 tier：HOME Rodriguez ⚪ Below Average vs AWAY Brazobán 🟢 Back-end 樣本零 → 雙弱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.3 | +0.7（AWAY 牛棚 3 核心 IL → ARI 末段攻擊極大化） | 4.0 |
| AWAY | 4.8 | +0.7（HOME 牛棚 3 核心 IL → NYM 末段攻擊極大化） | 5.5 |
| Total | 8.1 | +1.4 | 9.5 |

## 整體判斷

- **方向（基本面）**：**AWAY (NYM) 略有利**。Rodriguez 真實 ⚪ Below Average + ARI 牛棚 3 核心 IL → NYM 進攻面有空間；Brazobán 樣本零 + NYM 牛棚 3 核心 IL → ARI 也有空間。雙方都有得分能力但 NYM 微優（Rodriguez 真實水平差更多）。
- **總分（基本面）**：**9.5（base 8.1 + +1.4 信號）**，落點 8.5-11.0。雙弱 starter + 雙方牛棚都崩 → Total 上行；Chase Field HR -18% 部分壓制。
- **方向信心**：~58%（AWAY），微偏；雙方都極端不確定（Flag 8 + Flag 3 + 雙方牛棚崩）。
- **風險**：
  1. **Rodriguez ERA 2.50 vs xERA 4.72** — 本場可能繼續運氣加持（5 IP 1R），Mets 本場可能投手戰
  2. ARI last7 BABIP .203 unlucky-cold — 本場大幅反彈機率高，特別 Brazobán 樣本零
  3. Brazobán 3 IP 樣本 — 可能任一方向結果
  4. 雙方牛棚 3 核心 IL — 後段 7-9 局可能成為總分主戰場，極端高 Total（11+）也可能

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
