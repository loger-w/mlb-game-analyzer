## 投手對決

### Spencer Miles (HOME, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 0 / 4.7 IP 太薄 — 緊急 bullpen-to-starter 或 opener 角色）。原始 tier 🟡 Solid Starter — ERA 3.50 / xERA 3.58 / FIP 3.60 / xFIP 3.53 / K-BB% 14.6%。
  - **謹慎按 🟢 Back-end ~ 🟡 Solid 對待**：4.7 IP 樣本太薄，所有 ERA/FIP 數值都不可信；可能僅 2-4 IP 後接力。
- **vs LHB 弱點（.282/.300/.462，40 BF）**：對 LHB 弱（OPS .762），LAA 中段 LHB 多（Soler R, Adell R, Schanuel L #6）— LHB 攻擊點分散。
- **對手打線威脅**：🟡 中等。LAA matchup tier 🟢 Weak (vs RHP) — Trout vs RHP **.939** + Soler .809 + Neto .706 — Trout 是真實威脅，其他人 vs RHP 中等。

### José Soriano (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +1.1
  - **同意但謹慎**：ERA 1.74 / xERA **3.87** / FIP 3.40 / xFIP 2.93 / K-BB% **18.8%** / WHIP 1.09 / velo 93.5（max 101.3）— FIP-base 真實 elite 但 xERA 3.87 顯示 ERA 部分運氣。本場按 🟠 Strong Ace 對待較合理（ERA 2.5-3.0 區間）。
- **Flag 8 era_xera_delta=-2.13**：嚴重運氣加持，BABIP 偏低 + LOB% 偏高，但 K-BB% 18.8% 真實 elite。
- **TTO3 penalty（🟠 +0.148）**：第三輪 OPS 上升，TOR 中段（Okamoto/Sánchez/Straw）第三輪可能反彈。
- **對手打線威脅**：🟡 中等。TOR matchup tier 🟡 Average (vs RHP) — Guerrero vs RHP .751 / Okamoto vs RHP **.814** last7 1.123（火燙）/ Straw .952 — Okamoto 是真實爆分點，但深度有限。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 上修一檔；Okamoto last7 1.123 補強中心。
- **chain_break 信號（🔴 #8-9）**：Straw .814 → Heineman .378 — 嚴重，但 #8 是高 OPS 點，影響在 9 棒回頭時。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修；Trout 之外進攻深度差。
- **chain_break 信號（🔴 #7-8）**：Peraza .798 → Rivero .450 — 嚴重後段斷層。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.08 / 7 / **1 名（García IL60d）** | **5.42** / 5 / **1 名（Joyce 預期 IL）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（TOR）：ERA 4.08 中段稍弱，1 核心 IL（García setup）→ 🟠 中高。Hoffman (closer) 健康。對 LAA 弱進攻仍 OK。
- AWAY 牛棚（LAA）：ERA **5.42** 嚴重崩盤。連敗 2 + 守崩中。若 Spencer Miles 早下，LAA 牛棚對 TOR 中心（Okamoto/Straw）會被持續吃。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.13):
  - **運氣加持但真實水平強**：Soriano K-BB% 18.8% / FIP 3.40 是真實 elite；ERA 1.74 vs xERA 3.87 部分回歸壓力，但本場仍按 Strong Ace 對待。**不自動下修**。

### 額外信號
- 🟠 AWAY TTO3 penalty +0.148 — TOR 第三輪反彈關鍵。
- 🔴 HOME chain break #8-9 — TOR 後段熄火。
- 🔴 AWAY chain break #7-8 — LAA 後段嚴重熄火。
- 🟠 HOME 牛棚 1 核心 IL — 影響輕。

## 條件修正

- Park Factor: 99.0 → -0.05 run（Rogers Centre 中性）
- 天氣：室內（Roof Closed，不適用）
- 先發 tier：HOME Spencer Miles 樣本零 + 早下 vs AWAY Soriano 🟠 Strong Ace → 不對稱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.5 | 0（核心 IL 1 名，未達 ≥2 門檻） | 3.5 |
| AWAY | 3.9 | 0（核心 IL 1 名，未達 ≥2 門檻） | 3.9 |
| Total | 7.4 | 0 | 7.4 |

## 整體判斷

- **方向（基本面）**：**AWAY (LAA) 略有利**。Soriano 真實 Strong Ace vs Spencer Miles 樣本太薄 + LAA 連敗 2 但 Trout 中心仍可吃 Miles → LAA 進攻面有空間。但 LAA 牛棚 ERA 5.42 是嚴重隱憂，TOR Okamoto/Straw 末段可能反咬。整體微偏 AWAY 但不強。
- **總分（基本面）**：**7.4 接近實際**，落點 6.5-9.0。Soriano 強壓制 + Miles 樣本不確定 + 雙方後段 chain break → Total 中性。
- **方向信心**：~55%（AWAY），微偏但 LAA 牛棚崩 + TOR G1 14:1 大勝信心面對立。
- **風險**：
  1. Miles 樣本 4.7 IP — 本場可能 2-3 IP 後接力，TOR 牛棚 ERA 4.08 對 LAA Trout 等仍 OK
  2. Soriano ERA 1.74 vs xERA 3.87 — 部分回歸壓力存在
  3. LAA 連敗 2 + G1 1:14 慘敗 — 信心面崩盤，本場可能延續壓制
  4. Okamoto last7 1.123 火燙 — 真實爆分點，可能單棒打破投手戰

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
