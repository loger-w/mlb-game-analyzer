## 投手對決

### Keegan Akin (HOME, LHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本太薄，3 IP / 0 GS — 是 bullpen-to-starter 緊急轉換或 opener 角色）。原始 tier ⚪ Below Average。
  - **不適用 starter tier**：ERA **11.12** / xERA 5.28 / FIP 5.92 / xFIP 2.94 — xFIP 看起來不錯但 6.9% whiff% + 控球崩 → 真實水平 🟢 Back-end ~ ⚪ Below Average。LHB / RHB 雙邊都被打（vs LHB .364 / vs RHB **.438**）但樣本各 ~12-16 BF 太薄。本場按 ⚪ Below Average 對待。
- **Flag 8 era_xera_delta=+5.84**：嚴重結構警訊，ERA 11+ 即使有 BABIP 加持也是真實爆掉。
- **單一球種依賴（🟠 FF 54.9%）**：FF-heavy LHP，OAK 強打對 RHB 多的可以針對性 sit fastball。Akin 8 GS = 0，本場可能僅 2-3 IP 後接力。
- **對手打線威脅**：🔴 極高。OAK matchup tier 🟡 Average (vs LHP) — 但 Langeliers vs LHP **1.080** last7 1.304 / Cortes vs LHP **1.542** / Kurtz .621 / Rooker .472（差 vs LHP 但本場可能反向）— Langeliers + Cortes 是真實爆分點，Akin 1-2 IP 內可能掉 3-4 分。

### Luis Severino (AWAY, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p57, K-BB% p47），gap vs ERA-only = +2.8
  - **同意**：ERA 4.15 / xERA 4.47 / FIP 4.35 / xFIP 4.02 / K-BB% 9.7% — 各項一致 🟡 Solid。32 歲初期退化但本季數據撐住。
- **TTO3 penalty（🟠 -0.570）**：OPS TTO1 0.906 → TTO3 0.336 — 表面看是反向（TTO3 反而更壓制），但這是樣本噪音；K% 從 23.5% 掉到 16.3% 顯示真實第三輪壓制力下降。
- **vs LHB 弱點**：vs LHB .258/.378/.398 (111 BF) — OBP .378 偏高，控球對 LHB 較差。BAL 中段（Rutschman switch / Basallo）可吃。
- **對手打線威脅**：🟠 高。BAL matchup tier 🟠 Strong (vs RHP) — Rutschman vs RHP **.992** last7 .877 / Alonso .807 last7 .940 / Basallo .813 last7 .982 / Taveras .855 — 中心 3-6 棒對 Severino 是真實威脅。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong — 比 season tier 上修一檔；中心 Rutschman/Alonso/Basallo vs RHP 全 .800+ OPS。
- **chain_break 信號（🟠 #3-4）**：Rutschman .966 → Alonso .767 — 輕度，但 Alonso last7 .940 補強，影響輕。

### AWAY — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟡 Average — 與 season tier 一致；但 Langeliers + Cortes vs LHP 全爆表（1.080 / 1.542）。
- **chain_break 信號（🔴 #6-7）**：Cortes .992 → Butler .568 — 嚴重，但 OAK 前 5 棒 vs Akin 是 dream matchup；Butler/Gelof 第 7-8 棒 vs LHP 也不算太差。
- **lucky-hot 警訊（🟠 last7 BABIP .359）**：last7 OAK 火燙含運氣成分，部分回歸壓力，但 Langeliers/Cortes 的 EV/Barrel 是真實水平。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.56 / 8 / **2 名（Bautista closer + Helsley setup）** | 4.75 / 1 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（BAL）：ERA 4.56 偏弱 + **2 核心 IL 🔴 高**（Bautista closer + Helsley setup 雙缺陣）→ OAK 中段火力末段爆分機率極高，特別 7-9 局。
- AWAY 牛棚（OAK）：ERA 4.75 偏弱但無核心 IL，Miller (closer) 健康。對 BAL 強打（Rutschman/Alonso/Basallo）末段壓制力中等。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+5.84):
  - **嚴重結構性 + 樣本太薄**：Akin ERA 11.12 vs xERA 5.28 即使部分回歸仍是 5+ ERA 區間。本場按 ⚪ Below Average + 1-3 IP 預期離場。**不自動下修**，敘事上 OAK 失分基準應該按 base 6.5 + 大幅上修。

### 額外信號
- 🟠 HOME single-pitch dependent FF 54.9% — Akin FF-heavy 對 OAK 強打是噩夢。
- ℹ️ AWAY balanced 4+ pitches — Severino 多元球種對 BAL 強打前段是優勢。
- 🟠 AWAY TTO3 penalty -0.570（K% 顯示真實衰退）— BAL 第三輪反彈關鍵。
- 🔴 AWAY chain break #6-7 — OAK 後段斷層，但本場 Akin 太弱影響輕。
- 🔴 HOME 牛棚 2 核心 IL — 配合 Akin 早下，OAK 末段攻擊將吃 BAL 全部後段中繼。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Camden Yards 改造後 runs 96 偏輕度投手友善但 HR +7%）
- 天氣：Sunny 78°F, wind 8 mph **Out To RF** — 風吹向右外野，對 LHB pull HR（OAK Cortes/Soderstrom LHB）+ 一般 HR 都是順風助力 → 推升 HR 機率
- 先發 tier：HOME Akin ⚪ Below Average + 早下 vs AWAY Severino 🟡 Solid → 嚴重不對稱，AWAY 失分基準偏高
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.8 | 0（核心 IL 0 名） | 4.8 |
| AWAY | 6.5 | +0.8（HOME 牛棚 2 核心 IL → OAK 末段攻擊極大化，cap 上限） | 7.3 |
| Total | 11.3 | +0.8 | 12.1 |

## 整體判斷

- **方向（基本面）**：**AWAY (OAK) 強勢有利**。Akin ⚪ Below Average + 早下 + Langeliers/Cortes vs LHP 火燙 + BAL 牛棚 2 核心 IL → OAK 進攻面三重利好。Severino 🟡 Solid 中等水平但 BAL 中心打序強 → 雙方都有得分能力但 OAK 進攻優勢遠大於 BAL。
- **總分（基本面）**：**12.1（base 11.3 + +0.8 信號修正）**，落點 11.0-13.5。雙弱 starter + 雙方強打 + 順風 HR + BAL 牛棚崩 → Total 強上行。
- **方向信心**：~70%（AWAY），結構性數據（Akin 真實水平 + 牛棚 2 核心 IL）強支持。
- **風險**：
  1. **Akin 樣本僅 3 IP**：可能反常打出 3 IP 1R 賭運氣場面，但 xFIP 2.94 + xERA 5.28 衝突 → 本場仍多偏向爆掉
  2. OAK last7 BABIP .359 lucky-hot — 部分回歸壓力但 Langeliers/Cortes EV/Barrel 真實
  3. Severino 32 歲退化中 + BAL 強打 — 若 Severino 早下，OAK 牛棚 ERA 4.75 也可能被 BAL 中心打爆
  4. Camden 改造後球場特性混亂 + Out To RF 風 — HR 機率上升，極端高 Total（13+）也可能

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
