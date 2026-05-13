## 投手對決

### MacKenzie Gore (HOME, LHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p81, K-BB% p79），gap vs ERA-only = +53.2
  - 同意 Strong Ace。gap +53.2 = ERA 5.18 嚴重低估真實水平；xFIP 3.64 + K-BB% 15.4 + vs RHB .225/.310/.387 結構穩。屬「運氣壓低 ERA-only score 但 v2 認得」。不下修預測。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - fired Δ +0.085（vs LHB OPS .782 略高於 vs RHB OPS .697）。ARI 多右打（Marte/Perdomo/Arenado/Vargas/Moreno/Fernandez/Waldschmidt）— Gore reverse 效應僅對左打 Carroll 放大威脅，幅度 small。
- **對手打線威脅**：低。Gore vs RHB SLG .387 + xFIP 3.64，ARI 多右打反 platoon-disadvantage。主要威脅是 Carroll vs LHP 1.086（小心 1 球破壞分析）。

### Zac Gallen (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p52, K-BB% p33），gap vs ERA-only = +10.7
  - 不完全同意。K-BB% 7.7、vs LHB/RHB 雙邊被打、FIP 4.09 等同 xFIP — 結構性是 Back-end 邊緣。Solid Starter 偏樂觀。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：高。TEX 多右打，Gallen vs RHB .299/.341/.455 被打；Top 4 (Pederson/Nimmo/Seager/Jung) vs RHP 都過 .753 OPS，Jung .942 vs RHP 是 cleanup 威脅。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 同意。Top 4 Jung/Duran/Carter/Osuna vs RHP 都 .700+，Gallen 結構性可被打 → 評估上修至 Average 上緣。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #4-5 OPS gap 0.240 fired（Jung .942 → Carter .744）→ 中段 chain 中斷，影響大局得分連續性，−0.1 run。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong
  - 部分同意。Carroll 1.086 / Vargas .956 / Perdomo .762 vs LHP 數字漂亮，但 Cold last7 BABIP 0.229 → 整體出棒 timing 差。本場評估維持 Strong vs LHP，但 Cold 抵銷部分。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #5-6 OPS gap 0.396 fired（Vargas .956 → Gurriel .514 vs LHP）→ middle chain 嚴重斷層，−0.2 run；unlucky-cold ⏳ fired（BABIP 0.229）— 短期可能反彈，但 Gore xFIP 3.64 + K-BB 15.4 仍會壓制。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 2.68 / 6 / 0 | 4.34 / 6 / 3 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：TEX 2.68 ERA 全卡最強，0 core IL。Gore 7 IP 後完整火力銜接 → 末段防守極穩，對 ARI 末段得分機會極低。
- AWAY 牛棚：ARI 4.34 ERA + 3 core IL（Puk closer + Saalfrank setup + 1）= 🔴🔴 崩盤級。Gallen 若 5 局後被換投，ARI 末段隨時可能被連得分。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.229):
  - 可能反彈，但 Gore 真實水平壓制力強（xFIP 3.64、vs RHB SLG .387）→ 反彈幅度有限，不自動 ±run value。

### 額外信號
- 🟠 HOME reverse platoon Δ +0.085（vs LHB OPS 0.782 > vs RHB OPS 0.697）— LHP 對非預期手別反而吃虧
- 🔴 HOME TTO3 penalty：OPS Δ +0.423（TTO1 0.639 → TTO3 1.062），第三輪明顯衰退；K% 從 30.9% 掉到 24.2%（Δ -6.7pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.663（TTO1 0.676 → TTO3 1.339），第三輪明顯衰退；K% 從 11.0% 掉到 5.9%（Δ -5.1pp）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.240
- 🔴 AWAY chain breaks at #5-6：OPS 落差 0.396
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - ARI 3 core IL 崩盤級 + AWAY TTO3 +0.663 雙重後段壓力 → 若 Gallen 第3輪被打開，ARI 後段防守完全失能；TEX 受惠 +0.5 run 上界。

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：Gore Strong Ace > Gallen Back-end/Solid 一級；HOME 牛棚 2.68 vs AWAY 4.34（差 1.66 ERA）+ 3 core IL 差距，深度全面壓 ARI。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.4 | +0.6（AWAY core IL ×3 +0.5 + AWAY TTO3 +0.3 互動取 max+0.1 −0.1 chain HOME = +0.6 cap） | 5.0 |
| AWAY | 4.1 | +0.2（HOME TTO3 +0.3 + reverse platoon +0.1 互動 max+0.1 −0.2 chain AWAY = +0.2） | 4.3 |
| Total | 8.5 | +0.8 | 9.3 |

## 整體判斷

- **方向（基本面）**：HOME (TEX)
- **總分（基本面）**：9.3
- **方向信心**：65% — Gore 結構性壓 Gallen + TEX 牛棚 2.68 elite + ARI 3 core IL 崩盤；信心受 Carroll vs LHP 單槍威脅 + ARI cold BABIP 反彈面壓低。
- **風險**：
  1. ARI 3 core IL → 後段防守崩盤，TEX 即使僅取得小領先也能擴大
  2. Carroll 1.086 vs LHP + EV95 42.3 — Gore 仍可能被單槍敲穿
  3. 兩個 SP TTO3 都崩（Gore +0.423、Gallen +0.663）→ 4-5 局後雙方都被換投，但 TEX 牛棚 elite vs ARI 崩盤 — 該訊號偏 TEX
  4. ARI Cold BABIP 0.229 反彈面存在，但 Gore 真實水平壓制機會大

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
