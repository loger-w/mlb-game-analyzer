## 投手對決

### Bailey Ober (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p36, K-BB% p46），gap vs ERA-only = -4.1
  - 同意 Solid Starter 下緣。ERA 4.19 / xERA 3.84 / FIP 4.17 / K-BB% 9.5 / velo **84.1 avg**（極低）。velo 低限制 stuff，但球種混合 + 控球維持 Solid 表現。gap -4.1 微小已對齊。
- **Reverse platoon 信號**：未 fired（vs LHB SLG .429 / vs RHB SLG .337，差距 +0.092 接近 reverse 但未過 0.080 wOBA gap 標準）。
  - n/a
- **對手打線威脅**：高。MIA top 5 vs RHP 強過預期（Edwards .870 / Hicks 1.008 / Lopez .802 / Norby .814）+ Ober vs LHB SLG .429 → MIA 左打都有機會。Ober TTO3 OPS Δ +0.603（OPS 0.415→1.018）是重大警報。

### Eury Pérez (AWAY, RHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p69, K-BB% p69），gap vs ERA-only = +40.9
  - 部分同意。xFIP 3.86 + K-BB% 13.2 + velo **92.0 avg / 100.6 max** 結構優；但 ERA 5.01 + 3 GS 樣本 + vs RHB SLG .515 + Barrel 13.3% 顯示被打能力存在。實質 🟡 Solid Starter 上限（年輕成長股）。
- **Reverse platoon 信號**：未 fired（vs LHB .258/.368/.427 / vs RHB .221/.303/.515 — vs RHB SLG 高但 OPS 接近）。
  - n/a
- **對手打線威脅**：中。MIN top 5 vs RHP 強（Buxton .989 / Jeffers .954 / Martin .859 / Larnach .832）但 last7 個別 cold（Bell .407 / Lewis .263）。Pérez single-pitch FF 51.6% + vs LHB SLG .427 → Larnach 左打、Buxton 右打強拉 EV95 46.2 是威脅點。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 同意。Buxton / Jeffers / Larnach / Martin vs RHP 都過 .800 OPS；對 Pérez xERA 4.94 + Barrel 13.3% 結構性可被打投手有實質 edge。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #3-4 gap 0.307 fired（Jeffers .954 vs RHP → Bell .509）— middle chain 中斷影響 RBI 機會，−0.1 run。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 上修同意。MIA top 5 vs RHP 整支居然 .800+，整體 vs RHP 比 season Weak 強得多。Ober TTO3 OPS Δ +0.603 + vs LHB SLG .429 → MIA 多左打（Edwards/Stowers/Marsee/Morel/Hernandez）有 platoon 優勢。實質 🟠 Strong vs Ober 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #3-4 gap 0.224 fired（Lopez .802 → Stowers .716）— middle chain 微斷，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 5.54 / 7 / 1 | 3.37 / 3 / 2 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：**MIN 5.54 ERA 結構性差** + 1 core IL（Sands）。Ober TTO3 OPS 1.018 + K% -10.8pp 嚴重 → 5-6 局後高機率被換投，MIN 5.54 ERA 中繼會被持續打 — 本場最大 downside。
- AWAY 牛棚：MIA 3.37 ERA + 2 core IL（Fairbanks closer + Henriquez）— Fairbanks 是 closer，IL 影響度升級。9 局 close-out 風險 ↑，但前段牛棚仍可用。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.603（TTO1 0.415 → TTO3 1.018），第三輪明顯衰退
- 🟠 AWAY single-pitch dependent：主球種使用率 51.6%（≥45.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.049（TTO1 0.700 → TTO3 0.749），第三輪明顯衰退；K% 從 33.3% 掉到 22.5%（Δ -10.8pp）
- 🔴 HOME chain breaks at #3-4：OPS 落差 0.307
- 🟠 AWAY chain breaks at #3-4：OPS 落差 0.224
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **本場最強訊號 = Ober TTO3 OPS Δ +0.603（OPS 1.018 崩盤級）+ MIN 牛棚 5.54 + 1 core IL**。MIA 第3輪起到牛棚都有 edge；單側 +0.4 run 上界。

## 條件修正

- Park Factor: 106.0 → +0.30 run
- 天氣：Partly Cloudy, 70°F, wind 22 mph, Out To RF
  - 影響判讀：**22mph 出 RF 是 strong wind**（>20mph 必提）+ PF 106 + 70°F 中性 — 右打強拉 RF HR 機率大幅上修；Buxton (Barrel 20.5%) / Jeffers / Hicks (Barrel 7.1%) 右打都受惠 — Total +0.4 run（已部分含於 PF）。
- 先發 tier / doubleheader：Pérez Solid 上限 vs Ober Solid 下限 — SP 接近；但 Ober TTO3 崩盤 + MIN 5.54 牛棚是最大區別。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.4 | +0.3（AWAY single-pitch +0.2 + AWAY core IL ×2 Fairbanks +0.2 互動 max+0.1 −0.1 chain HOME） | 5.7 |
| AWAY | 4.6 | +0.3（HOME TTO3 +0.3 + HOME core IL +0.1 互動 max+0.1 −0.1 chain AWAY） | 4.9 |
| Total | 10.0 | +0.6（含 wind +0.3 已部分含於 PF） | 10.6 |

## 整體判斷

- **方向（基本面）**：持平（HOME 微傾）
- **總分（基本面）**：10.6（強 OVER 訊號）
- **方向信心**：53% — Ober TTO3 崩盤 + MIN 5.54 牛棚對 MIA 有利，但 MIN top 5 vs RHP 也強，加 22mph 出 RF 偏 right-handed MIN hitters；雙方接近。
- **風險**：
  1. **22mph 出 RF 風 + PF 106** — HR 機率大幅上升，OVER 是主訊號（distribution 厚尾向上）
  2. Ober TTO3 OPS 1.018 崩盤 → MIN 5.54 牛棚暴露
  3. Pérez 23 歲 100mph max — 可能單槍救主 6 IP 2 ER
  4. MIA / MIN 都連勝 +2 — momentum 對位無 edge

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
