## 投手對決

### Grant Holmes (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p37, K-BB% p32），gap vs ERA-only = -5.2
  - 同意 Back-end Starter。ERA 4.34 / xERA 4.35 已對齊，FIP 5.00 偏差但 xFIP 4.34 拉回；K-BB% 7.5 低；vs LHB/RHB 均 .224/.227 抗左右手能力對稱。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：高。CHC top 5 Conforto 1.155 vs RHP / Suzuki .946 / Happ .908 / Busch .718 + Ballesteros .849 — 整支對 RHP 有殺傷力；Holmes vs LHB SLG .395 → Conforto / Happ / Busch 左打 hunting zone。

### Colin Rea (AWAY, RHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p95, K-BB% p73），gap vs ERA-only = +23.5
  - 部分同意。xFIP 3.23 + K-BB% 13.9 + vs RHB .193/.258/.298 結構是 Solid/Strong 之間；但 ERA 4.03 + vs LHB SLG .415 + age 35 📉📉 將 Strong Ace 拉回 Solid 中緣。實質 🟡 Solid Starter 上緣。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：中。ATL Olson 1.133 vs RHP（左打）+ Harris .958 + Smith .948 + Baldwin .847 — 火力強，但 Rea vs RHB SLG .298（壓 RHB）對 Harris / Baldwin / Riley 右打有壓制；Olson 是主要威脅點。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 不完全同意 Average。Olson 1.133 / Harris .958 / Smith .948 vs RHP 三人 ace 級，整體應上修至 Strong 邊緣；但 Rea vs RHB 強壓 + Wind In LF 削弱 → 本場評估維持 Average，矛盾因子抵消。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #7-8 OPS gap 0.887 fired — 但 #8 Kim .000（樣本污染）+ #9 Yastrzemski .516 真實弱 → 部分採用，−0.1 run。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 同意。Conforto 1.155 vs RHP + 整支深度 → 對 Holmes 7.5 K-BB% 結構性弱投手有威脅；維持 Strong。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #2-3 OPS gap 0.471 fired（Conforto 1.155 → Bregman .637 vs RHP）→ chain 頂端斷層 — Bregman cold 影響 RBI 機會，−0.2 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.31 / 7 / 1 | 3.87 / 9 / 4 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ATL 3.31 ERA + 1 core IL（Young）— 略受影響但整體深度仍 OK。對 CHC 末段威脅完整。
- AWAY 牛棚：CHC 3.87 ERA 看起來尚可，但 **4 core IL（Thielbar + Harvey + 2）= 🔴🔴 崩盤級**。Rea 若 5-6 局後被換投，CHC 末段防守完全失能；ATL Olson/Harris 後段 RBI 機會大幅放大。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +-0.070（TTO1 0.653 → TTO3 0.583），第三輪明顯衰退；K% 從 19.7% 掉到 15.2%（Δ -4.5pp）
- 🔴 HOME chain breaks at #7-8：OPS 落差 0.887
- 🔴 AWAY chain breaks at #2-3：OPS 落差 0.471
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×4：🔴🔴 極高（牛棚崩盤級）
  - CHC 4 core IL 是本場最大訊號 — Rea TTO3 衰退無關（−0.070），但中繼一旦上場立刻暴露結構性弱點；ATL 受惠 +0.5 run 上界。

## 條件修正

- Park Factor: 98.0 → -0.10 run
- 天氣：Partly Cloudy, 75°F, wind 7 mph, In From LF
  - 影響判讀：風 In From LF 7mph 是輕度逆風 → 壓制左打拉打 HR（Olson / Conforto 都左打），−0.2 total。
- 先發 tier / doubleheader：Rea Solid > Holmes Back-end；HOME 牛棚 3.31 + 1 IL 完整 vs AWAY 牛棚 3.87 + 4 IL 崩盤級 — 後段優勢明顯偏 ATL。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.1 | +0.4（AWAY core IL ×4 +0.5 −0.1 chain HOME） | 4.5 |
| AWAY | 5.8 | -0.1（HOME core IL +0.1 −0.2 chain AWAY） | 5.7 |
| Total | 9.9 | +0.1（弱風壓 HR -0.2 vs 互動 +0.3） | 9.7 |

## 整體判斷

- **方向（基本面）**：HOME (ATL)
- **總分（基本面）**：9.7（風壓抑略下修）
- **方向信心**：60% — CHC 4 core IL 崩盤級是主要 edge，加 Rea age 35 + ATL top 4 vs RHP 火力；信心受 Conforto vs RHP 1.155 + Rea vs LHB .415 SLG 抵消。
- **風險**：
  1. CHC 4 core IL → 一旦進入 6 局後 ATL 末段持續加分機率高
  2. Conforto 1.155 vs RHP — 單支 HR 可破壞分析（最高 EV95 53.1）
  3. Truist Park HR -5% + wind In LF → 雙方拉打 HR 壓制，UNDER 偏面
  4. Rea age 35 📉📉 任何 velo 退化跡象都會放大 ATL 進攻

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
