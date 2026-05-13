## 投手對決

### Jovani Morán (HOME, LHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本 <30 BF，無打分），gap vs ERA-only = —
  - 樣本只 1 GS，tier_v2 無法判斷。原始：ERA 2.91 / xERA 3.48 / FIP 4.99 / K-BB% **6.6**（低）/ WHIP 1.25 / vs LHB .071/.212（極小樣本 33 BF）vs RHB .234/.362/.489（被打）。reliever-converted SP，ERA 2.91 為小樣本運氣假象。實質按 🟢 Back-end / Solid 邊緣處理。
- **Reverse platoon 信號**：未 fired（vs LHB 樣本太小無法穩定計算）。
  - n/a
- **對手打線威脅**：高。PHI top 3-5 vs LHP（Schwarber .796 / Harper .772 / García .877 / Stott 1.063 / Bohm .682）+ Hot last7（Schwarber 1.055 / Harper **1.568** / Marsh 1.126 / Bohm .911）— Morán vs RHB SLG .489 + K-BB% 6.6 → García/Realmuto/Schuemann 右打陣容 hunting zone。

### Zack Wheeler (AWAY, RHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p83, K-BB% p84），gap vs ERA-only = +14.7（< 15 沒 fire）
  - 同意 Strong Ace。ERA 3.12 / xERA 2.40 / FIP 2.81 / xFIP 3.26 / K-BB% 17.7 / WHIP 0.98 — 結構頂尖；age 35 📉📉 但 velo 89.3 仍維持。gap +14.7 接近但未過 ±15 fire 標準。
- **Reverse platoon 信號**：未 fired（vs LHB .139/.238/.250 / vs RHB .240/.269/.400 — vs RHB SLG 較高但 RHB 樣本小，未過 reverse 標準）。
  - n/a
- **對手打線威脅**：低。BOS vs RHP 整支結構性弱（Duran .544 / Story .489 / Mayer .577 / Narváez .535 / Durbin .545 — 多人 sub-replacement vs RHP）。Gasper #2 OPS 2.000 是樣本污染。Wheeler 結構性壓制 BOS chain。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 不同意 Strong。Gasper 2.000 OPS 是 1-2 PA 樣本污染，去除後 BOS top 5 vs RHP 多人 .500-.600 OPS — 實質 🟢 Weak vs RHP（與 season tier 一致）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #2-3 gap **1.150** — Gasper 2.000 樣本污染，**完全不採用**。實質 chain 連續性弱（Duran .544 → Abreu .827 → Yoshida .736 → Story .489），但這是 BOS 整體 vs RHP 弱的反映，非結構性 chain break。

### AWAY — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟢 Weak
  - 不同意 Weak。Schwarber/Harper/García top 3 vs LHP 都過 .770 OPS + Hot last7 — 整支看起來「Weak」是因 Turner .538、Crawford .206 等 vs LHP 數字拖低。對 Morán 1 GS reliever-converted SP 應上修至 🟡 Average vs LHP（top 3 是真實威脅）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #5-6 gap 0.257 fired（Marsh .680 vs LHP → Realmuto .488）— middle-back chain 中斷，影響 RBI 機會，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.39 / 6 / 1 | 3.96 / 3 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：BOS 3.39 ERA + 1 core IL（Coulombe LHP）— 中等深度，左側壓制變薄。PHI Schwarber/Harper 左打 cleanup 後段不再有 LHP specialist 應對。
- AWAY 牛棚：PHI 3.96 ERA + 0 core IL — 完整火力。Wheeler 7 IP 後鎖場順暢。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 AWAY TTO3 penalty：OPS Δ +0.142（TTO1 0.552 → TTO3 0.694），第三輪明顯衰退；K% 從 30.1% 掉到 26.2%（Δ -3.9pp）（career fallback）
- 🔴 HOME chain breaks at #2-3：OPS 落差 1.150
- 🟠 AWAY chain breaks at #5-6：OPS 落差 0.257
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - **本場最大訊號 = Wheeler Strong Ace + BOS top 9 結構性弱 vs RHP**。HOME chain_break #2-3 因 Gasper 樣本污染不採用；AWAY TTO3 Wheeler career +0.142 暗示第3輪可能被 BOS Yoshida/Abreu 等少數有 contact 的打者敲穿，BOS 6+ 局後可能 1-2 分。

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 天氣：Partly Cloudy, 62°F, wind 6 mph, L To R
  - 影響判讀：62°F 偏涼略利投手；6mph 橫風 L→R 無方向偏移影響；整體中性 0 run。Fenway HR -15% 已含於 PF 修正。
- 先發 tier / doubleheader：Wheeler Strong Ace > Morán Back-end 兩級以上；牛棚雙方接近但 PHI 0 IL 完整。Wheeler 期待 7 IP 1-2 ER。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.0 | +0.2（AWAY TTO3 Wheeler +0.2） | 3.2 |
| AWAY | 5.7 | +0.0（HOME core IL +0.1 −0.1 chain AWAY） | 5.7 |
| Total | 8.7 | +0.2 | 8.9 |

## 整體判斷

- **方向（基本面）**：AWAY (PHI)
- **總分（基本面）**：8.9（厚尾向 9.5+，因 Morán 樣本不確定 + PHI top 3 Hot）
- **方向信心**：70% — Wheeler 結構性壓倒 Morán 1 GS reliever-converted SP；BOS vs RHP 整支弱；PHI top 3 Hot 火力對 Morán 有 hunting zone。
- **風險**：
  1. Morán 1 GS 樣本，可能單場手感爆衝（reliever stuff playing up over short stint）
  2. Wheeler 35 歲，velo 任何下滑會放大 TTO3 風險（K% 30.1% → 26.2%）
  3. BOS Schwarber/Harper 都是左打，遇 LHP Morán 反而 vs LHP OPS .796/.772 不算崩盤
  4. Fenway 短左外野 + Schwarber 左打拉打 — 1 球可破壞分析

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
