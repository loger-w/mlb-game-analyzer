## 投手對決

### Brady Singer (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p52, K-BB% p42），gap vs ERA-only = +35.0
  - 不同意 Solid Starter。ERA 5.63 / xERA 5.77 / FIP 5.24 / xFIP 4.09 / K-BB% 9.0 / **vs LHB .374/.414/.571(!)** 被左打狠打 — 結構是 Below Avg。tier_v2 受 xFIP 4.09 拉抬但 contact 證據壓倒；實質 ⚪ Below Average。
- **Reverse platoon 信號**：未 fired（vs LHB SLG .571 vs RHB SLG .507 — vs LHB 更高但 RHB 也被打）。
  - 雖未 fire，vs LHB SLG .571 顯示對左打吃虧；WSH 多左打（Wood / García / Abrams / Lile / Ruiz）— Singer 反 platoon 對 WSH 是優勢。Single-pitch SI 48.4%.
- **對手打線威脅**：極高。WSH top 5 vs RHP（Wood .969 / García .692 / House .566 / Abrams 1.052 / Young .558）— Wood / Abrams 兩個左打 ace vs RHP + Singer vs LHB .571 SLG → WSH 左打 cleanup 完美 hunting zone。

### Miles Mikolas (AWAY, RHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p70, K-BB% p32），gap vs ERA-only = +42.0
  - 不同意 Solid Starter。ERA **7.44** / xERA 4.81 / FIP **6.34** / xFIP 3.84 / K-BB% 7.6 / age **37 📉📉📉 快速退化** / Barrel 11.5%（被打 barrel 高）— 結構是 Below Avg / 老化嚴重。tier_v2 受 xFIP 3.84 拉抬但 ERA/FIP/被擊球品質證據壓倒；**era_xera_delta +2.63（Flag 8）** = ERA 比 xERA 高但 K-BB 7.6 + Barrel 11.5 結構不行。實質 ⚪ Below Average。
- **Reverse platoon 信號**：未 fired（vs LHB .301/.354/.534 / vs RHB .279/.328/.541 — 兩邊都被打）。
  - 整支被打，無 platoon edge 可言。
- **對手打線威脅**：高。CIN top 5 vs RHP（Friedl .562 cold / Steer .714 / De La Cruz .834 / Stewart .739 / Bleday 1.194 vs RHP!） — Bleday 個別 elite vs RHP，整體中等。Mikolas vs LHB SLG .534 + vs RHB SLG .541 → CIN 全打線都有機會。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 上修同意。De La Cruz / Bleday / Stewart / Steer vs RHP 整體強 + Mikolas 兩邊被打 → 評估維持 🟠 Strong vs Mikolas 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #5-6 gap 0.410 fired — Bleday 1.194 → McLain .596 vs RHP（middle-back chain 大幅斷層）— Bleday 後段火力斷層，限制 RBI 機會，−0.2 run。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 上修同意。Wood .969 vs RHP + Abrams 1.052 vs RHP 兩人 elite，House / Tena last7 1.297 + Singer 結構性弱（K-BB 9.0 + vs LHB .571）→ 評估 🟠 Strong vs Singer 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #4-5 gap 0.324 fired — Abrams 1.052 → Young .558 vs RHP（middle-back chain 中斷）— 影響 RBI 機會，−0.2 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.46 / 4 / 2 | 4.84 / 7 / 2 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：CIN 4.46 ERA + 2 core IL（Ferguson + Pagán）= 🔴 高。後段顯著吃緊，Singer 5-6 局後被換投高機率，CIN 中繼受壓。
- AWAY 牛棚：WSH 4.84 ERA + 2 core IL（Beeter + Kranick）= 🔴 高。Mikolas 4-5 局後被換投高機率（ERA 7.44），WSH 中繼也吃緊。**雙方牛棚都崩盤級**。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+2.63):
  - 雙刃。ERA 7.44 高估 vs xERA 4.81 = 可能單場運氣回正（WSH 失分 ↓），但 K-BB% 7.6 + Barrel 11.5 + age 37 結構性差未解除。預期 5.5-6.5 ERA 區間。不自動下修預測。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 48.4%（≥45.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.068（TTO1 0.711 → TTO3 0.779），第三輪明顯衰退；K% 從 18.9% 掉到 14.5%（Δ -4.4pp）（career fallback）
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.410
- 🔴 AWAY chain breaks at #4-5：OPS 落差 0.324
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **本場最大訊號 = 雙方爛 SP + 雙方爛牛棚 + GABP HR +29%（隱性 HR 工廠）**。亂場警報；Total distribution 厚尾向上 ≥ 14。Wood/Abrams/Bleday 任何一根 HR 改變總分；單槍場景多。

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 天氣：Clear, 77°F, wind 7 mph, R To L
  - 影響判讀：77°F 中性偏暖（球皮鬆 slightly 利攻）；7mph 橫風 R→L 無方向偏移影響 — 整體中性，+0.1 total。**GABP HR +29% 是本場真正的 OVER 驅動**（已含於 PF runs 104，但 HR factor 比 runs factor 還偏 OVER）。
- 先發 tier / doubleheader：雙方都 Below Avg；雙方牛棚都 🔴 高 IL。亂場屬性，無 SP 方向 edge。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 7.8 | +0.1（AWAY core IL +0.2 + AWAY TTO3 +0.1 互動 max+0.1 −0.2 chain HOME = +0.1） | 7.9 |
| AWAY | 6.0 | +0.1（HOME core IL +0.2 + HOME single-pitch +0.1 互動 max+0.1 −0.2 chain AWAY = +0.1） | 6.1 |
| Total | 13.8 | +0.2 | 14.0 |

## 整體判斷

- **方向（基本面）**：持平（CIN 微傾）
- **總分（基本面）**：**14.0**（卡上最高 Total，亂場警報 🚨）
- **方向信心**：52%（卡上最低之一 — 基本面持平） — 雙方都 Below Avg SP + 雙方都 🔴 高 IL 牛棚 + 雙方都有 elite vs RHP 個別打者；distribution 極寬。CIN 微傾僅因主場優勢 + Bleday/De La Cruz 個別威脅 + Mikolas Flag 8 +2.63 可能延續高 ERA。
- **風險**：
  1. 🚨 **Total OVER 風險最大** — 14.0 base + GABP HR +29% + 雙方爛 SP + 雙方 🔴 高 IL 牛棚
  2. ⚠️ Mikolas Flag 8 +2.63 — 可能單場「運氣回正」（WSH 失分 ↓），但結構性差未解除
  3. Wood / Abrams / De La Cruz / Bleday — 4 個 vs RHP .830+ 打者，任何一根 HR 改變總分
  4. CIN 連敗 → 反彈面 vs WSH 連敗 → 抵銷，無明顯 momentum edge

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
