## 投手對決

### Jeffrey Springs (HOME, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p76, K-BB% p77），gap vs ERA-only = +15.7
  - 部分同意。xFIP 3.74 + K-BB% 14.9 + xERA 3.50 結構好，但 ERA 3.89 / FIP 4.21 / velo **85.4 avg**（低）+ age 33 — 實質 🟡 Solid Starter 上緣。tier_v2 受 K-BB% 拉抬至 Strong Ace；保守按 Solid。
- **Reverse platoon 信號**：未 fired（vs LHB SLG .467 / vs RHB SLG .395 — 接近 reverse 但未 fire）。
  - 雖未 fire，vs LHB SLG .467 顯示對左打吃虧；STL 多右打陣容反 platoon-disadvantage。
- **對手打線威脅**：中。STL vs Springs (LHP) 多右打 — Walker .956 vs LHP / Winn .912 vs LHP / Fermín .881 vs LHP + Herrera .799 / Wetherholt .679 — Top 5 vs LHP 多 .800+；對 Springs 結構性有 edge，但 STL Cold last7 BABIP 0.253。

### Andre Pallante (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p65, K-BB% p35），gap vs ERA-only = +10.5
  - 同意 Solid Starter 下緣。ERA 4.34 / xERA 4.47 / FIP 4.57 / xFIP 3.91 / K-BB% 7.9（低）— xFIP 拉抬至 Solid，但 K-BB% 低 + vs LHB SLG .413 — 實質 🟢 Back-end Starter 上緣。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：高。OAK Top 6 vs RHP 都強（Kurtz .962 / Langeliers .995 / Cortes .924 / Soderstrom .806 / Rooker .709 / Gelof .857）+ Pallante vs LHB SLG .413 → Kurtz / Soderstrom 左打有 hunting zone；OAK 預期 5+ 分。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 上修同意。Top 6 整支 vs RHP .700+ + Pallante Back-end 級結構性弱 → 評估維持 🟠 Strong vs Pallante。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #2-3 gap 0.311 fired — Langeliers .995 → Soderstrom .806 vs RHP（中段微斷，影響不大實際）；−0.1 run。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟢 Weak
  - 不完全同意 Weak。Walker .956 vs LHP / Winn .912 / Fermín .881 + Top 5 vs LHP 多 .800+ — 上修至 🟡 Average vs LHP；Cold BABIP 0.253 拉低但對 Springs vs LHB .467 SLG 仍有 edge。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #6-7 gap 0.410 fired — Fermín .881 vs LHP → Pozo .182（chain 後段大幅斷層）— STL chain 尾極弱，限制大局得分連續性，−0.2 run。unlucky-cold ⏳ fired (BABIP 0.253) — Walker / Winn 個別擊球品質好，反彈面存在但 chain 尾結構性弱仍是限制。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.67 / 1 / 0 | 4.65 / 1 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：OAK 4.67 ERA + 0 core IL — 中等深度但完整火力，無 IL 弱點。
- AWAY 牛棚：STL 4.65 ERA + 0 core IL — 中等深度但完整；Pallante 預期 5-6 IP，STL 中繼需擋 3-4 局。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.253):
  - 短期可能反彈，Walker .956 vs LHP + Winn .912 個別擊球品質仍強；但 STL chain 尾結構性弱（Pozo .182 / Saggese .408 vs LHP）限制反彈幅度。本場 distribution 偏 STL 4-5 分。不自動 ±run value。

### 額外信號
- 🟠 HOME ERA 低估真實水平 +15.7（v2 score 76.9 vs ERA-only 61.2）
- 🔴 HOME TTO3 penalty：OPS Δ +0.339（TTO1 0.603 → TTO3 0.942），第三輪明顯衰退
- 🔴 HOME chain breaks at #2-3：OPS 落差 0.311
- 🟠 ⏳ AWAY unlucky-cold：last7 BABIP 0.253 偏低，冷期可能反彈
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.410
  - **本場最強訊號 = Sutter Health Park PF 109 (打者天堂) + 86°F + 10mph Out LF 三項 OVER 信號疊加**；Springs TTO3 OPS Δ +0.339 第3輪 OPS 0.942 崩盤 → STL Top 5 有時間在後段 cash in。Total OVER 是主訊號。

## 條件修正

- Park Factor: 109.0 → +0.45 run
- 天氣：Clear, 86°F, wind 10 mph, Out To LF
  - 影響判讀：**86°F 高溫（>85°F 利攻）+ 10mph 出 LF（輕度順風利左打拉打）+ PF 109（打者天堂）三項疊加** → Total OVER 強烈訊號，+0.3 total（已部分含於 PF）。
- 先發 tier / doubleheader：Springs Solid > Pallante Back-end 一級；雙方牛棚接近 + 0 core IL。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.6 | +0.1（−0.1 chain HOME + 風略利右打 +0.1，相抵 +0.0；OAK 5+ 分 base 已強，無大幅 ±） | 5.7 |
| AWAY | 4.9 | +0.1（HOME TTO3 +0.3 −0.2 chain AWAY = +0.1） | 5.0 |
| Total | 10.5 | +0.2（含 +0.3 weather 已部分含於 PF） | 10.7 |

## 整體判斷

- **方向（基本面）**：HOME (OAK)
- **總分（基本面）**：10.7（Total OVER 強訊號）
- **方向信心**：55% — OAK Top 6 vs Pallante 結構性 edge + 雙方接近的 SP 但 OAK 進攻深度好；信心受 STL Cold BABIP 反彈面 + Walker vs LHP .956 單槍 hold。
- **風險**：
  1. **三項 OVER 訊號疊加**（PF 109 + 86°F + 10mph 出 LF）— Total OVER 是最強訊號，超越方向判斷
  2. STL Cold BABIP 0.253 — Walker .956 vs LHP 反彈面存在
  3. Springs TTO3 OPS 0.942 第3輪垮台 — STL 後段有機會 cash in
  4. OAK Top 6 vs Pallante 全部有機會 — 容易拉開（單槍 + 環境條件）

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
