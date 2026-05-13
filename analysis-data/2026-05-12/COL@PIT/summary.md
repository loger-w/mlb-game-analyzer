## 投手對決

### Paul Skenes (HOME, RHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +4.7
  - 完全同意 Elite Ace。ERA 2.36 / xERA 1.96 / FIP 2.65 / xFIP 2.68 / K-BB% 24.5 / WHIP 0.71 — 全部 elite。gap +4.7 微小、已對齊。當代頂尖 Cy Young 候選。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：低。COL top 5 vs RHP 帳面好看（Moniak 1.106 / Johnston .952 / McCarthy 1.235 last7）但 Skenes vs LHB .154/.198/.253、vs RHB .161 — 兩邊都壓制；COL 進攻基本廢手。

### Michael Lorenzen (AWAY, RHP, 34 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p61, K-BB% p31），gap vs ERA-only = +37.9
  - 不同意 Solid Starter。ERA 6.92 / FIP 5.02 / K-BB% 7.4 / WHIP 1.90 / vs LHB .427/.467/.793(!) — 結構性 Below Average。tier_v2 受 xFIP 3.96 拉抬但 BB 與 hard contact 證據壓倒；實質 ⚪ Below Average。不下修預測（已是地板），但敘事按 Below Average。
- **Reverse platoon 信號**：未 fired（但 vs LHB SLG .793 是 reverse 邊緣）。
  - n/a
- **對手打線威脅**：極高。PIT Lowe 1.074 vs RHP last7 1.177 + Cruz / Reynolds / O'Hearn / Gonzales / Horwitz — top 5 都 .700+；Lorenzen vs LHB .793 SLG → Lowe / Reynolds / O'Hearn / Davis 左打全是 hunting zone。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 上修同意。Lowe / Cruz / Reynolds / O'Hearn / Gonzales / Horwitz 整支 vs RHP 強，遇 Lorenzen vs LHB .793 → 評估 🔴 Elite 對位（單場），尤其左打 cleanup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #8-9 gap 0.174 — 小幅，影響 chain 底末段，-0.1 run。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 不同意 Strong。對 Skenes Elite 級任何「Strong」評級都不適用 — 維持 Average，但 Skenes vs LHB/RHB 雙邊壓制下實質本場 🟢 Weak。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #6-7 gap 0.184 — 小幅，-0.1 run；Hot last7 BABIP 0.346 但未過 0.370 Flag 3 門檻，敘事「短期可能含運氣，遇 Skenes 大概率清零」。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.14 / 2 / 0 | 4.44 / 4 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：PIT 4.14 + 0 core IL — 中等深度但完整。Skenes 預期 7 IP，牛棚負擔輕。
- AWAY 牛棚：COL 4.44 + 0 core IL — 中等深度完整。Lorenzen 5 局後早被換投，COL 牛棚需擋 4+ 局，疲勞累積大。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.052（TTO1 0.560 → TTO3 0.612），第三輪明顯衰退；K% 從 34.1% 掉到 24.5%（Δ -9.6pp）（career fallback）
- ℹ️ AWAY balanced 4+ pitches：最高球種僅 20.1%（<25.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.121（TTO1 1.026 → TTO3 1.147），第三輪明顯衰退；K% 從 17.6% 掉到 13.3%（Δ -4.3pp）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.174
- 🟠 AWAY chain breaks at #6-7：OPS 落差 0.184
  - Skenes TTO3 仍 OPS 0.612 表示能撐第三輪，PIT 牛棚負擔極低；Lorenzen balanced 4+ pitches 雖能難對位但 vs LHB SLG .793 顯示球速球質壓不下對手，「balanced」對 Below Avg 投手反成劣勢。HOME 受惠 +0.3 run。

## 條件修正

- Park Factor: 102.0 → +0.10 run
- 天氣：Partly Cloudy, 70°F, wind 9 mph, Out To LF
  - 影響判讀：9mph 出 LF 對左打拉打輕度有利（Lowe / Reynolds / O'Hearn / Cruz 左打陣容受惠 HR 機率上修）；但 PNC Park HR -17% 是極端壓 HR 球場，兩者抵銷後實質中性偏 OVER 微幅，+0.1 total。
- 先發 tier / doubleheader：Skenes Elite >>> Lorenzen Below Avg（全卡最大 mismatch）；牛棚雙方接近，但 Skenes 完整 7 IP 預期使 PIT 牛棚優勢進一步擴大。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.8 | +0.1（AWAY TTO3 +0.1 + balanced 受惠 max+0.0 −0.1 chain HOME） | 5.9 |
| AWAY | 2.9 | 0.0（HOME TTO3 +0.1 −0.1 chain AWAY） | 2.9 |
| Total | 8.7 | +0.1 | 8.8 |

## 整體判斷

- **方向（基本面）**：HOME (PIT) — 強烈
- **總分（基本面）**：8.8
- **方向信心**：78%（卡上最高） — Skenes Elite vs Lorenzen Below Avg 是全卡最大 mismatch；PIT 進攻又有 Lowe / Cruz vs Lorenzen vs LHB .793 完美 platoon；信心受 COL 短期 hot streak last7 BABIP 0.346 微壓。
- **風險**：
  1. Skenes Elite 高機率輸出 7+ IP 1-2 ER 鎖場 — COL 想得分需 Skenes 失誤或 HR 運氣
  2. PNC HR -17% + 9mph 出 LF — Lowe / O'Hearn / Reynolds 拉打可能變 fly out（壓制 HR 但不壓制安打）
  3. Lorenzen 6.92 ERA + vs LHB .793 SLG — distribution 多數場景 PIT 5-7 分
  4. COL Hot BABIP 0.346 含運氣，遇 Skenes 大概率清零

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
