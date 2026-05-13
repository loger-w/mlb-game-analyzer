## 投手對決

### Trevor Rogers (HOME, LHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p79, K-BB% p66），gap vs ERA-only = +37.2
  - 不完全同意 Strong Ace。xFIP 3.68 + K-BB% 12.8 結構是 Solid Starter 上限，但 ERA 4.75 / FIP 3.59 + vs RHB SLG **.475**（被右打打）— 對 NYY 多右打陣容是嚴重劣勢。實質 🟡 Solid Starter（不到 Strong Ace）。
- **Reverse platoon 信號**：未 fired（vs LHB .091/.192 vs RHB .323/.374 — 巨型「正向」platoon，預期 LHP 對左打強）。
  - vs LHB 26 BF 樣本太小，但極壓 LHB 是 trend；對 NYY 的 Rice (左打)、Bellinger (左打)、Wells (左打) 有理論優勢，但其他 cleanup 都是右打。
- **對手打線威脅**：極高。NYY 1-3 棒 Goldschmidt(R) .884 vs LHP / Judge(R) 1.056 vs LHP / Rice(L) 1.084 vs LHP + cleanup Bellinger(L) .971 / Rosario .808 — **整支 vs LHP 都過 .800 OPS**，Rogers vs RHB SLG .475 → Judge/Goldschmidt 右打 cleanup 完美 hunting zone。

### Will Warren (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +26.3
  - 完全同意 Elite Ace。xFIP **2.38**(!) + K-BB% **24.2**(!) + FIP 3.22 + WHIP 1.20 — 結構超強。ERA 3.46 已壓制，xERA 3.63 對齊；gap +26.3 主要因 K-BB% 極高（v2 抓得到）。當代頂尖。
- **Reverse platoon 信號**：未 fired（vs LHB .238/.299/.438 / vs RHB .247/.287/.333）。
  - vs LHB SLG .438 略高，BAL 多右打陣容反 platoon-disadvantage。
- **對手打線威脅**：低。BAL top 5 vs RHP（Henderson .672 / Ward .777 / Rutschman .909 / Alonso .779 / Basallo .811）對 Warren K-BB% 24.2 + vs RHB SLG .333 結構性壓制下 → 主要威脅是 Rutschman switch hitter (.909)。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 部分同意。Rutschman / Basallo / Alonso 個別強，但對 Warren Elite 級無人能 hunting zone — 維持 Average，本場 distribution 偏 🟢 Weak 結果。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #5-6 gap 0.258 fired（Basallo .811 → O'Neill .765 vs RHP）— 中段微斷，−0.1 run。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong
  - 上修同意。**Top 5 vs LHP 全部 .800+**（Goldschmidt .884 / Judge 1.056 / Rice 1.084 / Bellinger .971 / Rosario .808） — 史上最豪華 vs LHP cleanup 之一；對 Rogers vs RHB SLG .475 是巨大 edge。實質 🔴 Elite vs LHP 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #8-9 gap 0.242 — Schuemann .833 (small sample) → Wells .389 vs LHP，chain 尾不影響 RBI 主軸，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.29 / 7 / 2 | 3.28 / 3 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：**BAL 4.29 ERA + 2 core IL（Bautista closer + Helsley setup）= 🔴 高**。Bautista 是 BAL 王牌 closer；Helsley 也是 setup level。後段 8-9 局完全失能，NYY 末段加分機率極高。
- AWAY 牛棚：NYY 3.28 ERA + 0 core IL — 完整火力，Warren 7 IP 後鎖場無虞。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.168（TTO1 0.496 → TTO3 0.664），第三輪明顯衰退；K% 從 27.8% 掉到 18.8%（Δ -9.0pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.288（TTO1 0.522 → TTO3 0.810），第三輪明顯衰退；K% 從 40.7% 掉到 16.1%（Δ -24.6pp）
- 🟠 HOME chain breaks at #5-6：OPS 落差 0.258
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.242
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **本場最大訊號 = Warren Elite + BAL 2 core IL（含 Bautista closer）**。Warren TTO3 K% -24.6pp 是 hidden 風險但其 TTO3 OPS 仍 0.810（不算崩盤），NYY 6+ 分機率高；BAL 進攻被 Warren 結構性壓制。

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：Clear, 68°F, wind 8 mph, Out To LF
  - 影響判讀：8mph 出 LF 是輕度順風（噪音邊界），略利右打拉打 — Judge / Goldschmidt 右打 cleanup 受惠 HR 機率輕微 ↑，+0.1 total。Camden HR +7% 已含。
- 先發 tier / doubleheader：Warren Elite >>> Rogers Solid（兩級以上 mismatch）；BAL 2 core IL（含 closer）vs NYY 0 core IL — 後段結構性壓 NYY。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.3 | +0.1（AWAY TTO3 +0.2 −0.1 chain HOME） | 3.4 |
| AWAY | 4.1 | +0.3（HOME TTO3 +0.2 + HOME core IL ×2 含 closer +0.3 互動 max+0.1 −0.1 chain AWAY） | 4.4 |
| Total | 7.4 | +0.4（+0.1 風） | 7.8 |

## 整體判斷

- **方向（基本面）**：AWAY (NYY) — 強烈
- **總分（基本面）**：7.8（厚尾向 8.5+，因 Rogers Flag 8 +37.2 vs NYY top 5 vs LHP）
- **方向信心**：70% — Warren Elite + NYY 五人 vs LHP .800+ + BAL 2 core IL（含 Bautista closer）三層 edge 疊加。
- **風險**：
  1. Warren TTO3 K% -24.6pp 巨型衰退 → 5-6 局後 BAL 可能炸開（Rutschman/Basallo/Alonso 任何一棒）
  2. NYY 連敗 4 場（streak -4） — 心態風險
  3. Camden HR +7% + Out LF → Judge 1 球翻盤面，但 BAL Alonso/Basallo 也有單槍能力
  4. Rogers vs LHB SLG .182 極端小樣本（26 BF），可能單場反彈

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
