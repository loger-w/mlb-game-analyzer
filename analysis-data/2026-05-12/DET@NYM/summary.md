## 投手對決

### Freddy Peralta (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p78, K-BB% p70），gap vs ERA-only = +1.8
  - 同意 Strong Ace。ERA 3.12 / xERA 3.81 / FIP 3.56 / xFIP 3.71 / K-BB% 13.5 + vs LHB .212/.274 + vs RHB .200/.324/.217 — 結構穩。gap +1.8 微小，已對齊。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：高。DET season 🟢 Weak 但對 RHP 仍有 Greene .830 / McGonigle .896 / Keith .746 / Dingler .824 — top 5 集中。Peralta single-pitch FF 54.0% + 對 RHB SLG .217 → DET 多右打但仍會被壓制。

### Jack Flaherty (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p41, K-BB% p49），gap vs ERA-only = +30.6
  - 不同意 Solid Starter。ERA 5.56 / FIP 4.84 / vs LHB SLG .424 + 30 歲已退化 — 結構性 Back-end。tier_v2 受 xFIP 4.26 拉抬但 K-BB 9.9 + vs LHB 被打證據壓倒；實質 🟢 Back-end Starter。不下修預測（已含），但敘事按 Back-end。
- **Reverse platoon 信號**：未 fired（但 vs LHB SLG .424 vs RHB .393 接近反向）。
  - n/a
- **對手打線威脅**：極高。NYM season 🔴 Elite + Hot + vs RHP Elite — Soto .978 vs RHP / Melendez .852 / Alvarez .704 / Vientos .677。Flaherty single-pitch FF 47.1% + vs LHB SLG .424 → Soto / Bichette / Melendez 左打殺手 zone。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - 不同意 Weak。Soto .978 vs RHP + Melendez / Alvarez + Hot heat 整支 → 評估上修至 🔴 Elite vs RHP（與 season tier 一致）；script 標 Weak 可能因 Last 7 BABIP cold 拉低，但 Soto 本人 EV95 47.4 + Barrel 19.7 證據強。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #7-8 gap 0.594 — Ewing .000（空白樣本污染）+ Alvarez .697 真實普通 → 部分採用，−0.1 run。Flag 3 BABIP 0.250 cold — 但 Soto/Melendez OPS 仍 high，純運氣冷，敘事「短期應反彈」，不 ±run。

### AWAY — season tier 🔴 Elite / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🔴 Elite
  - 不完全同意 Elite。McGonigle .896 + Greene .830 + Dingler .824 + Keith .746 vs RHP 是強，但 Workman 2.500 是樣本污染 → 評估實質為 🟠 Strong vs RHP（去除噪音），仍是強隊。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #7-8 gap **1.763** — Workman 2.500 OPS 是 1 球小樣本污染，**完全不採用**。實質 chain 連續性正常。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.71 / 6 / 3 | 3.83 / 10 / 3 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：NYM 3.71 ERA + 3 core IL（Minter + Núñez + 1）= 🔴🔴 崩盤級。Peralta 若 6 局後被換投，DET 末段大幅得分機會。
- AWAY 牛棚：DET 3.83 ERA + 3 core IL（Brieske + Melton + 1）= 🔴🔴 崩盤級。Flaherty 若 5 局後被換投（高機率），NYM Hot 進攻可在後段炸開。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.25):
  - 短期可能反彈（Soto / Melendez 本人擊球品質 EV95/Barrel 強，純運氣冷期），但本場遇 Flaherty Back-end + vs LHB SLG .424 — Soto 等左打更有機會 → 反彈方向偏 NYM 進攻 +。不自動 ±run value。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 54.0%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +-0.001（TTO1 0.560 → TTO3 0.559），第三輪明顯衰退；K% 從 29.2% 掉到 19.5%（Δ -9.7pp）
- 🟠 AWAY single-pitch dependent：主球種使用率 47.1%（≥45.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.023（TTO1 0.761 → TTO3 0.784），第三輪明顯衰退；K% 從 28.2% 掉到 23.9%（Δ -4.3pp）（career fallback）
- 🔴 HOME chain breaks at #7-8：OPS 落差 0.594
- 🔴 AWAY chain breaks at #7-8：OPS 落差 1.763
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - **本場最大訊號 = 雙方都 3 core IL 崩盤級**。6 局後雙方都進中繼地獄，總得分 distribution 厚尾向上；單場 OVER 概率明顯高於 base 10.2。雙方信號相抵 → 方向不變但 Total 偏 OVER。

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：Clear, 61°F, wind 13 mph, Out To LF
  - 影響判讀：13mph 出 LF 中度順風利左打拉打（Soto / Bichette / Melendez 左打 cleanup 受惠）；61°F 偏涼略利投但風的效應更大 — 整體偏 OVER +0.3 total。
- 先發 tier / doubleheader：Peralta Strong Ace > Flaherty Back-end 一級以上；但 NYM Hot 進攻火力是 Flaherty 撐不住的關鍵；雙方 3 core IL 崩盤對總分有最大影響。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.1 | +0.5（AWAY core IL ×3 +0.5 + AWAY TTO3 +0.0 + AWAY single-pitch +0.1 互動 max+0.1 −0.1 chain HOME） | 5.6 |
| AWAY | 5.1 | +0.4（HOME core IL ×3 +0.5 + HOME single-pitch +0.1 互動 max+0.1 −0.1 chain AWAY 污染不採用） | 5.5 |
| Total | 10.2 | +0.9（+ 13mph 風 OVER +0.3） | 11.1 |

## 整體判斷

- **方向（基本面）**：AWAY (DET 微傾)
- **總分（基本面）**：11.1（強烈 OVER 訊號）
- **方向信心**：55%（信心低 — 雙崩盤亂場）— NYM Soto+Melendez vs Flaherty Back-end vs LHB 是 NYM 進攻優勢；但 DET top 4 vs Peralta single-pitch 也有機會；雙方 3 core IL 末段平衡。微傾 DET 因 Peralta 自己 TTO3 K% -9.7pp 暗示早被換投，暴露 NYM 自家 3 core IL 崩盤。
- **風險**：
  1. 雙方都 3 core IL 崩盤級 → 6 局後完全失控可能性最高，Total OVER 是主訊號
  2. Soto 1 球破壞分析（EV95 47.4 + Barrel 19.7 vs RHP）
  3. 13mph 出 LF 風 + Citi Field HR +7% → HR 機率上升
  4. NYM Hot last7 OPS .903 vs Flaherty Back-end + vs LHB SLG .424 → NYM 6+ 分可能性高

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
