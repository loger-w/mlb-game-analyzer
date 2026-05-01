## 投手對決

### Taj Bradley (HOME, RHP, 25 ⚡ 巔峰期)
- **Tier 覆寫**：🟡 Solid Starter（從腳本 🟠 Strong Ace 下修）
  - 理由：ERA 2.91 但 xERA 4.23（gap 1.32，逼近 Flag 13 門檻），FIP 3.60 / xFIP 3.64 與 xERA 一致；GS 6 小樣本。Bradley 過去三年生涯 ERA 都在 4 字頭，K-BB% 16.4% 為中等水準。近 3 場 ER/IP 2/16.7（1.08 ERA）有壓制力，但被擊球品質（hard_hit 24.3%、barrel 7.4%）與 Solid Starter 相符，並非 Strong Ace。
- 真實水平判斷：球速 91.9 avg / 100.0 max + FF/FC/FS 三球種主軸，球種 mix 健康；vs LHB（.215/.282/.369）優於 vs RHB（.262/.347/.400）。整體真實水平 mid-3.5 ~ 4.0 ERA 區間。
- 對手打線威脅：SEA 打線 🟠 Strong（xwOBA .343），Cal Raleigh / JRod / Arozarena / Naylor 近 7 天皆 OPS > .920，但 last7 BABIP 偏高（最高至 .474），具回歸風險（Flag 3 邊界 .311 未觸發但個別球員觸發）。Bradley 對 RHB 較弱 — SEA 打線 RHB（Raleigh switch / JRod / Arozarena）正好吃 platoon 優勢。

### George Kirby (AWAY, RHP, 28 ⚡ 巔峰期)
- **Tier 覆寫**：🟠 Strong Ace（沿用腳本）
- 真實水平判斷：ERA 2.97 ≈ xERA 2.85，FIP 3.63 / xFIP 3.40，數據一致無 luck gap；WHIP 1.04 + K-BB% 13.1（K 不爆但保送極少）符合 Kirby 以控球見長的特質。vs LHB 壓制力極強（.171/.266/.243，79 BF），vs RHB 較鬆（.278/.297/.417）。近 3 場 ER/IP 8/20.0（3.60 ERA）略微回落但仍穩定。
- 對手打線威脅：MIN 打線 🟡 Average（xwOBA .328），Buxton vs RHP OPS .950 為主要威脅；其他 Wallner / Bell 為 LHB，正好撞 Kirby vs LHB 殺傷力。MIN 近 10 場 RS 3.7（攻↓），整體威脅低。

## 打線評級

### HOME — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .328、OPS .725、K% 20.5% / BB% 12.2%。近 10 場 RS 3.70 偏冷，趨勢攻↓；core threat 集中 Buxton 一人。

### AWAY — 🟠 Strong / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .343、OPS .703（xwOBA > OPS 暗示 BABIP 略低、未來實際得分趨升）；K% 24.0% 偏高為主要漏洞。Top 5 打者近 7 天集體炸（Naylor 1.225 / Raleigh 1.134 / Young .931 / Arozarena .979 / JRod .923），雖 BABIP 高有回歸風險，但 EV95% / Barrel% 結構性數據佳，部分熱度有實質支撐。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 5.13 / 6 / 牛棚 ERA 5.13 為大幅劣勢；IL 名單以 SP/depth 為主（Pablo López SP 60d、Festa SP 60d、Laweryson、Abel、Acton、Adams），核心高槓桿身分難以從腳本確認，但牛棚整體 ERA 已反映劣勢 | 3.33 / 3 / IL 含 Bryce Miller (SP 15d) + Vargas + 1 名；牛棚 ERA 3.33 為相對優勢 |

### 牛棚雙向修正值
- HOME 牛棚（MIN）：對手 +0.4 run（牛棚 ERA 5.13 = league-average 4.20 高出近 1 run，6 局後 SEA 進攻將具明顯優勢） | HOME ML −2%
- AWAY 牛棚（SEA）：對手 −0.3 run（牛棚 ERA 3.33 為投手友善，封鎖能力佳） | AWAY ML +1~2%

## 風險提示

無自動觸發風險（dossier 標示）。

AI 敘事補充：
- Taj Bradley ERA-xERA gap 1.32（未達 Flag 13 1.5 門檻但接近），疑有 BABIP/HR luck — 不自動下修，但 Tier 已從 Strong Ace 下修為 Solid 反映此差距
- SEA 多名打者 last7 BABIP > .370（個別觸發 Flag 3 概念），但團隊 last7 BABIP .311 未觸發；個別熱度部分由 EV95%/Barrel% 結構性指標支撐 → 判讀為「半可持續」，不自動 ±run

## 條件修正

- Park Factor: 106.0 → +0.30 run（Target Field 利得分但抑制 HR -2%，整體偏打者友善）
- 先發 tier：Bradley 下修為 🟡 Solid + Kirby 🟠 Strong → 套用「雙方 🟡 Solid+」-0.5 total（折中估計，因 Kirby 偏 Strong）
- Doubleheader：否
- 天氣：未提供（4 月 Minnesota 偏冷，傳統上略抑制得分但無顯著修正）

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (MIN) | 4.3 | -0.25（雙方先發 Solid+ 對沖一半）-0.3（SEA 牛棚優勢）-0.3（MIN 攻↓ 近 10 場 RS 3.7） = **-0.85** | **3.45** |
| AWAY (SEA) | 4.5 | -0.25（雙方先發 Solid+ 對沖一半）+0.4（MIN 牛棚劣勢）+0.3（Bradley xERA 結構性差距） = **+0.45** | **4.95** |
| Total | 8.8 | **-0.4** | **8.4** |

## 整體判斷

- **方向（基本面）**：SEA 略優；run differential ~1.5 run；勝負偏向 SEA 但 Bradley 近期火熱仍可壓制使比賽變窄
- **總分（基本面）**：8.4 vs O/U 7.5，差距 +0.9 → 偏 OVER 但**未達 1.5 run 門檻**（D2/D5 紀律下需 PASS 或最低星級）
- **信心**：MEDIUM
- **風險**：
  1. Bradley 近 3 場 1.08 ERA 火熱，若延續則 SEA 進攻被壓制、總分下修
  2. SEA 主力 last7 BABIP 多人 > .400，回歸風險顯著
  3. MIN 牛棚實際核心可用度未從腳本完全確認（IL 多為 SP）— +0.4 修正可能高估
  4. Kirby 控球路線 K-BB% 僅 13.1，遇上 Buxton 等熱打可能爆 1-2 個失分包

⛔ MUST NOT contain：星級、明確盤口推薦
